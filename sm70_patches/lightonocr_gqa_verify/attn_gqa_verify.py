"""Patch « attn_gqa_verify » : attention de vérification spéculative groupée (GQA × jetons) pour le
backend FLASH_ATTN_V100 de 1Cat-vLLM 1.5.1 (V100, LightOnOCR-3-4B : 16 têtes q, 4 têtes KV, D = 256).

Interface du chargeur (scripts/vllm_ext/sitecustomize.py) : `CIBLE` est importé, puis `appliquer(module)`
est appelée juste après le premier import de la cible, dans chaque processus.

Méthodes remplacées sur `FlashAttnV100Impl` (repli explicite vers l'original pour toute forme non gérée) :
  1. `_flash_v100_small_query_prefill_as_decode` — lots de vérification (≤ smallq_decode_max_query_len
     jetons par requête), y compris en capture/rejeu de graphe CUDA : on lit les métadonnées persistantes
     du fork (smallq_query_start_loc, smallq_decode_block_table, smallq_decode_seq_lens), sinon (mode
     eager) attn_metadata.query_start_loc / block_table / seq_lens ;
  2. `_run_prefill_prefix_decode_rows` — lignes « petit q » des lots mixtes (prefill + décodage), eager.

Réglages (variables d'environnement) :
  GQA_VERIFY=0            désactive le patch à chaud (repli complet sur l'original) ;
  GQA_VERIFY_BACKEND      auto (défaut : CUDA WMMA si l'extension est chargée, sinon voie d'origine) | cuda
                          | triton (variante FMA, ~1,1-1,2× seulement) ;
  GQA_VERIFY_MIXTE=0      ne pas toucher aux lots mixtes ;
  GQA_VERIFY_PS=512       taille de partition de KV ;
  GQA_VERIFY_SO           chemin de gqa_verify_cuda.so (défaut : build/ à côté de ce fichier).
"""
from __future__ import annotations

import os
import sys

CIBLE = "vllm.v1.attention.backends.flash_attn_v100"

_ici = os.path.dirname(os.path.abspath(__file__))
try:  # chargé par le chargeur en tant que patchs.attn_gqa_verify
    from patchs import gqa_verify as _gv  # type: ignore
except Exception:  # noqa: BLE001
    if _ici not in sys.path:
        sys.path.insert(0, _ici)
    import gqa_verify as _gv  # type: ignore

_ETAT = {"appels": 0, "replis": {}, "mixte": 0}
_ATT = None


def _att():
    global _ATT
    if _ATT is None:
        _ATT = _gv.GQAVerifyAttention(partition_size=int(os.environ.get("GQA_VERIFY_PS", "512")),
                                      backend=os.environ.get("GQA_VERIFY_BACKEND", "auto"))
    return _ATT


def _repli(raison: str, logger=None) -> None:
    n = _ETAT["replis"].get(raison, 0)
    _ETAT["replis"][raison] = n + 1
    if n == 0 and logger is not None:
        logger.info("attn_gqa_verify : repli vers la voie d'origine (%s)", raison)


def _eligible(impl, layer, query, key_cache, value_cache, attn_metadata) -> str | None:
    """None si la forme est gérée, sinon la raison du repli."""
    import torch
    if os.environ.get("GQA_VERIFY", "1") == "0":
        return "désactivé"
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        return "dtype"
    if getattr(impl, "kv_cache_dtype", "auto") not in ("auto", "float16"):
        return "kv_cache_dtype"
    if query.dim() != 3 or key_cache.dim() != 4 or query.shape[2] != 256 or key_cache.shape[3] != 256:
        return "head_dim"
    hkv = int(key_cache.shape[2])
    if hkv <= 0 or query.shape[1] != 4 * hkv:
        return "groupe≠4"
    if int(getattr(impl, "smallq_decode_max_query_len", 0)) > 8:
        return "q>8"
    if getattr(impl, "alibi_slopes", None) is not None or getattr(impl, "sinks", None) is not None:
        return "alibi/sinks"
    try:
        if tuple(impl._flash_v100_window_size(causal=True)) != (-1, -1):
            return "fenêtre"
    except Exception:  # noqa: BLE001
        return "fenêtre?"
    if not getattr(attn_metadata, "causal", True):
        return "non causal"
    if getattr(attn_metadata, "is_dflash_selector_target", False):
        return "dflash"
    if getattr(impl, "prefix_anchored_decode_window", None) is not None:
        return "fenêtre ancrée"
    if float(getattr(layer, "_k_scale_float", 1.0)) != 1.0 or float(getattr(layer, "_v_scale_float", 1.0)) != 1.0:
        return "échelles kv"
    if query.stride(2) != 1 or key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return "pas"
    # Sans l'extension CUDA, la variante Triton (FMA) ne gagne que ~1,1-1,2× : on ne la prend
    # que si on la demande explicitement (GQA_VERIFY_BACKEND=triton).
    if os.environ.get("GQA_VERIFY_BACKEND", "auto") != "triton" and not _gv.charger_cuda():
        return "extension CUDA absente"
    return None


def _capacite(impl, attn_metadata, block_table, key_cache) -> int:
    cap = int(block_table.shape[1]) * int(key_cache.shape[1])
    mml = int(getattr(attn_metadata, "max_model_len", 0) or 0)
    return min(cap, mml) if mml > 0 else cap


def appliquer(module) -> None:
    import torch

    Impl = module.FlashAttnV100Impl
    logger = getattr(module, "logger", None)
    if getattr(Impl, "_gqa_verify_applique", False):
        return
    orig_smallq = Impl._flash_v100_small_query_prefill_as_decode
    orig_mixte = getattr(Impl, "_run_prefill_prefix_decode_rows", None)
    _gv.charger_cuda()  # avant toute capture de graphe

    def smallq(self, layer, query, key_cache, value_cache, attn_metadata, output, query_start_loc, _seq_lens):
        raison = _eligible(self, layer, query, key_cache, value_cache, attn_metadata)
        if raison is not None:
            _repli(raison, logger)
            return orig_smallq(self, layer, query, key_cache, value_cache, attn_metadata, output,
                               query_start_loc, _seq_lens)
        n = min(int(attn_metadata.num_actual_tokens), int(query.shape[0]), int(output.shape[0]))
        bt_tok = getattr(attn_metadata, "smallq_decode_block_table", None)
        sl_tok = getattr(attn_metadata, "smallq_decode_seq_lens", None)
        qsl_p = getattr(attn_metadata, "smallq_query_start_loc", None)
        if (bt_tok is not None and sl_tok is not None and qsl_p is not None
                and int(sl_tok.shape[0]) >= n and int(bt_tok.shape[0]) >= n):
            # métadonnées persistantes (seules sûres en capture/rejeu de graphe)
            bt, sl, qsl, per_token = bt_tok, sl_tok, qsl_p, True
        else:
            if module._is_cuda_graph_capturing(query):
                _repli("capture sans métadonnées persistantes", logger)
                return orig_smallq(self, layer, query, key_cache, value_cache, attn_metadata, output,
                                   query_start_loc, _seq_lens)
            bt, sl, qsl, per_token = (attn_metadata.block_table, attn_metadata.seq_lens,
                                      attn_metadata.query_start_loc, False)
        if (bt.dtype != torch.int32 or sl.dtype != torch.int32 or qsl.dtype != torch.int32
                or not bt.is_cuda or not qsl.is_cuda or bt.stride(1) != 1):
            _repli("métadonnées", logger)
            return orig_smallq(self, layer, query, key_cache, value_cache, attn_metadata, output,
                               query_start_loc, _seq_lens)
        maxq = max(1, int(getattr(self, "smallq_decode_max_query_len", 5)))
        _att()(query[:n], key_cache, value_cache, bt, sl, qsl, output[:n], scale=float(self.scale),
               max_query_len=maxq, kv_capacity=_capacite(self, attn_metadata, bt, key_cache),
               softcap=float(getattr(self, "logits_soft_cap", 0.0) or 0.0), per_token=per_token)
        _ETAT["appels"] += 1
        if _ETAT["appels"] == 1 and logger is not None:
            logger.info("attn_gqa_verify : voie de vérification groupée active (G=4, backend=%s, cuda=%s, "
                        "par_jeton=%s)", _att().backend, _gv.charger_cuda(), per_token)
        try:
            module._record_route("prefill_smallq_gqa_verify")
        except Exception:  # noqa: BLE001
            pass
        return output

    def mixte(self, layer, query, key_cache, value_cache, attn_metadata, out_view, query_start_loc, seq_lens,
              window_size):
        raison = _eligible(self, layer, query, key_cache, value_cache, attn_metadata)
        if raison is None and os.environ.get("GQA_VERIFY_MIXTE", "1") == "0":
            raison = "mixte désactivé"
        if raison is None and tuple(window_size) != (-1, -1):
            raison = "fenêtre"
        if raison is not None:
            _repli("mixte:" + raison, logger)
            return orig_mixte(self, layer, query, key_cache, value_cache, attn_metadata, out_view,
                              query_start_loc, seq_lens, window_size)
        num_seqs = len(query_start_loc) - 1
        qsl = [int(x) for x in query_start_loc[: num_seqs + 1].tolist()]
        sl_h = [int(x) for x in seq_lens[:num_seqs].tolist()]
        max_q = max(1, int(self.smallq_decode_max_query_len))
        rows = [i for i in range(num_seqs) if 1 <= qsl[i + 1] - qsl[i] <= max_q and sl_h[i] > qsl[i + 1] - qsl[i]]
        if not rows or len(rows) == num_seqs:
            return set()
        dev = query.device
        tok, cqsl = [], [0]
        for i in rows:
            tok.extend(range(qsl[i], qsl[i + 1]))
            cqsl.append(cqsl[-1] + qsl[i + 1] - qsl[i])
        start_idx = torch.tensor(tok, device=dev, dtype=torch.long)
        q_rows = query.index_select(0, start_idx)
        out_rows = torch.empty_like(q_rows)
        bt = attn_metadata.block_table.index_select(0, torch.tensor(rows, device=dev, dtype=torch.long))
        if bt.dtype != torch.int32:
            bt = bt.to(torch.int32)
        bt = bt.contiguous()
        sl_t = torch.tensor([sl_h[i] for i in rows], device=dev, dtype=torch.int32)
        qsl_t = torch.tensor(cqsl, device=dev, dtype=torch.int32)
        _att()(q_rows, key_cache, value_cache, bt, sl_t, qsl_t, out_rows, scale=float(self.scale),
               max_query_len=max_q, kv_capacity=_capacite(self, attn_metadata, attn_metadata.block_table, key_cache),
               softcap=float(getattr(self, "logits_soft_cap", 0.0) or 0.0), per_token=False)
        out_view.index_copy_(0, start_idx, out_rows)
        _ETAT["mixte"] += 1
        try:
            module._record_route("prefill_prefix_decode_rows_gqa_verify")
        except Exception:  # noqa: BLE001
            pass
        return set(rows)

    smallq.__wrapped__ = orig_smallq
    Impl._flash_v100_small_query_prefill_as_decode = smallq
    if orig_mixte is not None:
        mixte.__wrapped__ = orig_mixte
        Impl._run_prefill_prefix_decode_rows = mixte
    Impl._gqa_verify_applique = True
    Impl._gqa_verify_etat = _ETAT
    if logger is not None:
        logger.info("attn_gqa_verify : patch appliqué (cuda=%s)", _gv.charger_cuda())

"""Patch « vit_attn_fa » : attention de la tour vision (Qwen3-VL / LightOnOCR-3-4B, d = 64, non causale) par un
noyau FlashAttention CUDA WMMA pour sm_70 (build/vit_fa_cuda.so, op torch.ops.vit_fa.forward), à la place de
la SDPA « mem-efficient » (cutlassF sm70) de `vllm::torch_sdpa_wrapper`.

Deux gains :
  - le noyau (1,17× au meilleur temps, 1,23× en médiane, contre la SDPA sur une colonne de 5 888 patchs ;
    variante VIT_FA_VARIANTE=32 par défaut : blocs de 32 clés, 4 CTA/SM ; 64 : blocs de 64 clés, 2 CTA/SM) ;
  - plus de synchronisation GPU→CPU par couche : l'original fait `cu_seqlens.tolist()` (tenseur GPU) dans
    chacune des 24 couches ; ici la grille est bornée par la longueur totale (connue de l'hôte) et chaque CTA
    lit les bornes de sa séquence sur le GPU.

Interface du chargeur : CIBLE + appliquer(module). Remplace `MMEncoderAttention._forward_sdpa`, avec repli vers
l'original si : GQA (têtes kv ≠ têtes q), head_dim ≠ 64, dtype ≠ fp16, lot > 1 sans cu_seqlens, q et kv de
longueurs différentes, pas non multiples de 8, extension absente, ou VIT_FA=0.
"""
from __future__ import annotations

import os

CIBLE = "vllm.model_executor.layers.attention.mm_encoder_attention"

_ici = os.path.dirname(os.path.abspath(__file__))
_OK = None
_CU = {}
_ETAT = {"appels": 0, "replis": {}}


def charger(chemin: str | None = None) -> bool:
    global _OK
    if _OK is not None:
        return _OK
    import torch
    chemin = chemin or os.environ.get("VIT_FA_SO") or os.path.join(_ici, "build", "vit_fa_cuda.so")
    try:
        if not (hasattr(torch.ops, "vit_fa") and hasattr(torch.ops.vit_fa, "forward")):
            torch.ops.load_library(chemin)
        _OK = hasattr(torch.ops.vit_fa, "forward")
    except Exception:  # noqa: BLE001
        _OK = False
    return _OK


def _pas_ok(t) -> bool:
    return t.stride(-1) == 1 and t.stride(-2) % 8 == 0 and t.stride(-3) % 8 == 0 and t.data_ptr() % 16 == 0


def vit_fa(q3, k3, v3, cu, scale: float):
    """q3/k3/v3 [S, H, 64] fp16 (vues à pas), cu [nseq + 1] int32 sur le GPU. Retour [S, H, 64] contigu."""
    import torch
    out = torch.empty(q3.shape, dtype=q3.dtype, device=q3.device)
    torch.ops.vit_fa.forward(q3, k3, v3, out, cu, int(q3.shape[0]), float(scale),
                             int(os.environ.get("VIT_FA_VARIANTE", "32")))
    return out


def appliquer(module) -> None:
    import torch

    Cls = module.MMEncoderAttention
    if getattr(Cls, "_vit_fa_applique", False):
        return
    orig = Cls._forward_sdpa
    logger = getattr(module, "logger", None)
    charger()

    def repli(raison, *a):
        n = _ETAT["replis"].get(raison, 0)
        _ETAT["replis"][raison] = n + 1
        if n == 0 and logger is not None:
            logger.info("vit_attn_fa : repli vers la SDPA d'origine (%s)", raison)
        return orig(*a)

    def _forward_sdpa(self, query, key, value, cu_seqlens=None):
        a = (self, query, key, value, cu_seqlens)
        if os.environ.get("VIT_FA", "1") == "0":
            return repli("désactivé", *a)
        if not charger():
            return repli("extension absente", *a)
        if query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16 or not query.is_cuda:
            return repli("dtype", *a)
        if int(self.head_size) != 64 or int(self.num_heads) != int(self.num_kv_heads):
            return repli("forme (d≠64 ou GQA)", *a)
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        if kv_len != q_len or (bsz != 1):
            return repli("lot>1 ou q≠kv", *a)
        is_reshaped = query.dim() != 4
        q4, k4, v4 = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
        q3, k3, v3 = q4[0], k4[0], v4[0]
        if not (_pas_ok(q3) and _pas_ok(k3) and _pas_ok(v3)):
            q3, k3, v3 = q3.contiguous(), k3.contiguous(), v3.contiguous()
        if cu_seqlens is None:
            key_cu = (int(q_len), query.device)
            cu = _CU.get(key_cu)
            if cu is None:
                cu = torch.tensor([0, int(q_len)], dtype=torch.int32, device=query.device)
                _CU[key_cu] = cu
        else:
            cu = cu_seqlens
            if cu.device != query.device or cu.dtype != torch.int32 or cu.dim() != 1:
                return repli("cu_seqlens", *a)
            if not cu.is_contiguous():
                cu = cu.contiguous()
        scale = self.scale if self.scale is not None else 64 ** -0.5
        out = vit_fa(q3, k3, v3, cu, float(scale)).unsqueeze(0)
        _ETAT["appels"] += 1
        if _ETAT["appels"] == 1 and logger is not None:
            logger.info("vit_attn_fa : attention vision par le noyau FA WMMA sm70 (S=%d, H=%d)", q_len, q3.shape[1])
        if is_reshaped:
            out = out.reshape(bsz, q_len, -1)
        return out

    _forward_sdpa.__wrapped__ = orig
    Cls._forward_sdpa = _forward_sdpa
    Cls._vit_fa_applique = True
    Cls._vit_fa_etat = _ETAT
    if logger is not None:
        logger.info("vit_attn_fa : patch appliqué (extension=%s)", charger())

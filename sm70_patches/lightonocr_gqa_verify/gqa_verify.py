"""Attention de vérification spéculative groupée (GQA × jetons) pour SM70 (V100), en Triton.

Remplace, pour les lots « petit q » (≤ MAXQ jetons par requête) du backend FLASH_ATTN_V100 de
1Cat-vLLM, la voie actuelle qui déplie chaque jeton vérifié en une ligne de décodage indépendante
(`flash_attention_decode_partition_kernel`, une CTA par (jeton, tête q, partition) : le KV de la
requête est relu G × q_len fois).

Ici une CTA traite (requête, tête KV, partition de KV) : elle charge les G têtes q du groupe ×
q_len jetons de la requête (BLOCK_M lignes), lit chaque bloc K/V une seule fois, applique le masque
causal propre à chaque jeton spéculatif, softmax en ligne en fp32. Une seconde passe combine les
partitions (split-K).

Sémantique reprise de `FlashAttnV100Impl._flash_v100_small_query_prefill_as_decode` :
  - requête r : jetons [qsl[r], qsl[r+1]), q_len = qsl[r+1] - qsl[r] ;
  - eff = max(seq_lens[r], q_len) (seq_lens inclut les jetons de la requête) ;
  - le jeton j (0 ≤ j < q_len) voit les clés [0, eff - q_len + j + 1) ;
  - les jetons hors de toute requête (remplissage de graphe CUDA) reçoivent une sortie nulle ;
  - cache KV paginé [num_blocks, block_size, H_kv, D] (vues de kv_cache[:, 0] / [:, 1], pas
    forcément contiguës : on lit les pas réels) ; position p → bloc block_table[r, p // block_size],
    case p % block_size ;
  - logits = (q·k) × scale, softcap optionnel (cap × tanh(x / cap)).

Pas d'hypothèse de cœurs tenseurs : sur sm_70, Triton compile `tl.dot` en FMA (fp32).
Aucune synchronisation hôte, aucune allocation par appel (espace de travail mis en cache par
forme), donc compatible avec la capture en graphes CUDA.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_verify_partition_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, qsl_ptr,
    po_ptr, pm_ptr, pl_ptr,
    n_tokens, scale, softcap,
    s_qt, s_qh,
    s_kb, s_ks, s_kh,
    s_vb, s_vs, s_vh,
    s_btr,
    s_pot, s_poh, s_pop,
    s_pmt, s_pmh,
    PAGE: tl.constexpr, G: tl.constexpr, MAXQ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, PS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr, PER_TOKEN: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.program_id(2)

    qs = tl.load(qsl_ptr + r)
    qe = tl.load(qsl_ptr + r + 1)
    qlen = qe - qs
    if PER_TOKEN:
        eff = tl.load(seq_lens_ptr + tl.maximum(qe - 1, 0))
        bt_row = qs
    else:
        eff = tl.maximum(tl.load(seq_lens_ptr + r), qlen)
        bt_row = r
    ctx0 = eff - qlen
    kv_start = p * PS
    if (qlen > 0) & (kv_start < eff):
        kv_end = tl.minimum(kv_start + PS, eff)

        offs_m = tl.arange(0, BLOCK_M)
        j = offs_m // G
        g = offs_m % G
        tok = qs + j
        row_valid = (j < qlen) & (j < MAXQ) & (tok < n_tokens)
        qh = h * G + g
        offs_d = tl.arange(0, D)

        q = tl.load(
            q_ptr + tok[:, None].to(tl.int64) * s_qt + qh[:, None] * s_qh + offs_d[None, :],
            mask=row_valid[:, None], other=0.0,
        ).to(tl.float32)
        row_limit = ctx0 + j + 1  # clés visibles : [0, row_limit)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

        for start in range(kv_start, kv_end, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            n_valid = offs_n < kv_end
            page = tl.load(bt_ptr + bt_row.to(tl.int64) * s_btr + offs_n // PAGE, mask=n_valid, other=0)
            page = tl.maximum(page, 0).to(tl.int64)
            slot = offs_n % PAGE
            k = tl.load(
                k_ptr + page[:, None] * s_kb + slot[:, None] * s_ks + h * s_kh + offs_d[None, :],
                mask=n_valid[:, None], other=0.0,
            ).to(tl.float32)
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            if USE_SOFTCAP:
                # tanh(x) = 2σ(2x) − 1 (pas de tl.tanh portable)
                x = s / softcap
                s = softcap * (2.0 * tl.sigmoid(2.0 * x) - 1.0)
            vis = row_valid[:, None] & n_valid[None, :] & (offs_n[None, :] < row_limit[:, None])
            s = tl.where(vis, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            pr = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(pr, 1)
            v = tl.load(
                v_ptr + page[:, None] * s_vb + slot[:, None] * s_vs + h * s_vh + offs_d[None, :],
                mask=n_valid[:, None], other=0.0,
            ).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(pr, v, input_precision="ieee")
            m_i = m_new

        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / l_safe[:, None]
        tok64 = tok.to(tl.int64)
        tl.store(
            po_ptr + tok64[:, None] * s_pot + qh[:, None] * s_poh + p * s_pop + offs_d[None, :],
            o, mask=row_valid[:, None],
        )
        tl.store(pm_ptr + tok64 * s_pmt + qh * s_pmh + p, m_i, mask=row_valid)
        tl.store(pl_ptr + tok64 * s_pmt + qh * s_pmh + p, l_i, mask=row_valid)


@triton.jit
def _gqa_verify_reduce_kernel(
    out_ptr, po_ptr, pm_ptr, pl_ptr, seq_lens_ptr, qsl_ptr,
    num_reqs,
    s_ot, s_oh,
    s_pot, s_poh, s_pop,
    s_pmt, s_pmh,
    D: tl.constexpr, PS: tl.constexpr, P_BLOCK: tl.constexpr, REQ_BLOCK: tl.constexpr,
    PER_TOKEN: tl.constexpr,
):
    t = tl.program_id(0)
    hq = tl.program_id(1)

    offs_r = tl.arange(0, REQ_BLOCK)
    ends = tl.load(qsl_ptr + 1 + offs_r, mask=offs_r < num_reqs, other=0x7FFFFFFF)
    r = tl.sum((t >= ends).to(tl.int32), axis=0)
    in_req = r < num_reqs
    rc = tl.minimum(r, num_reqs - 1)
    qs = tl.load(qsl_ptr + rc)
    qe = tl.load(qsl_ptr + rc + 1)
    qlen = qe - qs
    jj = t - qs
    if PER_TOKEN:
        eff = tl.load(seq_lens_ptr + tl.maximum(qe - 1, 0))
    else:
        eff = tl.maximum(tl.load(seq_lens_ptr + rc), qlen)
    row_limit = eff - qlen + jj + 1
    valid = in_req & (jj >= 0) & (jj < qlen)
    n_p = tl.where(valid, (row_limit + PS - 1) // PS, 0)

    offs_p = tl.arange(0, P_BLOCK)
    pmask = offs_p < n_p
    t64 = t.to(tl.int64)
    base_m = pm_ptr + t64 * s_pmt + hq * s_pmh
    m = tl.load(base_m + offs_p, mask=pmask, other=float("-inf"))
    l = tl.load(pl_ptr + t64 * s_pmt + hq * s_pmh + offs_p, mask=pmask, other=0.0)
    mx = tl.max(m, 0)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.where(pmask, l * tl.exp(m - mx), 0.0)
    wsum = tl.sum(w, 0)
    offs_d = tl.arange(0, D)
    o = tl.load(
        po_ptr + t64 * s_pot + hq * s_poh + offs_p[:, None] * s_pop + offs_d[None, :],
        mask=pmask[:, None], other=0.0,
    )
    res = tl.sum(o * w[:, None], 0) / tl.where(wsum > 0.0, wsum, 1.0)
    res = tl.where(valid, res, 0.0)
    tl.store(out_ptr + t64 * s_ot + hq * s_oh + offs_d, res.to(out_ptr.dtype.element_ty))


_CUDA_OK: bool | None = None


def charger_cuda(chemin: str | None = None) -> bool:
    """Charge l'extension CUDA WMMA (build/gqa_verify_cuda.so, voir construire.py). Idempotent."""
    global _CUDA_OK
    if _CUDA_OK is not None:
        return _CUDA_OK
    chemin = chemin or os.environ.get("GQA_VERIFY_SO") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "build", "gqa_verify_cuda.so")
    try:
        if not hasattr(torch.ops, "gqa_verify") or not hasattr(torch.ops.gqa_verify, "partition"):
            torch.ops.load_library(chemin)
        _CUDA_OK = hasattr(torch.ops.gqa_verify, "partition")
    except Exception:  # noqa: BLE001
        _CUDA_OK = False
    return _CUDA_OK


class GQAVerifyAttention:
    """Lanceur avec espace de travail persistant (un par (capacité jetons, H_q, P, D, device)).

    backend : "cuda" (WMMA, cœurs tenseurs, G = 4 et D = 256 seulement), "triton" (FMA, générique)
    ou "auto" (cuda si l'extension est chargée et la forme compatible)."""

    def __init__(self, partition_size: int = 512, block_n: int = 32, num_warps: int = 8,
                 num_stages: int = 1, reduce_warps: int = 4, backend: str = "auto") -> None:
        self.backend = backend
        self.partition_size = int(partition_size)
        self.block_n = int(block_n)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)
        self.reduce_warps = int(reduce_warps)
        self._ws: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def num_partitions(self, kv_capacity: int) -> int:
        return max(1, math.ceil(int(kv_capacity) / self.partition_size))

    def workspace(self, n_tokens: int, num_q_heads: int, num_parts: int, head_dim: int,
                  device: torch.device):
        # Capacité arrondie à la puissance de 2 supérieure (≥ 16) : peu de tampons distincts,
        # et un tampon assez grand pour toutes les tailles de capture.
        cap = max(16, 1 << (int(n_tokens) - 1).bit_length())
        key = (cap, num_q_heads, num_parts, head_dim, str(device))
        ws = self._ws.get(key)
        if ws is None:
            # Réutilise un tampon plus grand déjà alloué si possible.
            for (c, hqq, pp, dd, dev), buf in self._ws.items():
                if c >= cap and hqq == num_q_heads and pp == num_parts and dd == head_dim and dev == str(device):
                    return buf
            po = torch.empty((cap, num_q_heads, num_parts, head_dim), dtype=torch.float32, device=device)
            pm = torch.empty((cap, num_q_heads, num_parts), dtype=torch.float32, device=device)
            pl = torch.empty_like(pm)
            ws = (po, pm, pl)
            self._ws[key] = ws
        return ws

    def __call__(self, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                 block_table: torch.Tensor, seq_lens: torch.Tensor, query_start_loc: torch.Tensor,
                 out: torch.Tensor, *, scale: float, max_query_len: int, kv_capacity: int,
                 softcap: float = 0.0, per_token: bool = False) -> torch.Tensor:
        """query/out [T, H_q, D] ; key/value_cache [num_blocks, page, H_kv, D] ;
        block_table [R, max_blocks] int32 ; seq_lens [R] ; query_start_loc [R + 1] (sur le GPU).
        per_token=True : block_table [T, max_blocks] et seq_lens [T] par jeton (métadonnées
        persistantes « small-query » du fork : smallq_decode_block_table / _seq_lens).

        `max_query_len` borne q_len par requête (constexpr) ; `kv_capacity` fixe le nombre
        de partitions (statique pour la capture en graphe : max_model_len ou capacité du
        block_table)."""
        n_tokens, hq, d = query.shape
        nb, page, hkv, dk = key_cache.shape
        assert d == dk and hq % hkv == 0
        assert query.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1
        assert out.stride(2) == 1
        g = hq // hkv
        num_reqs = int(query_start_loc.shape[0] - 1)
        if not per_token:
            num_reqs = int(min(num_reqs, seq_lens.shape[0], block_table.shape[0]))
        if n_tokens == 0 or num_reqs <= 0:
            out.zero_()
            return out
        maxq = int(max_query_len)
        block_m = max(16, triton.next_power_of_2(g * maxq))
        n_parts = self.num_partitions(kv_capacity)
        po, pm, pl = self.workspace(n_tokens, hq, n_parts, d, query.device)
        use_cuda = self.backend == "cuda" or (
            self.backend == "auto" and g == 4 and d == 256 and maxq <= 8
            and self.partition_size % 32 == 0 and charger_cuda())
        if use_cuda:
            torch.ops.gqa_verify.partition(query, key_cache, value_cache, block_table, seq_lens,
                                           query_start_loc, po, pm, pl, num_reqs, n_parts,
                                           self.partition_size, float(scale), float(softcap or 0.0),
                                           int(bool(per_token)))
        else:
            _gqa_verify_partition_kernel[(num_reqs, hkv, n_parts)](
                query, key_cache, value_cache, block_table, seq_lens, query_start_loc,
                po, pm, pl,
                n_tokens, float(scale), float(softcap) if softcap else 1.0,
                query.stride(0), query.stride(1),
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                block_table.stride(0),
                po.stride(0), po.stride(1), po.stride(2),
                pm.stride(0), pm.stride(1),
                PAGE=page, G=g, MAXQ=maxq, BLOCK_M=block_m, BLOCK_N=self.block_n, D=d,
                PS=self.partition_size, USE_SOFTCAP=bool(softcap), PER_TOKEN=bool(per_token),
                num_warps=self.num_warps, num_stages=self.num_stages,
            )
        _gqa_verify_reduce_kernel[(n_tokens, hq)](
            out, po, pm, pl, seq_lens, query_start_loc,
            num_reqs,
            out.stride(0), out.stride(1),
            po.stride(0), po.stride(1), po.stride(2),
            pm.stride(0), pm.stride(1),
            D=d, PS=self.partition_size, P_BLOCK=triton.next_power_of_2(n_parts),
            REQ_BLOCK=triton.next_power_of_2(num_reqs + 1), PER_TOKEN=bool(per_token),
            num_warps=self.reduce_warps,
        )
        return out

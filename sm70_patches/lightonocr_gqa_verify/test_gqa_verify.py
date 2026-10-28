"""Test autonome du noyau gqa_verify (hors serveur), aux formes de production.

  python test_gqa_verify.py [--page 544] [--mem-gb 1.4] [--fork]

- référence : attention fp32 en torch, sur le KV regroupé de chaque requête (masque causal décalé) ;
- `--fork` : compare aussi au noyau actuel du fork (flash_attn_v100.flash_attn_decode_paged, voie
  « small-query » : un jeton = une ligne de décodage), si le module est importable (image
  ocr/1cat-vllm) ;
- mesure : écart max / moyen, temps par appel (événements CUDA), bande passante effective.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gqa_verify import GQAVerifyAttention  # noqa: E402

HQ, HKV, D = 16, 4, 256
SCALE = 1.0 / math.sqrt(D)


def faire_cache(num_blocks: int, page: int, pad: int, dev):
    """Cache hybride comme en production : kv_cache[num_blocks, 2, page, H, D] vu par [:, 0] / [:, 1],
    avec un pas de bloc éventuellement plus grand (pages partagées avec l'état DeltaNet)."""
    per = 2 * page * HKV * D
    base = torch.empty(num_blocks * (per + pad), dtype=torch.float16, device=dev)
    base.normal_(0, 1)
    kv = base.as_strided((num_blocks, 2, page, HKV, D), (per + pad, page * HKV * D, HKV * D, D, 1))
    return kv[:, 0], kv[:, 1]


def faire_lot(qlens, ctxs, page, num_blocks, dev, gen):
    """qlens[r] jetons de requête, ctxs[r] = seq_len (inclut les jetons de requête)."""
    R = len(qlens)
    max_blocks = max(math.ceil(c / page) for c in ctxs) + 1
    perm = torch.randperm(num_blocks, generator=gen).tolist()
    bt = torch.full((R, max_blocks), -1, dtype=torch.int32)
    k = 0
    for r, c in enumerate(ctxs):
        nb = math.ceil(c / page)
        bt[r, :nb] = torch.tensor(perm[k:k + nb], dtype=torch.int32)
        k += nb
    assert k <= num_blocks
    qsl = torch.zeros(R + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(qlens, dtype=torch.int32), 0)
    return bt.to(dev), torch.tensor(ctxs, dtype=torch.int32, device=dev), qsl.to(dev)


def reference(q, kc, vc, bt, seq_lens, qsl, page, n_tokens):
    out = torch.zeros((n_tokens, HQ, D), dtype=torch.float32, device=q.device)
    bt_c, sl_c, qsl_c = bt.cpu(), seq_lens.cpu(), qsl.cpu()
    G = HQ // HKV
    for r in range(len(sl_c)):
        qs, qe = int(qsl_c[r]), int(qsl_c[r + 1])
        ql = qe - qs
        if ql <= 0:
            continue
        eff = max(int(sl_c[r]), ql)
        pos = torch.arange(eff)
        blocks = bt_c[r, pos // page].long().to(q.device)
        slots = (pos % page).to(q.device)
        K = kc[blocks, slots].float().repeat_interleave(G, dim=1)  # [eff, HQ, D]
        V = vc[blocks, slots].float().repeat_interleave(G, dim=1)
        Q = q[qs:qe].float()  # [ql, HQ, D]
        S = torch.einsum("qhd,khd->hqk", Q, K) * SCALE
        lim = eff - ql + torch.arange(ql, device=q.device) + 1
        mask = torch.arange(eff, device=q.device)[None, :] < lim[:, None]
        S = S.masked_fill(~mask[None], float("-inf"))
        P = torch.softmax(S, dim=-1)
        out[qs:qe] = torch.einsum("hqk,khd->qhd", P, V)
    return out


def chrono(fn, it=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it  # ms


def deplier(bt, seq_lens, qsl, n_tokens):
    """Métadonnées par jeton, comme le constructeur du fork (smallq_decode_block_table / _seq_lens) :
    ligne de block_table répétée, seq_len de décodage croissant, jetons de remplissage à 0."""
    qlens = (qsl[1:] - qsl[:-1]).long()
    eff = torch.maximum(seq_lens, qlens.to(seq_lens.dtype))
    nt = int(qsl[-1])
    bt_tok = torch.zeros((n_tokens, bt.shape[1]), dtype=torch.int32, device=bt.device)
    sl_tok = torch.zeros((n_tokens,), dtype=torch.int32, device=bt.device)
    bt_tok[:nt] = torch.repeat_interleave(bt.clamp_min(0), qlens, dim=0)
    starts = torch.repeat_interleave(qsl[:-1].long(), qlens)
    idx = torch.arange(nt, device=bt.device)
    sl_tok[:nt] = (torch.repeat_interleave(eff, qlens) - torch.repeat_interleave(qlens, qlens)
                   + (idx - starts) + 1).to(torch.int32)
    return bt_tok, sl_tok


def appel_fork(fork, q, kc, vc, bt, seq_lens, qsl, n_tokens, out):
    """Voie actuelle : chaque jeton = une ligne de décodage (block_table répétée, seq_len croissant)."""
    R = len(seq_lens)
    qlens = (qsl[1:] - qsl[:-1]).long()
    eff = torch.maximum(seq_lens, qlens.to(seq_lens.dtype))
    bt_tok = torch.repeat_interleave(bt.clamp_min(0), qlens, dim=0).contiguous()
    starts = torch.repeat_interleave(qsl[:-1].long(), qlens)
    eff_tok = torch.repeat_interleave(eff, qlens)
    ql_tok = torch.repeat_interleave(qlens, qlens)
    idx = torch.arange(int(qsl[-1]), device=q.device)
    sl_tok = (eff_tok - ql_tok + (idx - starts) + 1).to(torch.int32).contiguous()
    maxs = int(sl_tok.max())
    nt = int(qsl[-1])

    def run():
        fork.flash_attn_decode_paged(q[:nt], kc, vc, bt_tok, sl_tok, softmax_scale=SCALE, out=out[:nt],
                                     kv_cache_dtype="auto", k_scale=1.0, v_scale=1.0,
                                     max_seq_len_hint=maxs)
    return run


def cas(nom, qlens, ctxs, page, args, dev, gen, fork, res, extra_tokens=0, bench=False):
    R = len(qlens)
    num_blocks = sum(math.ceil(c / page) for c in ctxs) + 7
    kc, vc = faire_cache(num_blocks, page, args.pad, dev)
    bt, sl, qsl = faire_lot(qlens, ctxs, page, num_blocks, dev, gen)
    n_tokens = sum(qlens) + extra_tokens
    # query vue d'un qkv fusionné (pas de jeton > HQ*D), comme dans le modèle
    qkv = torch.randn((n_tokens, (HQ + 2 * HKV) * D), dtype=torch.float16, device=dev)
    q = qkv[:, : HQ * D].view(n_tokens, HQ, D)
    out = torch.full((n_tokens, HQ, D), float("nan"), dtype=torch.float16, device=dev)
    att = GQAVerifyAttention(partition_size=args.ps, block_n=args.bn, num_warps=args.warps, backend=args.backend)
    cap = args.max_model_len
    maxq = max(5, max(qlens))
    att(q, kc, vc, bt, sl, qsl, out, scale=SCALE, max_query_len=maxq, kv_capacity=cap)
    torch.cuda.synchronize()
    ref = reference(q, kc, vc, bt, sl, qsl, page, n_tokens)
    diff = (out.float() - ref).abs()
    ligne = {"cas": nom, "R": R, "jetons": n_tokens, "page": page,
             "ecart_max": float(diff.max()), "ecart_moy": float(diff.mean()),
             "nan": bool(torch.isnan(out).any()),
             "ref_absmax": float(ref.abs().max())}
    bt_tok, sl_tok = deplier(bt, sl, qsl, n_tokens)
    out_tok = torch.full_like(out, float("nan"))
    att(q, kc, vc, bt_tok, sl_tok, qsl, out_tok, scale=SCALE, max_query_len=maxq, kv_capacity=cap, per_token=True)
    torch.cuda.synchronize()
    ligne["par_jeton_identique"] = bool(torch.equal(out_tok, out))
    if extra_tokens:
        ligne["remplissage_nul"] = bool((out[sum(qlens):] == 0).all())
    if fork is not None:
        outf = torch.zeros_like(out)
        run_f = appel_fork(fork, q, kc, vc, bt, sl, qsl, n_tokens, outf)
        run_f()
        torch.cuda.synchronize()
        df = (outf[: sum(qlens)].float() - ref[: sum(qlens)]).abs()
        ligne["fork_ecart_max"] = float(df.max())
        ligne["fork_ecart_moy"] = float(df.mean())
        dd = (outf[: sum(qlens)].float() - out[: sum(qlens)].float()).abs()
        ligne["nouveau_vs_fork_max"] = float(dd.max())
        ligne["nouveau_vs_fork_identiques_%"] = float((dd == 0).float().mean() * 100)
    if bench:
        t_new = chrono(lambda: att(q, kc, vc, bt, sl, qsl, out, scale=SCALE, max_query_len=maxq, kv_capacity=cap))
        kv_bytes = sum(max(c, ql) for c, ql in zip(ctxs, qlens)) * HKV * D * 2 * 2
        ligne["ms_nouveau"] = round(t_new, 4)
        ligne["Go_s_nouveau"] = round(kv_bytes / t_new / 1e6, 1)
        ligne["Mo_kv_utiles"] = round(kv_bytes / 1e6, 1)
        if fork is not None:
            t_f = chrono(run_f)
            ligne["ms_fork"] = round(t_f, 4)
            ligne["Go_s_fork_utiles"] = round(kv_bytes / t_f / 1e6, 1)
            ligne["acceleration"] = round(t_f / t_new, 2)
    res.append(ligne)
    print(json.dumps(ligne, ensure_ascii=False), flush=True)
    del kc, vc, qkv, out, ref
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=544)
    ap.add_argument("--pad", type=int, default=0, help="éléments ajoutés au pas d'un bloc")
    ap.add_argument("--ps", type=int, default=512)
    ap.add_argument("--bn", type=int, default=32)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--mem-gb", type=float, default=1.4)
    ap.add_argument("--fork", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "triton"])
    ap.add_argument("--rapide", action="store_true", help="seulement le cas production + banc")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    tot = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / tot), dev)
    fork = None
    if args.fork:
        import flash_attn_v100 as fork  # noqa: F811
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    res = []
    print(json.dumps({"torch": torch.__version__, "triton": __import__("triton").__version__,
                      "gpu": torch.cuda.get_device_name(), "page": args.page, "ps": args.ps,
                      "bn": args.bn, "warps": args.warps, "pad": args.pad, "backend": args.backend,
                      "cuda_charge": __import__("gqa_verify").charger_cuda()}), flush=True)
    rnd = torch.Generator().manual_seed(1)
    ctx_prod = [int(x) for x in torch.randint(1100, 2501, (48,), generator=rnd)]
    cas("production 48×5, ctx 1100-2500", [5] * 48, ctx_prod, args.page, args, dev, gen, fork, res, bench=True)
    if not args.rapide:
        ql_mix = [5] * 40 + [1] * 8
        cas("48 req. dont 8 à 1 jeton", ql_mix, ctx_prod, args.page, args, dev, gen, fork, res, bench=True)
        cas("contextes courts (6-40)", [5, 5, 1, 3, 5, 2], [6, 9, 5, 17, 40, 2], args.page, args, dev, gen, fork, res)
        cas("frontières de page", [5] * 6, [args.page, args.page + 1, args.page - 1, 2 * args.page,
                                            2 * args.page + 4, 3 * args.page - 2], args.page, args, dev, gen, fork, res)
        cas("frontières de partition", [5] * 4, [args.ps, args.ps + 1, args.ps + 3, 2 * args.ps + 2],
            args.page, args, dev, gen, fork, res)
        cas("long (9 000)", [5, 5], [9000, 7777], args.page, args, dev, gen, fork, res)
        cas("remplissage graphe (+12 jetons hors requête)", [5] * 8, ctx_prod[:8], args.page, args, dev, gen,
            fork, res, extra_tokens=12)
        cas("requête à 0 jeton au milieu", [5, 0, 5, 1], [1300, 0, 1500, 1200], args.page, args, dev, gen, fork, res)
        cas("32 req. (taille de graphe 160)", [5] * 32, ctx_prod[:32], args.page, args, dev, gen, fork, res, bench=True)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

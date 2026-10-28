"""Test autonome du noyau vit_fa (attention de la tour vision, d = 64, non causale), hors serveur.

  python test_vit_fa.py [--mem-gb 0.8]

Référence : SDPA fp32 par séquence. Comparaison : SDPA fp16 actuelle (mem-efficient cutlassF sm70, la voie
du serveur). Entrées : vues à pas d'un qkv fusionné [S, 3·H·D] comme dans Qwen3-VL. Temps alternés."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

ici = os.path.dirname(os.path.abspath(__file__))
H, DH = 16, 64


def chrono(fn, it=5):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it


def sdpa_par_seq(q, k, v, lens, scale, dtype):
    outs, s = [], 0
    for n in lens:
        qi, ki, vi = (x[s:s + n].transpose(0, 1).unsqueeze(0).to(dtype) for x in (q, k, v))
        outs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=scale)[0].transpose(0, 1))
        s += n
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-gb", type=float, default=0.8)
    ap.add_argument("--json", default=None)
    ap.add_argument("--variante", type=int, default=32)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory, dev)
    torch.ops.load_library(os.path.join(ici, "build", "vit_fa_cuda.so"))
    scale = DH ** -0.5
    res = []
    cas = [("colonne 5 888 patchs", [5888], True), ("bandeau 256", [256], True), ("5 900 (dernier bloc partiel)", [5900], True),
           ("77", [77], False), ("1", [1], False), ("lot de 2 images 5 888 + 1 012", [5888, 1012], True),
           ("3 images 64 + 65 + 130", [64, 65, 130], False)]
    for nom, lens, banc in cas:
        torch.manual_seed(0)
        S = sum(lens)
        qkv = torch.randn((S, 3 * H * DH), dtype=torch.float16, device=dev)
        q, k, v = (qkv[:, i * H * DH:(i + 1) * H * DH].view(S, H, DH) for i in range(3))
        # ordre de grandeur réaliste : logits jusqu'à ~±30
        q.mul_(2.0)
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
        out = torch.full((S, H, DH), float("nan"), dtype=torch.float16, device=dev)
        f_new = lambda: torch.ops.vit_fa.forward(q, k, v, out, cu, S, scale, args.variante)  # noqa: E731
        f_new()
        torch.cuda.synchronize()
        ref = sdpa_par_seq(q, k, v, lens, scale, torch.float32)
        cur = sdpa_par_seq(q, k, v, lens, scale, torch.float16)
        d_new = (out.float() - ref).abs()
        d_cur = (cur.float() - ref).abs()
        dd = (out.float() - cur.float()).abs()
        ligne = {"cas": nom, "S": S, "ecart_max": float(d_new.max()), "ecart_moy": float(d_new.mean()),
                 "sdpa_actuelle_ecart_max": float(d_cur.max()), "sdpa_actuelle_ecart_moy": float(d_cur.mean()),
                 "nouveau_vs_actuel_max": float(dd.max()), "identiques_%": float((dd == 0).float().mean() * 100),
                 "nan": bool(torch.isnan(out).any()), "ref_absmax": float(ref.abs().max())}
        if banc:
            f_cur = lambda: sdpa_par_seq(q, k, v, lens, scale, torch.float16)  # noqa: E731
            tn, tc = [], []
            for _ in range(6):
                tn.append(chrono(f_new))
                tc.append(chrono(f_cur))
            flop = sum(4 * n * n * DH * H for n in lens)
            ligne.update({"ms_nouveau_med": round(statistics.median(tn), 3), "ms_actuel_med": round(statistics.median(tc), 3),
                          "ms_nouveau_min": round(min(tn), 3), "ms_actuel_min": round(min(tc), 3),
                          "acceleration_med": round(statistics.median(tc) / statistics.median(tn), 2),
                          "TFLOPS_nouveau_min": round(flop / min(tn) / 1e9, 1)})
        res.append(ligne)
        print(json.dumps(ligne, ensure_ascii=False), flush=True)
        del qkv, out, ref, cur
        torch.cuda.empty_cache()
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

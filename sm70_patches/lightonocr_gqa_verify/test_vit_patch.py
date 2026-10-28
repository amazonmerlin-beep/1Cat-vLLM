"""Test du branchement vit_attn_fa sur le VRAI module du fork (image ocr/1cat-vllm) : la méthode patchée
`MMEncoderAttention._forward_sdpa` est comparée à l'originale (`__wrapped__`, SDPA cutlassF), sur des entrées
au format de Qwen3-VL ([S, H·D] issues d'un qkv fusionné, cu_seqlens sur le GPU), plus les replis."""
from __future__ import annotations

import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vit_attn_fa as P  # noqa: E402

dev = torch.device("cuda", 0)
torch.cuda.set_per_process_memory_fraction(0.8 * 2**30 / torch.cuda.get_device_properties(0).total_memory, dev)
ok = True


def verifie(nom, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK   " if cond else "ÉCHEC"), nom, detail, flush=True)


mod = importlib.import_module(P.CIBLE)
P.appliquer(mod)
Cls = mod.MMEncoderAttention
verifie("méthode remplacée", getattr(Cls._forward_sdpa, "__wrapped__", None) is not None)
orig = Cls._forward_sdpa.__wrapped__


def instance(h=16, hkv=16, d=64):
    o = Cls.__new__(Cls)
    torch.nn.Module.__init__(o)
    o.num_heads, o.num_kv_heads, o.head_size, o.scale = h, hkv, d, d ** -0.5
    return o


H, D = 16, 64
for nom, lens in (("1 colonne 5 888", [5888]), ("2 images 1 024 + 777", [1024, 777]), ("cu rembourré (0 en fin)", [300, 0, 0])):
    S = sum(lens)
    torch.manual_seed(1)
    qkv = torch.randn((S, 3 * H * D), dtype=torch.float16, device=dev)
    q, k, v = (qkv[:, i * H * D:(i + 1) * H * D].unsqueeze(0) for i in range(3))  # [1, S, H·D] vues
    q = q * 2
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    o = instance()
    a = Cls._forward_sdpa(o, q, k, v, cu)
    b = orig(o, q, k, v, cu)
    torch.cuda.synchronize()
    e = (a.float() - b.float()).abs().max().item()
    verifie(f"{nom} : patché ≈ original", a.shape == b.shape and e < 4e-3 and not torch.isnan(a).any(),
            f"forme {tuple(a.shape)} écart max {e:.1e}")
# 4D sans cu_seqlens
q4 = torch.randn((1, 500, H, D), dtype=torch.float16, device=dev)
k4, v4 = torch.randn_like(q4), torch.randn_like(q4)
o = instance()
a, b = Cls._forward_sdpa(o, q4, k4, v4, None), orig(o, q4, k4, v4, None)
e = (a.float() - b.float()).abs().max().item()
verifie("4D sans cu_seqlens", a.shape == b.shape and e < 4e-3, f"{tuple(a.shape)} {e:.1e}")
# replis
n0 = dict(P._ETAT["replis"])
Cls._forward_sdpa(instance(d=80), torch.randn((1, 64, H * 80), dtype=torch.float16, device=dev),
                  torch.randn((1, 64, H * 80), dtype=torch.float16, device=dev),
                  torch.randn((1, 64, H * 80), dtype=torch.float16, device=dev), None)
verifie("repli d=80", P._ETAT["replis"].get("forme (d≠64 ou GQA)", 0) == n0.get("forme (d≠64 ou GQA)", 0) + 1)
x = torch.randn((1, 64, H * D), dtype=torch.float32, device=dev)
Cls._forward_sdpa(instance(), x, x, x, None)
verifie("repli fp32", P._ETAT["replis"].get("dtype", 0) >= 1)
x2 = torch.randn((2, 64, H * D), dtype=torch.float16, device=dev)
r = Cls._forward_sdpa(instance(), x2, x2, x2, None)
verifie("repli lot 2", P._ETAT["replis"].get("lot>1 ou q≠kv", 0) >= 1 and r.shape == (2, 64, H * D))
os.environ["VIT_FA"] = "0"
Cls._forward_sdpa(instance(), q4, k4, v4, None)
verifie("VIT_FA=0", P._ETAT["replis"].get("désactivé", 0) >= 1)
print("TOUT OK" if ok else "DES ÉCHECS", flush=True)
sys.exit(0 if ok else 1)

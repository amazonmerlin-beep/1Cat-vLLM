"""Tests du module de branchement attn_gqa_verify (hors serveur, image ocr/1cat-vllm) :

  a) aiguillage sur un faux module : forme gérée → noyau (= référence fp32) ; groupe ≠ 4, q > 8,
     GQA_VERIFY=0, kv fp8 → méthode d'origine ;
  b) capture en graphe CUDA puis rejeu après changement du CONTENU des métadonnées persistantes ;
  c) lot mixte (prefill + lignes petit q) : lignes de décodage = référence, lignes de prefill intactes ;
  d) (--vrai) le vrai module du fork : appliquer() remplace bien les deux méthodes.
"""
from __future__ import annotations

import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_gqa_verify as P  # noqa: E402
from test_gqa_verify import HKV, HQ, SCALE, D, deplier, faire_cache, faire_lot, reference  # noqa: E402

dev = torch.device("cuda", 0)
torch.cuda.set_per_process_memory_fraction(0.8 * 2**30 / torch.cuda.get_device_properties(0).total_memory, dev)
PAGE = 544
ok = True


def verifie(nom, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK   " if cond else "ÉCHEC"), nom, detail, flush=True)


class FauxImpl:
    def __init__(self):
        self.scale = SCALE
        self.kv_cache_dtype = "auto"
        self.smallq_decode_max_query_len = 5
        self.alibi_slopes = None
        self.sinks = None
        self.logits_soft_cap = 0
        self.prefix_anchored_decode_window = None

    def _flash_v100_window_size(self, causal=True):
        return (-1, -1)

    def _flash_v100_small_query_prefill_as_decode(self, *a, **k):
        return "ORIG"

    def _run_prefill_prefix_decode_rows(self, *a, **k):
        return "ORIG_MIXTE"


faux = types.SimpleNamespace(FlashAttnV100Impl=FauxImpl, logger=None,
                             _is_cuda_graph_capturing=lambda t: torch.cuda.is_current_stream_capturing(),
                             _record_route=lambda r: None)
P.appliquer(faux)
impl = FauxImpl()
layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)

gen = torch.Generator().manual_seed(0)
rnd = torch.Generator().manual_seed(3)
R = 16
ctx = [int(x) for x in torch.randint(600, 2600, (R,), generator=rnd)]
qlens = [5] * R
nb = R * math.ceil(2600 / PAGE) + 8
kc, vc = faire_cache(nb, PAGE, 64, dev)
bt, sl, qsl = faire_lot(qlens, ctx, PAGE, nb, dev, gen)
T = sum(qlens) + 5  # 5 jetons de remplissage (graphe)
q = torch.randn((T, HQ, D), dtype=torch.float16, device=dev)


def meta(bt, sl, qsl, persistant=True):
    m = types.SimpleNamespace(num_actual_tokens=T, causal=True, max_model_len=10240,
                              block_table=bt, seq_lens=sl, query_start_loc=qsl)
    if persistant:
        m.smallq_decode_block_table, m.smallq_decode_seq_lens = deplier(bt, sl, qsl, T)
        m.smallq_query_start_loc = qsl
    return m


# a) aiguillage
ref = reference(q, kc, vc, bt, sl, qsl, PAGE, T)
for persist in (True, False):
    out = torch.full((T, HQ, D), float("nan"), dtype=torch.float16, device=dev)
    r = impl._flash_v100_small_query_prefill_as_decode(layer, q, kc, vc, meta(bt, sl, qsl, persist), out, None, None)
    torch.cuda.synchronize()
    e = (out.float() - ref).abs().max().item()
    verifie(f"a) voie groupée (métadonnées {'persistantes' if persist else 'eager'})", r is out and e < 5e-4,
            f"écart max {e:.2e}")
q8 = torch.randn((T, 8 * HKV, D), dtype=torch.float16, device=dev)
verifie("a) groupe 8 → origine", impl._flash_v100_small_query_prefill_as_decode(
    layer, q8, kc, vc, meta(bt, sl, qsl), None, None, None) == "ORIG")
impl.smallq_decode_max_query_len = 9
verifie("a) q > 8 → origine", impl._flash_v100_small_query_prefill_as_decode(
    layer, q, kc, vc, meta(bt, sl, qsl), None, None, None) == "ORIG")
impl.smallq_decode_max_query_len = 5
impl.kv_cache_dtype = "fp8_e5m2"
verifie("a) kv fp8 → origine", impl._flash_v100_small_query_prefill_as_decode(
    layer, q, kc, vc, meta(bt, sl, qsl), None, None, None) == "ORIG")
impl.kv_cache_dtype = "auto"
os.environ["GQA_VERIFY"] = "0"
verifie("a) GQA_VERIFY=0 → origine", impl._flash_v100_small_query_prefill_as_decode(
    layer, q, kc, vc, meta(bt, sl, qsl), None, None, None) == "ORIG")
os.environ["GQA_VERIFY"] = "1"

# b) graphe CUDA : capture, puis on change le contenu des tampons persistants et on rejoue
m = meta(bt, sl, qsl)
out = torch.zeros((T, HQ, D), dtype=torch.float16, device=dev)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    impl._flash_v100_small_query_prefill_as_decode(layer, q, kc, vc, m, out, None, None)  # échauffement
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    impl._flash_v100_small_query_prefill_as_decode(layer, q, kc, vc, m, out, None, None)
ctx2 = [int(x) for x in torch.randint(600, 2600, (R,), generator=rnd)]
ql2 = [5] * (R - 3) + [1, 0, 0]  # requêtes de fin à 0 jeton = remplissage de graphe
bt2, sl2, qsl2 = faire_lot(ql2, [c if l else 0 for c, l in zip(ctx2, ql2)], PAGE, nb, dev, gen)
bt2_tok, sl2_tok = deplier(bt2, sl2, qsl2, T)
m.smallq_decode_block_table.copy_(bt2_tok)
m.smallq_decode_seq_lens.copy_(sl2_tok)
m.smallq_query_start_loc.copy_(qsl2)
q.copy_(torch.randn_like(q))
g.replay()
torch.cuda.synchronize()
ref2 = reference(q, kc, vc, bt2, sl2, qsl2, PAGE, T)
e = (out.float() - ref2).abs().max().item()
verifie("b) rejeu de graphe après changement des métadonnées", e < 5e-4, f"écart max {e:.2e}")
verifie("b) jetons hors requête nuls après rejeu", bool((out[sum(ql2):] == 0).all()))

# c) lot mixte : requête 0 et 3 en prefill (q = 300 et 40), les autres en petit q
qlm = [300, 5, 5, 40, 5, 1]
ctm = [300, 1500, 900, 1240, 2000, 700]
btm, slm, qslm = faire_lot(qlm, ctm, PAGE, nb, dev, gen)
Tm = sum(qlm)
qm = torch.randn((Tm, HQ, D), dtype=torch.float16, device=dev)
outm = torch.full((Tm, HQ, D), 7.0, dtype=torch.float16, device=dev)
mm = types.SimpleNamespace(num_actual_tokens=Tm, causal=True, max_model_len=10240, block_table=btm, seq_lens=slm,
                           query_start_loc=qslm)
rows = impl._run_prefill_prefix_decode_rows(layer, qm, kc, vc, mm, outm, qslm.cpu(), slm.cpu(), (-1, -1))
torch.cuda.synchronize()
refm = reference(qm, kc, vc, btm, slm, qslm, PAGE, Tm)
qc = qslm.cpu().tolist()
dec = torch.zeros(Tm, dtype=torch.bool)
for i in (1, 2, 4, 5):
    dec[qc[i]:qc[i + 1]] = True
e = (outm[dec.to(dev)].float() - refm[dec.to(dev)]).abs().max().item()
verifie("c) lignes rendues = {1, 2, 4, 5}", rows == {1, 2, 4, 5}, str(rows))
verifie("c) lignes petit q = référence", e < 5e-4, f"écart max {e:.2e}")
verifie("c) lignes de prefill intactes", bool((outm[~dec.to(dev)] == 7.0).all()))
# c') même attn_metadata pour 8 couches (cache par pas), puis pas suivant avec seq_lens +1 (nouvel objet)
ok_c = True
for couche in range(8):
    qm2 = torch.randn_like(qm)
    o2 = torch.full_like(outm, 7.0)
    r2 = impl._run_prefill_prefix_decode_rows(layer, qm2, kc, vc, mm, o2, qslm.cpu(), slm.cpu(), (-1, -1))
    torch.cuda.synchronize()
    ref2 = reference(qm2, kc, vc, btm, slm, qslm, PAGE, Tm)
    ok_c &= r2 == {1, 2, 4, 5} and (o2[dec.to(dev)].float() - ref2[dec.to(dev)]).abs().max().item() < 5e-4
verifie("c') 8 couches avec le même attn_metadata (cache)", ok_c)
slm3 = slm + torch.tensor([0, 1, 1, 0, 1, 1], dtype=slm.dtype, device=dev)
mm.seq_lens = slm3  # même objet réutilisé avec d'autres longueurs : le garde-fou doit reconstruire
o3 = torch.full_like(outm, 7.0)
impl._run_prefill_prefix_decode_rows(layer, qm, kc, vc, mm, o3, qslm.cpu(), slm3.cpu(), (-1, -1))
torch.cuda.synchronize()
ref3 = reference(qm, kc, vc, btm, slm3, qslm, PAGE, Tm)
e3 = (o3[dec.to(dev)].float() - ref3[dec.to(dev)]).abs().max().item()
verifie("c') objet réutilisé avec seq_lens changés → reconstruit", e3 < 5e-4, f"écart max {e3:.2e}")

if "--vrai" in sys.argv:
    import importlib
    mod = importlib.import_module(P.CIBLE)
    P.appliquer(mod)
    Impl = mod.FlashAttnV100Impl
    verifie("d) vrai module : méthode petit q remplacée",
            getattr(Impl._flash_v100_small_query_prefill_as_decode, "__wrapped__", None) is not None)
    verifie("d) vrai module : méthode lots mixtes remplacée",
            getattr(Impl._run_prefill_prefix_decode_rows, "__wrapped__", None) is not None)
    import inspect
    s1 = list(inspect.signature(Impl._flash_v100_small_query_prefill_as_decode.__wrapped__).parameters)
    s2 = list(inspect.signature(Impl._run_prefill_prefix_decode_rows.__wrapped__).parameters)
    verifie("d) signatures d'origine attendues", s1 == ["self", "layer", "query", "key_cache", "value_cache",
            "attn_metadata", "output", "query_start_loc", "_seq_lens"] and s2 == ["self", "layer", "query", "key_cache",
            "value_cache", "attn_metadata", "out_view", "query_start_loc", "seq_lens", "window_size"], f"{s1} {s2}")
    verifie("d) helpers du module présents", all(hasattr(mod, a) for a in ("_is_cuda_graph_capturing", "_record_route")))

print("TOUT OK" if ok else "DES ÉCHECS", flush=True)
sys.exit(0 if ok else 1)

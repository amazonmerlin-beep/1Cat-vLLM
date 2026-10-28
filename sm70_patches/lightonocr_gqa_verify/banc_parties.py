"""Banc : temps de la passe partition (CUDA) seule, de la réduction seule, et de l'appel complet."""
import math, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gqa_verify as gv
from test_gqa_verify import faire_cache, faire_lot, chrono, HQ, HKV, D, SCALE
dev = torch.device('cuda', 0)
torch.cuda.set_per_process_memory_fraction(0.8 * 2**30 / torch.cuda.get_device_properties(0).total_memory, dev)
gen = torch.Generator().manual_seed(0); rnd = torch.Generator().manual_seed(1)
ctx = [int(x) for x in torch.randint(1100, 2501, (48,), generator=rnd)]
page = 544; nb = sum(math.ceil(c / page) for c in ctx) + 7
kc, vc = faire_cache(nb, page, 0, dev); bt, sl, qsl = faire_lot([5] * 48, ctx, page, nb, dev, gen)
q = torch.randn((240, HQ, D), dtype=torch.float16, device=dev); out = torch.empty_like(q)
for ps in [int(x) for x in sys.argv[1:]] or [512]:
    att = gv.GQAVerifyAttention(partition_size=ps, backend='cuda')
    full = lambda: att(q, kc, vc, bt, sl, qsl, out, scale=SCALE, max_query_len=5, kv_capacity=10240)
    full()
    npart = att.num_partitions(10240)
    po, pm, pl = att.workspace(240, HQ, npart, D, dev)
    part = lambda: torch.ops.gqa_verify.partition(q, kc, vc, bt, sl, qsl, po, pm, pl, 48, npart, ps, SCALE, 0.0, 0)
    red = lambda: gv._gqa_verify_reduce_kernel[(240, HQ)](out, po, pm, pl, sl, qsl, 48, out.stride(0), out.stride(1),
        po.stride(0), po.stride(1), po.stride(2), pm.stride(0), pm.stride(1), D=D, PS=ps,
        P_BLOCK=gv.triton.next_power_of_2(npart), REQ_BLOCK=64, PER_TOKEN=False, num_warps=4)
    t = [chrono(f, 30) for f in (full, part, red, full, part, red)]
    print(f"ps={ps} complet {min(t[0],t[3]):.3f} ms | partition {min(t[1],t[4]):.3f} ms | réduction {min(t[2],t[5]):.3f} ms", flush=True)

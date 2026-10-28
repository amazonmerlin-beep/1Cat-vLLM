"""Banc alterné attention vision : SDPA fp16 actuelle vs vit_fa (variantes 32 et 64), colonne de 5 888 patchs."""
import os, statistics, sys, torch, torch.nn.functional as F
ici = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('cuda', 0)
torch.cuda.set_per_process_memory_fraction(0.8 * 2**30 / torch.cuda.get_device_properties(0).total_memory, dev)
torch.ops.load_library(os.path.join(ici, 'build', 'vit_fa_cuda.so'))
H, D, S = 16, 64, int(sys.argv[1]) if len(sys.argv) > 1 else 5888
qkv = torch.randn((S, 3 * H * D), dtype=torch.float16, device=dev)
q, k, v = (qkv[:, i * H * D:(i + 1) * H * D].view(S, H, D) for i in range(3))
cu = torch.tensor([0, S], dtype=torch.int32, device=dev); out = torch.empty((S, H, D), dtype=torch.float16, device=dev)
sc = D ** -0.5
def chrono(fn, it=3):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / it
f = {'sdpa': lambda: F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), scale=sc),
     'v32': lambda: torch.ops.vit_fa.forward(q, k, v, out, cu, S, sc, 32),
     'v64': lambda: torch.ops.vit_fa.forward(q, k, v, out, cu, S, sc, 64)}
r = {n: [] for n in f}
for _ in range(15):
    for n, fn in f.items(): r[n].append(chrono(fn))
for n, x in r.items():
    print(f"S={S} {n:5s} min {min(x):.3f} ms  médiane {statistics.median(x):.3f} ms  ({4*S*S*D*H/min(x)/1e9:.1f} TFLOPS au min)", flush=True)

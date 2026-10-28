"""Test du patch diag_cpu_pas avec de faux modules (runner, planificateur, cœur) : chronos par pas, classement
décodage / mixte, écart GPU mesuré par événements CUDA, journal toutes les DIAG_CPU_N pas, sorties inchangées."""
import os, sys, time, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["DIAG_CPU_N"] = "10"

class GPUModelRunner:
    speculative_config = types.SimpleNamespace(num_speculative_tokens=4)
    def execute_model(self, so):
        self._update_states(so); self._prepare_inputs(so)
        x = torch.ones(1 << 20, device="cuda"); [x.mul_(1.0001) for _ in range(50)]  # « forward »
        return None
    def _update_states(self, so): time.sleep(0.001)
    def _prepare_inputs(self, so): time.sleep(0.002)
    def sample_tokens(self, g):
        torch.cuda.synchronize(); r = self._sample(); self._bookkeeping_sync(); self.propose_draft_token_ids(); return r
    def _sample(self): return 42
    def _bookkeeping_sync(self): time.sleep(0.001)
    def propose_draft_token_ids(self): time.sleep(0.0005)

class Scheduler:
    def __init__(self): self.i = 0
    def schedule(self):
        self.i += 1; time.sleep(0.0015)
        return types.SimpleNamespace(num_scheduled_tokens={"a": 5, "b": 5} if self.i % 4 else {"a": 900, "b": 5})
    def update_from_output(self, so, out): time.sleep(0.001); return {"sortie": out}

class EngineCore:
    def __init__(self): self.s = Scheduler(); self.r = GPUModelRunner()
    def step(self):
        so = self.s.schedule(); self.r.execute_model(so); out = self.r.sample_tokens(None)
        time.sleep(0.003)  # hors chronos (sortie)
        return self.s.update_from_output(so, out), True

for nom, obj in (("vllm.v1.core.sched.scheduler", types.SimpleNamespace(Scheduler=Scheduler)),
                 ("vllm.v1.engine.core", types.SimpleNamespace(EngineCore=EngineCore))):
    sys.modules[nom] = obj
import diag_cpu_pas as P
P.appliquer(types.SimpleNamespace(GPUModelRunner=GPUModelRunner, logger=None))
ec = EngineCore()
outs = [ec.step() for _ in range(20)]
ok = all(o == ({"sortie": 42}, True) for o in outs) and GPUModelRunner.execute_model.__wrapped__ is not None
print("sorties inchangées et méthodes enveloppées :", ok)
sys.exit(0 if ok else 1)

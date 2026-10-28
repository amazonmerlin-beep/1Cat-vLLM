"""Patch de DIAGNOSTIC « diag_cpu_pas » (n'altère aucune sortie) : où passe le temps CPU d'un pas du moteur.

But : découper les ~17 ms de GPU inoccupé par pas de décodage pur (profil de 13:30) sans profileur torch (qui ne
capture qu'une fois par démarrage) ni py-spy. Mesure, par pas, le temps mur (perf_counter) de :
  EngineCore.step (total), Scheduler.schedule, Scheduler.update_from_output,
  GPUModelRunner.execute_model, ._update_states, ._prepare_inputs, .sample_tokens, ._sample,
  ._bookkeeping_sync, .propose_draft_token_ids (dont le proposeur brouillon.py),
et l'écart GPU (événements CUDA) entre la fin du travail enfilé par execute_model (forward) et le début du
travail enfilé au pas suivant : c'est le GPU inoccupé « entre les pas ».

Les pas sont classés « décodage » (≤ 1 + k jetons par requête) ou « mixte ». Toutes les DIAG_CPU_N pas (200 par
défaut), une ligne par catégorie est écrite sur stderr (et dans le journal vLLM) :
  diag_cpu_pas décodage n=… step=… schedule=… update=… execute=… update_states=… prepare_inputs=…
               sample_tokens=… _sample=… bookkeeping=… propose=… écart_gpu=…   (ms moyennes par pas)
Les temps sont inclusifs (execute ⊃ update_states, prepare_inputs ; sample_tokens ⊃ _sample, bookkeeping, propose ;
sample_tokens inclut aussi l'attente de la fin du forward GPU).

Interface du chargeur : CIBLE + appliquer(module). La cible est le module du runner (importé dans le processus
EngineCore) ; le planificateur et le cœur du moteur sont déjà importés à ce moment-là (sys.modules).
Coût : quelques perf_counter et deux événements CUDA par pas. DIAG_CPU=0 le neutralise.
"""
from __future__ import annotations

import collections
import os
import sys
import time

CIBLE = "vllm.v1.worker.gpu_model_runner"

_N = int(os.environ.get("DIAG_CPU_N", "200"))
_ETAT = {"pas": collections.defaultdict(float), "type": None, "ev_fin": None, "ev_debut": None,
         "tot": {"décodage": collections.defaultdict(float), "mixte": collections.defaultdict(float)},
         "n": collections.Counter(), "profondeur": 0}
_CLES = ("step", "schedule", "update", "execute", "update_states", "prepare_inputs", "sample_tokens", "_sample",
         "bookkeeping", "propose", "écart_gpu")


def _chrono(cls, nom, cle, avant=None, apres=None):
    f = cls.__dict__.get(nom)
    if f is None or getattr(f, "_diag_cpu", False):
        return False

    def w(*a, **k):
        if avant is not None:
            avant(a, k)
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            _ETAT["pas"][cle] += time.perf_counter() - t
            if apres is not None:
                apres(a, k)

    w.__wrapped__ = f
    w._diag_cpu = True
    w.__name__ = getattr(f, "__name__", nom)
    setattr(cls, nom, w)
    return True


def _journal(module_logger):
    for typ in ("décodage", "mixte"):
        n = _ETAT["n"][typ]
        if not n:
            continue
        tot = _ETAT["tot"][typ]
        champs = " ".join(f"{c}={tot[c] / n * 1000:.2f}" for c in _CLES if c in tot)
        ligne = f"diag_cpu_pas {typ} n={n} {champs} (ms/pas)"
        print(ligne, file=sys.stderr, flush=True)
        if module_logger is not None:
            module_logger.info(ligne)
        _ETAT["tot"][typ] = collections.defaultdict(float)
        _ETAT["n"][typ] = 0


def appliquer(module) -> None:
    if os.environ.get("DIAG_CPU", "1") == "0":
        return
    import torch

    logger = getattr(module, "logger", None)
    R = module.GPUModelRunner
    num_spec = {"k": None}

    def avant_execute(a, k):
        so = a[1] if len(a) > 1 else k.get("scheduler_output")
        try:
            vals = list(so.num_scheduled_tokens.values())
            if num_spec["k"] is None:
                sc = getattr(a[0], "speculative_config", None)
                num_spec["k"] = int(getattr(sc, "num_speculative_tokens", 0) or 0)
            _ETAT["type"] = "décodage" if vals and max(vals) <= 1 + num_spec["k"] else ("mixte" if vals else None)
        except Exception:  # noqa: BLE001
            _ETAT["type"] = None
        try:  # début du travail GPU de ce pas
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            prec = _ETAT["ev_fin"]
            if prec is not None:
                _ETAT["ev_debut"] = (prec, ev)
            _ETAT["ev_fin"] = None
        except Exception:  # noqa: BLE001
            pass

    def apres_execute(a, k):
        try:  # fin du forward enfilé
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            _ETAT["ev_fin"] = ev
        except Exception:  # noqa: BLE001
            pass

    _chrono(R, "execute_model", "execute", avant_execute, apres_execute)
    _chrono(R, "_update_states", "update_states")
    _chrono(R, "_prepare_inputs", "prepare_inputs")
    _chrono(R, "sample_tokens", "sample_tokens")
    _chrono(R, "_sample", "_sample")
    _chrono(R, "_bookkeeping_sync", "bookkeeping")
    _chrono(R, "propose_draft_token_ids", "propose")

    import importlib

    def mod(nom):
        m = sys.modules.get(nom)
        if m is None:
            try:
                m = importlib.import_module(nom)
            except Exception:  # noqa: BLE001
                m = None
        return m

    sched_mod = mod("vllm.v1.core.sched.scheduler")
    if sched_mod is not None and hasattr(sched_mod, "Scheduler"):
        _chrono(sched_mod.Scheduler, "schedule", "schedule")
        _chrono(sched_mod.Scheduler, "update_from_output", "update")

    core_mod = mod("vllm.v1.engine.core")
    if core_mod is not None and hasattr(core_mod, "EngineCore"):
        def apres_step(a, k):
            paire = _ETAT.pop("ev_debut", None)
            _ETAT["ev_debut"] = None
            if paire is not None:
                try:  # le pas courant est terminé : les deux événements sont passés
                    _ETAT["pas"]["écart_gpu"] += paire[0].elapsed_time(paire[1]) / 1000.0
                except Exception:  # noqa: BLE001
                    pass
            typ = _ETAT["type"]
            if typ in _ETAT["tot"]:
                for c, v in _ETAT["pas"].items():
                    _ETAT["tot"][typ][c] += v
                _ETAT["n"][typ] += 1
                if _ETAT["n"]["décodage"] + _ETAT["n"]["mixte"] >= _N:
                    _journal(logger)
            _ETAT["pas"] = collections.defaultdict(float)
            _ETAT["type"] = None

        _chrono(core_mod.EngineCore, "step", "step", None, apres_step)
    if logger is not None:
        logger.info("diag_cpu_pas : chronométrage par pas actif (journal toutes les %d pas)", _N)

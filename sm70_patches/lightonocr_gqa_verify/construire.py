"""Compile l'extension CUDA (sm_70) dans l'image de production (même torch/CUDA que le serveur) :

  docker run --rm --entrypoint /opt/venv/bin/python -v $PWD:/patchs ocr/1cat-vllm:1.5.1 /patchs/construire.py

Produit build/gqa_verify_cuda.so (op torch.ops.gqa_verify.partition) et build/vit_fa_cuda.so
(op torch.ops.vit_fa.forward), chargés par torch.ops.load_library. Argument optionnel : noms à construire."""
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
os.environ.setdefault("MAX_JOBS", "2")
import torch.utils.cpp_extension as ce  # noqa: E402

ici = os.path.dirname(os.path.abspath(__file__))
noms = [a for a in sys.argv[1:] if not a.startswith("-")] or ["gqa_verify_cuda", "vit_fa_cuda"]
for nom in noms:
    rep = os.path.join(ici, "build", nom)  # un répertoire par extension (build.ninja distincts)
    os.makedirs(rep, exist_ok=True)
    ce.load(name=nom, sources=[os.path.join(ici, "csrc", nom + ".cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-lineinfo"],
            build_directory=rep, is_python_module=False, verbose="-v" in sys.argv)
    os.replace(os.path.join(rep, nom + ".so"), os.path.join(ici, "build", nom + ".so")) if os.path.exists(
        os.path.join(rep, nom + ".so")) else None
    print("ok", os.path.join(ici, "build", nom + ".so"))

"""Compile l'extension CUDA (sm_70) dans l'image de production (même torch/CUDA que le serveur) :

  docker run --rm --entrypoint /opt/venv/bin/python -v $PWD:/patchs ocr/1cat-vllm:1.5.1 /patchs/construire.py

Produit build/gqa_verify_cuda.so, chargé par torch.ops.load_library (op torch.ops.gqa_verify.partition)."""
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
os.environ.setdefault("MAX_JOBS", "2")
import torch.utils.cpp_extension as ce  # noqa: E402

ici = os.path.dirname(os.path.abspath(__file__))
ce.load(name="gqa_verify_cuda", sources=[os.path.join(ici, "csrc", "gqa_verify_cuda.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-lineinfo"],
        build_directory=os.path.join(ici, "build"), is_python_module=False, verbose="-v" in sys.argv)
print("ok", os.path.join(ici, "build", "gqa_verify_cuda.so"))

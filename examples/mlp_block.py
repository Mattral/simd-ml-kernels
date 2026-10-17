"""Transformer MLP block, out = LayerNorm(x + GeLU(x @ W1) @ W2), with
simd_ml_kernels, checked against PyTorch (or NumPy float64 without torch).

    python examples/mlp_block.py [--tokens 64] [--d-model 768]
"""

import argparse
import sys

import numpy as np

import simd_ml_kernels as sk

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--d-model", type=int, default=768)
args = p.parse_args()
t, d, h = args.tokens, args.d_model, 4 * args.d_model

rng = np.random.default_rng(0)
x = rng.standard_normal((t, d)).astype(np.float32)
w1 = (rng.standard_normal((d, h)) / np.sqrt(d)).astype(np.float32)
w2 = (rng.standard_normal((h, d)) / np.sqrt(h)).astype(np.float32)
gamma = rng.uniform(0.5, 1.5, d).astype(np.float32)
beta = rng.uniform(-0.5, 0.5, d).astype(np.float32)

hidden = sk.sgemm(x, w1)
sk.gelu_inplace(hidden)                    # exact (erf) GeLU, PyTorch's default
y = sk.sgemm(hidden, w2, C=x.copy(), beta=1.0)   # x + hidden @ W2 in one call
out = sk.layer_norm(y, gamma=gamma, beta=beta)

try:
    import torch

    F = torch.nn.functional
    tx, tw1, tw2 = (torch.from_numpy(a) for a in (x, w1, w2))
    ref = F.layer_norm(tx + F.gelu(tx @ tw1) @ tw2, (d,),
                       torch.from_numpy(gamma), torch.from_numpy(beta)).numpy()
    against = f"PyTorch {torch.__version__} (float32)"
except ImportError:
    from math import erf

    x64 = x.astype(np.float64)
    a = x64 @ w1
    a = 0.5 * a * (1.0 + np.vectorize(erf)(a / np.sqrt(2.0)))
    z = x64 + a @ w2
    ref = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5) * gamma + beta
    against = "NumPy float64"

# GEMM rounding (K = 4 * d_model terms) dominates; LayerNorm rescales rows to
# unit std, so the error is absolute in units of that std. Measured on the
# reference laptop: 1.4e-6 at 64x768 and 3.6e-6 at 7x100, vs PyTorch and vs
# float64; 2e-5 leaves a margin of 5x for other summation orders.
err = float(np.max(np.abs(out - ref)))
tol = 2e-5
print(f"MLP block {t}x{d} -> {h} -> {d}, runtime ISA {sk.detected_isa()}: "
      f"max |ours - {against}| = {err:.2e} (tolerance {tol:g})")
sys.exit(0 if err <= tol else 1)

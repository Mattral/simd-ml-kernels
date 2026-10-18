"""Linear layer y = x @ W.T + b with W stored [out_features, in_features], as in
torch.nn.Linear, without transposing W: sgemm(x, W, C=bias rows, beta=1, trans_b=True).
Checked against PyTorch's F.linear (or NumPy float64) with the GEMM error bound
the tests use, for one row (decode) and a batch of rows.

    python examples/linear_trans_b.py [--in 768] [--out 3072]
"""

import argparse
import sys

import numpy as np

import simd_ml_kernels as sk

p = argparse.ArgumentParser()
p.add_argument("--in", dest="d_in", type=int, default=768)
p.add_argument("--out", dest="d_out", type=int, default=3072)
args = p.parse_args()

rng = np.random.default_rng(0)
W = (rng.standard_normal((args.d_out, args.d_in)) / np.sqrt(args.d_in)).astype(np.float32)
b = rng.standard_normal(args.d_out).astype(np.float32)

try:
    import torch

    def reference(x):
        return torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(W),
                                          torch.from_numpy(b)).double().numpy()
    against = f"PyTorch {torch.__version__}"
except ImportError:
    def reference(x):
        return x.astype(np.float64) @ W.T.astype(np.float64) + b
    against = "NumPy float64"

u = 2.0 ** -24
ok = True
for rows in (1, 64):                       # 1: dot-product path; 64: packed path
    x = rng.standard_normal((rows, args.d_in)).astype(np.float32)
    y = np.broadcast_to(b, (rows, args.d_out)).copy()    # C starts as the bias
    sk.sgemm(x, W, y, beta=1.0, trans_b=True)           # y = x @ W.T + y
    # Elementwise bound for both computations (ours and the reference's float32
    # sums): 2 * 2(K+2)u * (|x| @ |W|.T + |b|), K = in_features.
    scale = np.abs(x.astype(np.float64)) @ np.abs(W.T.astype(np.float64)) + np.abs(b)
    ratio = float(np.max(np.abs(y - reference(x)) / (4 * (args.d_in + 2) * u * scale)))
    ok &= ratio <= 1.0
    print(f"linear {rows}x{args.d_in} -> {args.d_out} (trans_b), runtime ISA "
          f"{sk.detected_isa()}: max err / bound vs {against} = {ratio:.3f}")
sys.exit(0 if ok else 1)

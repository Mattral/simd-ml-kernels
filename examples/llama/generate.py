"""Greedy text generation with Llama-3.x built from simd_ml_kernels GEMVs + NumPy.

    SIMD_ML_LLAMA_DIR=~/models/Llama-3.2-1B \\
        python examples/llama/generate.py --prompt "The capital of France is" --tokens 64 --threads 4

--model (or SIMD_ML_LLAMA_DIR) is a Hugging Face checkpoint directory with
config.json, tokenizer.json and *.safetensors. --threads > 1 needs an OpenMP
build (pip install . -C cmake.define.SIMD_ML_OPENMP=ON); the prompt is
prefilled one token at a time. Prints the text, then ms/token for prefill and
decode.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simd_ml_kernels as sk  # noqa: E402
from llama import Llama, encode, load_tokenizer  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--model", default=os.environ.get("SIMD_ML_LLAMA_DIR"),
               help="checkpoint directory (default: $SIMD_ML_LLAMA_DIR)")
p.add_argument("--prompt", required=True)
p.add_argument("--tokens", type=int, default=64, help="new tokens to generate (stops early at EOS)")
p.add_argument("--threads", type=int, default=1)
p.add_argument("--weights", choices=["fp32"], default="fp32")
args = p.parse_args()
if not args.model:
    p.error("--model or SIMD_ML_LLAMA_DIR is required")
if args.threads > 1 and not sk.build_info()["openmp"]:
    print(f"note: this build has no OpenMP; running on 1 thread, not {args.threads}", file=sys.stderr)
sk.set_num_threads(args.threads)

t = time.perf_counter()
model = Llama(args.model, weights=args.weights)
tok = load_tokenizer(args.model)
print(f"loaded {args.model} ({args.weights}, {model.cfg.num_hidden_layers} layers) in "
      f"{time.perf_counter() - t:.1f} s; ISA {sk.detected_isa()}, threads {sk.get_num_threads()}",
      file=sys.stderr)

ids = encode(tok, args.prompt)
stats: dict = {}
out: list[int] = []
printed = 0
sys.stdout.write(args.prompt)
for nxt in model.generate(ids, args.tokens, stats):
    out.append(nxt)
    text = tok.decode(out, skip_special_tokens=True)
    if not text.endswith("�"):  # wait for the rest of a split UTF-8 character
        sys.stdout.write(text[printed:])
        sys.stdout.flush()
        printed = len(text)
print()
pre_ms = 1e3 * stats["prefill_s"] / max(stats["prefill_tokens"], 1)
dec = (f"{1e3 * stats['decode_s'] / stats['decode_steps']:.1f} ms/token over {stats['decode_steps']} steps"
       if stats["decode_steps"] else "no decode steps")
print(f"prefill {stats['prefill_tokens']} tokens: {pre_ms:.1f} ms/token (one at a time); decode: {dec}")

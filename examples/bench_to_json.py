"""Run the benchmark into a JSON file and read it back: the same file format
as benchmarks/results/ (schema simd-ml-kernels-bench/1, docs/BENCHMARKS.md).

    taskset -c 2 python examples/bench_to_json.py out.json [--sizes 256,512]

Pin the process to one core (taskset on Linux, `start /affinity 4` on
Windows); the CLI refuses to run unless the BLAS thread variables are 1, so
this script sets them for the child process.
"""

import argparse
import json
import os
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("json_path")
p.add_argument("--sizes", default="256,512")
p.add_argument("--reps", type=int, default=30)
args = p.parse_args()

env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
subprocess.run([sys.executable, "-m", "simd_ml_kernels", "bench", "--json", args.json_path,
                "--sizes", args.sizes, "--reps", str(args.reps), "--elems", str(1 << 18),
                "--label", "example"], env=env, check=True, stdout=subprocess.DEVNULL)

with open(args.json_path, encoding="utf-8") as f:
    doc = json.load(f)
print(f"{doc['machine']['cpu_model']}, pinned CPUs {doc['machine']['pinned_cpus']}, "
      f"commit {doc['build']['git_commit']}, BLAS {doc['baselines']['blas']['name']}")

m = doc["machine"]
if m.get("stream_read_gbs"):
    print(f"  bandwidth roofs (one core): read {m['stream_read_gbs']:.1f} GB/s, "
          f"STREAM triad {m['stream_triad_gbs']:.1f} GB/s")

# Each case has one result per implementation; ours is "simd_ml_kernels".
# The stream_* rows are measurements of the machine, not comparisons: skipped.
cases = {}
for r in doc["results"]:
    cases.setdefault((r["kernel"], tuple(r["shape"])), {})[r["impl"]] = r
for (kernel, shape), impls in cases.items():
    ours = impls.pop("simd_ml_kernels", None)
    if ours is None or kernel.startswith("stream_"):
        continue
    for impl, base in impls.items():
        speed = base["median_s"] / ours["median_s"]   # > 1: ours is faster
        print(f"  {kernel:<11} {str(shape):<16} {speed:5.2f}x the speed of {impl} "
              f"(medians {ours['median_s'] * 1e3:.3f} / {base['median_s'] * 1e3:.3f} ms)")

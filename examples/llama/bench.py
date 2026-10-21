"""Llama-3.2-1B end to end: our decode (examples/llama) against llama.cpp on the
same GGUF files, plus perplexity. Writes a result JSON (schema
simd-ml-kernels-bench/1) that benchmarks/make_tables.py renders.

    export SIMD_ML_LLAMA_DIR=~/code/models/Llama-3.2-1B           # config, tokenizer, gguf/*.gguf
    export SIMD_ML_GGML_TEST_BACKEND_OPS=~/code/llama.cpp/build/bin/test-backend-ops   # llama.cpp build
    taskset -c 0-3 python examples/llama/bench.py --threads 1,2,4 --label lnl-226v-wsl \\
        --core "P-cores 0-3" --json benchmarks/results/<label>_<date>_<commit>_llama.json

Without SIMD_ML_LLAMA_DIR nothing is measured (README: "not measured"); without
the llama.cpp variable only our rows are. Speed: for each weight file (F32,
Q8_0, Q4_0) and thread count, our greedy decode of --gen tokens after a fixed
--prompt-tokens prompt (the first tokens of the perplexity text), --runs times
(median), timed per decode step; llama-bench -p 0 -n <gen> on the same GGUF
and threads, at depth 0 (as asked for) and at depth = prompt tokens (the same
KV length as ours). One profiled run per case splits our step time into
GEMVs, attention, norms/RoPE/SiLU and the rest (Python/NumPy glue).
Perplexity: llama-perplexity's method (llama.perplexity) on the committed text,
ours and llama-perplexity -c 512 -b 512, at the largest thread count.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import simd_ml_kernels as sk  # noqa: E402
from simd_ml_kernels import _cli  # noqa: E402
from llama import GGUF_TYPES, SECTIONS, Llama, encode, find_gguf, load_tokenizer, perplexity  # noqa: E402

TEXT = os.path.join(HERE, "data", "alice_ch1-2.txt")
LLAMA_ENV = "SIMD_ML_LLAMA_DIR"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def clock_stats(samples: list[float], min_ghz: float | None) -> dict:
    c = np.asarray(samples)
    return {"median": float(np.median(c)), "min": float(c.min()), "max": float(c.max()),
            "samples": int(c.size), "below_min_ghz": bool(min_ghz is not None and c.min() < min_ghz)}


def run_ours(model: Llama, prompt: list[int], gen: int, threads: int) -> tuple[float, float, list[float]]:
    """One timed run: prefill, then gen decode steps; returns (decode s, prefill s, clock samples)."""
    clock = _cli._clock_sample(threads)
    model.reset(len(prompt) + gen)
    t0 = time.perf_counter()
    for t in prompt:
        logits = model.step(t)
    t1 = time.perf_counter()
    decode = 0.0
    for _ in range(gen):  # the token chosen from the previous logits, as llama-bench's tg
        nxt = int(np.argmax(logits))
        t = time.perf_counter()
        logits = model.step(nxt)
        decode += time.perf_counter() - t
    clock += _cli._clock_sample(threads)
    return decode, t1 - t0, clock


def bench_ours(model, prompt, args, threads) -> dict:
    sk.set_num_threads(threads)
    run_ours(model, prompt[:8], 16, threads)  # warm-up (thread team, caches, page cache)
    for attempt in range(1, 4):
        runs, pre, clock = [], [], []
        for _ in range(args.runs):
            d, p, c = run_ours(model, prompt, args.gen, threads)
            runs.append(args.gen / d)
            pre.append(p / len(prompt))
            clock += c
        if args.min_ghz is None or min(clock) >= args.min_ghz:
            break
    model.profile = {}
    run_ours(model, prompt, args.gen, threads)
    prof, model.profile = model.profile, None
    steps = prof.pop("steps")
    ms = {k: 1e3 * v / steps for k, v in prof.items()}
    ms["other"] = ms["total"] - sum(ms[k] for k in SECTIONS)
    return {"tokens_per_s": float(np.median(runs)), "runs_tokens_per_s": runs,
            "ms_per_token": 1e3 / float(np.median(runs)),
            "prefill_ms_per_token": 1e3 * float(np.median(pre)), "attempts": attempt,
            "core_ghz": clock_stats(clock, args.min_ghz),
            "breakdown_ms_per_token": ms, "breakdown_steps": steps}


def llama_bench(bindir: str, gguf: str, args, threads: int) -> list[dict]:
    cmd = [os.path.join(bindir, "llama-bench"), "-m", gguf, "-p", "0", "-n", str(args.gen),
           "-d", f"0,{args.prompt_tokens}", "-t", str(threads), "-r", str(args.llama_reps), "-o", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return [{"not_measured": f"not measured (llama-bench exit {p.returncode}: {p.stderr.strip()[-200:]})",
                 "command": " ".join(cmd)}]
    out = []
    for e in json.loads(p.stdout):
        out.append({"tokens_per_s": e["avg_ts"], "stddev_tokens_per_s": e["stddev_ts"],
                    "runs_tokens_per_s": e.get("samples_ts"), "n_depth": e.get("n_depth", 0),
                    "n_threads": e["n_threads"], "n_gen": e["n_gen"], "model_type": e.get("model_type"),
                    "build_commit": e.get("build_commit"), "command": " ".join(cmd)})
    return out


def llama_perplexity(bindir: str, gguf: str, threads: int) -> dict:
    cmd = [os.path.join(bindir, "llama-perplexity"), "-m", gguf, "-f", TEXT, "-c", "512", "-b", "512",
           "-t", str(threads)]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    chunks = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", text)
    if p.returncode != 0 or not m:
        return {"not_measured": f"not measured (llama-perplexity exit {p.returncode})", "command": " ".join(cmd)}
    return {"ppl": float(m.group(1)), "ppl_stderr": float(m.group(2)),
            "chunks": int(chunks.group(1)) if chunks else None, "n_ctx": 512, "time_s": time.perf_counter() - t,
            "command": " ".join(cmd)}


def llama_tokens(bindir: str, gguf: str) -> list[int] | None:
    exe = os.path.join(bindir, "llama-tokenize")
    if not os.path.exists(exe):
        return None
    p = subprocess.run([exe, "-m", gguf, "-f", TEXT, "--ids", "--log-disable"], capture_output=True, text=True)
    m = re.search(r"\[([0-9, ]+)\]", p.stdout)
    return [int(v) for v in m.group(1).split(",")] if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=lambda v: [int(t) for t in v.split(",")], default=[1, 2, 4])
    ap.add_argument("--weights", type=lambda v: v.split(","), default=["fp32", "q8_0", "q4_0"])
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3, help="timed runs of ours per case (median)")
    ap.add_argument("--llama-reps", type=int, default=5, help="llama-bench -r")
    ap.add_argument("--cooldown", type=float, default=20.0, help="seconds idle before each case")
    ap.add_argument("--min-ghz", type=float, default=3.0, help="clock floor on the slowest thread (retry up to 3x)")
    ap.add_argument("--cases", type=lambda v: set(v.split(",")), default={"speed", "ppl"})
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--core", default="not recorded")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    model_dir = os.environ.get(LLAMA_ENV)
    if not model_dir:
        print(f"not measured ({LLAMA_ENV} not set)")
        return 0
    ggml_bin = os.environ.get(_cli.GGML_ENV)
    bindir = os.path.dirname(os.path.abspath(ggml_bin)) if ggml_bin else None
    llama_cpp = _cli._ggml_info(ggml_bin) if ggml_bin else f"not measured ({_cli.GGML_ENV} not set)"
    if isinstance(llama_cpp, dict):
        llama_cpp["tool"] = "llama.cpp llama-bench / llama-perplexity"
        for k in ("ggml_op_mul_mat", "env", "runs_per_case", "thread_check", "statistic"):
            llama_cpp.pop(k, None)
    tok = load_tokenizer(model_dir)
    with open(TEXT, encoding="utf-8") as f:
        text_ids = encode(tok, f.read())
    prompt = text_ids[:args.prompt_tokens]
    results: list[dict] = []
    files: dict[str, dict] = {}
    show = lambda r: print(json.dumps({k: v for k, v in r.items() if k not in ("runs_tokens_per_s", "command")}))
    for w in args.weights:
        gguf = find_gguf(model_dir, w)
        files[w] = {"path": gguf, "sha256": sha256(gguf), "type": GGUF_TYPES[w]}
        model = Llama(model_dir, gguf=gguf)
        files[w]["tensor_types"] = model.weight_types
        files[w]["matrix_bytes_per_token"] = model.weight_bytes()
        if "speed" in args.cases:
            for n in args.threads:
                time.sleep(args.cooldown)
                r = {"kernel": "llama_decode", "impl": "ours", "weights": w, "threads": n,
                     "prompt_tokens": len(prompt), "gen_tokens": args.gen, **bench_ours(model, prompt, args, n)}
                results.append(r)
                show(r)
                if bindir:
                    time.sleep(args.cooldown)
                    for e in llama_bench(bindir, gguf, args, n):
                        r = {"kernel": "llama_decode", "impl": "llama.cpp", "weights": w, "threads": n,
                             "gen_tokens": args.gen, **e}
                        results.append(r)
                        show(r)
        if "ppl" in args.cases:
            n = max(args.threads)
            sk.set_num_threads(n)
            t = time.perf_counter()
            pp = perplexity(model, text_ids)
            r = {"kernel": "llama_perplexity", "impl": "ours", "weights": w, "threads": n,
                 "time_s": time.perf_counter() - t, **{k: v for k, v in pp.items() if k != "running_ppl"}}
            results.append(r)
            show(r)
            if bindir:
                r = {"kernel": "llama_perplexity", "impl": "llama.cpp", "weights": w, "threads": n,
                     **llama_perplexity(bindir, gguf, n)}
                lt = llama_tokens(bindir, gguf)
                r["tokens_equal_ours"] = None if lt is None else lt == list(text_ids)
                results.append(r)
                show(r)
        del model

    if args.json:
        cfg = os.path.join(model_dir, "config.json")
        doc = {
            "schema": _cli.BENCH_SCHEMA,
            "machine": {"label": args.label, "cpu_model": _cli.cpu_model(), "core": args.core,
                        "pinned_cpus": _cli._pinned_cpus(), "caches": sk._core._cpu_caches(),
                        "logical_cpus": os.cpu_count(), "os": platform.platform(), "note": args.note},
            "build": _cli.info_dict()["build"],
            "baselines": {"numpy": np.__version__, "llama_cpp": llama_cpp},
            "env": {k: os.environ.get(k) for k in _cli.THREAD_ENV + _cli.KERNEL_ENV},
            "model": {"name": "meta-llama/Llama-3.2-1B", "config_sha256": sha256(cfg), "gguf": files,
                      "text": {"path": "examples/llama/data/alice_ch1-2.txt", "sha256": sha256(TEXT),
                               "tokens": len(text_ids)}},
            "protocol": {
                "decode": f"greedy decode of gen_tokens tokens after the first prompt_tokens tokens of the text "
                          f"(BOS included), prefill one token at a time; tokens_per_s = gen_tokens / sum of the "
                          f"decode step times, median of {args.runs} runs after a warm-up; clock sampled on every "
                          f"thread of a team of `threads` before and after each run, all runs repeated (up to 3 "
                          f"attempts) if a sample is below min_ghz; {args.cooldown:g} s idle before each case",
                "breakdown": "one more run with each section of the step timed (perf_counter around each call): "
                             "gemv = every projection and lm_head; attention = KV-cache write and per-KV-head "
                             "sgemm/softmax/sgemm; norm_rope_silu = RMSNorm, RoPE, SiLU and gate*up; other = "
                             "total - those (embedding row, residual adds, reshapes, Python, timer calls)",
                "llama_cpp": f"llama-bench -p 0 -n {args.gen} -d 0,{args.prompt_tokens} -t <threads> "
                             f"-r {args.llama_reps} on the same GGUF and CPUs (avg_ts; n_depth = KV entries before "
                             "the generated tokens); llama-perplexity -c 512 -b 512",
                "perplexity": "examples/llama/llama.py perplexity(): llama-perplexity's chunks of n_ctx = 512 "
                              "(first token BOS, second half scored)",
                "min_ghz": args.min_ghz, "runs": args.runs, "cooldown_s": args.cooldown},
            "command": " ".join(["python examples/llama/bench.py"] + sys.argv[1:]),
            "python": platform.python_version(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

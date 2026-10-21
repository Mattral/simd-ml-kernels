"""
tests/test_llama_gguf.py — examples/llama on llama.cpp's GGUF files
(SIMD_ML_LLAMA_DIR/gguf/*-{F32,Q8_0,Q4_0}.gguf, made with llama.cpp's
convert_hf_to_gguf.py and llama-quantize --pure; DESIGN.md section 9).

A module of its own: test_llama.py keeps an FP32 model (5 GB) for its whole
module, and a second one would not fit next to it in 8 GB. Skipped without
SIMD_ML_LLAMA_DIR or without the GGUF files.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from test_llama import (ATOL, LLAMA_DIR, NEW, NPOS, PROMPTS, RTOL, ROOT,  # noqa: F401
                        kernel_isa, llama_module, reference, tokenizer)

pytestmark = pytest.mark.skipif(not LLAMA_DIR, reason="SIMD_ML_LLAMA_DIR not set")


# ─── GGUF weights (llama.cpp's files, in SIMD_ML_LLAMA_DIR/gguf/) ──────────────


def _gguf(llama_module, weights):
    try:
        return llama_module.find_gguf(LLAMA_DIR, weights)
    except FileNotFoundError:
        pytest.skip(f"no {weights} GGUF in {LLAMA_DIR}/gguf")


@pytest.mark.parametrize("weights", ["q8_0", "q4_0"])
def test_gguf_blocks_match_our_quantizers(llama_module, weights):
    # llama-quantize --pure writes every weight matrix as Q8_0 / Q4_0 blocks;
    # quantizing the F32 GGUF's values with our quantize_q8_0 / quantize_q4_0
    # gives the same bytes, so gemv_q8_0 / gemv_q4_0 read llama.cpp's files as is.
    import simd_ml_kernels as sk
    from gguf_reader import GGUF

    quant = GGUF(_gguf(llama_module, weights))
    f32 = GGUF(_gguf(llama_module, "fp32"))
    fn = sk.quantize_q8_0 if weights == "q8_0" else sk.quantize_q4_0
    names = [n for n, t in quant.tensors.items() if t.type_name == weights.upper()]
    assert len(names) == 1 + 7 * 16  # token_embd (tied lm_head) + 7 per layer
    for n in names[:: max(1, len(names) // 12)] + ["token_embd.weight"]:
        ours = fn(np.ascontiguousarray(f32.tensors[n].data))
        assert np.array_equal(ours.reshape(-1), np.asarray(quant.tensors[n].data).reshape(-1)), n


def test_gguf_f32_greedy_matches_transformers(reference, llama_module):
    # The GGUF stores q/k rows in Meta's interleaved RoPE order (the converter
    # permutes them); the GGUF path rotates adjacent pairs, and must give the
    # same tokens as transformers.
    m = llama_module.Llama(LLAMA_DIR, gguf=_gguf(llama_module, "fp32"))
    assert m.rope_interleaved
    for i in range(len(PROMPTS)):
        ids = reference[f"ids{i}"]
        m.reset(NPOS)
        lg = np.stack([m.step(int(t)) for t in ids[:NPOS]])
        err = np.abs(lg.astype(np.float64) - reference[f"logits{i}"])
        assert (err / (ATOL + RTOL * np.abs(reference[f"logits{i}"]))).max() <= 1.0, f"prompt {i}"
        got = list(m.generate(ids.tolist(), NEW, stop_at_eos=False))
        assert got == reference[f"greedy{i}"].tolist(), f"prompt {i}"


def test_perplexity_text_tokens_match_llama_cpp(llama_module, tokenizer):
    # Same token ids as llama.cpp's tokenizer on the perplexity text, so the
    # two perplexities score the same tokens (needs llama-tokenize).
    import subprocess

    ggml = os.environ.get("SIMD_ML_GGML_TEST_BACKEND_OPS")
    exe = ggml and os.path.join(os.path.dirname(ggml), "llama-tokenize")
    if not exe or not os.path.exists(exe):
        pytest.skip("llama-tokenize not found next to SIMD_ML_GGML_TEST_BACKEND_OPS")
    text = os.path.join(ROOT, "examples", "llama", "data", "alice_ch1-2.txt")
    p = subprocess.run([exe, "-m", _gguf(llama_module, "q4_0"), "-f", text, "--ids", "--log-disable"],
                       capture_output=True, text=True, check=True)
    theirs = [int(v) for v in p.stdout.strip().strip("[]").split(",")]
    with open(text, encoding="utf-8") as f:
        assert llama_module.encode(tokenizer, f.read()) == theirs

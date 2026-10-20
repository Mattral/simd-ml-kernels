"""
tests/test_llama.py — end-to-end Llama-3.x decode (examples/llama) against
transformers (PyTorch CPU, float32).

Runs only when SIMD_ML_LLAMA_DIR points at a Hugging Face checkpoint directory
(config.json, tokenizer.json, *.safetensors; written for meta-llama/Llama-3.2-1B)
and transformers + torch are installed; otherwise every test is skipped (CI).

    SIMD_ML_LLAMA_DIR=~/code/models/Llama-3.2-1B python -m pytest tests/test_llama.py -s

The transformers outputs are computed once in a subprocess (both FP32 models
together need ~10 GB) and cached in .pytest_cache; -s prints the errors.
The reference alone:

    python tests/test_llama.py write-reference <model_dir> <out.npz>
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = os.environ.get("SIMD_ML_LLAMA_DIR")

# Five fixed prompts (each at least NPOS tokens with BOS).
PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "In 1969, Neil Armstrong became the first person to",
    "Photosynthesis is the process by which green plants",
    "Once upon a time, in a small village at the edge of the forest,",
]
NPOS = 8       # logits compared at the first 8 positions of each prompt
NEW = 64       # greedy tokens compared

# Logits tolerance, |ours - transformers| <= ATOL + RTOL*|transformers|.
# Both sides are FP32 with different summation orders (K = 2048 and 8192 dot
# products over 16 layers). Measured on the reference laptop (AVX2 table,
# transformers 5.20.2 / torch 2.14.1, sdpa attention): max |err| 1.0e-5 to
# 1.9e-5 per prompt on logits up to |29|, max relative error (|ref| >= 1)
# 1.7e-5. The bound, 1e-4 + 1e-5*|ref|, is at least 5x the measured error.
ATOL = 1e-4
RTOL = 1e-5
REF_FORMAT = 2  # bump when write_reference changes (part of the cache key)


def write_reference(model_dir: str, out: str) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    # NEW tokens on every prompt: no stop at EOS (ours: stop_at_eos=False).
    model.generation_config.eos_token_id = None
    res: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids
            res[f"ids{i}"] = ids[0].numpy()
            res[f"logits{i}"] = model(ids).logits[0, :NPOS].float().numpy()
            g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                               temperature=None, top_p=None, max_new_tokens=NEW,
                               output_logits=True, return_dict_in_generate=True,
                               pad_token_id=tok.eos_token_id)
            res[f"greedy{i}"] = g.sequences[0, ids.shape[1]:].numpy()
            raw = torch.stack(g.logits)[:, 0].float()        # [steps, vocab], unprocessed
            top = raw.topk(2)
            res[f"margin{i}"] = (top.values[:, 0] - top.values[:, 1]).numpy()
    res["versions"] = np.array(json.dumps({"torch": torch.__version__,
                                           "transformers": transformers.__version__,
                                           "attn": model.config._attn_implementation}))
    np.savez(out, **res)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(not LLAMA_DIR, reason="SIMD_ML_LLAMA_DIR not set")


@pytest.fixture(autouse=True, scope="module")
def kernel_isa():
    # Overrides conftest's per-table parametrization: a 1B-parameter model on
    # the scalar table would take minutes per prompt; the tables are tested
    # against float64 elsewhere. Runs on the table this CPU selects.
    import simd_ml_kernels

    yield simd_ml_kernels.detected_isa()


@pytest.fixture(scope="module")
def llama_module():
    sys.path.insert(0, os.path.join(ROOT, "examples", "llama"))
    import llama

    return llama


@pytest.fixture(scope="module")
def reference(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers

    st = sorted((f, os.path.getsize(os.path.join(LLAMA_DIR, f))) for f in os.listdir(LLAMA_DIR)
                if f.endswith(".safetensors") or f == "config.json")
    key = hashlib.sha256(json.dumps([REF_FORMAT, PROMPTS, NPOS, NEW, transformers.__version__, st,
                                     os.path.abspath(LLAMA_DIR)]).encode()).hexdigest()[:16]
    path = os.path.join(str(request.config.cache.mkdir("llama_reference")), f"{key}.npz")
    if not os.path.exists(path):
        tmp = path + ".tmp.npz"
        subprocess.run([sys.executable, os.path.abspath(__file__), "write-reference", LLAMA_DIR, tmp],
                       check=True)
        os.replace(tmp, path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def model(reference, llama_module):  # after `reference`: two FP32 models do not fit in 8 GB together
    return llama_module.Llama(LLAMA_DIR)


@pytest.fixture(scope="module")
def tokenizer(llama_module):
    return llama_module.load_tokenizer(LLAMA_DIR)


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_tokenizer_matches_transformers(reference, tokenizer, llama_module):
    for i, prompt in enumerate(PROMPTS):
        ids = llama_module.encode(tokenizer, prompt)
        assert ids == reference[f"ids{i}"].tolist(), f"prompt {i}"
        assert len(ids) >= NPOS
        assert ids[0] == 128000  # <|begin_of_text|> added, as by transformers


def test_rope_inv_freq_matches_transformers(llama_module):
    # llama3 rope_scaling, computed from config.json, against transformers' own
    # init function (plain RoPE differs by up to factor = 32x at low frequencies).
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    hf_cfg = AutoConfig.from_pretrained(LLAMA_DIR)
    cfg = llama_module.Config.from_dir(LLAMA_DIR)
    kind = cfg.rope_scaling.get("rope_type", cfg.rope_scaling.get("type"))
    want, scaling = ROPE_INIT_FUNCTIONS[kind](hf_cfg, "cpu")
    assert scaling == 1.0
    got = llama_module.rope_inv_freq(cfg)
    np.testing.assert_allclose(got, want.numpy(), rtol=2e-7, atol=0)


def test_logits_first_positions(reference, model):
    worst = 0.0
    for i in range(len(PROMPTS)):
        ids = reference[f"ids{i}"]
        ref = reference[f"logits{i}"]
        model.reset(NPOS)
        ours = np.stack([model.step(int(t)) for t in ids[:NPOS]])
        err = np.abs(ours.astype(np.float64) - ref)
        big = np.abs(ref) >= 1.0
        print(f"\nprompt {i}: max|err| = {err.max():.3e}, max|ref| = {np.abs(ref).max():.2f}, "
              f"max rel err (|ref| >= 1) = {(err[big] / np.abs(ref[big])).max():.3e}, "
              f"argmax equal at {int((ours.argmax(1) == ref.argmax(1)).sum())}/{NPOS} positions")
        ratio = (err / (ATOL + RTOL * np.abs(ref))).max()
        worst = max(worst, ratio)
        assert ratio <= 1.0, f"prompt {i}: logits error {ratio:.2f}x the tolerance"
    print(f"\nworst error / tolerance: {worst:.3f}; reference {reference['versions']}")


def test_greedy_decode_token_identical(reference, model):
    for i in range(len(PROMPTS)):
        ids = reference[f"ids{i}"].tolist()
        want = reference[f"greedy{i}"].tolist()
        margins: list[float] = []
        got = []
        for t in model.generate(ids, NEW, stop_at_eos=False):
            top2 = np.partition(model.last_logits, -2)[-2:]
            margins.append(float(top2[1] - top2[0]))
            got.append(t)
        n_same = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print(f"\nprompt {i}: {len(got)} tokens, smallest top-2 logit margin ours "
              f"{min(margins):.4f}, transformers {reference[f'margin{i}'].min():.4f}")
        assert len(want) == NEW
        assert got == want, (
            f"prompt {i}: first divergence at new token {n_same}: ours {got[n_same:n_same + 1]}, "
            f"transformers {want[n_same:n_same + 1]}; top-2 margin there: ours "
            f"{margins[n_same]:.4f}, transformers {reference[f'margin{i}'][n_same]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "write-reference":
        write_reference(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: python tests/test_llama.py write-reference <model_dir> <out.npz>")

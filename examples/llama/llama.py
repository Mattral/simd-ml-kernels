"""Llama-3.x decode from simd_ml_kernels GEMVs and NumPy (example, not library API).

Reads a Hugging Face checkpoint directory (config.json, tokenizer.json,
*.safetensors) and runs one token at a time with a KV cache:

    embedding -> L x [RMSNorm -> q/k/v GEMV -> RoPE -> GQA attention over the
    cache -> o GEMV -> residual -> RMSNorm -> gate/up GEMV -> SiLU(gate)*up
    -> down GEMV -> residual] -> RMSNorm -> lm_head GEMV

Every projection is sgemm(x, W, trans_b=True) with M = 1 (the decode GEMV,
DESIGN.md 2.7; threaded in OpenMP builds, 2.10); SiLU is simd_ml_kernels.silu;
attention uses sgemm and softmax per KV head; RMSNorm and RoPE are NumPy.
Prefill runs the prompt through the same one-token step, token by token (no
batched prefill yet). Every dimension comes from config.json.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

import simd_ml_kernels as sk

# ─── Checkpoint ───────────────────────────────────────────────────────────────


def _safetensors_files(model_dir: str) -> list[str]:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_dir, n) for n in names]
    single = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(single):
        raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {model_dir}")
    return [single]


def load_safetensors_fp32(path: str) -> dict[str, np.ndarray]:
    """Read every tensor of one .safetensors file as float32.

    The format is an 8-byte little-endian header length, a JSON header (name ->
    dtype, shape, byte range) and the raw data. The safetensors package cannot
    return BF16 as NumPy (NumPy has no bfloat16), so the file is parsed here:
    BF16 -> FP32 is exact (the 16 bits become the high half of the FP32 word).
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        b = raw[start:end]
        dtype = info["dtype"]
        if dtype == "BF16":
            a = b.view("<u2").astype(np.uint32)
            np.left_shift(a, 16, out=a)
            a = a.view(np.float32)
        elif dtype == "F16":
            a = b.view("<f2").astype(np.float32)
        elif dtype == "F32":
            a = np.array(b.view("<f4"))
        else:
            raise ValueError(f"{path}: tensor {name} has dtype {dtype}; expected BF16, F16 or F32")
        out[name] = a.reshape(info["shape"])
    del raw
    return out


@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_dir(cls, model_dir: str) -> "Config":
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        if c.get("model_type") != "llama":
            raise ValueError(f"model_type {c.get('model_type')!r}: only 'llama' is supported")
        if c.get("attention_bias") or c.get("mlp_bias") or c.get("hidden_act", "silu") != "silu":
            raise ValueError("attention_bias / mlp_bias / non-SiLU activation are not implemented")
        eos = c.get("eos_token_id")
        gen = os.path.join(model_dir, "generation_config.json")
        if os.path.exists(gen):  # generate() stops at generation_config's EOS ids
            with open(gen) as f:
                eos = json.load(f).get("eos_token_id", eos)
        eos_ids = tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,))
        heads = c["num_attention_heads"]
        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=c.get("num_key_value_heads", heads),
            head_dim=c.get("head_dim") or c["hidden_size"] // heads,
            vocab_size=c["vocab_size"],
            rms_norm_eps=float(c["rms_norm_eps"]),
            rope_theta=float(c.get("rope_theta", 10000.0)),
            rope_scaling=c.get("rope_scaling"),
            tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
            bos_token_id=c.get("bos_token_id"),
            eos_token_ids=eos_ids,
        )


# ─── RoPE ─────────────────────────────────────────────────────────────────────


def rope_inv_freq(cfg: Config) -> np.ndarray:
    """Inverse frequencies, float32, as transformers' rope init functions.

    Llama 3.x has rope_scaling {"rope_type": "llama3", factor, low_freq_factor,
    high_freq_factor, original_max_position_embeddings}: wavelengths longer than
    original/low_freq_factor are divided by factor, shorter than
    original/high_freq_factor are kept, and the band between is interpolated.
    Computed in float32 like transformers (torch float), so the angles match.
    """
    dim = cfg.head_dim
    exponent = np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)
    inv_freq = (np.float32(1.0) / (np.float32(cfg.rope_theta) ** exponent)).astype(np.float32)
    rs = cfg.rope_scaling
    kind = None if rs is None else rs.get("rope_type", rs.get("type"))
    if kind in (None, "default"):
        return inv_freq
    if kind != "llama3":
        raise ValueError(f"rope_scaling type {kind!r} is not implemented (llama3 and default are)")
    factor = rs["factor"]
    low, high = rs["low_freq_factor"], rs["high_freq_factor"]
    old_len = rs["original_max_position_embeddings"]
    low_freq_wavelen = old_len / low
    high_freq_wavelen = old_len / high
    wavelen = (np.float32(2 * math.pi) / inv_freq).astype(np.float32)
    scaled = np.where(wavelen > low_freq_wavelen, inv_freq / np.float32(factor), inv_freq).astype(np.float32)
    smooth = ((np.float32(old_len) / wavelen - np.float32(low)) / np.float32(high - low)).astype(np.float32)
    smoothed = ((1 - smooth) * scaled / np.float32(factor) + smooth * scaled).astype(np.float32)
    medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return np.where(medium, smoothed, scaled).astype(np.float32)


def rope_cos_sin(inv_freq: np.ndarray, pos: int) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin of one position, [head_dim], halves repeated (HF layout)."""
    angles = (np.float32(pos) * inv_freq).astype(np.float32)
    emb = np.concatenate([angles, angles])
    return np.cos(emb), np.sin(emb)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x [heads, head_dim] -> x*cos + rotate_half(x)*sin.

    rotate_half(x) = cat(-x[half:], x[:half]): the HF Llama layout, where the
    conversion from Meta's checkpoint permuted q/k rows so that the two
    elements of each rotated pair are half a head apart (not adjacent).
    """
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[:, half:], x[:, :half]], axis=-1)
    return x * cos + rot * sin


# ─── Model ────────────────────────────────────────────────────────────────────


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """weight * x / sqrt(mean(x^2) + eps), float32, in transformers' order."""
    var = np.mean(x * x, dtype=np.float32)
    return weight * (x * np.float32(1.0 / math.sqrt(float(var) + eps)))


def gemv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x [K] -> x @ w.T [N] with the decode GEMV (w is [N, K], nn.Linear layout)."""
    return sk.sgemm(x.reshape(1, -1), w, trans_b=True)[0]


@dataclass
class Layer:
    attn_norm: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    mlp_norm: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


class Llama:
    """FP32 weights in memory (BF16 checkpoints are widened exactly)."""

    def __init__(self, model_dir: str, weights: str = "fp32"):
        if weights != "fp32":
            raise ValueError(f"weights={weights!r}: only 'fp32' is implemented")
        self.cfg = cfg = Config.from_dir(model_dir)
        t: dict[str, np.ndarray] = {}
        for path in _safetensors_files(model_dir):
            t.update(load_safetensors_fp32(path))
        p = "model."
        self.embed = t.pop(p + "embed_tokens.weight")
        if "lm_head.weight" in t:
            self.lm_head = t.pop("lm_head.weight")
        elif cfg.tie_word_embeddings:
            self.lm_head = self.embed  # same array: 1 GB once, not twice
        else:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is false")
        self.norm = t.pop(p + "norm.weight")
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            q = f"{p}layers.{i}."
            self.layers.append(Layer(
                attn_norm=t.pop(q + "input_layernorm.weight"),
                wq=t.pop(q + "self_attn.q_proj.weight"),
                wk=t.pop(q + "self_attn.k_proj.weight"),
                wv=t.pop(q + "self_attn.v_proj.weight"),
                wo=t.pop(q + "self_attn.o_proj.weight"),
                mlp_norm=t.pop(q + "post_attention_layernorm.weight"),
                w_gate=t.pop(q + "mlp.gate_proj.weight"),
                w_up=t.pop(q + "mlp.up_proj.weight"),
                w_down=t.pop(q + "mlp.down_proj.weight"),
            ))
        leftover = [k for k in t if not k.endswith("rotary_emb.inv_freq")]
        if leftover:
            raise KeyError(f"unused tensors in checkpoint: {leftover[:5]}")
        self._check_shapes()
        self.inv_freq = rope_inv_freq(cfg)
        self.k_cache: np.ndarray | None = None
        self.v_cache: np.ndarray | None = None
        self.pos = 0

    def _check_shapes(self) -> None:
        c = self.cfg
        d, f, hd = c.hidden_size, c.intermediate_size, c.head_dim
        want = {"wq": (c.num_attention_heads * hd, d), "wk": (c.num_key_value_heads * hd, d),
                "wv": (c.num_key_value_heads * hd, d), "wo": (d, c.num_attention_heads * hd),
                "w_gate": (f, d), "w_up": (f, d), "w_down": (d, f)}
        for i, layer in enumerate(self.layers):
            for name, shape in want.items():
                if getattr(layer, name).shape != shape:
                    raise ValueError(f"layer {i} {name}: shape {getattr(layer, name).shape}, config says {shape}")
        if self.embed.shape != (c.vocab_size, d) or self.lm_head.shape != (c.vocab_size, d):
            raise ValueError("embedding / lm_head shape does not match config")

    def reset(self, capacity: int) -> None:
        """Empty KV cache for up to `capacity` positions: [layers, kv_heads, capacity, head_dim] each."""
        c = self.cfg
        shape = (c.num_hidden_layers, c.num_key_value_heads, capacity, c.head_dim)
        self.k_cache = np.zeros(shape, np.float32)
        self.v_cache = np.zeros(shape, np.float32)
        self.pos = 0

    def step(self, token: int) -> np.ndarray:
        """Run one token at position self.pos; returns float32 logits [vocab]."""
        c = self.cfg
        assert self.k_cache is not None and self.v_cache is not None, "call reset() first"
        pos = self.pos
        if pos >= self.k_cache.shape[2]:
            raise ValueError(f"KV cache full ({self.k_cache.shape[2]} positions)")
        nh, nkv, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        rep = nh // nkv
        scale = 1.0 / math.sqrt(hd)
        cos, sin = rope_cos_sin(self.inv_freq, pos)
        x = self.embed[token].copy()
        attn = np.empty((nh, hd), np.float32)
        for li, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, c.rms_norm_eps)
            q = apply_rope(gemv(h, L.wq).reshape(nh, hd), cos, sin)
            kc, vc = self.k_cache[li], self.v_cache[li]
            kc[:, pos] = apply_rope(gemv(h, L.wk).reshape(nkv, hd), cos, sin)
            vc[:, pos] = gemv(h, L.wv).reshape(nkv, hd)
            for g in range(nkv):  # GQA: rep query heads share KV head g
                qg = np.ascontiguousarray(q[g * rep:(g + 1) * rep])
                scores = sk.sgemm(qg, kc[g, :pos + 1], alpha=scale, trans_b=True)  # [rep, pos+1]
                attn[g * rep:(g + 1) * rep] = sk.sgemm(sk.softmax(scores), vc[g, :pos + 1])
            x = x + gemv(attn.reshape(-1), L.wo)
            h = rms_norm(x, L.mlp_norm, c.rms_norm_eps)
            x = x + gemv(sk.silu(gemv(h, L.w_gate)) * gemv(h, L.w_up), L.w_down)
        self.pos = pos + 1
        self.last_logits = gemv(rms_norm(x, self.norm, c.rms_norm_eps), self.lm_head)
        return self.last_logits

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int,
                 stats: dict | None = None, stop_at_eos: bool = True) -> Iterator[int]:
        """Greedy decode: yields up to max_new_tokens ids; stops after an EOS id
        (generation_config.json's, as transformers does) unless stop_at_eos is False.
        When an id is yielded, self.last_logits holds the logits it was chosen from.

        stats (optional dict) receives prefill_s, decode_s and decode_steps.
        """
        self.reset(len(prompt_ids) + max_new_tokens)
        t0 = time.perf_counter()
        for tok in prompt_ids:  # prefill, one token at a time
            logits = self.step(int(tok))
        if stats is not None:
            stats.update(prefill_s=time.perf_counter() - t0, prefill_tokens=len(prompt_ids),
                         decode_s=0.0, decode_steps=0)
        for i in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            yield nxt
            if (stop_at_eos and nxt in self.cfg.eos_token_ids) or i == max_new_tokens - 1:
                break
            t = time.perf_counter()  # the consumer's time between yields is not counted
            logits = self.step(nxt)
            if stats is not None:
                stats["decode_s"] += time.perf_counter() - t
                stats["decode_steps"] += 1


def load_tokenizer(model_dir: str):
    """tokenizers.Tokenizer from tokenizer.json. Its post-processor adds BOS
    (<|begin_of_text|> for Llama 3) exactly as transformers' fast tokenizer does
    with add_special_tokens=True (tested against AutoTokenizer)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=True).ids

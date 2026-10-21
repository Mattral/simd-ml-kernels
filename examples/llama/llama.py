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


class Linear:
    """y = x @ W.T for one row x [K]; W [N, K] (nn.Linear layout) as FP32, or
    as ggml Q8_0 / Q4_0 blocks read from a GGUF (byte-identical to
    quantize_q8_0 / quantize_q4_0), each through its decode GEMV."""

    def __init__(self, kind: str, data: np.ndarray, n: int, k: int):
        self.kind, self.data, self.shape = kind, data, (n, k)
        if kind == "f32":
            self._call = lambda x: sk.sgemm(x.reshape(1, -1), data, trans_b=True)[0]
        elif kind == "q8_0":
            self._call = lambda x: sk.gemv_q8_0(x, data, n, k)
        elif kind == "q4_0":
            self._call = lambda x: sk.gemv_q4_0(x, data, n, k)
        else:
            raise ValueError(f"weight kind {kind!r}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._call(x)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass
class Layer:
    attn_norm: np.ndarray
    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear
    mlp_norm: np.ndarray
    w_gate: Linear
    w_up: Linear
    w_down: Linear


# GGUF tensor names (llama.cpp) for each layer field.
_GGUF_NAMES = {"attn_norm": "attn_norm", "wq": "attn_q", "wk": "attn_k", "wv": "attn_v",
               "wo": "attn_output", "mlp_norm": "ffn_norm", "w_gate": "ffn_gate",
               "w_up": "ffn_up", "w_down": "ffn_down"}
_GGUF_KIND = {0: "f32", 8: "q8_0", 2: "q4_0"}  # ggml type -> Linear kind

# Sections of step() timed when profiling (Llama.profile).
SECTIONS = ("gemv", "attention", "norm_rope_silu")


class Llama:
    """Llama-3.x decoder for one sequence.

    weights="fp32": the safetensors checkpoint in model_dir (BF16 widened
    exactly to FP32). gguf=path: every tensor from a GGUF written by llama.cpp
    (F32, Q8_0 or Q4_0 weight matrices, F32 norms); config.json and
    tokenizer.json still come from model_dir, and the GGUF's own metadata
    must agree with config.json.
    """

    def __init__(self, model_dir: str, weights: str = "fp32", gguf: str | None = None):
        self.cfg = cfg = Config.from_dir(model_dir)
        if gguf is not None:
            self._load_gguf(gguf)
        elif weights == "fp32":
            self._load_safetensors(model_dir)
        else:
            raise ValueError(f"weights={weights!r} needs a GGUF file (gguf=...)")
        self._check_shapes()
        self.inv_freq = rope_inv_freq(cfg)
        self.k_cache: np.ndarray | None = None
        self.v_cache: np.ndarray | None = None
        self.pos = 0
        self.profile: dict[str, float] | None = None  # set to {} to time SECTIONS in step()

    def _load_safetensors(self, model_dir: str) -> None:
        cfg = self.cfg
        t: dict[str, np.ndarray] = {}
        for path in _safetensors_files(model_dir):
            t.update(load_safetensors_fp32(path))

        def lin(name: str) -> Linear:
            w = t.pop(name)
            return Linear("f32", w, *w.shape)

        p = "model."
        embed = t.pop(p + "embed_tokens.weight")
        self._embed_row = lambda tok: embed[tok].copy()
        if "lm_head.weight" in t:
            self.lm_head = lin("lm_head.weight")
        elif cfg.tie_word_embeddings:
            self.lm_head = Linear("f32", embed, *embed.shape)  # same array: 1 GB once, not twice
        else:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is false")
        self.norm = t.pop(p + "norm.weight")
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            q = f"{p}layers.{i}."
            self.layers.append(Layer(
                attn_norm=t.pop(q + "input_layernorm.weight"),
                wq=lin(q + "self_attn.q_proj.weight"),
                wk=lin(q + "self_attn.k_proj.weight"),
                wv=lin(q + "self_attn.v_proj.weight"),
                wo=lin(q + "self_attn.o_proj.weight"),
                mlp_norm=t.pop(q + "post_attention_layernorm.weight"),
                w_gate=lin(q + "mlp.gate_proj.weight"),
                w_up=lin(q + "mlp.up_proj.weight"),
                w_down=lin(q + "mlp.down_proj.weight"),
            ))
        leftover = [k for k in t if not k.endswith("rotary_emb.inv_freq")]
        if leftover:
            raise KeyError(f"unused tensors in checkpoint: {leftover[:5]}")
        self.rope_interleaved = False  # HF layout: rotate_half
        self.weight_types = {"all": "f32 (safetensors)"}

    def _load_gguf(self, path: str) -> None:
        from gguf_reader import GGUF, dequantize_rows

        cfg = self.cfg
        g = GGUF(path)
        self.gguf = g
        md = g.metadata
        if md.get("general.architecture") != "llama":
            raise ValueError(f"{path}: architecture {md.get('general.architecture')!r}, expected llama")
        want = {"llama.block_count": cfg.num_hidden_layers, "llama.embedding_length": cfg.hidden_size,
                "llama.feed_forward_length": cfg.intermediate_size,
                "llama.attention.head_count": cfg.num_attention_heads,
                "llama.attention.head_count_kv": cfg.num_key_value_heads}
        for key, val in want.items():
            if int(md[key]) != val:
                raise ValueError(f"{path}: {key} = {md[key]}, config.json says {val}")
        used: set[str] = set()

        def tensor(name: str):
            used.add(name)
            return g.tensors[name]

        def lin(name: str) -> Linear:
            t = tensor(name)
            if t.type not in _GGUF_KIND:
                raise ValueError(f"{path}: {name} is {t.type_name}; F32, Q8_0 or Q4_0 expected")
            return Linear(_GGUF_KIND[t.type], t.data, *t.shape)

        def vec(name: str) -> np.ndarray:
            t = tensor(name)
            if t.type_name != "F32":
                raise ValueError(f"{path}: {name} is {t.type_name}, F32 expected")
            return np.asarray(t.data)

        emb = tensor("token_embd.weight")
        self._embed_row = lambda tok: dequantize_rows(emb, tok)[0]
        # Tied embeddings: convert_hf_to_gguf writes no output.weight, and
        # llama.cpp multiplies by token_embd (same bytes, same type).
        self.lm_head = lin("output.weight" if "output.weight" in g.tensors else "token_embd.weight")
        self.norm = vec("output_norm.weight")
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            f = {k: f"blk.{i}.{v}.weight" for k, v in _GGUF_NAMES.items()}
            self.layers.append(Layer(attn_norm=vec(f["attn_norm"]), mlp_norm=vec(f["mlp_norm"]),
                                     **{k: lin(f[k]) for k in f if k not in ("attn_norm", "mlp_norm")}))
        # rope_freqs.weight holds llama3's per-frequency divisors; we compute
        # the same frequencies from config.json (rope_inv_freq, tested).
        leftover = sorted(set(g.tensors) - used - {"rope_freqs.weight"})
        if leftover:
            raise KeyError(f"{path}: unused tensors {leftover[:5]}")
        # convert_hf_to_gguf permutes q/k rows back to Meta's order, where each
        # rotated pair is two adjacent elements (llama.cpp's "normal" RoPE).
        self.rope_interleaved = True
        types: dict[str, int] = {}
        for t in g.tensors.values():
            types[t.type_name] = types.get(t.type_name, 0) + 1
        self.weight_types = types

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
        if self.lm_head.shape != (c.vocab_size, d):
            raise ValueError("lm_head shape does not match config")

    def weight_bytes(self) -> int:
        """Bytes of weight matrices read per token (every Linear once)."""
        lins = [self.lm_head] + [getattr(L, k) for L in self.layers
                                 for k in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")]
        return sum(w.nbytes for w in lins)

    def reset(self, capacity: int) -> None:
        """Empty KV cache for up to `capacity` positions: [layers, kv_heads, capacity, head_dim] each."""
        c = self.cfg
        shape = (c.num_hidden_layers, c.num_key_value_heads, capacity, c.head_dim)
        self.k_cache = np.zeros(shape, np.float32)
        self.v_cache = np.zeros(shape, np.float32)
        self.pos = 0

    def _rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        if self.rope_interleaved:  # pairs (2i, 2i+1), angle i: cos/sin halves are equal
            half = x.shape[-1] // 2
            a, b = x[:, 0::2], x[:, 1::2]
            c, s = cos[:half], sin[:half]
            out = np.empty_like(x)
            out[:, 0::2] = a * c - b * s
            out[:, 1::2] = b * c + a * s
            return out
        return apply_rope(x, cos, sin)

    def _attention(self, li: int, q: np.ndarray, k: np.ndarray, v: np.ndarray, pos: int) -> np.ndarray:
        """Write k, v at pos into layer li's cache; GQA attention of q [heads, hd] over positions 0..pos."""
        c = self.cfg
        nh, nkv, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        rep = nh // nkv
        kc, vc = self.k_cache[li], self.v_cache[li]
        kc[:, pos] = k
        vc[:, pos] = v
        out = np.empty((nh, hd), np.float32)
        scale = 1.0 / math.sqrt(hd)
        for g in range(nkv):  # rep query heads share KV head g
            qg = np.ascontiguousarray(q[g * rep:(g + 1) * rep])
            scores = sk.sgemm(qg, kc[g, :pos + 1], alpha=scale, trans_b=True)  # [rep, pos+1]
            out[g * rep:(g + 1) * rep] = sk.sgemm(sk.softmax(scores), vc[g, :pos + 1])
        return out.reshape(-1)

    def step(self, token: int, logits: bool = True) -> np.ndarray | None:
        """Run one token at position self.pos; returns float32 logits [vocab]
        (logits=False skips the lm_head GEMV and returns None)."""
        if self.profile is not None:
            return self._step_profiled(token)
        c = self.cfg
        assert self.k_cache is not None and self.v_cache is not None, "call reset() first"
        pos = self.pos
        if pos >= self.k_cache.shape[2]:
            raise ValueError(f"KV cache full ({self.k_cache.shape[2]} positions)")
        nh, nkv, hd, eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        cos, sin = rope_cos_sin(self.inv_freq, pos)
        rope = self._rope
        x = self._embed_row(token)
        for li, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, eps)
            q = rope(L.wq(h).reshape(nh, hd), cos, sin)
            k = rope(L.wk(h).reshape(nkv, hd), cos, sin)
            x = x + L.wo(self._attention(li, q, k, L.wv(h).reshape(nkv, hd), pos))
            h = rms_norm(x, L.mlp_norm, eps)
            x = x + L.w_down(sk.silu(L.w_gate(h)) * L.w_up(h))
        self.pos = pos + 1
        if not logits:
            return None
        self.last_logits = self.lm_head(rms_norm(x, self.norm, eps))
        return self.last_logits

    def _step_profiled(self, token: int) -> np.ndarray:
        """step() with each section timed into self.profile (seconds, summed):
        gemv (every Linear), attention (cache write, per-KV-head sgemm/softmax),
        norm_rope_silu (RMSNorm, RoPE, SiLU and the gate*up product), total.
        total minus the sections is Python and NumPy glue (embedding row,
        residual adds, reshapes, call overhead outside the timed calls) plus
        the timer calls themselves (about 0.1 us each, ~200 per token)."""
        c = self.cfg
        P = self.profile
        clock = time.perf_counter
        t_start = clock()
        pos = self.pos
        nh, nkv, hd, eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        gemv_s = attn_s = misc_s = 0.0
        t = clock()
        cos, sin = rope_cos_sin(self.inv_freq, pos)
        misc_s += clock() - t
        x = self._embed_row(token)
        for li, L in enumerate(self.layers):
            t = clock(); h = rms_norm(x, L.attn_norm, eps); misc_s += clock() - t
            t = clock(); q, k, v = L.wq(h), L.wk(h), L.wv(h); gemv_s += clock() - t
            t = clock(); q = self._rope(q.reshape(nh, hd), cos, sin); k = self._rope(k.reshape(nkv, hd), cos, sin)
            misc_s += clock() - t
            t = clock(); a = self._attention(li, q, k, v.reshape(nkv, hd), pos); attn_s += clock() - t
            t = clock(); o = L.wo(a); gemv_s += clock() - t
            x = x + o
            t = clock(); h = rms_norm(x, L.mlp_norm, eps); misc_s += clock() - t
            t = clock(); gate, up = L.w_gate(h), L.w_up(h); gemv_s += clock() - t
            t = clock(); m = sk.silu(gate) * up; misc_s += clock() - t
            t = clock(); d = L.w_down(m); gemv_s += clock() - t
            x = x + d
        t = clock(); h = rms_norm(x, self.norm, eps); misc_s += clock() - t
        t = clock(); self.last_logits = self.lm_head(h); gemv_s += clock() - t
        self.pos = pos + 1
        total = clock() - t_start
        for key, val in (("gemv", gemv_s), ("attention", attn_s), ("norm_rope_silu", misc_s), ("total", total)):
            P[key] = P.get(key, 0.0) + val
        P["steps"] = P.get("steps", 0) + 1
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


GGUF_TYPES = {"fp32": "F32", "q8_0": "Q8_0", "q4_0": "Q4_0"}


def find_gguf(model_dir: str, weights: str) -> str:
    """model_dir/gguf/*-<F32|Q8_0|Q4_0>.gguf (where the README's commands put them)."""
    import glob

    hits = sorted(glob.glob(os.path.join(model_dir, "gguf", f"*-{GGUF_TYPES[weights]}.gguf")))
    if not hits:
        raise FileNotFoundError(f"no {model_dir}/gguf/*-{GGUF_TYPES[weights]}.gguf (see examples/llama/README "
                                "commands in DESIGN.md section 9)")
    return hits[0]


def perplexity(model: Llama, ids: Sequence[int], n_ctx: int = 512, bos: int | None = None,
               progress=None) -> dict:
    """Perplexity as llama.cpp's llama-perplexity computes it: the token ids
    (BOS first) are cut into len(ids) // n_ctx chunks of n_ctx; each chunk
    starts from an empty cache with its first token replaced by BOS, and only
    the second half is scored: the logits at positions n_ctx/2 .. n_ctx-2
    predict the next token (n_ctx/2 - 1 tokens per chunk). PPL = exp(mean NLL).
    Positions before n_ctx/2 skip the lm_head GEMV (their logits are unused)."""
    bos = model.cfg.bos_token_id if bos is None else bos
    first = n_ctx // 2
    n_chunk = len(ids) // n_ctx
    if n_chunk < 1:
        raise ValueError(f"{len(ids)} tokens: at least n_ctx = {n_ctx} needed")
    nll, count, per_chunk = 0.0, 0, []
    for c in range(n_chunk):
        toks = [int(t) for t in ids[c * n_ctx:(c + 1) * n_ctx]]
        toks[0] = bos
        model.reset(n_ctx)
        for pos, t in enumerate(toks[:-1]):
            lg = model.step(t, logits=pos >= first)
            if lg is not None:
                z = lg.astype(np.float64)
                m = z.max()
                nll += m + math.log(np.exp(z - m).sum()) - z[toks[pos + 1]]
                count += 1
        per_chunk.append(math.exp(nll / count))
        if progress:
            progress(c + 1, n_chunk, per_chunk[-1])
    return {"ppl": math.exp(nll / count), "nll_mean": nll / count, "tokens_scored": count,
            "chunks": n_chunk, "n_ctx": n_ctx, "running_ppl": per_chunk}


def load_tokenizer(model_dir: str):
    """tokenizers.Tokenizer from tokenizer.json. Its post-processor adds BOS
    (<|begin_of_text|> for Llama 3) exactly as transformers' fast tokenizer does
    with add_special_tokens=True (tested against AutoTokenizer)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=True).ids

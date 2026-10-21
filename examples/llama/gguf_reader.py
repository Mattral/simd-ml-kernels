"""Minimal GGUF reader (llama.cpp's file format, version 2 and 3): metadata and
tensors as NumPy views of the memory-mapped file, no copies.

Layout: "GGUF", u32 version, u64 tensor count, u64 metadata count; metadata
(string key, u32 type, value); tensor infos (string name, u32 n_dims, u64
dims innermost first, u32 ggml type, u64 offset); data at the next multiple
of general.alignment (default 32). Q8_0 / Q4_0 tensors are returned as their
raw uint8 blocks, which is what gemv_q8_0 / gemv_q4_0 take (ggml block_q8_0:
34 bytes per 32 weights, block_q4_0: 18).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_BF16 = 0, 1, 2, 8, 30
TYPE_NAMES = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q4_0: "Q4_0", GGML_Q8_0: "Q8_0", GGML_BF16: "BF16"}
# bytes per block and values per block
_BLOCK = {GGML_F32: (4, 1), GGML_F16: (2, 1), GGML_BF16: (2, 1), GGML_Q8_0: (34, 32), GGML_Q4_0: (18, 32)}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass
class Tensor:
    name: str
    type: int
    shape: tuple[int, ...]   # NumPy order (outermost first): [N, K] for a weight matrix
    data: np.ndarray         # F32/F16: that dtype, shaped; Q8_0/Q4_0: uint8 [N, K/32*bytes]

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, str(self.type))


class GGUF:
    def __init__(self, path: str):
        self.path = path
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        self._pos = 0
        if bytes(self._mm[:4]) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self._pos = 4
        self.version = self._read("<I")
        if self.version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {self.version} (2 and 3 are supported)")
        n_tensors, n_kv = self._read("<Q"), self._read("<Q")
        self.metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = self._string()
            self.metadata[key] = self._value(self._read("<I"))
        infos = []
        for _ in range(n_tensors):
            name = self._string()
            dims = [self._read("<Q") for _ in range(self._read("<I"))]
            infos.append((name, dims, self._read("<I"), self._read("<Q")))
        align = int(self.metadata.get("general.alignment", 32))
        base = (self._pos + align - 1) // align * align
        self.tensors: dict[str, Tensor] = {}
        for name, dims, ttype, offset in infos:
            if ttype not in _BLOCK:
                raise ValueError(f"{path}: tensor {name} has ggml type {ttype} (not F32/F16/BF16/Q8_0/Q4_0)")
            bsize, bvals = _BLOCK[ttype]
            shape = tuple(reversed(dims))
            k = dims[0]
            rows = int(np.prod(dims[1:], dtype=np.int64)) if len(dims) > 1 else 1
            if k % bvals:
                raise ValueError(f"{path}: tensor {name}: row length {k} not a multiple of {bvals}")
            nbytes = rows * (k // bvals) * bsize
            raw = self._mm[base + offset: base + offset + nbytes]
            if ttype == GGML_F32:
                data = raw.view("<f4").reshape(shape)
            elif ttype == GGML_F16:
                data = raw.view("<f2").reshape(shape)
            elif ttype == GGML_BF16:
                data = raw.view("<u2").reshape(shape)
            else:
                data = raw.reshape(rows, (k // bvals) * bsize)
            self.tensors[name] = Tensor(name, ttype, shape, data)

    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        (v,) = struct.unpack_from(fmt, self._mm, self._pos)
        self._pos += size
        return v

    def _string(self) -> str:
        n = self._read("<Q")
        s = bytes(self._mm[self._pos:self._pos + n]).decode("utf-8", errors="replace")
        self._pos += n
        return s

    def _value(self, vtype: int):
        if vtype in _SCALAR:
            return self._read(_SCALAR[vtype])
        if vtype == 8:
            return self._string()
        if vtype == 9:
            etype, n = self._read("<I"), self._read("<Q")
            if etype in _SCALAR and etype != 7:  # numeric arrays (token scores, types) as NumPy
                dt = np.dtype(_SCALAR[etype])
                a = np.frombuffer(self._mm, dtype=dt, count=n, offset=self._pos)
                self._pos += n * dt.itemsize
                return a
            return [self._value(etype) for _ in range(n)]
        raise ValueError(f"{self.path}: unknown GGUF metadata type {vtype}")


def dequantize_rows(t: Tensor, rows: np.ndarray | int) -> np.ndarray:
    """float32 values of some rows of a 2-D tensor (embedding lookup)."""
    r = np.atleast_1d(np.asarray(rows))
    if t.type == GGML_F32:
        return np.array(t.data[r], dtype=np.float32)
    if t.type == GGML_F16:
        return t.data[r].astype(np.float32)
    if t.type == GGML_BF16:
        return (t.data[r].astype(np.uint32) << 16).view(np.float32)
    k = t.shape[-1]
    blocks = np.asarray(t.data[r]).reshape(len(r), k // 32, -1)
    d = blocks[:, :, :2].copy().view("<f2").astype(np.float32)        # [rows, nb, 1]
    if t.type == GGML_Q8_0:
        q = blocks[:, :, 2:].view(np.int8).astype(np.float32)          # [rows, nb, 32]
    else:  # Q4_0: value j in the low nibble of byte j, value 16 + j in the high nibble
        b = blocks[:, :, 2:]
        q = np.concatenate([b & 0x0F, b >> 4], axis=-1).astype(np.float32) - 8.0
    return (d * q).reshape(len(r), k)

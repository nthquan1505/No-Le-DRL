"""Lightweight vector math and NumPy .npy (v1.0 float32) I/O utilities.

Uses NumPy when installed and falls back transparently to pure Python (math + struct)
so deterministic tests and offline pipelines run cleanly even in minimal environments.
All saved .npy files strictly conform to the standard NumPy .npy v1.0 binary format.
"""

from __future__ import annotations

import ast
import math
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal Python envs
    _np = None

_NPY_MAGIC = b"\x93NUMPY"


def l2_norm(vec: Sequence[float]) -> float:
    """Return Euclidean L2 norm of a vector."""
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def l2_normalize(vec: Sequence[float]) -> List[float]:
    """Return L2-normalized copy of a vector (or zero vector if norm == 0)."""
    norm = l2_norm(vec)
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [float(x) / norm for x in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot-product cosine similarity between two equal-length vectors.

    If inputs are already L2-normalized, dot(a, b) equals cosine similarity directly.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def batch_dot(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> List[float]:
    """Compute dot product between query vector and each row in matrix."""
    if not matrix:
        return []
    if _np is not None:
        q = _np.asarray(query, dtype=_np.float32)
        m = _np.asarray(matrix, dtype=_np.float32)
        return [float(x) for x in (m @ q)]
    dim = len(query)
    scores: List[float] = []
    for row in matrix:
        if len(row) != dim:
            raise ValueError(f"Vector dimension mismatch: query={dim}, row={len(row)}")
        scores.append(sum(float(x) * float(y) for x, y in zip(query, row)))
    return scores


def save_npy_float32(path: Path, matrix: Sequence[Sequence[float]], dim: int) -> None:
    """Save a 2D float32 matrix in standard NumPy .npy v1.0 format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(matrix)
    for idx, row in enumerate(matrix):
        if len(row) != dim:
            raise ValueError(f"Row {idx} has dimension {len(row)}, expected {dim}")

    if _np is not None:
        arr = _np.asarray(matrix, dtype=_np.float32).reshape((rows, dim))
        _np.save(path, arr)
        return

    header_dict = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {dim}), }}"
    header_bytes = header_dict.encode("latin1")
    # Pad header so len(magic + version + header_len + header_bytes + '\n') is a multiple of 64
    preamble_len = len(_NPY_MAGIC) + 2 + 2
    pad_len = (64 - ((preamble_len + len(header_bytes) + 1) % 64)) % 64
    header_padded = header_bytes + (b" " * pad_len) + b"\n"

    with open(path, "wb") as handle:
        handle.write(_NPY_MAGIC)
        handle.write(struct.pack("<BB", 1, 0))
        handle.write(struct.pack("<H", len(header_padded)))
        handle.write(header_padded)
        if rows > 0 and dim > 0:
            fmt = f"<{dim}f"
            for row in matrix:
                handle.write(struct.pack(fmt, *(float(v) for v in row)))


def load_npy_float32(path: Path) -> Tuple[List[List[float]], Tuple[int, int]]:
    """Load a 2D float32 matrix from a standard NumPy .npy file."""
    if _np is not None:
        arr = _np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        return [[float(v) for v in row] for row in arr.tolist()], (rows, cols)

    with open(path, "rb") as handle:
        magic = handle.read(len(_NPY_MAGIC))
        if magic != _NPY_MAGIC:
            raise ValueError(f"Invalid .npy magic header in {path}")
        major, minor = struct.unpack("<BB", handle.read(2))
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major == 2:
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version {major}.{minor}")
        header_str = handle.read(header_len).decode("latin1").strip()
        header = ast.literal_eval(header_str)
        descr = header["descr"]
        if descr not in ("<f4", "|f4"):
            raise ValueError(f"Unsupported dtype '{descr}' in {path}; expected '<f4'")
        shape = header["shape"]
        if len(shape) != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {shape}")
        rows, cols = int(shape[0]), int(shape[1])
        data = handle.read()

    expected_bytes = rows * cols * 4
    if len(data) != expected_bytes:
        raise ValueError(f"Corrupt .npy data in {path}: expected {expected_bytes} bytes, got {len(data)}")

    matrix: List[List[float]] = []
    if rows > 0 and cols > 0:
        row_fmt = f"<{cols}f"
        row_bytes = cols * 4
        for i in range(rows):
            chunk = data[i * row_bytes : (i + 1) * row_bytes]
            matrix.append(list(struct.unpack(row_fmt, chunk)))
    return matrix, (rows, cols)

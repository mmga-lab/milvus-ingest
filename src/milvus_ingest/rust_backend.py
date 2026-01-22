"""Rust backend wrapper for high-performance data generation.

This module provides a unified interface for Rust-accelerated data generation functions,
with automatic fallback to pure Python implementations when Rust is unavailable.

Usage:
    from milvus_ingest.rust_backend import (
        USE_RUST,
        generate_float16_vectors,
        generate_bfloat16_vectors,
        generate_sparse_vectors,
        generate_text_fields,
        generate_json_fields,
        generate_varchar_array,
        generate_int_array,
    )
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Check if Rust acceleration is enabled
_RUST_DISABLED = os.environ.get("MILVUS_INGEST_USE_RUST", "1").lower() in (
    "0",
    "false",
    "no",
)

# Try to import Rust module
_rust_module = None
if not _RUST_DISABLED:
    try:
        from milvus_ingest import _rust

        _rust_module = _rust
    except ImportError:
        pass

# Export availability flag
USE_RUST = _rust_module is not None


def get_rust_version() -> str | None:
    """Get the version of the Rust module, or None if not available."""
    if _rust_module is not None:
        return _rust_module.get_version()
    return None


def get_cpu_features() -> list[str]:
    """Get available CPU SIMD features."""
    if _rust_module is not None:
        return _rust_module.get_cpu_features()
    return ["scalar"]


# =============================================================================
# Vector Generation Functions
# =============================================================================


def generate_float16_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Float16 vectors as uint8 bytes.

    Returns a 2D numpy array of shape (num_rows, dim * 2) containing
    Float16 vectors as raw bytes.
    """
    if _rust_module is not None:
        return _rust_module.generate_float16_vectors(num_rows, dim, seed)

    # Python fallback
    vectors = np.random.random((num_rows, dim)).astype(np.float16)
    return vectors.view(np.uint8)


def generate_bfloat16_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate BFloat16 vectors as uint8 bytes.

    Returns a 2D numpy array of shape (num_rows, dim * 2) containing
    BFloat16 vectors as raw bytes.
    """
    if _rust_module is not None:
        return _rust_module.generate_bfloat16_vectors(num_rows, dim, seed)

    # Python fallback
    from ml_dtypes import bfloat16

    vectors = np.random.random((num_rows, dim)).astype(bfloat16)
    return vectors.view(np.uint8)


def generate_normalized_float16_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate normalized (unit) Float16 vectors as uint8 bytes."""
    if _rust_module is not None:
        return _rust_module.generate_normalized_float16_vectors(num_rows, dim, seed)

    # Python fallback
    vectors = np.random.random((num_rows, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms
    return vectors.astype(np.float16).view(np.uint8)


def generate_float_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Float32 vectors.

    Returns a 2D numpy array of shape (num_rows, dim).
    """
    if _rust_module is not None:
        return _rust_module.generate_float_vectors(num_rows, dim, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    return np.random.random((num_rows, dim)).astype(np.float32)


def generate_normalized_float_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate normalized (unit) Float32 vectors.

    Returns a 2D numpy array of shape (num_rows, dim) with L2 norm = 1.
    """
    if _rust_module is not None:
        return _rust_module.generate_normalized_float_vectors(num_rows, dim, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    vectors = np.random.random((num_rows, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def generate_sparse_vectors(
    num_rows: int,
    max_dim: int = 1000,
    density_min: float = 0.01,
    density_max: float = 0.1,
    seed: int | None = None,
) -> list[dict[int, float]]:
    """Generate sparse float vectors as list of dicts.

    Returns a list of dicts, each mapping dimension index to value.
    """
    if _rust_module is not None:
        return _rust_module.generate_sparse_vectors(
            num_rows, max_dim, density_min, density_max, seed
        )

    # Python fallback
    sparse_vectors = []
    for _ in range(num_rows):
        density = np.random.uniform(density_min, density_max)
        nnz = max(1, int(max_dim * density))
        indices = np.random.choice(max_dim, nnz, replace=False)
        values = np.random.random(nnz)
        sparse_vectors.append(
            {int(idx): float(val) for idx, val in zip(indices, values, strict=False)}
        )
    return sparse_vectors


# =============================================================================
# Text Generation Functions
# =============================================================================


def generate_text_fields(
    num_rows: int,
    max_length: int = 256,
    text_type: str = "sentence",
    language: str = "en",
    seed: int | None = None,
) -> list[str]:
    """Generate realistic text fields for BM25 testing.

    Args:
        num_rows: Number of text strings to generate
        max_length: Maximum character length
        text_type: Type of text ("sentence", "paragraph", "title", "keywords")
        language: Language ("en", "zh", "mixed")
        seed: Random seed for reproducibility

    Returns:
        List of generated text strings
    """
    if _rust_module is not None:
        return _rust_module.generate_text_fields(
            num_rows, max_length, text_type, language, seed
        )

    # Python fallback - simple pattern
    return [f"text_{i}" for i in range(num_rows)]


def generate_simple_text(
    num_rows: int,
    prefix: str = "text",
    pk_offset: int = 0,
) -> list[str]:
    """Generate simple text fields (text_0, text_1, ...).

    Args:
        num_rows: Number of text strings to generate
        prefix: Prefix for the text
        pk_offset: Starting offset for numbering

    Returns:
        List of generated text strings
    """
    if _rust_module is not None:
        return _rust_module.generate_simple_text(num_rows, prefix, pk_offset)

    # Python fallback
    return [f"{prefix}_{pk_offset + i}" for i in range(num_rows)]


# =============================================================================
# JSON Generation Functions
# =============================================================================


def generate_json_fields(
    num_rows: int,
    pattern: str = "ecommerce",
    pk_offset: int = 0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate JSON fields with various patterns.

    Args:
        num_rows: Number of JSON objects to generate
        pattern: Pattern type ("ecommerce", "event", "config", "analytics", "document")
        pk_offset: Primary key offset for ID generation
        seed: Random seed for reproducibility

    Returns:
        List of Python dicts representing JSON objects
    """
    if _rust_module is not None:
        return _rust_module.generate_json_fields(num_rows, pattern, pk_offset, seed)

    # Python fallback - simple pattern
    return [{"id": pk_offset + i, "value": f"item_{i}"} for i in range(num_rows)]


# =============================================================================
# Array Generation Functions
# =============================================================================


def generate_varchar_array(
    num_rows: int,
    max_capacity: int = 10,
    max_length: int = 64,
    seed: int | None = None,
) -> list[list[str]]:
    """Generate Array[VarChar] fields.

    Args:
        num_rows: Number of arrays to generate
        max_capacity: Maximum number of elements per array
        max_length: Maximum length of each string element
        seed: Random seed for reproducibility

    Returns:
        List of lists of strings
    """
    if _rust_module is not None:
        return _rust_module.generate_varchar_array(
            num_rows, max_capacity, max_length, seed
        )

    # Python fallback
    result = []
    for _ in range(num_rows):
        arr_len = np.random.randint(1, max_capacity + 1)
        result.append([f"item_{j}" for j in range(arr_len)])
    return result


def generate_int_array(
    num_rows: int,
    max_capacity: int = 10,
    min_val: int = 0,
    max_val: int = 1000000,
    seed: int | None = None,
) -> list[list[int]]:
    """Generate Array[Int64] fields.

    Args:
        num_rows: Number of arrays to generate
        max_capacity: Maximum number of elements per array
        min_val: Minimum value for integers
        max_val: Maximum value for integers
        seed: Random seed for reproducibility

    Returns:
        List of lists of integers
    """
    if _rust_module is not None:
        return _rust_module.generate_int_array(
            num_rows, max_capacity, min_val, max_val, seed
        )

    # Python fallback
    result = []
    for _ in range(num_rows):
        arr_len = np.random.randint(1, max_capacity + 1)
        result.append(list(np.random.randint(min_val, max_val + 1, arr_len)))
    return result


def generate_float_array(
    num_rows: int,
    max_capacity: int = 10,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None,
) -> list[list[float]]:
    """Generate Array[Float] fields.

    Args:
        num_rows: Number of arrays to generate
        max_capacity: Maximum number of elements per array
        min_val: Minimum value for floats
        max_val: Maximum value for floats
        seed: Random seed for reproducibility

    Returns:
        List of lists of floats
    """
    if _rust_module is not None:
        return _rust_module.generate_float_array(
            num_rows, max_capacity, min_val, max_val, seed
        )

    # Python fallback
    result = []
    for _ in range(num_rows):
        arr_len = np.random.randint(1, max_capacity + 1)
        result.append(list(np.random.uniform(min_val, max_val, arr_len)))
    return result


# =============================================================================
# Numeric Batch Generation Functions
# =============================================================================


def generate_int64_batch(
    num_rows: int,
    min_val: int = 0,
    max_val: int = 1000000,
    seed: int | None = None,
) -> np.ndarray:
    """Generate batch of Int64 values.

    Args:
        num_rows: Number of values to generate
        min_val: Minimum value
        max_val: Maximum value
        seed: Random seed for reproducibility

    Returns:
        1D numpy array of int64
    """
    if _rust_module is not None:
        return _rust_module.generate_int64_batch(num_rows, min_val, max_val, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(min_val, max_val + 1, num_rows, dtype=np.int64)


def generate_float64_batch(
    num_rows: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate batch of Float64 values.

    Args:
        num_rows: Number of values to generate
        min_val: Minimum value
        max_val: Maximum value
        seed: Random seed for reproducibility

    Returns:
        1D numpy array of float64
    """
    if _rust_module is not None:
        return _rust_module.generate_float64_batch(num_rows, min_val, max_val, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(min_val, max_val, num_rows)

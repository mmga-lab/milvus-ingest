"""Type stubs for the Rust extension module."""

from typing import Any

import numpy as np
import numpy.typing as npt

def get_version() -> str: ...
def get_cpu_features() -> list[str]: ...

# Vector generation
def generate_float16_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> npt.NDArray[np.uint8]: ...

def generate_bfloat16_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> npt.NDArray[np.uint8]: ...

def generate_normalized_float16_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> npt.NDArray[np.uint8]: ...

def generate_float_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> npt.NDArray[np.float32]: ...

def generate_normalized_float_vectors(
    num_rows: int,
    dim: int,
    seed: int | None = None,
) -> npt.NDArray[np.float32]: ...

def generate_float_vectors_range(
    num_rows: int,
    dim: int,
    min_val: float,
    max_val: float,
    seed: int | None = None,
) -> npt.NDArray[np.float32]: ...

def generate_sparse_vectors(
    num_rows: int,
    max_dim: int = 1000,
    density_min: float = 0.01,
    density_max: float = 0.1,
    seed: int | None = None,
) -> list[dict[int, float]]: ...

def generate_sparse_vectors_nnz(
    num_rows: int,
    max_dim: int,
    nnz_min: int,
    nnz_max: int,
    seed: int | None = None,
) -> list[dict[int, float]]: ...

# Text generation
def generate_text_fields(
    num_rows: int,
    max_length: int = 256,
    text_type: str = "sentence",
    language: str = "en",
    seed: int | None = None,
) -> list[str]: ...

def generate_simple_text(
    num_rows: int,
    prefix: str = "text",
    pk_offset: int = 0,
) -> list[str]: ...

# JSON generation
def generate_json_fields(
    num_rows: int,
    pattern: str = "ecommerce",
    pk_offset: int = 0,
    seed: int | None = None,
) -> list[dict[str, Any]]: ...

# Array generation
def generate_varchar_array(
    num_rows: int,
    max_capacity: int = 10,
    max_length: int = 64,
    seed: int | None = None,
) -> list[list[str]]: ...

def generate_int_array(
    num_rows: int,
    max_capacity: int = 10,
    min_val: int = 0,
    max_val: int = 1000000,
    seed: int | None = None,
) -> list[list[int]]: ...

def generate_float_array(
    num_rows: int,
    max_capacity: int = 10,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None,
) -> list[list[float]]: ...

# Numeric batch generation
def generate_int64_batch(
    num_rows: int,
    min_val: int = 0,
    max_val: int = 1000000,
    seed: int | None = None,
) -> npt.NDArray[np.int64]: ...

def generate_float64_batch(
    num_rows: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None,
) -> npt.NDArray[np.float64]: ...

def generate_float32_batch_2d(
    num_rows: int,
    num_cols: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None,
) -> npt.NDArray[np.float32]: ...

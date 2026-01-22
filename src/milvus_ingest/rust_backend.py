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


# =============================================================================
# Faker Generation Functions (ported from synth-gen concepts)
# =============================================================================


def generate_faker_strings(
    count: int,
    faker_type: str,
    seed: int | None = None,
) -> list[str]:
    """Generate fake strings using the Rust faker module.

    Args:
        count: Number of strings to generate
        faker_type: Type of fake data. Supported types:
            - name, firstname, lastname: Person names
            - email, username: Internet identifiers
            - phone: Phone numbers
            - address, city, country: Location data
            - company, industry, jobtitle: Business data
            - url, domain, ipv4: Network data
            - useragent: Browser user agent strings
            - uuid: Unique identifiers
        seed: Random seed for reproducibility

    Returns:
        List of generated fake strings
    """
    if _rust_module is not None:
        return _rust_module.generate_faker_strings(count, faker_type, seed)

    # Python fallback - simple pattern-based generation
    return [f"{faker_type}_{i}" for i in range(count)]


def generate_categorical(
    count: int,
    categories: list[str],
    weights: list[float],
    seed: int | None = None,
) -> list[str]:
    """Generate categorical values from a weighted distribution.

    Args:
        count: Number of values to generate
        categories: List of category strings
        weights: List of weights (will be normalized to probabilities)
        seed: Random seed for reproducibility

    Returns:
        List of selected categories
    """
    if _rust_module is not None:
        return _rust_module.generate_categorical(count, categories, weights, seed)

    # Python fallback
    total = sum(weights)
    probs = [w / total for w in weights]
    indices = np.random.choice(len(categories), size=count, p=probs)
    return [categories[i] for i in indices]


def generate_range_numbers(
    count: int,
    low: float,
    high: float,
    step: float | None = None,
    seed: int | None = None,
) -> list[float]:
    """Generate range numbers with optional step.

    Args:
        count: Number of values to generate
        low: Minimum value (inclusive)
        high: Maximum value (exclusive)
        step: Optional step size
        seed: Random seed for reproducibility

    Returns:
        List of generated numbers
    """
    if _rust_module is not None:
        return _rust_module.generate_range_numbers(count, low, high, step, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    values = np.random.uniform(low, high, count)
    if step is not None and step > 0:
        values = low + np.floor((values - low) / step) * step
    return list(values)


def generate_range_integers(
    count: int,
    low: int,
    high: int,
    step: int | None = None,
    seed: int | None = None,
) -> list[int]:
    """Generate integers in a range with optional step.

    Args:
        count: Number of values to generate
        low: Minimum value (inclusive)
        high: Maximum value (exclusive)
        step: Optional step size
        seed: Random seed for reproducibility

    Returns:
        List of generated integers
    """
    if _rust_module is not None:
        return _rust_module.generate_range_integers(count, low, high, step, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    step_val = step if step else 1
    num_steps = (high - low) // step_val
    step_indices = np.random.randint(0, max(1, num_steps), count)
    return list(low + step_indices * step_val)


def generate_frequency_bool(
    count: int,
    true_probability: float = 0.5,
    seed: int | None = None,
) -> list[bool]:
    """Generate boolean values with specified true probability.

    Args:
        count: Number of values to generate
        true_probability: Probability of generating true (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of generated booleans
    """
    if _rust_module is not None:
        return _rust_module.generate_frequency_bool(count, true_probability, seed)

    # Python fallback
    if seed is not None:
        np.random.seed(seed)
    return list(np.random.random(count) < true_probability)


def generate_date_strings(
    count: int,
    start_year: int = 2020,
    end_year: int = 2024,
    seed: int | None = None,
) -> list[str]:
    """Generate date strings in ISO format (YYYY-MM-DD).

    Args:
        count: Number of dates to generate
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        seed: Random seed for reproducibility

    Returns:
        List of date strings
    """
    if _rust_module is not None:
        return _rust_module.generate_date_strings(count, start_year, end_year, seed)

    # Python fallback
    import random

    if seed is not None:
        random.seed(seed)
    dates = []
    for _ in range(count):
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Simplified
        dates.append(f"{year:04d}-{month:02d}-{day:02d}")
    return dates


def generate_datetime_strings(
    count: int,
    start_year: int = 2020,
    end_year: int = 2024,
    seed: int | None = None,
) -> list[str]:
    """Generate datetime strings in ISO format (YYYY-MM-DDTHH:MM:SS).

    Args:
        count: Number of datetimes to generate
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        seed: Random seed for reproducibility

    Returns:
        List of datetime strings
    """
    if _rust_module is not None:
        return _rust_module.generate_datetime_strings(count, start_year, end_year, seed)

    # Python fallback
    import random

    if seed is not None:
        random.seed(seed)
    datetimes = []
    for _ in range(count):
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Simplified
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        datetimes.append(
            f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
        )
    return datetimes


def generate_timestamptz_strings(
    count: int,
    start_year: int = 2020,
    end_year: int = 2024,
    timezone_offset_hours: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Generate TIMESTAMPTZ strings in ISO 8601 format with timezone.

    Output format: YYYY-MM-DDTHH:MM:SS+HH:MM (e.g., "2025-05-01T23:59:59+08:00")
    Compatible with Milvus TIMESTAMPTZ field type (Milvus 2.6.6+).

    Args:
        count: Number of timestamps to generate
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        timezone_offset_hours: Fixed timezone offset in hours (e.g., 8 for +08:00, -5 for -05:00).
                               If None, generates random offsets from common timezones.
        seed: Random seed for reproducibility

    Returns:
        List of TIMESTAMPTZ strings
    """
    if _rust_module is not None:
        return _rust_module.generate_timestamptz_strings(
            count, start_year, end_year, timezone_offset_hours, seed
        )

    # Python fallback using datetime
    import random
    from datetime import datetime, timedelta, timezone

    if seed is not None:
        random.seed(seed)

    tz_offsets = [
        -12,
        -11,
        -10,
        -9,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        0,
        1,
        2,
        3,
        5,
        8,
        9,
        10,
        12,
    ]
    timestamps = []
    for _ in range(count):
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        offset_hours = (
            timezone_offset_hours
            if timezone_offset_hours is not None
            else random.choice(tz_offsets)
        )
        tz = timezone(timedelta(hours=offset_hours))
        dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
        timestamps.append(
            dt.strftime("%Y-%m-%dT%H:%M:%S%z")[:22] + ":" + dt.strftime("%z")[3:]
        )
    return timestamps


def generate_sequential_ids(
    count: int,
    prefix: str = "id",
    start_index: int = 0,
) -> list[str]:
    """Generate unique IDs with prefix.

    Args:
        count: Number of IDs to generate
        prefix: Prefix for the ID
        start_index: Starting index

    Returns:
        List of generated IDs
    """
    if _rust_module is not None:
        return _rust_module.generate_sequential_ids(count, prefix, start_index)

    # Python fallback
    return [f"{prefix}_{start_index + i}" for i in range(count)]


# =============================================================================
# Geometry (WKT) Generation Functions
# =============================================================================


def generate_wkt_points(
    count: int,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    seed: int | None = None,
) -> list[str]:
    """Generate WKT POINT strings.

    Output format: POINT (longitude latitude)
    Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).

    Args:
        count: Number of points to generate
        lon_min: Minimum longitude (-180 to 180)
        lon_max: Maximum longitude
        lat_min: Minimum latitude (-90 to 90)
        lat_max: Maximum latitude
        seed: Random seed for reproducibility

    Returns:
        List of WKT POINT strings
    """
    if _rust_module is not None:
        return _rust_module.generate_wkt_points(
            count, lon_min, lon_max, lat_min, lat_max, seed
        )

    # Python fallback
    import random

    if seed is not None:
        random.seed(seed)
    return [
        f"POINT ({random.uniform(lon_min, lon_max):.6f} {random.uniform(lat_min, lat_max):.6f})"
        for _ in range(count)
    ]


def generate_wkt_linestrings(
    count: int,
    points_min: int = 2,
    points_max: int = 5,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    seed: int | None = None,
) -> list[str]:
    """Generate WKT LINESTRING strings.

    Output format: LINESTRING (x1 y1, x2 y2, ...)
    Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).

    Args:
        count: Number of linestrings to generate
        points_min: Minimum number of points per linestring
        points_max: Maximum number of points per linestring
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        seed: Random seed for reproducibility

    Returns:
        List of WKT LINESTRING strings
    """
    if _rust_module is not None:
        return _rust_module.generate_wkt_linestrings(
            count, points_min, points_max, lon_min, lon_max, lat_min, lat_max, seed
        )

    # Python fallback
    import random

    if seed is not None:
        random.seed(seed)
    result = []
    for _ in range(count):
        num_points = random.randint(max(2, points_min), points_max)
        points = [
            f"{random.uniform(lon_min, lon_max):.6f} {random.uniform(lat_min, lat_max):.6f}"
            for _ in range(num_points)
        ]
        result.append(f"LINESTRING ({', '.join(points)})")
    return result


def generate_wkt_polygons(
    count: int,
    vertices_min: int = 4,
    vertices_max: int = 8,
    center_lon: float = 0.0,
    center_lat: float = 0.0,
    radius: float = 1.0,
    seed: int | None = None,
) -> list[str]:
    """Generate WKT POLYGON strings.

    Output format: POLYGON ((x1 y1, x2 y2, ..., x1 y1))
    Generates simple convex polygons (no holes).
    Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).

    Args:
        count: Number of polygons to generate
        vertices_min: Minimum number of vertices (3+)
        vertices_max: Maximum number of vertices
        center_lon: Center longitude for polygon generation
        center_lat: Center latitude for polygon generation
        radius: Approximate radius in degrees
        seed: Random seed for reproducibility

    Returns:
        List of WKT POLYGON strings
    """
    if _rust_module is not None:
        return _rust_module.generate_wkt_polygons(
            count, vertices_min, vertices_max, center_lon, center_lat, radius, seed
        )

    # Python fallback
    import math
    import random

    if seed is not None:
        random.seed(seed)
    result = []
    for _ in range(count):
        num_vertices = random.randint(max(3, vertices_min), vertices_max)
        cx = center_lon + random.uniform(-radius, radius)
        cy = center_lat + random.uniform(-radius, radius)
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        points = []
        for angle in angles:
            r = radius * (0.5 + random.uniform(0, 0.5))
            lon = cx + r * math.cos(angle)
            lat = cy + r * math.sin(angle)
            points.append(f"{lon:.6f} {lat:.6f}")
        points.append(points[0])  # Close the ring
        result.append(f"POLYGON (({', '.join(points)}))")
    return result


def generate_wkt_geometries(
    count: int,
    geometry_type: str = "point",
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    seed: int | None = None,
) -> list[str]:
    """Generate WKT geometry strings of specified type.

    Generates POINT, LINESTRING, POLYGON, or mixed geometries.
    Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).

    Args:
        count: Number of geometries to generate
        geometry_type: Type of geometry ("point", "linestring", "polygon", "mixed")
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        seed: Random seed for reproducibility

    Returns:
        List of WKT geometry strings
    """
    if _rust_module is not None:
        return _rust_module.generate_wkt_geometries(
            count, geometry_type, lon_min, lon_max, lat_min, lat_max, seed
        )

    # Python fallback
    geom_type = geometry_type.lower()
    if geom_type == "point":
        return generate_wkt_points(count, lon_min, lon_max, lat_min, lat_max, seed)
    elif geom_type == "linestring":
        return generate_wkt_linestrings(
            count, 2, 5, lon_min, lon_max, lat_min, lat_max, seed
        )
    elif geom_type == "polygon":
        cx = (lon_min + lon_max) / 2
        cy = (lat_min + lat_max) / 2
        radius = min(lon_max - lon_min, lat_max - lat_min) * 0.1
        return generate_wkt_polygons(count, 4, 8, cx, cy, radius, seed)
    else:  # mixed
        import random

        if seed is not None:
            random.seed(seed)
        result = []
        for _ in range(count):
            t = random.choice(["point", "linestring", "polygon"])
            if t == "point":
                result.extend(generate_wkt_points(1, lon_min, lon_max, lat_min, lat_max))
            elif t == "linestring":
                result.extend(
                    generate_wkt_linestrings(1, 2, 5, lon_min, lon_max, lat_min, lat_max)
                )
            else:
                cx = (lon_min + lon_max) / 2
                cy = (lat_min + lat_max) / 2
                radius = min(lon_max - lon_min, lat_max - lat_min) * 0.1
                result.extend(generate_wkt_polygons(1, 4, 6, cx, cy, radius))
        return result


def list_faker_types() -> list[str]:
    """List all available faker types.

    Returns:
        List of supported faker type strings
    """
    if _rust_module is not None:
        return _rust_module.list_faker_types()

    # Python fallback
    return [
        "name",
        "firstname",
        "lastname",
        "email",
        "username",
        "phone",
        "address",
        "city",
        "country",
        "company",
        "industry",
        "jobtitle",
        "url",
        "domain",
        "ipv4",
        "useragent",
        "uuid",
        "word",
        "sentence",
        "paragraph",
    ]


# =============================================================================
# Lorem Text Generation Functions
# =============================================================================


def generate_lorem_words(
    count: int,
    words_per_item: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Generate Lorem Ipsum words.

    Args:
        count: Number of word items to generate
        words_per_item: Number of words per item (default 1)
        seed: Random seed for reproducibility

    Returns:
        List of generated word strings
    """
    if _rust_module is not None:
        return _rust_module.generate_lorem_words(count, words_per_item, seed)

    # Python fallback - simple lorem vocabulary
    lorem_words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
        "incididunt",
        "ut",
        "labore",
        "et",
        "dolore",
        "magna",
        "aliqua",
    ]
    import random

    if seed is not None:
        random.seed(seed)
    result = []
    for _ in range(count):
        words = [random.choice(lorem_words) for _ in range(words_per_item)]
        result.append(" ".join(words))
    return result


def generate_lorem_sentences(
    count: int,
    word_count_min: int = 4,
    word_count_max: int = 10,
    seed: int | None = None,
) -> list[str]:
    """Generate Lorem Ipsum sentences.

    Args:
        count: Number of sentences to generate
        word_count_min: Minimum words per sentence (default 4)
        word_count_max: Maximum words per sentence (default 10)
        seed: Random seed for reproducibility

    Returns:
        List of generated sentences
    """
    if _rust_module is not None:
        return _rust_module.generate_lorem_sentences(
            count, word_count_min, word_count_max, seed
        )

    # Python fallback
    import random

    if seed is not None:
        random.seed(seed)
    lorem_words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
    ]
    result = []
    for _ in range(count):
        word_count = random.randint(word_count_min, word_count_max)
        words = [random.choice(lorem_words) for _ in range(word_count)]
        words[0] = words[0].capitalize()
        result.append(" ".join(words) + ".")
    return result


def generate_lorem_paragraphs(
    count: int,
    sentence_count_min: int = 3,
    sentence_count_max: int = 6,
    seed: int | None = None,
) -> list[str]:
    """Generate Lorem Ipsum paragraphs.

    Args:
        count: Number of paragraphs to generate
        sentence_count_min: Minimum sentences per paragraph (default 3)
        sentence_count_max: Maximum sentences per paragraph (default 6)
        seed: Random seed for reproducibility

    Returns:
        List of generated paragraphs
    """
    if _rust_module is not None:
        return _rust_module.generate_lorem_paragraphs(
            count, sentence_count_min, sentence_count_max, seed
        )

    # Python fallback
    import random

    if seed is not None:
        random.seed(seed)
    result = []
    for i in range(count):
        sentence_count = random.randint(sentence_count_min, sentence_count_max)
        sentences = generate_lorem_sentences(
            sentence_count, seed=seed + i if seed else None
        )
        result.append(" ".join(sentences))
    return result


def generate_lorem_text(
    count: int,
    text_type: str = "sentence",
    min_units: int = 4,
    max_units: int = 10,
    seed: int | None = None,
) -> list[str]:
    """Generate Lorem Ipsum text with configurable type.

    This is a flexible text generator for realistic-looking text.

    Args:
        count: Number of text items to generate
        text_type: Type of text - "word", "sentence", "paragraph", "text"
        min_units: Minimum units (words/sentences depending on type)
        max_units: Maximum units
        seed: Random seed for reproducibility

    Returns:
        List of generated text strings
    """
    if _rust_module is not None:
        return _rust_module.generate_lorem_text(
            count, text_type, min_units, max_units, seed
        )

    # Python fallback
    text_type = text_type.lower()
    if text_type in ("word", "words"):
        return generate_lorem_words(count, max_units, seed)
    elif text_type in ("paragraph", "paragraphs", "text"):
        return generate_lorem_paragraphs(count, min_units, max_units, seed)
    else:
        return generate_lorem_sentences(count, min_units, max_units, seed)

"""Core logic for generating mock data from a Milvus collection schema (offline file).

The module exposes a single public function `generate_mock_data`, which returns a
pandas `DataFrame` containing random data following the provided schema.

Supported field types (case-insensitive):
    • Int8 / Int16 / Int32 / Int64
    • Float / Double
    • Bool
    • VarChar (string)
    • FloatVector (requires `dim` in field definition)
    • BinaryVector (requires `dim` in field definition, dim is number of bits)

Schema file format (JSON / YAML):
{
  "collection_name": "my_collection",
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true},
    {"name": "age", "type": "Int32"},
    {"name": "embedding", "type": "FloatVector", "dim": 128}
  ]
}

If the top-level key `fields` is missing we assume the whole dict is already a
list of fields.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import pandas as pd
import yaml
from faker import Faker
from ml_dtypes import bfloat16
from pydantic import ValidationError

from .logging_config import (
    get_logger,
    log_error_with_context,
    log_performance,
    log_schema_validation,
)
from .models import get_schema_help, validate_schema_data

faker = Faker()
NULL_PROB = 0.1  # probability to generate null for nullable fields

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_mock_data_batches(
    schema_path: str | Path,
    rows: int = 1000,
    batch_size: int = 10000,
    seed: int | None = None,
    show_progress: bool = True,
) -> Iterator[pd.DataFrame]:
    """Generate mock data in batches to handle large datasets efficiently.

    Args:
        schema_path: Path to JSON or YAML schema file.
        rows: Total number of rows to generate.
        batch_size: Number of rows per batch.
        seed: Optional random seed for reproducibility.
        show_progress: Whether to display progress bar.

    Yields:
        pandas.DataFrame containing batch_size rows (or fewer for last batch).
    """
    import time

    logger = get_logger(__name__)
    start_time = time.time()

    logger.info(
        "Starting batch data generation",
        schema_file=str(schema_path),
        rows=rows,
        batch_size=batch_size,
        seed=seed,
    )

    if seed is not None:
        logger.debug("Setting random seeds", seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

    schema_data = _load_schema(schema_path)
    logger.debug("Schema loaded successfully", schema_file=str(schema_path))
    try:
        validated_schema = validate_schema_data(schema_data)
        if isinstance(validated_schema, list):
            # List of field schemas
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema
            ]
            schema_name = f"schema_from_{Path(schema_path).stem}"
            fields_count = len(validated_schema)
        else:
            # Collection schema
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema.fields
            ]
            schema_name = validated_schema.collection_name or "unknown"
            fields_count = len(validated_schema.fields)

        log_schema_validation(schema_name, fields_count, "success")
        logger.debug(
            "Schema validation completed",
            schema_name=schema_name,
            fields_count=fields_count,
        )
    except ValidationError as e:
        # Format validation errors with helpful messages
        error_list = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            error_list.append(f"{loc}: {error['msg']}")

        log_schema_validation("unknown_schema", 0, "failed", error_list)
        log_error_with_context(
            e, {"schema_file": str(schema_path), "operation": "schema_validation"}
        )
        error_msg = "Schema validation failed:\n\n"
        error_msg += "\n".join(f"• {err}" for err in error_list)
        error_msg += f"\n{get_schema_help()}"
        raise ValueError(error_msg) from e

    # Generate timestamp base for reproducible tests
    if seed is not None:
        # Use deterministic base for reproducible tests
        base_ts = (seed * 1000) << 18
    else:
        import time

        base_ts = int(time.time() * 1000) << 18

    pk_field = next((f for f in fields if f.get("is_primary")), None)
    total_batches = (rows + batch_size - 1) // batch_size

    logger.info(
        "Starting batch generation",
        total_batches=total_batches,
        pk_field=pk_field["name"] if pk_field else None,
    )

    # Generate data in batches
    if show_progress:
        print(f"🎯 Generating {rows:,} rows of mock data...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, rows)
        
        if show_progress and batch_idx % 10 == 0:  # Print progress every 10 batches
            print(f"📊 Progress: {end_idx:,}/{rows:,} rows ({100.0 * end_idx / rows:.1f}%)")

        rows_data: list[dict[str, Any]] = []
        for idx in range(start_idx, end_idx):
            row: dict[str, Any] = {}
            for f in fields:
                name = f["name"]
                f_type = f["type"].upper()
                # Skip auto_id fields
                if f.get("auto_id"):
                    continue
                # Primary key handling (monotonic unique)
                if pk_field and name == pk_field["name"]:
                    row[name] = _gen_pk_value(f_type, base_ts, idx)
                    continue
                # Nullable handling
                if f.get("nullable") and random.random() < NULL_PROB:
                    row[name] = None
                    continue
                # Generate value by type
                row[name] = _gen_value_by_field(f)
            rows_data.append(row)

        logger.debug(
            "Batch generated",
            batch_idx=batch_idx + 1,
            batch_size=len(rows_data),
            start_idx=start_idx,
            end_idx=end_idx,
        )
        yield pd.DataFrame(rows_data)

    # Log performance metrics
    total_time = time.time() - start_time
    log_performance(
        "generate_mock_data_batches",
        total_time,
        rows=rows,
        batch_size=batch_size,
        total_batches=total_batches,
    )
    logger.info(
        "Batch generation completed",
        total_rows=rows,
        total_batches=total_batches,
        duration_seconds=total_time,
    )


def generate_mock_data(
    schema_path: str | Path,
    rows: int = 1000,
    seed: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Generate mock data according to the schema described by *schema_path*.

    Args:
        schema_path: Path to JSON or YAML schema file.
        rows: Number of rows to generate.
        seed: Optional random seed for reproducibility.
        show_progress: Whether to display progress bar for large datasets.

    Returns:
        A *pandas.DataFrame* containing mock data with *rows* rows.
    """
    # For backward compatibility, use batch generation and concatenate results
    batches = list(
        generate_mock_data_batches(
            schema_path,
            rows=rows,
            batch_size=10000,
            seed=seed,
            show_progress=show_progress,
        )
    )

    if not batches:
        # Return empty DataFrame with correct columns if no data
        schema_data = _load_schema(schema_path)
        validated_schema = validate_schema_data(schema_data)
        if isinstance(validated_schema, list):
            fields = validated_schema
        else:
            fields = validated_schema.fields
        columns = [f.name for f in fields if not f.auto_id]
        return pd.DataFrame(columns=pd.Index(columns))

    return pd.concat(batches, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_pk_value(f_type: str, base_ts: int, idx: int) -> int | str:
    """Generate unique & monotonic primary key value."""
    if f_type == "INT64":
        return base_ts + idx
    return str(base_ts + idx)


def _gen_value_by_field(field: dict[str, Any]) -> Any:
    """Generate a random value matching field definition."""
    f_type = field["type"].upper()
    # Canonicalize vector type names like FLOATVECTOR -> FLOAT_VECTOR
    if "VECTOR" in f_type and "_VECTOR" not in f_type:
        # Handle special case for SparseFloatVector
        if f_type == "SPARSEFLOATVECTOR":
            f_type = "SPARSE_FLOAT_VECTOR"
        else:
            f_type = f_type.replace("VECTOR", "_VECTOR")
    dim = int(field.get("dim", 8))
    max_length = int(field.get("max_length", 128))
    if f_type == "BOOL":
        return random.choice([True, False])
    if f_type in {"INT8", "INT16", "INT32", "INT64"}:
        low, high = field.get("min", 0), field.get("max", 1_000_000)
        return random.randint(low, high)
    if f_type in {"FLOAT", "DOUBLE"}:
        low, high = field.get("min", 0.0), field.get("max", 1_000.0)
        return random.uniform(low, high)
    if f_type in {"VARCHAR", "STRING"}:
        # Use 80% of max_length to avoid hitting the limit, minimum 5 chars for faker
        safe_max_length = max(5, int(max_length * 0.8))
        return faker.text(max_nb_chars=safe_max_length)
    if f_type == "JSON":
        return {"key": faker.text(max_nb_chars=16)}
    if f_type == "ARRAY":
        element_type = field.get("element_type", "INT32").upper()
        max_capacity = int(field.get("max_capacity", 5))
        arr_len = random.randint(1, max_capacity)
        # Create element field with proper constraints
        element_field = {"type": element_type}
        # Pass max_length to VARCHAR elements in arrays
        if element_type in {"VARCHAR", "STRING"} and "max_length" in field:
            element_field["max_length"] = field["max_length"]
        return [_gen_value_by_field(element_field) for _ in range(arr_len)]
    # Vector types
    if f_type == "BINARY_VECTOR":
        # Binary vector: each int represents 8 dimensions
        # If binary vector dimension is 16, use [x, y] where x and y are 0-255
        byte_dim = dim // 8
        return [random.randint(0, 255) for _ in range(byte_dim)]
    if f_type in {"FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"}:
        if f_type == "FLOAT_VECTOR":
            dtype = np.float32
            return np.random.random(dim).astype(dtype).tolist()
        elif f_type == "FLOAT16_VECTOR":
            # Generate float16 vector data using uint8 representation
            raw_vector = [random.random() for _ in range(dim)]
            fp16_vector = np.array(raw_vector, dtype=np.float16).view(np.uint8).tolist()
            return fp16_vector
        elif f_type == "BFLOAT16_VECTOR":
            # Generate bfloat16 vector data using uint8 representation
            raw_vector = [random.random() for _ in range(dim)]
            bf16_vector = np.array(raw_vector, dtype=bfloat16).view(np.uint8).tolist()
            return bf16_vector
    if f_type == "SPARSE_FLOAT_VECTOR":
        # Generate sparse float vector as dict with indices as keys and values as floats
        # Use default dimension of 1000 for sparse vectors, with ~10% density
        max_dim = 1000
        non_zero_count = random.randint(10, max_dim // 10)  # 10-100 non-zero values
        indices = random.sample(range(max_dim), non_zero_count)
        values = [random.random() for _ in range(non_zero_count)]
        sparse_vector = {
            str(index): value for index, value in zip(indices, values, strict=False)
        }
        return sparse_vector
    raise ValueError(f"Unsupported field type: {f_type}")


def _load_schema(path: str | Path) -> dict[str, Any] | list[Any]:
    """Load JSON or YAML schema file and return as dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return cast("dict[str, Any]", yaml.safe_load(content))
    elif path.suffix.lower() == ".json":
        return cast("dict[str, Any]", json.loads(content))
    else:
        # fallback: try json then yaml
        try:
            return cast("dict[str, Any]", json.loads(content))
        except json.JSONDecodeError:
            return cast("dict[str, Any]", yaml.safe_load(content))

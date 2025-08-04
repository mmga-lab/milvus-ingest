"""Simplified optimized writer focusing on core performance improvements."""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from ml_dtypes import bfloat16

# Pure NumPy optimizations - consistently outperforms JIT for vector operations
# Uses optimized BLAS libraries for multi-core CPU utilization
# Note: Removed module-level debug log to avoid interfering with progress bars


class GenerationParams:
    """Container for validated generation parameters."""
    
    def __init__(self, file_size: str, file_count: int, total_rows: int):
        self.file_size = file_size
        self.file_count = file_count
        self.total_rows = total_rows


def validate_and_calculate_params(
    file_size: str | None,
    file_count: int | None,
    total_rows: int,
    fields: list[dict[str, Any]],
    format: str,
    schema: dict[str, Any],
    seed: int | None = None,
) -> GenerationParams:
    """
    Validate generation parameters and calculate the missing third parameter.
    
    Rules:
    - Exactly two of (file_size, file_count, total_rows) must be provided
    - The third parameter will be calculated based on sampling estimation
    
    Args:
        file_size: Target file size (e.g., "1GB", "512MB")
        file_count: Number of files to generate
        total_rows: Total number of rows to generate
        fields: Schema field definitions for sampling
        format: Output format ("parquet" or "json")
        schema: Full schema definition
        seed: Random seed for sampling
    
    Returns:
        GenerationParams with all three parameters set
        
    Raises:
        ValueError: If parameter combination is invalid
    """
    # Count provided parameters
    # Note: total_rows has a default value of 1000 from CLI, but if we have both 
    # file_size and file_count, we should ignore total_rows and calculate it instead
    provided_params = []
    if file_size is not None:
        provided_params.append("file_size")
    if file_count is not None:
        provided_params.append("file_count") 
    
    # If both file_size and file_count are provided, we ignore total_rows 
    # Note: CLI sets total_rows=0 when both file_size and file_count are provided
    if file_size is not None and file_count is not None:
        # Special case: both provided, ignore total_rows
        pass  # Don't add total_rows to provided_params
    elif total_rows > 0:
        provided_params.append("total_rows")
    
    if len(provided_params) != 2:
        if len(provided_params) == 0:
            # Use default behavior: total_rows is provided, calculate others
            provided_params = ["total_rows"]
        elif len(provided_params) == 3:
            raise ValueError(
                "Cannot specify all three parameters (file_size, file_count, total_rows) simultaneously. "
                "Please provide exactly two parameters, and the third will be calculated automatically."
            )
        elif len(provided_params) == 1:
            if "total_rows" in provided_params:
                # Only total_rows provided - use default file size (256MB) to calculate file_count
                file_size = "256MB"
                provided_params.append("file_size")
            else:
                raise ValueError(
                    f"Only {provided_params[0]} provided. Please provide exactly two of: "
                    "file_size, file_count, total_rows"
                )
    
    logger.info(f"Parameter mode: {' + '.join(sorted(provided_params))} → calculating remaining parameter")
    
    # Calculate the missing parameter using sampling estimation
    if "total_rows" not in provided_params:
        # Calculate total_rows from file_size + file_count
        return _calculate_total_rows_from_size_and_count(
            file_size, file_count, fields, format, schema, seed
        )
    elif "file_count" not in provided_params:
        # Calculate file_count from file_size + total_rows
        return _calculate_file_count_from_size_and_rows(
            file_size, total_rows, fields, format, schema, seed
        )
    else:  # "file_size" not in provided_params
        # Calculate file_size from file_count + total_rows
        return _calculate_file_size_from_count_and_rows(
            file_count, total_rows, fields, format, schema, seed
        )


def _calculate_total_rows_from_size_and_count(
    file_size: str,
    file_count: int,
    fields: list[dict[str, Any]],
    format: str,
    schema: dict[str, Any],
    seed: int | None,
) -> GenerationParams:
    """Calculate total_rows from file_size + file_count using sampling."""
    logger.info("Calculating total_rows from file_size + file_count...")
    
    # Use enhanced sampling to estimate bytes per row
    sample_size = 5000  # Good balance for estimation
    actual_row_size_bytes, estimation_stats = _enhanced_estimate_row_size_from_sample(
        fields, sample_size, 0, format, schema, seed, num_iterations=3
    )
    
    # Calculate rows per file based on target file size
    target_size_bytes = _parse_file_size(file_size)
    rows_per_file = max(1, int(target_size_bytes // actual_row_size_bytes))
    calculated_total_rows = file_count * rows_per_file
    
    logger.info(
        f"Calculated: {calculated_total_rows:,} total rows "
        f"({file_count} files × {rows_per_file:,} rows/file, "
        f"~{actual_row_size_bytes:.1f} bytes/row)"
    )
    
    return GenerationParams(file_size, file_count, calculated_total_rows)


def _calculate_file_count_from_size_and_rows(
    file_size: str,
    total_rows: int,
    fields: list[dict[str, Any]],
    format: str,
    schema: dict[str, Any],
    seed: int | None,
) -> GenerationParams:
    """Calculate file_count from file_size + total_rows using sampling."""
    logger.info("Calculating file_count from file_size + total_rows...")
    
    # Use enhanced sampling to estimate bytes per row
    sample_size = min(max(1000, total_rows // 50), 10000)
    actual_row_size_bytes, estimation_stats = _enhanced_estimate_row_size_from_sample(
        fields, sample_size, 0, format, schema, seed, num_iterations=3
    )
    
    # Calculate how many files are needed
    target_size_bytes = _parse_file_size(file_size)
    rows_per_file = max(1, int(target_size_bytes // actual_row_size_bytes))
    calculated_file_count = max(1, (total_rows + rows_per_file - 1) // rows_per_file)
    
    logger.info(
        f"Calculated: {calculated_file_count} files needed "
        f"({total_rows:,} rows ÷ {rows_per_file:,} rows/file, "
        f"~{actual_row_size_bytes:.1f} bytes/row)"
    )
    
    return GenerationParams(file_size, calculated_file_count, total_rows)


def _calculate_file_size_from_count_and_rows(
    file_count: int,
    total_rows: int,
    fields: list[dict[str, Any]],
    format: str,
    schema: dict[str, Any],
    seed: int | None,
) -> GenerationParams:
    """Calculate file_size from file_count + total_rows using sampling."""
    logger.info("Calculating file_size from file_count + total_rows...")
    
    # Use enhanced sampling to estimate bytes per row
    sample_size = min(max(1000, total_rows // 50), 10000)
    actual_row_size_bytes, estimation_stats = _enhanced_estimate_row_size_from_sample(
        fields, sample_size, 0, format, schema, seed, num_iterations=3
    )
    
    # Calculate target file size
    rows_per_file = max(1, total_rows // file_count)
    target_size_bytes = int(rows_per_file * actual_row_size_bytes)
    
    # Convert to human readable format
    if target_size_bytes >= 1024**3:
        calculated_file_size = f"{target_size_bytes / (1024**3):.1f}GB"
    elif target_size_bytes >= 1024**2:
        calculated_file_size = f"{target_size_bytes / (1024**2):.0f}MB"
    else:
        calculated_file_size = f"{target_size_bytes / 1024:.0f}KB"
    
    logger.info(
        f"Calculated: {calculated_file_size} per file "
        f"({total_rows:,} rows ÷ {file_count} files = {rows_per_file:,} rows/file, "
        f"~{actual_row_size_bytes:.1f} bytes/row)"
    )
    
    return GenerationParams(calculated_file_size, file_count, total_rows)


def calculate_chunk_parameters(
    file_size: str,
    chunk_size: str,
    fields: list[dict[str, Any]],
    format: str,
    schema: dict[str, Any],
    seed: int | None = None,
) -> tuple[int, int, int]:
    """
    Calculate chunk parameters based on file size and chunk size using sampling.
    
    Args:
        file_size: Target file size (e.g., "1GB", "512MB")
        chunk_size: Chunk size (e.g., "512MB", "1GB")
        fields: Schema field definitions for sampling
        format: Output format ("parquet" or "json")
        schema: Full schema definition
        seed: Random seed for sampling
    
    Returns:
        tuple of (rows_per_chunk, rows_per_file, chunks_per_file)
    """
    # Use enhanced sampling to estimate bytes per row
    sample_size = 5000  # Good balance for estimation
    actual_row_size_bytes, estimation_stats = _enhanced_estimate_row_size_from_sample(
        fields, sample_size, 0, format, schema, seed, num_iterations=3
    )
    
    # Parse sizes
    chunk_size_bytes = _parse_file_size(chunk_size)
    file_size_bytes = _parse_file_size(file_size)
    
    # Calculate rows per chunk and file
    rows_per_chunk = max(1, int(chunk_size_bytes // actual_row_size_bytes))
    rows_per_file = max(1, int(file_size_bytes // actual_row_size_bytes))
    
    # Calculate chunks per file
    chunks_per_file = max(1, (rows_per_file + rows_per_chunk - 1) // rows_per_chunk)
    
    logger.info(
        f"Chunk calculation: {actual_row_size_bytes:.1f} bytes/row → "
        f"{rows_per_chunk:,} rows/chunk, {chunks_per_file} chunks/file"
    )
    
    return rows_per_chunk, rows_per_file, chunks_per_file


def determine_generation_strategy(
    file_size: str,
    file_count: int,
    chunk_size: str = "512MB",
) -> dict[str, Any]:
    """
    Determine the optimal generation strategy based on file characteristics.
    
    Enhanced strategy selection for optimal multi-processing utilization:
    - Small files (<chunk_size): Direct parallel file generation
    - Large files + few count: Chunk-parallel with merge
    - Many small files: Batch-parallel processing
    - Many large files: Adaptive batch-chunk processing
    
    Args:
        file_size: Target file size (e.g., "1GB", "512MB")
        file_count: Number of files to generate
        chunk_size: Chunk size for chunk-and-merge strategy
    
    Returns:
        Dictionary with strategy information including worker optimization
    """
    file_size_bytes = _parse_file_size(file_size)
    chunk_size_bytes = _parse_file_size(chunk_size)
    file_size_mb = file_size_bytes / (1024**2)
    file_size_gb = file_size_bytes / (1024**3)
    cpu_count = multiprocessing.cpu_count()
    
    chunks_per_file = (file_size_bytes + chunk_size_bytes - 1) // chunk_size_bytes if file_size_bytes >= chunk_size_bytes else 1
    total_chunks = chunks_per_file * file_count
    
    # Strategy 1: Small files (< chunk_size) - Direct parallel
    if file_size_bytes < chunk_size_bytes:
        if file_count <= cpu_count * 2:
            # Few small files: full parallelism
            return {
                "strategy": "direct_parallel",
                "reason": f"Small files ({file_size_mb:.0f}MB < {chunk_size}), count manageable ({file_count})",
                "chunks_per_file": 1,
                "use_chunk_and_merge": False,
                "max_parallel_files": min(file_count, cpu_count * 2),
                "memory_profile": "low",
                "optimization": "cpu_bound"
            }
        else:
            # Many small files: batch processing
            batch_size = cpu_count * 4  # Process more files per batch for small files
            return {
                "strategy": "batch_direct_parallel",
                "reason": f"Many small files ({file_count}), using batch processing",
                "chunks_per_file": 1,
                "use_chunk_and_merge": False,
                "batch_size": batch_size,
                "max_parallel_files": cpu_count * 2,
                "memory_profile": "low",
                "optimization": "throughput_optimized"
            }
    
    # Strategy 2: Large files (>= chunk_size)
    
    # 2.1: Few large files - Full chunk parallelism (e.g., 10×10GB)
    if file_count <= 20 and file_size_gb >= 2:
        return {
            "strategy": "chunk_parallel_merge",
            "reason": f"Few large files ({file_count}×{file_size_gb:.1f}GB), maximize chunk parallelism",
            "chunks_per_file": chunks_per_file,
            "use_chunk_and_merge": True,
            "max_parallel_chunks": min(total_chunks, cpu_count * 3),  # Aggressive parallelism
            "memory_profile": "high" if file_size_gb >= 5 else "medium",
            "optimization": "chunk_parallel_optimized"
        }
    
    # 2.2: Many medium files - Balanced approach (e.g., 1000×100MB)
    elif file_count >= 100 and file_size_bytes < 2 * 1024**3:  # < 2GB per file
        # For many medium files, use file-level batching rather than chunk-level
        return {
            "strategy": "batch_file_parallel",
            "reason": f"Many medium files ({file_count}×{file_size_mb:.0f}MB), file-level batching",
            "chunks_per_file": chunks_per_file,
            "use_chunk_and_merge": True,
            "batch_size": max(10, cpu_count),  # Process 10+ files per batch
            "max_parallel_files": cpu_count,
            "memory_profile": "medium",
            "optimization": "balanced_throughput"
        }
    
    # 2.3: Medium scale - Standard chunk processing
    elif total_chunks <= cpu_count * 4:
        return {
            "strategy": "chunk_parallel_merge",
            "reason": f"Medium scale ({total_chunks} chunks), standard chunk processing",
            "chunks_per_file": chunks_per_file,
            "use_chunk_and_merge": True,
            "max_parallel_chunks": min(total_chunks, cpu_count * 2),
            "memory_profile": "medium",
            "optimization": "standard"
        }
    
    # 2.4: Large scale - Adaptive batch-chunk processing
    else:
        # For very large scale, use adaptive batching
        if file_size_gb >= 5:
            # Very large files: conservative batching
            batch_size = max(5, cpu_count // 2)
        elif file_size_gb >= 1:
            # Large files: moderate batching
            batch_size = max(10, cpu_count)
        else:
            # Medium files: aggressive batching
            batch_size = cpu_count * 2
            
        return {
            "strategy": "adaptive_batch_chunk_merge",
            "reason": f"Large scale ({total_chunks} chunks), adaptive batch processing",
            "chunks_per_file": chunks_per_file,
            "use_chunk_and_merge": True,
            "batch_size": batch_size,
            "max_parallel_chunks": cpu_count,
            "memory_profile": "high" if file_size_gb >= 2 else "medium",
            "optimization": "memory_conscious"
        }


def enhanced_sampling_for_chunks(
    schema: dict[str, Any], 
    file_size: str, 
    chunk_size: str = "512MB", 
    format: str = "parquet",
    seed: int | None = None
) -> tuple[int, int, float]:
    """
    智能採样系统：生成少量样本数据来准确估算每行的字节数和chunk参数
    
    Args:
        schema: Complete schema definition
        file_size: Target file size
        chunk_size: Chunk size (default 512MB)
        format: Output format
        seed: Random seed
    
    Returns:
        tuple of (rows_per_chunk, rows_per_file, bytes_per_row)
    """
    fields = schema.get("fields", schema)
    
    return calculate_chunk_parameters(
        file_size=file_size,
        chunk_size=chunk_size,
        fields=fields,
        format=format,
        schema=schema,
        seed=seed,
    )


def _is_bm25_output_field(field_name: str, schema: dict[str, Any]) -> bool:
    """Check if a field is a BM25 function output field that should be skipped during generation."""
    functions = schema.get("functions", [])
    for func in functions:
        if func.get("type") == "BM25" and field_name in func.get(
            "output_field_names", []
        ):
            return True
    return False


def _find_partition_key_field(fields: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the partition key field in schema fields."""
    for field in fields:
        if field.get("is_partition_key", False):
            return field
    return None


def _find_primary_key_field(fields: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the primary key field in schema fields."""
    for field in fields:
        if field.get("is_primary", False):
            return field
    return None


def _calculate_shard_id(primary_key_value: Any, num_shards: int) -> int:
    """Calculate shard ID based on primary key hash (simulates Milvus VChannel distribution)."""
    return hash(str(primary_key_value)) % num_shards


def _calculate_partition_id(partition_key_value: Any, num_partitions: int) -> int:
    """Calculate partition ID based on partition key hash."""
    return hash(str(partition_key_value)) % num_partitions


def _parse_file_size(size_str: str) -> int:
    """
    Parse file size string to bytes.

    Args:
        size_str: Size string like '10GB', '200MB', '1TB', or plain number (MB)

    Returns:
        Size in bytes
    """
    if not size_str:
        return 0

    size_str = str(size_str).upper().strip()

    # Handle different units
    if size_str.endswith("TB"):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024 * 1024)
    elif size_str.endswith("GB"):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith("MB"):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("B"):
        return int(float(size_str[:-1]))
    else:
        # If no unit, assume it's MB (for backward compatibility)
        try:
            return int(float(size_str) * 1024 * 1024)
        except ValueError:
            raise ValueError(
                f"Invalid file size format: {size_str}. Use formats like '10GB', '200MB', or '256' (MB)"
            )


def adjust_workers_by_strategy(
    num_workers: int, 
    strategy_info: dict[str, Any], 
    file_size_bytes: int,
    file_count: int
) -> int:
    """
    Adjust number of workers based on generation strategy and constraints.
    
    This replaces the old file-size-only approach with strategy-aware optimization.
    
    Args:
        num_workers: Initial worker count
        strategy_info: Strategy information from determine_generation_strategy()
        file_size_bytes: Estimated file size in bytes
        file_count: Number of files to generate
    
    Returns:
        Optimized worker count based on strategy
    """
    strategy = strategy_info.get("strategy", "unknown")
    memory_profile = strategy_info.get("memory_profile", "medium")
    optimization = strategy_info.get("optimization", "standard")
    
    file_size_gb = file_size_bytes / (1024**3)
    cpu_count = multiprocessing.cpu_count()
    
    # Strategy-based worker adjustment
    if strategy == "direct_parallel":
        # For direct parallel: use strategy recommendation
        recommended_workers = strategy_info.get("max_parallel_files", num_workers)
        adjusted_workers = min(num_workers, recommended_workers)
        
    elif strategy == "batch_direct_parallel":
        # For batch direct: limit workers but allow higher throughput
        max_workers = strategy_info.get("max_parallel_files", cpu_count * 2)
        adjusted_workers = min(num_workers, max_workers)
        
    elif strategy == "chunk_parallel_merge":
        # For chunk parallel: adjust based on memory profile
        if memory_profile == "high":
            # Conservative for large files
            adjusted_workers = min(num_workers, max(1, cpu_count // 2))
        elif memory_profile == "medium":
            # Balanced approach
            adjusted_workers = min(num_workers, cpu_count)
        else:
            # Aggressive for small chunks
            max_chunks = strategy_info.get("max_parallel_chunks", cpu_count * 2)
            adjusted_workers = min(num_workers, max_chunks)
            
    elif strategy == "batch_file_parallel":
        # For batch file processing: use batch-optimized workers
        batch_size = strategy_info.get("batch_size", cpu_count)
        max_workers = strategy_info.get("max_parallel_files", cpu_count)
        adjusted_workers = min(num_workers, max_workers)
        
    elif strategy == "adaptive_batch_chunk_merge":
        # For adaptive processing: memory-conscious approach
        if file_size_gb >= 5:
            adjusted_workers = min(num_workers, max(1, cpu_count // 4))
        elif file_size_gb >= 2:
            adjusted_workers = min(num_workers, max(2, cpu_count // 2))  
        else:
            adjusted_workers = min(num_workers, cpu_count)
    else:
        # Fallback: use original size-based logic
        if file_size_gb >= 5:
            adjusted_workers = 1
        elif file_size_gb >= 2:
            adjusted_workers = min(num_workers, 2)
        else:
            adjusted_workers = num_workers
    
    # Log strategy and adjustment
    if adjusted_workers != num_workers:
        logger.info(
            f"Strategy-based worker adjustment: {num_workers} → {adjusted_workers} "
            f"(strategy: {strategy}, memory: {memory_profile}, file: {file_size_gb:.1f}GB)"
        )
        
    # Additional memory safety warnings
    if memory_profile == "high" and adjusted_workers > 2:
        logger.warning(
            f"High memory profile detected. Using {adjusted_workers} workers for "
            f"{file_count}×{file_size_gb:.1f}GB files. Monitor memory usage."
        )
    elif optimization == "chunk_parallel_optimized":
        logger.info(
            f"Chunk-parallel optimization enabled: {adjusted_workers} workers for "
            f"maximum chunk throughput ({strategy_info.get('chunks_per_file', 1)} chunks/file)"
        )
        
    return max(1, adjusted_workers)  # Always at least 1 worker


def adjust_workers_by_file_size(num_workers: int, file_size_bytes: int) -> int:
    """
    Legacy function: Adjust number of workers based on file size only.
    
    DEPRECATED: Use adjust_workers_by_strategy() for better optimization.
    Kept for backward compatibility.
    """
    file_size_gb = file_size_bytes / (1024**3)
    initial_workers = num_workers
    
    if file_size_gb < 1:
        adjusted_workers = initial_workers
    elif file_size_gb <= 5:
        adjusted_workers = min(initial_workers, 2)
    else:
        adjusted_workers = 1
        
    if adjusted_workers != initial_workers:
        logger.info(
            f"Legacy worker adjustment: {initial_workers} → {adjusted_workers} "
            f"(file size: {file_size_gb:.1f}GB)"
        )
        
    return adjusted_workers


def _generate_partition_key_with_cardinality(
    partition_field: dict[str, Any],
    num_rows: int,
    unique_count: int,
    pk_offset: int = 0,
) -> np.ndarray | list:
    """
    Generate partition key values with specified unique count.

    Args:
        partition_field: Partition key field definition
        num_rows: Number of rows to generate
        unique_count: Number of unique values to generate
        pk_offset: Offset for generation

    Returns:
        Generated partition key values
    """
    field_type = partition_field["type"]

    if field_type == "Int64":
        # Generate unique base values
        base_value = (
            pk_offset + 1
        ) * 100000  # Ensure different offsets produce different ranges
        unique_values = np.arange(base_value, base_value + unique_count, dtype=np.int64)

        # Randomly select from unique values to fill all rows
        partition_keys = np.random.choice(unique_values, size=num_rows)
        return partition_keys

    elif field_type in ["VarChar", "String"]:
        # Generate unique string values
        unique_values = [f"partition_key_{pk_offset}_{i}" for i in range(unique_count)]

        # Randomly select from unique values to fill all rows
        partition_keys = np.random.choice(unique_values, size=num_rows)
        return partition_keys.tolist()

    else:
        # Fallback to regular generation for unsupported types
        return _generate_scalar_field_data(partition_field, num_rows, pk_offset)


def _generate_dynamic_field_data(
    dynamic_field: dict[str, Any], batch_size: int
) -> list[Any]:
    """Generate data for a single dynamic field.

    Args:
        dynamic_field: Dynamic field definition from schema
        batch_size: Number of rows to generate

    Returns:
        List of generated values for the dynamic field
    """
    field_type = dynamic_field.get("type", "String")
    probability = dynamic_field.get("probability", 1.0)
    values = dynamic_field.get("values")

    # Determine which rows should have this field (based on probability)
    should_include = np.random.random(batch_size) < probability
    result: list[Any] = []

    for i in range(batch_size):
        if not should_include[i]:
            # Skip this field for this row (will not be added to the data)
            result.append(None)
            continue

        if values:
            # Choose from predefined values
            value = np.random.choice(values)
            result.append(value)
            continue

        # Generate based on type
        if field_type == "Bool":
            result.append(bool(np.random.randint(0, 2)))
        elif field_type == "Int":
            min_val = dynamic_field.get("min_value", 0)
            max_val = dynamic_field.get("max_value", 1000)
            result.append(int(np.random.randint(min_val, max_val + 1)))
        elif field_type == "Float":
            min_val = dynamic_field.get("min_value", 0.0)
            max_val = dynamic_field.get("max_value", 1.0)
            result.append(float(np.random.uniform(min_val, max_val)))
        elif field_type == "String":
            min_len = dynamic_field.get("min_length", 5)
            max_len = dynamic_field.get("max_length", 20)
            length = np.random.randint(min_len, max_len + 1)
            # Generate random string with letters and numbers
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            value = "".join(np.random.choice(list(chars), size=length))
            result.append(value)
        elif field_type == "Array":
            min_len = dynamic_field.get("array_min_length", 0)
            max_len = dynamic_field.get("array_max_length", 5)
            length = np.random.randint(min_len, max_len + 1)
            # Generate array of strings or integers
            if np.random.random() < 0.5:
                # String array
                array_data = [f"item_{j}" for j in range(length)]
            else:
                # Integer array
                array_data = np.random.randint(0, 100, size=length).tolist()
            result.append(array_data)
        elif field_type == "JSON":
            # Generate simple JSON object
            json_obj = {
                "id": int(np.random.randint(1, 1000)),
                "name": f"dynamic_item_{np.random.randint(1, 1000)}",
                "active": bool(np.random.randint(0, 2)),
                "score": float(np.random.uniform(0, 100)),
            }
            result.append(json_obj)
        else:
            # Default to string
            result.append(f"dynamic_value_{np.random.randint(1, 1000)}")

    return result


def _generate_single_file(file_info: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a single data file in a separate process.

    Args:
        file_info: Dictionary containing:
            - schema: Schema definition
            - file_index: Index of this file
            - total_files: Total number of files
            - current_batch_rows: Number of rows for this file
            - pk_offset: Primary key offset
            - output_dir: Output directory path
            - format: Output format ('parquet' or 'json')
            - seed: Random seed (will be adjusted per process)
            - num_partitions: Number of partitions (optional)
            - num_shards: Number of shards (optional)

    Returns:
        Dictionary with generation results
    """
    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from loguru import logger
    from ml_dtypes import bfloat16

    # Configure logging for worker process to suppress INFO/DEBUG messages during progress display
    # This prevents worker processes from interfering with progress bars
    logger.remove()  # Remove default handler
    logger.add(
        __import__("sys").stderr,
        level="WARNING",  # Only show warnings and errors from worker processes
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        colorize=True,
    )

    try:
        # Extract parameters
        schema = file_info["schema"]
        file_index = file_info["file_index"]
        total_files = file_info["total_files"]
        current_batch_rows = file_info["current_batch_rows"]
        pk_offset = file_info["pk_offset"]
        output_dir = Path(file_info["output_dir"])
        format = file_info["format"]
        seed = file_info["seed"]
        num_partitions = file_info.get("num_partitions")

        # Set process-specific seed to ensure different random data per process
        if seed:
            process_seed = seed + file_index * 12345
            np.random.seed(process_seed)

        fields = schema.get("fields", schema)
        enable_dynamic = schema.get("enable_dynamic_field", False)
        dynamic_fields = schema.get("dynamic_fields", []) if enable_dynamic else []

        generation_start = time.time()

        # Pre-analyze schema for optimization and distribution
        vector_fields = []
        scalar_fields = []
        partition_key_field = None
        primary_key_field = None

        for field in fields:
            # Skip auto_id fields - they should not be generated
            if field.get("auto_id", False):
                continue

            # Skip BM25 function output fields - they are auto-generated by Milvus
            if _is_bm25_output_field(field["name"], schema):
                continue

            # Identify special fields
            if field.get("is_partition_key", False):
                partition_key_field = field
            # Note: primary_key_field is determined but not used in single file generation

            if "Vector" in field["type"]:
                vector_fields.append(field)
            else:
                scalar_fields.append(field)

        # Generate data
        data: dict[str, Any] = {}

        # Generate partition key with enough unique values (if partition key exists)
        if partition_key_field and num_partitions:
            # Generate partition key values with unique count = num_partitions * 10
            partition_unique_count = num_partitions * 10
            partition_values = _generate_partition_key_with_cardinality(
                partition_key_field,
                current_batch_rows,
                partition_unique_count,
                pk_offset,
            )
            data[partition_key_field["name"]] = partition_values

        # Generate remaining scalar fields efficiently
        for field in scalar_fields:
            field_name = field["name"]
            field_type = field["type"]

            # Skip fields already generated (partition key, primary key)
            if field_name in data:
                continue

            if field_type in [
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Float",
                "Double",
                "Bool",
                "VarChar",
                "String",
            ]:
                # Use the new cardinality-aware generation function
                data[field_name] = _generate_scalar_field_data(
                    field, current_batch_rows, pk_offset
                )
            elif field_type == "JSON":
                # Generate diverse JSON data with multiple patterns
                json_data = []

                # Pre-generate random data for efficiency
                random_types = np.random.randint(0, 5, size=current_batch_rows)
                random_ints = np.random.randint(1, 1000, size=current_batch_rows)
                random_floats = np.random.random(current_batch_rows)
                random_bools = np.random.randint(
                    0, 2, size=current_batch_rows, dtype=bool
                )

                # Pre-generate string pools
                categories = [
                    "electronics",
                    "books",
                    "clothing",
                    "food",
                    "toys",
                    "sports",
                    "health",
                    "home",
                ]
                statuses = ["active", "pending", "completed", "cancelled", "processing"]
                tags_pool = [
                    "new",
                    "featured",
                    "sale",
                    "limited",
                    "exclusive",
                    "popular",
                    "trending",
                    "clearance",
                ]

                for i in range(current_batch_rows):
                    json_type = random_types[i]

                    if json_type == 0:
                        # E-commerce product metadata
                        json_obj = {
                            "product_id": int(pk_offset + i),
                            "category": categories[i % len(categories)],
                            "price": round(float(random_floats[i] * 999.99), 2),
                            "in_stock": bool(random_bools[i]),
                            "attributes": {
                                "brand": f"Brand_{random_ints[i] % 50}",
                                "weight": round(float(random_floats[i] * 10), 2),
                                "dimensions": {
                                    "length": int(random_ints[i] % 100),
                                    "width": int((random_ints[i] + 10) % 100),
                                    "height": int((random_ints[i] + 20) % 100),
                                },
                            },
                            "tags": tags_pool[: random_ints[i] % 4 + 1],
                        }
                    elif json_type == 1:
                        # User activity/event data
                        json_obj = {
                            "event_id": int(pk_offset + i),
                            "event_type": [
                                "click",
                                "view",
                                "purchase",
                                "share",
                                "like",
                            ][i % 5],
                            "timestamp": int(1600000000 + random_ints[i] * 1000),
                            "user": {
                                "id": f"user_{random_ints[i] % 1000}",
                                "session": f"session_{random_ints[i] % 100}",
                                "device": ["mobile", "desktop", "tablet"][i % 3],
                            },
                            "metrics": {
                                "duration_ms": int(random_ints[i] * 10),
                                "clicks": int(random_ints[i] % 10),
                                "score": round(float(random_floats[i] * 5), 2),
                            },
                        }
                    elif json_type == 2:
                        # Configuration/settings data
                        json_obj = {
                            "config_id": int(pk_offset + i),
                            "name": f"config_{i}",
                            "settings": {
                                "enabled": bool(random_bools[i]),
                                "threshold": float(random_floats[i]),
                                "max_retries": int(random_ints[i] % 10),
                                "timeout_seconds": int(random_ints[i] % 300),
                                "features": {
                                    "feature_a": bool(random_bools[i]),
                                    "feature_b": bool(not random_bools[i]),
                                    "feature_c": bool(i % 3 == 0),
                                },
                            },
                            "metadata": {
                                "version": f"{random_ints[i] % 3}.{random_ints[i] % 10}.{random_ints[i] % 20}",
                                "last_updated": int(1600000000 + random_ints[i] * 1000),
                            },
                        }
                    elif json_type == 3:
                        # Analytics/metrics data
                        json_obj = {
                            "metric_id": int(pk_offset + i),
                            "type": "analytics",
                            "values": {
                                "count": int(random_ints[i]),
                                "sum": round(float(random_floats[i] * 10000), 2),
                                "avg": round(float(random_floats[i] * 100), 2),
                                "min": round(float(random_floats[i] * 10), 2),
                                "max": round(float(random_floats[i] * 1000), 2),
                            },
                            "dimensions": {
                                "region": ["north", "south", "east", "west"][i % 4],
                                "category": categories[i % len(categories)],
                                "segment": f"segment_{random_ints[i] % 10}",
                            },
                            "percentiles": {
                                "p50": round(float(random_floats[i] * 50), 2),
                                "p90": round(float(random_floats[i] * 90), 2),
                                "p99": round(float(random_floats[i] * 99), 2),
                            },
                        }
                    else:
                        # Document/content metadata
                        json_obj = {
                            "doc_id": int(pk_offset + i),
                            "title": f"Document_{i}",
                            "status": statuses[i % len(statuses)],
                            "metadata": {
                                "author": f"author_{random_ints[i] % 100}",
                                "created_at": int(1600000000 + random_ints[i] * 1000),
                                "word_count": int(random_ints[i] * 10),
                                "language": ["en", "es", "fr", "de", "zh"][i % 5],
                                "sentiment": {
                                    "positive": round(float(random_floats[i]), 3),
                                    "negative": round(float(1 - random_floats[i]), 3),
                                    "neutral": round(float(random_floats[i] * 0.5), 3),
                                },
                            },
                            "tags": tags_pool[: random_ints[i] % 3 + 1],
                            "properties": {
                                "public": bool(random_bools[i]),
                                "archived": bool(not random_bools[i]),
                                "priority": int(random_ints[i] % 5),
                            },
                        }

                    json_data.append(json_obj)

                data[field_name] = json_data

            elif field_type == "Array":
                element_type = field.get("element_type", "Int32")
                max_capacity = field.get("max_capacity", 5)

                # Optimized vectorized array generation
                lengths = np.random.randint(
                    0, max_capacity + 1, size=current_batch_rows
                )

                if element_type in ["Int32", "Int64"]:
                    # Vectorized integer array generation
                    total_elements = np.sum(lengths)
                    if total_elements > 0:
                        # Pre-generate large pool of integers and slice as needed
                        int_pool = np.random.randint(
                            -999, 999, size=total_elements, dtype=np.int32
                        )
                        array_data = []
                        start_idx = 0
                        for length in lengths:
                            if length > 0:
                                array_data.append(
                                    int_pool[start_idx : start_idx + length].tolist()
                                )
                                start_idx += length
                            else:
                                array_data.append([])
                    else:
                        array_data = [[] for _ in range(current_batch_rows)]
                    data[field_name] = array_data
                else:
                    # Optimized string arrays with pre-computed strings
                    str_templates = [f"item_{j}" for j in range(max_capacity)]
                    array_data = []
                    for length in lengths:
                        if length > 0:
                            array_data.append(str_templates[:length])
                        else:
                            array_data.append([])
                    data[field_name] = array_data

        # Generate vector fields efficiently
        for field in vector_fields:
            field_name = field["name"]
            field_type = field["type"]
            dim = field.get("dim", 128)

            if field_type == "FloatVector":
                # Generate normalized float vectors efficiently (uses multiple CPU cores)
                vectors = np.random.randn(current_batch_rows, dim).astype(np.float32)

                # Always use NumPy for vector normalization (consistently faster due to optimized BLAS)
                # JIT compilation overhead typically outweighs benefits for this simple operation
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms

                # Store as list of arrays for pandas
                data[field_name] = list(vectors)

            elif field_type == "BinaryVector":
                # Binary vector: each int represents 8 dimensions
                # If binary vector dimension is 16, use [x, y] where x and y are 0-255
                byte_dim = dim // 8
                binary_vectors = np.random.randint(
                    0, 256, size=(current_batch_rows, byte_dim), dtype=np.uint8
                )
                data[field_name] = list(binary_vectors)

            elif field_type in ["Float16Vector", "BFloat16Vector"]:
                if field_type == "Float16Vector":
                    # Generate float16 vectors using uint8 representation
                    fp16_vectors = []
                    for _ in range(current_batch_rows):
                        raw_vector = np.random.random(dim)
                        fp16_vector = (
                            np.array(raw_vector, dtype=np.float16)
                            .view(np.uint8)
                            .tolist()
                        )
                        fp16_vectors.append(fp16_vector)
                    data[field_name] = fp16_vectors
                else:  # BFloat16Vector
                    # Generate bfloat16 vectors using uint8 representation
                    bf16_vectors = []
                    for _ in range(current_batch_rows):
                        raw_vector = np.random.random(dim)
                        bf16_vector = (
                            np.array(raw_vector, dtype=bfloat16).view(np.uint8).tolist()
                        )
                        bf16_vectors.append(bf16_vector)
                    data[field_name] = bf16_vectors

            elif field_type == "SparseFloatVector":
                # Generate sparse float vectors as dict with indices as keys and values as floats
                sparse_vectors = []
                for _ in range(current_batch_rows):
                    max_dim = 1000
                    non_zero_count = np.random.randint(
                        10, max_dim // 10
                    )  # 10-100 non-zero values
                    indices = np.random.choice(max_dim, non_zero_count, replace=False)
                    values = np.random.random(non_zero_count)
                    sparse_vector = {
                        str(index): float(value)
                        for index, value in zip(indices, values, strict=False)
                    }
                    sparse_vectors.append(sparse_vector)
                data[field_name] = sparse_vectors

        # Generate dynamic fields if enabled
        if enable_dynamic and dynamic_fields:
            meta_data = []

            # Generate $meta field containing all dynamic fields
            for _ in range(current_batch_rows):
                row_meta = {}

                for dynamic_field in dynamic_fields:
                    field_name = dynamic_field.get("name")
                    field_values = _generate_dynamic_field_data(
                        dynamic_field, 1
                    )  # Generate one value at a time

                    # Only add the field if it has a non-None value
                    if field_values[0] is not None:
                        row_meta[field_name] = field_values[0]

                # Add the meta object (can be empty if no dynamic fields were generated for this row)
                meta_data.append(row_meta if row_meta else {})

            # Add $meta field to main data
            if meta_data:
                data["$meta"] = meta_data

        batch_generation_time = time.time() - generation_start

        # Create DataFrame
        df = pd.DataFrame(data)

        # Apply Parquet column optimization (only for Parquet format)
        optimization_info = {}
        if format.lower() == "parquet":
            df, optimization_info = _optimize_parquet_columns(df, fields, format)

        # Convert JSON fields to strings for Parquet storage
        if format.lower() == "parquet":
            # Handle regular JSON fields
            for field in fields:
                if field["type"] == "JSON":
                    field_name = field["name"]
                    if field_name in df.columns:
                        # Convert JSON objects to JSON strings for Parquet storage
                        df[field_name] = df[field_name].apply(
                            lambda x: json.dumps(x, ensure_ascii=False)
                            if x is not None
                            else None
                        )

            # Handle $meta field (contains all dynamic fields as JSON)
            if "$meta" in df.columns:
                df["$meta"] = df["$meta"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if x is not None and x != {}
                    else None
                )

        # Write file
        write_start = time.time()

        if format.lower() == "parquet":
            if total_files == 1:
                output_file = output_dir / "data.parquet"
            else:
                file_num = file_index + 1
                output_file = (
                    output_dir / f"data-{file_num:05d}-of-{total_files:05d}.parquet"
                )

            # Convert to PyArrow table for efficient writing
            table = pa.Table.from_pandas(df)

            # Write with optimized settings
            pq.write_table(
                table,
                output_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=False,
                row_group_size=min(50000, current_batch_rows),
            )

        elif format.lower() == "json":
            if total_files == 1:
                output_file = output_dir / "data.json"
            else:
                file_num = file_index + 1
                output_file = (
                    output_dir / f"data-{file_num:05d}-of-{total_files:05d}.json"
                )

            # Write JSON in Milvus bulk import format (list of dict)
            rows_data = []
            for record in df.to_dict(orient="records"):
                # Handle numpy types and field omission for nullable/default fields
                row_record = {}
                for key, value in record.items():
                    # Find field definition for this key
                    field_def = None
                    for field in fields:
                        if field["name"] == key:
                            field_def = field
                            break

                    # Handle value conversion
                    if isinstance(value, np.ndarray):
                        converted_value = value.tolist()
                    elif isinstance(value, np.generic):
                        converted_value = value.item()
                    else:
                        converted_value = value

                    # Handle NaN values for JSON compatibility
                    import math

                    if isinstance(converted_value, float) and math.isnan(
                        converted_value
                    ):
                        converted_value = (
                            None  # Convert NaN to None for JSON serialization
                        )

                    # Decide how to handle this field in JSON
                    if field_def:
                        import random

                        # For nullable fields: randomly omit with 40% probability (Milvus will handle as null)
                        if field_def.get("nullable", False):
                            should_omit = random.random() < 0.4
                            if not should_omit:
                                row_record[key] = converted_value

                        # For default_value fields: randomly omit with 30% probability (Milvus will use default)
                        elif field_def.get("default_value") is not None:
                            should_omit = random.random() < 0.3
                            if not should_omit:
                                row_record[key] = converted_value
                        else:
                            # Regular field - always include
                            row_record[key] = converted_value
                    else:
                        # Field definition not found - include as-is
                        row_record[key] = converted_value

                rows_data.append(row_record)

            # Write as JSON array for Milvus bulk import compatibility (list of dict)
            with open(output_file, "w") as f:
                json.dump(rows_data, f, ensure_ascii=False, separators=(",", ":"))

        else:
            raise ValueError(
                f"Unsupported format: {format}. High-performance mode only supports 'parquet' and 'json' formats."
            )

        batch_write_time = time.time() - write_start
        total_time = batch_generation_time + batch_write_time

        # Return results
        return {
            "success": True,
            "file_path": str(output_file),
            "file_index": file_index,
            "rows": current_batch_rows,
            "generation_time": batch_generation_time,
            "write_time": batch_write_time,
            "total_time": total_time,
            "rows_per_second": current_batch_rows / batch_generation_time
            if batch_generation_time > 0
            else 0,
            "optimization_info": optimization_info
            if optimization_info and "_summary" in optimization_info
            else None,
        }

    except Exception as e:
        # Return error information instead of raising
        return {
            "success": False,
            "error": str(e),
            "file_index": file_info.get("file_index", -1),
        }


def _generate_files_parallel(
    schema: dict[str, Any],
    fields: list[dict[str, Any]],
    enable_dynamic: bool,
    dynamic_fields: list[dict[str, Any]],
    total_files: int,
    effective_max_rows_per_file: int,
    rows: int,
    output_dir: Path,
    format: str,
    seed: int | None,
    num_partitions: int | None,
    num_shards: int | None,
    partition_key_field: dict[str, Any] | None,
    primary_key_field: dict[str, Any] | None,
    vector_fields: list[dict[str, Any]],
    scalar_fields: list[dict[str, Any]],
    num_workers: int,
    progress_callback: Any,
    start_time: float,
    estimation_stats: dict[str, Any],
    file_size: str | None = None,
    skip_validation: bool = False,
) -> tuple[list[str], int]:
    """
    Generate multiple files in parallel using ProcessPoolExecutor.
    
    Returns:
        tuple[list[str], int]: A tuple containing (list of created file paths, actual total rows generated)
    """
    import threading

    # Prepare file generation tasks
    file_tasks = []
    remaining_rows = rows
    pk_offset = 0

    for file_index in range(total_files):
        current_batch_rows = min(remaining_rows, effective_max_rows_per_file)
        if current_batch_rows <= 0:
            break

        file_info = {
            "schema": schema,
            "file_index": file_index,
            "total_files": total_files,
            "current_batch_rows": current_batch_rows,
            "pk_offset": pk_offset,
            "output_dir": str(output_dir),
            "format": format,
            "seed": seed,
            "num_partitions": num_partitions,
            "num_shards": num_shards,
        }

        file_tasks.append(file_info)
        remaining_rows -= current_batch_rows
        pk_offset += current_batch_rows

    logger.info(f"Prepared {len(file_tasks)} file generation tasks")

    # Progress tracking
    completed_files = 0
    completed_rows = 0
    completed_files_lock = threading.Lock()
    all_files_created = []
    all_files_info = []  # Store detailed file information (path, rows, etc.)
    all_optimization_info = []
    total_generation_time = 0.0
    total_write_time = 0.0
    error_count = 0

    def update_progress_thread_safe(result: dict[str, Any]) -> None:
        """Thread-safe progress update callback."""
        nonlocal \
            completed_files, \
            completed_rows, \
            total_generation_time, \
            total_write_time, \
            error_count

        with completed_files_lock:
            if result.get("success", False):
                completed_files += 1
                completed_rows += result.get("rows", 0)
                total_generation_time += result.get("generation_time", 0)
                total_write_time += result.get("write_time", 0)
                file_path = result.get("file_path", "")
                all_files_created.append(file_path)
                
                # Store detailed file info
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                all_files_info.append({
                    "file_name": Path(file_path).name,
                    "file_path": file_path,
                    "rows": result.get("rows", 0),
                    "file_index": result.get("file_index", -1),
                    "file_size_bytes": file_size,
                })

                # Collect optimization info if available
                if result.get("optimization_info"):
                    all_optimization_info.append(
                        {
                            "file_index": result.get("file_index", -1),
                            "file_name": Path(result.get("file_path", "")).name,
                            "rows": result.get("rows", 0),
                            "optimization": result.get("optimization_info"),
                        }
                    )

                # Update progress callback if provided
                if progress_callback:
                    progress_callback(completed_rows)

                # Log progress for debugging
                logger.debug(
                    f"File {result.get('file_index', -1) + 1}/{total_files} completed: "
                    f"{result.get('rows', 0):,} rows "
                    f"({result.get('rows_per_second', 0):.0f} rows/sec)"
                )
            else:
                error_count += 1
                logger.error(
                    f"File {result.get('file_index', -1) + 1} failed: {result.get('error', 'Unknown error')}"
                )

    # Execute parallel file generation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        logger.info(f"Starting parallel generation with {num_workers} workers...")

        # Submit all tasks
        future_to_task = {
            executor.submit(_generate_single_file, task): task for task in file_tasks
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                update_progress_thread_safe(result)
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e),
                    "file_index": task.get("file_index", -1),
                }
                update_progress_thread_safe(error_result)
                logger.error(f"Task execution failed: {e}")

    # Final progress update
    if progress_callback:
        progress_callback(rows)

    logger.info(
        f"Parallel generation completed: {completed_files}/{total_files} files successful, {error_count} errors"
    )

    if error_count > 0:
        raise RuntimeError(
            f"Parallel generation failed: {error_count} out of {total_files} files failed"
        )

    # Calculate final statistics
    total_time = time.time() - start_time

    # Create metadata file
    meta_file = output_dir / "meta.json"

    # Calculate actual file size limit in MB from user input or default
    if file_size:
        actual_file_size_mb = _parse_file_size(file_size) / (1024 * 1024)
    else:
        actual_file_size_mb = 256  # Default 256MB when no file size specified

    metadata = {
        "schema": schema,
        "generation_info": {
            "total_rows": rows,
            "format": format,
            "seed": seed,
            "data_files": all_files_info,
            "file_count": len(all_files_created),
            "max_rows_per_file": effective_max_rows_per_file,
            "max_file_size_mb": actual_file_size_mb,
            "generation_time": total_generation_time,
            "write_time": total_write_time,
            "total_time": total_time,
            "rows_per_second": rows / total_time if total_time > 0 else 0,
            "parallel_workers_used": num_workers,
            "size_estimation": estimation_stats,
        },
    }

    # Add collection configuration for Milvus create_collection
    collection_config = {}

    # Add partition configuration if specified
    if num_partitions and partition_key_field:
        collection_config["num_partitions"] = num_partitions
        collection_config["partition_key_field"] = partition_key_field["name"]

    # Add shard configuration if specified
    if num_shards:
        collection_config["num_shards"] = num_shards

    # Add collection config to metadata if any settings were specified
    if collection_config:
        metadata["collection_config"] = collection_config

    # Add optimization information if any occurred
    if all_optimization_info:
        # Calculate total optimization statistics
        total_optimized_columns = sum(
            info["optimization"].get("_summary", {}).get("columns_optimized", 0)
            for info in all_optimization_info
        )
        total_savings_mb = sum(
            info["optimization"].get("_summary", {}).get("total_memory_savings_mb", 0.0)
            for info in all_optimization_info
        )

        metadata["generation_info"]["parquet_optimization"] = {
            "enabled": True,
            "files_optimized": len(all_optimization_info),
            "total_files": len(all_files_created),
            "total_columns_optimized": total_optimized_columns,
            "total_savings_mb": round(total_savings_mb, 2),
            "optimization_details": all_optimization_info,
        }
    elif format.lower() == "parquet":
        metadata["generation_info"]["parquet_optimization"] = {
            "enabled": True,
            "files_optimized": 0,
            "total_files": len(all_files_created),
            "note": "No uniform columns detected for optimization",
        }

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Convert metadata to handle numpy types
    serializable_metadata = convert_numpy_types(metadata)

    with open(meta_file, "w") as f:
        json.dump(serializable_metadata, f, indent=2)

    logger.info(
        f"✅ Parallel generation completed: {len(all_files_created)} files, {rows:,} rows, {total_time:.2f}s ({rows / total_time:.0f} rows/sec)"
    )

    # Run minimal validation unless skipped (e.g., for temporary chunks)
    if not skip_validation:
        print(f"🎯 Generation completed, proceeding to validation...")
        print("📊 Starting minimal file validation...")
        from .minimal_validator import MinimalValidator
        from rich.console import Console

        try:
            # Create validator
            console = Console()
            validator = MinimalValidator(output_dir, console)

            # Run validation
            print("🔍 Running minimal validation (schema + count)...")
            logger.info("Running minimal file validation...")
            validation_results = validator.validate()

            # Display results
            validator.display_results(validation_results)

            # Log validation status with higher visibility
            if validation_results["valid"]:
                print(
                    f"✅ File validation passed: {validation_results['summary']['total_files']} files, {validation_results['summary']['total_rows']:,} rows"
                )
                logger.info("✅ File validation passed")
            else:
                print(
                    f"⚠️ File validation failed with {len(validation_results['errors'])} errors"
                )
                logger.warning(
                    f"⚠️ File validation failed with {len(validation_results['errors'])} errors"
                )
                # Don't fail the generation, just warn
                for error in validation_results["errors"]:
                    print(f"  - {error}")
                    logger.warning(f"  - {error}")

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            logger.warning(f"Failed to run validation: {e}")
            # Don't fail the generation if validation fails

    return all_files_created, rows


def _estimate_row_size(fields: list[dict[str, Any]]) -> int:
    """
    Estimate the average size in bytes of a single row based on schema.

    This is used to determine how many rows can fit within the file size limit.
    Estimates are conservative (slightly higher) to avoid exceeding size limits.
    """
    total_size = 0

    for field in fields:
        field_type = field["type"]

        if field_type == "Int8":
            total_size += 1
        elif field_type == "Int16":
            total_size += 2
        elif field_type in ["Int32", "Float"]:
            total_size += 4
        elif field_type in ["Int64", "Double"]:
            total_size += 8
        elif field_type == "Bool":
            total_size += 1
        elif field_type in ["VarChar", "String"]:
            # Estimate based on max_length or use conservative default
            max_length = field.get("max_length", 100)
            total_size += max_length * 2  # UTF-8 can be up to 2 bytes per char
        elif field_type == "JSON":
            # Conservative estimate for JSON objects (can vary widely)
            total_size += 200  # Average JSON object size
        elif field_type == "Array":
            element_type = field.get("element_type", "Int32")
            max_capacity = field.get("max_capacity", 5)

            if element_type in ["Int8"]:
                element_size = 1
            elif element_type in ["Int16"]:
                element_size = 2
            elif element_type in ["Int32", "Float"]:
                element_size = 4
            elif element_type in ["Int64", "Double"]:
                element_size = 8
            elif element_type in ["VarChar", "String"]:
                max_length = field.get("max_length", 50)
                element_size = max_length * 2
            else:
                element_size = 4  # Default

            # Average array size (assume half capacity on average)
            total_size += (
                max_capacity // 2
            ) * element_size + 8  # +8 for array overhead
        elif field_type == "FloatVector":
            dim = field.get("dim", 128)
            total_size += dim * 4  # 4 bytes per float32
        elif field_type == "BinaryVector":
            dim = field.get("dim", 128)
            total_size += dim // 8  # 1 bit per dimension, 8 bits per byte
        elif field_type in ["Float16Vector", "BFloat16Vector"]:
            dim = field.get("dim", 128)
            total_size += dim * 2  # 2 bytes per 16-bit float
        elif field_type == "SparseFloatVector":
            # Sparse vectors vary widely, estimate ~50 non-zero entries
            # Each entry is index (int) + value (float) = 8 bytes
            total_size += 50 * 8  # index + value pairs
        else:
            # Unknown type, use conservative estimate
            total_size += 8

    # Add overhead for serialization format (Parquet/JSON metadata, etc.)
    overhead = max(50, total_size // 10)  # At least 50 bytes or 10% overhead

    return total_size + overhead


def _estimate_row_size_from_sample(
    fields: list[dict[str, Any]],
    sample_size: int,
    pk_offset: int,
    format: str,
    schema: dict[str, Any],
    seed: int | None = None,
) -> float:
    """
    Generate a small sample of real data and measure its actual size.
    This provides much more accurate estimation than theoretical calculations.
    """
    import tempfile

    # Set seed for reproducible sampling
    if seed:
        np.random.seed(seed)

    # Generate sample data using the same logic as the main generation
    data: dict[str, Any] = {}

    # Generate scalar fields
    for field in fields:
        # Skip auto_id fields
        if field.get("auto_id", False):
            continue
        # Skip BM25 function output fields - they are auto-generated by Milvus
        if _is_bm25_output_field(field["name"], schema):
            continue
        if "Vector" in field["type"]:
            continue  # Skip vectors for now, handle separately

        field_name = field["name"]
        field_type = field["type"]

        if field_type in [
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Float",
            "Double",
            "Bool",
            "VarChar",
            "String",
        ]:
            # Use the new cardinality-aware generation function
            data[field_name] = _generate_scalar_field_data(
                field, sample_size, pk_offset
            )
        elif field_type == "JSON":
            # Use simplified JSON for sampling (faster generation)
            ids = np.arange(pk_offset, pk_offset + sample_size)
            values = np.random.random(sample_size)
            data[field_name] = [
                {
                    "id": int(ids[i]),
                    "value": float(values[i]),
                    "category": f"cat_{i % 10}",
                }
                for i in range(sample_size)
            ]
        elif field_type == "Array":
            element_type = field.get("element_type", "Int32")
            max_capacity = field.get("max_capacity", 5)
            lengths = np.random.randint(0, max_capacity + 1, size=sample_size)

            if element_type in ["Int32", "Int64"]:
                total_elements = np.sum(lengths)
                if total_elements > 0:
                    int_pool = np.random.randint(
                        -999, 999, size=total_elements, dtype=np.int32
                    )
                    array_data = []
                    start_idx = 0
                    for length in lengths:
                        if length > 0:
                            array_data.append(
                                int_pool[start_idx : start_idx + length].tolist()
                            )
                            start_idx += length
                        else:
                            array_data.append([])
                else:
                    array_data = [[] for _ in range(sample_size)]
                data[field_name] = array_data
            else:
                str_templates = [f"item_{j}" for j in range(max_capacity)]
                array_data = []
                for length in lengths:
                    if length > 0:
                        array_data.append(str_templates[:length])
                    else:
                        array_data.append([])
                data[field_name] = array_data

    # Generate vector fields
    for field in fields:
        # Skip auto_id fields
        if field.get("auto_id", False):
            continue
        # Skip BM25 function output fields - they are auto-generated by Milvus
        if _is_bm25_output_field(field["name"], schema):
            continue
        if "Vector" not in field["type"]:
            continue

        field_name = field["name"]
        field_type = field["type"]
        dim = field.get("dim", 128)

        if field_type == "FloatVector":
            vectors = np.random.randn(sample_size, dim).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            data[field_name] = list(vectors)
        elif field_type == "BinaryVector":
            # Binary vector: each int represents 8 dimensions
            # If binary vector dimension is 16, use [x, y] where x and y are 0-255
            byte_dim = dim // 8
            binary_vectors = np.random.randint(
                0, 256, size=(sample_size, byte_dim), dtype=np.uint8
            )
            data[field_name] = list(binary_vectors)
        elif field_type in ["Float16Vector", "BFloat16Vector"]:
            if field_type == "Float16Vector":
                # Generate float16 vectors using uint8 representation
                fp16_vectors = []
                for _ in range(sample_size):
                    raw_vector = np.random.random(dim)
                    fp16_vector = (
                        np.array(raw_vector, dtype=np.float16).view(np.uint8).tolist()
                    )
                    fp16_vectors.append(fp16_vector)
                data[field_name] = fp16_vectors
            else:  # BFloat16Vector
                # Generate bfloat16 vectors using uint8 representation
                bf16_vectors = []
                for _ in range(sample_size):
                    raw_vector = np.random.random(dim)
                    bf16_vector = (
                        np.array(raw_vector, dtype=bfloat16).view(np.uint8).tolist()
                    )
                    bf16_vectors.append(bf16_vector)
                data[field_name] = bf16_vectors

        elif field_type == "SparseFloatVector":
            # Generate sparse float vectors as dict with indices as keys and values as floats
            sparse_vectors = []
            for _ in range(sample_size):
                max_dim = 1000
                non_zero_count = np.random.randint(
                    10, max_dim // 10
                )  # 10-100 non-zero values
                indices = np.random.choice(max_dim, non_zero_count, replace=False)
                values = np.random.random(non_zero_count)
                sparse_vector = {
                    str(index): float(value)
                    for index, value in zip(indices, values, strict=False)
                }
                sparse_vectors.append(sparse_vector)
            data[field_name] = sparse_vectors

    # Create DataFrame and measure its size
    df = pd.DataFrame(data)

    # Write to temporary file and measure size
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        if format.lower() == "parquet":
            # Use same settings as main generation
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                tmp_file.name,
                compression="snappy",
                use_dictionary=True,
                write_statistics=False,
                row_group_size=min(50000, sample_size),
            )
        else:  # JSON
            import json

            with open(tmp_file.name, "w") as f:
                for i, record in enumerate(df.to_dict(orient="records")):
                    # Apply the same field omission logic as the main JSON generation
                    filtered_record = {}
                    for key, value in record.items():
                        # Find field definition for this key
                        field_def = None
                        for field in fields:
                            if field["name"] == key:
                                field_def = field
                                break

                        # Handle numpy types
                        if isinstance(value, np.ndarray):
                            converted_value = value.tolist()
                        elif isinstance(value, np.generic):
                            converted_value = value.item()
                        else:
                            converted_value = value

                        # Decide whether to include this field in JSON
                        should_omit = False
                        if field_def:
                            import random

                            # For nullable fields: randomly omit with 40% probability
                            if field_def.get("nullable", False):
                                should_omit = random.random() < 0.4

                            # For default_value fields: randomly omit with 30% probability
                            elif field_def.get("default_value") is not None:
                                should_omit = random.random() < 0.3

                        # Only add field to record if we shouldn't omit it
                        if not should_omit:
                            filtered_record[key] = converted_value

                    if i > 0:
                        f.write("\n")
                    json.dump(
                        filtered_record, f, ensure_ascii=False, separators=(",", ":")
                    )

        # Get file size
        file_size = Path(tmp_file.name).stat().st_size

    # Calculate bytes per row
    bytes_per_row = file_size / sample_size

    logger.debug(
        f"Sample file size: {file_size:,} bytes for {sample_size:,} rows = {bytes_per_row:.1f} bytes/row"
    )

    return bytes_per_row


def _enhanced_estimate_row_size_from_sample(
    fields: list[dict[str, Any]],
    sample_size: int,
    pk_offset: int,
    format: str,
    schema: dict[str, Any],
    seed: int | None = None,
    num_iterations: int = 3,  # 多次采样
    base_sample_size: int = 10000,  # 增加基础采样量到10000
) -> tuple[float, dict[str, Any]]:
    """
    Enhanced version of row size estimation with multiple sampling and statistics.

    Improvements:
    1. Multiple sampling iterations for better accuracy
    2. Increased base sample size to 10000 (from 1000)
    3. Statistical analysis with confidence metrics
    4. Adaptive adjustment based on variance
    5. Detailed estimation statistics

    Args:
        fields: Field definitions
        sample_size: Target sample size (will be adapted)
        pk_offset: Primary key offset
        format: Output format
        schema: Schema definition
        seed: Random seed
        num_iterations: Number of sampling iterations
        base_sample_size: Base sample size (default 10000)

    Returns:
        (adjusted_bytes_per_row, estimation_stats)
    """
    from loguru import logger

    # Use larger sample size, but respect memory limits
    adaptive_sample_size = min(max(base_sample_size, sample_size), 50000)

    logger.info(
        f"🔍 Enhanced estimation: {num_iterations} iterations × {adaptive_sample_size:,} rows each"
    )

    sample_results = []

    # Multiple sampling iterations
    for i in range(num_iterations):
        # Use different seed for each iteration to avoid identical samples
        iteration_seed = (seed + i * 1000) if seed else None

        # Call the original estimation function for each sample
        bytes_per_row = _estimate_row_size_from_sample(
            fields,
            adaptive_sample_size,
            pk_offset + i * adaptive_sample_size,
            format,
            schema,
            iteration_seed,
        )

        sample_results.append(bytes_per_row)
        logger.debug(f"   📊 Sample {i + 1}: {bytes_per_row:.2f} bytes/row")

    # Calculate statistical metrics
    mean_size = np.mean(sample_results)
    std_size = np.std(sample_results)
    min_size = np.min(sample_results)
    max_size = np.max(sample_results)

    # Calculate coefficient of variation
    cv = std_size / mean_size if mean_size > 0 else 0

    # Intelligent adjustment based on variance
    if cv < 0.03:  # Very low variance (< 3%)
        adjusted_size = mean_size
        confidence = "Very High"
        adjustment_reason = "Low variance, using mean"
        adjustment_factor = 1.0
    elif cv < 0.08:  # Low variance (< 8%)
        adjusted_size = mean_size * 1.02  # 2% safety margin
        confidence = "High"
        adjustment_reason = "Low variance, small safety margin"
        adjustment_factor = 1.02
    elif cv < 0.15:  # Medium variance (< 15%)
        adjusted_size = mean_size * 1.05  # 5% safety margin
        confidence = "Medium"
        adjustment_reason = "Medium variance, moderate safety margin"
        adjustment_factor = 1.05
    else:  # High variance (>= 15%)
        adjusted_size = max_size * 1.08  # Use max + 8% safety margin
        confidence = "Lower"
        adjustment_reason = "High variance, using max + safety margin"
        adjustment_factor = max_size / mean_size * 1.08

    # Create detailed statistics
    estimation_stats = {
        "method": "enhanced_multi_sample",
        "num_iterations": num_iterations,
        "sample_size_per_iteration": adaptive_sample_size,
        "total_samples": num_iterations * adaptive_sample_size,
        "samples": sample_results,
        "mean_bytes_per_row": float(mean_size),
        "std_bytes_per_row": float(std_size),
        "min_bytes_per_row": float(min_size),
        "max_bytes_per_row": float(max_size),
        "coefficient_of_variation": float(cv),
        "confidence_level": confidence,
        "adjusted_bytes_per_row": float(adjusted_size),
        "adjustment_factor": float(adjustment_factor),
        "adjustment_reason": adjustment_reason,
    }

    logger.info("📈 Estimation results:")
    logger.info(f"   Mean: {mean_size:.2f} bytes/row (±{std_size:.2f})")
    logger.info(f"   Range: {min_size:.2f} - {max_size:.2f}")
    logger.info(f"   CV: {cv:.1%}, Confidence: {confidence}")
    logger.info(f"   Adjusted: {adjusted_size:.2f} bytes/row ({adjustment_reason})")

    return adjusted_size, estimation_stats


def _detect_uniform_column(
    column: pd.Series, field: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Detect if a column has uniform values that can be optimized.

    Args:
        column: pandas Series to analyze
        field: Field definition for context

    Returns:
        Optimization info if column can be optimized, None otherwise
    """
    default_value = field.get("default_value")
    is_nullable = field.get("nullable", False)

    # Case 1: All values are null (for nullable fields)
    if is_nullable and column.isna().all():
        return {
            "type": "all_null",
            "value": None,
            "reason": "naturally_all_null",
            "recovery_strategy": "skip_field_use_null",
        }

    # Case 2: All non-null values are the same as default_value
    if default_value is not None:
        non_null_mask = column.notna()
        if non_null_mask.any():  # Has some non-null values
            non_null_values = column[non_null_mask]
            if (
                len(non_null_values.unique()) == 1
                and non_null_values.iloc[0] == default_value
            ):
                # All non-null values are default values
                if non_null_mask.all():
                    # All values are non-null defaults
                    return {
                        "type": "all_default",
                        "value": default_value,
                        "reason": "naturally_all_default",
                        "recovery_strategy": "skip_field_use_default",
                    }
                else:
                    # Mix of nulls and defaults - can optimize if nullable
                    if is_nullable:
                        return {
                            "type": "null_and_default_only",
                            "value": default_value,
                            "null_count": column.isna().sum(),
                            "default_count": non_null_mask.sum(),
                            "reason": "naturally_mixed_null_default",
                            "recovery_strategy": "skip_field_use_mixed",
                        }

    # Case 3: All values are null and we have a default value (nullable + default)
    if default_value is not None and is_nullable and column.isna().all():
        return {
            "type": "all_null_with_default",
            "value": default_value,
            "reason": "naturally_all_null_but_has_default",
            "recovery_strategy": "skip_field_use_default",
        }

    # Case 4: All values are the same (even without default_value)
    # Skip uniform detection for columns containing unhashable types (lists, dicts, etc.)
    try:
        unique_vals = column.dropna().unique()
        if len(unique_vals) == 1 and column.notna().all():
            return {
                "type": "uniform_value",
                "value": unique_vals[0],
                "reason": "naturally_uniform",
                "recovery_strategy": "skip_field_recreate_uniform",
            }
    except TypeError:
        # Column contains unhashable types (e.g., lists, dicts), skip uniform detection
        pass

    return None


def _optimize_parquet_columns(
    df: pd.DataFrame, schema_fields: list[dict[str, Any]], format_type: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Optimize Parquet DataFrame by detecting and removing columns with uniform values.
    This naturally occurs when data generation results in all same values for a column.

    Args:
        df: DataFrame to optimize
        schema_fields: Original schema field definitions
        format_type: Output format ('parquet' or 'json')

    Returns:
        Tuple of (optimized_df, optimization_info)
    """
    if format_type.lower() != "parquet":
        # Only optimize Parquet format
        return df, {}

    optimization_info = {}
    columns_to_drop = []
    original_size = df.memory_usage(deep=True).sum()

    # Create field lookup by name
    field_by_name = {field["name"]: field for field in schema_fields}

    for column_name in df.columns:
        if column_name not in field_by_name:
            continue

        field = field_by_name[column_name]
        column = df[column_name]

        # Skip optimization for certain field types
        if field.get("is_primary", False) or field["type"] in [
            "FloatVector",
            "BinaryVector",
            "Float16Vector",
            "BFloat16Vector",
            "SparseFloatVector",
        ]:
            continue

        # Detect if this column can be optimized
        uniform_info = _detect_uniform_column(column, field)
        if uniform_info:
            optimization_info[column_name] = uniform_info
            optimization_info[column_name]["memory_savings_bytes"] = (
                column.memory_usage(deep=True)
            )
            columns_to_drop.append(column_name)

    # Remove optimized columns
    optimized_df = df.drop(columns=columns_to_drop) if columns_to_drop else df

    # Calculate space savings
    if columns_to_drop:
        optimized_size = optimized_df.memory_usage(deep=True).sum()
        total_savings = original_size - optimized_size

        optimization_info["_summary"] = {
            "columns_optimized": len(columns_to_drop),
            "optimized_columns": list(columns_to_drop),
            "total_memory_savings_bytes": total_savings,
            "total_memory_savings_mb": round(total_savings / 1024 / 1024, 2),
            "original_columns": len(df.columns),
            "remaining_columns": len(optimized_df.columns),
            "optimization_ratio": round(
                (len(columns_to_drop) / len(df.columns)) * 100, 1
            )
            if df.columns.size > 0
            else 0,
        }

        logger.info(
            f"Natural optimization: detected {len(columns_to_drop)} uniform columns ({', '.join(columns_to_drop)}), saved {optimization_info['_summary']['total_memory_savings_mb']:.2f}MB"
        )

    return optimized_df, optimization_info


def _generate_scalar_field_data(
    field: dict[str, Any], num_rows: int, pk_offset: int = 0
) -> np.ndarray | list:
    """
    Generate data for a scalar field with support for cardinality, constraints, and default values.

    Args:
        field: Field definition with type, constraints, cardinality, and default_value
        num_rows: Number of rows to generate
        pk_offset: Offset for primary key generation

    Returns:
        Generated data as numpy array or list
    """
    field_type = field["type"]

    # Handle primary key generation - ensure global uniqueness
    if field.get("is_primary", False) and not field.get("auto_id", False):
        if field_type == "Int64":
            # Generate consecutive unique primary keys starting from pk_offset + 1
            return np.arange(pk_offset + 1, pk_offset + num_rows + 1, dtype=np.int64)
        elif field_type == "VarChar":
            # Generate unique VarChar primary keys using UUID
            import uuid

            return [str(uuid.uuid4()) for _ in range(num_rows)]

    # Check if field is nullable (null probability from legacy generator)
    is_nullable = field.get("nullable", False)
    null_probability = 0.1  # 10% chance of null values, matching legacy generator

    # Check if field has a default value
    default_value = field.get("default_value")

    # Handle fields with nullable or default_value
    if is_nullable or default_value is not None:
        # Small probability to force entire column to be uniform (for Parquet optimization testing)
        uniform_column_probability = 0.05  # 5% chance
        force_uniform = np.random.random() < uniform_column_probability

        if force_uniform:
            # Decide what uniform value to use
            if is_nullable and (default_value is None or np.random.random() < 0.5):
                # Force all null (for nullable fields)
                result = [None] * num_rows
                logger.debug(
                    f"Forced field '{field['name']}' to be all null for Parquet optimization testing"
                )
            elif default_value is not None:
                # Force all default values
                result = [default_value] * num_rows
                logger.debug(
                    f"Forced field '{field['name']}' to be all default value ({default_value}) for Parquet optimization testing"
                )
            else:
                # Force all same generated value
                uniform_value = _generate_regular_scalar_data(field, 1)[0]
                result = [uniform_value] * num_rows
                logger.debug(
                    f"Forced field '{field['name']}' to be uniform value ({uniform_value}) for Parquet optimization testing"
                )
        else:
            # Normal mixed generation
            # Handle nullable fields - can generate null values
            if is_nullable:
                # For nullable fields, some values can be null
                should_be_null = np.random.random(num_rows) < null_probability
            else:
                should_be_null = np.zeros(num_rows, dtype=bool)

            # Handle default_value fields - can use default values
            if default_value is not None:
                # For fields with defaults, some values can use the default
                should_use_default = np.random.random(num_rows) < 0.3  # 30% use default
                # But null takes precedence over default
                should_use_default = should_use_default & (~should_be_null)
            else:
                should_use_default = np.zeros(num_rows, dtype=bool)

            # Generate regular values for remaining rows
            need_regular_values = (~should_be_null) & (~should_use_default)
            regular_count = np.sum(need_regular_values)

            if regular_count > 0:
                regular_data = _generate_regular_scalar_data(field, regular_count)
            else:
                regular_data = []

            # Build result array
            result = []
            regular_index = 0

            for i in range(num_rows):
                if should_be_null[i]:
                    result.append(None)
                elif should_use_default[i]:
                    result.append(default_value)
                else:
                    result.append(regular_data[regular_index])
                    regular_index += 1

        # Convert to appropriate type based on field type
        # Handle None values properly for numpy arrays
        if field_type in ["Int8", "Int16", "Int32", "Int64"]:
            if any(x is None for x in result):
                # Use nullable integer type for pandas/numpy compatibility
                return result  # Return as list to preserve None values
            else:
                dtype_map = {
                    "Int8": np.int8,
                    "Int16": np.int16,
                    "Int32": np.int32,
                    "Int64": np.int64,
                }
                return np.array(result, dtype=dtype_map[field_type])
        elif field_type in ["Float", "Double"]:
            if any(x is None for x in result):
                # Convert None to NaN for float types
                result_with_nan = [np.nan if x is None else x for x in result]
                dtype = np.float32 if field_type == "Float" else np.float64
                return np.array(result_with_nan, dtype=dtype)
            else:
                dtype = np.float32 if field_type == "Float" else np.float64
                return np.array(result, dtype=dtype)
        elif field_type == "Bool":
            if any(x is None for x in result):
                # Return as list to preserve None values for boolean fields
                return result
            else:
                return np.array(result, dtype=bool)
        else:  # String types
            return result

    # No default value - use regular generation logic
    return _generate_regular_scalar_data(field, num_rows, pk_offset)


def _generate_regular_scalar_data(
    field: dict[str, Any], num_rows: int, pk_offset: int = 0
) -> np.ndarray | list:
    """
    Generate regular scalar field data without default value considerations.

    Args:
        field: Field definition with type, constraints, and cardinality
        num_rows: Number of rows to generate
        pk_offset: Offset for primary key generation

    Returns:
        Generated data as numpy array or list
    """
    field_name = field["name"]
    field_type = field["type"]

    # Get constraints
    min_val = field.get("min")
    max_val = field.get("max")
    cardinality_ratio = field.get("cardinality_ratio")
    enum_values = field.get("enum_values")

    # Handle enum values (highest priority)
    if enum_values:
        # Randomly select from enum values
        indices = np.random.randint(0, len(enum_values), size=num_rows)
        enum_array = np.array(enum_values)
        return enum_array[indices]

    # Handle cardinality ratio constraint
    if cardinality_ratio is not None:
        # Calculate number of unique values
        num_unique = max(1, int(num_rows * cardinality_ratio))

        # Generate based on field type
        if field_type in ["Int8", "Int16", "Int32", "Int64"]:
            # Set defaults based on type
            type_ranges = {
                "Int8": (-128, 127),
                "Int16": (-32768, 32767),
                "Int32": (-2147483648, 2147483647),
                "Int64": (-999999, 999999),  # Limited range for readability
            }
            default_min, default_max = type_ranges[field_type]
            min_val = min_val if min_val is not None else default_min
            max_val = max_val if max_val is not None else default_max

            # Generate unique values
            unique_values = np.random.randint(min_val, max_val + 1, size=num_unique)
            # Repeat to fill all rows
            indices = np.random.randint(0, num_unique, size=num_rows)

            # Convert to appropriate dtype
            dtype_map = {
                "Int8": np.int8,
                "Int16": np.int16,
                "Int32": np.int32,
                "Int64": np.int64,
            }
            return unique_values[indices].astype(dtype_map[field_type])

        elif field_type in ["Float", "Double"]:
            min_val = min_val if min_val is not None else 0.0
            max_val = max_val if max_val is not None else 1000.0

            # Generate unique values
            unique_float_values = np.random.uniform(min_val, max_val, size=num_unique)
            # Repeat to fill all rows
            indices = np.random.randint(0, num_unique, size=num_rows)
            result = unique_float_values[indices]
            return result.astype(np.float32) if field_type == "Float" else result

        elif field_type in ["VarChar", "String"]:
            # Generate string pool based on cardinality
            string_pool = [f"{field_name}_{i}" for i in range(num_unique)]
            indices = np.random.randint(0, len(string_pool), size=num_rows)
            string_array = np.array(string_pool)
            return list(string_array[indices])

    # Default generation (no cardinality specified)
    if field_type in ["Int8", "Int16", "Int32", "Int64"]:
        type_info = {
            "Int8": (np.int8, -128, 127),
            "Int16": (np.int16, -32768, 32767),
            "Int32": (np.int32, -2147483648, 2147483647),
            "Int64": (np.int64, -999999, 999999),
        }
        dtype, default_min, default_max = type_info[field_type]
        min_val = min_val if min_val is not None else default_min
        max_val = max_val if max_val is not None else default_max
        return np.random.randint(min_val, max_val + 1, size=num_rows).astype(dtype)

    elif field_type == "Float":
        min_val = min_val if min_val is not None else 0.0
        max_val = max_val if max_val is not None else 1.0
        return np.random.uniform(min_val, max_val, size=num_rows).astype(np.float32)

    elif field_type == "Double":
        min_val = min_val if min_val is not None else 0.0
        max_val = max_val if max_val is not None else 1.0
        return np.random.uniform(min_val, max_val, size=num_rows)

    elif field_type == "Bool":
        return np.random.randint(0, 2, size=num_rows, dtype=bool)

    elif field_type in ["VarChar", "String"]:
        # Default string generation
        string_pool = [f"text_{i % 1000}" for i in range(1000)]
        indices = np.random.randint(0, len(string_pool), size=num_rows)
        string_array = np.array(string_pool)
        return list(string_array[indices])

    else:
        raise ValueError(f"Unsupported field type: {field_type}")


def generate_data_optimized(
    schema_path: Path,
    total_rows: int,
    output_dir: Path,
    format: str = "parquet",
    batch_size: int = 50000,
    seed: int | None = None,
    file_size: str | None = None,
    rows_per_file: int = 1000000,
    progress_callback: Any = None,
    num_partitions: int | None = None,
    num_shards: int | None = None,
    file_count: int | None = None,
    num_workers: int | None = None,
    chunk_and_merge: bool = False,
    chunk_size: str = "512MB",
    skip_validation: bool = False,
) -> tuple[list[str], int]:
    """
    Optimized data generation using vectorized NumPy operations with file partitioning.

    Key optimizations:
    1. Vectorized data generation with NumPy
    2. Pre-allocated arrays
    3. Batch processing with automatic file partitioning
    4. Efficient Parquet/JSON writing
    5. Minimal memory copying
    6. Smart file splitting based on size and row count

    Args:
        max_file_size_mb: Maximum size per file in MB (default: 256MB)
        max_rows_per_file: Maximum rows per file (default: 1M rows)
    
    Returns:
        tuple[list[str], int]: A tuple containing (list of created file paths, actual total rows generated)
    """
    start_time = time.time()

    # If chunk-and-merge is explicitly requested and we have file_count + file_size, 
    # we need to defer it until we calculate the actual total rows
    defer_chunk_merge = chunk_and_merge and file_size and file_count
    
    # If chunk-and-merge is requested but we're not deferring, use it now
    if chunk_and_merge and not defer_chunk_merge:
        return _generate_with_chunk_and_merge(
            schema_path=schema_path,
            total_rows=total_rows,
            output_dir=output_dir,
            format=format,
            batch_size=batch_size,
            seed=seed,
            file_size=file_size,
            rows_per_file=rows_per_file,
            progress_callback=progress_callback,
            num_partitions=num_partitions,
            num_shards=num_shards,
            file_count=file_count,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

    # Load schema
    with open(schema_path) as f:
        schema = json.load(f)

    fields = schema.get("fields", schema)
    enable_dynamic = schema.get("enable_dynamic_field", False)
    dynamic_fields = schema.get("dynamic_fields", []) if enable_dynamic else []

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    if seed:
        np.random.seed(seed)

    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    elif num_workers <= 0:
        num_workers = 1

    logger.info(f"Using {num_workers} worker processes for parallel file generation")

    # Pre-analyze schema for optimization and distribution
    vector_fields = []
    scalar_fields = []
    partition_key_field = None
    primary_key_field = None

    for field in fields:
        # Skip auto_id fields - they should not be generated
        if field.get("auto_id", False):
            logger.info(f"Skipping auto_id field: {field['name']}")
            continue

        # Skip BM25 function output fields - they are auto-generated by Milvus
        if _is_bm25_output_field(field["name"], schema):
            logger.info(f"Skipping BM25 output field: {field['name']}")
            continue

        # Identify special fields
        if field.get("is_partition_key", False):
            partition_key_field = field
        if field.get("is_primary", False):
            primary_key_field = field

        if "Vector" in field["type"]:
            vector_fields.append(field)
        else:
            scalar_fields.append(field)

    # Validate distribution parameters
    if num_partitions and not partition_key_field:
        logger.warning(
            "num_partitions specified but no partition key field found in schema. Ignoring partitions parameter."
        )
        num_partitions = None

    if num_partitions and partition_key_field:
        logger.info(
            f"Partition distribution enabled: {num_partitions} partitions using field '{partition_key_field['name']}'"
        )

    if num_shards and primary_key_field:
        logger.info(
            f"Shard distribution enabled: {num_shards} shards using primary key '{primary_key_field['name']}'"
        )
    elif num_shards:
        logger.warning(
            "num_shards specified but no primary key field found in schema. Ignoring shards parameter."
        )
        num_shards = None

    # Use new parameter validation and calculation system
    try:
        validated_params = validate_and_calculate_params(
            file_size=file_size,
            file_count=file_count,
            total_rows=total_rows,
            fields=fields,
            format=format,
            schema=schema,
            seed=seed,
        )
        
        # Extract validated parameters
        final_file_size = validated_params.file_size
        final_file_count = validated_params.file_count
        final_total_rows = validated_params.total_rows
        
        logger.info(
            f"✓ Parameter validation complete: "
            f"{final_total_rows:,} rows → {final_file_count} files × {final_file_size} each"
        )
        
        # Debug: Log the validated parameters
        logger.info(f"🔍 DEBUG: Validated params - rows={final_total_rows}, files={final_file_count}, size={final_file_size}")
        logger.info(f"🔍 DEBUG: Original input params - total_rows={total_rows}, file_count={file_count}, file_size={file_size}")
        
    except ValueError as e:
        logger.error(f"Parameter validation failed: {e}")
        raise

    # Update local variables for compatibility with rest of function
    rows = final_total_rows
    file_size = final_file_size
    file_count = final_file_count
    
    # Debug: Log the updated local variables
    logger.info(f"🔍 DEBUG: Updated local vars - rows={rows}, file_size={file_size}, file_count={file_count}")

    # Determine optimal generation strategy based on validated parameters
    strategy_info = determine_generation_strategy(
        file_size=file_size,
        file_count=file_count,
        chunk_size=chunk_size,
    )
    
    logger.info(
        f"📋 Generation strategy: {strategy_info['strategy']} "
        f"({strategy_info['reason']})"
    )
    logger.info(
        f"📊 Strategy details: memory={strategy_info.get('memory_profile', 'N/A')}, "
        f"optimization={strategy_info.get('optimization', 'N/A')}"
    )

    # Check if we should use chunk-and-merge strategy
    should_use_chunk_merge = chunk_and_merge or defer_chunk_merge
    
    if file_size and not should_use_chunk_merge:
        file_size_bytes = _parse_file_size(file_size)
        file_size_mb = file_size_bytes / (1024**2)
        logger.debug(f"File size: {file_size} = {file_size_mb:.1f}MB")
        
        # Auto-enable chunk-and-merge for files >= 512MB (updated threshold)
        if file_size_mb >= 512:
            should_use_chunk_merge = True
            logger.info(
                f"Auto-enabling chunk-and-merge for file size ≥512MB ({file_size_mb:.1f}MB). "
                f"This improves performance for medium-to-large file generation."
            )

    
    if should_use_chunk_merge:
        # Use chunk-and-merge strategy with validated parameters
        logger.info(f"🔍 DEBUG: Using chunk-and-merge with rows={rows}, files={file_count}, size={file_size}")
        return _generate_with_chunk_and_merge(
            schema_path=schema_path,
            total_rows=rows,
            output_dir=output_dir,
            format=format,
            batch_size=batch_size,
            seed=seed,
            file_size=file_size,
            rows_per_file=rows_per_file,
            progress_callback=progress_callback,
            num_partitions=num_partitions,
            num_shards=num_shards,
            file_count=file_count,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

    # Calculate effective parameters for regular (non-chunk) generation
    logger.info(f"🔍 DEBUG: Using regular generation with rows={rows}, files={file_count}, size={file_size}")
    sample_size = min(max(1000, rows // 50), 10000)
    actual_row_size_bytes, estimation_stats = _enhanced_estimate_row_size_from_sample(
        fields, sample_size, 0, format, schema, seed, num_iterations=3
    )
    
    target_size_bytes = _parse_file_size(file_size)
    effective_max_rows_per_file = max(1, int(target_size_bytes // actual_row_size_bytes))
    total_files = file_count
    
    logger.info(f"🔍 DEBUG: Regular generation parameters:")
    logger.info(f"    - effective_max_rows_per_file: {effective_max_rows_per_file:,}")
    logger.info(f"    - total_files: {total_files}")
    
    # Adjust worker count based on generation strategy and file constraints
    estimated_file_size_bytes = effective_max_rows_per_file * actual_row_size_bytes
    num_workers = adjust_workers_by_strategy(
        num_workers=num_workers,
        strategy_info=strategy_info,
        file_size_bytes=estimated_file_size_bytes,
        file_count=total_files,
    )
    
    logger.info(
        f"Regular generation: ~{estimated_file_size_bytes / (1024**3):.1f}GB per file, "
        f"{effective_max_rows_per_file:,} rows/file, using {num_workers} worker(s)"
    )

    # Continue with the rest of the function (parallel vs serial decision)
    logger.info(f"🔍 DEBUG: Final parameters for generation:")
    logger.info(f"    - rows: {rows}")
    logger.info(f"    - total_files: {total_files}")
    logger.info(f"    - effective_max_rows_per_file: {effective_max_rows_per_file}")
    logger.info(f"    - num_workers: {num_workers}")

    # Decide whether to use parallel or serial generation
    use_parallel = num_workers > 1 and total_files > 1

    if use_parallel:
        logger.info(
            f"🚀 Using parallel generation: {num_workers} workers for {total_files} files"
        )
        return _generate_files_parallel(
            schema=schema,
            fields=fields,
            enable_dynamic=enable_dynamic,
            dynamic_fields=dynamic_fields,
            total_files=total_files,
            effective_max_rows_per_file=effective_max_rows_per_file,
            rows=rows,
            output_dir=output_dir,
            format=format,
            seed=seed,
            num_partitions=num_partitions,
            num_shards=num_shards,
            partition_key_field=partition_key_field,
            primary_key_field=primary_key_field,
            vector_fields=vector_fields,
            scalar_fields=scalar_fields,
            num_workers=num_workers,
            progress_callback=progress_callback,
            start_time=start_time,
            estimation_stats=estimation_stats,
            file_size=file_size,
            skip_validation=skip_validation,
        )
    else:
        logger.info(f"📝 Using serial generation: {total_files} file(s)")
        # Fall back to original serial generation

    # Generate data in batches and write multiple files if needed (SERIAL MODE)
    remaining_rows = rows
    file_index = 0
    pk_offset = 0
    all_files_created = []
    all_files_info = []  # Store detailed file information (path, rows, etc.)
    total_generation_time = 0.0
    total_write_time = 0.0
    all_optimization_info = []  # Collect optimization info from all files

    while remaining_rows > 0:
        # Determine batch size for this file (respect both size and row constraints)
        current_batch_rows = min(remaining_rows, effective_max_rows_per_file)

        # Use debug level when progress callback is provided to avoid mixing with progress bar
        if progress_callback:
            logger.debug(
                f"Generating file {file_index + 1}: {current_batch_rows:,} rows"
            )
        else:
            logger.info(
                f"Generating file {file_index + 1}: {current_batch_rows:,} rows"
            )

        generation_start = time.time()
        data: dict[str, Any] = {}

        # Generate partition key with enough unique values (if partition key exists)
        if partition_key_field and num_partitions:
            # Generate partition key values with unique count = num_partitions * 10
            partition_unique_count = num_partitions * 10
            partition_values = _generate_partition_key_with_cardinality(
                partition_key_field,
                current_batch_rows,
                partition_unique_count,
                pk_offset,
            )
            data[partition_key_field["name"]] = partition_values

        # Generate remaining scalar fields efficiently
        for field in scalar_fields:
            field_name = field["name"]
            field_type = field["type"]

            # Skip fields already generated (partition key, primary key)
            if field_name in data:
                continue

            if field_type in [
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Float",
                "Double",
                "Bool",
                "VarChar",
                "String",
            ]:
                # Use the new cardinality-aware generation function
                data[field_name] = _generate_scalar_field_data(
                    field, current_batch_rows, pk_offset
                )
            elif field_type == "JSON":
                # Generate diverse JSON data with multiple patterns
                json_data = []

                # Pre-generate random data for efficiency
                random_types = np.random.randint(0, 5, size=current_batch_rows)
                random_ints = np.random.randint(1, 1000, size=current_batch_rows)
                random_floats = np.random.random(current_batch_rows)
                random_bools = np.random.randint(
                    0, 2, size=current_batch_rows, dtype=bool
                )

                # Pre-generate string pools
                categories = [
                    "electronics",
                    "books",
                    "clothing",
                    "food",
                    "toys",
                    "sports",
                    "health",
                    "home",
                ]
                statuses = ["active", "pending", "completed", "cancelled", "processing"]
                tags_pool = [
                    "new",
                    "featured",
                    "sale",
                    "limited",
                    "exclusive",
                    "popular",
                    "trending",
                    "clearance",
                ]

                for i in range(current_batch_rows):
                    json_type = random_types[i]

                    if json_type == 0:
                        # E-commerce product metadata
                        json_obj = {
                            "product_id": int(pk_offset + i),
                            "category": categories[i % len(categories)],
                            "price": round(float(random_floats[i] * 999.99), 2),
                            "in_stock": bool(random_bools[i]),
                            "attributes": {
                                "brand": f"Brand_{random_ints[i] % 50}",
                                "weight": round(float(random_floats[i] * 10), 2),
                                "dimensions": {
                                    "length": int(random_ints[i] % 100),
                                    "width": int((random_ints[i] + 10) % 100),
                                    "height": int((random_ints[i] + 20) % 100),
                                },
                            },
                            "tags": tags_pool[: random_ints[i] % 4 + 1],
                        }
                    elif json_type == 1:
                        # User activity/event data
                        json_obj = {
                            "event_id": int(pk_offset + i),
                            "event_type": [
                                "click",
                                "view",
                                "purchase",
                                "share",
                                "like",
                            ][i % 5],
                            "timestamp": int(1600000000 + random_ints[i] * 1000),
                            "user": {
                                "id": f"user_{random_ints[i] % 1000}",
                                "session": f"session_{random_ints[i] % 100}",
                                "device": ["mobile", "desktop", "tablet"][i % 3],
                            },
                            "metrics": {
                                "duration_ms": int(random_ints[i] * 10),
                                "clicks": int(random_ints[i] % 10),
                                "score": round(float(random_floats[i] * 5), 2),
                            },
                        }
                    elif json_type == 2:
                        # Configuration/settings data
                        json_obj = {
                            "config_id": int(pk_offset + i),
                            "name": f"config_{i}",
                            "settings": {
                                "enabled": bool(random_bools[i]),
                                "threshold": float(random_floats[i]),
                                "max_retries": int(random_ints[i] % 10),
                                "timeout_seconds": int(random_ints[i] % 300),
                                "features": {
                                    "feature_a": bool(random_bools[i]),
                                    "feature_b": bool(not random_bools[i]),
                                    "feature_c": bool(i % 3 == 0),
                                },
                            },
                            "metadata": {
                                "version": f"{random_ints[i] % 3}.{random_ints[i] % 10}.{random_ints[i] % 20}",
                                "last_updated": int(1600000000 + random_ints[i] * 1000),
                            },
                        }
                    elif json_type == 3:
                        # Analytics/metrics data
                        json_obj = {
                            "metric_id": int(pk_offset + i),
                            "type": "analytics",
                            "values": {
                                "count": int(random_ints[i]),
                                "sum": round(float(random_floats[i] * 10000), 2),
                                "avg": round(float(random_floats[i] * 100), 2),
                                "min": round(float(random_floats[i] * 10), 2),
                                "max": round(float(random_floats[i] * 1000), 2),
                            },
                            "dimensions": {
                                "region": ["north", "south", "east", "west"][i % 4],
                                "category": categories[i % len(categories)],
                                "segment": f"segment_{random_ints[i] % 10}",
                            },
                            "percentiles": {
                                "p50": round(float(random_floats[i] * 50), 2),
                                "p90": round(float(random_floats[i] * 90), 2),
                                "p99": round(float(random_floats[i] * 99), 2),
                            },
                        }
                    else:
                        # Document/content metadata
                        json_obj = {
                            "doc_id": int(pk_offset + i),
                            "title": f"Document_{i}",
                            "status": statuses[i % len(statuses)],
                            "metadata": {
                                "author": f"author_{random_ints[i] % 100}",
                                "created_at": int(1600000000 + random_ints[i] * 1000),
                                "word_count": int(random_ints[i] * 10),
                                "language": ["en", "es", "fr", "de", "zh"][i % 5],
                                "sentiment": {
                                    "positive": round(float(random_floats[i]), 3),
                                    "negative": round(float(1 - random_floats[i]), 3),
                                    "neutral": round(float(random_floats[i] * 0.5), 3),
                                },
                            },
                            "tags": tags_pool[: random_ints[i] % 3 + 1],
                            "properties": {
                                "public": bool(random_bools[i]),
                                "archived": bool(not random_bools[i]),
                                "priority": int(random_ints[i] % 5),
                            },
                        }

                    json_data.append(json_obj)

                data[field_name] = json_data

            elif field_type == "Array":
                element_type = field.get("element_type", "Int32")
                max_capacity = field.get("max_capacity", 5)

                # Optimized vectorized array generation
                lengths = np.random.randint(
                    0, max_capacity + 1, size=current_batch_rows
                )

                if element_type in ["Int32", "Int64"]:
                    # Vectorized integer array generation
                    total_elements = np.sum(lengths)
                    if total_elements > 0:
                        # Pre-generate large pool of integers and slice as needed
                        int_pool = np.random.randint(
                            -999, 999, size=total_elements, dtype=np.int32
                        )
                        array_data = []
                        start_idx = 0
                        for length in lengths:
                            if length > 0:
                                array_data.append(
                                    int_pool[start_idx : start_idx + length].tolist()
                                )
                                start_idx += length
                            else:
                                array_data.append([])
                    else:
                        array_data = [[] for _ in range(current_batch_rows)]
                    data[field_name] = array_data
                else:
                    # Optimized string arrays with pre-computed strings
                    str_templates = [f"item_{j}" for j in range(max_capacity)]
                    array_data = []
                    for length in lengths:
                        if length > 0:
                            array_data.append(str_templates[:length])
                        else:
                            array_data.append([])
                    data[field_name] = array_data

        # Generate vector fields efficiently
        for field in vector_fields:
            field_name = field["name"]
            field_type = field["type"]
            dim = field.get("dim", 128)

            if field_type == "FloatVector":
                # Generate normalized float vectors efficiently (uses multiple CPU cores)
                vectors = np.random.randn(current_batch_rows, dim).astype(np.float32)

                # Always use NumPy for vector normalization (consistently faster due to optimized BLAS)
                # JIT compilation overhead typically outweighs benefits for this simple operation
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms

                # Store as list of arrays for pandas
                data[field_name] = list(vectors)

            elif field_type == "BinaryVector":
                # Binary vector: each int represents 8 dimensions
                # If binary vector dimension is 16, use [x, y] where x and y are 0-255
                byte_dim = dim // 8
                binary_vectors = np.random.randint(
                    0, 256, size=(current_batch_rows, byte_dim), dtype=np.uint8
                )
                data[field_name] = list(binary_vectors)

            elif field_type in ["Float16Vector", "BFloat16Vector"]:
                if field_type == "Float16Vector":
                    # Generate float16 vectors using uint8 representation
                    fp16_vectors = []
                    for _ in range(current_batch_rows):
                        raw_vector = np.random.random(dim)
                        fp16_vector = (
                            np.array(raw_vector, dtype=np.float16)
                            .view(np.uint8)
                            .tolist()
                        )
                        fp16_vectors.append(fp16_vector)
                    data[field_name] = fp16_vectors
                else:  # BFloat16Vector
                    # Generate bfloat16 vectors using uint8 representation
                    bf16_vectors = []
                    for _ in range(current_batch_rows):
                        raw_vector = np.random.random(dim)
                        bf16_vector = (
                            np.array(raw_vector, dtype=bfloat16).view(np.uint8).tolist()
                        )
                        bf16_vectors.append(bf16_vector)
                    data[field_name] = bf16_vectors

            elif field_type == "SparseFloatVector":
                # Generate sparse float vectors as dict with indices as keys and values as floats
                sparse_vectors = []
                for _ in range(current_batch_rows):
                    max_dim = 1000
                    non_zero_count = np.random.randint(
                        10, max_dim // 10
                    )  # 10-100 non-zero values
                    indices = np.random.choice(max_dim, non_zero_count, replace=False)
                    values = np.random.random(non_zero_count)
                    sparse_vector = {
                        str(index): float(value)
                        for index, value in zip(indices, values, strict=False)
                    }
                    sparse_vectors.append(sparse_vector)
                data[field_name] = sparse_vectors

        # Generate dynamic fields if enabled
        if enable_dynamic and dynamic_fields:
            logger.debug(f"Generating {len(dynamic_fields)} dynamic fields")
            meta_data = []

            # Generate $meta field containing all dynamic fields
            for _ in range(current_batch_rows):
                row_meta = {}

                for dynamic_field in dynamic_fields:
                    field_name = dynamic_field.get("name")
                    field_values = _generate_dynamic_field_data(
                        dynamic_field, 1
                    )  # Generate one value at a time

                    # Only add the field if it has a non-None value
                    if field_values[0] is not None:
                        row_meta[field_name] = field_values[0]

                # Add the meta object (can be empty if no dynamic fields were generated for this row)
                meta_data.append(row_meta if row_meta else {})

            # Add $meta field to main data
            if meta_data:
                data["$meta"] = meta_data
                non_empty_count = sum(1 for meta in meta_data if meta)
                logger.debug(
                    f"Generated $meta field with dynamic data for {non_empty_count}/{current_batch_rows} rows"
                )

        batch_generation_time = time.time() - generation_start
        total_generation_time += batch_generation_time

        # Create DataFrame
        df = pd.DataFrame(data)

        # Apply Parquet column optimization (only for Parquet format)
        optimization_info = {}
        if format.lower() == "parquet":
            df, optimization_info = _optimize_parquet_columns(df, fields, format)

        # Collect optimization info if any optimizations were made
        if optimization_info and "_summary" in optimization_info:
            file_optimization = {
                "file_index": file_index,
                "file_name": f"data-{file_index + 1:05d}-of-{total_files:05d}.parquet"
                if total_files > 1
                else "data.parquet",
                "rows": current_batch_rows,
                "optimization": optimization_info,
            }
            all_optimization_info.append(file_optimization)

        # No need to collect complex distribution info - just record basic stats

        # Convert JSON fields to strings for Parquet storage
        if format.lower() == "parquet":
            # Handle regular JSON fields
            for field in fields:
                if field["type"] == "JSON":
                    field_name = field["name"]
                    if field_name in df.columns:
                        # Convert JSON objects to JSON strings for Parquet storage
                        df[field_name] = df[field_name].apply(
                            lambda x: json.dumps(x, ensure_ascii=False)
                            if x is not None
                            else None
                        )

            # Handle $meta field (contains all dynamic fields as JSON)
            if "$meta" in df.columns:
                df["$meta"] = df["$meta"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if x is not None and x != {}
                    else None
                )

        # Write file
        write_start = time.time()

        if format.lower() == "parquet":
            if total_files == 1:
                output_file = output_dir / "data.parquet"
            else:
                file_num = file_index + 1
                output_file = (
                    output_dir / f"data-{file_num:05d}-of-{total_files:05d}.parquet"
                )

            # Convert to PyArrow table for efficient writing
            table = pa.Table.from_pandas(df)

            # Write with optimized settings
            pq.write_table(
                table,
                output_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=False,
                row_group_size=min(50000, current_batch_rows),
            )

        elif format.lower() == "json":
            if total_files == 1:
                output_file = output_dir / "data.json"
            else:
                file_num = file_index + 1
                output_file = (
                    output_dir / f"data-{file_num:05d}-of-{total_files:05d}.json"
                )

            # Write JSON in Milvus bulk import format (list of dict)
            rows_data = []
            for record in df.to_dict(orient="records"):
                # Handle numpy types and field omission for nullable/default fields
                row_record = {}
                for key, value in record.items():
                    # Find field definition for this key
                    field_def = None
                    for field in fields:
                        if field["name"] == key:
                            field_def = field
                            break

                    # Handle value conversion
                    if isinstance(value, np.ndarray):
                        converted_value = value.tolist()
                    elif isinstance(value, np.generic):
                        converted_value = value.item()
                    else:
                        converted_value = value

                    # Handle NaN values for JSON compatibility
                    import math

                    if isinstance(converted_value, float) and math.isnan(
                        converted_value
                    ):
                        converted_value = (
                            None  # Convert NaN to None for JSON serialization
                        )

                    # Decide how to handle this field in JSON
                    if field_def:
                        import random

                        # For nullable fields: randomly omit with 40% probability (Milvus will handle as null)
                        if field_def.get("nullable", False):
                            should_omit = random.random() < 0.4
                            if not should_omit:
                                row_record[key] = converted_value

                        # For default_value fields: randomly omit with 30% probability (Milvus will use default)
                        elif field_def.get("default_value") is not None:
                            should_omit = random.random() < 0.3
                            if not should_omit:
                                row_record[key] = converted_value
                        else:
                            # Regular field - always include
                            row_record[key] = converted_value
                    else:
                        # Field definition not found - include as-is
                        row_record[key] = converted_value

                rows_data.append(row_record)

            # Write as JSON array for Milvus bulk import compatibility (list of dict)
            with open(output_file, "w") as f:
                json.dump(rows_data, f, ensure_ascii=False, separators=(",", ":"))

        else:
            raise ValueError(
                f"Unsupported format: {format}. High-performance mode only supports 'parquet' and 'json' formats."
            )

        batch_write_time = time.time() - write_start
        total_write_time += batch_write_time

        all_files_created.append(str(output_file))
        
        # Store detailed file info
        file_size = output_file.stat().st_size if output_file.exists() else 0
        all_files_info.append({
            "file_name": Path(output_file).name,
            "file_path": str(output_file),
            "rows": current_batch_rows,
            "file_index": file_index,
            "file_size_bytes": file_size,
        })

        # Log progress - use debug level when progress callback is provided to avoid mixing with progress bar
        completion_msg = (
            f"File {file_index + 1} completed: {current_batch_rows:,} rows "
            f"({current_batch_rows / batch_generation_time:.0f} rows/sec generation)"
        )
        if progress_callback:
            logger.debug(completion_msg)
        else:
            logger.info(completion_msg)

        # Update counters
        remaining_rows -= current_batch_rows
        pk_offset += current_batch_rows
        file_index += 1

        # Update progress if callback provided
        if progress_callback:
            progress_callback(pk_offset)

    # Write metadata
    meta_file = output_dir / "meta.json"
    total_time = time.time() - start_time

    # Calculate actual file size limit in MB
    if file_size:
        target_file_size_mb = _parse_file_size(file_size) / (1024 * 1024)
    else:
        target_file_size_mb = 256  # Default 256MB

    metadata = {
        "schema": schema,
        "generation_info": {
            "total_rows": rows,
            "format": format,
            "seed": seed,
            "data_files": all_files_info,
            "file_count": len(all_files_created),
            "max_rows_per_file": rows_per_file,
            "max_file_size_mb": target_file_size_mb,
            "generation_time": total_generation_time,
            "write_time": total_write_time,
            "total_time": total_time,
            "rows_per_second": rows / total_time,
            "size_estimation": estimation_stats,
        },
    }

    # Add collection configuration for Milvus create_collection
    collection_config = {}

    # Add partition configuration if specified
    if num_partitions and partition_key_field:
        collection_config["num_partitions"] = num_partitions
        collection_config["partition_key_field"] = partition_key_field["name"]
        logger.info(
            f"Collection config: {num_partitions} partitions using field '{partition_key_field['name']}'"
        )

    # Add shard configuration if specified
    if num_shards:
        collection_config["num_shards"] = num_shards
        logger.info(f"Collection config: {num_shards} shards")

    # Add collection config to metadata if any settings were specified
    if collection_config:
        metadata["collection_config"] = collection_config

    # Add optimization information if any occurred
    if all_optimization_info:
        # Calculate total optimization statistics
        total_optimized_columns = 0
        total_savings_mb = 0.0
        optimized_files = 0

        for file_info in all_optimization_info:
            if "_summary" in file_info["optimization"]:
                summary = file_info["optimization"]["_summary"]
                total_optimized_columns += summary.get("columns_optimized", 0)
                total_savings_mb += summary.get("total_memory_savings_mb", 0.0)
                optimized_files += 1

        metadata["generation_info"]["parquet_optimization"] = {
            "enabled": True,
            "files_optimized": optimized_files,
            "total_files": len(all_files_created),
            "total_columns_optimized": total_optimized_columns,
            "total_savings_mb": round(total_savings_mb, 2),
            "optimization_details": all_optimization_info,
        }
    elif format.lower() == "parquet":
        metadata["generation_info"]["parquet_optimization"] = {
            "enabled": True,
            "files_optimized": 0,
            "total_files": len(all_files_created),
            "note": "No uniform columns detected for optimization",
        }

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Convert metadata to handle numpy types
    serializable_metadata = convert_numpy_types(metadata)

    with open(meta_file, "w") as f:
        json.dump(serializable_metadata, f, indent=2)

    # Show summary logs only when no progress callback (to avoid mixing with progress bar)
    if not progress_callback:
        logger.info(
            f"Total generation completed: {rows:,} rows in {len(all_files_created)} file(s)"
        )
        logger.info(f"Total time: {total_time:.2f}s ({rows / total_time:.0f} rows/sec)")

    # Run minimal validation unless skipped (for serial generation path)
    if not skip_validation:
        print("📊 Starting minimal file validation...")
        from .minimal_validator import MinimalValidator
        from rich.console import Console

        try:
            # Create validator
            console = Console()
            validator = MinimalValidator(output_dir, console)

            # Run validation
            print("🔍 Running minimal validation (schema + count)...")
            logger.info("Running minimal file validation...")
            validation_results = validator.validate()

            # Display results
            validator.display_results(validation_results)

            # Log validation status with higher visibility
            if validation_results["valid"]:
                print(
                    f"✅ File validation passed: {validation_results['summary']['total_files']} files, {validation_results['summary']['total_rows']:,} rows"
                )
                logger.info("✅ File validation passed")
            else:
                print(
                    f"⚠️ File validation failed with {len(validation_results['errors'])} errors"
                )
                logger.warning(
                    f"⚠️ File validation failed with {len(validation_results['errors'])} errors"
                )
                # Don't fail the generation, just warn
                for error in validation_results["errors"]:
                    print(f"  - {error}")
                    logger.warning(f"  - {error}")

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            logger.warning(f"Failed to run validation: {e}")
            # Don't fail the generation if validation fails

    return all_files_created, rows


def _generate_single_chunk(
    schema_path: Path,
    chunk_output_dir: Path,
    chunk_rows: int,
    format: str,
    batch_size: int,
    seed: int | None,
    num_partitions: int | None,
    num_shards: int | None,
    chunk_id: int,
    chunk_size: str,  # Add chunk_size parameter
) -> list[str]:
    """
    Generate a single chunk file directly without further subdivision.
    This function is designed to be called from ProcessPoolExecutor.
    
    Returns:
        List of generated file paths (should be exactly one file)
    """
    # Use single-threaded generation for individual chunks
    # This generates chunks as single files without internal subdivision
    chunk_files_created, _ = generate_data_optimized(
        schema_path=schema_path,
        total_rows=chunk_rows,
        output_dir=chunk_output_dir,
        format=format,
        batch_size=batch_size,
        seed=seed,
        file_size=None,  # Don't use file size control for chunks
        rows_per_file=chunk_rows,  # Single file per chunk
        progress_callback=None,  # Disable individual progress bars
        num_partitions=num_partitions,
        num_shards=num_shards,
        file_count=None,
        num_workers=1,  # IMPORTANT: Use single worker for each chunk
        chunk_and_merge=False,  # Prevent recursion
        skip_validation=True,  # Skip validation for temporary chunks
    )
    
    return chunk_files_created


def _generate_with_chunk_and_merge(
    schema_path: Path,
    total_rows: int,
    output_dir: Path,
    format: str = "parquet",
    batch_size: int = 50000,
    seed: int | None = None,
    file_size: str | None = None,
    rows_per_file: int = 1000000,
    progress_callback: Any = None,
    num_partitions: int | None = None,
    num_shards: int | None = None,
    file_count: int | None = None,
    num_workers: int | None = None,
    chunk_size: str = "512MB",
) -> tuple[list[str], int]:
    """
    Generate data using chunk-and-merge strategy for better performance on large files.

    This strategy:
    1. Determines if chunk-and-merge would be beneficial
    2. Generates multiple smaller chunks in parallel
    3. Merges chunks into final large files
    4. Cleans up temporary files
    """
    from .file_merger import FileMerger
    import shutil

    start_time = time.time()
    logger.info("🚀 Using chunk-and-merge strategy for large file generation")

    # Parse chunk size
    chunk_size_bytes = _parse_file_size(chunk_size)
    target_file_size_bytes = (
        _parse_file_size(file_size) if file_size else (256 * 1024 * 1024)
    )  # 256MB default

    # Determine if chunk-and-merge is beneficial
    if target_file_size_bytes < (2 * 1024 * 1024 * 1024):  # Less than 2GB
        logger.info("Target file size < 2GB, using standard generation instead")
        # Call original function without chunk_and_merge flag
        return generate_data_optimized(
            schema_path=schema_path,
            total_rows=total_rows,
            output_dir=output_dir,
            format=format,
            batch_size=batch_size,
            seed=seed,
            file_size=file_size,
            rows_per_file=rows_per_file,
            progress_callback=progress_callback,
            num_partitions=num_partitions,
            num_shards=num_shards,
            file_count=file_count,
            num_workers=num_workers,
            chunk_and_merge=False,
            skip_validation=False,  # Still validate when falling back to standard generation
        )

    # Calculate chunk parameters
    num_chunks = max(1, target_file_size_bytes // chunk_size_bytes)
    rows_per_chunk = max(1000, total_rows // num_chunks)  # At least 1000 rows per chunk

    logger.info(
        f"Generating {num_chunks} chunks of ~{chunk_size} each ({rows_per_chunk:,} rows per chunk)"
    )

    # Create temporary directory for chunks
    temp_dir = output_dir / ".chunks_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate all chunks in parallel using ProcessPoolExecutor
        chunk_files = []
        
        # Prepare chunk generation tasks
        chunk_tasks = []
        remaining_rows = total_rows
        
        for i in range(num_chunks):
            chunk_rows = min(rows_per_chunk, remaining_rows)
            if chunk_rows <= 0:
                break
                
            chunk_seed = (seed + i * 10000) if seed else None
            chunk_output_dir = temp_dir / f"chunk_{i:05d}"
            
            chunk_tasks.append({
                'chunk_id': i,
                'chunk_rows': chunk_rows,
                'chunk_seed': chunk_seed,
                'chunk_output_dir': chunk_output_dir,
            })
            
            remaining_rows -= chunk_rows

        logger.info(f"Generating {len(chunk_tasks)} chunks in parallel using {min(num_workers, len(chunk_tasks))} worker(s)")

        # Use ProcessPoolExecutor to generate chunks in parallel
        actual_workers = min(num_workers, len(chunk_tasks))  # Don't use more workers than chunks
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all chunk generation tasks
            future_to_chunk = {}
            for task in chunk_tasks:
                future = executor.submit(
                    _generate_single_chunk,
                    schema_path=schema_path,
                    chunk_output_dir=task['chunk_output_dir'],
                    chunk_rows=task['chunk_rows'],
                    format=format,
                    batch_size=batch_size,
                    seed=task['chunk_seed'],
                    num_partitions=num_partitions,
                    num_shards=num_shards,
                    chunk_id=task['chunk_id'],
                    chunk_size=chunk_size,
                )
                future_to_chunk[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                task = future_to_chunk[future]
                try:
                    chunk_file_paths = future.result()
                    for chunk_file_path in chunk_file_paths:
                        chunk_files.append(Path(chunk_file_path))
                    logger.info(f"Completed chunk {task['chunk_id'] + 1}/{len(chunk_tasks)} ({task['chunk_rows']:,} rows)")
                except Exception as e:
                    logger.error(f"Chunk {task['chunk_id'] + 1} failed: {e}")
                    raise

        logger.info(f"Generated {len(chunk_files)} chunk files, now merging...")

        # Merge chunks into final file(s)
        merger = FileMerger(temp_dir=temp_dir)
        final_files = []

        if file_count and file_count > 1:
            # Multiple output files: distribute chunks across files
            chunks_per_file = len(chunk_files) // file_count
            extra_chunks = len(chunk_files) % file_count

            chunk_idx = 0
            for file_idx in range(file_count):
                file_chunks_count = chunks_per_file + (
                    1 if file_idx < extra_chunks else 0
                )
                file_chunk_list = chunk_files[chunk_idx : chunk_idx + file_chunks_count]

                if file_chunk_list:
                    final_filename = (
                        f"data-{file_idx + 1:05d}-of-{file_count:05d}.{format}"
                    )
                    final_file_path = output_dir / final_filename

                    merge_stats = merger.merge_files(
                        chunk_files=file_chunk_list,
                        output_file=final_file_path,
                        file_format=format,
                        cleanup_chunks=False,  # Do not cleanup yet
                    )

                    final_files.append(str(final_file_path))
                    logger.info(
                        f"Merged {merge_stats['chunks_merged']} chunks into {final_filename}"
                    )

                chunk_idx += file_chunks_count
        else:
            # Single output file
            final_filename = f"data.{format}"
            final_file_path = output_dir / final_filename

            merge_stats = merger.merge_files(
                chunk_files=chunk_files,
                output_file=final_file_path,
                file_format=format,
                cleanup_chunks=False,  # Do not cleanup yet
            )

            final_files.append(str(final_file_path))
            logger.info(
                f"Merged {merge_stats['chunks_merged']} chunks into {final_filename}"
            )

        total_time = time.time() - start_time
        logger.info(f"Chunk-and-merge completed in {total_time:.1f}s")
        logger.info(
            f"Generated {len(final_files)} final file(s) from {len(chunk_files)} chunks"
        )

        # Generate meta.json file for the final merged files
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Load schema from file
        with open(schema_path) as f:
            schema = json.load(f)

        # Create metadata for final files
        data_files_info = []
        for i, file_path in enumerate(final_files):
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size
            
            data_files_info.append({
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj.relative_to(output_dir)),
                "rows": total_rows,  # All rows in the single merged file
                "file_index": i,
                "file_size_bytes": file_size
            })

        metadata = {
            "schema": schema,
            "generation_info": {
                "total_rows": total_rows,
                "format": format,
                "data_files": data_files_info,
                "file_count": len(final_files),
                "generation_method": "chunk_and_merge",
                "chunk_size": chunk_size,
                "chunks_generated": len(chunk_files),
                "total_time_seconds": total_time,
                "rows_per_second": int(total_rows / total_time) if total_time > 0 else 0,
            }
        }

        # Add partition/shard info if applicable
        if num_partitions:
            metadata["collection_config"] = {"num_partitions": num_partitions}
            if num_shards:
                metadata["collection_config"]["num_shards"] = num_shards

        # Save metadata
        meta_file = output_dir / "meta.json"
        serializable_metadata = convert_numpy_types(metadata)
        with open(meta_file, "w") as f:
            json.dump(serializable_metadata, f, indent=2)

        return final_files, total_rows

    finally:
        # Always cleanup temporary directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary chunk files")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")

"""Centralized constants for milvus-ingest.

This module consolidates magic numbers and configuration constants
that were previously scattered across multiple modules.
"""

from __future__ import annotations

# ==============================================================================
# Performance and Batching
# ==============================================================================

# Default batch size for data generation and processing
DEFAULT_BATCH_SIZE = 50_000

# Sample size for row size estimation
DEFAULT_SAMPLE_SIZE = 5_000

# Default number of iterations for row size estimation
DEFAULT_ESTIMATION_ITERATIONS = 3

# ==============================================================================
# File Generation
# ==============================================================================

# Default file size in MB for output files
DEFAULT_FILE_SIZE_MB = 256

# Default maximum rows per file
DEFAULT_MAX_ROWS_PER_FILE = 1_000_000

# Multipart upload chunk size in bytes (256 MB)
MULTIPART_CHUNK_SIZE = 256 * 1024 * 1024

# ==============================================================================
# Milvus Limits
# ==============================================================================

# Maximum query limit for Milvus queries
MILVUS_QUERY_LIMIT = 16_384

# Maximum vector dimension
MAX_VECTOR_DIMENSION = 32_768

# Default string max length
DEFAULT_STRING_MAX_LENGTH = 256

# ==============================================================================
# Data Generation
# ==============================================================================

# Probability to generate null for nullable fields
DEFAULT_NULL_PROBABILITY = 0.1

# Default seed for reproducible random generation
DEFAULT_SEED = None

# ==============================================================================
# Verification
# ==============================================================================

# Default verification sample ratio (10%)
DEFAULT_VERIFICATION_SAMPLE_RATIO = 0.1

# Maximum verification sample size
MAX_VERIFICATION_SAMPLE_SIZE = 1_000_000

# Minimum verification sample size
MIN_VERIFICATION_SAMPLE_SIZE = 1_000

# ==============================================================================
# Export all constants
# ==============================================================================

__all__ = [
    # Performance
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_ESTIMATION_ITERATIONS",
    # File Generation
    "DEFAULT_FILE_SIZE_MB",
    "DEFAULT_MAX_ROWS_PER_FILE",
    "MULTIPART_CHUNK_SIZE",
    # Milvus Limits
    "MILVUS_QUERY_LIMIT",
    "MAX_VECTOR_DIMENSION",
    "DEFAULT_STRING_MAX_LENGTH",
    # Data Generation
    "DEFAULT_NULL_PROBABILITY",
    "DEFAULT_SEED",
    # Verification
    "DEFAULT_VERIFICATION_SAMPLE_RATIO",
    "MAX_VERIFICATION_SAMPLE_SIZE",
    "MIN_VERIFICATION_SAMPLE_SIZE",
]

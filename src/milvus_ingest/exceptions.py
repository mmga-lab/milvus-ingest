"""Custom exceptions for milvus-ingest.

This module defines a hierarchical exception system for specific error handling
across different components of the library.
"""

from __future__ import annotations


class MilvusIngestError(Exception):
    """Base exception for milvus-ingest."""

    pass


# ==============================================================================
# Schema-related exceptions
# ==============================================================================


class SchemaError(MilvusIngestError):
    """Raised when there's an issue with the schema format or content."""

    pass


class UnsupportedFieldTypeError(SchemaError):
    """Raised when an unsupported field type is encountered."""

    pass


class PrimaryKeyException(SchemaError):
    """Raised when there's an issue with primary key configuration."""

    pass


# ==============================================================================
# Data generation exceptions
# ==============================================================================


class GenerationError(MilvusIngestError):
    """Raised when there's an error during data generation."""

    pass


# ==============================================================================
# Storage and upload exceptions
# ==============================================================================


class S3UploadError(MilvusIngestError):
    """Raised when S3 upload operations fail."""

    pass


class FileOperationError(MilvusIngestError):
    """Raised when file read/write operations fail."""

    pass


# ==============================================================================
# Milvus operation exceptions
# ==============================================================================


class MilvusConnectionError(MilvusIngestError):
    """Raised when connection to Milvus fails."""

    pass


class MilvusInsertError(MilvusIngestError):
    """Raised when data insertion to Milvus fails."""

    pass


class MilvusImportError(MilvusIngestError):
    """Raised when bulk import to Milvus fails."""

    pass


# ==============================================================================
# Verification exceptions
# ==============================================================================


class VerificationError(MilvusIngestError):
    """Raised when data verification fails."""

    pass


class DataMismatchError(VerificationError):
    """Raised when source and Milvus data don't match."""

    pass


# ==============================================================================
# Configuration exceptions
# ==============================================================================


class ConfigurationError(MilvusIngestError):
    """Raised when configuration is invalid or missing."""

    pass

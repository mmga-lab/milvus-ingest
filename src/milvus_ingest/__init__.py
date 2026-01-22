"""Milvus Ingest package.
High-performance data ingestion tool for Milvus vector database.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .exceptions import (
    ConfigurationError,
    DataMismatchError,
    FileOperationError,
    GenerationError,
    MilvusConnectionError,
    MilvusImportError,
    MilvusIngestError,
    MilvusInsertError,
    PrimaryKeyException,
    S3UploadError,
    SchemaError,
    UnsupportedFieldTypeError,
    VerificationError,
)
from .generator import generate_mock_data

__all__ = [
    "__version__",
    "generate_mock_data",
    # Base exception
    "MilvusIngestError",
    # Schema exceptions
    "SchemaError",
    "UnsupportedFieldTypeError",
    "PrimaryKeyException",
    # Generation exceptions
    "GenerationError",
    # Storage exceptions
    "S3UploadError",
    "FileOperationError",
    # Milvus exceptions
    "MilvusConnectionError",
    "MilvusInsertError",
    "MilvusImportError",
    # Verification exceptions
    "VerificationError",
    "DataMismatchError",
    # Configuration exceptions
    "ConfigurationError",
]

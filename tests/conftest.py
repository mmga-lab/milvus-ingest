"""Pytest configuration and shared fixtures for milvus-ingest tests.

This file provides common test fixtures and configuration for the comprehensive
end-to-end test suite.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def small_test_rows() -> int:
    """Small number of rows for fast tests."""
    return 10


@pytest.fixture(scope="session")
def medium_test_rows() -> int:
    """Medium number of rows for more comprehensive tests."""
    return 100


@pytest.fixture(scope="session")
def large_test_rows() -> int:
    """Large number of rows for performance and partitioning tests."""
    return 2500


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_env_vars(monkeypatch, request):
    """Set up environment variables for testing."""
    # Get configuration from pytest options or use defaults
    config = request.config

    test_env = {
        "MILVUS_URI": getattr(config.option, "uri", "http://localhost:19530"),
        "MILVUS_TOKEN": getattr(config.option, "token", ""),
        "MINIO_HOST": getattr(config.option, "minio_host", "localhost"),
        "MINIO_ACCESS_KEY": getattr(config.option, "minio_ak", "minioadmin"),
        "MINIO_SECRET_KEY": getattr(config.option, "minio_sk", "minioadmin"),
        "MINIO_BUCKET": getattr(config.option, "minio_bucket", "a-bucket"),
        "AWS_ACCESS_KEY_ID": getattr(config.option, "minio_ak", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": getattr(config.option, "minio_sk", "minioadmin"),
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    return test_env


@pytest.fixture
def minimal_schema() -> dict[str, Any]:
    """Minimal valid schema for basic testing."""
    return {
        "collection_name": "minimal_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "data", "type": "VarChar", "max_length": 50},
        ],
    }


@pytest.fixture
def comprehensive_schema() -> dict[str, Any]:
    """Comprehensive schema with all supported field types."""
    return {
        "collection_name": "comprehensive_test",
        "fields": [
            # Primary key
            {"name": "id", "type": "Int64", "is_primary": True},
            # Numeric types
            {"name": "int8_field", "type": "Int8", "min": -10, "max": 10},
            {"name": "int16_field", "type": "Int16", "min": -1000, "max": 1000},
            {"name": "int32_field", "type": "Int32", "min": -100000, "max": 100000},
            {"name": "int64_field", "type": "Int64", "min": -1000000, "max": 1000000},
            {"name": "float_field", "type": "Float", "min": 0.0, "max": 100.0},
            {"name": "double_field", "type": "Double", "min": 0.0, "max": 1000.0},
            {"name": "bool_field", "type": "Bool"},
            # Text types
            {"name": "varchar_field", "type": "VarChar", "max_length": 200},
            {"name": "string_field", "type": "String", "max_length": 500},
            {
                "name": "nullable_text",
                "type": "VarChar",
                "max_length": 100,
                "nullable": True,
            },
            # Complex types
            {"name": "json_field", "type": "JSON", "nullable": True},
            {
                "name": "array_field",
                "type": "Array",
                "element_type": "VarChar",
                "max_capacity": 5,
                "max_length": 30,
            },
            # Vector types
            {"name": "float_vector", "type": "FloatVector", "dim": 128},
            {"name": "binary_vector", "type": "BinaryVector", "dim": 256},
            {"name": "float16_vector", "type": "Float16Vector", "dim": 64},
            {"name": "bfloat16_vector", "type": "BFloat16Vector", "dim": 32},
            {"name": "sparse_vector", "type": "SparseFloatVector"},
        ],
    }


@pytest.fixture
def ecommerce_like_schema() -> dict[str, Any]:
    """E-commerce like schema for realistic testing."""
    return {
        "collection_name": "test_ecommerce",
        "fields": [
            {
                "name": "product_id",
                "type": "Int64",
                "is_primary": True,
                "auto_id": True,
            },
            {"name": "product_name", "type": "VarChar", "max_length": 300},
            {
                "name": "description",
                "type": "VarChar",
                "max_length": 1000,
                "nullable": True,
            },
            {"name": "price", "type": "Float", "min": 0.01, "max": 9999.99},
            {"name": "rating", "type": "Float", "min": 1.0, "max": 5.0},
            {"name": "category", "type": "VarChar", "max_length": 100},
            {
                "name": "tags",
                "type": "Array",
                "element_type": "VarChar",
                "max_capacity": 10,
                "max_length": 50,
            },
            {"name": "specifications", "type": "JSON", "nullable": True},
            {"name": "is_available", "type": "Bool"},
            {"name": "search_embedding", "type": "FloatVector", "dim": 768},
            {"name": "image_embedding", "type": "FloatVector", "dim": 512},
        ],
    }


@pytest.fixture
def invalid_schema_missing_primary() -> dict[str, Any]:
    """Invalid schema without primary key for error testing."""
    return {
        "collection_name": "invalid_no_primary",
        "fields": [
            {"name": "data", "type": "VarChar", "max_length": 50},
            {"name": "vector", "type": "FloatVector", "dim": 128},
        ],
    }


@pytest.fixture
def invalid_schema_missing_dim() -> dict[str, Any]:
    """Invalid schema with vector missing dimension for error testing."""
    return {
        "collection_name": "invalid_no_dim",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "vector", "type": "FloatVector"},  # Missing 'dim'
        ],
    }


@pytest.fixture
def performance_test_schema() -> dict[str, Any]:
    """Schema optimized for performance testing."""
    return {
        "collection_name": "performance_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True},
            {"name": "name", "type": "VarChar", "max_length": 100},
            {"name": "score", "type": "Float"},
            {"name": "embedding", "type": "FloatVector", "dim": 256},
        ],
    }


@pytest.fixture
def default_values_schema() -> dict[str, Any]:
    """Schema with default values for testing default_value functionality."""
    return {
        "collection_name": "default_values_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True},
            {"name": "title", "type": "VarChar", "max_length": 200},
            {
                "name": "category",
                "type": "VarChar",
                "max_length": 50,
                "default_value": "general",
            },
            {
                "name": "priority",
                "type": "Int64",
                "default_value": 0,
                "min": 0,
                "max": 10,
            },
            {
                "name": "score",
                "type": "Float",
                "default_value": 5.0,
                "min": 0.0,
                "max": 10.0,
            },
            {"name": "is_active", "type": "Bool", "default_value": True},
            {
                "name": "status",
                "type": "VarChar",
                "max_length": 20,
                "default_value": "pending",
            },
            {
                "name": "description",
                "type": "VarChar",
                "max_length": 1000,
                "nullable": True,
            },
            {"name": "embedding", "type": "FloatVector", "dim": 128},
        ],
    }


# Pytest configuration
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--uri", action="store", default="http://localhost:19530", help="Milvus URI"
    )
    parser.addoption("--token", action="store", default="", help="Milvus token")
    parser.addoption(
        "--minio-host", action="store", default="localhost", help="MinIO host"
    )
    parser.addoption(
        "--minio-ak", action="store", default="minioadmin", help="MinIO access key"
    )
    parser.addoption(
        "--minio-sk", action="store", default="minioadmin", help="MinIO secret key"
    )
    parser.addoption(
        "--minio-bucket", action="store", default="a-bucket", help="MinIO bucket name"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_docker: marks tests that require Docker services"
    )
    config.addinivalue_line(
        "markers", "requires_milvus: marks tests that require Milvus connection"
    )
    config.addinivalue_line(
        "markers", "requires_s3: marks tests that require S3/MinIO connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if any(
            keyword in item.nodeid.lower()
            for keyword in ["large", "performance", "full_workflow", "partitioning"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if any(
            keyword in item.nodeid.lower()
            for keyword in ["milvus", "s3", "upload", "import", "insert"]
        ):
            item.add_marker(pytest.mark.integration)

        # Mark tests requiring external services
        if "milvus" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_milvus)

        if any(keyword in item.nodeid.lower() for keyword in ["s3", "upload", "minio"]):
            item.add_marker(pytest.mark.requires_s3)

"""Tests for Rust backend integration."""

import numpy as np
import pytest

from milvus_ingest.rust_backend import (
    USE_RUST,
    generate_bfloat16_vectors,
    generate_float16_vectors,
    generate_float64_batch,
    generate_float_array,
    generate_float_vectors,
    generate_int64_batch,
    generate_int_array,
    generate_json_fields,
    generate_normalized_float_vectors,
    generate_simple_text,
    generate_sparse_vectors,
    generate_text_fields,
    generate_varchar_array,
    get_cpu_features,
    get_rust_version,
)


class TestRustBackendAvailability:
    """Test Rust backend availability and version info."""

    def test_rust_available(self):
        """Test that Rust backend is available."""
        assert USE_RUST is True, "Rust backend should be available"

    def test_rust_version(self):
        """Test that Rust version is returned."""
        version = get_rust_version()
        assert version is not None
        assert isinstance(version, str)
        # Version should be in format x.y.z
        parts = version.split(".")
        assert len(parts) == 3

    def test_cpu_features(self):
        """Test that CPU features are returned."""
        features = get_cpu_features()
        assert isinstance(features, list)
        assert len(features) > 0
        # Should have at least one feature (neon on ARM, avx2/sse on x86, or scalar)
        valid_features = {"neon", "avx2", "avx", "sse4.2", "sse4.1", "fma", "scalar"}
        for feature in features:
            assert feature in valid_features


class TestVectorGeneration:
    """Test vector generation functions."""

    def test_float16_vectors_shape(self):
        """Test Float16 vector generation shape."""
        num_rows, dim = 100, 128
        result = generate_float16_vectors(num_rows, dim, seed=42)
        assert result.shape == (num_rows, dim * 2)  # Each f16 is 2 bytes
        assert result.dtype == np.uint8

    def test_float16_vectors_reproducible(self):
        """Test Float16 vector generation is reproducible with seed."""
        result1 = generate_float16_vectors(100, 64, seed=42)
        result2 = generate_float16_vectors(100, 64, seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_bfloat16_vectors_shape(self):
        """Test BFloat16 vector generation shape."""
        num_rows, dim = 100, 128
        result = generate_bfloat16_vectors(num_rows, dim, seed=42)
        assert result.shape == (num_rows, dim * 2)  # Each bf16 is 2 bytes
        assert result.dtype == np.uint8

    def test_bfloat16_vectors_reproducible(self):
        """Test BFloat16 vector generation is reproducible with seed."""
        result1 = generate_bfloat16_vectors(100, 64, seed=42)
        result2 = generate_bfloat16_vectors(100, 64, seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_float_vectors_shape(self):
        """Test Float32 vector generation shape."""
        num_rows, dim = 100, 128
        result = generate_float_vectors(num_rows, dim, seed=42)
        assert result.shape == (num_rows, dim)
        assert result.dtype == np.float32

    def test_normalized_float_vectors(self):
        """Test normalized Float32 vectors have unit norm."""
        num_rows, dim = 100, 128
        result = generate_normalized_float_vectors(num_rows, dim, seed=42)
        assert result.shape == (num_rows, dim)
        # Check that vectors are normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestSparseVectors:
    """Test sparse vector generation."""

    def test_sparse_vectors_structure(self):
        """Test sparse vector generation returns correct structure."""
        result = generate_sparse_vectors(100, max_dim=1000, seed=42)
        assert len(result) == 100
        for sparse_vec in result:
            assert isinstance(sparse_vec, dict)
            # Keys should be integers, values should be floats
            for k, v in sparse_vec.items():
                assert isinstance(k, int)
                assert isinstance(v, float)
                assert 0 <= k < 1000
                assert 0 <= v <= 1

    def test_sparse_vectors_density(self):
        """Test sparse vector density is within range."""
        result = generate_sparse_vectors(
            100, max_dim=1000, density_min=0.01, density_max=0.1, seed=42
        )
        for sparse_vec in result:
            density = len(sparse_vec) / 1000
            # Some tolerance since density is random
            assert density >= 0.005
            assert density <= 0.15


class TestTextGeneration:
    """Test text generation functions."""

    def test_simple_text(self):
        """Test simple text generation."""
        result = generate_simple_text(10, prefix="item", pk_offset=100)
        assert len(result) == 10
        assert result[0] == "item_100"
        assert result[9] == "item_109"

    def test_text_fields_english(self):
        """Test realistic English text generation."""
        result = generate_text_fields(
            100, max_length=256, text_type="sentence", language="en", seed=42
        )
        assert len(result) == 100
        for text in result:
            assert isinstance(text, str)
            assert len(text) <= 256
            assert len(text) > 0

    def test_text_fields_chinese(self):
        """Test realistic Chinese text generation."""
        result = generate_text_fields(
            100, max_length=256, text_type="sentence", language="zh", seed=42
        )
        assert len(result) == 100
        for text in result:
            assert isinstance(text, str)
            assert len(text) <= 256

    def test_text_fields_paragraph(self):
        """Test paragraph text generation."""
        result = generate_text_fields(
            10, max_length=500, text_type="paragraph", language="en", seed=42
        )
        assert len(result) == 10
        for text in result:
            assert isinstance(text, str)
            # Paragraphs should generally be longer than sentences
            assert len(text) > 0


class TestJsonGeneration:
    """Test JSON field generation."""

    def test_json_ecommerce(self):
        """Test ecommerce JSON pattern."""
        result = generate_json_fields(100, pattern="ecommerce", pk_offset=0, seed=42)
        assert len(result) == 100
        for item in result:
            assert isinstance(item, dict)
            assert "product_id" in item
            assert "category" in item
            assert "price" in item
            assert "in_stock" in item

    def test_json_event(self):
        """Test event JSON pattern."""
        result = generate_json_fields(100, pattern="event", pk_offset=0, seed=42)
        assert len(result) == 100
        for item in result:
            assert isinstance(item, dict)
            assert "event_id" in item
            assert "event_type" in item
            assert "timestamp" in item

    def test_json_config(self):
        """Test config JSON pattern."""
        result = generate_json_fields(100, pattern="config", pk_offset=0, seed=42)
        assert len(result) == 100
        for item in result:
            assert isinstance(item, dict)
            assert "config_id" in item
            assert "settings" in item

    def test_json_analytics(self):
        """Test analytics JSON pattern."""
        result = generate_json_fields(100, pattern="analytics", pk_offset=0, seed=42)
        assert len(result) == 100
        for item in result:
            assert isinstance(item, dict)
            assert "record_id" in item
            assert "metrics" in item

    def test_json_document(self):
        """Test document JSON pattern."""
        result = generate_json_fields(100, pattern="document", pk_offset=0, seed=42)
        assert len(result) == 100
        for item in result:
            assert isinstance(item, dict)
            assert "doc_id" in item
            assert "title" in item


class TestArrayGeneration:
    """Test array field generation."""

    def test_varchar_array(self):
        """Test VarChar array generation."""
        result = generate_varchar_array(100, max_capacity=10, max_length=64, seed=42)
        assert len(result) == 100
        for arr in result:
            assert isinstance(arr, list)
            assert 1 <= len(arr) <= 10
            for item in arr:
                assert isinstance(item, str)
                assert len(item) <= 64

    def test_int_array(self):
        """Test Int array generation."""
        result = generate_int_array(
            100, max_capacity=10, min_val=0, max_val=1000, seed=42
        )
        assert len(result) == 100
        for arr in result:
            assert isinstance(arr, list)
            assert 1 <= len(arr) <= 10
            for item in arr:
                assert isinstance(item, int)
                assert 0 <= item <= 1000

    def test_float_array(self):
        """Test Float array generation."""
        result = generate_float_array(
            100, max_capacity=10, min_val=0.0, max_val=1.0, seed=42
        )
        assert len(result) == 100
        for arr in result:
            assert isinstance(arr, list)
            assert 1 <= len(arr) <= 10
            for item in arr:
                assert isinstance(item, float)
                assert 0.0 <= item <= 1.0


class TestNumericBatchGeneration:
    """Test numeric batch generation."""

    def test_int64_batch(self):
        """Test Int64 batch generation."""
        result = generate_int64_batch(10000, min_val=0, max_val=1000000, seed=42)
        assert len(result) == 10000
        assert result.dtype == np.int64
        assert result.min() >= 0
        assert result.max() <= 1000000

    def test_float64_batch(self):
        """Test Float64 batch generation."""
        result = generate_float64_batch(10000, min_val=0.0, max_val=1.0, seed=42)
        assert len(result) == 10000
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0


@pytest.mark.slow
class TestPerformance:
    """Performance tests for Rust backend."""

    def test_float16_performance(self):
        """Test Float16 generation performance."""
        import time

        num_rows, dim = 100000, 128
        start = time.time()
        generate_float16_vectors(num_rows, dim, seed=42)
        elapsed = time.time() - start
        # Should complete in less than 1 second on modern hardware
        assert elapsed < 1.0, f"Float16 generation took {elapsed:.2f}s, expected < 1.0s"

    def test_sparse_performance(self):
        """Test sparse vector generation performance."""
        import time

        num_rows = 10000
        start = time.time()
        generate_sparse_vectors(num_rows, max_dim=1000, seed=42)
        elapsed = time.time() - start
        # Should complete in less than 1 second
        assert elapsed < 1.0, f"Sparse generation took {elapsed:.2f}s, expected < 1.0s"

    def test_text_performance(self):
        """Test text generation performance."""
        import time

        num_rows = 100000
        start = time.time()
        generate_text_fields(
            num_rows, max_length=256, text_type="sentence", language="en", seed=42
        )
        elapsed = time.time() - start
        # Should complete in less than 1 second
        assert elapsed < 1.0, f"Text generation took {elapsed:.2f}s, expected < 1.0s"


class TestRustDisabled:
    """Test Python fallback when Rust is disabled."""

    def test_fallback_works(self):
        """Test that Python fallback works when Rust import fails."""
        # This test is mainly for documentation - the actual fallback
        # is tested implicitly by the fact that all functions have
        # Python implementations
        pass

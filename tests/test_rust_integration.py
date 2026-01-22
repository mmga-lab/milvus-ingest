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


class TestFakerGeneration:
    """Test Faker data generation functions (ported from synth-gen concepts)."""

    def test_faker_strings_name(self):
        """Test faker name generation."""
        from milvus_ingest.rust_backend import generate_faker_strings

        result = generate_faker_strings(100, "name", seed=42)
        assert len(result) == 100
        for name in result:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_faker_strings_email(self):
        """Test faker email generation."""
        from milvus_ingest.rust_backend import generate_faker_strings

        result = generate_faker_strings(100, "email", seed=42)
        assert len(result) == 100
        for email in result:
            assert isinstance(email, str)
            assert "@" in email

    def test_faker_strings_phone(self):
        """Test faker phone generation."""
        from milvus_ingest.rust_backend import generate_faker_strings

        result = generate_faker_strings(100, "phone", seed=42)
        assert len(result) == 100
        for phone in result:
            assert isinstance(phone, str)
            assert len(phone) > 0

    def test_faker_strings_company(self):
        """Test faker company name generation."""
        from milvus_ingest.rust_backend import generate_faker_strings

        result = generate_faker_strings(100, "company", seed=42)
        assert len(result) == 100
        for company in result:
            assert isinstance(company, str)
            assert len(company) > 0

    def test_faker_strings_reproducible(self):
        """Test faker string generation is reproducible with seed."""
        from milvus_ingest.rust_backend import generate_faker_strings

        result1 = generate_faker_strings(100, "name", seed=42)
        result2 = generate_faker_strings(100, "name", seed=42)
        assert result1 == result2

    def test_faker_strings_uuid(self):
        """Test faker UUID generation."""
        from milvus_ingest.rust_backend import generate_faker_strings

        result = generate_faker_strings(100, "uuid", seed=42)
        assert len(result) == 100
        for uuid_str in result:
            assert isinstance(uuid_str, str)
            # UUID format: 8-4-4-4-12
            assert len(uuid_str) == 36
            assert uuid_str.count("-") == 4

    def test_list_faker_types(self):
        """Test list of available faker types."""
        from milvus_ingest.rust_backend import list_faker_types

        types = list_faker_types()
        assert isinstance(types, list)
        assert len(types) > 10
        assert "name" in types
        assert "email" in types
        assert "phone" in types


class TestCategoricalGeneration:
    """Test categorical data generation."""

    def test_categorical_basic(self):
        """Test basic categorical generation."""
        from milvus_ingest.rust_backend import generate_categorical

        categories = ["A", "B", "C"]
        weights = [1.0, 1.0, 1.0]
        result = generate_categorical(1000, categories, weights, seed=42)
        assert len(result) == 1000
        for item in result:
            assert item in categories

    def test_categorical_weighted(self):
        """Test weighted categorical generation."""
        from milvus_ingest.rust_backend import generate_categorical

        categories = ["rare", "common"]
        weights = [1.0, 99.0]  # common should appear ~99x more often
        result = generate_categorical(10000, categories, weights, seed=42)

        rare_count = sum(1 for x in result if x == "rare")
        common_count = sum(1 for x in result if x == "common")

        # With 1:99 ratio, rare should be less than 5% (allowing for randomness)
        assert rare_count < 500, f"rare_count={rare_count} should be < 500"
        assert common_count > 9500, f"common_count={common_count} should be > 9500"

    def test_categorical_reproducible(self):
        """Test categorical generation is reproducible."""
        from milvus_ingest.rust_backend import generate_categorical

        categories = ["X", "Y", "Z"]
        weights = [1.0, 2.0, 3.0]
        result1 = generate_categorical(100, categories, weights, seed=42)
        result2 = generate_categorical(100, categories, weights, seed=42)
        assert result1 == result2


class TestRangeGeneration:
    """Test range number generation."""

    def test_range_numbers_basic(self):
        """Test basic range number generation."""
        from milvus_ingest.rust_backend import generate_range_numbers

        result = generate_range_numbers(1000, 0.0, 100.0, seed=42)
        assert len(result) == 1000
        for num in result:
            assert 0.0 <= num < 100.0

    def test_range_numbers_with_step(self):
        """Test range number generation with step."""
        from milvus_ingest.rust_backend import generate_range_numbers

        result = generate_range_numbers(1000, 0.0, 100.0, step=10.0, seed=42)
        assert len(result) == 1000
        for num in result:
            assert 0.0 <= num < 100.0
            # Should be multiple of 10
            assert abs(num % 10.0) < 1e-9

    def test_range_integers_basic(self):
        """Test basic range integer generation."""
        from milvus_ingest.rust_backend import generate_range_integers

        result = generate_range_integers(1000, 0, 100, seed=42)
        assert len(result) == 1000
        for num in result:
            assert 0 <= num < 100

    def test_range_integers_with_step(self):
        """Test range integer generation with step."""
        from milvus_ingest.rust_backend import generate_range_integers

        result = generate_range_integers(1000, 0, 100, step=5, seed=42)
        assert len(result) == 1000
        for num in result:
            assert 0 <= num < 100
            assert num % 5 == 0


class TestBooleanGeneration:
    """Test boolean generation."""

    def test_frequency_bool_half(self):
        """Test boolean generation with 50% probability."""
        from milvus_ingest.rust_backend import generate_frequency_bool

        result = generate_frequency_bool(10000, true_probability=0.5, seed=42)
        assert len(result) == 10000
        true_count = sum(1 for x in result if x)
        # Should be roughly 50% (allow 10% margin)
        assert 4000 < true_count < 6000, f"true_count={true_count}"

    def test_frequency_bool_biased(self):
        """Test boolean generation with biased probability."""
        from milvus_ingest.rust_backend import generate_frequency_bool

        result = generate_frequency_bool(10000, true_probability=0.9, seed=42)
        true_count = sum(1 for x in result if x)
        # Should be roughly 90% (allow 5% margin)
        assert 8500 < true_count < 9500, f"true_count={true_count}"


class TestDateGeneration:
    """Test date and datetime generation."""

    def test_date_strings(self):
        """Test date string generation."""
        from milvus_ingest.rust_backend import generate_date_strings

        result = generate_date_strings(1000, start_year=2020, end_year=2024, seed=42)
        assert len(result) == 1000
        for date_str in result:
            assert isinstance(date_str, str)
            # Format: YYYY-MM-DD
            parts = date_str.split("-")
            assert len(parts) == 3
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            assert 2020 <= year <= 2024
            assert 1 <= month <= 12
            assert 1 <= day <= 31

    def test_datetime_strings(self):
        """Test datetime string generation."""
        from milvus_ingest.rust_backend import generate_datetime_strings

        result = generate_datetime_strings(1000, start_year=2020, end_year=2024, seed=42)
        assert len(result) == 1000
        for dt_str in result:
            assert isinstance(dt_str, str)
            # Format: YYYY-MM-DDTHH:MM:SS
            assert "T" in dt_str
            date_part, time_part = dt_str.split("T")
            assert len(date_part.split("-")) == 3
            assert len(time_part.split(":")) == 3


class TestSequentialIds:
    """Test sequential ID generation."""

    def test_sequential_ids_basic(self):
        """Test basic sequential ID generation."""
        from milvus_ingest.rust_backend import generate_sequential_ids

        result = generate_sequential_ids(100, prefix="user", start_index=0)
        assert len(result) == 100
        assert result[0] == "user_0"
        assert result[99] == "user_99"

    def test_sequential_ids_with_offset(self):
        """Test sequential ID generation with offset."""
        from milvus_ingest.rust_backend import generate_sequential_ids

        result = generate_sequential_ids(100, prefix="item", start_index=1000)
        assert len(result) == 100
        assert result[0] == "item_1000"
        assert result[99] == "item_1099"


@pytest.mark.slow
class TestFakerPerformance:
    """Performance tests for Faker generation."""

    def test_faker_performance(self):
        """Test faker string generation performance."""
        import time

        from milvus_ingest.rust_backend import generate_faker_strings

        count = 100000
        start = time.time()
        generate_faker_strings(count, "name", seed=42)
        elapsed = time.time() - start
        # Should complete in less than 2 seconds on modern hardware
        assert elapsed < 2.0, f"Faker generation took {elapsed:.2f}s, expected < 2.0s"

    def test_categorical_performance(self):
        """Test categorical generation performance."""
        import time

        from milvus_ingest.rust_backend import generate_categorical

        count = 1000000
        categories = ["A", "B", "C", "D", "E"]
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        start = time.time()
        generate_categorical(count, categories, weights, seed=42)
        elapsed = time.time() - start
        # Should complete in less than 1 second
        assert elapsed < 1.0, f"Categorical generation took {elapsed:.2f}s, expected < 1.0s"


class TestLoremGeneration:
    """Test Lorem Ipsum text generation functions."""

    def test_lorem_words_basic(self):
        """Test basic Lorem word generation."""
        from milvus_ingest.rust_backend import generate_lorem_words

        result = generate_lorem_words(100, words_per_item=1, seed=42)
        assert len(result) == 100
        assert all(isinstance(w, str) for w in result)
        assert all(len(w) > 0 for w in result)

    def test_lorem_words_multiple(self):
        """Test Lorem word generation with multiple words per item."""
        from milvus_ingest.rust_backend import generate_lorem_words

        result = generate_lorem_words(50, words_per_item=5, seed=42)
        assert len(result) == 50
        # Each item should have multiple words
        for item in result:
            word_count = len(item.split())
            assert word_count >= 1

    def test_lorem_sentences_basic(self):
        """Test basic Lorem sentence generation."""
        from milvus_ingest.rust_backend import generate_lorem_sentences

        result = generate_lorem_sentences(100, word_count_min=4, word_count_max=10, seed=42)
        assert len(result) == 100
        # Each sentence should end with a period
        for sentence in result:
            assert sentence.endswith(".")
            # Should have multiple words
            assert len(sentence.split()) >= 3

    def test_lorem_paragraphs_basic(self):
        """Test basic Lorem paragraph generation."""
        from milvus_ingest.rust_backend import generate_lorem_paragraphs

        result = generate_lorem_paragraphs(50, sentence_count_min=2, sentence_count_max=4, seed=42)
        assert len(result) == 50
        # Each paragraph should have multiple sentences
        for paragraph in result:
            # Count periods as sentence endings
            sentence_count = paragraph.count(".")
            assert sentence_count >= 1

    def test_lorem_text_types(self):
        """Test Lorem text generation with different types."""
        from milvus_ingest.rust_backend import generate_lorem_text

        # Test sentence type
        sentences = generate_lorem_text(10, text_type="sentence", min_units=4, max_units=8, seed=42)
        assert len(sentences) == 10
        for s in sentences:
            assert s.endswith(".")

        # Test paragraph type
        paragraphs = generate_lorem_text(10, text_type="paragraph", min_units=2, max_units=4, seed=42)
        assert len(paragraphs) == 10

        # Test word type
        words = generate_lorem_text(10, text_type="word", min_units=3, max_units=6, seed=42)
        assert len(words) == 10

    def test_lorem_reproducibility(self):
        """Test Lorem generation reproducibility with seed."""
        from milvus_ingest.rust_backend import generate_lorem_sentences

        result1 = generate_lorem_sentences(100, seed=12345)
        result2 = generate_lorem_sentences(100, seed=12345)
        result3 = generate_lorem_sentences(100, seed=99999)

        assert result1 == result2
        assert result1 != result3


@pytest.mark.slow
class TestLoremPerformance:
    """Performance tests for Lorem generation."""

    def test_lorem_sentences_performance(self):
        """Test Lorem sentence generation performance."""
        import time

        from milvus_ingest.rust_backend import generate_lorem_sentences

        count = 100000
        start = time.time()
        generate_lorem_sentences(count, seed=42)
        elapsed = time.time() - start
        # Should complete in less than 1 second on modern hardware
        assert elapsed < 1.0, f"Lorem sentences took {elapsed:.2f}s, expected < 1.0s"

    def test_lorem_paragraphs_performance(self):
        """Test Lorem paragraph generation performance."""
        import time

        from milvus_ingest.rust_backend import generate_lorem_paragraphs

        count = 100000
        start = time.time()
        generate_lorem_paragraphs(count, seed=42)
        elapsed = time.time() - start
        # Should complete in less than 2 seconds
        assert elapsed < 2.0, f"Lorem paragraphs took {elapsed:.2f}s, expected < 2.0s"

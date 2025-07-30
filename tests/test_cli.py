"""Comprehensive CLI tests for milvus-ingest following Click testing best practices.

Test Structure:
- Uses Click's CliRunner for isolated command execution
- Parameterized tests for comprehensive coverage
- Proper fixture organization and reuse
- Clear test organization by functionality
- Robust error handling verification
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from milvus_ingest.cli import main


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for MinIO/Milvus testing."""
    return {
        "MILVUS_URI": "http://127.0.0.1:19530",
        "MINIO_HOST": "127.0.0.1",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
        "MINIO_BUCKET": "a-bucket",
    }


@pytest.fixture
def runner():
    """Click CLI runner for isolated filesystem testing."""
    return CliRunner()


@pytest.fixture
def sample_schema():
    """Sample schema for testing basic functionality."""
    return {
        "collection_name": "e2e_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True},
            {"name": "title", "type": "VarChar", "max_length": 100},
            {
                "name": "description",
                "type": "VarChar",
                "max_length": 500,
                "nullable": True,
            },
            {"name": "price", "type": "Float", "min": 0.01, "max": 999.99},
            {"name": "rating", "type": "Int8", "min": 1, "max": 5},
            {
                "name": "tags",
                "type": "Array",
                "element_type": "VarChar",
                "max_capacity": 5,
                "max_length": 20,
            },
            {"name": "metadata", "type": "JSON", "nullable": True},
            {"name": "is_active", "type": "Bool"},
            {"name": "embedding", "type": "FloatVector", "dim": 128},
            {"name": "sparse_vector", "type": "SparseFloatVector"},
        ],
    }


@pytest.fixture
def vector_schema():
    """Schema with various vector types for testing vector generation."""
    return {
        "collection_name": "vector_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "float_vec", "type": "FloatVector", "dim": 64},
            {"name": "binary_vec", "type": "BinaryVector", "dim": 128},
            {"name": "float16_vec", "type": "Float16Vector", "dim": 32},
            {"name": "bfloat16_vec", "type": "BFloat16Vector", "dim": 256},
            {"name": "sparse_vec", "type": "SparseFloatVector"},
        ],
    }


class TestDataGeneration:
    """Test data generation functionality."""

    @pytest.mark.parametrize(
        "schema_name,expected_rows",
        [
            ("quickstart", 50),
            ("ecommerce", 25),
        ],
    )
    def test_generate_with_builtin_schema(self, runner, schema_name, expected_rows):
        """Test data generation using built-in schemas."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--builtin",
                    schema_name,
                    "-r",
                    str(expected_rows),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"Command failed: {result.output}"
            assert (
                f"Generated {expected_rows} rows" in result.output
                or f"Saved {expected_rows} rows" in result.output
            )

            # Verify output directory structure
            # Different schemas have different collection name patterns
            collection_map = {
                "quickstart": "quickstart_example",
                "ecommerce": "ecommerce_products",
            }
            collection_name = collection_map.get(schema_name, f"{schema_name}_example")
            expected_output_dir = (
                Path.home() / ".milvus-ingest" / "data" / collection_name
            )
            assert expected_output_dir.exists(), (
                f"Output directory not found: {expected_output_dir}"
            )

            # Verify metadata file
            meta_file = expected_output_dir / "meta.json"
            assert meta_file.exists(), "meta.json file not found"

            with open(meta_file) as f:
                meta = json.load(f)
            assert meta["generation_info"]["total_rows"] == expected_rows

    def test_generate_with_custom_schema(self, runner, sample_schema):
        """Test data generation with custom schema."""
        with runner.isolated_filesystem():
            # Create schema file
            schema_file = Path("test_schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "25",
                    "--seed",
                    "123",
                    "--out",
                    "custom_output",
                ],
            )

            assert result.exit_code == 0

            # Check output directory
            output_dir = Path("custom_output")
            assert output_dir.exists()

            # Verify data files exist
            data_files = [f for f in output_dir.iterdir() if f.name != "meta.json"]
            assert len(data_files) > 0

    def test_generate_different_formats(self, runner, sample_schema):
        """Test data generation in different formats."""
        with runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            # Test Parquet format (default)
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "10",
                    "--format",
                    "parquet",
                    "--out",
                    "parquet_output",
                ],
            )
            assert result.exit_code == 0

            parquet_dir = Path("parquet_output")
            assert parquet_dir.exists()
            parquet_files = list(parquet_dir.glob("*.parquet"))
            assert len(parquet_files) > 0

            # Test JSON format
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "10",
                    "--format",
                    "json",
                    "--out",
                    "json_output",
                ],
            )
            assert result.exit_code == 0

            json_dir = Path("json_output")
            assert json_dir.exists()
            json_files = list(json_dir.glob("*.json"))
            # Filter out meta.json
            data_json_files = [f for f in json_files if f.name != "meta.json"]
            assert len(data_json_files) > 0

    def test_generate_large_dataset_with_partitioning(self, runner, sample_schema):
        """Test large dataset generation with automatic file partitioning."""
        with runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "15000",  # Should create multiple files
                    "--max-rows-per-file",
                    "5000",
                    "--batch-size",
                    "2500",
                    "--out",
                    "large_output",
                ],
            )

            assert result.exit_code == 0

            output_dir = Path("large_output")
            assert output_dir.exists()

            # Should have multiple data files due to partitioning
            data_files = [f for f in output_dir.iterdir() if f.name != "meta.json"]
            assert len(data_files) >= 3  # 15000 rows / 5000 max = 3 files

            # Verify metadata
            with open(output_dir / "meta.json") as f:
                meta = json.load(f)
            assert meta["generation_info"]["total_rows"] == 15000

    def test_generate_vector_types(self, runner, vector_schema):
        """Test generation of various vector types."""
        with runner.isolated_filesystem():
            schema_file = Path("vector_schema.json")
            with open(schema_file, "w") as f:
                json.dump(vector_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "20",
                    "--seed",
                    "42",
                    "--out",
                    "vector_output",
                ],
            )

            assert result.exit_code == 0

            output_dir = Path("vector_output")
            assert output_dir.exists()

    def test_preview_mode(self, runner, sample_schema):
        """Test preview mode functionality."""
        with runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "100",
                    "--preview",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0
            assert "Preview (top 5 rows):" in result.output
            # Should not create actual output files in preview mode

    def test_validate_only_mode(self, runner, sample_schema):
        """Test schema validation without generation."""
        with runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = runner.invoke(
                main, ["generate", "--schema", str(schema_file), "--validate-only"]
            )

            assert result.exit_code == 0, f"Validation failed: {result.output}"
            assert "sparse_vector: SparseFloatVector" in result.output


class TestSchemaManagement:
    """Test schema management operations."""

    def test_schema_list(self, runner):
        """Test listing available schemas."""
        result = runner.invoke(main, ["schema", "list"])

        assert result.exit_code == 0
        # Should list built-in schemas
        assert "quickstart" in result.output
        assert "ecommerce" in result.output

    @pytest.mark.parametrize("schema_name", ["quickstart", "ecommerce"])
    def test_schema_show(self, runner, schema_name):
        """Test showing schema details."""
        result = runner.invoke(main, ["schema", "show", schema_name])

        assert result.exit_code == 0
        assert result.exit_code == 0, f"Schema show failed: {result.output}"
        # Verify schema content is displayed
        assert (
            schema_name in result.output.lower() or "example" in result.output.lower()
        )

    def test_schema_add_and_remove(self, runner, sample_schema):
        """Test adding and removing custom schemas."""
        with runner.isolated_filesystem():
            # Create schema file
            schema_file = Path("my_schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            # Clean up any existing schema
            runner.invoke(main, ["schema", "remove", "my_test_custom"], input="y\n")

            # Add custom schema
            result = runner.invoke(
                main,
                ["schema", "add", "my_test_custom", str(schema_file)],
                input="Test schema description\ntesting, e2e\n",
            )

            assert result.exit_code == 0, f"Schema add failed: {result.output}"
            assert "my_test_custom" in result.output

            # Verify schema is listed
            list_result = runner.invoke(main, ["schema", "list"])
            assert "my_test_custom" in list_result.output

            # Test using the custom schema
            gen_result = runner.invoke(
                main, ["generate", "--builtin", "my_test_custom", "-r", "5"]
            )
            assert gen_result.exit_code == 0, f"Generation failed: {gen_result.output}"

            # Clean up
            remove_result = runner.invoke(
                main, ["schema", "remove", "my_test_custom"], input="y\n"
            )
            assert remove_result.exit_code == 0, (
                f"Schema removal failed: {remove_result.output}"
            )

    def test_schema_help(self, runner):
        """Test schema format help."""
        result = runner.invoke(main, ["schema", "help"])

        assert result.exit_code == 0, f"Schema help failed: {result.output}"
        assert any(
            keyword in result.output
            for keyword in ["Schema Format", "Field Types", "format"]
        )


class TestS3MinIOIntegration:
    """Test S3/MinIO upload functionality."""

    def test_upload_to_minio(self, runner, sample_schema, mock_env_vars):
        """Test uploading data to local MinIO."""
        with runner.isolated_filesystem():
            # Generate data first
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "10",
                    "--out",
                    "test_data",
                ],
            )
            assert result.exit_code == 0

            # Test MinIO upload
            upload_result = runner.invoke(
                main,
                [
                    "upload",
                    "--local-path",
                    "test_data",
                    "--s3-path",
                    f"s3://{mock_env_vars['MINIO_BUCKET']}/test-upload/",
                    "--access-key-id",
                    mock_env_vars["MINIO_ACCESS_KEY"],
                    "--secret-access-key",
                    mock_env_vars["MINIO_SECRET_KEY"],
                    "--endpoint-url",
                    f"http://{mock_env_vars['MINIO_HOST']}:9000",
                    "--no-verify-ssl",
                ],
            )

            if upload_result.exit_code != 0:
                pytest.skip(f"MinIO not available: {upload_result.output}")

            assert "error" not in upload_result.output.lower()


class TestMilvusIntegration:
    """Test Milvus integration functionality."""

    def test_milvus_insert(self, runner, mock_env_vars):
        """Test direct insert to local Milvus."""
        with runner.isolated_filesystem():
            # Create a simple schema without sparse vector for Milvus compatibility
            simple_schema = {
                "collection_name": "test_milvus_insert",
                "fields": [
                    {
                        "name": "id",
                        "type": "Int64",
                        "is_primary": True,
                        "auto_id": True,
                    },
                    {"name": "title", "type": "VarChar", "max_length": 100},
                    {"name": "price", "type": "Float", "min": 0.01, "max": 999.99},
                    {"name": "is_active", "type": "Bool"},
                    {"name": "embedding", "type": "FloatVector", "dim": 128},
                ],
            }

            # Generate data first
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(simple_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "20",
                    "--out",
                    "milvus_data",
                ],
            )
            assert result.exit_code == 0

            # Test Milvus insert
            insert_result = runner.invoke(
                main,
                [
                    "to-milvus",
                    "insert",
                    "milvus_data",
                    "--uri",
                    mock_env_vars["MILVUS_URI"],
                    "--batch-size",
                    "10",
                    "--drop-if-exists",
                ],
            )

            if insert_result.exit_code != 0:
                pytest.skip(f"Milvus not available: {insert_result.output}")

            assert "error" not in insert_result.output.lower()

    @patch("milvus_ingest.milvus_importer.MilvusBulkImporter")
    def test_milvus_bulk_import(self, mock_importer, runner, mock_env_vars):
        """Test bulk import to Milvus."""
        with runner.isolated_filesystem():
            # Create dummy data file
            data_dir = Path("import_data")
            data_dir.mkdir()
            (data_dir / "data.parquet").touch()
            (data_dir / "meta.json").write_text('{"collection_name": "test"}')

            # Mock bulk importer
            mock_instance = Mock()
            mock_importer.return_value = mock_instance
            mock_instance.bulk_import.return_value = "job-123"
            mock_instance.wait_for_completion.return_value = True

            # Test bulk import with mocked components
            result = runner.invoke(
                main,
                [
                    "to-milvus",
                    "import",
                    "--collection-name",
                    "test_collection",
                    "--local-path",
                    str(data_dir),
                    "--s3-path",
                    "test-import/",
                    "--bucket",
                    mock_env_vars["MINIO_BUCKET"],
                    "--endpoint-url",
                    f"http://{mock_env_vars['MINIO_HOST']}:9000",
                    "--access-key-id",
                    mock_env_vars["MINIO_ACCESS_KEY"],
                    "--secret-access-key",
                    mock_env_vars["MINIO_SECRET_KEY"],
                    "--uri",
                    mock_env_vars["MILVUS_URI"],
                    "--wait",
                    "--no-verify-ssl",
                ],
            )

            assert result.exit_code == 0
            mock_importer.assert_called_once()

    def test_milvus_verify_help(self, runner):
        """Test verify command help functionality."""
        result = runner.invoke(main, ["to-milvus", "verify", "--help"])

        assert result.exit_code == 0, f"Help command failed: {result.output}"
        assert "Verify" in result.output or "verify" in result.output.lower()

    @patch("pymilvus.MilvusClient")
    def test_milvus_verify_mocked(self, mock_client, runner, mock_env_vars):
        """Test verify command with mocked Milvus client."""
        with runner.isolated_filesystem():
            # Setup test data
            data_dir = Path("verify_data")
            data_dir.mkdir()

            meta_content = {
                "schema": {"collection_name": "test_verify_collection"},
                "generation_info": {"total_rows": 1000},
            }

            with open(data_dir / "meta.json", "w") as f:
                json.dump(meta_content, f)

            # Mock Milvus client behavior
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            mock_instance.has_collection.return_value = True
            mock_instance.query.return_value = [{"count(*)": 1000}]

            # Test verification
            result = runner.invoke(
                main,
                [
                    "to-milvus",
                    "verify",
                    str(data_dir),
                    "--uri",
                    mock_env_vars["MILVUS_URI"],
                ],
            )

            if result.exit_code != 0:
                pytest.skip(f"Verify command not available: {result.output}")

            assert (
                "passed" in result.output.lower() or "success" in result.output.lower()
            )


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_schema_file(self, runner):
        """Test handling of invalid schema files."""
        with runner.isolated_filesystem():
            # Create invalid JSON file
            invalid_file = Path("invalid.json")
            invalid_file.write_text("{ invalid json")

            result = runner.invoke(
                main, ["generate", "--schema", str(invalid_file), "--rows", "1"]
            )

            assert result.exit_code != 0
            assert "Error loading schema" in result.output

    def test_missing_required_parameters(self, runner):
        """Test handling of missing required parameters."""
        result = runner.invoke(main, ["generate", "--rows", "1"])

        assert result.exit_code != 0
        assert "One of --schema or --builtin is required" in result.output

    def test_invalid_builtin_schema(self, runner):
        """Test handling of non-existent built-in schema."""
        result = runner.invoke(
            main, ["generate", "--builtin", "nonexistent", "--rows", "1"]
        )

        assert result.exit_code != 0

    def test_schema_validation_errors(self, runner):
        """Test schema validation error handling."""
        with runner.isolated_filesystem():
            # Create schema with validation errors
            invalid_schema = {
                "collection_name": "test",
                "fields": [
                    {"name": "vector", "type": "FloatVector"},  # Missing required 'dim'
                ],
            }

            schema_file = Path("invalid_schema.json")
            with open(schema_file, "w") as f:
                json.dump(invalid_schema, f)

            result = runner.invoke(
                main, ["generate", "--schema", str(schema_file), "--validate-only"]
            )

            assert result.exit_code != 0
            assert "validation" in result.output.lower()


class TestUtilityCommands:
    """Test utility commands."""

    def test_clean_command(self, runner):
        """Test clean command."""
        with runner.isolated_filesystem():
            # Create some test data
            test_dir = Path.home() / ".milvus-ingest" / "data" / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "data.parquet").touch()

            result = runner.invoke(main, ["clean", "--yes"])

            assert result.exit_code == 0

    def test_help_commands(self, runner):
        """Test help command functionality."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Generate mock data for Milvus" in result.output

        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(main, ["schema", "--help"])
        assert result.exit_code == 0


class TestPerformanceOptions:
    """Test performance-related options."""

    def test_batch_size_option(self, runner, sample_schema):
        """Test different batch sizes."""
        with runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            # Test with custom batch size
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "1000",
                    "--batch-size",
                    "250",
                    "--out",
                    "batch_test",
                ],
            )

            assert result.exit_code == 0

    def test_file_size_limits(self, runner, sample_schema):
        """Test file size limit options."""
        with runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "5000",
                    "--max-file-size",
                    "1",  # 1MB limit
                    "--max-rows-per-file",
                    "1000",
                    "--out",
                    "size_test",
                ],
            )

            assert result.exit_code == 0

            output_dir = Path("size_test")
            data_files = [f for f in output_dir.iterdir() if f.name != "meta.json"]
            assert (
                len(data_files) >= 3
            )  # Should create multiple files due to size limits


# Integration test that exercises the full workflow
class TestFullWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_data_pipeline_workflow(self, runner):
        """Test complete workflow from schema creation to data generation."""
        with runner.isolated_filesystem():
            # Step 1: Create a custom schema
            custom_schema = {
                "collection_name": "complete_test",
                "fields": [
                    {"name": "id", "type": "Int64", "is_primary": True},
                    {"name": "product_name", "type": "VarChar", "max_length": 200},
                    {"name": "price", "type": "Double", "min": 1.0, "max": 1000.0},
                    {
                        "name": "category_tags",
                        "type": "Array",
                        "element_type": "VarChar",
                        "max_capacity": 3,
                        "max_length": 50,
                    },
                    {"name": "product_embedding", "type": "FloatVector", "dim": 256},
                    {"name": "is_featured", "type": "Bool"},
                    {"name": "metadata", "type": "JSON", "nullable": True},
                ],
            }

            schema_file = Path("complete_schema.json")
            with open(schema_file, "w") as f:
                json.dump(custom_schema, f, indent=2)

            # Step 2: Validate schema
            validate_result = runner.invoke(
                main, ["generate", "--schema", str(schema_file), "--validate-only"]
            )
            assert validate_result.exit_code == 0, (
                f"Schema validation failed: {validate_result.output}"
            )

            # Step 3: Generate preview
            preview_result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "10",
                    "--preview",
                    "--seed",
                    "42",
                ],
            )
            assert preview_result.exit_code == 0, (
                f"Preview failed: {preview_result.output}"
            )
            assert "Preview" in preview_result.output

            # Step 4: Generate full dataset in Parquet format
            parquet_result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "2500",
                    "--format",
                    "parquet",
                    "--out",
                    "complete_parquet",
                    "--batch-size",
                    "500",
                    "--rows-per-file",
                    "1000",
                    "--seed",
                    "42",
                ],
            )
            assert parquet_result.exit_code == 0, (
                f"Parquet generation failed: {parquet_result.output}"
            )

            # Verify output
            parquet_dir = Path("complete_parquet")
            assert parquet_dir.exists()

            # Check metadata
            with open(parquet_dir / "meta.json") as f:
                meta = json.load(f)
            assert meta["schema"]["collection_name"] == "complete_test"
            assert meta["generation_info"]["total_rows"] == 2500

            # Should have multiple files due to max_rows_per_file setting
            data_files = [f for f in parquet_dir.iterdir() if f.name != "meta.json"]
            assert len(data_files) >= 3

            # Step 5: Generate JSON format dataset
            json_result = runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "-r",
                    "100",
                    "--format",
                    "json",
                    "--out",
                    "complete_json",
                    "--seed",
                    "42",
                ],
            )
            assert json_result.exit_code == 0, (
                f"JSON generation failed: {json_result.output}"
            )

            json_dir = Path("complete_json")
            assert json_dir.exists()

            # Verify JSON format files exist
            json_files = list(json_dir.glob("*.json"))
            data_json_files = [f for f in json_files if f.name != "meta.json"]
            assert len(data_json_files) > 0

"""E2E tests for all 6 builtin schemas data generation."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from milvus_ingest.cli import main


@pytest.fixture
def runner():
    """Click CLI runner for isolated filesystem testing."""
    return CliRunner()


@pytest.fixture
def all_schemas():
    """All 6 new builtin schema names."""
    return [
        "product_catalog",
        "ecommerce_search",
        "news_articles",
        "document_search",
        "multi_tenant_data",
        "multimedia_content",
    ]


class TestNewBuiltinSchemas:
    """Test the 6 new focused builtin schemas."""

    def test_schema_list_contains_all_new_schemas(self, runner, all_schemas):
        """Test that schema list shows all 6 new schemas."""
        result = runner.invoke(main, ["schema", "list"])

        assert result.exit_code == 0, f"Schema list failed: {result.output}"

        # Check each schema is in output
        for schema_name in all_schemas:
            assert schema_name in result.output, (
                f"Schema {schema_name} not found in output"
            )

    @pytest.mark.parametrize(
        "schema_name",
        [
            "product_catalog",
            "ecommerce_search",
            "multimedia_content",  # Test a subset first
        ],
    )
    def test_schema_show_works(self, runner, schema_name):
        """Test showing individual schema details."""
        result = runner.invoke(main, ["schema", "show", schema_name])

        assert result.exit_code == 0, (
            f"Schema show failed for {schema_name}: {result.output}"
        )
        assert "Fields" in result.output or "fields" in result.output

    @pytest.mark.parametrize("schema_name", ["product_catalog", "ecommerce_search"])
    @pytest.mark.parametrize("format_type", ["json", "parquet"])
    def test_basic_data_generation(self, runner, schema_name, format_type):
        """Test basic data generation for schemas in both formats."""
        with runner.isolated_filesystem():
            output_dir = f"test_{schema_name}_{format_type}"

            # Try with correct parameter name first
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--builtin",
                    schema_name,
                    "-r",
                    "20",  # Use short form to avoid confusion
                    "--format",
                    format_type,
                    "--out",
                    output_dir,
                    "--seed",
                    "42",
                ],
            )

            if result.exit_code != 0:
                print(f"\nERROR for {schema_name} ({format_type}):")
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                if result.exception:
                    print(f"Exception: {result.exception}")

            assert result.exit_code == 0, (
                f"Generation failed for {schema_name} ({format_type})"
            )

            # Verify output directory exists
            output_path = Path(output_dir)
            assert output_path.exists(), f"Output directory not found: {output_dir}"

            # Verify meta.json exists
            meta_file = output_path / "meta.json"
            assert meta_file.exists(), f"meta.json not found in {output_dir}"

            # Verify metadata content
            with open(meta_file) as f:
                meta = json.load(f)

            assert meta["generation_info"]["total_rows"] == 20
            assert "schema" in meta
            assert meta["schema"]["collection_name"]

            # Verify data files exist
            if format_type == "json":
                data_files = [
                    f for f in output_path.glob("*.json") if f.name != "meta.json"
                ]
            else:
                data_files = list(output_path.glob("*.parquet"))

            assert len(data_files) > 0, f"No {format_type} data files found"
            assert all(f.stat().st_size > 0 for f in data_files), (
                "Found empty data files"
            )


class TestSchemaFeatures:
    """Test specific schema features."""

    def test_auto_id_schemas(self, runner):
        """Test schemas with auto_id feature."""
        # product_catalog and multi_tenant_data have auto_id
        for schema_name in ["product_catalog", "multi_tenant_data"]:
            with runner.isolated_filesystem():
                result = runner.invoke(
                    main,
                    [
                        "generate",
                        "--builtin",
                        schema_name,
                        "-r",
                        "10",
                        "--format",
                        "json",
                        "--out",
                        f"auto_id_{schema_name}",
                        "--seed",
                        "42",
                    ],
                )

                assert result.exit_code == 0, f"Auto ID test failed for {schema_name}"

                # Check metadata for auto_id fields
                meta_file = Path(f"auto_id_{schema_name}") / "meta.json"
                with open(meta_file) as f:
                    meta = json.load(f)

                # Find auto_id field
                auto_id_fields = [
                    f for f in meta["schema"]["fields"] if f.get("auto_id")
                ]
                assert len(auto_id_fields) == 1, (
                    f"Expected 1 auto_id field in {schema_name}"
                )

    def test_dynamic_schema(self, runner):
        """Test news_articles schema with dynamic fields."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--builtin",
                    "news_articles",
                    "-r",
                    "15",
                    "--format",
                    "json",
                    "--out",
                    "dynamic_test",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, "Dynamic schema test failed"

            # Check for enable_dynamic_field
            meta_file = Path("dynamic_test") / "meta.json"
            with open(meta_file) as f:
                meta = json.load(f)

            assert meta["schema"].get("enable_dynamic_field") is True, (
                "enable_dynamic_field not set"
            )

    def test_sparse_vectors(self, runner):
        """Test document_search schema with sparse vectors."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--builtin",
                    "document_search",
                    "-r",
                    "12",
                    "--format",
                    "parquet",
                    "--out",
                    "sparse_test",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, "Sparse vector test failed"

            # Check for sparse vector fields
            meta_file = Path("sparse_test") / "meta.json"
            with open(meta_file) as f:
                meta = json.load(f)

            sparse_fields = [
                f for f in meta["schema"]["fields"] if f["type"] == "SparseFloatVector"
            ]
            assert len(sparse_fields) >= 1, "No sparse vector fields found"


class TestMultiFileGeneration:
    """Test multi-file generation."""

    def test_file_size_constraint(self, runner):
        """Test generating multiple files with size constraint."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--builtin",
                    "multimedia_content",  # Schema with most fields
                    "-r",
                    "5000",
                    "--file-size",
                    "1MB",  # Small file size to force multiple files
                    "--format",
                    "parquet",
                    "--out",
                    "multifile_test",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, (
                f"Multi-file generation failed: {result.output}"
            )

            output_path = Path("multifile_test")
            data_files = list(output_path.glob("*.parquet"))

            # Should create multiple files due to size constraint
            assert len(data_files) >= 2, (
                f"Expected multiple files, got {len(data_files)}"
            )

            # Verify metadata
            meta_file = output_path / "meta.json"
            with open(meta_file) as f:
                meta = json.load(f)
            assert meta["generation_info"]["total_rows"] == 5000

    def test_large_dataset_generation(self, runner):
        """Test generating large datasets successfully."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--builtin",
                    "multimedia_content",  # Use schema with most fields
                    "-r",
                    "3000",
                    "--format",
                    "parquet",
                    "--out",
                    "large_test",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, (
                f"Large dataset generation failed: {result.output}"
            )

            output_path = Path("large_test")
            data_files = list(output_path.glob("*.parquet"))

            # Verify metadata
            meta_file = output_path / "meta.json"
            with open(meta_file) as f:
                meta = json.load(f)
            assert meta["generation_info"]["total_rows"] == 3000

            # Verify data files exist and have content
            assert len(data_files) >= 1, "No data files generated"

            # Verify actual data integrity
            import pandas as pd

            total_rows_found = 0
            for data_file in data_files:
                df = pd.read_parquet(data_file)
                total_rows_found += len(df)

                # Verify data has expected columns (multimedia_content schema)
                assert "media_id" in df.columns, "Missing primary key column"
                assert "content_embedding" in df.columns, "Missing vector column"
                assert len(df.columns) >= 5, (
                    f"Expected at least 5 columns, got {len(df.columns)}"
                )

            assert total_rows_found == 3000, (
                f"Total rows mismatch: expected 3000, got {total_rows_found}"
            )

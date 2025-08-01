"""Lightweight file validation for generated data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from rich.console import Console
from rich.table import Table

from .exceptions import MilvusIngestError


class LightweightValidator:
    """Lightweight validator for generated data files."""

    def __init__(self, output_dir: Path, console: Console | None = None):
        """Initialize validator with output directory.

        Args:
            output_dir: Directory containing generated files and meta.json
            console: Rich console for output (optional)
        """
        self.output_dir = output_dir
        self.console = console or Console()
        self.meta_file = output_dir / "meta.json"
        self.metadata = None
        self.schema = None
        self.generation_info = None

    def validate(self, sample_rows: int = 100) -> dict[str, Any]:
        """Run lightweight validation on generated files.

        Args:
            sample_rows: Number of rows to sample from each file

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_checks": {},
            "schema_checks": {},
            "summary": {},
        }

        try:
            # 1. Check meta.json exists and is valid
            if not self.meta_file.exists():
                results["valid"] = False
                results["errors"].append(f"meta.json not found in {self.output_dir}")
                return results

            # Load metadata
            try:
                with open(self.meta_file) as f:
                    self.metadata = json.load(f)
                    self.schema = self.metadata.get("schema", {})
                    self.generation_info = self.metadata.get("generation_info", {})
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Failed to parse meta.json: {e}")
                return results

            # 2. Verify all expected files exist
            expected_files = self.generation_info.get("data_files", [])
            if not expected_files:
                results["warnings"].append("No data files listed in meta.json")
                return results

            existing_files = []
            missing_files = []

            for file_name in expected_files:
                file_path = self.output_dir / file_name
                if file_path.exists():
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_name)
                    results["valid"] = False
                    results["errors"].append(f"Expected file not found: {file_name}")

            results["file_checks"]["expected"] = len(expected_files)
            results["file_checks"]["found"] = len(existing_files)
            results["file_checks"]["missing"] = missing_files

            # 3. Validate each existing file
            total_rows_verified = 0
            total_size_bytes = 0
            file_format = self.generation_info.get("format", "parquet")

            for file_path in existing_files:
                file_result = self._validate_file(file_path, file_format, sample_rows)
                results["file_checks"][file_path.name] = file_result

                if not file_result["valid"]:
                    results["valid"] = False
                    results["errors"].extend(file_result.get("errors", []))

                total_rows_verified += file_result.get("row_count", 0)
                total_size_bytes += file_result.get("size_bytes", 0)

            # 4. Verify total row count matches expectation
            expected_rows = self.generation_info.get("total_rows", 0)
            if expected_rows > 0:
                row_diff = abs(total_rows_verified - expected_rows)
                if row_diff > 0:
                    results["warnings"].append(
                        f"Row count mismatch: expected {expected_rows:,}, found {total_rows_verified:,}"
                    )
                    if row_diff > expected_rows * 0.01:  # More than 1% difference
                        results["valid"] = False
                        results["errors"].append(
                            f"Significant row count mismatch (>{1}% difference)"
                        )

            # 5. Summary statistics
            results["summary"] = {
                "total_files": len(existing_files),
                "total_rows": total_rows_verified,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
                "expected_rows": expected_rows,
                "format": file_format,
            }

            # Log results
            if results["valid"]:
                logger.info(
                    f"Validation passed: {len(existing_files)} files, "
                    f"{total_rows_verified:,} rows, "
                    f"{results['summary']['total_size_mb']:.2f} MB"
                )
            else:
                logger.error(
                    f"Validation failed with {len(results['errors'])} errors"
                )

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {e}")
            logger.error(f"Validation failed: {e}")

        return results

    def _validate_file(
        self, file_path: Path, file_format: str, sample_rows: int
    ) -> dict[str, Any]:
        """Validate a single data file.

        Args:
            file_path: Path to the file
            file_format: Expected format (parquet or json)
            sample_rows: Number of rows to sample

        Returns:
            Dictionary with file validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "row_count": 0,
            "size_bytes": 0,
            "schema_valid": True,
            "samples_valid": True,
        }

        try:
            # Get file size
            result["size_bytes"] = file_path.stat().st_size

            # Validate based on format
            if file_format.lower() == "parquet":
                result.update(self._validate_parquet_file(file_path, sample_rows))
            elif file_format.lower() == "json":
                result.update(self._validate_json_file(file_path, sample_rows))
            else:
                result["valid"] = False
                result["errors"].append(f"Unsupported format: {file_format}")

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"File validation error: {e}")

        return result

    def _validate_parquet_file(
        self, file_path: Path, sample_rows: int
    ) -> dict[str, Any]:
        """Validate a parquet file."""
        result = {"parquet_metadata": {}}

        try:
            # Read parquet metadata
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata

            result["row_count"] = metadata.num_rows
            result["parquet_metadata"] = {
                "num_row_groups": metadata.num_row_groups,
                "columns": metadata.num_columns,
                "format_version": str(metadata.format_version),
            }

            # Sample first and last rows
            df = pd.read_parquet(file_path)

            # Basic integrity check
            if len(df) != metadata.num_rows:
                result["valid"] = False
                result["errors"].append(
                    f"Row count mismatch: metadata says {metadata.num_rows}, "
                    f"but DataFrame has {len(df)}"
                )

            # Validate schema if we have field definitions
            if self.schema and "fields" in self.schema:
                schema_errors = self._validate_dataframe_schema(df)
                if schema_errors:
                    result["schema_valid"] = False
                    result["errors"].extend(schema_errors)

            # Sample validation
            sample_size = min(sample_rows, len(df))
            if sample_size > 0:
                # Check first N rows
                head_sample = df.head(sample_size // 2)
                # Check last N rows
                tail_sample = df.tail(sample_size // 2)

                # Validate samples don't have critical issues
                for sample_df in [head_sample, tail_sample]:
                    # Check for NaN in non-nullable fields
                    for field in self.schema.get("fields", []):
                        field_name = field["name"]
                        if field_name in sample_df.columns:
                            if not field.get("nullable", False):
                                if sample_df[field_name].isna().any():
                                    result["samples_valid"] = False
                                    result["errors"].append(
                                        f"Found NaN in non-nullable field: {field_name}"
                                    )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Parquet validation error: {e}")

        return result

    def _validate_json_file(self, file_path: Path, sample_rows: int) -> dict[str, Any]:
        """Validate a JSON file."""
        result = {}

        try:
            # For JSON files, we expect an array of objects
            with open(file_path) as f:
                data = json.load(f)

            if not isinstance(data, list):
                result["valid"] = False
                result["errors"].append("JSON file is not an array of objects")
                return result

            result["row_count"] = len(data)

            # Sample validation
            sample_size = min(sample_rows, len(data))
            if sample_size > 0:
                # Check first and last samples
                samples = data[:sample_size // 2] + data[-(sample_size // 2):]

                # Basic validation
                for i, record in enumerate(samples):
                    if not isinstance(record, dict):
                        result["samples_valid"] = False
                        result["errors"].append(f"Record {i} is not a dictionary")
                        break

                # Schema validation if available
                if self.schema and "fields" in self.schema and samples:
                    # Convert to DataFrame for easier validation
                    try:
                        df = pd.DataFrame(samples)
                        schema_errors = self._validate_dataframe_schema(df)
                        if schema_errors:
                            result["schema_valid"] = False
                            result["errors"].extend(schema_errors)
                    except Exception as e:
                        result["errors"].append(f"Failed to parse JSON samples: {e}")

        except json.JSONDecodeError as e:
            result["valid"] = False
            result["errors"].append(f"Invalid JSON format: {e}")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"JSON validation error: {e}")

        return result

    def _validate_dataframe_schema(self, df: pd.DataFrame) -> list[str]:
        """Validate DataFrame against schema definition."""
        errors = []
        fields = {field["name"]: field for field in self.schema.get("fields", [])}

        # Check for missing required fields
        for field_name, field_def in fields.items():
            if field_def.get("name") == "id" and field_def.get("auto_id", False):
                # Skip auto-generated ID fields
                continue

            if field_name not in df.columns:
                # Only error if field is not nullable and has no default
                if not field_def.get("nullable", False) and "default_value" not in field_def:
                    errors.append(f"Missing required field: {field_name}")

        # Validate data types for existing columns
        for col in df.columns:
            if col in fields:
                field_def = fields[col]
                field_type = field_def["type"]

                # Basic type validation
                try:
                    if field_type in ["Int8", "Int16", "Int32", "Int64"]:
                        if not pd.api.types.is_integer_dtype(df[col]):
                            errors.append(f"Field {col} should be integer, got {df[col].dtype}")
                    elif field_type == "Float":
                        if not pd.api.types.is_float_dtype(df[col]):
                            errors.append(f"Field {col} should be float, got {df[col].dtype}")
                    elif field_type == "Bool":
                        if not pd.api.types.is_bool_dtype(df[col]):
                            errors.append(f"Field {col} should be boolean, got {df[col].dtype}")
                    elif field_type in ["VarChar", "String"]:
                        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                            errors.append(f"Field {col} should be string, got {df[col].dtype}")
                except Exception as e:
                    errors.append(f"Type validation error for {col}: {e}")

        return errors

    def display_results(self, results: dict[str, Any]) -> None:
        """Display validation results in a formatted table."""
        if results["valid"]:
            self.console.print("\n✅ [bold green]Validation Passed[/bold green]")
        else:
            self.console.print("\n❌ [bold red]Validation Failed[/bold red]")

        # Summary table
        table = Table(title="Validation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        summary = results.get("summary", {})
        table.add_row("Total Files", f"{summary.get('total_files', 0)}")
        table.add_row("Total Rows", f"{summary.get('total_rows', 0):,}")
        table.add_row("Expected Rows", f"{summary.get('expected_rows', 0):,}")
        table.add_row("Total Size", f"{summary.get('total_size_mb', 0):.2f} MB")
        table.add_row("Format", summary.get("format", "unknown"))

        self.console.print(table)

        # Errors
        if results.get("errors"):
            self.console.print("\n[bold red]Errors:[/bold red]")
            for error in results["errors"]:
                self.console.print(f"  • {error}")

        # Warnings
        if results.get("warnings"):
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in results["warnings"]:
                self.console.print(f"  • {warning}")

        # File details
        if results.get("file_checks"):
            file_checks = results["file_checks"]
            if isinstance(file_checks.get("expected"), int):
                self.console.print(
                    f"\n[dim]Files: {file_checks.get('found', 0)}/{file_checks.get('expected', 0)} found[/dim]"
                )
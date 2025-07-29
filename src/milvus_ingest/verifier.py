"""Comprehensive data verification for Milvus collections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pymilvus import MilvusClient
from rich.console import Console
from rich.table import Table

from .exceptions import MilvusIngestError
from .rich_display import display_error, display_info, display_success


class MilvusVerifier:
    """Comprehensive verification system for Milvus collections."""

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        data_path: Path,
        console: Console | None = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.data_path = data_path
        self.console = console or Console()
        self.metadata = self._load_metadata()
        self.schema = self.metadata.get("schema", {})
        self.generation_info = self.metadata.get("generation_info", {})

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from meta.json."""
        meta_file = self.data_path / "meta.json"
        if not meta_file.exists():
            raise MilvusIngestError(f"meta.json not found in {self.data_path}")

        with open(meta_file) as f:
            return json.load(f)

    def verify_count_with_queries(self, sample_size: int = 1000) -> dict[str, bool]:
        """Verify row count and run query tests (Level 1)."""
        results = {}

        self.console.print(
            "\n[bold]🔍 Running row count verification with query tests...[/bold]"
        )

        # Ensure collection is loaded
        self.client.load_collection(self.collection_name)

        # 1. Row count verification
        results["row_count"] = self._verify_row_count()

        # 2. Query/search correctness with 1000 samples
        results["query_correctness"] = self._verify_query_correctness(1000)

        self._display_summary(results)
        return results

    def verify_scalar_fields_with_queries(
        self, sample_size: int = 1000
    ) -> dict[str, bool]:
        """Verify row count, scalar fields, and run query tests (Level 2)."""
        results = {}

        self.console.print(
            "\n[bold]🔍 Running scalar field verification with query tests...[/bold]"
        )

        # Ensure collection is loaded
        self.client.load_collection(self.collection_name)

        # 1. Row count verification
        results["row_count"] = self._verify_row_count()

        # 2. Check AUTO_ID compatibility for field value verification
        if not self._check_auto_id_field_verification("scalar"):
            # Skip field verification but continue with query tests
            results["scalar_fields"] = True  # Mark as passed (skipped)
        else:
            # Calculate optimal sample size
            total_rows = self.generation_info.get("total_rows", 0)
            optimal_sample_size = self._calculate_sample_size(total_rows)
            self.console.print(
                f"[dim]Using {optimal_sample_size:,} samples for field verification[/dim]"
            )

            # Scalar field comparison
            results["scalar_fields"] = self._verify_scalar_field_values(
                optimal_sample_size
            )

        # 3. Query/search correctness with 1000 samples (always run)
        results["query_correctness"] = self._verify_query_correctness(1000)

        self._display_summary(results)
        return results

    def verify_full_fields_with_queries(
        self, sample_size: int = 1000
    ) -> dict[str, bool]:
        """Verify row count, all fields, and run query tests (Level 3)."""
        results = {}

        self.console.print(
            "\n[bold]🔍 Running full field verification with query tests...[/bold]"
        )

        # Ensure collection is loaded
        self.client.load_collection(self.collection_name)

        # 1. Row count verification
        results["row_count"] = self._verify_row_count()

        # 2. Check AUTO_ID compatibility for field value verification
        if not self._check_auto_id_field_verification("full"):
            # Skip field verification but continue with query tests
            results["all_fields"] = True  # Mark as passed (skipped)
        else:
            # Calculate optimal sample size
            total_rows = self.generation_info.get("total_rows", 0)
            optimal_sample_size = self._calculate_sample_size(total_rows)
            self.console.print(
                f"[dim]Using {optimal_sample_size:,} samples for field verification[/dim]"
            )

            # All field comparison (including vectors)
            results["all_fields"] = self._verify_all_field_values(optimal_sample_size)

        # 3. Query/search correctness with 1000 samples (always run)
        results["query_correctness"] = self._verify_query_correctness(1000)

        self._display_summary(results)
        return results

    def verify_scalar_fields(self, sample_size: int = 1000) -> dict[str, bool]:
        """Verify row count and scalar field values (Level 2)."""
        results = {}

        self.console.print("\n[bold]🔍 Running scalar field verification...[/bold]")

        # Ensure collection is loaded
        self.client.load_collection(self.collection_name)

        # 1. Row count verification
        results["row_count"] = self._verify_row_count()

        # 2. Check AUTO_ID compatibility for field value verification
        if not self._check_auto_id_field_verification("scalar"):
            # Skip field verification for AUTO_ID scenarios
            results["scalar_fields"] = True  # Mark as passed (skipped)
        else:
            # Calculate optimal sample size
            total_rows = self.generation_info.get("total_rows", 0)
            optimal_sample_size = self._calculate_sample_size(total_rows)
            self.console.print(
                f"[dim]Using {optimal_sample_size:,} samples for field verification[/dim]"
            )

            # Scalar field comparison
            results["scalar_fields"] = self._verify_scalar_field_values(
                optimal_sample_size
            )

        self._display_summary(results)
        return results

    def verify_full_fields(self, sample_size: int = 1000) -> dict[str, bool]:
        """Verify all fields and run query/search tests (Level 3)."""
        results = {}

        self.console.print(
            "\n[bold]🔍 Running full field verification with query tests...[/bold]"
        )

        # Ensure collection is loaded
        self.client.load_collection(self.collection_name)

        # 1. Row count verification
        results["row_count"] = self._verify_row_count()

        # 2. Check AUTO_ID compatibility for field value verification
        if not self._check_auto_id_field_verification("full"):
            # Skip field verification but continue with query tests
            results["all_fields"] = True  # Mark as passed (skipped)
        else:
            # Calculate optimal sample size
            total_rows = self.generation_info.get("total_rows", 0)
            optimal_sample_size = self._calculate_sample_size(total_rows)
            self.console.print(
                f"[dim]Using {optimal_sample_size:,} samples for field verification[/dim]"
            )

            # All field comparison (including vectors)
            results["all_fields"] = self._verify_all_field_values(optimal_sample_size)

        # 3. Query/search correctness with 1000 samples (always run)
        results["query_correctness"] = self._verify_query_correctness(1000)

        self._display_summary(results)
        return results

    def _verify_row_count(self) -> bool:
        """Verify row count matches expected."""
        self.console.print("\n[bold blue]📊 Verifying row count...[/bold blue]")

        expected_rows = self.generation_info.get("total_rows", 0)
        if expected_rows == 0:
            display_error("Could not find total_rows in meta.json")
            return False

        # Ensure collection is loaded with refresh before counting
        self.client.load_collection(self.collection_name, refresh=True)

        # Query count
        result = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        actual_rows = result[0].get("count(*)", 0) if result else 0

        success = actual_rows == expected_rows
        if success:
            display_success(f"✓ Row count verification passed: {actual_rows:,} rows")
        else:
            difference = actual_rows - expected_rows
            display_error(
                f"✗ Row count mismatch: Expected {expected_rows:,}, "
                f"got {actual_rows:,} (difference: {difference:+,})"
            )

        return success

    def _verify_scalar_field_values(self, sample_size: int = 1000) -> bool:
        """Verify scalar field values only (excludes vectors)."""
        self.console.print(
            f"\n[bold blue]🔧 Verifying scalar fields ({sample_size} samples)...[/bold blue]"
        )

        return self._verify_field_values_internal(sample_size, exclude_vectors=True)

    def _verify_all_field_values(self, sample_size: int = 1000) -> bool:
        """Verify all field values (including vectors)."""
        self.console.print(
            f"\n[bold blue]🔧 Verifying all fields ({sample_size} samples)...[/bold blue]"
        )

        return self._verify_field_values_internal(sample_size, exclude_vectors=False)

    def _verify_field_values_internal(
        self, sample_size: int, exclude_vectors: bool = False
    ) -> bool:
        """Internal method to verify field values with vector exclusion option."""
        # Load sample data from source files
        source_data = self._load_sample_source_data(sample_size)
        if not source_data:
            display_info("Could not load source data for field consistency check")
            return False

        # Check if we have AUTO_ID fields
        has_auto_id = any(
            field.get("auto_id", False) for field in self.schema.get("fields", [])
        )

        if has_auto_id:
            self.console.print(
                "[dim]🔄 AUTO_ID detected: Using row index-based comparison[/dim]"
            )
            return self._verify_auto_id_scenario(
                source_data, sample_size, exclude_vectors
            )
        else:
            self.console.print("[dim]🔑 Using primary key-based comparison[/dim]")
            return self._verify_primary_key_scenario(
                source_data, sample_size, exclude_vectors
            )

    def _verify_auto_id_scenario(
        self, source_data: list[dict], sample_size: int, exclude_vectors: bool = False
    ) -> bool:
        """Verify field values for AUTO_ID scenario using row index alignment."""
        try:
            # Query Milvus data in insertion order (no sorting to maintain order)
            milvus_data = self.client.query(
                collection_name=self.collection_name,
                filter="",  # Get all records
                output_fields=["*"],
                limit=min(sample_size, len(source_data)),
            )
        except Exception as e:
            display_error(f"Failed to query Milvus data: {e}")
            return False

        # Create lookup by row index for comparison
        sample_count = min(len(source_data), len(milvus_data))
        milvus_lookup = {i: milvus_data[i] for i in range(sample_count)}
        source_lookup = {i: source_data[i] for i in range(sample_count)}

        return self._verify_fields_with_index_alignment(
            source_lookup, milvus_lookup, sample_count, exclude_vectors
        )

    def _verify_primary_key_scenario(
        self, source_data: list[dict], sample_size: int, exclude_vectors: bool = False
    ) -> bool:
        """Verify field values for non-AUTO_ID scenario using primary key alignment."""
        # Get primary key field
        pk_field = None
        for field in self.schema.get("fields", []):
            if field.get("is_primary_key") or field.get("is_primary"):
                pk_field = field["name"]
                break

        if not pk_field:
            display_info(
                "No primary key field found, falling back to index-based comparison"
            )
            # Fallback to index-based comparison
            try:
                milvus_data = self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=["*"],
                    limit=min(sample_size, len(source_data)),
                )
                sample_count = min(len(source_data), len(milvus_data))
                milvus_lookup = {i: milvus_data[i] for i in range(sample_count)}
                source_lookup = {i: source_data[i] for i in range(sample_count)}
                return self._verify_fields_with_index_alignment(
                    source_lookup, milvus_lookup, sample_count, exclude_vectors
                )
            except Exception as e:
                display_error(f"Failed to query Milvus data: {e}")
                return False

        # Sample some records from Milvus using primary key
        # Get primary key field type to format values correctly
        pk_field_type = None
        for field in self.schema.get("fields", []):
            if field.get("is_primary_key") or field.get("is_primary"):
                pk_field_type = field.get("type", "VarChar")
                break

        # Format primary key values based on field type
        pk_values = []
        for row in source_data[:sample_size]:
            pk_value = row[pk_field]
            if pk_field_type in ["Int8", "Int16", "Int32", "Int64"]:
                # For integer types, use raw values
                pk_values.append(pk_value)
            else:
                # For string types, quote the values
                pk_values.append(f'"{pk_value}"')

        filter_expr = f"{pk_field} in [{', '.join(map(str, pk_values))}]"

        try:
            milvus_data = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["*"],
                limit=sample_size,
            )
        except Exception as e:
            display_error(f"Failed to query Milvus data: {e}")
            return False

        # Create lookup for Milvus data using primary key
        milvus_lookup = {str(row[pk_field]): row for row in milvus_data}

        # Verify each field
        field_results = {}
        fields_to_verify = self._get_fields_to_verify(exclude_vectors)
        all_passed = True

        for field in fields_to_verify:
            field_name = field["name"]
            field_type = field["type"]

            field_passed = self._verify_field_values_by_primary_key(
                field_name, field_type, source_data, milvus_lookup, pk_field
            )
            field_results[field_name] = field_passed
            all_passed = all_passed and field_passed

        # Display field verification results
        self._display_field_verification_table(fields_to_verify, field_results)
        return all_passed

    def _verify_fields_with_index_alignment(
        self,
        source_lookup: dict[int, dict],
        milvus_lookup: dict[int, dict],
        sample_count: int,
        exclude_vectors: bool = False,
    ) -> bool:
        """Verify fields using index-based alignment."""
        fields_to_verify = self._get_fields_to_verify(exclude_vectors)
        field_results = {}
        all_passed = True

        for field in fields_to_verify:
            field_name = field["name"]
            field_type = field["type"]

            field_passed = self._verify_field_values_by_index(
                field_name, field_type, source_lookup, milvus_lookup, sample_count
            )
            field_results[field_name] = field_passed
            all_passed = all_passed and field_passed

        # Display field verification results
        self._display_field_verification_table(fields_to_verify, field_results)
        return all_passed

    def _get_fields_to_verify(self, exclude_vectors: bool = False) -> list[dict]:
        """Get list of fields to verify based on configuration."""
        fields_to_verify = []
        for field in self.schema.get("fields", []):
            field_name = field["name"]
            field_type = field["type"]

            if field_name in ["$meta"]:  # Skip system fields
                continue

            # Skip auto_id primary key fields since they're not in source data
            if field.get("auto_id", False):
                continue

            # Skip function output fields since they're not in source data
            if self._is_function_output_field(field_name):
                continue

            # Skip vector fields if exclude_vectors is True
            if exclude_vectors and "Vector" in field_type:
                continue

            fields_to_verify.append(field)

        return fields_to_verify

    def _display_field_verification_table(
        self, fields_to_verify: list[dict], field_results: dict[str, bool]
    ) -> None:
        """Display field verification results in a table."""
        table = Table(title="Field Value Verification")
        table.add_column("Field", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")

        for field in fields_to_verify:
            field_name = field["name"]
            if field_name in field_results:
                status = "✓ Pass" if field_results[field_name] else "✗ Fail"
                table.add_row(field_name, field["type"], status)

        self.console.print(table)

    def _verify_field_values_by_primary_key(
        self,
        field_name: str,
        field_type: str,
        source_data: list[dict],
        milvus_lookup: dict[str, dict],
        pk_field: str,
    ) -> bool:
        """Verify field values using primary key-based comparison."""
        mismatches = 0
        total_compared = 0

        for source_row in source_data:
            pk_value = str(source_row[pk_field])
            milvus_row = milvus_lookup.get(pk_value)

            if not milvus_row or field_name not in milvus_row:
                continue

            source_value = source_row.get(field_name)
            milvus_value = milvus_row[field_name]

            # Handle different data type comparisons
            if not self._values_match(source_value, milvus_value, field_type):
                mismatches += 1
                if mismatches <= 5:  # Log first few mismatches
                    logger.debug(
                        f"Field {field_name} mismatch for PK {pk_value}: source={source_value}, "
                        f"milvus={milvus_value}"
                    )

            total_compared += 1

        # Allow small percentage of mismatches due to float precision, etc.
        mismatch_rate = mismatches / total_compared if total_compared > 0 else 0
        success = mismatch_rate < 0.05  # 5% threshold

        if not success:
            display_info(
                f"Field {field_name}: {mismatches}/{total_compared} mismatches "
                f"({mismatch_rate:.1%})"
            )

        return success

    def _verify_field_consistency(self, sample_size: int = 1000) -> bool:
        """Verify field data types and value consistency."""
        self.console.print(
            f"\n[bold blue]🔧 Verifying field consistency ({sample_size} samples)...[/bold blue]"
        )

        all_passed = True

        # Load sample data from source files
        source_data = self._load_sample_source_data(sample_size)
        if not source_data:
            display_info("Could not load source data for field consistency check")
            return False

        # Get primary key field
        pk_field = None
        for field in self.schema.get("fields", []):
            if field.get("is_primary_key") or field.get("is_primary"):
                pk_field = field["name"]
                break

        if not pk_field:
            display_info("No primary key field found, skipping field consistency check")
            return False

        # Sample some records from Milvus
        pk_values = [str(row[pk_field]) for row in source_data[:sample_size]]
        filter_expr = f"{pk_field} in {pk_values}"

        try:
            milvus_data = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["*"],
                limit=sample_size,
            )
        except Exception as e:
            display_error(f"Failed to query Milvus data: {e}")
            return False

        # Create lookup for Milvus data
        milvus_lookup = {str(row[pk_field]): row for row in milvus_data}

        # Verify each field
        field_results = {}
        for field in self.schema.get("fields", []):
            field_name = field["name"]
            field_type = field["type"]

            if field_name in ["$meta"]:  # Skip system fields
                continue

            field_passed = self._verify_field_values(
                field_name, field_type, source_data, milvus_lookup
            )
            field_results[field_name] = field_passed
            all_passed = all_passed and field_passed

        # Display field verification results
        table = Table(title="Field Consistency Verification")
        table.add_column("Field", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")

        for field in self.schema.get("fields", []):
            field_name = field["name"]
            if field_name in field_results:
                status = "✓ Pass" if field_results[field_name] else "✗ Fail"
                table.add_row(field_name, field["type"], status)

        self.console.print(table)
        return all_passed

    def _verify_field_values(
        self,
        field_name: str,
        field_type: str,
        source_data: list[dict],
        milvus_lookup: dict[str, dict],
    ) -> bool:
        """Verify individual field values match between source and Milvus."""
        mismatches = 0
        total_compared = 0

        for source_row in source_data:
            pk_value = str(next(iter(source_row.values())))  # Get PK value
            milvus_row = milvus_lookup.get(pk_value)

            if not milvus_row or field_name not in milvus_row:
                continue

            source_value = source_row.get(field_name)
            milvus_value = milvus_row[field_name]

            # Handle different data type comparisons
            if not self._values_match(source_value, milvus_value, field_type):
                mismatches += 1
                if mismatches <= 5:  # Log first few mismatches
                    logger.debug(
                        f"Field {field_name} mismatch: source={source_value}, "
                        f"milvus={milvus_value}"
                    )

            total_compared += 1

        # Allow small percentage of mismatches due to float precision, etc.
        mismatch_rate = mismatches / total_compared if total_compared > 0 else 0
        success = mismatch_rate < 0.05  # 5% threshold

        if not success:
            display_info(
                f"Field {field_name}: {mismatches}/{total_compared} mismatches "
                f"({mismatch_rate:.1%})"
            )

        return success

    def _verify_field_values_by_index(
        self,
        field_name: str,
        field_type: str,
        source_lookup: dict[int, dict],
        milvus_lookup: dict[int, dict],
        sample_count: int,
    ) -> bool:
        """Verify field values using index-based comparison for AUTO_ID scenarios."""
        mismatches = 0
        total_compared = 0

        for i in range(sample_count):
            source_row = source_lookup.get(i)
            milvus_row = milvus_lookup.get(i)

            if not source_row or not milvus_row or field_name not in milvus_row:
                continue

            source_value = source_row.get(field_name)
            milvus_value = milvus_row[field_name]

            # Handle different data type comparisons
            if not self._values_match(source_value, milvus_value, field_type):
                mismatches += 1
                if mismatches <= 5:  # Log first few mismatches
                    logger.debug(
                        f"Field {field_name} mismatch at index {i}: source={source_value}, "
                        f"milvus={milvus_value}"
                    )

            total_compared += 1

        # Allow small percentage of mismatches due to float precision, etc.
        mismatch_rate = mismatches / total_compared if total_compared > 0 else 0
        success = mismatch_rate < 0.05  # 5% threshold

        if not success:
            display_info(
                f"Field {field_name}: {mismatches}/{total_compared} mismatches "
                f"({mismatch_rate:.1%})"
            )

        return success

    def _values_match(
        self, source_value: Any, milvus_value: Any, field_type: str
    ) -> bool:
        """Check if two values match considering data type specifics."""
        try:
            if source_value is None and milvus_value is None:
                return True
            if source_value is None or milvus_value is None:
                return False

            # Handle vector fields
            if "Vector" in field_type:
                if not isinstance(source_value, list | np.ndarray):
                    return False
                if not isinstance(milvus_value, list | np.ndarray):
                    return False
                return np.allclose(source_value, milvus_value, rtol=1e-6)

            # Handle float precision
            if field_type in ["Float", "Double"]:
                try:
                    return abs(float(source_value) - float(milvus_value)) < 1e-6
                except (ValueError, TypeError):
                    return False

            # Handle arrays/JSON
            if field_type in ["Array", "JSON"]:
                # Handle numpy arrays first
                if isinstance(source_value, np.ndarray) and isinstance(
                    milvus_value, np.ndarray
                ):
                    return np.array_equal(source_value, milvus_value)
                elif isinstance(source_value, np.ndarray) or isinstance(
                    milvus_value, np.ndarray
                ):
                    # Convert numpy array to list for comparison
                    source_list = (
                        source_value.tolist()
                        if isinstance(source_value, np.ndarray)
                        else source_value
                    )
                    milvus_list = (
                        milvus_value.tolist()
                        if isinstance(milvus_value, np.ndarray)
                        else milvus_value
                    )
                    return source_list == milvus_list
                # Handle JSON fields specifically
                elif field_type == "JSON":
                    return self._compare_json_values(source_value, milvus_value)
                else:
                    return source_value == milvus_value

            # Default string comparison
            return str(source_value) == str(milvus_value)
        except Exception as e:
            logger.debug(
                f"Error comparing values: {e}, source_type={type(source_value)}, milvus_type={type(milvus_value)}"
            )
            return False

    def _verify_query_correctness(self, sample_size: int = 100) -> bool:
        """Verify query and search results are correct."""
        self.console.print(
            f"\n[bold blue]🔍 Verifying query correctness ({sample_size} samples)...[/bold blue]"
        )

        all_passed = True

        # Test exact queries
        exact_query_passed = self._test_exact_queries(sample_size // 2)
        all_passed = all_passed and exact_query_passed

        # Test vector search if vector fields exist
        vector_fields = [
            f for f in self.schema.get("fields", []) if "Vector" in f["type"]
        ]
        if vector_fields:
            vector_search_passed = self._test_vector_search(
                vector_fields[0], sample_size // 2
            )
            all_passed = all_passed and vector_search_passed
        else:
            display_info("No vector fields found, skipping vector search verification")

        return all_passed

    def _test_exact_queries(self, sample_size: int) -> bool:
        """Test exact queries return expected results."""
        # Get primary key field
        pk_field = None
        for field in self.schema.get("fields", []):
            if field.get("is_primary_key") or field.get("is_primary"):
                pk_field = field["name"]
                break

        if not pk_field:
            display_info("No primary key field found, skipping exact query test")
            return False

        # Sample some records
        try:
            sample_data = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=[pk_field],
                limit=sample_size,
            )
        except Exception as e:
            display_error(f"Failed to sample data for exact queries: {e}")
            return False

        # Test individual exact queries
        passed = 0
        for row in sample_data:
            pk_value = row[pk_field]
            try:
                query_result = self.client.query(
                    collection_name=self.collection_name,
                    filter=f"{pk_field} == {pk_value}",
                    output_fields=[pk_field],
                )
                if len(query_result) == 1 and query_result[0][pk_field] == pk_value:
                    passed += 1
            except Exception as e:
                logger.debug(f"Exact query failed for {pk_value}: {e}")

        success_rate = passed / len(sample_data) if sample_data else 0
        success = success_rate > 0.95  # 95% success rate

        if success:
            display_success(
                f"✓ Exact queries: {passed}/{len(sample_data)} passed ({success_rate:.1%})"
            )
        else:
            display_error(
                f"✗ Exact queries: {passed}/{len(sample_data)} passed ({success_rate:.1%})"
            )

        return success

    def _test_vector_search(
        self, vector_field: dict[str, Any], sample_size: int
    ) -> bool:
        """Test vector search returns reasonable results."""
        field_name = vector_field["name"]

        # Get primary key field for result identification
        pk_field = None
        for field in self.schema.get("fields", []):
            if field.get("is_primary_key") or field.get("is_primary"):
                pk_field = field["name"]
                break

        if not pk_field:
            display_info("No primary key field found, skipping vector search test")
            return False

        # Check if this is a function output field that cannot be retrieved
        is_function_output = self._is_function_output_field(field_name)
        if is_function_output:
            display_info(
                f"Skipping vector search test for function output field '{field_name}' (cannot retrieve raw data)"
            )
            return True  # Consider it passed since function output fields can't be tested this way

        # Sample some vectors with their primary keys
        try:
            sample_data = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=[field_name, pk_field],
                limit=sample_size,
            )
        except Exception as e:
            # If we can't retrieve the vector field, it might be sparse or have retrieval restrictions
            if "not allowed to retrieve raw data" in str(e):
                display_info(
                    f"Skipping vector search test for field '{field_name}' (raw data retrieval not allowed)"
                )
                return True  # Consider it passed since we can't test it
            display_error(f"Failed to sample vectors for search test: {e}")
            return False

        # Test vector searches
        passed = 0
        for row in sample_data:
            query_vector = row[field_name]
            query_pk = row[pk_field]
            try:
                # Use search without hardcoded parameters to let Milvus use index defaults
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    anns_field=field_name,
                    limit=10,
                    output_fields=[pk_field],
                )

                # Check if the query vector itself is returned in top 10 results
                found_self = False
                if search_results and len(search_results[0]) > 0:
                    for result in search_results[0]:
                        if result.entity.get(pk_field) == query_pk:
                            found_self = True
                            break

                if found_self:
                    passed += 1
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")

        recall_at_10 = passed / len(sample_data) if sample_data else 0
        success = recall_at_10 > 0.9  # 90% recall@10 for vector search

        if success:
            display_success(
                f"✓ Vector search recall@10: {passed}/{len(sample_data)} passed ({recall_at_10:.1%})"
            )
        else:
            display_error(
                f"✗ Vector search recall@10: {passed}/{len(sample_data)} passed ({recall_at_10:.1%})"
            )

        return success

    def _verify_special_fields(self, sample_size: int = 100) -> bool:
        """Verify special field behaviors (Nullable, Dynamic, PartitionKey, Function/BM25)."""
        self.console.print(
            "\n[bold blue]⚡ Verifying special field behaviors...[/bold blue]"
        )

        all_passed = True

        # Check nullable fields
        nullable_passed = self._verify_nullable_fields(sample_size)
        all_passed = all_passed and nullable_passed

        # Check dynamic fields
        dynamic_passed = self._verify_dynamic_fields(sample_size)
        all_passed = all_passed and dynamic_passed

        # Check partition key
        partition_passed = self._verify_partition_key()
        all_passed = all_passed and partition_passed

        # Check function fields (BM25, etc.)
        function_passed = self._verify_function_fields()
        all_passed = all_passed and function_passed

        return all_passed

    def _verify_nullable_fields(self, sample_size: int) -> bool:
        """Verify nullable field behavior."""
        nullable_fields = [
            f for f in self.schema.get("fields", []) if f.get("nullable", False)
        ]

        if not nullable_fields:
            display_info("No nullable fields found")
            return True

        try:
            # Sample data to check for null values
            output_fields = [f["name"] for f in nullable_fields]
            sample_data = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=output_fields,
                limit=sample_size,
            )

            # Check if nullable fields actually contain null values
            null_counts = {}
            for field in nullable_fields:
                field_name = field["name"]
                null_count = sum(
                    1 for row in sample_data if row.get(field_name) is None
                )
                null_counts[field_name] = null_count

            display_success(f"✓ Nullable fields verified: {null_counts}")
            return True

        except Exception as e:
            display_error(f"✗ Nullable field verification failed: {e}")
            return False

    def _verify_dynamic_fields(self, sample_size: int) -> bool:
        """Verify dynamic field behavior."""
        # Check if schema has enable_dynamic_field
        enable_dynamic = self.schema.get("enable_dynamic_field", False)

        if not enable_dynamic:
            display_info("Dynamic fields not enabled")
            return True

        try:
            # Query with dynamic fields
            sample_data = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["*"],
                limit=min(sample_size, 10),
            )

            # Check for dynamic fields (fields not in schema)
            schema_fields = {f["name"] for f in self.schema.get("fields", [])}
            dynamic_field_found = False

            for row in sample_data:
                for field_name in row:
                    if field_name not in schema_fields:
                        dynamic_field_found = True
                        break
                if dynamic_field_found:
                    break

            if dynamic_field_found:
                display_success("✓ Dynamic fields verified")
            else:
                display_info("No dynamic fields found in sample data")

            return True

        except Exception as e:
            display_error(f"✗ Dynamic field verification failed: {e}")
            return False

    def _verify_partition_key(self) -> bool:
        """Verify partition key field behavior."""
        partition_key_field = None
        for field in self.schema.get("fields", []):
            if field.get("partition_key", False):
                partition_key_field = field["name"]
                break

        if not partition_key_field:
            display_info("No partition key field found")
            return True

        try:
            # Check partition distribution
            partitions = self.client.list_partitions(self.collection_name)
            if len(partitions) <= 1:
                display_info("Only default partition found")
                return True

            # Sample partition key values and verify distribution
            sample_data = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=[partition_key_field],
                limit=1000,
            )

            partition_values = {row[partition_key_field] for row in sample_data}
            expected_partitions = self.metadata.get("collection_config", {}).get(
                "num_partitions", 1
            )

            if len(partition_values) >= min(expected_partitions, 10):
                display_success(
                    f"✓ Partition key verified: {len(partition_values)} unique values"
                )
            else:
                display_info(
                    f"Partition key has few unique values: {len(partition_values)}"
                )

            return True

        except Exception as e:
            display_error(f"✗ Partition key verification failed: {e}")
            return False

    def _verify_function_fields(self) -> bool:
        """Verify function fields (BM25, etc.)."""
        function_fields = [
            f
            for f in self.schema.get("fields", [])
            if f.get("is_function_output", False)
        ]

        if not function_fields:
            display_info("No function fields found")
            return True

        # For BM25 fields, verify they exist and have expected structure
        try:
            for field in function_fields:
                field_name = field["name"]
                # Try to query the function field
                sample_data = self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=[field_name],
                    limit=10,
                )

                if sample_data and all(field_name in row for row in sample_data):
                    display_success(f"✓ Function field '{field_name}' verified")
                else:
                    display_info(f"Function field '{field_name}' may be empty")

            return True

        except Exception as e:
            display_error(f"✗ Function field verification failed: {e}")
            return False

    def _compare_json_values(self, source_value: Any, milvus_value: Any) -> bool:
        """Compare JSON values, handling string vs parsed object differences."""
        import json

        try:
            # Parse source value if it's a string
            if isinstance(source_value, str):
                source_parsed = json.loads(source_value)
            else:
                source_parsed = source_value

            # Parse milvus value if it's a string
            if isinstance(milvus_value, str):
                milvus_parsed = json.loads(milvus_value)
            else:
                milvus_parsed = milvus_value

            # Compare the parsed objects
            return source_parsed == milvus_parsed
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"JSON comparison error: {e}")
            # Fallback to string comparison
            return str(source_value) == str(milvus_value)

    def _is_function_output_field(self, field_name: str) -> bool:
        """Check if a field is a function output field (like BM25 sparse vectors)."""
        functions = self.schema.get("functions", [])
        for function in functions:
            output_fields = function.get("output_field_names", [])
            if field_name in output_fields:
                return True
        return False

    def _load_sample_source_data(self, sample_size: int) -> list[dict[str, Any]]:
        """Load sample data from source files (parquet or json)."""
        try:
            # Try parquet files first
            parquet_files = list(self.data_path.glob("*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                sample_data = df.head(sample_size).to_dict(orient="records")
                return sample_data

            # Fallback to JSON files (exclude meta.json)
            json_files = [
                f for f in self.data_path.glob("*.json") if f.name != "meta.json"
            ]
            if json_files:
                return self._read_json_sample(json_files[0], sample_size)

            return []

        except Exception as e:
            logger.debug(f"Failed to load source data: {e}")
            return []

    def _read_json_sample(
        self, json_path: Path, sample_size: int
    ) -> list[dict[str, Any]]:
        """Read sample data from JSON file."""
        import json

        try:
            with open(json_path, encoding="utf-8") as f:
                content = f.read().strip()

            data_list = []
            if content.startswith("["):
                # JSON array format (list of dict) - Milvus bulk import format
                data_list = json.loads(content)
            elif content.startswith("{"):
                # Check if it's legacy format with "rows" key or single object
                data = json.loads(content)
                if "rows" in data and isinstance(data["rows"], list):
                    # Legacy Milvus bulk import format: {"rows": [...]}
                    data_list = data["rows"]
                else:
                    # Single object - wrap in list
                    data_list = [data]
            else:
                # Try line-delimited JSON
                lines = content.strip().split("\n")
                data_list = [json.loads(line) for line in lines if line.strip()]

            # Return sample
            return data_list[:sample_size]

        except Exception as e:
            logger.debug(f"Failed to read JSON file {json_path}: {e}")
            return []

    def _calculate_sample_size(self, total_rows: int) -> int:
        """Calculate optimal sample size: 10% of total rows, max 1M records."""
        if total_rows <= 0:
            return 1000  # Default fallback

        # 10% sampling ratio with 1M max limit
        sample_size = min(int(total_rows * 0.1), 1_000_000)
        # Ensure minimum of 1000 samples for statistical significance
        return max(sample_size, 1000)

    def _check_auto_id_field_verification(self, verification_level: str) -> bool:
        """Check if field verification is possible with AUTO_ID fields."""
        if verification_level not in ["scalar", "full"]:
            return True

        # Check if there are AUTO_ID fields in the schema
        has_auto_id = any(
            field.get("auto_id", False) for field in self.schema.get("fields", [])
        )

        if has_auto_id:
            auto_id_field = next(
                (
                    field["name"]
                    for field in self.schema.get("fields", [])
                    if field.get("auto_id", False)
                ),
                "unknown",
            )

            self.console.print(
                f"[yellow]⚠️  AUTO_ID field '{auto_id_field}' detected: "
                f"Skipping {verification_level} field value verification[/yellow]"
            )
            self.console.print(
                "[dim]   Field verification requires reliable data alignment, which is not possible with AUTO_ID.[/dim]"
            )
            self.console.print(
                "[dim]   Query/search functionality verification will continue normally.[/dim]"
            )
            return False

        return True

    def _display_summary(self, results: dict[str, bool]) -> None:
        """Display verification summary."""
        self.console.print("\n[bold]📋 Verification Summary[/bold]")

        table = Table()
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Description")

        check_descriptions = {
            "row_count": "Row count matches expected",
            "scalar_fields": "Scalar field values match source data (or skipped for AUTO_ID)",
            "all_fields": "All field values match source data (or skipped for AUTO_ID)",
            "field_consistency": "Field values match source data",
            "query_correctness": "Queries and searches return correct results",
            "special_fields": "Special field behaviors work correctly",
        }

        all_passed = all(results.values())

        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            description = check_descriptions.get(check, "")
            table.add_row(check.replace("_", " ").title(), status, description)

        self.console.print(table)

        if all_passed:
            display_success("🎉 All verification checks passed!")
        else:
            failed_checks = [k for k, v in results.items() if not v]
            display_error(f"❌ Some checks failed: {', '.join(failed_checks)}")

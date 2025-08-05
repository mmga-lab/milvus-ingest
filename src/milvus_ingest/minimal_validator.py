"""Minimal file validation for generated data files - checks file integrity only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table


class MinimalValidator:
    """Minimal validator that only checks file integrity after generation."""

    def __init__(self, output_dir: Path, console: Console | None = None):
        """Initialize validator with output directory.

        Args:
            output_dir: Directory containing generated files and meta.json
            console: Rich console for output (optional)
        """
        self.output_dir = output_dir
        self.console = console or Console()
        self.meta_file = output_dir / "meta.json"

    def validate(self) -> dict[str, Any]:
        """Run minimal validation on generated files.
        
        Only validates file integrity (readability, format, size).
        Does not validate row count against metadata since the actual
        generated data IS the ground truth.

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "summary": {
                "total_files": 0,
                "total_rows": 0,
                "total_size": 0,
                "format": "unknown",
            },
        }

        try:
            # 1. Check meta.json exists and load it
            if not self.meta_file.exists():
                results["valid"] = False
                results["errors"].append(f"meta.json not found in {self.output_dir}")
                return results

            try:
                with open(self.meta_file) as f:
                    metadata = json.load(f)
                    generation_info = metadata.get("generation_info", {})
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Failed to parse meta.json: {e}")
                return results

            # 2. Get expected files from metadata
            expected_files = generation_info.get("data_files", [])
            if not expected_files:
                results["errors"].append("No data files listed in meta.json")
                return results

            file_format = generation_info.get("format", "parquet")
            results["summary"]["format"] = file_format

            # Handle both old format (list of strings) and new format (list of dicts)
            if expected_files and isinstance(expected_files[0], str):
                # Old format: convert to new format for compatibility
                expected_files = [{"file_name": name, "rows": None} for name in expected_files]
            elif expected_files and isinstance(expected_files[0], dict):
                # New format: use as-is
                pass
            else:
                results["errors"].append("Invalid data_files format in meta.json")
                return results

            # 3. For each file, do minimal validation
            total_rows = 0
            total_size = 0
            valid_files = 0

            for file_info in expected_files:
                # Support both old and new format
                if isinstance(file_info, str):
                    file_name = file_info
                    expected_file_rows = None
                    expected_file_size = None
                else:
                    file_name = file_info.get("file_name")
                    expected_file_rows = file_info.get("rows")
                    expected_file_size = file_info.get("file_size_bytes")

                file_path = self.output_dir / file_name

                if not file_path.exists():
                    results["valid"] = False
                    results["errors"].append(f"File not found: {file_name}")
                    continue

                try:
                    # Get file size
                    file_size = file_path.stat().st_size
                    total_size += file_size

                    # Validate file size if expected size is available
                    if expected_file_size is not None:
                        if file_size != expected_file_size:
                            results["valid"] = False
                            results["errors"].append(
                                f"File size mismatch in {file_name}: expected {expected_file_size} bytes, got {file_size} bytes"
                            )
                            continue

                    # Try to get row count based on format
                    if file_format.lower() == "parquet":
                        row_count = self._get_parquet_row_count(file_path)
                    elif file_format.lower() == "json":
                        row_count = self._get_json_row_count(file_path)
                    else:
                        results["valid"] = False
                        results["errors"].append(f"Unsupported format: {file_format}")
                        continue

                    total_rows += row_count
                    valid_files += 1

                except Exception as e:
                    results["valid"] = False
                    results["errors"].append(f"Failed to read {file_name}: {e}")

            # Update summary
            results["summary"]["total_files"] = valid_files
            results["summary"]["total_rows"] = total_rows
            results["summary"]["total_size"] = total_size

            # Note: In generation phase, we don't verify row count against metadata
            # because the actual generated rows ARE the ground truth.
            # The metadata should reflect what was actually generated.
            # We only validate file integrity (readability, size) here.

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {e}")

        return results

    def _get_parquet_row_count(self, file_path: Path) -> int:
        """Get row count from parquet file using metadata only."""
        try:
            # Only read metadata, not the actual data
            parquet_file = pq.ParquetFile(file_path)
            return parquet_file.metadata.num_rows
        except Exception as e:
            raise Exception(f"Cannot read parquet metadata: {e}")

    def _get_json_row_count(self, file_path: Path) -> int:
        """Get row count from JSON Array file (Milvus bulk import format)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                else:
                    # Single JSON object (valid but uncommon for bulk import)
                    return 1
        except Exception as e:
            raise Exception(f"Cannot read JSON file: {e}")

    def display_results(self, results: dict[str, Any]) -> None:
        """Display validation results using Rich formatting."""
        if results["valid"]:
            self.console.print("\n✅ [bold green]File Integrity Validation Passed[/bold green]")
        else:
            self.console.print("\n❌ [bold red]File Integrity Validation Failed[/bold red]")

        # Summary table
        table = Table(title="File Integrity Validation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        summary = results["summary"]
        table.add_row("Total Files", str(summary["total_files"]))
        table.add_row("Total Rows", f"{summary['total_rows']:,}")
        table.add_row("Total Size", f"{summary['total_size'] / (1024 * 1024):.2f} MB")
        table.add_row("Format", summary["format"])
        table.add_row("Validation", "File readability and size only")

        self.console.print(table)

        # Show errors if any
        if results["errors"]:
            self.console.print("\n[bold red]Errors:[/bold red]")
            for error in results["errors"]:
                self.console.print(f"  • [red]{error}[/red]")

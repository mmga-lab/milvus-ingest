"""Minimal S3 file validation - only checks file existence and size."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from boto3 import client as boto3_client

from rich.console import Console
from rich.table import Table


class S3MinimalValidator:
    """Minimal validator for S3 uploaded files - only checks file existence and size."""

    def __init__(
        self,
        s3_client: boto3_client,
        bucket: str,
        prefix: str,
        local_path: Path,
        console: Console | None = None,
    ):
        """Initialize S3 validator.

        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            prefix: S3 prefix (folder path)
            local_path: Local directory containing meta.json
            console: Rich console for output (optional)
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")  # Remove trailing slash
        self.local_path = local_path
        self.console = console or Console()
        self.meta_file = local_path / "meta.json"

    def validate(self) -> dict[str, Any]:
        """Run minimal validation on S3 uploaded files.
        
        Only validates file existence and size for upload integrity.

        Returns:
            Dictionary with validation results
        """
        results: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "summary": {
                "total_files": 0,
                "total_size": 0,
                "format": "unknown",
            },
        }

        try:
            # 1. Check meta.json exists locally and load it
            if not self.meta_file.exists():
                results["valid"] = False
                results["errors"].append(f"meta.json not found in {self.local_path}")
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

            # 3. For each file, validate existence and size only
            total_size = 0
            valid_files = 0

            for file_info in expected_files:
                # Support both old and new format
                if isinstance(file_info, str):
                    file_name = file_info
                    expected_file_size = None
                else:
                    file_name = file_info.get("file_name")
                    expected_file_size = file_info.get("file_size_bytes")
                
                # Build S3 key
                s3_key = f"{self.prefix}/{file_name}" if self.prefix else file_name

                try:
                    # Check file exists and get size
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket, Key=s3_key
                    )
                    actual_file_size = head_response.get("ContentLength", 0)
                    total_size += actual_file_size

                    # Validate file size if expected size is available
                    if expected_file_size is not None:
                        if actual_file_size != expected_file_size:
                            results["valid"] = False
                            results["errors"].append(
                                f"File size mismatch in {file_name}: expected {expected_file_size} bytes, got {actual_file_size} bytes"
                            )
                            continue

                    valid_files += 1

                except self.s3_client.exceptions.NoSuchKey:
                    results["valid"] = False
                    results["errors"].append(f"File not found in S3: {s3_key}")
                except Exception as e:
                    results["valid"] = False
                    results["errors"].append(f"Failed to validate {file_name}: {e}")

            # Update summary
            results["summary"]["total_files"] = valid_files
            results["summary"]["total_size"] = total_size

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {e}")

        return results

    def display_results(self, results: dict[str, Any]) -> None:
        """Display validation results using Rich formatting."""
        if results["valid"]:
            self.console.print(
                "\n✅ [bold green]S3 Upload Validation Passed[/bold green]"
            )
        else:
            self.console.print("\n❌ [bold red]S3 Upload Validation Failed[/bold red]")

        # Summary table
        table = Table(title="S3 Upload Validation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        summary = results["summary"]
        table.add_row("S3 Location", f"s3://{self.bucket}/{self.prefix}/")
        table.add_row("Total Files", str(summary["total_files"]))
        table.add_row("Total Size", f"{summary['total_size'] / (1024 * 1024):.2f} MB")
        table.add_row("Format", summary["format"])
        table.add_row("Validation", "File existence and size only")

        self.console.print(table)

        # Show errors if any
        if results["errors"]:
            self.console.print("\n[bold red]Errors:[/bold red]")
            for error in results["errors"]:
                self.console.print(f"  • [red]{error}[/red]")

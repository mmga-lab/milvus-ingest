"""Minimal S3 file validation - only checks file accessibility and row count."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from boto3 import client as boto3_client

import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table


class S3MinimalValidator:
    """Minimal validator for S3 uploaded files - only checks accessibility and row count."""

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

        Returns:
            Dictionary with validation results
        """
        results: dict[str, Any] = {
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

            # 3. For each file, do minimal S3 validation
            total_rows = 0
            total_size = 0
            valid_files = 0

            for file_name in expected_files:
                # Build S3 key
                s3_key = f"{self.prefix}/{file_name}" if self.prefix else file_name

                try:
                    # Check file exists and get size
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket, Key=s3_key
                    )
                    file_size = head_response.get("ContentLength", 0)
                    total_size += file_size

                    # Get row count based on format
                    if file_format.lower() == "parquet":
                        row_count = self._get_s3_parquet_row_count(s3_key)
                    elif file_format.lower() == "json":
                        row_count = self._get_s3_json_row_count(s3_key)
                    else:
                        results["valid"] = False
                        results["errors"].append(f"Unsupported format: {file_format}")
                        continue

                    total_rows += row_count
                    valid_files += 1

                except self.s3_client.exceptions.NoSuchKey:
                    results["valid"] = False
                    results["errors"].append(f"File not found in S3: {s3_key}")
                except Exception as e:
                    results["valid"] = False
                    results["errors"].append(f"Failed to validate {file_name}: {e}")

            # Update summary
            results["summary"]["total_files"] = valid_files
            results["summary"]["total_rows"] = total_rows
            results["summary"]["total_size"] = total_size

            # Verify against expected totals from metadata
            expected_rows = generation_info.get("total_rows")
            if expected_rows and total_rows != expected_rows:
                results["valid"] = False
                results["errors"].append(
                    f"Row count mismatch: expected {expected_rows}, got {total_rows}"
                )

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {e}")

        return results

    def _get_s3_parquet_row_count(self, s3_key: str) -> int:
        """Get row count from S3 parquet file using metadata only.

        For Parquet files, we can read just the file footer which contains
        all metadata including row count, without downloading the entire file.
        """
        try:
            # First, get file size to read footer
            head_response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            file_size = head_response.get("ContentLength", 0)

            # Parquet footer is typically small, read last 8KB which should be enough
            footer_size = min(8192, file_size)
            footer_start = max(0, file_size - footer_size)

            # Read the footer
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=s3_key,
                Range=f"bytes={footer_start}-{file_size - 1}",
            )
            footer_bytes = response["Body"].read()

            # Create a file-like object that pyarrow can read
            # We need to provide the full file size for proper parsing
            class S3ParquetFile:
                def __init__(self, s3_client: Any, bucket: str, key: str, footer_bytes: bytes, file_size: int) -> None:
                    self.s3_client = s3_client
                    self.bucket = bucket
                    self.key = key
                    self.footer_bytes = footer_bytes
                    self.file_size = file_size
                    self._position = 0

                def size(self) -> int:
                    return self.file_size

                def tell(self) -> int:
                    return self._position

                def seek(self, position: int, whence: int = 0) -> int:
                    if whence == 0:  # absolute
                        self._position = position
                    elif whence == 1:  # relative to current
                        self._position += position
                    elif whence == 2:  # relative to end
                        self._position = self.file_size + position
                    return self._position

                def read(self, nbytes: int | None = None) -> bytes:
                    # For metadata reading, pyarrow typically reads from the end
                    if self._position >= self.file_size - len(self.footer_bytes):
                        # We're reading from the footer we already have
                        footer_offset = self._position - (
                            self.file_size - len(self.footer_bytes)
                        )
                        if nbytes is None:
                            data = self.footer_bytes[footer_offset:]
                        else:
                            data = self.footer_bytes[
                                footer_offset : footer_offset + nbytes
                            ]
                        self._position += len(data)
                        return data
                    else:
                        # Need to fetch from S3 (this shouldn't happen for metadata-only reads)
                        if nbytes is None:
                            nbytes = self.file_size - self._position
                        response = self.s3_client.get_object(
                            Bucket=self.bucket,
                            Key=self.key,
                            Range=f"bytes={self._position}-{self._position + nbytes - 1}",
                        )
                        data = response["Body"].read()
                        self._position += len(data)
                        return data

                def close(self) -> None:
                    pass

            # Create the file-like object and read metadata
            s3_file = S3ParquetFile(
                self.s3_client, self.bucket, s3_key, footer_bytes, file_size
            )
            parquet_file = pq.ParquetFile(s3_file)
            return parquet_file.metadata.num_rows

        except Exception as e:
            raise Exception(f"Cannot read S3 parquet metadata: {e}") from e

    def _get_s3_json_row_count(self, s3_key: str) -> int:
        """Get row count from S3 JSON file using streaming."""
        try:
            # For JSON, we need to count array elements
            # Use streaming to avoid loading entire file into memory
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)

            # Stream and count rows
            row_count = 0
            in_string = False
            escape_next = False
            brace_depth = 0

            # Read in chunks
            for chunk in response["Body"].iter_chunks(chunk_size=8192):
                for byte in chunk:
                    char = chr(byte)

                    if escape_next:
                        escape_next = False
                        continue

                    if char == "\\" and in_string:
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == "{":
                            brace_depth += 1
                            if brace_depth == 1:  # Top-level object
                                row_count += 1
                        elif char == "}":
                            brace_depth = max(0, brace_depth - 1)

            return row_count

        except Exception as e:
            # Fallback: download and parse (for smaller files)
            try:
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                content = response["Body"].read().decode("utf-8")
                data = json.loads(content)
                if isinstance(data, list):
                    return len(data)
                else:
                    return 1
            except Exception as parse_error:
                raise Exception(
                    f"Cannot read S3 JSON file: {e}, Parse error: {parse_error}"
                ) from parse_error

    def display_results(self, results: dict[str, Any]) -> None:
        """Display validation results using Rich formatting."""
        if results["valid"]:
            self.console.print(
                "\n✅ [bold green]S3 Upload Validation Passed[/bold green]"
            )
        else:
            self.console.print("\n❌ [bold red]S3 Upload Validation Failed[/bold red]")

        # Summary table
        table = Table(title="S3 Validation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        summary = results["summary"]
        table.add_row("S3 Location", f"s3://{self.bucket}/{self.prefix}/")
        table.add_row("Total Files", str(summary["total_files"]))
        table.add_row("Total Rows", f"{summary['total_rows']:,}")
        table.add_row("Total Size", f"{summary['total_size'] / (1024 * 1024):.2f} MB")
        table.add_row("Format", summary["format"])

        self.console.print(table)

        # Show errors if any
        if results["errors"]:
            self.console.print("\n[bold red]Errors:[/bold red]")
            for error in results["errors"]:
                self.console.print(f"  • [red]{error}[/red]")

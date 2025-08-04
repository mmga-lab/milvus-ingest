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

            # 3. For each file, do precise S3 validation
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

                    # Get row count based on format
                    if file_format.lower() == "parquet":
                        actual_row_count = self._get_s3_parquet_row_count(s3_key)
                    elif file_format.lower() == "json":
                        actual_row_count = self._get_s3_json_row_count(s3_key)
                    else:
                        results["valid"] = False
                        results["errors"].append(f"Unsupported format: {file_format}")
                        continue

                    # Validate row count if expected count is available
                    if expected_file_rows is not None:
                        if actual_row_count != expected_file_rows:
                            results["valid"] = False
                            results["errors"].append(
                                f"Row count mismatch in {file_name}: expected {expected_file_rows}, got {actual_row_count}"
                            )
                            continue

                    total_rows += actual_row_count
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

            # Verify against expected totals from metadata (strict validation)
            expected_rows = generation_info.get("total_rows")
            if expected_rows and total_rows != expected_rows:
                results["valid"] = False
                results["errors"].append(
                    f"Total row count mismatch: expected {expected_rows}, got {total_rows}"
                )

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {e}")

        return results

    def _get_s3_parquet_row_count(self, s3_key: str) -> int:
        """Get exact row count from S3 parquet file by reading only the footer metadata.
        
        This is highly efficient as it only downloads the Parquet footer (typically 8-32KB)
        instead of the entire file, while still providing 100% accurate row counts.
        """
        try:
            # Get file size first
            head_response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            file_size = head_response.get("ContentLength", 0)
            
            # For very small files, just download completely
            if file_size < 8192:  # Less than 8KB
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                file_content = response["Body"].read()
                import io
                file_like = io.BytesIO(file_content)
                parquet_file = pq.ParquetFile(file_like)
                return parquet_file.metadata.num_rows
            
            # Try progressively larger footer reads until we get valid metadata
            footer_sizes = [8192, 16384, 32768, 65536]  # 8KB to 64KB
            
            for footer_size in footer_sizes:
                try:
                    # Calculate range to read from end of file
                    footer_start = max(0, file_size - footer_size)
                    
                    # Read only the footer portion using S3 Range request
                    response = self.s3_client.get_object(
                        Bucket=self.bucket,
                        Key=s3_key,
                        Range=f"bytes={footer_start}-{file_size - 1}"
                    )
                    footer_bytes = response["Body"].read()
                    
                    # Create a file-like object with padding to simulate full file
                    import io
                    temp_buffer = io.BytesIO()
                    
                    # Add minimal padding if needed (only for files that would be reasonably sized)
                    padding_size = file_size - len(footer_bytes)
                    if padding_size < 10 * 1024 * 1024:  # Less than 10MB padding
                        temp_buffer.write(b'\x00' * padding_size)
                    
                    # Write the actual footer data
                    temp_buffer.write(footer_bytes)
                    temp_buffer.seek(0)
                    
                    # Try to parse with PyArrow
                    try:
                        parquet_file = pq.ParquetFile(temp_buffer)
                        return parquet_file.metadata.num_rows
                    except Exception:
                        # This footer size wasn't enough, try the next larger size
                        continue
                        
                except Exception:
                    # Error with this range request, try larger size
                    continue
            
            # If progressive reading failed, fall back to full download for reasonable sized files
            if file_size < 100 * 1024 * 1024:  # Less than 100MB
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                file_content = response["Body"].read()
                import io
                file_like = io.BytesIO(file_content)
                parquet_file = pq.ParquetFile(file_like)
                return parquet_file.metadata.num_rows
            else:
                raise Exception(
                    f"Could not read Parquet metadata from large file ({file_size / (1024*1024):.1f}MB). "
                    "Footer may be larger than 64KB or file may be corrupted."
                )

        except Exception as e:
            raise Exception(f"Cannot read S3 parquet row count: {e}") from e

    def _get_s3_json_row_count(self, s3_key: str) -> int:
        """Get row count from S3 JSON file using optimized approach.
        
        Only supports JSON Array format (Milvus bulk import requirement).
        Uses streaming to minimize memory usage and bandwidth for large files.
        """
        try:
            # Get file size to decide strategy
            head_response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            file_size = head_response.get("ContentLength", 0)
            
            # For small files, just download and parse directly
            if file_size < 32768:  # Less than 32KB
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                content = response["Body"].read().decode("utf-8")
                return self._count_json_objects(content)
            
            # For moderate files, download and parse directly
            if file_size < 10 * 1024 * 1024:  # Less than 10MB
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                content = response["Body"].read().decode("utf-8")
                return self._count_json_objects(content)
            else:
                # For large JSON arrays, use streaming parser
                return self._count_json_array_streaming(s3_key)
                
        except Exception as e:
            # Fallback: download and parse (for smaller files only)
            try:
                if file_size < 50 * 1024 * 1024:  # Less than 50MB
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                    content = response["Body"].read().decode("utf-8")
                    return self._count_json_objects(content)
                else:
                    raise Exception(f"Cannot process large JSON file ({file_size / (1024*1024):.1f}MB): {e}")
            except Exception as parse_error:
                raise Exception(
                    f"Cannot read S3 JSON file: {e}, Parse error: {parse_error}"
                ) from parse_error


    def _count_json_array_streaming(self, s3_key: str) -> int:
        """Count objects in JSON Array format using streaming parser."""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)

        # Stream and count objects in array
        row_count = 0
        in_string = False
        escape_next = False
        brace_depth = 0
        bracket_depth = 0
        found_array_start = False

        # Read in chunks
        for chunk in response["Body"].iter_chunks(chunk_size=65536):  # 64KB chunks
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
                    if char == "[":
                        bracket_depth += 1
                        if bracket_depth == 1:
                            found_array_start = True
                    elif char == "]":
                        bracket_depth = max(0, bracket_depth - 1)
                    elif char == "{" and found_array_start and bracket_depth == 1:
                        brace_depth += 1
                        if brace_depth == 1:  # Top-level object in array
                            row_count += 1
                    elif char == "}":
                        brace_depth = max(0, brace_depth - 1)

        return row_count

    def _count_json_objects(self, content: str) -> int:
        """Count JSON objects in JSON Array format."""
        import json
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return len(data)
            else:
                # Single JSON object
                return 1
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format (expected JSON Array): {e}")

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

"""S3/MinIO upload functionality for generated data files."""

from __future__ import annotations

import os
import threading
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Callable

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .logging_config import get_logger
from .rich_display import display_error, display_info


class ProgressPercentage:
    """Track upload progress for S3 transfers."""

    def __init__(self, file_path: Path, callback: Callable[[int], None] | None = None):
        """Initialize progress tracker.

        Args:
            file_path: Path to the file being uploaded
            callback: Optional callback to receive progress updates
        """
        self._filename = file_path.name
        self._size = file_path.stat().st_size
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._callback = callback
        self.logger = get_logger(__name__)

    def __call__(self, bytes_amount: int) -> None:
        """Called by boto3 during upload to report progress.

        Args:
            bytes_amount: Number of bytes transferred
        """
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100

            # Log progress periodically (every 10%)
            if int(percentage) % 10 == 0 and int(percentage) != int(
                (self._seen_so_far - bytes_amount) / self._size * 100
            ):
                self.logger.debug(
                    f"Upload progress for {self._filename}: {percentage:.0f}% "
                    f"({self._seen_so_far:,} / {self._size:,} bytes)"
                )

            # Call external callback if provided
            if self._callback:
                self._callback(int(percentage))


class S3Uploader:
    """Handle uploads to S3-compatible storage (S3, MinIO, etc.)."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region_name: str = "us-east-1",
        verify_ssl: bool = True,
    ):
        """Initialize S3 client.

        Args:
            endpoint_url: S3-compatible endpoint URL (e.g., http://localhost:9000 for MinIO)
            access_key_id: AWS access key ID (can also be set via AWS_ACCESS_KEY_ID env var)
            secret_access_key: AWS secret access key (can also be set via AWS_SECRET_ACCESS_KEY env var)
            region_name: AWS region name (default: us-east-1)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        self.logger = get_logger(__name__)

        # Get credentials from environment if not provided
        if not access_key_id:
            access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        if not secret_access_key:
            secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        # Configure retry behavior for rate limiting
        retry_config = Config(
            retries={
                "max_attempts": 10,
                "mode": "adaptive",  # Automatic exponential backoff
            }
        )

        # Configure default multipart upload settings for large files
        self.default_transfer_config = TransferConfig(
            multipart_threshold=1024
            * 1024
            * 1024,  # 1GB - files larger than this use multipart
            multipart_chunksize=256 * 1024 * 1024,  # 256MB chunks
            max_concurrency=5,  # Reduced from default 10 to prevent rate limiting
            num_download_attempts=10,  # Retry attempts for failed parts
            use_threads=True,  # Enable threading for parallel uploads
        )

        # Create S3 client with retry configuration
        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
                verify=verify_ssl,
                config=retry_config,
            )
            self.endpoint_url = endpoint_url
            self.logger.info(
                "S3 client initialized",
                extra={
                    "endpoint": endpoint_url or "AWS S3",
                    "region": region_name,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _get_transfer_config(self, file_size: int) -> TransferConfig:
        """Get appropriate TransferConfig based on file size.

        Args:
            file_size: File size in bytes

        Returns:
            TransferConfig optimized for the file size
        """
        gb_size = file_size / (1024**3)

        if gb_size <= 5:
            # For files <= 5GB, use default config
            return self.default_transfer_config
        elif gb_size <= 50:
            # For files 5-50GB, use larger chunks (512MB) to reduce parts
            return TransferConfig(
                multipart_threshold=1024 * 1024 * 1024,  # 1GB
                multipart_chunksize=512 * 1024 * 1024,  # 512MB chunks
                max_concurrency=3,  # Further reduced concurrency
                num_download_attempts=10,
                use_threads=True,
            )
        else:
            # For files > 50GB, use 1GB chunks to stay under 10,000 part limit
            # 10,000 parts * 1GB = 10TB max file size
            return TransferConfig(
                multipart_threshold=1024 * 1024 * 1024,  # 1GB
                multipart_chunksize=1024 * 1024 * 1024,  # 1GB chunks
                max_concurrency=2,  # Minimal concurrency for huge files
                num_download_attempts=10,
                use_threads=True,
            )

    def upload_directory(
        self,
        local_path: Path,
        bucket: str,
        prefix: str = "",
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Upload a directory to S3/MinIO.

        Args:
            local_path: Local directory path containing files to upload
            bucket: S3 bucket name
            prefix: Optional prefix (folder) in the bucket
            show_progress: Whether to show upload progress

        Returns:
            Dictionary with upload statistics
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Directory not found: {local_path}")

        if not local_path.is_dir():
            raise ValueError(f"Path is not a directory: {local_path}")

        # Check if bucket exists, create if not
        self._ensure_bucket_exists(bucket)

        # Collect all files to upload
        files_to_upload = []
        total_size = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                files_to_upload.append(file_path)
                total_size += file_path.stat().st_size

        if not files_to_upload:
            display_info(f"No files found in {local_path}")
            return {"uploaded_files": 0, "total_size": 0, "failed_files": []}

        # Upload files
        uploaded_files = 0
        failed_files = []

        # Identify large files (> 1GB) for special handling
        large_file_threshold = 1024 * 1024 * 1024  # 1GB
        large_files = {
            f for f in files_to_upload if f.stat().st_size > large_file_threshold
        }

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task(
                    f"Uploading {len(files_to_upload)} files",
                    total=len(files_to_upload),
                )

                for file_path in files_to_upload:
                    # Calculate S3 key
                    relative_path = file_path.relative_to(local_path)
                    relative_path_str = str(relative_path).replace(
                        "\\", "/"
                    )  # Convert to forward slashes

                    # Build S3 key, ensuring no double slashes
                    if prefix:
                        # Remove trailing slash from prefix if present
                        clean_prefix = prefix.rstrip("/")
                        s3_key = f"{clean_prefix}/{relative_path_str}"
                    else:
                        s3_key = relative_path_str

                    try:
                        # Log large file uploads
                        if file_path in large_files:
                            file_size_gb = file_path.stat().st_size / (1024**3)
                            self.logger.info(
                                f"Uploading large file: {file_path.name} ({file_size_gb:.1f}GB) "
                                f"using multipart upload with integrity verification"
                            )

                        upload_result = self._upload_file(file_path, bucket, s3_key)
                        if upload_result["success"] and upload_result["integrity_valid"]:
                            uploaded_files += 1
                        else:
                            failed_files.append({
                                "file": str(file_path), 
                                "error": "Upload integrity verification failed",
                                "details": upload_result
                            })
                        progress.update(task, advance=1)
                    except Exception as e:
                        self.logger.error(f"Failed to upload {file_path}: {e}")
                        failed_files.append({"file": str(file_path), "error": str(e)})
                        progress.update(task, advance=1)
        else:
            # Upload without progress bar
            for file_path in files_to_upload:
                relative_path = file_path.relative_to(local_path)
                relative_path_str = str(relative_path).replace(
                    "\\", "/"
                )  # Convert to forward slashes

                # Build S3 key, ensuring no double slashes
                if prefix:
                    # Remove trailing slash from prefix if present
                    clean_prefix = prefix.rstrip("/")
                    s3_key = f"{clean_prefix}/{relative_path_str}"
                else:
                    s3_key = relative_path_str

                try:
                    # Log large file uploads
                    if file_path in large_files:
                        file_size_gb = file_path.stat().st_size / (1024**3)
                        self.logger.info(
                            f"Uploading large file: {file_path.name} ({file_size_gb:.1f}GB) "
                            f"using multipart upload with integrity verification"
                        )

                    upload_result = self._upload_file(file_path, bucket, s3_key)
                    if upload_result["success"] and upload_result["integrity_valid"]:
                        uploaded_files += 1
                    else:
                        failed_files.append({
                            "file": str(file_path), 
                            "error": "Upload integrity verification failed",
                            "details": upload_result
                        })
                except Exception as e:
                    self.logger.error(f"Failed to upload {file_path}: {e}")
                    failed_files.append({"file": str(file_path), "error": str(e)})

        # Validate uploaded files if all uploads succeeded
        validation_results = None
        if uploaded_files > 0 and len(failed_files) == 0 and local_path.exists():
            validation_results = self._validate_uploaded_files(
                local_path, bucket, prefix, files_to_upload
            )
        
        return {
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "total_size": total_size,
            "bucket": bucket,
            "prefix": prefix,
            "validation": validation_results,
        }

    def _ensure_bucket_exists(self, bucket: str) -> None:
        """Ensure bucket exists, create if it doesn't."""
        try:
            self.s3_client.head_bucket(Bucket=bucket)
            self.logger.debug(f"Bucket '{bucket}' exists")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    if self.endpoint_url and "amazonaws.com" not in (
                        self.endpoint_url or ""
                    ):
                        # For MinIO and other S3-compatible services
                        self.s3_client.create_bucket(Bucket=bucket)
                    else:
                        # For AWS S3, need to specify LocationConstraint for non-us-east-1
                        response = self.s3_client.get_bucket_location(Bucket=bucket)
                        region = response.get("LocationConstraint", "us-east-1")
                        if region and region != "us-east-1":
                            self.s3_client.create_bucket(
                                Bucket=bucket,
                                CreateBucketConfiguration={
                                    "LocationConstraint": region
                                },
                            )
                        else:
                            self.s3_client.create_bucket(Bucket=bucket)
                    self.logger.info(f"Created bucket '{bucket}'")
                except Exception as create_error:
                    self.logger.error(
                        f"Failed to create bucket '{bucket}': {create_error}"
                    )
                    raise
            else:
                # Other error
                self.logger.error(f"Error checking bucket '{bucket}': {e}")
                raise

    def _upload_file(
        self,
        file_path: Path,
        bucket: str,
        key: str,
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Upload a single file to S3 using multipart upload for large files with integrity validation.

        Args:
            file_path: Local file path to upload
            bucket: S3 bucket name
            key: S3 object key
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with upload results including integrity validation
        """
        import hashlib
        
        result = {
            "success": False,
            "file_path": str(file_path),
            "s3_key": key,
            "local_size": 0,
            "remote_size": 0,
            "local_etag": None,
            "remote_etag": None,
            "integrity_valid": False
        }
        
        try:
            # Get local file information
            file_size = file_path.stat().st_size
            result["local_size"] = file_size
            
            # Calculate MD5 for small files (< 1GB) for ETag comparison
            # For larger files, use multipart ETag calculation
            if file_size < 1024 * 1024 * 1024:  # < 1GB
                result["local_etag"] = self._calculate_file_md5(file_path)
            else:
                # For large files, calculate multipart ETag
                transfer_config = self._get_transfer_config(file_size)
                chunk_size = transfer_config.multipart_chunksize
                result["local_etag"] = self._calculate_multipart_etag(file_path, chunk_size)

            # Get optimized transfer config based on file size
            transfer_config = self._get_transfer_config(file_size)

            # Log config details for large files
            if file_size > 5 * 1024**3:  # > 5GB
                chunk_size_mb = transfer_config.multipart_chunksize / (1024**2)
                self.logger.info(
                    f"Uploading large file {file_path.name} ({file_size / (1024**3):.1f}GB) "
                    f"using {chunk_size_mb:.0f}MB chunks with {transfer_config.max_concurrency} concurrent uploads"
                )

            # Create a callback wrapper if progress tracking is requested
            callback = None
            if progress_callback:
                callback = ProgressPercentage(file_path, progress_callback)

            # Upload the file
            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=bucket,
                Key=key,
                Config=transfer_config,
                Callback=callback,
                ExtraArgs={},
            )
            
            # Verify upload integrity
            integrity_check = self._verify_upload_integrity(bucket, key, result)
            result.update(integrity_check)
            
            if result["integrity_valid"]:
                self.logger.debug(f"✅ Upload verified: {file_path} -> s3://{bucket}/{key}")
                result["success"] = True
            else:
                self.logger.error(f"❌ Upload integrity check failed: {file_path}")
                # Clean up failed upload
                try:
                    self.s3_client.delete_object(Bucket=bucket, Key=key)
                    self.logger.info(f"Cleaned up failed upload: s3://{bucket}/{key}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up failed upload: {cleanup_error}")
                raise ValueError(f"Upload integrity validation failed for {file_path}")
                
        except NoCredentialsError as e:
            raise ValueError(
                "No credentials found. Please provide access_key_id and secret_access_key "
                "or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            ) from e
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "SlowDown":
                self.logger.warning(
                    f"Rate limited while uploading {file_path}. "
                    "The file will be retried automatically with exponential backoff."
                )
            self.logger.error(f"Failed to upload {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to upload {file_path}: {e}")
            raise
            
        return result

    def _calculate_file_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        import hashlib
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in 64kb chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_multipart_etag(self, file_path: Path, chunk_size: int) -> str:
        """Calculate multipart ETag for large files (S3 multipart upload format)."""
        import hashlib
        
        chunk_hashes = []
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunk_hashes.append(hashlib.md5(chunk).digest())
        
        if len(chunk_hashes) == 1:
            # Single part, return simple MD5
            return chunk_hashes[0].hex()
        else:
            # Multipart, combine all chunk hashes
            combined_hash = hashlib.md5(b''.join(chunk_hashes)).hexdigest()
            return f"{combined_hash}-{len(chunk_hashes)}"
    
    def _verify_upload_integrity(self, bucket: str, key: str, upload_info: dict) -> dict[str, Any]:
        """Verify uploaded file integrity by comparing size and ETag."""
        result = {
            "remote_size": 0,
            "remote_etag": None,
            "integrity_valid": False
        }
        
        try:
            # Get remote file metadata
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            result["remote_size"] = response.get("ContentLength", 0)
            result["remote_etag"] = response.get("ETag", "").strip('"')
            
            # Check size match
            size_match = result["remote_size"] == upload_info["local_size"]
            
            # Check ETag match (more complex for multipart uploads)
            etag_match = False
            local_etag = upload_info.get("local_etag", "")
            remote_etag = result["remote_etag"]
            
            if local_etag and remote_etag:
                # Handle multipart ETags (contain dashes)
                if "-" in remote_etag:
                    # Multipart upload - ETags format: hash-partcount
                    etag_match = local_etag == remote_etag
                else:
                    # Single part upload - simple MD5 comparison
                    etag_match = local_etag.split("-")[0] == remote_etag
            
            result["integrity_valid"] = size_match and etag_match
            
            if not size_match:
                self.logger.warning(
                    f"Size mismatch - Local: {upload_info['local_size']}, Remote: {result['remote_size']}"
                )
            if not etag_match:
                self.logger.warning(
                    f"ETag mismatch - Local: {local_etag}, Remote: {remote_etag}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to verify upload integrity: {e}")
            
        return result

    def _validate_uploaded_files(
        self, 
        local_path: Path, 
        bucket: str, 
        prefix: str, 
        uploaded_files: list[Path]
    ) -> dict[str, Any]:
        """Validate uploaded files by reading them directly from S3.
        
        Args:
            local_path: Local directory path
            bucket: S3 bucket name
            prefix: S3 prefix
            uploaded_files: List of local files that were uploaded
            
        Returns:
            Dictionary with validation results
        """
        import json
        import pandas as pd
        import pyarrow.parquet as pq
        
        self.logger.info("Validating uploaded files in S3...")
        
        validation_results = {
            "valid": True,
            "total_files": len(uploaded_files),
            "validated_files": 0,
            "failed_validations": [],
            "file_details": []
        }
        
        # Find meta.json to understand the expected format
        meta_file = local_path / "meta.json"
        file_format = "parquet"  # default
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
                    file_format = metadata.get("generation_info", {}).get("format", "parquet")
            except Exception as e:
                self.logger.warning(f"Could not read meta.json: {e}")
        
        # Validate each uploaded file by reading from S3
        for file_path in uploaded_files:
            # Skip meta.json as it's not a data file
            if file_path.name == "meta.json":
                continue
                
            try:
                # Calculate S3 key for this file
                relative_path = file_path.relative_to(local_path)
                relative_path_str = str(relative_path).replace("\\", "/")
                
                if prefix:
                    clean_prefix = prefix.rstrip("/")
                    s3_key = f"{clean_prefix}/{relative_path_str}"
                else:
                    s3_key = relative_path_str
                
                # Build S3 URL for pandas/pyarrow
                if self.endpoint_url:
                    s3_url = f"{self.endpoint_url.rstrip('/')}/{bucket}/{s3_key}"
                else:
                    s3_url = f"s3://{bucket}/{s3_key}"
                
                # Validate file by reading from S3
                file_validation = self._validate_single_s3_file(
                    s3_url, file_path.name, file_format, bucket, s3_key
                )
                
                validation_results["file_details"].append(file_validation)
                
                if file_validation["valid"]:
                    validation_results["validated_files"] += 1
                else:
                    validation_results["valid"] = False
                    validation_results["failed_validations"].append({
                        "file": file_path.name,
                        "s3_key": s3_key,
                        "errors": file_validation.get("errors", [])
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to validate {file_path.name}: {e}")
                validation_results["valid"] = False
                validation_results["failed_validations"].append({
                    "file": file_path.name,
                    "errors": [f"Validation error: {e}"]
                })
        
        # Log results
        if validation_results["valid"]:
            self.logger.info(
                f"✅ All {validation_results['validated_files']} uploaded files validated successfully"
            )
        else:
            self.logger.warning(
                f"⚠️ Upload validation failed for {len(validation_results['failed_validations'])} files"
            )
            
        return validation_results
    
    def _validate_single_s3_file(
        self, 
        s3_url: str, 
        filename: str, 
        file_format: str,
        bucket: str,
        s3_key: str
    ) -> dict[str, Any]:
        """Validate a single file directly from S3.
        
        Args:
            s3_url: Full S3 URL to the file
            filename: Original filename  
            file_format: Expected format (parquet or json)
            bucket: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            Dictionary with file validation results
        """
        import pandas as pd
        import pyarrow.parquet as pq
        import json
        
        result = {
            "filename": filename,
            "s3_key": s3_key,
            "valid": True,
            "errors": [],
            "row_count": 0,
            "file_size_bytes": 0
        }
        
        try:
            # Get file metadata from S3
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            result["file_size_bytes"] = response.get("ContentLength", 0)
            
            # Validate based on format
            if file_format.lower() == "parquet":
                result.update(self._validate_s3_parquet_file(s3_url, bucket, s3_key))
            elif file_format.lower() == "json":
                result.update(self._validate_s3_json_file(s3_url, bucket, s3_key))
            else:
                result["valid"] = False
                result["errors"].append(f"Unsupported format: {file_format}")
                
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate file: {e}")
            
        return result
    
    def _validate_s3_parquet_file(self, s3_url: str, bucket: str, s3_key: str) -> dict[str, Any]:
        """Validate a parquet file directly from S3."""
        import io
        import pandas as pd
        import pyarrow.parquet as pq
        
        result = {"parquet_validation": {}}
        
        try:
            # Get file size first to determine validation strategy
            head_response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            file_size = head_response.get("ContentLength", 0)
            
            # For files > 100MB, use streaming/partial validation
            if file_size > 100 * 1024 * 1024:  # > 100MB
                self.logger.info(f"Large file detected ({file_size / (1024*1024):.1f}MB), using streaming validation")
                result.update(self._validate_large_parquet_streaming(bucket, s3_key))
            else:
                # For smaller files, use full download validation
                response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
                file_content = response['Body'].read()
                
                # Create a BytesIO buffer for parquet reading
                buffer = io.BytesIO(file_content)
                
                # Try to read the parquet file metadata
                parquet_file = pq.ParquetFile(buffer)
                metadata = parquet_file.metadata
                
                result["row_count"] = metadata.num_rows
                result["parquet_validation"] = {
                    "num_row_groups": metadata.num_row_groups,
                    "columns": metadata.num_columns,
                    "format_version": str(metadata.format_version)
                }
                
                # Reset buffer and try to read a small sample
                buffer.seek(0)
                df_sample = pd.read_parquet(buffer, nrows=10)
                
                if len(df_sample) == 0 and metadata.num_rows > 0:
                    result["valid"] = False
                    result["errors"].append("File appears empty but metadata indicates rows")
                
        except Exception as e:
            result["valid"] = False  
            result["errors"].append(f"Parquet validation error: {e}")
            
        return result
    
    def _validate_large_parquet_streaming(self, bucket: str, s3_key: str) -> dict[str, Any]:
        """Validate large parquet files using partial/streaming reads."""
        import io
        import pandas as pd
        import pyarrow.parquet as pq
        
        result = {"parquet_validation": {}}
        
        try:
            # Read only the first 10MB to get metadata and validate structure
            partial_response = self.s3_client.get_object(
                Bucket=bucket, 
                Key=s3_key,
                Range="bytes=0-10485759"  # First 10MB
            )
            partial_content = partial_response['Body'].read()
            
            # Try to read parquet metadata from partial content
            buffer = io.BytesIO(partial_content)
            
            try:
                parquet_file = pq.ParquetFile(buffer)
                metadata = parquet_file.metadata
                
                result["row_count"] = metadata.num_rows
                result["parquet_validation"] = {
                    "num_row_groups": metadata.num_row_groups,
                    "columns": metadata.num_columns,
                    "format_version": str(metadata.format_version),
                    "validation_method": "partial_read"
                }
                
                # Try to read first few rows as sample
                buffer.seek(0)
                df_sample = pd.read_parquet(buffer, nrows=5)
                
                if len(df_sample) == 0 and metadata.num_rows > 0:
                    result["valid"] = False
                    result["errors"].append("File appears empty but metadata indicates rows")
                    
            except Exception:
                # If partial read fails, try reading parquet footer only
                # Parquet files store metadata at the end
                head_response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
                file_size = head_response.get("ContentLength", 0)
                
                if file_size > 1024:  # Only if file is reasonably sized
                    # Read last 1KB which should contain the footer
                    footer_response = self.s3_client.get_object(
                        Bucket=bucket,
                        Key=s3_key,
                        Range=f"bytes={file_size-1024}-{file_size-1}"
                    )
                    footer_content = footer_response['Body'].read()
                    
                    # Basic validation - check for parquet magic bytes
                    if b'PAR1' in footer_content:
                        result["parquet_validation"] = {
                            "has_parquet_footer": True,
                            "validation_method": "footer_check"
                        }
                    else:
                        result["valid"] = False
                        result["errors"].append("Invalid parquet footer signature")
                
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Large parquet validation error: {e}")
            
        return result
        
    def _validate_s3_json_file(self, s3_url: str, bucket: str, s3_key: str) -> dict[str, Any]:
        """Validate a JSON file directly from S3."""
        import json
        
        result = {"json_validation": {}}
        
        try:
            # Download file content to memory for validation
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            file_content = response['Body'].read().decode('utf-8')
            
            # Try to read as JSON (should be array of objects)
            data = json.loads(file_content)
                
            if isinstance(data, list):
                result["row_count"] = len(data)
                result["json_validation"]["is_array"] = True
                
                # Check first few records
                if data:
                    sample_records = data[:min(5, len(data))]
                    all_dicts = all(isinstance(record, dict) for record in sample_records)
                    result["json_validation"]["records_are_dicts"] = all_dicts
                    
                    if not all_dicts:
                        result["valid"] = False
                        result["errors"].append("JSON contains non-dictionary records")
            else:
                result["valid"] = False
                result["errors"].append("JSON file is not an array of objects")
                
        except json.JSONDecodeError as e:
            result["valid"] = False
            result["errors"].append(f"Invalid JSON format: {e}")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"JSON validation error: {e}")
            
        return result

    def test_connection(self) -> bool:
        """Test connection to S3/MinIO."""
        try:
            # Try to list buckets as a connection test
            response = self.s3_client.list_buckets()
            self.logger.info(
                f"Successfully connected to S3. Found {len(response['Buckets'])} buckets."
            )
            return True
        except NoCredentialsError:
            display_error(
                "No credentials found. Please provide credentials via:\n"
                "  - Command line options (--access-key-id, --secret-access-key)\n"
                "  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                "  - AWS credentials file (~/.aws/credentials)"
            )
            return False
        except Exception as e:
            display_error(f"Failed to connect to S3: {e}")
            return False


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse S3 URL into bucket and prefix.

    Args:
        url: S3 URL in format s3://bucket/prefix or s3://bucket

    Returns:
        Tuple of (bucket, prefix)
    """
    if not url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL format: {url}. Must start with 's3://'")

    parsed = urlparse(url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError(f"No bucket specified in URL: {url}")

    return bucket, prefix

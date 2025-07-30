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
                                f"using multipart upload with 256MB chunks"
                            )

                        self._upload_file(file_path, bucket, s3_key)
                        uploaded_files += 1
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
                            f"using multipart upload with 256MB chunks"
                        )

                    self._upload_file(file_path, bucket, s3_key)
                    uploaded_files += 1
                except Exception as e:
                    self.logger.error(f"Failed to upload {file_path}: {e}")
                    failed_files.append({"file": str(file_path), "error": str(e)})

        return {
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "total_size": total_size,
            "bucket": bucket,
            "prefix": prefix,
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
    ) -> None:
        """Upload a single file to S3 using multipart upload for large files.

        Args:
            file_path: Local file path to upload
            bucket: S3 bucket name
            key: S3 object key
            progress_callback: Optional callback function for progress updates
        """
        try:
            # Use upload_file which automatically handles multipart for large files
            extra_args: dict[str, Any] = {}

            # Get optimized transfer config based on file size
            file_size = file_path.stat().st_size
            transfer_config = self._get_transfer_config(file_size)

            # Log config details for large files
            if file_size > 5 * 1024**3:  # > 5GB
                chunk_size_mb = transfer_config.multipart_chunksize / (1024**2)
                self.logger.info(
                    f"Using {chunk_size_mb:.0f}MB chunks with {transfer_config.max_concurrency} "
                    f"concurrent uploads for {file_path.name}"
                )

            # Create a callback wrapper if progress tracking is requested
            callback = None
            if progress_callback:
                callback = ProgressPercentage(file_path, progress_callback)

            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=bucket,
                Key=key,
                Config=transfer_config,
                Callback=callback,
                ExtraArgs=extra_args,
            )
            self.logger.debug(f"Uploaded {file_path} to s3://{bucket}/{key}")
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

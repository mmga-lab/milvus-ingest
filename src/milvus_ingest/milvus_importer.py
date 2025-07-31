"""Bulk import functionality for Milvus using bulk_import API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient
from pymilvus.bulk_writer import bulk_import, get_import_progress, list_import_jobs
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .logging_config import get_logger
from .milvus_schema_builder import MilvusSchemaBuilder


class MilvusBulkImporter:
    """Handle bulk importing data to Milvus using bulk_import API."""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        db_name: str = "default",
    ):
        """Initialize Milvus connection.

        Args:
            uri: Milvus server URI (e.g., http://localhost:19530)
            token: Token for authentication
            db_name: Database name
        """
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.logger = get_logger(__name__)
        self.console = Console(stderr=True)

        # Initialize Milvus client for collection management
        try:
            self.client = MilvusClient(
                uri=uri,
                token=token,
                db_name=db_name,
            )
            # Initialize schema builder
            self.schema_builder = MilvusSchemaBuilder(self.client)
            self.logger.info(f"Connected to Milvus at {uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def ensure_collection_exists(
        self,
        collection_name: str,
        schema_metadata: dict[str, Any] | None = None,
        drop_if_exists: bool = False,
        use_flat_index: bool = True,
    ) -> bool:
        """Ensure collection exists, create if needed.

        Args:
            collection_name: Target collection name
            schema_metadata: Schema metadata for creating collection (from meta.json)
            drop_if_exists: Drop collection if it already exists
            use_flat_index: Use FLAT index for dense vector fields only (default: True, provides 100% recall)

        Returns:
            True if collection was created, False if it already existed
        """
        # Check if collection exists
        if self.client.has_collection(collection_name):
            if drop_if_exists:
                self.client.drop_collection(collection_name)
                self.logger.info(f"Dropped existing collection: {collection_name}")
            else:
                self.logger.info(f"Collection '{collection_name}' already exists")
                return False

        # Create collection if we have metadata
        if schema_metadata:
            # Use unified schema builder to create collection
            return self.schema_builder.create_collection_with_schema(
                collection_name, schema_metadata, drop_if_exists=False, use_flat_index=use_flat_index
            )
        else:
            raise ValueError(
                f"Collection '{collection_name}' does not exist and no schema metadata provided"
            )

    def _try_load_metadata(self, files: list[str]) -> dict[str, Any] | None:
        """Try to load metadata from file paths.

        Args:
            files: List of file paths

        Returns:
            Metadata dict if found, None otherwise
        """
        # Try to find meta.json in the same directory as data files
        for file_path in files:
            try:
                if file_path.startswith("s3://"):
                    # For S3 paths, we can't easily access meta.json
                    # This would require S3 client integration
                    continue

                path = Path(file_path)
                if path.is_dir():
                    # Check for meta.json in directory
                    meta_path = path / "meta.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            metadata: dict[str, Any] = json.load(f)
                        self.logger.info(f"Found metadata in {meta_path}")
                        return metadata
                else:
                    # Check for meta.json in same directory as file
                    meta_path = path.parent / "meta.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            file_metadata: dict[str, Any] = json.load(f)
                        self.logger.info(f"Found metadata in {meta_path}")
                        return file_metadata
            except Exception as e:
                self.logger.debug(f"Failed to load metadata from {file_path}: {e}")

        return None

    def bulk_import_files(
        self,
        collection_name: str,
        files: list[str],
        import_files: list[str] | None = None,
        show_progress: bool = True,
        create_collection: bool = True,
        drop_if_exists: bool = False,
        use_flat_index: bool = True,
    ) -> str:
        """Start bulk import job.

        Args:
            collection_name: Target collection name
            files: List of directory paths for metadata loading
            import_files: List of relative file paths to import (relative to bucket, supports parquet and json files)
            show_progress: Show progress bar
            create_collection: Try to create collection if it doesn't exist
            drop_if_exists: Drop collection if it already exists
            use_flat_index: Use FLAT index for dense vector fields only (default: True, provides 100% recall)

        Returns:
            Job ID for the import task
        """
        try:
            # Try to ensure collection exists
            if create_collection:
                # Try to load metadata from files
                metadata = self._try_load_metadata(files)

                if metadata:
                    # Use unified schema builder to create collection if needed
                    self.schema_builder.create_collection_with_schema(
                        collection_name, metadata, drop_if_exists=drop_if_exists, use_flat_index=use_flat_index
                    )
                else:
                    # Just check if collection exists
                    if not self.client.has_collection(collection_name):
                        self.logger.warning(
                            f"Collection '{collection_name}' does not exist and no metadata found. "
                            "Please create the collection first or ensure meta.json is available."
                        )
                        raise ValueError(
                            f"Collection '{collection_name}' does not exist. "
                            "Create it first using 'to-milvus insert' or provide meta.json"
                        )
            # Use import_files if provided, otherwise use files
            actual_import_files = import_files if import_files is not None else files

            # Log import preparation details
            self.logger.info(f"Preparing bulk import for collection: {collection_name}")
            self.logger.info(f"Target Milvus URI: {self.uri}")
            self.logger.info(f"Database: {self.db_name}")
            self.logger.info(f"Number of files to import: {len(actual_import_files)}")

            # Log file details
            for i, file_path in enumerate(actual_import_files, 1):
                file_ext = (
                    ".parquet"
                    if file_path.endswith(".parquet")
                    else ".json"
                    if file_path.endswith(".json")
                    else "unknown"
                )
                if file_path.startswith("s3://"):
                    self.logger.info(f"File {i}: {file_path} ({file_ext}, S3/MinIO)")
                else:
                    # For relative paths, just show the filename
                    self.logger.info(
                        f"File {i}: {file_path} ({file_ext}, relative to bucket)"
                    )

            # Prepare files as list of lists (each inner list is a batch)
            file_batches = [[f] for f in actual_import_files]
            self.logger.info(f"Organized files into {len(file_batches)} import batches")

            # Start bulk import using bulk_writer
            self.logger.info("Initiating bulk import request to Milvus...")
            resp = bulk_import(
                url=self.uri,
                collection_name=collection_name,
                files=file_batches,
            )

            # Extract job ID from response
            response_data = resp.json()
            job_id: str = response_data["data"]["jobId"]

            self.logger.info("✓ Bulk import request accepted successfully")
            self.logger.info(f"Job ID: {job_id}")
            self.logger.info(f"Collection: {collection_name}")
            self.logger.info(
                "Status: Import job queued and will be processed asynchronously"
            )

            return job_id

        except Exception as e:
            self.logger.error(f"Failed to start bulk import: {e}")
            self.logger.error(f"Collection: {collection_name}")
            self.logger.error(f"Files: {files}")
            raise

    def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 300,
        show_progress: bool = True,
    ) -> bool:
        """Wait for bulk import job to complete.

        Args:
            job_id: Import job ID
            timeout: Timeout in seconds
            show_progress: Show progress bar

        Returns:
            True if import completed successfully
        """
        start_time = time.time()

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=True,
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Waiting for import completion...", total=timeout
                )

                while time.time() - start_time < timeout:
                    # Check job status
                    resp = get_import_progress(
                        url=self.uri,
                        job_id=job_id,
                    )
                    job_info = resp.json()["data"]
                    state = job_info.get("state", "unknown")
                    progress_percent = job_info.get("progress", 0)
                    imported_rows = job_info.get("importedRows", 0)
                    total_rows = job_info.get("totalRows", 0)
                    file_size = job_info.get("fileSize", 0)

                    # Log detailed progress information
                    elapsed = time.time() - start_time
                    if int(elapsed) % 10 < 2:  # Log every 10 seconds
                        # Stop the progress temporarily and print logs
                        progress.stop()
                        self.logger.info(f"Import progress update for job {job_id}:")
                        self.logger.info(f"  State: {state}")
                        self.logger.info(f"  Progress: {progress_percent}%")
                        self.logger.info(
                            f"  Imported rows: {imported_rows:,} / {total_rows:,}"
                        )
                        self.logger.info(f"  File size processed: {file_size:,} bytes")
                        self.logger.info(f"  Elapsed time: {elapsed:.1f}s")
                        # Resume progress bar
                        progress.start()

                    if state == "ImportCompleted" or state == "Completed":
                        progress.update(task, completed=timeout)
                        # Stop progress before final logs
                        progress.stop()
                        self.logger.info("🎉 Bulk import completed successfully!")
                        self.logger.info(f"Job ID: {job_id}")
                        self.logger.info(f"Total rows imported: {imported_rows:,}")
                        self.logger.info(f"Total file size: {file_size:,} bytes")
                        self.logger.info(f"Total time: {elapsed:.2f}s")
                        if imported_rows > 0 and elapsed > 0:
                            rate = imported_rows / elapsed
                            self.logger.info(f"Import rate: {rate:.0f} rows/second")
                        return True
                    elif state == "ImportFailed" or state == "Failed":
                        progress.update(task, completed=timeout)
                        # Stop progress before error logs
                        progress.stop()
                        reason = job_info.get("reason", "Unknown error")
                        self.logger.error("❌ Bulk import failed!")
                        self.logger.error(f"Job ID: {job_id}")
                        self.logger.error(f"Failure reason: {reason}")
                        self.logger.error(f"State: {state}")
                        self.logger.error(f"Progress when failed: {progress_percent}%")
                        self.logger.error(
                            f"Rows imported before failure: {imported_rows:,}"
                        )
                        return False

                    # Update progress
                    progress.update(task, completed=min(elapsed, timeout))

                    time.sleep(2)

        else:
            # Wait without progress bar
            self.logger.info(f"Monitoring import job {job_id} (no progress bar)")
            last_log_time = 0.0

            while time.time() - start_time < timeout:
                resp = get_import_progress(
                    url=self.uri,
                    job_id=job_id,
                )
                job_info = resp.json()["data"]
                state = job_info.get("state", "unknown")
                progress_percent = job_info.get("progress", 0)
                imported_rows = job_info.get("importedRows", 0)
                total_rows = job_info.get("totalRows", 0)
                file_size = job_info.get("fileSize", 0)

                # Log detailed progress information every 10 seconds
                elapsed = time.time() - start_time
                if elapsed - last_log_time >= 10:
                    self.logger.info(f"Import progress update for job {job_id}:")
                    self.logger.info(f"  State: {state}")
                    self.logger.info(f"  Progress: {progress_percent}%")
                    self.logger.info(
                        f"  Imported rows: {imported_rows:,} / {total_rows:,}"
                    )
                    self.logger.info(f"  File size processed: {file_size:,} bytes")
                    self.logger.info(f"  Elapsed time: {elapsed:.1f}s")
                    last_log_time = elapsed

                if state == "ImportCompleted" or state == "Completed":
                    self.logger.info("🎉 Bulk import completed successfully!")
                    self.logger.info(f"Job ID: {job_id}")
                    self.logger.info(f"Total rows imported: {imported_rows:,}")
                    self.logger.info(f"Total file size: {file_size:,} bytes")
                    self.logger.info(f"Total time: {elapsed:.2f}s")
                    if imported_rows > 0 and elapsed > 0:
                        rate = imported_rows / elapsed
                        self.logger.info(f"Import rate: {rate:.0f} rows/second")
                    return True
                elif state == "ImportFailed" or state == "Failed":
                    reason = job_info.get("reason", "Unknown error")
                    self.logger.error("❌ Bulk import failed!")
                    self.logger.error(f"Job ID: {job_id}")
                    self.logger.error(f"Failure reason: {reason}")
                    self.logger.error(f"State: {state}")
                    self.logger.error(f"Progress when failed: {progress_percent}%")
                    self.logger.error(
                        f"Rows imported before failure: {imported_rows:,}"
                    )
                    return False

                time.sleep(2)

        # Timeout reached
        elapsed = time.time() - start_time
        self.logger.error(f"⏰ Bulk import timeout after {timeout} seconds")
        self.logger.error(f"Job ID: {job_id}")
        self.logger.error(f"Final state: {job_info.get('state', 'unknown')}")
        self.logger.error(f"Progress at timeout: {job_info.get('progress', 0)}%")
        return False

    def list_import_jobs(
        self,
        collection_name: str | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """List all import jobs.

        Args:
            collection_name: Filter by collection name
            show_progress: Show progress bar

        Returns:
            List of import job information
        """
        try:
            self.logger.info(f"Listing import jobs from Milvus: {self.uri}")
            if collection_name:
                self.logger.info(f"Filtering by collection: {collection_name}")
                resp = list_import_jobs(
                    url=self.uri,
                    collection_name=collection_name,
                )
            else:
                self.logger.info("Listing all import jobs")
                resp = list_import_jobs(
                    url=self.uri,
                )

            jobs: list[dict[str, Any]] = resp.json()["data"]["records"]

            self.logger.info(f"📋 Found {len(jobs)} import jobs")

            # Log summary of jobs by state
            if jobs:
                states: dict[str, int] = {}
                for job in jobs:
                    state = job.get("state", "unknown")
                    states[state] = states.get(state, 0) + 1

                self.logger.info("Job summary by state:")
                for state, count in states.items():
                    self.logger.info(f"  {state}: {count} jobs")

                # Log details of recent jobs
                recent_jobs = sorted(
                    jobs, key=lambda x: x.get("jobId", ""), reverse=True
                )[:5]
                self.logger.info(f"Recent {min(5, len(jobs))} jobs:")
                for job in recent_jobs:
                    job_id = job.get("jobId", "unknown")
                    state = job.get("state", "unknown")
                    collection = job.get("collectionName", "unknown")
                    imported_rows = job.get("importedRows", 0)
                    self.logger.info(
                        f"  Job {job_id}: {state} | Collection: {collection} | Rows: {imported_rows:,}"
                    )

            return jobs

        except Exception as e:
            self.logger.error(f"Failed to list import jobs: {e}")
            self.logger.error(f"URI: {self.uri}")
            if collection_name:
                self.logger.error(f"Collection filter: {collection_name}")
            raise

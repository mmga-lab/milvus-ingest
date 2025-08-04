"""Cache management for milvus-ingest data generation.

This module provides caching functionality to avoid regenerating identical datasets
when the same generation parameters are used multiple times.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .logging_config import get_logger

# Default cache directory: ~/.milvus-ingest/cache
DEFAULT_CACHE_DIR = Path.home() / ".milvus-ingest" / "cache"

# Cache version for compatibility management
CACHE_VERSION = "1.0"

# Parameters that affect data generation and should be included in cache key
CACHE_KEY_PARAMS = {
    "schema_content",  # The actual schema content (not path)
    "total_rows",
    "seed",
    "format",
    "batch_size",
    "file_size",
    "rows_per_file",
    "file_count",
    "num_partitions",
    "num_shards",
    "num_workers",  # Can affect random seed distribution
}

# Parameters that don't affect generation results and should be excluded
EXCLUDED_PARAMS = {
    "output_path",
    "preview",
    "no_progress",
    "verbose",
    "force",
    "validate_only",
}


class CacheManager:
    """Manages cache storage and retrieval for generated datasets."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize cache manager.

        Args:
            cache_dir: Directory to store cache. Defaults to ~/.milvus-ingest/cache
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def generate_cache_key(
        self, schema_path: Path, generation_params: dict[str, Any]
    ) -> str:
        """Generate a cache key from schema and generation parameters.

        Args:
            schema_path: Path to schema file
            generation_params: Dictionary of generation parameters

        Returns:
            SHA256 hash string to use as cache key
        """
        # Read schema content
        try:
            schema_content = schema_path.read_text("utf-8")
            # Parse and re-serialize to normalize JSON formatting
            schema_data = json.loads(schema_content)
            normalized_schema = json.dumps(
                schema_data, sort_keys=True, separators=(",", ":")
            )
        except Exception as e:
            self.logger.warning(f"Failed to read schema for cache key: {e}")
            normalized_schema = str(schema_path)

        # Filter parameters that affect generation
        cache_params = {}
        for key, value in generation_params.items():
            if key in CACHE_KEY_PARAMS and value is not None:
                cache_params[key] = value

        # Add schema content to params
        cache_params["schema_content"] = normalized_schema

        # Create deterministic string representation
        param_string = json.dumps(cache_params, sort_keys=True, separators=(",", ":"))

        # Generate SHA256 hash
        cache_key = hashlib.sha256(param_string.encode("utf-8")).hexdigest()

        self.logger.debug(f"Generated cache key: {cache_key}")
        self.logger.debug(f"Cache params: {cache_params}")

        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the cache directory path for a given cache key."""
        return self.cache_dir / cache_key

    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists and is valid for given cache key."""
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return False

        # Check for required files
        cache_info_file = cache_path / "cache_info.json"
        meta_file = cache_path / "meta.json"

        if not cache_info_file.exists() or not meta_file.exists():
            self.logger.warning(f"Cache {cache_key} missing required files")
            return False

        try:
            # Validate cache info
            with open(cache_info_file) as f:
                cache_info = json.load(f)

            # Check cache version compatibility
            if cache_info.get("cache_version") != CACHE_VERSION:
                self.logger.warning(f"Cache {cache_key} has incompatible version")
                return False

            # Check if data files exist
            data_files = cache_info.get("data_files", [])
            for data_file in data_files:
                if not (cache_path / data_file).exists():
                    self.logger.warning(
                        f"Cache {cache_key} missing data file: {data_file}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Failed to validate cache {cache_key}: {e}")
            return False

    def store_cache(
        self,
        cache_key: str,
        generated_files: list[str],
        meta_file: Path,
        generation_params: dict[str, Any],
    ) -> bool:
        """Store generated files in cache.

        Args:
            cache_key: Cache key to store under
            generated_files: List of paths to generated data files
            meta_file: Path to meta.json file
            generation_params: Parameters used for generation

        Returns:
            True if cache stored successfully, False otherwise
        """
        cache_path = self.get_cache_path(cache_key)

        try:
            # Create cache directory
            cache_path.mkdir(parents=True, exist_ok=True)

            # Copy data files
            data_files = []
            for file_path_str in generated_files:
                file_path = Path(file_path_str)
                if file_path.exists():
                    dest_file = cache_path / file_path.name
                    self._copy_file(file_path, dest_file)
                    data_files.append(file_path.name)

            # Copy meta.json
            if meta_file.exists():
                dest_meta = cache_path / "meta.json"
                self._copy_file(meta_file, dest_meta)

            # Create cache info
            cache_info = {
                "cache_version": CACHE_VERSION,
                "cache_key": cache_key,
                "created_at": time.time(),
                "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "data_files": data_files,
                "generation_params": generation_params,
            }

            # Write cache info
            cache_info_file = cache_path / "cache_info.json"
            with open(cache_info_file, "w") as f:
                json.dump(cache_info, f, indent=2)

            # Calculate cache size
            cache_size = sum(
                f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
            )
            cache_size_mb = cache_size / (1024 * 1024)

            self.logger.info(
                f"Stored cache {cache_key}: {len(data_files)} files, {cache_size_mb:.1f} MB"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to store cache {cache_key}: {e}")
            # Clean up partial cache on failure
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
            return False

    def retrieve_cache(
        self, cache_key: str, output_dir: Path
    ) -> tuple[bool, list[str], dict[str, Any] | None]:
        """Retrieve files from cache to output directory.

        Args:
            cache_key: Cache key to retrieve
            output_dir: Directory to copy cached files to

        Returns:
            Tuple of (success, list_of_copied_files, meta_json_data)
        """
        if not self.cache_exists(cache_key):
            return False, [], None

        cache_path = self.get_cache_path(cache_key)

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load cache info
            cache_info_file = cache_path / "cache_info.json"
            with open(cache_info_file) as f:
                cache_info = json.load(f)

            # Copy data files
            copied_files = []
            data_files = cache_info.get("data_files", [])

            for data_file in data_files:
                src_file = cache_path / data_file
                dest_file = output_dir / data_file

                if src_file.exists():
                    self._copy_file(src_file, dest_file)
                    copied_files.append(str(dest_file))

            # Copy meta.json
            src_meta = cache_path / "meta.json"
            dest_meta = output_dir / "meta.json"
            meta_data = None

            if src_meta.exists():
                self._copy_file(src_meta, dest_meta)
                with open(dest_meta) as f:
                    meta_data = json.load(f)

            self.logger.info(f"Retrieved cache {cache_key}: {len(copied_files)} files")
            return True, copied_files, meta_data

        except Exception as e:
            self.logger.error(f"Failed to retrieve cache {cache_key}: {e}")
            return False, [], None

    def list_caches(self) -> list[dict[str, Any]]:
        """List all available caches with their information.

        Returns:
            List of cache information dictionaries
        """
        caches: list[dict[str, Any]] = []

        if not self.cache_dir.exists():
            return caches

        for cache_dir in self.cache_dir.iterdir():
            if not cache_dir.is_dir():
                continue

            cache_info_file = cache_dir / "cache_info.json"
            if not cache_info_file.exists():
                continue

            try:
                with open(cache_info_file) as f:
                    cache_info = json.load(f)

                # Calculate cache size
                cache_size = sum(
                    f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
                )
                cache_info["size_bytes"] = cache_size
                cache_info["size_mb"] = cache_size / (1024 * 1024)

                caches.append(cache_info)

            except Exception as e:
                self.logger.warning(
                    f"Failed to read cache info for {cache_dir.name}: {e}"
                )
                continue

        # Sort by creation time (newest first)
        caches.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return caches

    def get_cache_info(self, cache_key: str) -> dict[str, Any] | None:
        """Get detailed information about a specific cache.

        Args:
            cache_key: Cache key to get info for

        Returns:
            Cache information dictionary or None if not found
        """
        cache_path = self.get_cache_path(cache_key)
        cache_info_file = cache_path / "cache_info.json"

        if not cache_info_file.exists():
            return None

        try:
            with open(cache_info_file) as f:
                cache_info: dict[str, Any] = json.load(f)

            # Add current cache size
            cache_size = sum(
                f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
            )
            cache_info["size_bytes"] = cache_size
            cache_info["size_mb"] = cache_size / (1024 * 1024)

            return cache_info

        except Exception as e:
            self.logger.error(f"Failed to get cache info for {cache_key}: {e}")
            return None

    def clear_cache(self, cache_key: str) -> bool:
        """Remove a specific cache.

        Args:
            cache_key: Cache key to remove

        Returns:
            True if successfully removed, False otherwise
        """
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return True

        try:
            shutil.rmtree(cache_path)
            self.logger.info(f"Cleared cache {cache_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear cache {cache_key}: {e}")
            return False

    def clear_all_caches(self) -> tuple[int, int]:
        """Remove all caches.

        Returns:
            Tuple of (successful_removals, failed_removals)
        """
        if not self.cache_dir.exists():
            return 0, 0

        successful = 0
        failed = 0

        for cache_dir in self.cache_dir.iterdir():
            if cache_dir.is_dir():
                if self.clear_cache(cache_dir.name):
                    successful += 1
                else:
                    failed += 1

        return successful, failed

    def get_cache_stats(self) -> dict[str, Any]:
        """Get overall cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                "total_caches": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "oldest_cache": None,
                "newest_cache": None,
            }

        caches = self.list_caches()

        total_size = sum(cache.get("size_bytes", 0) for cache in caches)

        stats = {
            "total_caches": len(caches),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_cache": caches[-1] if caches else None,
            "newest_cache": caches[0] if caches else None,
        }

        return stats

    def _copy_file(self, src: Path, dest: Path) -> None:
        """Copy a file using the most efficient method available.

        Tries hard link first (same filesystem), falls back to copy.
        """
        try:
            # Try hard link first (most efficient)
            if dest.exists():
                dest.unlink()
            os.link(src, dest)
            self.logger.debug(f"Hard linked {src} -> {dest}")
        except OSError:
            # Fall back to regular copy (different filesystem or permission issues)
            shutil.copy2(src, dest)
            self.logger.debug(f"Copied {src} -> {dest}")

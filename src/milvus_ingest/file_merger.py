"""File merger module for combining chunked data files."""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


class FileMerger:
    """Handles merging of multiple small files into larger ones."""

    def __init__(self, temp_dir: Path | None = None):
        """Initialize file merger.

        Args:
            temp_dir: Directory for temporary files. If None, uses system temp.
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())

    def merge_parquet_files(
        self,
        chunk_files: list[Path],
        output_file: Path,
        cleanup_chunks: bool = True,
    ) -> dict[str, Any]:
        """Merge multiple Parquet files into a single file.

        Args:
            chunk_files: List of Parquet chunk files to merge
            output_file: Path for the merged output file
            cleanup_chunks: Whether to delete chunk files after merging

        Returns:
            dict with merge statistics
        """
        if not chunk_files:
            raise ValueError("No chunk files provided for merging")

        start_time = time.time()
        total_rows = 0
        total_size_bytes = 0

        logger.info(f"Merging {len(chunk_files)} Parquet chunks into {output_file}")

        try:
            # Read all chunk files and combine into a single table
            tables = []
            for chunk_file in chunk_files:
                if not chunk_file.exists():
                    logger.warning(f"Chunk file not found: {chunk_file}")
                    continue

                table = pq.read_table(chunk_file)
                tables.append(table)
                total_rows += table.num_rows
                total_size_bytes += chunk_file.stat().st_size

            if not tables:
                raise ValueError("No valid chunk files found")

            # Concatenate all tables
            merged_table = pa.concat_tables(tables)

            # Write merged table with optimized settings
            pq.write_table(
                merged_table,
                output_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=False,
                row_group_size=min(50000, total_rows),
            )

            merge_time = time.time() - start_time
            output_size = output_file.stat().st_size

            # Cleanup chunk files if requested
            if cleanup_chunks:
                for chunk_file in chunk_files:
                    try:
                        chunk_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")

            logger.info(
                f"Parquet merge completed: {total_rows:,} rows, "
                f"{output_size / (1024**3):.1f}GB in {merge_time:.1f}s"
            )

            return {
                "chunks_merged": len(tables),
                "total_rows": total_rows,
                "input_size_bytes": total_size_bytes,
                "output_size_bytes": output_size,
                "merge_time_seconds": merge_time,
                "compression_ratio": total_size_bytes / output_size
                if output_size > 0
                else 1.0,
            }

        except Exception as e:
            logger.error(f"Failed to merge Parquet files: {e}")
            raise

    def merge_json_files(
        self,
        chunk_files: list[Path],
        output_file: Path,
        cleanup_chunks: bool = True,
    ) -> dict[str, Any]:
        """Merge multiple JSON files into a single file.

        Args:
            chunk_files: List of JSON chunk files to merge
            output_file: Path for the merged output file
            cleanup_chunks: Whether to delete chunk files after merging

        Returns:
            dict with merge statistics
        """
        if not chunk_files:
            raise ValueError("No chunk files provided for merging")

        start_time = time.time()
        total_rows = 0
        total_size_bytes = 0

        logger.info(f"Merging {len(chunk_files)} JSON chunks into {output_file}")

        try:
            # Read and combine all JSON data
            all_records = []

            for chunk_file in chunk_files:
                if not chunk_file.exists():
                    logger.warning(f"Chunk file not found: {chunk_file}")
                    continue

                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)

                # Handle both single objects and arrays
                if isinstance(chunk_data, list):
                    all_records.extend(chunk_data)
                    total_rows += len(chunk_data)
                else:
                    all_records.append(chunk_data)
                    total_rows += 1

                total_size_bytes += chunk_file.stat().st_size

            if not all_records:
                raise ValueError("No valid records found in chunk files")

            # Write merged JSON data
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, separators=(",", ":"))

            merge_time = time.time() - start_time
            output_size = output_file.stat().st_size

            # Cleanup chunk files if requested
            if cleanup_chunks:
                for chunk_file in chunk_files:
                    try:
                        chunk_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")

            logger.info(
                f"JSON merge completed: {total_rows:,} records, "
                f"{output_size / (1024**3):.1f}GB in {merge_time:.1f}s"
            )

            return {
                "chunks_merged": len(chunk_files),
                "total_rows": total_rows,
                "input_size_bytes": total_size_bytes,
                "output_size_bytes": output_size,
                "merge_time_seconds": merge_time,
                "compression_ratio": total_size_bytes / output_size
                if output_size > 0
                else 1.0,
            }

        except Exception as e:
            logger.error(f"Failed to merge JSON files: {e}")
            raise

    def merge_files(
        self,
        chunk_files: list[Path],
        output_file: Path,
        file_format: str,
        cleanup_chunks: bool = True,
    ) -> dict[str, Any]:
        """Merge files based on format.

        Args:
            chunk_files: List of chunk files to merge
            output_file: Path for the merged output file
            file_format: File format ('parquet' or 'json')
            cleanup_chunks: Whether to delete chunk files after merging

        Returns:
            dict with merge statistics
        """
        if file_format.lower() == "parquet":
            return self.merge_parquet_files(chunk_files, output_file, cleanup_chunks)
        elif file_format.lower() == "json":
            return self.merge_json_files(chunk_files, output_file, cleanup_chunks)
        else:
            raise ValueError(f"Unsupported file format for merging: {file_format}")


def merge_chunked_files(
    chunk_dir: Path,
    final_output_file: Path,
    file_format: str,
    cleanup_chunks: bool = True,
) -> dict[str, Any]:
    """Convenience function to merge all chunk files in a directory.

    Args:
        chunk_dir: Directory containing chunk files
        final_output_file: Path for the final merged file
        file_format: File format ('parquet' or 'json')
        cleanup_chunks: Whether to delete chunk files after merging

    Returns:
        dict with merge statistics
    """
    # Find all chunk files in the directory
    if file_format.lower() == "parquet":
        pattern = "chunk-*.parquet"
    elif file_format.lower() == "json":
        pattern = "chunk-*.json"
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    chunk_files = sorted(chunk_dir.glob(pattern))

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_dir} with pattern {pattern}")

    merger = FileMerger()
    return merger.merge_files(
        chunk_files, final_output_file, file_format, cleanup_chunks
    )

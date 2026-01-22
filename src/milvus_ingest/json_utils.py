"""Unified JSON loading utilities for milvus-ingest.

This module provides consistent JSON loading logic that handles multiple formats:
- JSON array format: [{"field": "value"}, ...]
- Legacy Milvus format: {"rows": [...]}
- Single object format: {"field": "value"}
- JSONL format: one JSON object per line
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonLoadError(Exception):
    """Raised when JSON loading fails."""

    pass


def load_json_data(file_path: Path | str) -> list[dict[str, Any]]:
    """Load data from a JSON file, supporting multiple formats.

    Supports the following formats:
    - JSON array: [{"field": "value"}, ...]
    - Legacy Milvus bulk import: {"rows": [...]}
    - Single JSON object: {"field": "value"} -> wrapped in list
    - JSONL (line-delimited): one JSON object per line

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of dictionaries containing the data.

    Raises:
        JsonLoadError: If the file cannot be read or parsed.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise JsonLoadError(f"Empty JSON file: {file_path}")

        return _parse_json_content(content)

    except json.JSONDecodeError as e:
        raise JsonLoadError(f"Invalid JSON in file {file_path}: {e}") from e
    except UnicodeDecodeError as e:
        raise JsonLoadError(f"Encoding error reading {file_path}: {e}") from e


def load_json_sample(
    file_path: Path | str, sample_size: int
) -> list[dict[str, Any]]:
    """Load a sample of data from a JSON file.

    Args:
        file_path: Path to the JSON file.
        sample_size: Maximum number of records to return.

    Returns:
        List of dictionaries, limited to sample_size records.
    """
    data = load_json_data(file_path)
    return data[:sample_size]


def _parse_json_content(content: str) -> list[dict[str, Any]]:
    """Parse JSON content string into a list of dictionaries.

    Args:
        content: JSON content string (already stripped).

    Returns:
        List of dictionaries.

    Raises:
        JsonLoadError: If the content format is not recognized.
    """
    if content.startswith("["):
        # JSON array format (list of dict) - Milvus bulk import format
        data = json.loads(content)
        if not isinstance(data, list):
            raise JsonLoadError("Expected JSON array but got different type")
        return data

    elif content.startswith("{"):
        # Could be legacy format with "rows" key or single object
        data = json.loads(content)
        if not isinstance(data, dict):
            raise JsonLoadError("Expected JSON object but got different type")

        if "rows" in data and isinstance(data["rows"], list):
            # Legacy Milvus bulk import format: {"rows": [...]}
            return data["rows"]
        else:
            # Single JSON object, wrap in list
            return [data]

    else:
        # Try line-delimited JSON (JSONL)
        return _parse_jsonl_content(content)


def _parse_jsonl_content(content: str) -> list[dict[str, Any]]:
    """Parse JSONL (line-delimited JSON) content.

    Args:
        content: JSONL content string.

    Returns:
        List of dictionaries.

    Raises:
        JsonLoadError: If no valid JSON lines found.
    """
    data_list: list[dict[str, Any]] = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                data_list.append(obj)
            else:
                raise JsonLoadError(
                    f"Line {line_num}: Expected JSON object, got {type(obj).__name__}"
                )
        except json.JSONDecodeError as e:
            raise JsonLoadError(f"Line {line_num}: Invalid JSON - {e}") from e

    if not data_list:
        raise JsonLoadError("No valid JSON data found in file")

    return data_list


def parse_json_string(content: str) -> list[dict[str, Any]]:
    """Parse a JSON string directly without file I/O.

    Useful for testing or when JSON content is already in memory.

    Args:
        content: JSON content string.

    Returns:
        List of dictionaries.
    """
    content = content.strip()
    if not content:
        raise JsonLoadError("Empty JSON content")
    return _parse_json_content(content)


__all__ = [
    "JsonLoadError",
    "load_json_data",
    "load_json_sample",
    "parse_json_string",
]

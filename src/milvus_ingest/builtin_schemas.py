"""Built-in schema management for milvus-ingest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import validate_schema_data

# Path to built-in schemas directory
SCHEMAS_DIR = Path(__file__).parent / "schemas"

# Schema metadata
BUILTIN_SCHEMAS = {
    "product_catalog": {
        "name": "Product Catalog",
        "description": "Simple product catalog for getting started with auto_id",
        "use_cases": ["Learning", "Basic product search", "Auto ID demonstration"],
        "fields_count": 4,
        "vector_dims": [128],
        "file": "product_catalog.json",
    },
    "ecommerce_search": {
        "name": "E-commerce Search",
        "description": "E-commerce product search with nullable fields",
        "use_cases": ["Product search", "E-commerce", "Nullable field handling"],
        "fields_count": 5,
        "vector_dims": [256],
        "file": "ecommerce_search.json",
    },
    "news_articles": {
        "name": "News Articles",
        "description": "News article storage with dynamic fields",
        "use_cases": ["News search", "Dynamic schema", "Content management"],
        "fields_count": 4,
        "vector_dims": [768],
        "file": "news_articles.json",
    },
    "document_search": {
        "name": "Document Search",
        "description": "Document search with user-provided sparse vectors and BM25",
        "use_cases": ["Sparse vector search", "BM25 functions", "Hybrid search"],
        "fields_count": 5,
        "vector_dims": [768],
        "file": "document_search.json",
    },
    "multi_tenant_data": {
        "name": "Multi-tenant Support",
        "description": "Multi-tenant customer support system with partitioning",
        "use_cases": ["Multi-tenant SaaS", "Partition keys", "Support tickets"],
        "fields_count": 5,
        "vector_dims": [256],
        "file": "multi_tenant_data.json",
    },
    "multimedia_content": {
        "name": "Multimedia Gallery",
        "description": "Multimedia content with multiple vector types and nullable fields",
        "use_cases": ["Media search", "Multiple vector types", "Default values"],
        "fields_count": 7,
        "vector_dims": [256, 384, 128],
        "file": "multimedia_content.json",
    },
}


def list_builtin_schemas() -> dict[str, dict[str, Any]]:
    """Get list of all built-in schemas with metadata.

    Returns:
        Dictionary mapping schema IDs to their metadata.
    """
    return BUILTIN_SCHEMAS.copy()


def get_schema_info(schema_id: str) -> dict[str, Any] | None:
    """Get metadata for a specific schema.

    Args:
        schema_id: The schema identifier.

    Returns:
        Schema metadata dictionary or None if not found.
    """
    return BUILTIN_SCHEMAS.get(schema_id)


def load_builtin_schema(schema_id: str) -> dict[str, Any]:
    """Load a built-in schema by ID.

    Args:
        schema_id: The schema identifier (e.g., 'ecommerce', 'documents').

    Returns:
        The schema dictionary.

    Raises:
        ValueError: If schema ID is not found.
        FileNotFoundError: If schema file is missing.
    """
    if schema_id not in BUILTIN_SCHEMAS:
        available = ", ".join(BUILTIN_SCHEMAS.keys())
        raise ValueError(
            f"Unknown schema ID '{schema_id}'. Available schemas: {available}"
        )

    schema_file = SCHEMAS_DIR / BUILTIN_SCHEMAS[schema_id]["file"]  # type: ignore[operator]
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    try:
        with open(schema_file, encoding="utf-8") as f:
            schema_data = json.load(f)

        # Validate the schema to ensure it's correct
        validate_schema_data(schema_data)

        return schema_data  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file {schema_file}: {e}") from e


def save_schema_to_file(schema_data: dict[str, Any], output_path: str | Path) -> None:
    """Save a schema to a file.

    Args:
        schema_data: The schema dictionary to save.
        output_path: Path where to save the schema file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=2, ensure_ascii=False)


def get_schema_summary() -> str:
    """Get a formatted summary of all built-in schemas.

    Returns:
        Formatted string describing all available schemas.
    """
    summary = "# Built-in Schema Collection\n\n"
    summary += "The following schemas are available for quick data generation:\n\n"

    for schema_id, info in BUILTIN_SCHEMAS.items():
        summary += f"## {info['name']} (`{schema_id}`)\n"
        summary += f"**Description:** {info['description']}\n\n"
        summary += f"**Use Cases:** {', '.join(info['use_cases'])}\n\n"  # type: ignore[arg-type]
        summary += f"**Fields:** {info['fields_count']} fields\n\n"

        if info["vector_dims"]:
            dims_str = ", ".join(map(str, info["vector_dims"]))  # type: ignore[call-overload]
            summary += f"**Vector Dimensions:** {dims_str}\n\n"

        summary += "**Usage:**\n"
        summary += "```bash\n"
        summary += "# Use built-in schema\n"
        summary += f"milvus-ingest generate --builtin {schema_id} --total-rows 1000\n\n"
        summary += "# Save schema to file for customization\n"
        summary += f"milvus-ingest schema show {schema_id} > my_{schema_id}.json\n"
        summary += "```\n\n"
        summary += "---\n\n"

    return summary


def validate_all_builtin_schemas() -> list[str]:
    """Validate all built-in schemas.

    Returns:
        List of any validation errors found.
    """
    errors = []

    for schema_id in BUILTIN_SCHEMAS:
        try:
            load_builtin_schema(schema_id)
        except Exception as e:
            errors.append(f"Schema '{schema_id}': {e}")

    return errors

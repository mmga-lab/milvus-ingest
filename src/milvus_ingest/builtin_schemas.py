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
    "quickstart": {
        "name": "Quick Start",
        "description": "Basic schema for getting started with essential fields (uses auto_id=true)",
        "use_cases": ["Learning", "Testing", "Quick start", "Auto ID demonstration"],
        "fields_count": 6,
        "vector_dims": [128],
        "file": "quickstart.json",
    },
    "dynamic_fields": {
        "name": "Dynamic Fields",
        "description": "Schema demonstrating dynamic field capabilities with VarChar auto_id for flexible data insertion",
        "use_cases": [
            "Dynamic data",
            "Schema evolution",
            "Flexible fields",
            "VarChar auto ID demonstration",
        ],
        "fields_count": 4,
        "vector_dims": [384],
        "file": "dynamic_fields.json",
    },
    "ecommerce": {
        "name": "E-commerce Products",
        "description": "Product catalog with multiple embeddings for search and recommendation",
        "use_cases": ["Product search", "Recommendation systems", "Catalog management"],
        "fields_count": 12,
        "vector_dims": [768, 512],
        "file": "ecommerce.json",
    },
    "documents": {
        "name": "Document Search",
        "description": "Document collection with semantic search capabilities",
        "use_cases": ["Document search", "Knowledge base", "Content management"],
        "fields_count": 12,
        "vector_dims": [1536],
        "file": "documents.json",
    },
    "images": {
        "name": "Image Gallery",
        "description": "Image collection with visual similarity search",
        "use_cases": ["Image search", "Visual similarity", "Photo management"],
        "fields_count": 14,
        "vector_dims": [2048, 512],
        "file": "images.json",
    },
    "users": {
        "name": "User Profiles",
        "description": "User profiles with behavioral embeddings for recommendation",
        "use_cases": ["User recommendation", "Personalization", "Customer analytics"],
        "fields_count": 15,
        "vector_dims": [256],
        "file": "users.json",
    },
    "videos": {
        "name": "Video Library",
        "description": "Video content with multimodal embeddings",
        "use_cases": ["Video search", "Content recommendation", "Media management"],
        "fields_count": 18,
        "vector_dims": [512, 1024],
        "file": "videos.json",
    },
    "news": {
        "name": "News Articles",
        "description": "News articles with semantic search and sentiment analysis",
        "use_cases": ["News search", "Content analysis", "Media monitoring"],
        "fields_count": 19,
        "vector_dims": [384, 768],
        "file": "news.json",
    },
    "ecommerce_partitioned": {
        "name": "E-commerce Multi-tenant",
        "description": "Multi-tenant e-commerce catalog with partition key for tenant isolation",
        "use_cases": [
            "Multi-tenant SaaS",
            "Partitioned product search",
            "Tenant data isolation",
        ],
        "fields_count": 13,
        "vector_dims": [768],
        "file": "ecommerce_partitioned.json",
    },
    "data_distribution": {
        "name": "Data Distribution",
        "description": "Schema showcasing different data distribution patterns using cardinality_ratio and enum_values",
        "use_cases": [
            "Data distribution testing",
            "Performance benchmarking",
            "Schema design patterns",
        ],
        "fields_count": 10,
        "vector_dims": [128],
        "file": "data_distribution.json",
    },
    "audio_transcripts": {
        "name": "Audio Transcripts",
        "description": "Audio transcription service with semantic search using Float16Vector embeddings",
        "use_cases": [
            "Speech-to-text search",
            "Podcast transcription",
            "Meeting recordings",
        ],
        "fields_count": 10,
        "vector_dims": [768],
        "file": "audio_transcripts.json",
    },
    "ai_conversations": {
        "name": "AI Conversations",
        "description": "AI chat conversation history with BFloat16Vector embeddings for semantic search",
        "use_cases": [
            "Chatbot analytics",
            "Conversation search",
            "AI interaction history",
        ],
        "fields_count": 13,
        "vector_dims": [1024],
        "file": "ai_conversations.json",
    },
    "face_recognition": {
        "name": "Face Recognition",
        "description": "Facial recognition system with BinaryVector for efficient biometric matching",
        "use_cases": [
            "Security systems",
            "Identity verification",
            "Access control",
        ],
        "fields_count": 14,
        "vector_dims": [512, 256],
        "file": "face_recognition.json",
    },
    "default_values": {
        "name": "Default Values",
        "description": "Schema with default_value field parameters for handling missing data during insertion",
        "use_cases": [
            "Data migration with missing values",
            "Flexible data insertion",
            "Schema with fallback values",
        ],
        "fields_count": 9,
        "vector_dims": [768],
        "file": "default_values.json",
    },
    "full_text_search": {
        "name": "Full-Text Search",
        "description": "Full-text search with BM25 function and semantic embeddings for hybrid search",
        "use_cases": [
            "Full-text search",
            "Hybrid search (BM25 + semantic)",
            "Document retrieval",
            "Keyword search",
        ],
        "fields_count": 11,
        "vector_dims": [768],
        "file": "full_text_search.json",
    },
    "text_search_advanced": {
        "name": "Advanced Text Search",
        "description": "Comprehensive BM25 functions with multiple text fields for advanced full-text search",
        "use_cases": [
            "Advanced BM25 search",
            "Multi-field full-text search",
            "Hybrid search systems",
            "Advanced text retrieval",
            "Search relevance optimization",
        ],
        "fields_count": 17,
        "vector_dims": [768],
        "file": "text_search_advanced.json",
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
        summary += f"milvus-ingest --builtin {schema_id} --rows 1000\n\n"
        summary += "# Save schema to file for customization\n"
        summary += (
            f"milvus-ingest --builtin {schema_id} --save-schema my_{schema_id}.json\n"
        )
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

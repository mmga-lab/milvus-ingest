"""Unified Milvus schema builder to eliminate code duplication between inserter and importer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymilvus import DataType, Function, FunctionType

from .logging_config import get_logger

if TYPE_CHECKING:
    from pymilvus import MilvusClient


class MilvusSchemaBuilder:
    """Unified schema builder for Milvus collections."""

    def __init__(self, client: MilvusClient):
        """Initialize schema builder with Milvus client.

        Args:
            client: MilvusClient instance
        """
        self.client = client
        self.logger = get_logger(__name__)

    def create_schema_from_metadata(self, metadata: dict[str, Any]) -> Any:
        """Create Milvus collection schema from metadata.

        Args:
            metadata: Schema metadata containing field definitions and functions

        Returns:
            Milvus schema object
        """
        # Use MilvusClient's create_schema method
        enable_dynamic = metadata["schema"].get("enable_dynamic_field", False)
        schema = self.client.create_schema(enable_dynamic_field=enable_dynamic)

        # Identify BM25 input fields that need enable_analyzer=True
        bm25_input_fields = set()
        if "functions" in metadata["schema"]:
            for func_info in metadata["schema"]["functions"]:
                if func_info["type"] == "BM25":
                    bm25_input_fields.update(func_info["input_field_names"])

        for field_info in metadata["schema"]["fields"]:
            field_name = field_info["name"]
            field_type = field_info["type"]

            # Map field type to Milvus DataType
            milvus_type = self._get_milvus_datatype(field_type)

            # Add field to schema
            if field_type in ["VarChar", "String"]:
                kwargs = {
                    "field_name": field_name,
                    "datatype": milvus_type,
                    "max_length": field_info.get("max_length", 65535),
                    "is_primary": field_info.get("is_primary", False),
                    "auto_id": field_info.get("auto_id", False),
                }
                # Enable analyzer for BM25 input fields
                if field_name in bm25_input_fields:
                    kwargs["enable_analyzer"] = True
                # Add nullable and default_value if present
                if "nullable" in field_info:
                    kwargs["nullable"] = field_info["nullable"]
                if "default_value" in field_info:
                    kwargs["default_value"] = self._convert_default_value(
                        field_info["default_value"], field_type
                    )
                schema.add_field(**kwargs)
            elif "Vector" in field_type:
                # SparseFloatVector doesn't need dim parameter
                if field_type == "SparseFloatVector":
                    kwargs = {
                        "field_name": field_name,
                        "datatype": milvus_type,
                        "is_primary": field_info.get("is_primary", False),
                        "auto_id": field_info.get("auto_id", False),
                    }
                else:
                    kwargs = {
                        "field_name": field_name,
                        "datatype": milvus_type,
                        "dim": field_info.get("dim"),
                        "is_primary": field_info.get("is_primary", False),
                        "auto_id": field_info.get("auto_id", False),
                    }
                # Vector fields typically don't have nullable/default_value, but add if present
                if "nullable" in field_info:
                    kwargs["nullable"] = field_info["nullable"]
                if "default_value" in field_info:
                    kwargs["default_value"] = field_info["default_value"]
                schema.add_field(**kwargs)
            elif field_type == "Array":
                kwargs = {
                    "field_name": field_name,
                    "datatype": milvus_type,
                    "max_capacity": field_info.get("max_capacity"),
                    "element_type": self._get_milvus_datatype(
                        field_info.get("element_type")
                    )
                    if field_info.get("element_type")
                    else None,
                    "is_primary": field_info.get("is_primary", False),
                    "auto_id": field_info.get("auto_id", False),
                }
                # Add max_length for VarChar element type
                if field_info.get("element_type") in ["VarChar", "String"]:
                    kwargs["max_length"] = field_info.get("max_length", 65535)
                # Add nullable and default_value if present
                if "nullable" in field_info:
                    kwargs["nullable"] = field_info["nullable"]
                if "default_value" in field_info:
                    kwargs["default_value"] = self._convert_default_value(
                        field_info["default_value"], field_type
                    )
                schema.add_field(**kwargs)
            else:
                # Scalar fields (Int64, Float, Bool, etc.)
                kwargs = {
                    "field_name": field_name,
                    "datatype": milvus_type,
                    "is_primary": field_info.get("is_primary", False),
                    "auto_id": field_info.get("auto_id", False),
                }
                # Add nullable and default_value if present
                if "nullable" in field_info:
                    kwargs["nullable"] = field_info["nullable"]
                if "default_value" in field_info:
                    kwargs["default_value"] = self._convert_default_value(
                        field_info["default_value"], field_type
                    )
                schema.add_field(**kwargs)

        # Add functions if defined in schema
        if "functions" in metadata["schema"]:
            for func_info in metadata["schema"]["functions"]:
                if func_info["type"] == "BM25":
                    # Create BM25 function
                    bm25_function = Function(
                        name=func_info["name"],
                        input_field_names=func_info["input_field_names"],
                        output_field_names=func_info["output_field_names"],
                        function_type=FunctionType.BM25,
                    )
                    schema.add_function(bm25_function)
                    self.logger.info(f"Added BM25 function: {func_info['name']}")

        return schema

    def create_index_params_from_metadata(self, metadata: dict[str, Any]) -> Any:
        """Create index parameters from metadata.

        Args:
            metadata: Schema metadata containing field definitions

        Returns:
            Milvus index parameters object
        """
        # Use MilvusClient's prepare_index_params method
        index_params = self.client.prepare_index_params()

        for field_info in metadata["schema"]["fields"]:
            if "Vector" in field_info["type"]:
                field_name = field_info["name"]
                field_type = field_info["type"]

                if field_type == "SparseFloatVector":
                    # Determine metric type based on whether this field is BM25 function output
                    metric_type = self._get_sparse_vector_metric_type(
                        field_name, metadata
                    )

                    # Add sparse vector index
                    index_params.add_index(
                        field_name=field_name,
                        index_name=f"sparse_inverted_index_{field_name}",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type=metric_type,
                        params={"drop_ratio_build": 0.2},
                    )
                else:
                    # Determine metric type based on vector type
                    metric_type = "HAMMING" if field_type == "BinaryVector" else "L2"

                    # Add index for vector field
                    index_params.add_index(
                        field_name=field_name, metric_type=metric_type
                    )

        return index_params

    def create_collection_with_schema(
        self,
        collection_name: str,
        metadata: dict[str, Any],
        drop_if_exists: bool = False,
    ) -> bool:
        """Create collection with schema from metadata.

        Args:
            collection_name: Name of the collection to create
            metadata: Schema metadata
            drop_if_exists: Whether to drop existing collection

        Returns:
            True if collection was created, False if it already existed
        """
        # Check if collection exists
        collection_exists = self.client.has_collection(collection_name)

        if collection_exists:
            if drop_if_exists:
                self.client.drop_collection(collection_name)
                self.logger.info(f"Dropped existing collection: {collection_name}")
                collection_exists = False
            else:
                self.logger.info(
                    f"Collection '{collection_name}' already exists. "
                    "Skipping collection creation."
                )
                return False

        # Create collection only if it doesn't exist
        if not collection_exists:
            # Create collection schema
            schema = self.create_schema_from_metadata(metadata)

            # Create index params
            index_params = self.create_index_params_from_metadata(metadata)

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            self.logger.info(f"Created collection: {collection_name}")
            return True

        return False

    def get_index_info_from_metadata(
        self, collection_name: str, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get index information from metadata for return values.

        Args:
            collection_name: Collection name
            metadata: Schema metadata

        Returns:
            List of index information dictionaries
        """
        index_info = []

        for field_info in metadata["schema"]["fields"]:
            if "Vector" in field_info["type"]:
                field_name = field_info["name"]
                field_type = field_info["type"]

                if field_type == "SparseFloatVector":
                    metric_type = self._get_sparse_vector_metric_type(
                        field_name, metadata
                    )
                    index_info.append(
                        {
                            "field": field_name,
                            "index_type": "SPARSE_INVERTED_INDEX",
                            "metric_type": metric_type,
                        }
                    )
                else:
                    # Determine metric type based on vector type
                    metric_type = "HAMMING" if field_type == "BinaryVector" else "L2"

                    index_info.append(
                        {
                            "field": field_name,
                            "index_type": "AUTOINDEX",  # MilvusClient uses AUTOINDEX by default
                            "metric_type": metric_type,
                        }
                    )

        return index_info

    def _convert_default_value(self, default_value: Any, field_type: str) -> Any:
        """Convert default value to correct type for Milvus schema.

        Args:
            default_value: The default value to convert
            field_type: The field type

        Returns:
            Converted default value
        """
        if default_value is None:
            return None

        # Type conversion for default values
        if field_type == "Float":
            # Milvus Float (DataType.FLOAT) expects float32
            import numpy as np

            return np.float32(default_value)
        elif field_type == "Double":
            return float(default_value)
        elif field_type in ["Int64", "Int32", "Int16", "Int8"]:
            return int(default_value)
        elif field_type == "Bool":
            return bool(default_value)
        elif field_type in ["String", "VarChar"]:
            return str(default_value)
        else:
            # For other types, return as-is
            return default_value

    def _get_sparse_vector_metric_type(
        self, field_name: str, metadata: dict[str, Any]
    ) -> str:
        """Determine the correct metric type for SparseFloatVector based on BM25 function usage.

        Args:
            field_name: The name of the SparseFloatVector field
            metadata: Schema metadata containing functions information

        Returns:
            "BM25" if field is output of BM25 function, "IP" otherwise
        """
        # Check if this field is output of a BM25 function
        functions = metadata.get("schema", {}).get("functions", [])
        for func in functions:
            if func.get("type") == "BM25" and field_name in func.get(
                "output_field_names", []
            ):
                return "BM25"

        # Default to IP for non-BM25 sparse vectors (e.g., from sparse neural models)
        return "IP"

    def _get_milvus_datatype(self, field_type: str) -> DataType:
        """Map field type string to Milvus DataType.

        Args:
            field_type: String representation of field type

        Returns:
            Milvus DataType enum value
        """
        type_mapping = {
            "Bool": DataType.BOOL,
            "Int8": DataType.INT8,
            "Int16": DataType.INT16,
            "Int32": DataType.INT32,
            "Int64": DataType.INT64,
            "Float": DataType.FLOAT,
            "Double": DataType.DOUBLE,
            "String": DataType.VARCHAR,
            "VarChar": DataType.VARCHAR,
            "JSON": DataType.JSON,
            "Array": DataType.ARRAY,
            "FloatVector": DataType.FLOAT_VECTOR,
            "BinaryVector": DataType.BINARY_VECTOR,
            "Float16Vector": DataType.FLOAT16_VECTOR,
            "BFloat16Vector": DataType.BFLOAT16_VECTOR,
            "SparseFloatVector": DataType.SPARSE_FLOAT_VECTOR,
        }

        if field_type not in type_mapping:
            raise ValueError(f"Unknown field type: {field_type}")

        return type_mapping[field_type]

"""Insert generated data directly to Milvus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from pymilvus import MilvusClient
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .logging_config import get_logger
from .milvus_schema_builder import MilvusSchemaBuilder
from .rich_display import display_error

if TYPE_CHECKING:
    from pathlib import Path


class MilvusInserter:
    """Handle inserting data to Milvus."""

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
        self.logger = get_logger(__name__)
        self.uri = uri
        self.db_name = db_name

        try:
            # Use MilvusClient
            self.client = MilvusClient(
                uri=uri,
                token=token,
                db_name=db_name,
            )
            # Initialize schema builder
            self.schema_builder = MilvusSchemaBuilder(self.client)
            self.logger.info(
                f"Connected to Milvus at {uri}", extra={"db_name": db_name}
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def insert_data(
        self,
        data_path: Path,
        collection_name: str | None = None,
        drop_if_exists: bool = False,
        create_index: bool = True,
        batch_size: int = 10000,
        show_progress: bool = True,
        use_flat_index: bool = True,
    ) -> dict[str, Any]:
        """Insert data from generated files to Milvus.

        Args:
            data_path: Path to the data directory containing parquet files and meta.json
            collection_name: Override collection name from meta.json
            drop_if_exists: Drop collection if it already exists
            create_index: Create index on vector fields after insert
            batch_size: Batch size for inserting data
            show_progress: Show progress bar
            use_flat_index: Use FLAT index for dense vector fields only (default: True, provides 100% recall but uses more memory)

        Returns:
            Dictionary with insert statistics
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Load metadata
        meta_path = data_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {data_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Get collection name
        final_collection_name = collection_name or metadata["schema"]["collection_name"]

        # Check if collection exists
        collection_exists = self.client.has_collection(final_collection_name)

        if collection_exists:
            if drop_if_exists:
                self.client.drop_collection(final_collection_name)
                self.logger.info(
                    f"Dropped existing collection: {final_collection_name}"
                )
                collection_exists = False
            else:
                self.logger.info(
                    f"Collection '{final_collection_name}' already exists. "
                    "Skipping collection creation and will insert data into existing collection."
                )

        # Create collection only if it doesn't exist
        if not collection_exists:
            # Use unified schema builder to create collection
            self.schema_builder.create_collection_with_schema(
                final_collection_name, metadata, drop_if_exists=False, use_flat_index=use_flat_index
            )

        # Find data files (parquet or json)
        data_files = []
        parquet_files = sorted(data_path.glob("*.parquet"))
        json_files = sorted(data_path.glob("*.json"))

        # Exclude meta.json from json files
        json_files = [f for f in json_files if f.name != "meta.json"]

        if parquet_files:
            data_files = parquet_files
            file_format = "parquet"
        elif json_files:
            data_files = json_files
            file_format = "json"
        else:
            raise FileNotFoundError(
                f"No parquet or json data files found in {data_path}"
            )

        self.logger.info(f"Found {len(data_files)} {file_format} file(s) to process")

        # Insert data from all data files
        total_inserted = 0
        failed_batches = []

        for data_file in data_files:
            self.logger.info(f"Processing {data_file.name}")

            # Read data file based on format
            if file_format == "parquet":
                df = pd.read_parquet(data_file)
                data_source = df
                total_rows = len(df)
            else:  # json format
                data_list = self._read_json_file(data_file)
                data_source = data_list
                total_rows = len(data_list)

            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                ) as progress:
                    task = progress.add_task(
                        f"Inserting {data_file.name}", total=total_rows
                    )

                    # Insert in batches
                    for i in range(0, total_rows, batch_size):
                        if file_format == "parquet":
                            batch_df = data_source.iloc[i : i + batch_size]
                            try:
                                # Convert DataFrame to list of dictionaries
                                data = self._convert_dataframe_to_dict_list(
                                    batch_df, metadata
                                )
                                batch_size_actual = len(batch_df)
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to convert batch {i // batch_size}: {e}"
                                )
                                failed_batches.append(
                                    {
                                        "file": data_file.name,
                                        "batch": i // batch_size,
                                        "error": str(e),
                                    }
                                )
                                progress.update(task, advance=batch_size)
                                continue
                        else:  # json format
                            # For JSON, data is already in the correct format
                            batch_data = data_source[i : i + batch_size]
                            data = self._process_json_batch(batch_data, metadata)
                            batch_size_actual = len(batch_data)

                        try:
                            # Insert using MilvusClient
                            self.client.insert(
                                collection_name=final_collection_name, data=data
                            )
                            total_inserted += batch_size_actual
                            progress.update(task, advance=batch_size_actual)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to insert batch {i // batch_size}: {e}"
                            )
                            failed_batches.append(
                                {
                                    "file": data_file.name,
                                    "batch": i // batch_size,
                                    "error": str(e),
                                }
                            )
                            progress.update(task, advance=batch_size_actual)
            else:
                # Insert without progress bar
                for i in range(0, total_rows, batch_size):
                    if file_format == "parquet":
                        batch_df = data_source.iloc[i : i + batch_size]
                        try:
                            # Convert DataFrame to list of dictionaries
                            data = self._convert_dataframe_to_dict_list(
                                batch_df, metadata
                            )
                            batch_size_actual = len(batch_df)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to convert batch {i // batch_size}: {e}"
                            )
                            failed_batches.append(
                                {
                                    "file": data_file.name,
                                    "batch": i // batch_size,
                                    "error": str(e),
                                }
                            )
                            continue
                    else:  # json format
                        # For JSON, data is already in the correct format
                        batch_data = data_source[i : i + batch_size]
                        data = self._process_json_batch(batch_data, metadata)
                        batch_size_actual = len(batch_data)

                    try:
                        # Insert using MilvusClient
                        self.client.insert(
                            collection_name=final_collection_name, data=data
                        )
                        total_inserted += batch_size_actual
                    except Exception as e:
                        self.logger.error(
                            f"Failed to insert batch {i // batch_size}: {e}"
                        )
                        failed_batches.append(
                            {
                                "file": data_file.name,
                                "batch": i // batch_size,
                                "error": str(e),
                            }
                        )

        # Flush data
        self.client.flush(collection_name=final_collection_name)
        self.logger.info("Data flushed to disk")

        # Load collection (indexes are already created during collection creation)
        self.client.load_collection(collection_name=final_collection_name)
        self.logger.info(f"Collection '{final_collection_name}' loaded")

        # Get index info for return value
        index_info = self.schema_builder.get_index_info_from_metadata(
            final_collection_name, metadata, use_flat_index
        )

        return {
            "collection_name": final_collection_name,
            "total_inserted": total_inserted,
            "failed_batches": failed_batches,
            "indexes_created": index_info,
            "collection_loaded": True,
        }

    def _convert_dataframe_to_dict_list(
        self, df: pd.DataFrame, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert DataFrame to list of dictionaries with missing column handling."""
        import numpy as np

        # Convert DataFrame to list of dictionaries
        data_list = []

        for _, row in df.iterrows():
            record = {}
            for field_info in metadata["schema"]["fields"]:
                field_name = field_info["name"]
                field_type = field_info["type"]

                # Skip auto_id columns
                if self._is_auto_id_field(field_name, metadata):
                    continue

                # Skip BM25 function output fields - they are auto-generated by Milvus
                if self._is_bm25_output_field(field_name, metadata):
                    continue

                # Handle missing columns based on nullable/default_value properties
                if field_name not in df.columns:
                    # Column is completely missing from DataFrame - skip it, let Milvus handle
                    self.logger.debug(
                        f"Field '{field_name}' missing from DataFrame - Milvus will handle with null/default"
                    )
                    continue

                # Also handle cases where field exists in DataFrame but value is missing from this specific row
                if field_name not in row:
                    # Field is missing from this row - skip it, let Milvus handle
                    self.logger.debug(
                        f"Field '{field_name}' missing from row - Milvus will handle with null/default"
                    )
                    continue

                # Get the value
                value = row[field_name]

                # Convert vector data if needed
                if "Vector" in field_type:
                    # Convert single vector value
                    converted_value = self._convert_single_vector_data(
                        value, field_type
                    )
                    record[field_name] = converted_value
                elif field_type == "Array":
                    # Convert array field - ensure it's a list
                    # Check for numpy array first to avoid ambiguous truth value error
                    if isinstance(value, np.ndarray):
                        record[field_name] = value.tolist()
                    elif isinstance(value, list):
                        record[field_name] = value
                    elif pd.isna(value):
                        record[field_name] = []
                    else:
                        record[field_name] = [value] if value is not None else []
                elif field_type == "JSON":
                    # Convert JSON field - handle both dict and string formats
                    if pd.isna(value):
                        record[field_name] = None
                    elif isinstance(value, str):
                        # JSON stored as string in Parquet - convert back to dict for insert
                        try:
                            import json

                            record[field_name] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, treat as string
                            record[field_name] = value
                    elif isinstance(value, dict | list):
                        # Already a JSON object
                        record[field_name] = value
                    else:
                        record[field_name] = value
                else:
                    # Scalar fields - convert to native Python types with proper type conversion
                    # Handle pd.isna() carefully to avoid array ambiguity errors
                    try:
                        is_na = pd.isna(value)
                    except (ValueError, TypeError):
                        # Handle cases where pd.isna fails (like on arrays)
                        is_na = value is None

                    if is_na:
                        # Handle missing values based on field properties
                        if field_info.get("nullable", False):
                            record[field_name] = None
                        elif field_info.get("default_value") is not None:
                            record[field_name] = field_info["default_value"]
                        else:
                            record[field_name] = None
                    elif hasattr(value, "to_pydatetime"):
                        try:
                            record[field_name] = (
                                value.to_pydatetime() if not pd.isna(value) else None
                            )
                        except (ValueError, TypeError):
                            record[field_name] = value
                    else:
                        # Convert to correct data type based on field definition
                        converted_value = self._convert_scalar_value(
                            value, field_type, field_info
                        )
                        record[field_name] = converted_value

            # Handle $meta field - unpack dynamic fields
            if "$meta" in df.columns and "$meta" in row:
                meta_value = row["$meta"]
                if not pd.isna(meta_value) and meta_value is not None:
                    # Parse $meta field (may be stored as JSON string in Parquet)
                    if isinstance(meta_value, str):
                        try:
                            import json

                            meta_dict = json.loads(meta_value)
                        except (json.JSONDecodeError, TypeError):
                            meta_dict = {}
                    elif isinstance(meta_value, dict):
                        meta_dict = meta_value
                    else:
                        meta_dict = {}

                    # Add all dynamic fields from $meta to the record
                    for key, value in meta_dict.items():
                        if value is not None:  # Only add non-None dynamic fields
                            record[key] = value

            data_list.append(record)

        return data_list

    def _convert_scalar_value(
        self, value: Any, field_type: str, field_info: dict[str, Any]
    ) -> Any:
        """Convert scalar value to correct Python type for Milvus insertion."""
        import numpy as np

        # Handle NaN values with proper error handling for arrays
        try:
            is_na = pd.isna(value)
        except (ValueError, TypeError):
            # Handle cases where pd.isna fails (like on arrays)
            is_na = value is None

        if is_na:
            if field_info.get("nullable", False):
                return None
            elif field_info.get("default_value") is not None:
                return field_info["default_value"]
            else:
                return None

        # Type conversion mapping
        if field_type == "Int64":
            # Convert to Python int (not numpy int64)
            if isinstance(value, (int, np.integer)):
                return int(value)
            elif isinstance(value, (float, np.floating)):
                return int(value) if not np.isnan(value) else None
            elif isinstance(value, str):
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            else:
                return int(value) if value is not None else None

        elif field_type in ["Int32", "Int16", "Int8"]:
            # Convert to Python int
            if isinstance(value, (int, np.integer)):
                return int(value)
            elif isinstance(value, (float, np.floating)):
                return int(value) if not np.isnan(value) else None
            else:
                return int(value) if value is not None else None

        elif field_type in ["Float", "Double"]:
            # Convert to Python float
            if isinstance(value, (float, np.floating)) or isinstance(value, (int, np.integer)):
                return float(value)
            else:
                return float(value) if value is not None else None

        elif field_type == "Bool":
            # Convert to Python bool
            if isinstance(value, (bool, np.bool_)) or isinstance(value, (int, np.integer)):
                return bool(value)
            else:
                return bool(value) if value is not None else None

        elif field_type in ["String", "VarChar"]:
            # Convert to Python str
            if isinstance(value, str):
                return value
            else:
                return str(value) if value is not None else None

        else:
            # Return as-is for unknown types
            return value

    def _read_json_file(self, json_path: Path) -> pd.DataFrame:
        """Read JSON file and convert to DataFrame."""
        import json

        self.logger.info(f"Reading JSON file: {json_path}")

        data_list = []
        with open(json_path, encoding="utf-8") as f:
            # Handle both formats: single JSON array or line-delimited JSON
            content = f.read().strip()

            if content.startswith("["):
                # JSON array format (list of dict) - Milvus bulk import format
                data_list = json.loads(content)
            elif content.startswith("{"):
                # Check if it's legacy format with "rows" key or single object
                data = json.loads(content)
                if "rows" in data and isinstance(data["rows"], list):
                    # Legacy Milvus bulk import format: {"rows": [...]}
                    data_list = data["rows"]
                else:
                    # Single JSON object, treat as one record
                    data_list = [data]
            else:
                # Line-delimited JSON format (JSONL)
                f.seek(0)  # Reset file pointer
                for line in f:
                    line = line.strip()
                    if line:
                        data_list.append(json.loads(line))

        if not data_list:
            raise ValueError(f"No data found in JSON file: {json_path}")

        # Return the data_list directly for JSON files to preserve field omission
        self.logger.info(f"Loaded {len(data_list)} rows from JSON file")

        return data_list

    def _process_json_batch(
        self, batch_data: list[dict], metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process JSON batch data for Milvus insertion - mainly type conversion."""
        processed_data = []

        for record in batch_data:
            processed_record = {}

            # Process each field according to schema definition
            for field_info in metadata["schema"]["fields"]:
                field_name = field_info["name"]
                field_type = field_info["type"]

                # Skip auto_id fields
                if field_info.get("auto_id", False):
                    continue

                # Skip BM25 function output fields - they are auto-generated by Milvus
                if self._is_bm25_output_field(field_name, metadata):
                    continue

                # If field is present in the record, convert it
                if field_name in record:
                    value = record[field_name]

                    # Handle vector fields
                    if "Vector" in field_type:
                        processed_record[field_name] = self._convert_single_vector_data(
                            value, field_type
                        )
                    elif field_type == "Array":
                        # Convert array field
                        if isinstance(value, list):
                            processed_record[field_name] = value
                        else:
                            processed_record[field_name] = (
                                [value] if value is not None else []
                            )
                    elif field_type == "JSON":
                        # JSON field - keep as is
                        processed_record[field_name] = value
                    else:
                        # Scalar field - convert type
                        processed_record[field_name] = self._convert_scalar_value(
                            value, field_type, field_info
                        )

                # If field is missing, skip it - let Milvus handle with null/default
                # This is the key difference from DataFrame processing

            # Handle $meta field if present
            if "$meta" in record:
                meta_value = record["$meta"]
                if isinstance(meta_value, dict):
                    # Add dynamic fields from $meta
                    for key, value in meta_value.items():
                        if value is not None:
                            processed_record[key] = value

            processed_data.append(processed_record)

        return processed_data

    def _convert_single_vector_data(self, vector_data: Any, field_type: str) -> Any:
        """Convert single vector data to appropriate format for Milvus insert."""
        import numpy as np

        if field_type == "Float16Vector":
            # Convert data to float16 numpy array
            if isinstance(vector_data, list | np.ndarray):
                # First ensure it's uint8
                if (
                    isinstance(vector_data, np.ndarray)
                    and vector_data.dtype != np.uint8
                ):
                    uint8_array = vector_data.astype(np.uint8)
                else:
                    uint8_array = np.array(vector_data, dtype=np.uint8)
                # Then view as float16
                float16_array = uint8_array.view(np.float16)
                return np.ascontiguousarray(
                    float16_array
                )  # Return numpy array for Milvus
            return vector_data
        elif field_type == "BFloat16Vector":
            # Convert data to bfloat16 numpy array
            try:
                import ml_dtypes

                bfloat16 = ml_dtypes.bfloat16
            except ImportError:
                self.logger.error(
                    "ml_dtypes not available, cannot convert BFloat16Vector"
                )
                return vector_data

            if isinstance(vector_data, list | np.ndarray):
                # First ensure it's uint8
                if (
                    isinstance(vector_data, np.ndarray)
                    and vector_data.dtype != np.uint8
                ):
                    uint8_array = vector_data.astype(np.uint8)
                else:
                    uint8_array = np.array(vector_data, dtype=np.uint8)
                # Then view as bfloat16
                bfloat16_array = uint8_array.view(bfloat16)
                return np.ascontiguousarray(
                    bfloat16_array
                )  # Return numpy array for Milvus
            return vector_data
        elif field_type == "BinaryVector":
            # Convert data to bytes
            if isinstance(vector_data, list | np.ndarray):
                # First ensure it's uint8
                if (
                    isinstance(vector_data, np.ndarray)
                    and vector_data.dtype != np.uint8
                ):
                    uint8_array = vector_data.astype(np.uint8)
                else:
                    uint8_array = np.array(vector_data, dtype=np.uint8)
                return uint8_array.tobytes()
            return vector_data
        elif field_type == "SparseFloatVector":
            # Convert sparse vector from string-keyed dict to int-keyed dict with only non-null values
            if isinstance(vector_data, dict):
                # Filter out null values and convert keys to int
                sparse_vector = {
                    int(k): v for k, v in vector_data.items() if v is not None
                }
                return sparse_vector
            return vector_data
        else:
            # FloatVector - keep as is
            return vector_data

    def _get_field_type(self, column_name: str, metadata: dict[str, Any]) -> str:
        """Get the field type for a column from metadata."""
        for field_info in metadata["schema"]["fields"]:
            if field_info["name"] == column_name:
                return str(field_info["type"])
        return "unknown"

    def _is_auto_id_field(self, column_name: str, metadata: dict[str, Any]) -> bool:
        """Check if a column is an auto_id field that should be skipped during insert."""
        for field_info in metadata["schema"]["fields"]:
            if field_info["name"] == column_name and field_info.get("auto_id", False):
                return True
        return False

    def _is_bm25_output_field(self, field_name: str, metadata: dict[str, Any]) -> bool:
        """Check if a field is a BM25 function output field that should be skipped during insert."""
        functions = metadata.get("schema", {}).get("functions", [])
        for func in functions:
            if func.get("type") == "BM25" and field_name in func.get(
                "output_field_names", []
            ):
                return True
        return False

    def close(self) -> None:
        """Close Milvus connection."""
        try:
            self.client.close()
            self.logger.info("Disconnected from Milvus")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {e}")

    def test_connection(self) -> bool:
        """Test Milvus connection."""
        try:
            # Try to list collections
            collections = self.client.list_collections()
            self.logger.info(
                f"Successfully connected to Milvus. Found {len(collections)} collections."
            )
            return True
        except Exception as e:
            display_error(f"Failed to connect to Milvus: {e}")
            return False

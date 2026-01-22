"""Insert generated data directly to Milvus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from pymilvus import MilvusClient
from pymilvus.exceptions import ConnectError, MilvusException

from .json_utils import load_json_data
from .logging_config import get_logger
from .milvus_schema_builder import MilvusSchemaBuilder
from .rich_display import display_error
from .utils import is_bm25_output_field

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
        except (ConnectError, MilvusException) as e:
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
                final_collection_name,
                metadata,
                drop_if_exists=False,
                use_flat_index=use_flat_index,
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

            # Insert data in batches
            if show_progress:
                print(f"📥 Inserting {data_file.name} ({total_rows:,} rows)...")

            for i in range(0, total_rows, batch_size):
                batch_num = i // batch_size + 1

                # Show progress periodically
                if show_progress and batch_num % 10 == 0:
                    progress_pct = 100.0 * i / total_rows
                    print(
                        f"📊 Progress: {i:,}/{total_rows:,} rows ({progress_pct:.1f}%)"
                    )

                if file_format == "parquet":
                    batch_df = data_source.iloc[i : i + batch_size]
                    try:
                        # Convert DataFrame to list of dictionaries
                        data = self._convert_dataframe_to_dict_list(batch_df, metadata)
                        batch_size_actual = len(batch_df)
                    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                        self.logger.error(f"Failed to convert batch {batch_num}: {e}")
                        failed_batches.append(
                            {
                                "file": data_file.name,
                                "batch": batch_num,
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
                    self.client.insert(collection_name=final_collection_name, data=data)
                    total_inserted += batch_size_actual
                except MilvusException as e:
                    self.logger.error(f"Failed to insert batch {batch_num}: {e}")
                    failed_batches.append(
                        {
                            "file": data_file.name,
                            "batch": batch_num,
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
        """Convert DataFrame to list of dictionaries with missing column handling.

        Uses vectorized to_dict() for ~100x performance improvement over iterrows().
        """
        # Build field info map for fast lookup
        field_map = {
            f["name"]: f
            for f in metadata["schema"]["fields"]
            if not self._is_auto_id_field(f["name"], metadata)
            and not is_bm25_output_field(f["name"], metadata)
        }

        # Log missing columns once (not per-row)
        missing_cols = [f for f in field_map if f not in df.columns]
        if missing_cols:
            self.logger.debug(
                f"Fields missing from DataFrame (Milvus will handle): {missing_cols}"
            )

        # Convert DataFrame to list of dicts - MUCH faster than iterrows()
        raw_records = df.to_dict(orient="records")
        has_meta = "$meta" in df.columns

        data_list = []
        for raw_record in raw_records:
            record = self._convert_record(raw_record, field_map, has_meta)
            data_list.append(record)

        return data_list

    def _convert_record(
        self,
        raw_record: dict[str, Any],
        field_map: dict[str, dict[str, Any]],
        has_meta: bool,
    ) -> dict[str, Any]:
        """Convert a single record from raw dict format to Milvus-compatible format."""
        record = {}

        for field_name, field_info in field_map.items():
            if field_name not in raw_record:
                continue

            value = raw_record[field_name]
            field_type = field_info["type"]

            # Convert based on field type
            if "Vector" in field_type:
                record[field_name] = self._convert_single_vector_data(value, field_type)
            elif field_type == "Array":
                record[field_name] = self._convert_array_value(value)
            elif field_type == "JSON":
                record[field_name] = self._convert_json_value(value)
            else:
                record[field_name] = self._convert_scalar_field(value, field_type, field_info)

        # Handle $meta field - unpack dynamic fields
        if has_meta and "$meta" in raw_record:
            meta_value = raw_record["$meta"]
            meta_dict = self._parse_meta_value(meta_value)
            for key, val in meta_dict.items():
                if val is not None:
                    record[key] = val

        return record

    def _convert_array_value(self, value: Any) -> list:
        """Convert value to array format for Milvus."""
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return value
        elif self._is_na(value):
            return []
        else:
            return [value] if value is not None else []

    def _convert_json_value(self, value: Any) -> Any:
        """Convert value to JSON format for Milvus."""
        if self._is_na(value):
            return None
        elif isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        elif isinstance(value, dict | list):
            return value
        else:
            return value

    def _convert_scalar_field(
        self, value: Any, field_type: str, field_info: dict[str, Any]
    ) -> Any:
        """Convert scalar field value for Milvus insertion."""
        if self._is_na(value):
            if field_info.get("nullable", False):
                return None
            elif field_info.get("default_value") is not None:
                return field_info["default_value"]
            else:
                return None
        elif hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime() if not self._is_na(value) else None
            except (ValueError, TypeError):
                return value
        else:
            return self._convert_scalar_value(value, field_type, field_info)

    def _parse_meta_value(self, meta_value: Any) -> dict[str, Any]:
        """Parse $meta field value to dictionary."""
        if self._is_na(meta_value) or meta_value is None:
            return {}
        elif isinstance(meta_value, str):
            try:
                return json.loads(meta_value)
            except (json.JSONDecodeError, TypeError):
                return {}
        elif isinstance(meta_value, dict):
            return meta_value
        else:
            return {}

    def _is_na(self, value: Any) -> bool:
        """Check if value is NA/NaN safely."""
        try:
            return pd.isna(value)
        except (ValueError, TypeError):
            return value is None

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
            if isinstance(value, (float, np.floating, int, np.integer)):
                return float(value)
            else:
                return float(value) if value is not None else None

        elif field_type == "Bool":
            # Convert to Python bool
            if isinstance(value, (bool, np.bool_, int, np.integer)):
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

    def _read_json_file(self, json_path: Path) -> list[dict[str, Any]]:
        """Read JSON file and return as list of dictionaries.

        Uses shared JSON loading utility that handles multiple formats:
        - JSON array: [{"field": "value"}, ...]
        - Legacy Milvus: {"rows": [...]}
        - Single object: {"field": "value"}
        - JSONL: one JSON object per line
        """
        self.logger.info(f"Reading JSON file: {json_path}")

        data_list = load_json_data(json_path)

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
                if is_bm25_output_field(field_name, metadata):
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

    def close(self) -> None:
        """Close Milvus connection."""
        try:
            self.client.close()
            self.logger.info("Disconnected from Milvus")
        except MilvusException as e:
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
        except (ConnectError, MilvusException) as e:
            display_error(f"Failed to connect to Milvus: {e}")
            return False

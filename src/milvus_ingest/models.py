"""Pydantic models for schema validation with helpful error messages."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from pydantic import ValidationInfo


class FieldType(str, Enum):
    """Supported Milvus field types."""

    BOOL = "Bool"
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    FLOAT = "Float"
    DOUBLE = "Double"
    VARCHAR = "VarChar"
    STRING = "String"
    JSON = "JSON"
    ARRAY = "Array"
    FLOAT_VECTOR = "FloatVector"
    BINARY_VECTOR = "BinaryVector"
    FLOAT16_VECTOR = "Float16Vector"
    BFLOAT16_VECTOR = "BFloat16Vector"
    SPARSE_FLOAT_VECTOR = "SparseFloatVector"


class DynamicFieldType(str, Enum):
    """Supported dynamic field types for data generation."""

    BOOL = "Bool"
    INT = "Int"
    FLOAT = "Float"
    STRING = "String"
    ARRAY = "Array"
    JSON = "JSON"


class ArrayElementType(str, Enum):
    """Supported array element types."""

    BOOL = "Bool"
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    FLOAT = "Float"
    DOUBLE = "Double"
    VARCHAR = "VarChar"
    STRING = "String"


class FunctionType(str, Enum):
    """Supported Milvus function types."""

    BM25 = "BM25"


class DynamicFieldSchema(BaseModel):
    """Schema definition for dynamic fields that should be generated."""

    name: str = Field(
        ..., description="Dynamic field name", min_length=1, max_length=255
    )
    type: DynamicFieldType = Field(..., description="Dynamic field data type")
    probability: float = Field(
        default=1.0,
        description="Probability that this field appears in a row (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    # Field type-specific constraints
    min_value: int | float | None = Field(
        default=None, description="Minimum value for numeric fields"
    )
    max_value: int | float | None = Field(
        default=None, description="Maximum value for numeric fields"
    )
    min_length: int | None = Field(
        default=None, description="Minimum length for string fields", ge=1
    )
    max_length: int | None = Field(
        default=None, description="Maximum length for string fields", ge=1, le=1000
    )
    values: list[Any] | None = Field(
        default=None, description="List of possible values to choose from"
    )
    array_min_length: int | None = Field(
        default=None, description="Minimum array length", ge=0, le=100
    )
    array_max_length: int | None = Field(
        default=None, description="Maximum array length", ge=1, le=100
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate field name."""
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"Dynamic field name '{v}' is invalid. Field names must contain only letters, numbers, and underscores."
            )
        return v

    @model_validator(mode="after")
    def validate_constraints(self) -> DynamicFieldSchema:
        """Validate field-specific constraints."""
        field_type = self.type.value.upper()

        # Validate min/max for numeric types
        if field_type in {"INT", "FLOAT"}:
            if (
                self.min_value is not None
                and self.max_value is not None
                and self.min_value >= self.max_value
            ):
                raise ValueError(
                    f"Dynamic field '{self.name}': min_value ({self.min_value}) must be less than max_value ({self.max_value})"
                )
        else:
            if self.min_value is not None or self.max_value is not None:
                raise ValueError(
                    f"Dynamic field '{self.name}' with type '{self.type}' cannot have min_value/max_value constraints"
                )

        # Validate length constraints for strings
        if field_type == "STRING":
            if (
                self.min_length is not None
                and self.max_length is not None
                and self.min_length > self.max_length
            ):
                raise ValueError(
                    f"Dynamic field '{self.name}': min_length ({self.min_length}) must be <= max_length ({self.max_length})"
                )
        else:
            if self.min_length is not None or self.max_length is not None:
                raise ValueError(
                    f"Dynamic field '{self.name}' with type '{self.type}' cannot have length constraints"
                )

        # Validate array constraints
        if field_type == "ARRAY":
            if (
                self.array_min_length is not None
                and self.array_max_length is not None
                and self.array_min_length > self.array_max_length
            ):
                raise ValueError(
                    f"Dynamic field '{self.name}': array_min_length ({self.array_min_length}) must be <= array_max_length ({self.array_max_length})"
                )
        else:
            if self.array_min_length is not None or self.array_max_length is not None:
                raise ValueError(
                    f"Dynamic field '{self.name}' with type '{self.type}' cannot have array length constraints"
                )

        return self


class FieldSchema(BaseModel):
    """Pydantic model for field schema validation."""

    name: str = Field(..., description="Field name", min_length=1, max_length=255)
    type: FieldType = Field(..., description="Field data type")
    is_primary: bool = Field(
        default=False, description="Whether this field is the primary key"
    )
    auto_id: bool = Field(
        default=False, description="Whether to auto-generate ID values"
    )
    is_partition_key: bool = Field(
        default=False, description="Whether this field is a partition key"
    )
    nullable: bool = Field(default=False, description="Whether this field can be null")
    default_value: Any | None = Field(
        default=None,
        description="Default value for scalar fields when no value is provided during insert",
    )

    # Numeric field constraints
    min: int | float | None = Field(
        default=None, description="Minimum value for numeric fields"
    )
    max: int | float | None = Field(
        default=None, description="Maximum value for numeric fields"
    )

    # String field constraints
    max_length: int | None = Field(
        default=None, description="Maximum length for string fields", ge=1, le=65535
    )

    # Vector field constraints
    dim: int | None = Field(
        default=None, description="Dimension for vector fields", ge=1, le=32768
    )

    # Array field constraints
    element_type: ArrayElementType | None = Field(
        default=None, description="Element type for array fields"
    )
    max_capacity: int | None = Field(
        default=None, description="Maximum capacity for array fields", ge=1, le=4096
    )

    # Data distribution constraints (for scalar fields)
    cardinality_ratio: float | None = Field(
        default=None,
        description="Ratio of unique values to total rows (0.0-1.0). Lower values create more duplicates.",
        ge=0.0,
        le=1.0,
    )
    enum_values: list[str | int | float] | None = Field(
        default=None,
        description="Fixed set of values to randomly select from (overrides cardinality_ratio)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate field name."""
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"Field name '{v}' is invalid. Field names must contain only letters, numbers, and underscores."
            )
        return v

    @field_validator("auto_id")
    @classmethod
    def validate_auto_id(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate auto_id can only be True for primary key fields."""
        if v and not info.data.get("is_primary", False):
            raise ValueError(
                "auto_id can only be True for primary key fields. Set is_primary=true first."
            )
        return v

    @model_validator(mode="after")
    def validate_field_constraints(self) -> FieldSchema:
        """Validate field-specific constraints."""
        field_type = self.type.value.upper()

        # String fields must have max_length
        if field_type in {"VARCHAR", "STRING"} and self.max_length is None:
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' must specify max_length.\n"
                f'Example: {{"name": "{self.name}", "type": "{self.type}", "max_length": 128}}'
            )

        # Vector fields must have dim (except SparseFloatVector)
        if (
            "VECTOR" in field_type
            and field_type != "SPARSEFLOATVECTOR"
            and self.dim is None
        ):
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' must specify dim (dimension).\n"
                f'Example: {{"name": "{self.name}", "type": "{self.type}", "dim": 128}}'
            )

        # SparseFloatVector should not have dim
        if field_type == "SPARSEFLOATVECTOR" and self.dim is not None:
            raise ValueError(
                f"Field '{self.name}' with type 'SparseFloatVector' should not specify dim.\n"
                f'Example: {{"name": "{self.name}", "type": "SparseFloatVector"}}'
            )

        # Array fields must have element_type and max_capacity
        if field_type == "ARRAY":
            if self.element_type is None:
                raise ValueError(
                    f"Field '{self.name}' with type 'Array' must specify element_type.\n"
                    f'Example: {{"name": "{self.name}", "type": "Array", "element_type": "VarChar", "max_capacity": 10}}'
                )
            if self.max_capacity is None:
                raise ValueError(
                    f"Field '{self.name}' with type 'Array' must specify max_capacity.\n"
                    f'Example: {{"name": "{self.name}", "type": "Array", "element_type": "VarChar", "max_capacity": 10}}'
                )
            # Array of VarChar needs max_length
            if (
                self.element_type in {ArrayElementType.VARCHAR, ArrayElementType.STRING}
                and self.max_length is None
            ):
                raise ValueError(
                    f"Field '{self.name}' with Array of VarChar/String must specify max_length for elements.\n"
                    f'Example: {{"name": "{self.name}", "type": "Array", "element_type": "VarChar", "max_capacity": 10, "max_length": 50}}'
                )

        # Validate min/max constraints
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(
                f"Field '{self.name}': min value ({self.min}) must be less than max value ({self.max})"
            )

        # Check incompatible field attributes
        if not field_type.startswith(("INT", "FLOAT", "DOUBLE")) and (
            self.min is not None or self.max is not None
        ):
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' cannot have min/max constraints. "
                f"These are only valid for numeric types (Int8, Int16, Int32, Int64, Float, Double)."
            )

        # Validate cardinality constraints
        if self.cardinality_ratio is not None or self.enum_values is not None:
            # Cardinality constraints only apply to scalar types
            scalar_types = {
                FieldType.INT8,
                FieldType.INT16,
                FieldType.INT32,
                FieldType.INT64,
                FieldType.FLOAT,
                FieldType.DOUBLE,
                FieldType.VARCHAR,
                FieldType.STRING,
            }
            if self.type not in scalar_types:
                raise ValueError(
                    f"Field '{self.name}' with type '{self.type}' cannot have cardinality constraints. "
                    f"cardinality_ratio and enum_values are only valid for scalar types."
                )

        if (
            field_type not in {"VARCHAR", "STRING"}
            and field_type != "ARRAY"
            and self.max_length is not None
        ):
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' cannot have max_length. "
                f"This is only valid for VarChar, String, and Array fields."
            )

        if "VECTOR" not in field_type and self.dim is not None:
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' cannot have dim. "
                f"This is only valid for vector fields (FloatVector, BinaryVector, etc.)."
            )

        # Validate default_value constraints
        if self.default_value is not None:
            # Primary key fields cannot have default values
            if self.is_primary:
                raise ValueError(
                    f"Primary key field '{self.name}' cannot have a default_value. "
                    f"Primary keys must be explicitly provided or use auto_id=true."
                )

            # Only scalar fields (non-array, non-JSON, non-vector) support default values
            scalar_types = {
                FieldType.BOOL,
                FieldType.INT8,
                FieldType.INT16,
                FieldType.INT32,
                FieldType.INT64,
                FieldType.FLOAT,
                FieldType.DOUBLE,
                FieldType.VARCHAR,
                FieldType.STRING,
            }
            if self.type not in scalar_types:
                raise ValueError(
                    f"Field '{self.name}' with type '{self.type}' cannot have a default_value. "
                    f"Only scalar fields (Bool, Int8-Int64, Float, Double, VarChar, String) support default values."
                )

            # Validate default value type matches field type
            self._validate_default_value_type()

        return self

    def _validate_default_value_type(self) -> None:
        """Validate that default_value type matches field type."""
        if self.default_value is None:
            return

        field_type = self.type.value.upper()

        # Boolean fields
        if field_type == "BOOL":
            if not isinstance(self.default_value, bool):
                raise ValueError(
                    f"Field '{self.name}' with type Bool must have a boolean default_value, "
                    f"got {type(self.default_value).__name__}: {self.default_value}"
                )

        # Integer fields
        elif field_type in {"INT8", "INT16", "INT32", "INT64"}:
            if not isinstance(self.default_value, int):
                raise ValueError(
                    f"Field '{self.name}' with type {self.type} must have an integer default_value, "
                    f"got {type(self.default_value).__name__}: {self.default_value}"
                )
            # Validate range constraints if specified
            if self.min is not None and self.default_value < self.min:
                raise ValueError(
                    f"Field '{self.name}' default_value ({self.default_value}) is less than min ({self.min})"
                )
            if self.max is not None and self.default_value > self.max:
                raise ValueError(
                    f"Field '{self.name}' default_value ({self.default_value}) is greater than max ({self.max})"
                )

        # Float/Double fields
        elif field_type in {"FLOAT", "DOUBLE"}:
            if not isinstance(self.default_value, int | float):
                raise ValueError(
                    f"Field '{self.name}' with type {self.type} must have a numeric default_value, "
                    f"got {type(self.default_value).__name__}: {self.default_value}"
                )
            # Convert int to float for consistency
            if isinstance(self.default_value, int):
                self.default_value = float(self.default_value)
            # Validate range constraints if specified
            if self.min is not None and self.default_value < self.min:
                raise ValueError(
                    f"Field '{self.name}' default_value ({self.default_value}) is less than min ({self.min})"
                )
            if self.max is not None and self.default_value > self.max:
                raise ValueError(
                    f"Field '{self.name}' default_value ({self.default_value}) is greater than max ({self.max})"
                )

        # String fields
        elif field_type in {"VARCHAR", "STRING"}:
            if not isinstance(self.default_value, str):
                raise ValueError(
                    f"Field '{self.name}' with type {self.type} must have a string default_value, "
                    f"got {type(self.default_value).__name__}: {self.default_value}"
                )
            # Validate max_length constraint
            if (
                self.max_length is not None
                and len(self.default_value) > self.max_length
            ):
                raise ValueError(
                    f"Field '{self.name}' default_value length ({len(self.default_value)}) exceeds max_length ({self.max_length})"
                )


class FunctionSchema(BaseModel):
    """Schema definition for Milvus functions (e.g., BM25 for full-text search)."""

    name: str = Field(..., description="Function name", min_length=1, max_length=255)
    type: FunctionType = Field(..., description="Function type (e.g., BM25)")
    input_field_names: list[str] = Field(
        ...,
        description="List of input field names that this function processes",
        min_length=1,
    )
    output_field_names: list[str] = Field(
        ...,
        description="List of output field names that this function generates",
        min_length=1,
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Additional function parameters"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate function name."""
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"Function name '{v}' is invalid. Function names must contain only letters, numbers, and underscores."
            )
        return v

    @model_validator(mode="after")
    def validate_function(self) -> FunctionSchema:
        """Validate function-specific constraints."""
        # BM25 specific validations
        if self.type == FunctionType.BM25:
            # BM25 accepts only one input field
            if len(self.input_field_names) != 1:
                raise ValueError(
                    f"BM25 function '{self.name}' must have exactly one input field, "
                    f"got {len(self.input_field_names)}: {', '.join(self.input_field_names)}"
                )
            # BM25 produces only one output field
            if len(self.output_field_names) != 1:
                raise ValueError(
                    f"BM25 function '{self.name}' must have exactly one output field, "
                    f"got {len(self.output_field_names)}: {', '.join(self.output_field_names)}"
                )

        return self


class CollectionSchema(BaseModel):
    """Pydantic model for collection schema validation."""

    collection_name: str | None = Field(
        default=None, description="Collection name", min_length=1, max_length=255
    )
    fields: list[FieldSchema] = Field(
        ..., description="List of field schemas", min_length=1
    )
    num_partitions: int | None = Field(
        default=None,
        description="Number of partitions for the collection",
        ge=1,
        le=4096,
    )
    enable_dynamic_field: bool = Field(
        default=False,
        description="Enable dynamic fields to store undefined fields in Milvus",
    )
    dynamic_fields: list[DynamicFieldSchema] | None = Field(
        default=None,
        description="Dynamic fields to generate when enable_dynamic_field is True",
    )
    functions: list[FunctionSchema] | None = Field(
        default=None,
        description="Functions to apply to fields (e.g., BM25 for full-text search)",
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str | None) -> str | None:
        """Validate collection name."""
        if v is not None and not v.replace("_", "").isalnum():
            raise ValueError(
                f"Collection name '{v}' is invalid. Collection names must contain only letters, numbers, and underscores."
            )
        return v

    @model_validator(mode="after")
    def validate_schema(self) -> CollectionSchema:
        """Validate the overall schema."""
        # Check for at least one primary key
        primary_fields = [f for f in self.fields if f.is_primary]
        if not primary_fields:
            raise ValueError(
                "Schema must have exactly one primary key field.\n"
                "Add is_primary=true to one of your fields.\n"
                'Example: {"name": "id", "type": "Int64", "is_primary": true}'
            )

        if len(primary_fields) > 1:
            primary_names = [f.name for f in primary_fields]
            raise ValueError(
                f"Schema can have only one primary key field, but found {len(primary_fields)}: {', '.join(primary_names)}\n"
                "Set is_primary=true for only one field."
            )

        # Check for duplicate field names
        field_names = [f.name for f in self.fields]
        duplicates = [name for name in set(field_names) if field_names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate field names found: {', '.join(duplicates)}\n"
                "Each field must have a unique name."
            )

        # Validate auto_id constraint
        primary_field = primary_fields[0]
        if primary_field.auto_id and primary_field.type not in {
            FieldType.INT64,
            FieldType.VARCHAR,
        }:
            raise ValueError(
                f"Primary key field '{primary_field.name}' with auto_id=true must be of type Int64 or VarChar, not {primary_field.type}.\n"
                "Change type to Int64/VarChar or set auto_id=false."
            )

        # Validate partition key constraints
        partition_key_fields = [f for f in self.fields if f.is_partition_key]
        if len(partition_key_fields) > 1:
            partition_names = [f.name for f in partition_key_fields]
            raise ValueError(
                f"Schema can have only one partition key field, but found {len(partition_key_fields)}: {', '.join(partition_names)}\n"
                "Set is_partition_key=true for only one field."
            )

        if partition_key_fields:
            partition_field = partition_key_fields[0]
            # Partition key cannot be nullable
            if partition_field.nullable:
                raise ValueError(
                    f"Partition key field '{partition_field.name}' cannot be nullable.\n"
                    "Set nullable=false for partition key fields."
                )
            # Partition key must be scalar type (VARCHAR or INT64 are most common)
            if partition_field.type not in {
                FieldType.VARCHAR,
                FieldType.INT8,
                FieldType.INT16,
                FieldType.INT32,
                FieldType.INT64,
            }:
                raise ValueError(
                    f"Partition key field '{partition_field.name}' must be a scalar type (VARCHAR or integer), not {partition_field.type}.\n"
                    "Use VARCHAR or INT64 for partition keys."
                )

        # Validate dynamic fields constraints
        if self.dynamic_fields is not None:
            if not self.enable_dynamic_field:
                raise ValueError(
                    "dynamic_fields can only be specified when enable_dynamic_field is True.\n"
                    "Set enable_dynamic_field=true to use dynamic fields."
                )

            # Check for duplicate dynamic field names
            dynamic_field_names = [f.name for f in self.dynamic_fields]
            duplicates = [
                name
                for name in set(dynamic_field_names)
                if dynamic_field_names.count(name) > 1
            ]
            if duplicates:
                raise ValueError(
                    f"Duplicate dynamic field names found: {', '.join(duplicates)}\n"
                    "Each dynamic field must have a unique name."
                )

            # Check for conflicts with regular field names
            regular_field_names = [f.name for f in self.fields]
            conflicts = [
                name for name in dynamic_field_names if name in regular_field_names
            ]
            if conflicts:
                raise ValueError(
                    f"Dynamic field names conflict with regular field names: {', '.join(conflicts)}\n"
                    "Dynamic field names must be different from regular field names."
                )

        # Validate functions
        if self.functions is not None:
            # Get all field names for validation
            all_field_names = {f.name for f in self.fields}
            function_output_fields: set[str] = set()

            for func in self.functions:
                # Validate input fields exist
                for input_field in func.input_field_names:
                    if input_field not in all_field_names:
                        raise ValueError(
                            f"Function '{func.name}' references non-existent input field '{input_field}'. "
                            f"Available fields: {', '.join(sorted(all_field_names))}"
                        )

                # For BM25, output field must be defined in schema
                # For other functions, output fields should not conflict
                if func.type == FunctionType.BM25:
                    # BM25 output field must already exist in schema
                    for output_field in func.output_field_names:
                        if output_field not in all_field_names:
                            raise ValueError(
                                f"BM25 function '{func.name}' output field '{output_field}' must be defined in fields. "
                                f"Add a SparseFloatVector field named '{output_field}' to the schema."
                            )
                else:
                    # Other function types create new fields, so check for conflicts
                    for output_field in func.output_field_names:  # type: ignore[unreachable]
                        if output_field in all_field_names:
                            raise ValueError(
                                f"Function '{func.name}' output field '{output_field}' conflicts with existing field. "
                                f"Function output fields must have unique names."
                            )
                        if output_field in function_output_fields:
                            raise ValueError(
                                f"Multiple functions produce the same output field '{output_field}'. "
                                f"Each function must produce uniquely named output fields."
                            )
                        function_output_fields.add(output_field)

                # Additional BM25-specific validations
                if func.type == FunctionType.BM25:
                    # Validate input field is VARCHAR
                    input_field_name = func.input_field_names[0]
                    field_obj = next(
                        f for f in self.fields if f.name == input_field_name
                    )
                    if field_obj.type not in {FieldType.VARCHAR, FieldType.STRING}:
                        raise ValueError(
                            f"BM25 function '{func.name}' input field '{input_field_name}' must be VarChar or String type, "
                            f"got {field_obj.type}"
                        )

                    # Validate that a corresponding SparseFloatVector field exists
                    output_field_name = func.output_field_names[0]
                    # Find if there's a SparseFloatVector field for the output
                    has_sparse_field = any(
                        f.name == output_field_name
                        and f.type == FieldType.SPARSE_FLOAT_VECTOR
                        for f in self.fields
                    )
                    if not has_sparse_field:
                        # Add a note that the output field should be defined as SparseFloatVector
                        raise ValueError(
                            f"BM25 function '{func.name}' requires output field '{output_field_name}' "
                            f"to be defined as SparseFloatVector in the fields list.\n"
                            f'Example: {{"name": "{output_field_name}", "type": "SparseFloatVector"}}'
                        )

        return self


# Support for list-only schemas (backward compatibility)
SchemaModel = CollectionSchema | list[FieldSchema]


def validate_schema_data(data: dict[str, Any] | list[dict[str, Any]]) -> SchemaModel:
    """Validate schema data and return appropriate model."""
    if isinstance(data, list):
        # List of fields only
        return [FieldSchema.model_validate(field) for field in data]
    else:
        # Full collection schema
        return CollectionSchema.model_validate(data)


def get_schema_help() -> str:
    """Get comprehensive help text for writing schema files."""
    return """
# Milvus Schema File Format

## Basic Structure

### Full Schema Format:
```json
{
  "collection_name": "my_collection",
  "enable_dynamic_field": true,
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true},
    {"name": "title", "type": "VarChar", "max_length": 128},
    {"name": "embedding", "type": "FloatVector", "dim": 768}
  ]
}
```

### Simplified Format (fields only):
```json
[
  {"name": "id", "type": "Int64", "is_primary": true},
  {"name": "title", "type": "VarChar", "max_length": 128}
]
```

## Field Types and Requirements

### Numeric Types
- **Int8, Int16, Int32, Int64**: Integer types
- **Float, Double**: Floating-point types
- Optional: `min`, `max` for value ranges

### String Types
- **VarChar, String**: Variable-length strings
- Required: `max_length` (1-65535)

### Other Types
- **Bool**: Boolean values
- **JSON**: JSON objects

### Vector Types
- **FloatVector**: 32-bit float vectors
- **BinaryVector**: Binary vectors
- **Float16Vector**: 16-bit float vectors
- **BFloat16Vector**: Brain float vectors
- **SparseFloatVector**: Sparse float vectors
- Required: `dim` (dimension, 1-32768)

### Array Type
- **Array**: Array of elements
- Required: `element_type`, `max_capacity`
- If element_type is VarChar/String: also need `max_length`

## Collection Properties

### Optional Collection Properties
- `collection_name`: Name of the collection
- `enable_dynamic_field`: Enable dynamic fields (default: false)
  - When true, allows inserting fields not defined in the schema
  - Dynamic fields are stored in the special `$meta` field
  - Can be queried using field names directly in filter expressions
- `num_partitions`: Number of partitions for the collection (1-4096)

## Field Properties

### Required Properties
- `name`: Field name (letters, numbers, underscores only)
- `type`: Field type (see above)

### Optional Properties
- `is_primary`: Primary key flag (exactly one field must be true)
- `auto_id`: Auto-generate ID (only for Int64 primary keys)
- `nullable`: Allow null values
- `default_value`: Default value for scalar fields when no value provided during insert
- `min`, `max`: Value constraints (numeric types only)
- `max_length`: String length limit (string/array types)
- `dim`: Vector dimension (vector types only)
- `element_type`: Array element type (array only)
- `max_capacity`: Array capacity (array only)

## Functions

Milvus supports functions that process fields to generate derived data automatically.

### BM25 Function (Full-Text Search)
The BM25 function enables full-text search by converting text fields into sparse vectors.

```json
{
  "name": "text_bm25",
  "type": "BM25",
  "input_field_names": ["content"],
  "output_field_names": ["content_sparse"],
  "params": {}
}
```

**Important Notes for BM25:**
- Input field must be VarChar or String type
- Output field must be defined as SparseFloatVector in the fields list
- BM25 automatically generates sparse vectors during insert/import (no data needed for output field)
- Only one input and one output field allowed per BM25 function

## Examples

### E-commerce Collection:
```json
{
  "collection_name": "products",
  "enable_dynamic_field": false,
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true, "auto_id": true},
    {"name": "title", "type": "VarChar", "max_length": 200},
    {"name": "price", "type": "Float", "min": 0.0, "max": 10000.0},
    {"name": "tags", "type": "Array", "element_type": "VarChar", "max_capacity": 10, "max_length": 50},
    {"name": "in_stock", "type": "Bool", "nullable": true},
    {"name": "metadata", "type": "JSON"},
    {"name": "embedding", "type": "FloatVector", "dim": 768}
  ]
}
```

### Collection with Default Values:
```json
{
  "collection_name": "user_profiles",
  "enable_dynamic_field": false,
  "fields": [
    {"name": "user_id", "type": "Int64", "is_primary": true, "auto_id": true},
    {"name": "name", "type": "VarChar", "max_length": 100},
    {"name": "age", "type": "Int64", "default_value": 18, "min": 0, "max": 120},
    {"name": "status", "type": "VarChar", "max_length": 20, "default_value": "active"},
    {"name": "score", "type": "Float", "default_value": 0.0, "min": 0.0, "max": 100.0},
    {"name": "is_verified", "type": "Bool", "default_value": false},
    {"name": "embedding", "type": "FloatVector", "dim": 512}
  ]
}
```


### Dynamic Fields Collection:
```json
{
  "collection_name": "documents",
  "enable_dynamic_field": true,
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true, "auto_id": true},
    {"name": "content", "type": "VarChar", "max_length": 5000},
    {"name": "embedding", "type": "FloatVector", "dim": 384}
  ]
}
```

When `enable_dynamic_field` is true, you can insert additional fields:
```json
{
  "id": 1,
  "content": "Document content",
  "embedding": [0.1, 0.2, ...],
  "author": "John Doe",
  "tags": ["AI", "ML"],
  "publish_date": "2024-01-15",
  "metadata": {"source": "blog", "views": 1500}
}
```

### Simple Document Collection:
```json
[
  {"name": "doc_id", "type": "VarChar", "max_length": 100, "is_primary": true},
  {"name": "content", "type": "VarChar", "max_length": 5000},
  {"name": "vector", "type": "FloatVector", "dim": 384}
]
```

### Full-Text Search Collection with BM25:
```json
{
  "collection_name": "full_text_search",
  "enable_dynamic_field": false,
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true, "auto_id": true},
    {"name": "title", "type": "VarChar", "max_length": 500},
    {"name": "content", "type": "VarChar", "max_length": 10000},
    {"name": "content_sparse", "type": "SparseFloatVector"},
    {"name": "embedding", "type": "FloatVector", "dim": 768}
  ],
  "functions": [
    {
      "name": "content_bm25",
      "type": "BM25",
      "input_field_names": ["content"],
      "output_field_names": ["content_sparse"],
      "params": {}
    }
  ]
}
```

**Note:** When inserting data, you only need to provide values for `title`, `content`, and `embedding`.
The `content_sparse` field will be automatically populated by the BM25 function.
""".strip()

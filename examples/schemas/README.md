# Example Schemas

This directory contains example schema files for testing and documentation purposes.

## Available Schemas

### performance_test.json
A comprehensive schema designed for performance testing that includes:
- Primary key field (Int64)
- Dense vector field (FloatVector, 128-dim)
- Sparse vector field (SparseFloatVector)
- Text field (VarChar)
- Numeric field (Float)
- JSON metadata field
- Array field (VarChar array)

**Usage:**
```bash
# Test performance with this schema
milvus-ingest generate --schema examples/schemas/performance_test.json --total-rows 100000

# Validate the schema
milvus-ingest generate --schema examples/schemas/performance_test.json --validate-only
```

## Creating Custom Schemas

For schema format documentation, see:
- [Schema Format Guide](../docs/tutorials/json-format-guide.md)
- [Schema Command Reference](../docs/commands/schema.md)

You can also use the built-in schemas:
```bash
# List all built-in schemas
milvus-ingest schema list

# Show details of a built-in schema
milvus-ingest schema show simple
```

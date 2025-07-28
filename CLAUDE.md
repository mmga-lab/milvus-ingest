# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
pdm install        # Install with development dependencies (default behavior)
pdm install --prod # Install production dependencies only
```

### Code Quality & Testing
```bash
# Formatting
pdm run ruff format src tests

# Linting & Type Checking
pdm run ruff check src tests
pdm run mypy src

# Testing
pdm run pytest                                          # Run all tests
pdm run pytest tests/test_generator.py                  # Run specific test file
pdm run pytest tests/test_generator.py::TestClass      # Run specific test class
pdm run pytest tests/test_generator.py::test_function  # Run specific test function
pdm run pytest --cov=src --cov-report=html             # Run tests with coverage
pdm run pytest --cov=src --cov-report=term-missing     # Show missing lines in terminal
pdm run pytest -v -s                                    # Verbose output with print statements
pdm run pytest -m "not slow"                            # Skip slow tests
pdm run pytest -m integration                           # Run only integration tests

# Combined quality checks (via Makefile)
make lint          # Run ruff format + check + mypy
make test          # Run pytest
make test-cov      # Run tests with coverage report
make check         # Run lint + test together
make clean         # Clean build artifacts and caches
make build         # Build the package
make publish       # Publish to PyPI
make security      # Run security checks (safety, pip-audit, bandit)
```

### Test Environment Configuration

**Local Test Environment (Recommended):**
Use the Docker Compose setup in `deploy/docker-compose.yml` to start local test services:

```bash
# Start local test environment
cd deploy/
docker-compose up -d

# Environment variables for local testing
export MILVUS_URI=http://127.0.0.1:19530
export MINIO_HOST=127.0.0.1
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=a-bucket

# Or use the .env.example file in project root
cp .env.example .env  # Create and edit with your values
```

**Note:** The `deploy/docker-compose.yml` provides a complete local testing stack including Milvus and MinIO.

### Building & Publishing
```bash
pdm build         # Build the package
pdm publish       # Publish to PyPI (requires PDM_PUBLISH_TOKEN)
```

### CLI Usage for Testing
```bash
# Install in development mode first
pdm install

# High-performance data generation commands (optimized for large-scale datasets)
milvus-ingest generate --builtin simple --total-rows 100000 --preview        # Generate 100K rows
milvus-ingest generate --schema example_schema.json --total-rows 1000000     # Generate 1M rows
milvus-ingest generate --schema schema.json --total-rows 5000000 --batch-size 100000 # Use large batch size

# Advanced file control options (for pipeline testing scenarios)
milvus-ingest generate --schema schema.json --total-rows 1000000 --file-size 256MB   # Set specific file size
milvus-ingest generate --schema schema.json --file-count 10 --file-size 10GB         # Generate 10×10GB files
milvus-ingest generate --schema schema.json --file-count 500 --file-size 200MB       # Generate 500×200MB files
milvus-ingest generate --schema schema.json --total-rows 1000000 --rows-per-file 500000 # Set max rows per file

# Partition and shard configuration (for distributed testing)
milvus-ingest generate --schema schema.json --total-rows 1000000 --partitions 8 --shards 4  # Multi-partition setup
milvus-ingest generate --schema schema.json --total-rows 5000000 --partitions 1024 --shards 16  # Max partitions/shards

# Additional generation options
milvus-ingest generate --schema schema.json --validate-only            # Validate schema without generating
milvus-ingest generate --schema schema.json --total-rows 1000000 --no-progress # Disable progress bar
milvus-ingest generate --schema schema.json --out mydata --force   # Force overwrite existing output

# Schema management commands
milvus-ingest schema list                    # List all schemas
milvus-ingest schema show simple            # Show schema details
milvus-ingest schema add myschema file.json # Add custom schema
milvus-ingest schema remove myschema        # Remove custom schema
milvus-ingest schema help                   # Schema format help

# Utility commands
milvus-ingest clean                         # Clean up generated files

# Upload to S3/MinIO (standalone upload, useful for separate upload/import workflow)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/              # Upload to AWS S3
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --endpoint-url http://localhost:9000  # Upload to MinIO
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --no-verify-ssl  # Disable SSL verification
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --access-key-id KEY --secret-access-key SECRET  # With credentials

# Send data to Milvus
# Direct insert to Milvus (reads local parquet and JSON files and creates collection)
milvus-ingest to-milvus insert ./output                                # Insert to local Milvus
milvus-ingest to-milvus insert ./output --uri http://192.168.1.100:19530 --token your-token  # Remote Milvus with auth
milvus-ingest to-milvus insert ./output --drop-if-exists               # Drop existing collection and recreate
milvus-ingest to-milvus insert ./output --collection-name my_collection --batch-size 5000  # Custom settings

# Bulk import to Milvus (upload + import in one step)
# Note: Combines upload and import for convenience, includes auto-collection creation
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000  # Upload and import
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --collection-name my_collection  # Override collection name
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait  # Wait for completion
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret  # With credentials
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --drop-if-exists  # Drop and recreate

# Verify data in Milvus
# Comprehensive verification system with three levels (all include query/search correctness tests)

# Level 1: Row count + query tests (default, fastest)
milvus-ingest to-milvus verify ./output                                # Row count + query correctness
milvus-ingest to-milvus verify ./output --level count                  # Explicit level specification

# Level 2: Scalar fields + query tests (excludes vectors for performance)
milvus-ingest to-milvus verify ./output --level scalar                 # Verify scalar fields + queries

# Level 3: All fields + query tests (includes vectors, most comprehensive)
milvus-ingest to-milvus verify ./output --level full                   # Full field verification + queries

# Additional verification options
milvus-ingest to-milvus verify ./output --level full --collection-name my_collection  # Specific collection
milvus-ingest to-milvus verify ./output --level scalar --uri http://remote:19530 --token your-token  # Remote Milvus
```

## Architecture Overview

### Project Structure
```
milvus-ingest/
├── src/milvus_ingest/          # Main package (renamed from milvus_fake_data)
│   ├── cli.py                  # CLI entry point with Click commands
│   ├── optimized_writer.py     # High-performance vectorized data generation
│   ├── models.py               # Pydantic schema validation models
│   ├── schema_manager.py       # Schema management and storage
│   ├── milvus_inserter.py      # Direct Milvus insertion with batch processing
│   ├── milvus_importer.py      # Bulk import from S3/MinIO with job tracking
│   ├── milvus_schema_builder.py # Milvus collection schema builder
│   ├── uploader.py             # S3/MinIO upload functionality
│   ├── builtin_schemas.py      # Built-in schema definitions and metadata
│   ├── generator.py            # Legacy generator (for compatibility)
│   ├── rich_display.py         # Rich terminal formatting and progress bars
│   ├── logging_config.py       # Loguru-based structured logging
│   ├── exceptions.py           # Custom exception classes
│   └── schemas/                # 16 built-in schema JSON files
├── tests/                      # Test suite
│   ├── conftest.py            # Common test fixtures
│   └── test_*.py              # Test modules
├── deploy/                     # Docker development environment
│   └── docker-compose.yml     # Local Milvus + MinIO stack
├── pyproject.toml             # PDM configuration and metadata
├── Makefile                   # Common development tasks
└── README.md                  # User documentation
```

### High-Level Architecture

This is a high-performance mock data generator for Milvus vector databases with several key design principles:

1. **Performance-First Design**: Uses vectorized NumPy operations to achieve 10,000-100,000+ rows/second generation speed. The `optimized_writer.py` module handles batch processing and memory-efficient streaming.

2. **Schema-Driven Generation**: All data generation is driven by JSON schemas validated with Pydantic models. Schemas define field types, dimensions, cardinality, and generation modes (faker patterns, ranges, custom values).

3. **Flexible Output Pipeline**:
   - Generate → Parquet/JSON files → Direct insert to Milvus
   - Generate → Parquet files → Upload to S3/MinIO → Bulk import to Milvus

4. **Smart File Partitioning**: Automatically splits output into multiple files based on configurable limits (default: 256MB or 1M rows per file) to prevent memory issues and optimize import performance. Supports precise file size control for pipeline testing scenarios.

5. **Built-in Schema Library**: Includes 16 pre-configured schemas for common use cases (simple vectors, e-commerce, documents, images, etc.) with support for custom schemas.

6. **CLI Architecture**: Click-based command groups (generate, schema, upload, to-milvus, clean) with rich terminal output for better user experience.

7. **Milvus Integration**: Complete integration with Milvus including collection creation, partition/shard configuration, direct insertion, bulk import from S3/MinIO storage, and comprehensive data verification.

8. **Comprehensive Verification System**: Multi-level verification system that ensures data integrity, field consistency, and functional correctness through automated query and search testing.

## Key Implementation Details

### Verification System Architecture
The verification system provides three progressive levels of data validation:

#### Level 1: Count + Query Tests (Default)
- **Row Count Verification**: Compares total rows in Milvus collection with expected count from meta.json
- **Query Correctness Tests**: Executes 1000 sample exact queries to verify data retrieval functionality 
- **Vector Search Tests**: Performs 1000 sample vector similarity searches to verify search functionality
- **Use Case**: Quick verification of data import success and basic functionality
- **Performance**: Fastest option, minimal resource usage

#### Level 2: Scalar + Query Tests  
- **Row Count Verification**: Same as Level 1
- **Scalar Field Validation**: Compares scalar field values between source data and Milvus (excludes vectors)
- **AUTO_ID Handling**: Intelligently skips auto-generated primary key fields  
- **Query Correctness Tests**: Same as Level 1
- **Vector Search Tests**: Same as Level 1
- **Use Case**: Business data accuracy verification with performance optimization
- **Performance**: Moderate resource usage, skips compute-intensive vector comparisons

#### Level 3: Full + Query Tests
- **Row Count Verification**: Same as Level 1
- **All Field Validation**: Compares ALL field values including vector fields between source and Milvus
- **Vector Field Precision**: Uses numpy.allclose() for floating-point vector comparison with 1e-6 tolerance
- **Query Correctness Tests**: Same as Level 1  
- **Vector Search Tests**: Same as Level 1
- **Use Case**: Complete data quality assurance and comprehensive validation
- **Performance**: Most resource-intensive, includes vector field comparisons

#### Verification Features
- **Smart Primary Key Detection**: Supports both `is_primary` and `is_primary_key` field attributes
- **Index-based Comparison**: For AUTO_ID scenarios, uses row index matching instead of primary key lookup
- **Data Type Handling**: Specialized comparison logic for vectors, floats, strings, JSON, and arrays
- **Error Tolerance**: Allows up to 5% field mismatch rate for floating-point precision differences
- **Rich Output**: Detailed verification tables and summary reports with pass/fail status
- **Query Testing**: Each level includes 1000-sample query correctness validation
  - Exact queries: 95% success rate threshold
  - Vector searches: 80% success rate threshold (accounts for approximate search nature)

### Parameter Naming Convention
The CLI uses consistent parameter naming after recent updates:
- `--total-rows` (not `--rows`) - Total number of rows to generate
- `--file-size` (not `--max-file-size`) - File size limit (supports units like "10GB", "256MB")
- `--rows-per-file` (not `--max-rows-per-file`) - Maximum rows per file
- `--partitions` - Number of Milvus partitions (requires partition key field in schema)
- `--shards` - Number of Milvus shards/VChannels (distributes data based on primary key hash)

### File Size Control Logic
The system handles parameter conflicts intelligently:
- When both `--file-count` and `--file-size` are specified, `--total-rows` is ignored
- Total rows are calculated as: `file_count × estimated_rows_per_file`
- File size estimation uses sampling of 1000 rows for accuracy

### Meta.json Structure
Generated data includes a `meta.json` file with:
```json
{
  "schema": { ... },           // Original schema definition
  "generation_info": { ... },  // Generation statistics and file list
  "collection_config": {       // Milvus collection configuration
    "num_partitions": 8,
    "partition_key_field": "user_id",
    "num_shards": 4
  }
}
```

### Data Distribution Rules
- **Primary Key Uniqueness**: Ensured across all files using offset-based generation
- **Partition Key Cardinality**: Unique values = `num_partitions × 10` for proper distribution
- **Shard Distribution**: Based on primary key hash for even distribution across VChannels

## Important Technical Patterns

### Error Handling
- All user-facing errors use custom exceptions from `exceptions.py`
- Schema validation errors provide detailed fix suggestions using Pydantic's validation
- CLI commands use Rich library for formatted error display

### Performance Optimization Patterns
1. **Vectorized Operations**: The `optimized_writer.py` uses NumPy for batch generation instead of row-by-row operations
2. **Memory Management**: Data is written in configurable batches (default 50K rows) to prevent memory exhaustion
3. **File Partitioning**: Automatic file splitting based on size (256MB) or row count (1M) limits
4. **Parallel Processing**: Vector normalization and certain generation tasks use multiple CPU cores

### Schema Extension Points
- Custom field types can be added by extending the `FieldSchema` model in `models.py`
- Generation strategies are defined in `optimized_writer.py` using the `_generate_*` methods
- New built-in schemas go in `schemas/` directory with metadata in `builtin_schemas.py`

### Testing Patterns
- Unit tests mock external dependencies (Milvus, S3/MinIO)
- Integration tests use the Docker Compose stack in `deploy/`
- Performance tests are marked with `@pytest.mark.slow` decorator
- Test fixtures are defined in `tests/conftest.py`

### CLI Design Patterns
- Commands are grouped: `generate`, `schema`, `upload`, `to-milvus`, `clean`
- All commands support `--help` for detailed documentation
- Progress bars and rich formatting are provided by `rich_display.py`
- Verbose logging controlled by `--verbose` flag using loguru

## Common Development Tasks

### Adding a New Schema Field Type
1. Add the type to `models.py` in the `FieldType` enum
2. Implement generation logic in `optimized_writer.py` 
3. Update validation in `models.py` if needed
4. Add test cases in `tests/test_models.py`

### Adding a New Built-in Schema
1. Create JSON schema file in `src/milvus_ingest/schemas/`
2. Register in `builtin_schemas.py` with metadata
3. Add example usage to README.md
4. Test with: `milvus-ingest generate --builtin <schema_name> --preview`

### Debugging Performance Issues
1. Enable verbose logging: `LOGURU_LEVEL=DEBUG milvus-ingest generate ...`
2. Check batch processing in logs for memory usage
3. Profile with: `python -m cProfile -o profile.stats src/milvus_ingest/cli.py ...`
4. Analyze with: `python -m pstats profile.stats`

### Release Process
1. Update version in `pyproject.toml`
2. Run full test suite: `make check`
3. Build package: `make build`
4. Test installation: `pip install dist/*.whl`
5. Publish: `PDM_PUBLISH_TOKEN=<token> make publish`

## Performance Benchmarking

The `bench/` directory contains a comprehensive benchmarking suite for testing data generation performance:

### Benchmark Tools
1. **benchmark_generator.py** - Full pipeline benchmarking across schemas and configurations
2. **benchmark_optimized.py** - Direct OptimizedDataWriter performance testing
3. **profile_memory.py** - Memory usage profiling and leak detection

### Running Benchmarks
```bash
# Install additional benchmark dependencies
pip install memory-profiler matplotlib psutil

# Quick full pipeline benchmark
python bench/benchmark_generator.py --quick

# Comprehensive benchmark with multiple schemas
python bench/benchmark_generator.py \
    --schemas simple ecommerce documents \
    --rows 10000 100000 1000000 \
    --batch-sizes 10000 50000 100000

# Test OptimizedWriter performance directly
python bench/benchmark_optimized.py --mode all --rows 1000000

# Profile memory usage
python bench/profile_memory.py \
    --schema ecommerce \
    --rows 500000 \
    --batch-sizes 10000 50000 100000 \
    --plot --detailed
```

### Expected Performance
- **Numeric fields**: 50,000-100,000+ rows/sec
- **String fields**: 30,000-70,000 rows/sec  
- **Vector fields (768d)**: 15,000-25,000 rows/sec
- **Memory usage**: ~100-200MB base + 50-200MB per 50K batch
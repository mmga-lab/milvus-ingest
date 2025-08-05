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

data path parent is ./data/

```bash
# Install in development mode first
pdm install

# High-performance data generation commands (optimized for large-scale datasets)
milvus-ingest generate --builtin quickstart --total-rows 100000 --preview        # Generate 100K rows (formerly --rows)
milvus-ingest generate --schema example_schema.json --total-rows 1000000     # Generate 1M rows
milvus-ingest generate --schema schema.json --total-rows 5000000 --batch-size 100000 # Use large batch size

# Advanced file control options (for pipeline testing scenarios)
milvus-ingest generate --schema schema.json --total-rows 1000000 --file-size 256MB   # Set specific file size (formerly --max-file-size)
milvus-ingest generate --schema schema.json --file-count 10 --file-size 10GB         # Generate 10×10GB files
milvus-ingest generate --schema schema.json --file-count 500 --file-size 200MB       # Generate 500×200MB files
milvus-ingest generate --schema schema.json --total-rows 1000000 --rows-per-file 500000 # Set max rows per file (formerly --max-rows-per-file)

# Partition and shard configuration (for distributed testing)
milvus-ingest generate --schema schema.json --total-rows 1000000 --partitions 8 --shards 4  # Multi-partition setup
milvus-ingest generate --schema schema.json --total-rows 5000000 --partitions 1024 --shards 16  # Max partitions/shards

# Worker configuration for parallel processing
milvus-ingest generate --schema schema.json --total-rows 10000000 --workers 8  # Use 8 parallel workers

# Chunk-and-merge strategy for large files (auto-enabled for >=2GB files)
# Note: chunk-and-merge is automatically enabled for files >= 2GB to improve performance
milvus-ingest generate --schema schema.json --file-size 10GB  # Auto-uses chunk-and-merge for 10GB file
milvus-ingest generate --schema schema.json --file-size 5GB --chunk-size 500MB  # Auto-enabled with custom chunk size
milvus-ingest generate --schema schema.json --file-count 3 --file-size 8GB  # Auto-enabled for multiple large files
milvus-ingest generate --schema schema.json --file-size 1GB --chunk-and-merge  # Manually enable for files <2GB

# Additional generation options
milvus-ingest generate --schema schema.json --validate-only            # Validate schema without generating
milvus-ingest generate --schema schema.json --total-rows 1000000 --no-progress # Disable progress bar
milvus-ingest generate --schema schema.json --out mydata --force   # Force overwrite existing output

# Cache functionality (reuse previously generated data with identical parameters)
milvus-ingest generate --schema schema.json --total-rows 1000000 --use-cache           # Enable cache, reuse if available
milvus-ingest generate --schema schema.json --total-rows 1000000 --force-regenerate    # Force regeneration, update cache

# Schema management commands
milvus-ingest schema list                    # List all schemas
milvus-ingest schema show quickstart            # Show schema details
milvus-ingest schema add myschema file.json # Add custom schema
milvus-ingest schema remove myschema        # Remove custom schema
milvus-ingest schema help                   # Schema format help

# Cache management commands
milvus-ingest cache list                            # List all cached datasets
milvus-ingest cache info <cache_key>                # Show detailed cache information  
milvus-ingest cache stats                           # Show cache statistics and usage
milvus-ingest cache clean --all --yes               # Remove all cached datasets
milvus-ingest cache clean --older-than 7           # Remove caches older than 7 days
milvus-ingest cache clean <cache_key1> <cache_key2> # Remove specific caches

# Utility commands
milvus-ingest clean                         # Clean up generated files

# Upload to S3/MinIO (standalone upload, useful for separate upload/import workflow)
# Note: Multiple upload methods supported (AWS CLI default, mc CLI for MinIO, boto3 legacy)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/              # Upload to AWS S3 (uses AWS CLI by default)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --endpoint-url http://localhost:9000  # Upload to MinIO (uses AWS CLI by default)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --endpoint-url http://localhost:9000 --use-mc  # Upload to MinIO using mc CLI (recommended for MinIO)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --no-verify-ssl  # Disable SSL verification
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --access-key-id KEY --secret-access-key SECRET --use-mc  # With credentials using mc CLI
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --use-boto3   # Use boto3 instead of AWS CLI (legacy mode)

# Send data to Milvus
# Direct insert to Milvus (reads local parquet and JSON files and creates collection)
milvus-ingest to-milvus insert ./output                                # Insert to local Milvus (uses FLAT index by default)
milvus-ingest to-milvus insert ./output --uri http://192.168.1.100:19530 --token your-token  # Remote Milvus with auth
milvus-ingest to-milvus insert ./output --drop-if-exists               # Drop existing collection and recreate
milvus-ingest to-milvus insert ./output --collection-name my_collection --batch-size 5000  # Custom settings
milvus-ingest to-milvus insert ./output --use-autoindex                # Use AUTOINDEX for better performance (lower recall)

# Bulk import to Milvus (upload + import in one step)  
# Note: Combines upload and import for convenience, includes auto-collection creation
# Note: Multiple upload methods supported (AWS CLI default, mc CLI for MinIO, boto3 legacy)
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000  # Upload and import (uses AWS CLI + FLAT index by default)
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --use-mc  # Upload and import using mc CLI (recommended for MinIO)
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --collection-name my_collection --use-mc  # Override collection name with mc CLI
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait --use-mc  # Wait for completion with mc CLI
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret --use-mc  # With credentials using mc CLI
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --drop-if-exists --use-mc  # Drop and recreate with mc CLI
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --use-autoindex --use-mc  # Use AUTOINDEX for better performance with mc CLI
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --use-boto3  # Use boto3 instead of AWS CLI (legacy mode)

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

### Validation System

The system includes comprehensive validation at multiple stages to ensure data integrity:

#### Local File Validation (Default)
Automatically runs after data generation to ensure file integrity:
- **File Readability**: Verifies files can be opened and schema can be read
- **Row Count Verification**: Confirms actual row count matches expected count from meta.json
- **File Size Validation**: Validates exact file size matches recorded size (byte-level precision)
- **Performance**: Very fast, only reads file metadata (not actual data)
- **Coverage**: Validates all generated files for basic integrity

```bash
# Minimal validation is automatically run after generation
milvus-ingest generate --builtin quickstart --total-rows 10000
# Output includes: "✅ File validation passed: 1 files, 10,000 rows"
```

#### S3 Upload Validation
Comprehensive validation after S3/MinIO upload to ensure upload integrity:
- **File Existence**: Confirms all files exist in S3/MinIO
- **Row Count Verification**: Uses optimized streaming to verify row counts without full download
- **File Size Validation**: Validates uploaded file sizes match local file sizes exactly
- **Format-Optimized Reading**: 
  - **Parquet**: Reads only footer metadata (8-64KB) using S3 Range requests
  - **JSON Array**: Streaming parser for large files, direct parsing for small files (<10MB)

The validation system performs three essential checks:
1. **Schema Access**: Can the file be opened and metadata read? (proves file is not corrupted)
2. **Count Verification**: Does the actual row count match the expected count? (proves file is complete)
3. **Size Verification**: Does the file size match exactly? (proves upload integrity)

This multi-layered approach ensures complete data integrity from generation through upload while maintaining high performance.

## Upload Methods

The system supports three upload methods for S3-compatible storage, with intelligent selection based on the scenario:

### 1. AWS CLI (Default for AWS S3)

**Automatic Installation:**
AWS CLI is the default upload method for AWS S3 and general S3-compatible storage. If AWS CLI is not installed, the tool will automatically attempt to install it:

```bash
# AWS CLI will be automatically installed if not found
milvus-ingest upload --local-path ./output --s3-path s3://bucket/data/

# Installation attempts (in order), all using --user flag for permission-free install:
# 1. pip install awscli --upgrade --user
# 2. pip3 install awscli --upgrade --user  
# 3. python -m pip install awscli --upgrade --user
```

**Key Features:**
- **Permission-Free**: Uses `--user` flag to install in user directory
- **Auto PATH Setup**: Automatically adds AWS CLI to PATH for current session
- **PATH Guidance**: Provides instructions to add to PATH permanently
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Fallback Options
If automatic installation fails, you can:
1. **Manual Installation**: `pip install awscli --upgrade --user`
2. **Use boto3 Legacy Mode**: Add `--use-boto3` flag to use the previous method

### Example PATH Setup Output
When AWS CLI is installed with `--user`, you'll see guidance like:
```
⚠️  AWS CLI installed to /Users/yourname/.local/bin
   Add this to your PATH for future sessions:
   export PATH="/Users/yourname/.local/bin:$PATH"
```

### Benefits of AWS CLI (Default)
- **More Reliable**: Better error handling and retry mechanisms
- **Large File Support**: Optimized for files >1GB with automatic multipart upload
- **Network Resilience**: Better handling of network interruptions and timeouts
- **Production Ready**: Battle-tested upload reliability
- **Automatic Installation**: Zero-configuration setup

### 2. MinIO Client (mc) CLI (Recommended for MinIO)

**Automatic Installation:**
The MinIO Client (mc) is the native CLI for MinIO and provides optimal performance for MinIO deployments. If mc CLI is not installed, the tool will automatically attempt to install it:

```bash
# mc CLI will be automatically installed if not found when using --use-mc
milvus-ingest upload --local-path ./output --s3-path s3://bucket/data/ --endpoint-url http://localhost:9000 --use-mc
```

**Installation Process:**
- Downloads the appropriate mc binary for your OS/architecture from MinIO's official repository
- Installs to `~/.local/bin/` directory with proper permissions
- Automatically configures PATH for the current session
- Provides guidance for permanent PATH setup

**Key Features:**
- **Native MinIO Support**: Designed specifically for MinIO servers
- **Optimal Performance**: Direct integration with MinIO protocols
- **Automatic Configuration**: Creates temporary aliases for connection management
- **Cross-Platform**: Supports Linux, macOS, and Windows (x86_64, ARM64)
- **SSL Control**: Supports `--insecure` flag for self-signed certificates

**Usage Examples:**
```bash
# Upload to MinIO using mc CLI (recommended)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/data/ --endpoint-url http://localhost:9000 --use-mc

# With credentials
milvus-ingest upload --local-path ./output --s3-path s3://bucket/data/ --endpoint-url http://localhost:9000 --access-key-id mykey --secret-access-key mysecret --use-mc

# Disable SSL verification
milvus-ingest upload --local-path ./output --s3-path s3://bucket/data/ --endpoint-url http://localhost:9000 --no-verify-ssl --use-mc

# Bulk import with mc CLI
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --use-mc
```

**Benefits of mc CLI for MinIO:**
- **Native Protocol Support**: Optimized for MinIO's native protocols
- **Better Error Messages**: More detailed MinIO-specific error reporting
- **Alias Management**: Automatic temporary alias creation and cleanup
- **Performance**: Often faster than generic S3 clients for MinIO
- **MinIO Features**: Access to MinIO-specific features and optimizations

### 3. boto3 (Legacy Mode)

**When to Use boto3 Legacy Mode:**
- AWS CLI or mc CLI installation issues in restricted environments
- Specific boto3 compatibility requirements
- Environments where subprocess execution is restricted
- Testing and debugging scenarios

```bash
# Force boto3 usage (legacy mode)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/data/ --use-boto3
```

### Upload Method Selection Priority

The system automatically selects the upload method based on the following priority:
1. **mc CLI** (if `--use-mc` flag is specified)
2. **AWS CLI** (if `--use-boto3` is not specified, default)
3. **boto3** (if `--use-boto3` flag is specified, legacy fallback)

**Recommendations:**
- **For MinIO**: Use `--use-mc` for optimal performance and compatibility
- **For AWS S3**: Use default AWS CLI (no additional flags needed)
- **For Other S3-Compatible**: Use AWS CLI (default) or boto3 (`--use-boto3`) based on compatibility
- **For Restricted Environments**: Use `--use-boto3` as a fallback option

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
│   ├── verifier.py             # Comprehensive Milvus data verification system
│   ├── minimal_validator.py    # Fast local file integrity validation (schema + count + size)
│   ├── s3_minimal_validator.py # S3/MinIO upload validation with optimized streaming
│   ├── lightweight_validator.py # More thorough validation with sampling (legacy)
│   ├── file_merger.py          # File merging for chunk-and-merge strategy
│   ├── rich_display.py         # Rich terminal formatting and progress bars
│   ├── logging_config.py       # Loguru-based structured logging
│   ├── exceptions.py           # Custom exception classes
│   ├── cache_manager.py        # Dataset caching for parameter-based reuse
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

9. **Intelligent Caching System**: SHA256-based parameter caching that automatically reuses previously generated datasets when identical generation parameters are detected, dramatically reducing time for repeated operations on large datasets.

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

### Parameter Naming Convention (Updated)
The CLI uses consistent parameter naming after recent updates:
- `--total-rows` (not `--rows`) - Total number of rows to generate
- `--file-size` (not `--max-file-size`) - File size limit (supports units like "10GB", "256MB")
- `--rows-per-file` (not `--max-rows-per-file`) - Maximum rows per file
- `--partitions` - Number of Milvus partitions (requires partition key field in schema)
- `--shards` - Number of Milvus shards/VChannels (distributes data based on primary key hash)
- `--workers` - Number of parallel worker processes for file generation (default: CPU count)
- `--file-count` - Target number of files (when used with --file-size, calculates total rows automatically)

### File Size Control Logic
The system handles parameter conflicts intelligently:
- When both `--file-count` and `--file-size` are specified, `--total-rows` is ignored
- Total rows are calculated as: `file_count × estimated_rows_per_file`
- File size estimation uses sampling of 1000 rows for accuracy

### Meta.json Structure
Generated data includes a `meta.json` file with detailed file information:
```json
{
  "schema": { ... },           // Original schema definition
  "generation_info": {
    "total_rows": 1000000,
    "format": "parquet",
    "data_files": [              // Detailed file information (new format)
      {
        "file_name": "data.parquet",
        "file_path": "output/data.parquet", 
        "rows": 1000000,
        "file_index": 0,
        "file_size_bytes": 157041    // Exact file size for validation
      }
    ],
    "file_count": 1,
    // ... other generation statistics
  },
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

### JSON Format Support
The system generates JSON files in **JSON Array format only** (`[{...}, {...}, ...]`) as required by Milvus bulk import:

#### JSON Generation Strategy
- **Format**: Always JSON Array format for Milvus compatibility
- **Size Handling**: Small files (<32KB) for direct parsing, streaming for larger files
- **Validation**: Optimized row counting with format-specific strategies:
  - **Small files (<32KB)**: Direct JSON parsing 
  - **Medium files (<10MB)**: Complete download and parsing
  - **Large files (>10MB)**: Streaming array parser with 64KB chunks

#### JSON vs Parquet Performance
- **Parquet**: Fastest generation and validation (metadata-only reads)
- **JSON**: Slower generation but better human readability
- **Recommendation**: Use Parquet for large datasets, JSON for debugging/inspection

## Vector Index Strategy

### FLAT vs AUTOINDEX for Vector Fields

The system provides two indexing strategies for dense vector fields (FloatVector, Float16Vector, BFloat16Vector). Note that sparse vectors (SparseFloatVector) and binary vectors (BinaryVector) always use their specialized index types regardless of this setting:

#### AUTOINDEX (`--use-autoindex`)
- **Performance**: Optimized for speed and memory efficiency
- **Recall**: ~90-95% typical recall (approximate search)
- **Memory**: Lower memory usage
- **Use Cases**: Production workloads, large datasets, when speed > perfect accuracy

#### FLAT Index (Default for Dense Vectors)
- **Performance**: Slower but guaranteed exact results  
- **Recall**: 100% recall (brute-force exact search)
- **Memory**: Higher memory usage (stores all vectors in memory)
- **Use Cases**: Verification, testing, small datasets, when accuracy is critical
- **Applies to**: FloatVector, Float16Vector, BFloat16Vector only

#### When to Use AUTOINDEX
```bash
# For production workloads where speed > perfect accuracy
milvus-ingest to-milvus insert ./output --use-autoindex

# For large datasets with memory constraints
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket prod --use-autoindex

# When you need better performance and can accept ~90-95% recall
```

#### Index Types by Vector Field Type
- **FloatVector, Float16Vector, BFloat16Vector**: FLAT (default) or AUTOINDEX (with `--use-autoindex`)
- **SparseFloatVector**: Always uses SPARSE_INVERTED_INDEX (regardless of setting)
- **BinaryVector**: Always uses AUTOINDEX with HAMMING metric (FLAT not supported)

⚠️ **Memory Considerations**: FLAT index (default for dense vectors) loads all vectors into memory. Ensure sufficient RAM for your dataset size. Use `--use-autoindex` for memory-constrained environments.

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

### Validation Architecture Patterns
1. **Zero-Tolerance Validation**: File sizes and row counts must match exactly (byte-level precision)
2. **Optimized S3 Reading**: 
   - **Parquet**: Uses S3 Range requests to read only metadata footer (8-64KB)
   - **JSON**: Streaming parser for large files, direct parsing for small files
3. **Multi-Stage Validation**: Local validation → Upload validation → Optional Milvus verification
4. **Metadata-Driven**: All validation based on detailed file information stored in meta.json

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

## Complete CLI Command Reference

### Data Generation (`generate` command)
```bash
milvus-ingest generate [OPTIONS]

Options:
  --schema PATH                Path to schema JSON/YAML file
  --builtin TEXT              Use a built-in schema (e.g., 'ecommerce', 'documents')
  --total-rows INTEGER        Total number of rows to generate (default: 1000)
  --format {parquet|json}     Output file format (default: parquet)
  --preview                   Print first 5 rows to terminal after generation
  --out PATH                  Output directory path (default: <collection_name>/)
  --seed INTEGER              Random seed for reproducibility
  --validate-only             Only validate schema without generating data
  --no-progress               Disable progress bar display
  --batch-size INTEGER        Batch size for processing (default: 50000)
  --force                     Force overwrite output directory if it exists
  --file-size TEXT            File size limit (e.g., '10GB', '256MB', default: 256MB)
  --rows-per-file INTEGER     Maximum rows per file (default: 1000000)
  --file-count INTEGER        Target number of files (overrides --total-rows when used with --file-size)
  --partitions INTEGER        Number of Milvus partitions to simulate
  --shards INTEGER            Number of shards (VChannels) to simulate
  --workers INTEGER           Number of parallel worker processes (default: CPU count)
  -v, --verbose               Enable verbose logging
```

### Schema Management (`schema` commands)
```bash
milvus-ingest schema list              # List all available schemas
milvus-ingest schema show SCHEMA_ID    # Show details of a specific schema
milvus-ingest schema add SCHEMA_ID FILE # Add a custom schema
milvus-ingest schema remove SCHEMA_ID  # Remove a custom schema
milvus-ingest schema help              # Show schema format help
```

### Data Upload (`upload` command)
```bash
milvus-ingest upload [OPTIONS]

Required Options:
  --local-path PATH           Local path to data directory to upload
  --s3-path TEXT              S3 destination path (e.g., s3://bucket/prefix/)

Optional Options:
  --endpoint-url TEXT         S3-compatible endpoint URL (e.g., http://localhost:9000)
  --access-key-id TEXT        AWS access key ID (or use AWS_ACCESS_KEY_ID env var)
  --secret-access-key TEXT    AWS secret access key (or use AWS_SECRET_ACCESS_KEY env var)
  --region TEXT               AWS region name (default: us-east-1)
  --no-verify-ssl             Disable SSL certificate verification
  --no-progress               Disable progress bar during upload
```

### Milvus Operations (`to-milvus` commands)

#### Direct Insert
```bash
milvus-ingest to-milvus insert DATA_PATH [OPTIONS]

Options:
  --uri TEXT                  Milvus server URI (default: http://localhost:19530)
  --token TEXT                Token for authentication
  --db-name TEXT              Database name (default: default)
  --collection-name TEXT      Override collection name from metadata
  --drop-if-exists            Drop collection if it already exists
  --no-index                  Skip creating indexes on vector fields
  --batch-size INTEGER        Batch size for inserting (default: 10000)
  --no-progress               Disable progress bar
  --use-autoindex             Use AUTOINDEX for dense vector fields (faster but ~90-95% recall, overrides default FLAT index)
```

#### Bulk Import (Upload + Import)
```bash
milvus-ingest to-milvus import [OPTIONS]

Required Options:
  --local-path PATH           Local data directory path
  --s3-path TEXT              S3 path (relative to bucket)
  --bucket TEXT               S3/MinIO bucket name

Optional Options:
  --collection-name TEXT      Target collection name (overrides metadata)
  --endpoint-url TEXT         S3/MinIO endpoint URL
  --access-key-id TEXT        S3/MinIO access key ID
  --secret-access-key TEXT    S3/MinIO secret access key
  --no-verify-ssl             Disable SSL verification
  --uri TEXT                  Milvus URI (default: http://127.0.0.1:19530)
  --token TEXT                Authentication token
  --wait                      Wait for import to complete
  --timeout INTEGER           Timeout in seconds when waiting
  --drop-if-exists            Drop collection if exists before creating
  --use-autoindex             Use AUTOINDEX for dense vector fields (faster but ~90-95% recall, overrides default FLAT index)
```

#### Data Verification
```bash
milvus-ingest to-milvus verify DATA_PATH [OPTIONS]

Options:
  --collection-name TEXT      Collection name to verify
  --uri TEXT                  Milvus server URI (default: http://localhost:19530)
  --token TEXT                Token for authentication
  --db-name TEXT              Database name (default: default)
  --level {count|scalar|full} Verification level (default: count)
                              - count: Row count + query tests
                              - scalar: Scalar fields + query tests
                              - full: All fields + query tests
```

### Utility Commands
```bash
milvus-ingest clean [OPTIONS]          # Clean up generated output files
  --yes, -y                            # Auto-confirm all prompts
```

## Cache System Architecture

The caching system provides intelligent reuse of previously generated datasets when identical parameters are used:

### Cache Key Generation
- **SHA256-based**: Generates deterministic hashes from normalized generation parameters
- **Parameter Filtering**: Only includes parameters that affect data generation results
- **Schema Normalization**: JSON schema content is parsed and re-serialized for consistency
- **Excluded Parameters**: Output paths, progress flags, and UI options don't affect cache keys

### Cache Storage Structure
```
~/.milvus-ingest/cache/
├── <cache_key>/                 # SHA256 hash directory
│   ├── cache_info.json         # Cache metadata and generation parameters  
│   ├── meta.json               # Original dataset metadata
│   └── data-*.parquet          # Cached data files (hard-linked when possible)
```

### Cache Workflow
1. **Generation Request**: User runs generate command with `--use-cache`
2. **Key Generation**: System creates SHA256 hash from normalized parameters
3. **Cache Check**: Validates cache existence and file integrity
4. **Cache Hit**: Hard-links or copies files to target directory (seconds vs hours)
5. **Cache Miss**: Generates data normally, then stores in cache for future use

### Cache Management Features
- **Intelligent File Operations**: Uses hard links for same-filesystem operations, falls back to copying
- **Cache Validation**: Checks file existence, size, and metadata consistency
- **Flexible Cleanup**: Remove by age, size, specific keys, or all caches
- **Rich Management UI**: Detailed cache listings with creation time, size, and parameters
- **Partial Key Matching**: Support shortened cache keys for user convenience

### Performance Benefits
- **Large Dataset Reuse**: Multi-million row datasets cached in seconds instead of hours
- **Development Workflow**: Rapid iteration on downstream processing without regeneration
- **CI/CD Optimization**: Consistent test data generation with deterministic caching
- **Storage Efficiency**: Hard links minimize disk usage when possible

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

## Important Architecture and Usage Notes

### Running Tests Against Remote Test Environment

If you need to run tests against a remote test environment instead of local Docker Compose:

```bash
# Remote test environment variables (example)
export MILVUS_URI=http://10.104.17.43:19530
export MINIO_HOST=10.104.17.44
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=milvus-bucket

# Run tests with remote environment
pdm run pytest
```

### Important Parameter Changes

The CLI underwent significant parameter naming changes. Always use the new names:
- Use `--total-rows` (NOT `--rows`)
- Use `--file-size` (NOT `--max-file-size`)
- Use `--rows-per-file` (NOT `--max-rows-per-file`)

### Common Issues and Solutions

#### Large File Generation (>2GB)
The system automatically enables chunk-and-merge strategy for files ≥2GB:
```bash
# Automatically uses chunk-and-merge
milvus-ingest generate --schema schema.json --file-size 10GB

# Manually control chunk size
milvus-ingest generate --schema schema.json --file-size 5GB --chunk-size 500MB
```

#### Memory Issues with Large Datasets
- Adjust `--batch-size` (default: 50000) to lower values
- Use `--workers` to control parallel processing
- Monitor memory usage with verbose logging: `LOGURU_LEVEL=DEBUG`

#### Upload Failures
- AWS CLI is now the default (more reliable for large files)
- Falls back to boto3 if AWS CLI not available
- Force boto3 with `--use-boto3` flag if needed

### Key Performance Considerations

1. **File Format Choice**: Always use Parquet for large datasets (default) - it's significantly faster than JSON
2. **Batch Size Tuning**: For very large vector dimensions (>1024), consider reducing batch size to 10K-20K
3. **Worker Processes**: Default is CPU count, but for I/O-bound operations, 2x CPU count may be optimal
4. **File Partitioning**: The system automatically partitions at 256MB/1M rows - this is optimized for Milvus import

### Development Workflow Tips

1. **Testing Schema Changes**: Always use `--validate-only` first when modifying schemas
2. **Debugging Generation**: Use `--preview` to inspect first 5 rows without full generation
3. **Cache Usage**: When iterating on downstream processing, use `--use-cache` to avoid regeneration
4. **Verification**: Always verify after import - use appropriate level based on needs (count → scalar → full)
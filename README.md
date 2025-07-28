# Milvus Ingest - High-Performance Data Ingestion Tool

🚀 **Ultra-fast data ingestion tool for Milvus vector databases** - Built for large-scale data generation and ingestion with vectorized operations, parallel processing, and optimized file I/O. Generate and ingest millions of rows in seconds with automatic file partitioning and intelligent memory management.

## ⚡ Performance Highlights

- **🏎️ 10,000-100,000+ rows/sec** - Vectorized NumPy operations for maximum speed
- **📈 Large-scale optimized** - Designed for datasets >100K rows with intelligent batching  
- **🔥 Smart file partitioning** - Automatic splitting (256MB chunks, 1M rows/file)
- **💾 Memory efficient** - Streaming generation prevents memory exhaustion
- **⚡ Direct PyArrow I/O** - Optimized Parquet writing with Snappy compression
- **🔄 Parallel processing** - Multi-core CPU utilization with configurable workers

## ✨ Key Features

- 🎯 **Ready-to-use schemas** - Pre-built schemas for e-commerce, documents, images, users, news, and videos
- 📚 **Schema management** - Add, organize, and reuse custom schemas with metadata
- 🚀 **High-performance generation** - Vectorized operations optimized for large datasets
- 🔧 **Complete Milvus support** - All field types including vectors, arrays, JSON, and primitive types
- ✅ **Smart validation** - Pydantic-based validation with detailed error messages and suggestions
- 📊 **High-performance formats** - Parquet (fastest I/O), JSON (structured data)
- 🌱 **Reproducible results** - Seed support for consistent data generation
- 🎨 **Rich customization** - Field constraints, nullable fields, auto-generated IDs
- 🔍 **Schema exploration** - Validation, help commands, and schema details
- 🏠 **Unified interface** - Use custom and built-in schemas interchangeably
- ✅ **Comprehensive verification** - 3-level validation system with query/search correctness testing
- 🧪 **Functional testing** - Automated query and vector search validation (1000 samples)

## Installation

```bash
# Install from source (recommended for development)
git clone https://github.com/zilliz/milvus-ingest.git
cd milvus-ingest
pdm install  # Installs with development dependencies

# For production use only
pdm install --prod

# After installation, the CLI tool is available as:
milvus-ingest --help
```

## 🚀 Quick Start

### 1. Use Built-in Schemas (Recommended)

Get started instantly with pre-built schemas optimized for large-scale generation:

```bash
# List all available built-in schemas
milvus-ingest schema list

# Generate data using a built-in schema (high-performance by default)
milvus-ingest generate --builtin simple --total-rows 100000 --preview

# Generate large e-commerce dataset with automatic file partitioning
milvus-ingest generate --builtin ecommerce --total-rows 2500000 --out products/

# Import to Milvus and verify data integrity
milvus-ingest to-milvus insert ./products/ --uri http://localhost:19530
milvus-ingest to-milvus verify ./products/ --level scalar --uri http://localhost:19530
```

**Available Built-in Schemas:**
| Schema | Description | Use Cases |
|--------|-------------|-----------|
| `simple` | Basic example with common field types | Learning, testing |
| `ecommerce` | Product catalog with search embeddings | Online stores, recommendations |
| `documents` | Document search with semantic embeddings | Knowledge bases, document search |
| `images` | Image gallery with visual similarity | Media platforms, image search |
| `users` | User profiles with behavioral embeddings | User analytics, personalization |
| `videos` | Video library with multimodal embeddings | Video platforms, content discovery |
| `news` | News articles with sentiment analysis | News aggregation, content analysis |
| `audio_transcripts` | Audio transcription with FP16 embeddings | Speech-to-text search, podcasts |
| `ai_conversations` | AI chat history with BF16 embeddings | Chatbot analytics, conversation search |
| `face_recognition` | Facial recognition with binary vectors | Security systems, identity verification |
| `ecommerce_partitioned` | Partitioned e-commerce schema | Scalable product catalogs |
| `cardinality_demo` | Schema demonstrating cardinality features | Testing cardinality constraints |

### 2. Create Custom Schemas

Define your own collection structure with JSON or YAML:

```json
{
  "collection_name": "my_collection",
  "fields": [
    {
      "name": "id",
      "type": "Int64",
      "is_primary": true
    },
    {
      "name": "title",
      "type": "VarChar",
      "max_length": 256
    },
    {
      "name": "embedding",
      "type": "FloatVector",
      "dim": 128
    }
  ]
}
```

```bash
# Generate large dataset from custom schema with high-performance mode
milvus-ingest generate --schema my_schema.json --total-rows 1000000 --format parquet --preview
```

**Note:** Output is always a directory containing data files (in the specified format) and a `meta.json` file with collection metadata.

### 3. Schema Management

Store and organize your schemas for reuse:

```bash
# Add a custom schema to your library
milvus-ingest schema add my_products product_schema.json

# List all schemas (built-in + custom)
milvus-ingest schema list

# Use your custom schema like a built-in one (optimized for large datasets)
milvus-ingest generate --builtin my_products --total-rows 500000

# Show detailed schema information
milvus-ingest schema show my_products
```

### 4. Python API

```python
from milvus_ingest.generator import generate_mock_data
from milvus_ingest.schema_manager import get_schema_manager
from milvus_ingest.builtin_schemas import load_builtin_schema
import tempfile
import json

# Use the schema manager to work with schemas
manager = get_schema_manager()

# List all available schemas
all_schemas = manager.list_all_schemas()
print("Available schemas:", list(all_schemas.keys()))

# Load any schema (built-in or custom)
schema = manager.get_schema("ecommerce")  # Built-in
# schema = manager.get_schema("my_products")  # Custom

# Generate data from schema file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(schema, f, indent=2)
    f.flush()
    
    # Generate data (returns dict with DataFrame and metadata)
    result = generate_mock_data(f.name, rows=10000, seed=42, output_format="dict")
    df = result["data"]
    metadata = result["metadata"]

print(df.head())
print(f"Generated {len(df)} rows for collection: {metadata['collection_name']}")

# Add a custom schema programmatically
custom_schema = {
    "collection_name": "my_collection",
    "fields": [
        {"name": "id", "type": "Int64", "is_primary": True},
        {"name": "text", "type": "VarChar", "max_length": 100},
        {"name": "vector", "type": "FloatVector", "dim": 256}
    ]
}

manager.add_schema("my_custom", custom_schema, "Custom schema", ["testing"])
print("Added custom schema!")
```

## 📋 Schema Reference

### Supported Field Types

| Type | Description | Required Parameters | Optional Parameters |
|------|-------------|-------------------|-------------------|
| **Numeric Types** | | | |
| `Int8`, `Int16`, `Int32`, `Int64` | Integer types | - | `min`, `max` |
| `Float`, `Double` | Floating point | - | `min`, `max` |
| `Bool` | Boolean values | - | - |
| **Text Types** | | | |
| `VarChar`, `String` | Variable length string | `max_length` | - |
| `JSON` | JSON objects | - | - |
| **Vector Types** | | | |
| `FloatVector` | 32-bit float vectors | `dim` | - |
| `BinaryVector` | Binary vectors | `dim` | - |
| `Float16Vector` | 16-bit float vectors | `dim` | - |
| `BFloat16Vector` | Brain float vectors | `dim` | - |
| `SparseFloatVector` | Sparse float vectors | `dim` | - |
| **Complex Types** | | | |
| `Array` | Array of elements | `element_type`, `max_capacity` | `max_length` (for string elements) |

### Field Properties

| Property | Description | Applicable Types |
|----------|-------------|------------------|
| `is_primary` | Mark field as primary key (exactly one required) | All types |
| `auto_id` | Auto-generate ID values | Int64 primary keys only |
| `nullable` | Allow null values (10% probability) | All types |
| `min`, `max` | Value constraints | Numeric types |
| `max_length` | String/element length limit | String and Array types |
| `dim` | Vector dimension (1-32768) | Vector types |
| `element_type` | Array element type | Array type |
| `max_capacity` | Array capacity (1-4096) | Array type |

### Complete Example

```yaml
collection_name: "advanced_catalog"
fields:
  # Primary key with auto-generated IDs
  - name: "id"
    type: "Int64"
    is_primary: true
    auto_id: true
  
  # Text fields with constraints
  - name: "title"
    type: "VarChar"
    max_length: 200
  
  - name: "description"
    type: "VarChar"
    max_length: 1000
    nullable: true
  
  # Numeric fields with ranges
  - name: "price"
    type: "Float"
    min: 0.01
    max: 9999.99
  
  - name: "rating"
    type: "Int8"
    min: 1
    max: 5
  
  # Vector for semantic search
  - name: "embedding"
    type: "FloatVector"
    dim: 768
  
  # Array of tags
  - name: "tags"
    type: "Array"
    element_type: "VarChar"
    max_capacity: 10
    max_length: 50
  
  # Structured metadata
  - name: "metadata"
    type: "JSON"
    nullable: true
  
  # Boolean flags
  - name: "in_stock"
    type: "Bool"
```

## 📚 CLI Reference

### Command Structure

The CLI uses a clean grouped structure:

```bash
# Main command groups
milvus-ingest generate [options]  # Data generation
milvus-ingest schema [command]    # Schema management
milvus-ingest clean [options]     # Utility commands
```

### Data Generation Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--schema PATH` | Generate from custom schema file | `milvus-ingest generate --schema my_schema.json` |
| `--builtin SCHEMA_ID` | Use built-in or managed schema | `milvus-ingest generate --builtin ecommerce` |
| `--total-rows INTEGER` | Total number of rows to generate | `milvus-ingest generate --total-rows 5000` |
| `--format FORMAT` | Output format (parquet, json) | `milvus-ingest generate --format json` |
| `--out DIRECTORY` | Output directory path | `milvus-ingest generate --out my_data/` |
| `--preview` | Show first 5 rows | `milvus-ingest generate --preview` |
| `--seed INTEGER` | Random seed for reproducibility | `milvus-ingest generate --seed 42` |
| `--validate-only` | Validate schema without generating | `milvus-ingest generate --validate-only` |
| `--no-progress` | Disable progress bar display | `milvus-ingest generate --no-progress` |
| `--batch-size INTEGER` | Batch size for memory efficiency (default: 50000) | `milvus-ingest generate --batch-size 100000` |
| `--file-size TEXT` | File size limit (e.g., '10GB', '256MB', default: 256MB) | `milvus-ingest generate --file-size 100MB` |
| `--rows-per-file INTEGER` | Maximum rows per file (default: 1000000) | `milvus-ingest generate --rows-per-file 500000` |
| `--file-count INTEGER` | Target number of files (overrides --total-rows when used with --file-size) | `milvus-ingest generate --file-count 10 --file-size 1GB` |
| `--partitions INTEGER` | Number of Milvus partitions to simulate | `milvus-ingest generate --partitions 8` |
| `--shards INTEGER` | Number of shards (VChannels) to simulate | `milvus-ingest generate --shards 4` |
| `--workers INTEGER` | Number of parallel worker processes (default: CPU count) | `milvus-ingest generate --workers 8` |
| `--force` | Force overwrite output directory | `milvus-ingest generate --force` |

### Schema Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `schema list` | List all schemas (built-in + custom) | `milvus-ingest schema list` |
| `schema show SCHEMA_ID` | Show schema details | `milvus-ingest schema show ecommerce` |
| `schema add SCHEMA_ID FILE` | Add custom schema | `milvus-ingest schema add products schema.json` |
| `schema remove SCHEMA_ID` | Remove custom schema | `milvus-ingest schema remove products` |
| `schema help` | Show schema format help | `milvus-ingest schema help` |

### Utility Commands

| Command | Description | Example |
|---------|-------------|---------|
| `clean` | Clean up generated output files | `milvus-ingest clean --yes` |
| `--help` | Show help message | `milvus-ingest --help` |

### Common Usage Patterns

```bash
# Quick start with built-in schema (high-performance by default)
milvus-ingest generate --builtin simple --total-rows 100000 --preview

# Generate massive datasets with automatic file partitioning 
milvus-ingest generate --builtin ecommerce --total-rows 5000000 --format parquet --out products/

# Test custom schema validation
milvus-ingest generate --schema my_schema.json --validate-only

# Reproducible large-scale data generation
milvus-ingest generate --builtin users --total-rows 2000000 --seed 42 --out users/

# Control file partitioning (smaller files for easier handling)
milvus-ingest generate --builtin ecommerce --total-rows 5000000 --file-size 128MB --rows-per-file 500000

# Schema management workflow
milvus-ingest schema list
milvus-ingest schema show ecommerce
milvus-ingest schema add my_ecommerce ecommerce_base.json

# Clean up generated output files
milvus-ingest clean --yes
```

## 🔗 Milvus Integration

### Direct Insert to Milvus

Insert generated data directly into Milvus with automatic collection creation:

```bash
# Generate data first
milvus-ingest generate --builtin ecommerce --total-rows 100000 --out products/

# Insert to local Milvus (default: localhost:19530)
milvus-ingest to-milvus insert ./products/

# Insert to remote Milvus with authentication
milvus-ingest to-milvus insert ./products/ \
    --uri http://192.168.1.100:19530 \
    --token your-api-token \
    --db-name custom_db

# Insert with custom settings
milvus-ingest to-milvus insert ./products/ \
    --collection-name product_catalog \
    --batch-size 5000 \
    --drop-if-exists
```

**Direct Insert Features:**
- ✅ Automatic collection creation from metadata
- ✅ Smart index creation based on vector dimensions
- ✅ Progress tracking with batch processing
- ✅ Support for authentication and custom databases
- ✅ Connection testing before import

### Verify Data in Milvus

Comprehensive verification system with three progressive levels of validation (all include query/search correctness tests):

```bash
# Level 1: Row count + Query tests (default, fastest)
milvus-ingest to-milvus verify ./products/                    # Quick verification
milvus-ingest to-milvus verify ./products/ --level count      # Explicit level

# Level 2: Scalar fields + Query tests (excludes vectors for performance)  
milvus-ingest to-milvus verify ./products/ --level scalar     # Business data accuracy

# Level 3: All fields + Query tests (includes vectors, most comprehensive)
milvus-ingest to-milvus verify ./products/ --level full       # Complete validation

# Remote Milvus verification
milvus-ingest to-milvus verify ./products/ --level full \
    --uri http://192.168.1.100:19530 \
    --token your-api-token \
    --collection-name product_catalog
```

**🔍 Comprehensive Verification Features:**

**📊 Progressive Verification Levels:**
- **Level 1 (count)**: Row count verification + 1000-sample query/search tests
- **Level 2 (scalar)**: + Scalar field value validation (excludes vectors)  
- **Level 3 (full)**: + Vector field validation (complete data quality assurance)

**🧪 Query Correctness Testing:**
- ✅ **1000-sample exact queries** - Verifies data retrieval functionality (95% success threshold)
- ✅ **1000-sample vector searches** - Validates similarity search accuracy (80% success threshold)
- ✅ **Functional validation** - Ensures imported data works correctly, not just exists

**🎯 Smart Field Handling:**
- ✅ **AUTO_ID detection** - Automatically skips auto-generated primary key fields
- ✅ **Vector precision** - Uses numpy.allclose() with 1e-6 tolerance for float comparisons
- ✅ **Data type support** - Specialized validation for vectors, floats, strings, JSON, arrays
- ✅ **Error tolerance** - Allows up to 5% mismatch rate for floating-point precision differences

**📋 Rich Output & Reporting:**
- ✅ Detailed field-by-field verification tables with pass/fail status
- ✅ Comprehensive summary reports with verification statistics  
- ✅ Clear debugging information for failed validations
- ✅ Performance metrics and success rates for query tests

### Bulk Import from S3/MinIO

For very large datasets, use bulk import with pre-uploaded files:

```bash
# First, upload to S3/MinIO
milvus-ingest upload ./products/ s3://bucket/data/ \
    --endpoint-url http://minio:9000 \
    --access-key-id minioadmin \
    --secret-access-key minioadmin

# Then bulk import to Milvus
milvus-ingest to-milvus import product_catalog s3://bucket/data/file1.parquet

# Import multiple files
milvus-ingest to-milvus import product_catalog \
    s3://bucket/data/file1.parquet \
    s3://bucket/data/file2.parquet

# Import all files from directory
milvus-ingest to-milvus import product_catalog ./products/

# Import and wait for completion
milvus-ingest to-milvus import product_catalog ./products/ \
    --wait \
    --timeout 300
```

**Bulk Import Features:**
- ⚡ High-performance import for millions of rows
- 📁 Support for single/multiple files or directories
- ⏳ Asynchronous operation with job tracking
- 🔄 Wait for completion with timeout support
- 📊 Import job status monitoring

### S3/MinIO Upload

Upload generated data to S3-compatible storage:

```bash
# Upload to AWS S3 (using default credentials)
milvus-ingest upload ./output s3://my-bucket/data/

# Upload to MinIO with custom endpoint
milvus-ingest upload ./output s3://my-bucket/data/ \
    --endpoint-url http://localhost:9000 \
    --access-key-id minioadmin \
    --secret-access-key minioadmin

# Upload with environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
milvus-ingest upload ./output s3://my-bucket/data/

# Disable SSL verification for local MinIO
milvus-ingest upload ./output s3://my-bucket/data/ \
    --endpoint-url http://localhost:9000 \
    --no-verify-ssl
```

### Complete Workflow Example

```bash
# 1. Generate large dataset
milvus-ingest generate --builtin ecommerce --rows 5000000 --out products/

# 2. Option A: Direct insert (for smaller datasets)
milvus-ingest to-milvus insert ./products/ \
    --uri http://milvus:19530 \
    --collection-name ecommerce_products

# 2. Option B: Bulk import (for very large datasets)
# First upload to MinIO
milvus-ingest upload ./products/ s3://milvus-data/products/ \
    --endpoint-url http://minio:9000

# Then bulk import
milvus-ingest to-milvus import ecommerce_products \
    s3://milvus-data/products/ \
    --wait

# 3. Verify data integrity and functionality
# Quick verification (row count + query tests)
milvus-ingest to-milvus verify ./products/ --level count

# Business data accuracy (scalar fields + query tests)  
milvus-ingest to-milvus verify ./products/ --level scalar

# Complete validation (all fields + query tests)
milvus-ingest to-milvus verify ./products/ --level full
```

### Import Method Comparison

| Method | Best For | Speed | Max Size | Features |
|--------|----------|-------|----------|----------|
| **Direct Insert** | <1M rows | Moderate | Limited by memory | Automatic collection creation, progress bar |
| **Bulk Import** | >1M rows | Very Fast | 16GB per file | Async operation, job tracking |

**Important Notes:**
- Files must be uploaded to S3/MinIO before bulk import
- Maximum 1024 files per import request
- Each file should not exceed 16GB
- Collection must exist for bulk import (create with direct insert first if needed)
- **Always verify data after import** using the comprehensive verification system

**Verification Recommendations:**
- Use `--level count` for quick post-import validation
- Use `--level scalar` for business-critical data accuracy
- Use `--level full` for complete data quality assurance

## 🛠️ Development

This project uses PDM for dependency management and follows modern Python development practices.

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/your-org/milvus-ingest.git
cd milvus-ingest
pdm install  # Install development dependencies
```

### Development Workflow

```bash
# Code formatting and linting
pdm run ruff format src tests    # Format code
pdm run ruff check src tests     # Check linting
pdm run mypy src                 # Type checking

# Testing
pdm run pytest                           # Run all tests
pdm run pytest --cov=src --cov-report=html  # With coverage
pdm run pytest tests/test_generator.py   # Specific test file

# Combined quality checks
make lint test                   # Run linting and tests together
```

### Project Structure

```
src/milvus_fake_data/
├── cli.py              # Click-based CLI interface
├── generator.py        # Core data generation logic  
├── optimized_writer.py # High-performance vectorized data generation
├── models.py           # Pydantic schema validation models
├── schema_manager.py   # Schema management system
├── builtin_schemas.py  # Built-in schema definitions and metadata
├── rich_display.py     # Rich terminal formatting and UI
├── logging_config.py   # Loguru-based structured logging
├── exceptions.py       # Custom exception classes
├── uploader.py         # S3/MinIO upload functionality
├── milvus_inserter.py  # Direct Milvus insertion
├── milvus_importer.py  # Bulk import from S3/MinIO
└── schemas/            # Built-in schema JSON files (12 schemas)
    ├── simple.json
    ├── ecommerce.json
    ├── documents.json
    ├── images.json
    ├── users.json
    ├── videos.json
    ├── news.json
    ├── audio_transcripts.json
    ├── ai_conversations.json
    ├── face_recognition.json
    ├── ecommerce_partitioned.json
    └── cardinality_demo.json
```

## 📊 Performance Benchmarks

The high-performance engine delivers exceptional speed for large-scale data generation:

| Dataset Size | Time | Throughput | Memory Usage | File Output |
|-------------|------|------------|--------------|-------------|
| 100K rows | ~13s | 7,500 rows/sec | <1GB | Single file |
| 1M rows | ~87s | 11,500 rows/sec | <2GB | Single file |
| 2.5M rows | ~217s | 11,500 rows/sec | <3GB | 5 files (auto-partitioned) |
| 10M rows | ~870s | 11,500 rows/sec | <4GB | 10 files (auto-partitioned) |

**Key Performance Features:**
- **Vectorized Operations**: NumPy-based batch processing for maximum CPU efficiency
- **Smart Memory Management**: Streaming generation prevents memory exhaustion
- **Automatic File Partitioning**: Files split at 256MB/1M rows for optimal handling
- **Optimized I/O**: Direct PyArrow integration with Snappy compression
- **Parallel Processing**: Multi-core utilization for vector generation and normalization

**Recommended Settings for Large Datasets:**
- Use `--format parquet` for fastest I/O (default)
- Batch size 50K-100K rows for optimal memory/speed balance
- Enable automatic file partitioning for datasets >1M rows

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with tests
4. **Ensure** quality checks pass: `make lint test`
5. **Commit** changes: `git commit -m 'Add amazing feature'`
6. **Push** to branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### Contribution Guidelines

- Add tests for new functionality
- Update documentation for API changes
- Follow existing code style (ruff + mypy)
- Include helpful error messages for user-facing features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the [Milvus](https://milvus.io/) vector database ecosystem
- Optimized with [NumPy](https://numpy.org/) vectorized operations for maximum performance
- Uses [PyArrow](https://arrow.apache.org/docs/python/) for efficient Parquet I/O
- Powered by [Pandas](https://pandas.pydata.org/) and [Faker](https://faker.readthedocs.io/) for realistic data generatio
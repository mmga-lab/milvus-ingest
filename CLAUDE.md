# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
pdm install        # Install with development dependencies
pdm install --prod # Install production dependencies only
```

### Code Quality & Testing
```bash
# Formatting & Linting (always run before commits)
pdm run ruff format src tests   # Auto-format code
pdm run ruff check src tests     # Lint checks
pdm run mypy src                 # Type checking

# Testing
pdm run pytest                                    # Run all tests
pdm run pytest --cov=src --cov-report=html       # With coverage report
pdm run pytest -v -s                             # Verbose with output
pdm run pytest -m "not slow"                     # Skip integration tests
pdm run pytest tests/test_cli.py::test_specific  # Run specific test

# Makefile shortcuts (recommended)
make lint          # Run ruff format + check + mypy
make test          # Run pytest
make check         # Run lint + test together (CI equivalent)
make build         # Build distribution package
make security      # Run security checks (safety, bandit, audit)
```

### Test Environment
```bash
# Start local test stack
cd deploy/
docker-compose up -d

# Environment variables
export MILVUS_URI=http://127.0.0.1:19530
export MINIO_HOST=127.0.0.1
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=a-bucket
```

## CLI Usage Examples

### Data Generation Patterns
```bash
# Quick testing with preview
milvus-ingest generate --builtin simple --total-rows 1000 --preview

# Large-scale generation (optimized patterns)
milvus-ingest generate --builtin ecommerce --total-rows 5000000 --out products/
milvus-ingest generate --schema custom.json --file-size 512MB --file-count 10

# Performance tuning
milvus-ingest generate --builtin users --total-rows 2000000 --workers 8 --batch-size 100000

# Reproducible datasets
milvus-ingest generate --builtin news --total-rows 1000000 --seed 42
```

### Schema Workflow
```bash
# Explore available schemas
milvus-ingest schema list
milvus-ingest schema show ecommerce

# Custom schema management
milvus-ingest schema add my_products schema.json
milvus-ingest schema remove my_products

# Schema validation
milvus-ingest generate --schema schema.json --validate-only
```

### Upload Methods
```bash
# AWS S3 (uses AWS CLI, auto-installs if missing)
milvus-ingest upload ./output s3://bucket/data/

# MinIO with mc CLI (recommended for MinIO)
milvus-ingest upload ./output s3://bucket/data/ --endpoint-url http://minio:9000 --use-mc

# boto3 fallback (legacy)
milvus-ingest upload ./output s3://bucket/data/ --use-boto3
```

### Milvus Integration
```bash
# Direct insert (< 1M rows)
milvus-ingest to-milvus insert ./output --drop-if-exists
milvus-ingest to-milvus insert ./output --use-autoindex --collection-name products

# Bulk import workflow (> 1M rows)
# 1. Upload first
milvus-ingest upload ./output s3://bucket/data/ --use-mc
# 2. Import from S3
milvus-ingest to-milvus import products s3://bucket/data/ --wait

# Comprehensive verification (always verify after import)
milvus-ingest to-milvus verify ./output --level count    # Fast: count + queries
milvus-ingest to-milvus verify ./output --level scalar   # Business data accuracy
milvus-ingest to-milvus verify ./output --level full     # Complete validation
```


## Architecture Overview

### Core Components
- **`optimized_writer.py`** - Vectorized data generation engine using NumPy/BLAS
- **`cli.py`** - Click-based command interface with grouped commands
- **`generator.py`** - Legacy data generation (use optimized_writer for >100K rows)
- **`schema_manager.py`** - Built-in + custom schema management system
- **`milvus_inserter.py`** - Direct insertion via pymilvus (small datasets)
- **`milvus_importer.py`** - Bulk import via S3 (large datasets)
- **`verifier.py`** - 3-level verification with query/search testing
- **`uploader.py`** - Multi-method S3 upload (AWS CLI/mc/boto3)
- **`builtin_schemas.py`** - 15+ production-ready schemas

### Data Flow Patterns
1. **Small datasets (<1M rows)**: Generate → Direct Insert → Verify
2. **Large datasets (>1M rows)**: Generate → Upload to S3 → Bulk Import → Verify
3. **Schema development**: Validate → Generate sample → Test → Scale up

### Performance Architecture
- **Vectorized operations** using NumPy with BLAS optimization
- **Streaming generation** prevents memory exhaustion on large datasets
- **Automatic file partitioning** at 256MB/1M rows for optimal I/O
- **Multi-process generation** with configurable worker pools
- **Smart batching** with 50K default batch size

## Development Guidelines

### Performance Optimization
- **Always use `optimized_writer.py`** for datasets >100K rows (10x faster than generator.py)
- **File size defaults**: 256MB chunks, 1M rows/file for optimal Milvus import
- **Memory management**: Use `--batch-size 50000` default, increase for more RAM
- **Vectorized operations**: NumPy arrays are pre-allocated for maximum efficiency
- **Multi-processing**: Default to CPU count, tune with `--workers N`

### Testing Environments
```bash
# Local development stack
cd deploy/ && docker-compose up -d
export MILVUS_URI=http://127.0.0.1:19530
export MINIO_HOST=127.0.0.1
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=a-bucket

# Remote test environment
export MILVUS_URI=http://10.104.17.43:19530
export MINIO_HOST=10.104.17.44
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=milvus-bucket
```

### Schema Development
- Built-in schemas in `src/milvus_ingest/schemas/` are production-tested
- Custom schemas managed via `schema_manager.py` with validation
- Use `--validate-only` to test schema structure without generation
- Vector dimensions limited to 1-32768, consider memory impact

### Code Quality Requirements
- **Always run `make check`** before commits (runs lint + test)
- **Type hints required** - enforced by mypy with strict settings
- **Test coverage** - aim for >80% on new functionality
- **Performance tests** in `bench/` directory for optimization work
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

### Data Generation
```bash
# Quick testing with preview (shows first 5 rows)
milvus-ingest generate --builtin product_catalog --total-rows 1000 --preview

# Large-scale generation with optimized vectorized operations
milvus-ingest generate --builtin ecommerce_search --total-rows 5000000 --out products/
milvus-ingest generate --schema custom.json --file-size 512MB --file-count 10

# Performance tuning with batch size and workers
milvus-ingest generate --builtin news_articles --total-rows 2000000 --workers 8 --batch-size 100000

# Reproducible datasets with seed
milvus-ingest generate --builtin document_search --total-rows 1000000 --seed 42

# Generate with file partitioning
milvus-ingest generate --builtin multimedia_content --total-rows 10000000 --file-size 256MB

# Validate schema without generating data
milvus-ingest generate --schema schema.json --validate-only
```

### Schema Management
```bash
# List all available schemas (built-in + custom)
milvus-ingest schema list

# Show detailed schema information
milvus-ingest schema show product_catalog
milvus-ingest schema show ecommerce_search

# Add custom schema to library
milvus-ingest schema add my_products schema.json

# Remove custom schema
milvus-ingest schema remove my_products

# Get schema format help and examples
milvus-ingest schema help
```

### Upload to S3/MinIO
```bash
# Upload to AWS S3 (uses AWS CLI by default)
milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/

# Upload to MinIO using mc CLI (recommended for MinIO)
milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ \
    --endpoint-url http://localhost:9000 --use-mc

# Upload with explicit credentials
milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ \
    --access-key-id mykey --secret-access-key mysecret --use-mc

# Upload with boto3 (legacy fallback)
milvus-ingest upload --local-path ./output --s3-path s3://my-bucket/data/ --use-boto3
```

### Milvus Operations
```bash
# Direct insert (recommended for < 1M rows)
milvus-ingest to-milvus insert ./output --drop-if-exists
milvus-ingest to-milvus insert ./output --use-autoindex --collection-name products
milvus-ingest to-milvus insert ./output --uri http://192.168.1.100:19530 --token your_token

# Combined upload + bulk import (recommended for > 1M rows)
milvus-ingest to-milvus import --local-path ./output --s3-path data/ \
    --bucket my-bucket --endpoint-url http://minio:9000 --use-mc --wait

# With custom collection name and credentials
milvus-ingest to-milvus import --collection-name my_collection \
    --local-path ./output --s3-path data/ --bucket my-bucket \
    --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret \
    --use-mc --wait --output-import-info import_info.json

# Data verification (always verify after import)
milvus-ingest to-milvus verify ./output --level count    # Fast: row count + queries
milvus-ingest to-milvus verify ./output --level scalar   # Medium: scalar fields validation
milvus-ingest to-milvus verify ./output --level full     # Complete: all fields including vectors
```

### Utility Commands
```bash
# Clean up generated output files
milvus-ingest clean --yes

# Show help for any command
milvus-ingest --help
milvus-ingest generate --help
milvus-ingest schema --help
milvus-ingest to-milvus --help

# Enable verbose logging
milvus-ingest -v generate --builtin product_catalog --total-rows 1000
```

### Report Generation
```bash
# Generate GLM-powered analysis report (requires API key)
export GLM_API_KEY=your-api-key
milvus-ingest report generate --job-ids job1 job2 --format analysis --output analysis.md

# Export raw data without analysis
milvus-ingest report generate --job-ids job1 --format raw --output raw_data.json

# Generate report for specific time range
milvus-ingest report generate --start-time 2024-01-01T10:00:00 --end-time 2024-01-01T11:00:00 \
    --format analysis --output report.md

# Generate report with custom data sources
milvus-ingest report generate --job-ids abc123 \
    --loki-url http://loki:3100 --prometheus-url http://prometheus:9090 \
    --format analysis --output import_analysis.md

# Generate report with additional metadata
milvus-ingest report generate --job-ids job1 --collection-name products \
    --test-scenario "Large Parquet Import" --notes "Testing with 10M rows" \
    --import-info-file import_info.json --format analysis --output detailed_report.md

# Specify custom GLM model
milvus-ingest report generate --job-ids job1 --glm-model glm-4.5-flash \
    --format analysis --output analysis.md
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
- **`report/`** - Performance analysis and report generation module
  - **`report_generator.py`** - Core report orchestrator with multi-format output
  - **`llm_generator.py`** - Report analyzer supporting raw data export and GLM-powered analysis
  - **`glm_analyzer.py`** - GLM API client for intelligent performance analysis
  - **`loki_collector.py`** - Loki log data collection and parsing
  - **`prometheus_collector.py`** - Prometheus metrics collection and aggregation
  - **`models.py`** - Pydantic data models for reports and configurations
  - **`report_templates.py`** - Report templates and GLM prompts
  - **`templates/`** - Jinja2 HTML templates with Chart.js visualizations

### Data Flow Patterns
1. **Small datasets (<1M rows)**: Generate → Direct Insert → Verify → Report
2. **Large datasets (>1M rows)**: Generate → Upload to S3 → Bulk Import → Verify → Report
3. **Schema development**: Validate → Generate sample → Test → Scale up
4. **Performance analysis**: Import → Monitor (Loki/Prometheus) → Generate Reports → Optimize
5. **GLM-powered analysis**: Collect data → Summarize metrics → GLM analysis → Markdown report

### Performance Architecture
- **Vectorized operations** using NumPy with BLAS optimization
- **Streaming generation** prevents memory exhaustion on large datasets
- **Automatic file partitioning** at 256MB/1M rows for optimal I/O
- **Multi-process generation** with configurable worker pools
- **Smart batching** with 50K default batch size

## Environment Variables

### API Keys
```bash
# GLM API key for intelligent report analysis
export GLM_API_KEY=your-glm-api-key

# AWS credentials for S3 upload
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-west-2
```

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

# Remote test environment (stable cluster)
export MILVUS_URI=http://10.104.17.43:19530
export MINIO_HOST=10.104.17.44
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=milvus-bucket

# Alternative remote test environment
export MILVUS_URI=http://10.104.23.145:19530
export MINIO_HOST=10.104.19.3
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

### Monitoring Infrastructure
- **Loki** - Log aggregation for import job analysis (default: http://10.100.36.154:80)
- **Prometheus** - Metrics collection for performance monitoring (default: http://10.100.36.157:9090)
- **GLM API** - Intelligent analysis via GLM-4.5-flash model (requires API key)
- **Report formats** - HTML (with charts), JSON (raw data), CSV (tabular), Markdown (GLM analysis)

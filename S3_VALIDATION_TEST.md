# S3 Minimal Validation Testing Guide

## Overview
This document describes how to test the new S3 minimal validation feature that has been implemented.

## What Has Changed

### 1. New S3MinimalValidator Class
- Located in `src/milvus_ingest/s3_minimal_validator.py`
- Performs minimal validation similar to local MinimalValidator:
  - Checks file accessibility in S3
  - Verifies row count without downloading entire files
  - Compares with meta.json expectations

### 2. Updated S3Uploader
- Modified `upload_directory` method in `src/milvus_ingest/uploader.py`
- Now uses S3MinimalValidator instead of complex validation
- Removed old validation methods that downloaded and sampled data

## How It Works

### For Parquet Files
- Reads only the file footer (last 8KB) to get metadata
- Extracts row count from metadata without downloading entire file
- Much faster than previous approach

### For JSON Files
- Uses streaming to count objects
- Falls back to downloading for small files
- Avoids loading large files into memory

## Testing Instructions

### Prerequisites
```bash
# Install development environment
pdm install

# Start local MinIO/Milvus for testing
cd deploy/
docker-compose up -d
```

### Test Case 1: Upload with Validation
```bash
# Generate test data
milvus-ingest generate --builtin quickstart --total-rows 10000 --out test_data

# Upload to MinIO with validation
milvus-ingest upload --local-path ./test_data --s3-path s3://test-bucket/upload-test/ \
  --endpoint-url http://localhost:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin
```

Expected output:
- Files upload successfully
- Minimal validation runs automatically
- Shows validation summary table
- Reports "S3 Upload Validation Passed"

### Test Case 2: Large File Upload
```bash
# Generate larger dataset
milvus-ingest generate --builtin ecommerce --total-rows 1000000 --out large_data

# Upload large files
milvus-ingest upload --local-path ./large_data --s3-path s3://test-bucket/large-test/ \
  --endpoint-url http://localhost:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin
```

Expected behavior:
- Validation should be fast (only reads metadata)
- No memory issues with large files
- Row count verification works correctly

### Test Case 3: Bulk Import with Validation
```bash
# Use to-milvus import command
milvus-ingest to-milvus import \
  --local-path ./test_data/ \
  --s3-path data/ \
  --bucket test-bucket \
  --endpoint-url http://localhost:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin
```

Expected output:
- Upload step includes minimal validation
- Import to Milvus proceeds if validation passes

## Validation Details

The validation checks:
1. **File Existence**: Each file listed in meta.json exists in S3
2. **File Accessibility**: Can read file metadata from S3
3. **Row Count**: Actual rows match expected count from meta.json
4. **Format Support**: Handles both Parquet and JSON formats

## Performance Comparison

### Before (Old Validation)
- Downloaded entire files for small files (<100MB)
- Downloaded 10MB chunks for large files
- Read and sampled actual data
- Slow and memory-intensive

### After (Minimal Validation)
- Reads only file metadata
- For Parquet: Reads last 8KB (footer)
- For JSON: Streams and counts objects
- Fast and memory-efficient

## Troubleshooting

### Common Issues

1. **"File not found in S3" error**
   - Check S3 bucket and prefix are correct
   - Ensure files were uploaded successfully

2. **"Row count mismatch" error**
   - Verify data generation completed successfully
   - Check meta.json has correct totals

3. **Connection errors**
   - Verify MinIO/S3 endpoint is accessible
   - Check credentials are correct

## Implementation Notes

- The validation is automatically run after successful uploads
- It's designed to be fast and lightweight
- Focuses on the same checks as local MinimalValidator
- Removes complex data sampling and integrity checks for simplicity
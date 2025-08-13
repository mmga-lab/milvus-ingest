# Jenkins Import Test Configuration

This directory contains Jenkins pipeline configurations for comprehensive Milvus import testing. The pipelines are designed to run within the milvus-ingest repository workspace and test various import scenarios including large-scale data generation, bulk import, and verification.

## Overview

The Jenkins test suite provides:
- **Comprehensive Schema Testing**: 10 built-in schemas covering all Milvus field types and advanced features
- **Scale Testing**: Support for both large files (10×10GB) and small files (500×200MB) scenarios
- **Format Support**: JSON and Parquet format testing with identical functionality
- **Storage Validation**: Tests both Storage V1 and V2 architectures
- **Multi-tenant Testing**: Up to 1024 partitions and 16 VChannels for distributed scenarios
- **Automated Verification**: 3-level verification system with query correctness testing

## Files Structure

```
jenkins/
├── milvus_import_batch_test.groovy      # Batch test orchestrator (triggers multiple scenarios)
├── milvus_import_stable_test.groovy     # Individual test executor (single scenario)
├── pods/
│   └── import-test-client.yaml          # Kubernetes pod configuration
├── values/
│   ├── cluster-storagev1.yaml           # Cluster mode + Storage V1 configuration
│   └── cluster-storagev2.yaml           # Cluster mode + Storage V2 configuration
└── README.md                            # This documentation
```

## Pipeline Parameters

### milvus_import_stable_test.groovy

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `image_repository` | Milvus Docker image repository | `harbor.milvus.io/milvus/milvus` | - |
| `image_tag` | Milvus Docker image tag | `master-latest` | - |
| `create_index` | Whether to create index | `not_create` | `not_create`, `create_index` |
| `querynode_nums` | Number of QueryNodes | `3` | - |
| `datanode_nums` | Number of DataNodes | `3` | - |
| `indexnode_nums` | Number of IndexNodes | `3` | - |
| `proxy_nums` | Number of Proxy nodes | `1` | - |
| `keep_env` | Keep environment after test | `false` | `true`, `false` |
| `schema_type` | Built-in schema for comprehensive testing | `text_search_advanced` | See [Built-in Schemas](#built-in-schemas) |
| `file_count` | Number of files to generate | `10` | - |
| `file_size` | Size per file | `10GB` | e.g., `10GB`, `200MB` |
| `file_format` | File format | `parquet` | `parquet`, `json` |
| `storage_version` | Storage version | `V1` | `V1`, `V2` |
| `partition_count` | Number of partitions | `1024` | - |
| `shard_count` | Number of VChannels | `16` | - |

## Built-in Schemas

The test supports these comprehensive built-in schemas for different testing scenarios:

### Advanced Testing Schemas (Primary Focus)
- `text_search_advanced` - **17 fields**, BM25 functions, all data types, **768d vectors**
- `full_text_search` - **11 fields**, BM25 + semantic search, nullable fields, **768d vectors**
- `default_values` - **9 fields**, default_value parameters, missing data handling, **768d vectors**
- `dynamic_fields` - **4 fields**, dynamic field capabilities, schema evolution, **384d vectors**

### Domain-Specific Schemas (Additional Coverage)
- `ecommerce` - **12 fields**, product catalog, multiple embeddings, **768d+512d vectors**
- `documents` - **12 fields**, document search, semantic capabilities, **1536d vectors**
- `images` - **14 fields**, image gallery, visual similarity, **2048d+512d vectors**
- `users` - **15 fields**, user profiles, behavioral embeddings, **256d vectors**
- `videos` - **18 fields**, video library, multimodal embeddings, **512d+1024d vectors**
- `news` - **19 fields**, news articles, sentiment analysis, **384d+768d vectors**

## Test Scenarios Covered

### Large File Import Testing
- **Configuration**: 10 files × 10GB each
- **Purpose**: Test large-scale data import performance
- **Formats**: Both JSON and Parquet

### Small File Import Testing
- **Configuration**: 500 files × 200MB each
- **Purpose**: Test many-file import scenarios
- **Formats**: Both JSON and Parquet

### Storage Version Testing
- **V1**: Traditional storage configuration
- **V2**: New storage architecture with optimizations

### Multi-tenant Testing
- **Partitions**: Up to 1024 partitions for tenant isolation
- **VChannels**: Up to 16 VChannels for distributed processing

## Message Queue Configuration

All cluster values files are pre-configured with:
- **Kafka**: Enabled (reliable message queue for import workloads)
- **Pulsar/PulsarV3**: Disabled
- **Cluster mode**: Only cluster deployment is supported (no standalone mode)
- **Resource allocation**: Optimized for large-scale import testing

The import process is MQ-agnostic, so Kafka is used as the default reliable message queue.

## Quick Start

### Prerequisites
- Jenkins environment with Kubernetes plugin
- Access to Kubernetes cluster in `chaos-testing` namespace
- Helm installed for Milvus deployment
- NFS storage available for data persistence

### Running Tests

The simplest way to run tests is using the batch test orchestrator which handles multiple scenarios automatically.

## Usage Examples

### Run Single Test
```bash
# Trigger individual test with specific parameters
jenkins> build 'milvus_import_stable_test' with parameters:
  - schema_type: 'text_search_advanced'
  - file_count: '10'
  - file_size: '10GB'
  - file_format: 'parquet'
  - storage_version: 'V1'
```

### Run Batch Tests
```bash
# Trigger comprehensive test matrix
jenkins> build 'milvus_import_batch_test' with parameters:
  - run_large_files: true
  - run_small_files: true
  - test_json: true
  - test_parquet: true
  - test_storage_v1: true
  - test_storage_v2: true
```

## Test Flow

The test pipeline executes the following stages:

1. **Install Dependencies**: Install PDM and yq tools
2. **Prepare Values**: Select appropriate Helm values file based on storage version (cluster mode only)
3. **Deploy Milvus**: Deploy Milvus cluster using Helm with configured parameters
4. **Wait for Stability**: Allow Milvus cluster to fully initialize
5. **Install milvus-ingest**: Install CLI tool from current workspace using PDM
6. **Generate Data**: Generate test data using specified schema and parameters
7. **Import Data**: Upload to MinIO and bulk import to Milvus with timeout protection (80 minutes)
8. **Verify Data**: Comprehensive verification with query/search correctness testing
9. **Cleanup**: Archive logs and cleanup resources (unless keep_env=true)

## Output Artifacts

- `artifacts-{release-name}-test-data.tar.gz` - Generated test data
- `artifacts-{release-name}-server-logs.tar.gz` - Milvus server logs
- Test summary with performance metrics

## Technical Notes

### Resource Configuration
- **Pod resources**: 64Gi memory limit, 16 CPU limit for large-scale testing
- **NFS storage**: Mounted at `/root/milvus_ingest_data` for data persistence
- **Milvus cluster**: Optimized resource allocation for QueryNode, DataNode, IndexNode

### Pipeline Features
- **Workspace execution**: Runs directly in milvus-ingest repository (no external clone)
- **Timeout protection**: 80-minute timeout for import operations
- **Comprehensive verification**: 3-level verification system (count, scalar, full)
- **Query correctness**: Automated testing of 1000 sample queries and vector searches
- **Automatic cleanup**: Resources cleaned unless `keep_env=true` is specified

### Supported Test Scenarios
- **File sizes**: 10×10GB (large files) and 500×200MB (small files)
- **Formats**: JSON and Parquet with identical schema support
- **Storage versions**: V1 (traditional) and V2 (optimized) architectures
- **Schema variety**: 10 built-in schemas covering all Milvus field types and features
- **Scale testing**: Up to 1024 partitions and 16 VChannels for multi-tenant scenarios

### Error Handling
- **Stage timeouts**: Each stage has appropriate timeout limits
- **Retry mechanisms**: Built-in retry for transient failures
- **Log collection**: Comprehensive log archival for debugging
- **Resource cleanup**: Automatic cleanup prevents resource leaks

// Enhanced pipeline for Milvus import stability testing with comprehensive configurations
// Supports multiple schemas, file formats, storage versions, upload methods, and large-scale testing

pipeline {
    options {
        timestamps()
        timeout(time: 1000, unit: 'MINUTES')
    }
    agent {
        kubernetes {
            cloud '4am'
            defaultContainer 'main'
            yamlFile 'jenkins/pods/import-test-client.yaml'
            customWorkspace '/home/jenkins/agent/workspace'
            idleMinutes 5
        }
    }
    parameters {
        booleanParam(
            description: 'Use existing Milvus instance instead of deploying a new one',
            name: 'use_existing_instance',
            defaultValue: false
        )
        string(
            description: 'Existing Milvus release name (when use_existing_instance is true)',
            name: 'existing_release_name',
            defaultValue: 'long-run-data-verify'
        )
        string(
            description: 'Existing Milvus namespace (when use_existing_instance is true)',
            name: 'existing_namespace',
            defaultValue: 'qa-tools'
        )
        string(
            description: 'MinIO Access Key (required if use_existing_instance is true)',
            name: 'minio_access_key',
            defaultValue: 'minioadmin'
        )
        string(
            description: 'MinIO Secret Key (required if use_existing_instance is true)',
            name: 'minio_secret_key',
            defaultValue: 'minioadmin'
        )
        string(
            description: 'MinIO Bucket Name',
            name: 'minio_bucket',
            defaultValue: 'milvus-bucket'
        )
        string(
            description: 'Image Repository',
            name: 'image_repository',
            defaultValue: 'harbor.milvus.io/milvus/milvus'
        )
        string(
            description: 'Image Tag',
            name: 'image_tag',
            defaultValue: 'master-latest'
        )
        string(
            description: 'QueryNode Nums',
            name: 'querynode_nums',
            defaultValue: '1'
        )
        string(
            description: 'DataNode Nums',
            name: 'datanode_nums',
            defaultValue: '1'
        )
        string(
            description: 'Proxy Nums',
            name: 'proxy_nums',
            defaultValue: '1'
        )
        booleanParam(
            description: 'Keep Environment',
            name: 'keep_env',
            defaultValue: false
        )
        choice(
            description: '''Built-in schema for comprehensive import testing. Available schemas:

CURRENT AVAILABLE SCHEMAS:
• product_catalog - Simple product catalog with auto_id (4 fields, 128d vectors)
• ecommerce_search - E-commerce with nullable fields (5 fields, 256d vectors)
• news_articles - News with dynamic fields (4 fields, 768d vectors)
• document_search - Document search with sparse vectors + BM25 (5 fields, 768d vectors)
• multi_tenant_data - Multi-tenant with partitioning (5 fields, 256d vectors)
• multimedia_content - Multiple vector types + nullable fields (7 fields, 256d+384d+128d vectors)

Select based on specific testing requirements (BM25, dynamic fields, multi-vector, partitioning, etc.)''',
            name: 'schema_type',
            choices: [
                'product_catalog',     // Simple product catalog with auto_id (4 fields, 128d)
                'ecommerce_search',    // E-commerce with nullable fields (5 fields, 256d)
                'news_articles',       // News with dynamic fields (4 fields, 768d)
                'document_search',     // Document search with sparse vectors + BM25 (5 fields, 768d)
                'multi_tenant_data',   // Multi-tenant with partitioning (5 fields, 256d)
                'multimedia_content'   // Multiple vector types + nullable fields (7 fields, 256d+384d+128d)
            ]
        )
        string(
            description: 'File Count',
            name: 'file_count',
            defaultValue: '200'
        )
        string(
            description: 'File Size (e.g., 1GB, 100MB)',
            name: 'file_size',
            defaultValue: '100MB'
        )
        choice(
            description: 'File Format',
            name: 'file_format',
            choices: ['parquet', 'json']
        )
        choice(
            description: 'Storage Version',
            name: 'storage_version',
            choices: ['V2', 'V1']
        )
        string(
            description: 'Partition Count',
            name: 'partition_count',
            defaultValue: '16'
        )
        string(
            description: 'Shard Count (VChannels)',
            name: 'shard_count',
            defaultValue: '2'
        )
        choice(
            description: '''Upload Method Selection:

• mc_cli - Use MinIO Client (mc) CLI (Recommended for MinIO)
  - Native MinIO protocol support
  - Optimal performance for MinIO servers  
  - Automatic installation and configuration
  - Best choice for MinIO deployments

• aws_cli - Use AWS CLI (Default, Recommended for AWS S3)
  - Reliable for large files with multipart upload
  - Battle-tested upload reliability
  - Good for AWS S3 and general S3-compatible storage
  - Automatic retry and error handling

• boto3 - Use boto3 Python library (Legacy)
  - Python-based upload method
  - Fallback for restricted environments
  - Use when CLI tools are not available''',
            name: 'upload_method',
            choices: [
                'mc_cli',    // MinIO Client (mc) CLI (Recommended for MinIO)
                'aws_cli',    // AWS CLI (Recommended for AWS S3)
                'boto3'       // boto3 library (legacy fallback)
            ]
        )
        booleanParam(
            description: 'Generate performance report from Loki and Prometheus data',
            name: 'generate_report',
            defaultValue: true
        )
        string(
            description: 'Loki URL for log collection (for performance reports)',
            name: 'loki_url',
            defaultValue: 'http://10.100.36.154:80'
        )
        string(
            description: 'Prometheus URL for metrics collection (for performance reports)',
            name: 'prometheus_url',
            defaultValue: 'http://10.100.36.157:9090'
        )
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        RELEASE_NAME = "import-stable-test-${env.BUILD_ID}"
        NAMESPACE = "chaos-testing"
        DATA_PATH = "/root/milvus_ingest_data/${env.BUILD_ID}"
        IMPORT_INFO = "/tmp/import_info_${env.BUILD_ID}.json"
        REPORT_DIR = "/tmp/reports_${env.BUILD_ID}"
    }

    stages {
        stage('Validate Parameters') {
            when {
                expression { params.use_existing_instance == true }
            }
            steps {
                container('main') {
                    script {
                        if (!params.existing_release_name || params.existing_release_name.trim().isEmpty()) {
                            error("Existing release name is required when using existing instance")
                        }
                        if (!params.existing_namespace || params.existing_namespace.trim().isEmpty()) {
                            error("Existing namespace is required when using existing instance")
                        }
                        
                        echo "Using existing Milvus instance:"
                        echo "  Release Name: ${params.existing_release_name}"
                        echo "  Namespace: ${params.existing_namespace}"
                        echo "  MinIO Bucket: ${params.minio_bucket}"
                        
                        // Verify the existing instance is accessible
                        echo "Verifying existing Milvus instance..."
                        sh """
                        # Check if the services exist in the specified namespace
                        kubectl get svc/${params.existing_release_name}-milvus -n ${params.existing_namespace} --no-headers || {
                            echo "ERROR: Milvus service '${params.existing_release_name}-milvus' not found in namespace '${params.existing_namespace}'"
                            exit 1
                        }
                        kubectl get svc/${params.existing_release_name}-minio -n ${params.existing_namespace} --no-headers || {
                            echo "ERROR: MinIO service '${params.existing_release_name}-minio' not found in namespace '${params.existing_namespace}'"
                            exit 1
                        }
                        echo "✅ Both Milvus and MinIO services found in the specified namespace"
                        """
                    }
                }
            }
        }
        
        stage('Install Dependencies') {
            steps {
                container('main') {
                    script {
                        sh "pip install pdm -i https://pypi.tuna.tsinghua.edu.cn/simple"
                        sh "pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple"
                    }
                }
            }
        }
        stage('Install milvus-ingest') {
            steps {
                container('main') {
                    script {
                        sh """
                        # Set UV cache directory to use NFS mounted path
                        export UV_CACHE_DIR=/tmp/.uv-cache
                        
                        # Install PDM if not available
                        which pdm || pip install pdm
                        
                        # Use Python 3.10 specifically
                        pdm use python3.10
                        pdm config use_uv true
                        
                        # Install milvus-ingest from current workspace
                        # Fix lockfile if needed
                        # pdm lock --update-reuse || true
                        rm -rf pdm.lock
                        pdm install
                        
                        # Verify installation
                        pdm run milvus-ingest --help
                        """
                    }
                }
            }
        }        

        stage('Prepare Milvus Values') {
            when {
                expression { params.use_existing_instance == false }
            }
            steps {
                container('main') {
                    script {
                        sh """
                        # Create working directory for values
                        mkdir -p /tmp/milvus-values
                        
                        # Select appropriate values file based on storage version (cluster mode only)
                        if [ "${params.storage_version}" = "V2" ]; then
                            cp jenkins/values/cluster-storagev2.yaml /tmp/milvus-values/values.yaml
                            echo "Using cluster Storage V2 configuration"
                        else
                            cp jenkins/values/cluster-storagev1.yaml /tmp/milvus-values/values.yaml
                            echo "Using cluster Storage V1 configuration"
                        fi
                        
                        # Customize values based on parameters
                        cd /tmp/milvus-values
                        
                        # Update node replicas
                        yq -i '.queryNode.replicas = "${params.querynode_nums}"' values.yaml
                        yq -i '.dataNode.replicas = "${params.datanode_nums}"' values.yaml
                        yq -i '.proxy.replicas = "${params.proxy_nums}"' values.yaml
                        
                        echo "Final values configuration:"
                        cat values.yaml
                        """
                    }
                }
            }
        }

        stage('Deploy Milvus') {
            when {
                expression { params.use_existing_instance == false }
            }
            options {
                timeout(time: 15, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        def image_tag_modified = ''
                        
                        if ("${params.image_tag}" =~ 'latest') {
                            image_tag_modified = sh(returnStdout: true, script: "tagfinder get-tag -t ${params.image_tag}").trim()
                        }
                        else {
                            image_tag_modified = "${params.image_tag}"
                        }
                        
                        sh 'helm repo add milvus https://zilliztech.github.io/milvus-helm'
                        sh 'helm repo update'
                        
                        sh """
                        cd /tmp/milvus-values
                        
                        echo "Deploying Milvus cluster with configuration:"
                        echo "Image Repository: ${params.image_repository}"
                        echo "Image Tag: ${params.image_tag}"
                        echo "Resolved Image Tag: ${image_tag_modified}"
                        echo "Storage Version: ${params.storage_version}"
                        
                        helm install --wait --debug --timeout 600s ${env.RELEASE_NAME} milvus/milvus \\
                            --set image.all.repository=${params.image_repository} \\
                            --set image.all.tag=${image_tag_modified} \\
                            --set metrics.serviceMonitor.enabled=true \\
                            --set quotaAndLimits.enabled=false \\
                            -f values.yaml -n=${env.NAMESPACE}
                        """
                        sh 'cat /tmp/milvus-values/values.yaml'
                        sh "kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl wait --for=condition=Ready pod -l release=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl get pods -o wide|grep ${env.RELEASE_NAME}"
                    }   
                }
            }
        }

        stage('Generate Test Data') {
            options {
                timeout(time: 120, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        def outputPath = "/root/milvus_ingest_data/${env.BUILD_ID}"
                        sh """
                        echo "Generating test data with configuration:"
                        echo "Schema: ${params.schema_type}"
                        echo "File Count: ${params.file_count}"
                        echo "File Size: ${params.file_size}"
                        echo "Format: ${params.file_format}"
                        echo "Partitions: ${params.partition_count}"
                        echo "Shards: ${params.shard_count}"
                        echo "Output Path: ${outputPath}"
                        
                        # Generate data using milvus-ingest CLI from current workspace
                        pdm run milvus-ingest generate \\
                            --builtin ${params.schema_type} \\
                            --file-count ${params.file_count} \\
                            --file-size ${params.file_size} \\
                            --format ${params.file_format} \\
                            --partitions ${params.partition_count} \\
                            --shards ${params.shard_count} \\
                            --out ${outputPath} \\
                            --workers 8 \\
                            --verbose \\
                            --force
                        
                        echo "Data generation completed. Checking output:"
                        ls -lah ${outputPath}/
                        if [ -f "${outputPath}/meta.json" ]; then
                            echo "Meta.json contents:"
                            cat ${outputPath}/meta.json | jq .
                        fi
                        """
                    }
                }
            }
        }
        
        stage('Import to Milvus') {
            options {
                timeout(time: 240, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        def outputPath = "/root/milvus_ingest_data/${env.BUILD_ID}"
                        def milvusUri = ""
                        def minioEndpoint = ""
                        def minioAccessKey = params.minio_access_key
                        def minioSecretKey = params.minio_secret_key
                        def minioBucket = params.minio_bucket
                        
                        if (params.use_existing_instance) {
                            // Get URIs from existing services using release name and namespace
                            def host = sh(returnStdout: true, script: "kubectl get svc/${params.existing_release_name}-milvus -n ${params.existing_namespace} -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            def minioHost = sh(returnStdout: true, script: "kubectl get svc/${params.existing_release_name}-minio -n ${params.existing_namespace} -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            milvusUri = "http://${host}:19530"
                            minioEndpoint = "http://${minioHost}:9000"
                            
                            if (!milvusUri || !minioEndpoint) {
                                error("Could not retrieve service IPs from existing instance")
                            }
                        } else {
                            // Get URIs from deployed services
                            def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            def minioHost = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-minio -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            milvusUri = "http://${host}:19530"
                            minioEndpoint = "http://${minioHost}:9000"
                        }
                        
                        // Determine upload method flags based on user selection
                        def uploadFlags = ""
                        switch(params.upload_method) {
                            case 'mc_cli':
                                uploadFlags = "--use-mc"
                                break
                            case 'boto3':
                                uploadFlags = "--use-boto3"
                                break
                            case 'aws_cli':
                            default:
                                uploadFlags = ""  // AWS CLI is default, no flag needed
                                break
                        }
                        
                        sh """
                        echo "Starting bulk import to Milvus:"
                        echo "Milvus URI: ${milvusUri}"
                        echo "MinIO Endpoint: ${minioEndpoint}"
                        echo "MinIO Bucket: ${minioBucket}"
                        echo "Data Path: ${outputPath}"
                        echo "Upload Method: ${params.upload_method}"
                        echo "Upload Flags: ${uploadFlags}"
                        
                        # Prepare directories for import info and reports
                        mkdir -p ${env.REPORT_DIR}
                        mkdir -p ${env.ARTIFACTS}
                        
                        # Import data to Milvus via MinIO from current workspace
                        # Use the new --output-import-info option to get structured information
                        pdm run milvus-ingest to-milvus import \\
                            --local-path ${outputPath} \\
                            --s3-path test-data/${env.BUILD_ID} \\
                            --bucket ${minioBucket} \\
                            --endpoint-url ${minioEndpoint} \\
                            --access-key-id ${minioAccessKey} \\
                            --secret-access-key ${minioSecretKey} \\
                            --uri ${milvusUri} \\
                            --drop-if-exists \\
                            --wait \\
                            --timeout 4800 \\
                            --output-import-info ${env.IMPORT_INFO} \\
                            ${uploadFlags}
                        
                        # Verify the import info file was created and enhance it with Jenkins metadata
                        if [ -f "${env.IMPORT_INFO}" ]; then
                            echo "✓ Import information captured successfully"
                            
                            # Add Jenkins-specific metadata to the import info
                            jq '. + {
                                "jenkins_job": "${env.JOB_NAME}",
                                "jenkins_build": "${env.BUILD_NUMBER}",
                                "jenkins_build_id": "${env.BUILD_ID}",
                                "jenkins_url": "${env.BUILD_URL}",
                                "test_parameters": {
                                    "schema_type": "${params.schema_type}",
                                    "file_count": "${params.file_count}",
                                    "file_size": "${params.file_size}",
                                    "file_format": "${params.file_format}",
                                    "storage_version": "${params.storage_version}",
                                    "partition_count": "${params.partition_count}",
                                    "shard_count": "${params.shard_count}",
                                    "upload_method": "${params.upload_method}"
                                }
                            }' ${env.IMPORT_INFO} > ${env.IMPORT_INFO}.tmp && mv ${env.IMPORT_INFO}.tmp ${env.IMPORT_INFO}
                            
                            echo "Enhanced import info with Jenkins metadata:"
                            cat ${env.IMPORT_INFO} | jq .
                        else
                            echo "⚠ Warning: Import info file not created, generating fallback"
                            # Create a minimal fallback info file
                            cat > ${env.IMPORT_INFO} << EOF
{
    "status": "unknown",
    "collection_name": "${params.schema_type}",
    "job_ids": ["fallback-${env.BUILD_ID}"],
    "import_start_time": "\$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "import_end_time": "\$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "jenkins_job": "${env.JOB_NAME}",
    "jenkins_build": "${env.BUILD_NUMBER}",
    "jenkins_build_id": "${env.BUILD_ID}",
    "jenkins_url": "${env.BUILD_URL}",
    "test_parameters": {
        "schema_type": "${params.schema_type}",
        "file_count": "${params.file_count}",
        "file_size": "${params.file_size}",
        "file_format": "${params.file_format}",
        "storage_version": "${params.storage_version}",
        "partition_count": "${params.partition_count}",
        "shard_count": "${params.shard_count}",
        "upload_method": "${params.upload_method}"
    }
}
EOF
                        fi
                        
                        echo "Import completed successfully using ${params.upload_method} method"
                        """
                    }
                }
            }
        }
        
        stage('Verify Data') {
            options {
                timeout(time: 60, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        def outputPath = "/root/milvus_ingest_data/${env.BUILD_ID}"
                        def milvusUri = ""
                        
                        if (params.use_existing_instance) {
                            def host = sh(returnStdout: true, script: "kubectl get svc/${params.existing_release_name}-milvus -n ${params.existing_namespace} -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            milvusUri = "http://${host}:19530"
                        } else {
                            def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            milvusUri = "http://${host}:19530"
                        }
                        
                        sh """
                        echo "Starting data verification:"
                        echo "Milvus URI: ${milvusUri}"
                        echo "Data Path: ${outputPath}"
                        
                        # Verify imported data from current workspace
                        pdm run milvus-ingest to-milvus verify \\
                            ${outputPath} \\
                            --uri ${milvusUri} \\
                            --level full
                        
                        echo "Verification completed successfully"
                        """
                    }
                }
            }
        }
        
        stage('Generate Performance Report') {
            when {
                expression { params.generate_report == true }
            }
            options {
                timeout(time: 30, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        sh """
                        echo "Starting performance report generation"
                        echo "Reading import information from ${env.IMPORT_INFO}"
                        
                        if [ ! -f "${env.IMPORT_INFO}" ]; then
                            echo "Warning: Import info file not found. Cannot generate detailed report."
                            exit 0
                        fi
                        
                        # Load import information from structured JSON
                        COLLECTION_NAME=\$(cat ${env.IMPORT_INFO} | jq -r '.collection_name // "${params.schema_type}"')
                        
                        # Handle job_ids as either array or string
                        JOB_IDS_RAW=\$(cat ${env.IMPORT_INFO} | jq -r '.job_ids')
                        if echo "\${JOB_IDS_RAW}" | jq -e 'type == "array"' > /dev/null 2>&1; then
                            # job_ids is an array, join with commas
                            JOB_IDS=\$(echo "\${JOB_IDS_RAW}" | jq -r 'join(",")')
                        else
                            # job_ids is already a string
                            JOB_IDS="\${JOB_IDS_RAW}"
                        fi
                        
                        IMPORT_START_TIME=\$(cat ${env.IMPORT_INFO} | jq -r '.import_start_time')
                        IMPORT_END_TIME=\$(cat ${env.IMPORT_INFO} | jq -r '.import_end_time')
                        IMPORT_STATUS=\$(cat ${env.IMPORT_INFO} | jq -r '.status // "unknown"')
                        
                        echo "Collection name: \${COLLECTION_NAME}"
                        echo "Job IDs: \${JOB_IDS}"
                        echo "Import time range: \${IMPORT_START_TIME} to \${IMPORT_END_TIME}"
                        
                        # Extend time range for report analysis (add 15 minutes before and after)
                        REPORT_START_TIME=\$(date -u -d "\${IMPORT_START_TIME} - 15 minutes" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo \${IMPORT_START_TIME})
                        REPORT_END_TIME=\$(date -u -d "\${IMPORT_END_TIME} + 15 minutes" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo \${IMPORT_END_TIME})
                        
                        echo "Extended report time range: \${REPORT_START_TIME} to \${REPORT_END_TIME}"
                        
                        # Get specific pod pattern and namespace for this deployment
                        if [ "${params.use_existing_instance}" = "true" ]; then
                            POD_PATTERN="${params.existing_release_name}.*"
                            REPORT_NAMESPACE="${params.existing_namespace}"
                            REPORT_RELEASE_NAME="${params.existing_release_name}"
                            echo "Using existing instance for report generation:"
                            echo "  Release Name: ${params.existing_release_name}"
                            echo "  Namespace: ${params.existing_namespace}"
                        else
                            POD_PATTERN="${env.RELEASE_NAME}.*"
                            REPORT_NAMESPACE="${env.NAMESPACE}"
                            REPORT_RELEASE_NAME="${env.RELEASE_NAME}"
                            echo "Using deployed instance for report generation:"
                            echo "  Release Name: ${env.RELEASE_NAME}"
                            echo "  Namespace: ${env.NAMESPACE}"
                        fi
                        
                        echo "Using pod pattern: \${POD_PATTERN} (namespace: \${REPORT_NAMESPACE})"
                        
                        # List actual pods for verification
                        echo "Pods matching pattern:"
                        kubectl get pods -n \${REPORT_NAMESPACE} -l app.kubernetes.io/instance=\${REPORT_RELEASE_NAME} --no-headers -o custom-columns=":metadata.name" || true
                        
                        # Generate CSV performance report
                        CSV_REPORT="${env.REPORT_DIR}/import-performance-summary-${env.BUILD_ID}.csv"
                        
                        echo "Generating CSV performance report with Prometheus metrics..."
                        # Parse job IDs - handle both single ID and comma-separated list
                        if [[ "\${JOB_IDS}" == *","* ]]; then
                            # Multiple job IDs - split and add --job-ids for each
                            JOB_PARAMS=""
                            IFS=',' read -ra JOB_ARRAY <<< "\${JOB_IDS}"
                            for job_id in "\${JOB_ARRAY[@]}"; do
                                JOB_PARAMS="\${JOB_PARAMS} --job-ids \${job_id}"
                            done
                        else
                            # Single job ID
                            JOB_PARAMS="--job-ids \${JOB_IDS}"
                        fi
                        
                        pdm run milvus-ingest report generate \\
                            \${JOB_PARAMS} \\
                            --collection-name "\${COLLECTION_NAME}" \\
                            --start-time "\${REPORT_START_TIME}" \\
                            --end-time "\${REPORT_END_TIME}" \\
                            --output "\${CSV_REPORT}" \\
                            --loki-url "${params.loki_url}" \\
                            --prometheus-url "${params.prometheus_url}" \\
                            --pod-pattern "\${POD_PATTERN}" \\
                            --release-name "\${REPORT_RELEASE_NAME}" \\
                            --milvus-namespace "\${REPORT_NAMESPACE}" \\
                            --test-scenario "${params.schema_type} Import Test (${params.file_count} × ${params.file_size} ${params.file_format})" \\
                            --notes "Jenkins Build ${env.BUILD_NUMBER} - Storage ${params.storage_version}, Upload: ${params.upload_method}" \\
                            --timeout 60 || echo "CSV report generation failed, but continuing..."
                        
                        # Copy reports to Jenkins artifacts directory
                        mkdir -p ${env.ARTIFACTS}/reports
                        cp -f ${env.IMPORT_INFO} ${env.ARTIFACTS}/import_info.json || true
                        cp -f "\${CSV_REPORT}" ${env.ARTIFACTS}/reports/ || true
                        
                        # Generate a summary report for easy consumption
                        cat > ${env.ARTIFACTS}/reports/report_summary.txt << EOF
Performance Report Summary
========================

Build Information:
- Jenkins Job: ${env.JOB_NAME}
- Build Number: ${env.BUILD_NUMBER}  
- Build ID: ${env.BUILD_ID}
- Build URL: ${env.BUILD_URL}

Test Configuration:
- Schema Type: ${params.schema_type}
- Collection Name: \${COLLECTION_NAME}
- File Count: ${params.file_count}
- File Size: ${params.file_size}
- File Format: ${params.file_format}
- Storage Version: ${params.storage_version}
- Partitions: ${params.partition_count}
- Shards: ${params.shard_count}
- Upload Method: ${params.upload_method}

Import Details:
- Job IDs: \${JOB_IDS}
- Import Start: \${IMPORT_START_TIME}
- Import End: \${IMPORT_END_TIME}

Report Files:
- CSV Summary: import-performance-summary-${env.BUILD_ID}.csv
- Import Info: import_info.json

Data Sources:
- Loki URL: ${params.loki_url}
- Prometheus URL: ${params.prometheus_url}

Generated at: \$(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF
                        
                        echo "Performance reports generated successfully!"
                        echo "Report files:"
                        ls -la ${env.ARTIFACTS}/reports/
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Upload logs and cleanup'
            container('main') {
                script {
                    // Collect logs from the appropriate instance
                    def logReleaseName = params.use_existing_instance ? params.existing_release_name : env.RELEASE_NAME
                    def logNamespace = params.use_existing_instance ? params.existing_namespace : env.NAMESPACE
                    
                    echo "Collecting logs from release: ${logReleaseName} in namespace: ${logNamespace}"
                    sh "kubectl get pods -n ${logNamespace} -o wide | grep ${logReleaseName} || true"
                    
                    // Collect logs using kubectl
                    sh """
                    mkdir -p k8s_log/${logReleaseName}
                    kubectl logs -l app.kubernetes.io/instance=${logReleaseName} -n ${logNamespace} --all-containers=true --tail=-1 > k8s_log/${logReleaseName}/milvus-logs.txt || true
                    kubectl describe pods -l app.kubernetes.io/instance=${logReleaseName} -n ${logNamespace} > k8s_log/${logReleaseName}/pod-descriptions.txt || true
                    """

                    // Archive logs
                    sh "tar -zcvf artifacts-${logReleaseName}-server-logs.tar.gz k8s_log/ --remove-files || true"

                    archiveArtifacts artifacts: "artifacts-${logReleaseName}-server-logs.tar.gz", allowEmptyArchive: true
                    
                    // Always archive performance reports if they exist
                    if (fileExists("${env.ARTIFACTS}/reports/")) {
                        echo "Archiving performance reports..."
                        archiveArtifacts artifacts: "_artifacts/**/*", allowEmptyArchive: true
                    } else {
                        echo "No performance reports found to archive"
                    }

                    // Cleanup test data
                    sh "rm -rf ${env.DATA_PATH} || true"

                    if ("${params.keep_env}" == "false" && params.use_existing_instance == false) {
                        sh "helm uninstall ${env.RELEASE_NAME} -n ${env.NAMESPACE} || true"
                    }
                }
            }
        }
        success {
            echo 'Test completed successfully!'
            container('main') {
                script {
                    echo "Test Summary:"
                    echo "Schema: ${params.schema_type}"
                    echo "File Count: ${params.file_count}"
                    echo "File Size: ${params.file_size}"
                    echo "Format: ${params.file_format}"
                    echo "Storage Version: ${params.storage_version}"
                    echo "Upload Method: ${params.upload_method}"
                    echo "Partitions: ${params.partition_count}"
                    echo "Shards: ${params.shard_count}"

                    if ("${params.keep_env}" == "false" && params.use_existing_instance == false) {
                        sh "helm uninstall ${env.RELEASE_NAME} -n ${env.NAMESPACE} || true"
                    }
                }
            }
        }
        unstable {
            echo 'Test completed with some issues'
        }
        failure {
            echo 'Test failed'
        }
        changed {
            echo 'Test results changed from previous run'
        }
    }
}

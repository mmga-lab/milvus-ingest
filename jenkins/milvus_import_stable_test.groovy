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
            description: 'Milvus URI (required if use_existing_instance is true)',
            name: 'milvus_uri',
            defaultValue: 'http://10.255.11.79:19530'
        )
        string(
            description: 'MinIO Endpoint URL (required if use_existing_instance is true)',
            name: 'minio_endpoint',
            defaultValue: 'http://10.255.194.30:9000'
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
            defaultValue: '10'
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
            defaultValue: '1024'
        )
        string(
            description: 'Shard Count (VChannels)',
            name: 'shard_count',
            defaultValue: '16'
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
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        RELEASE_NAME = "import-stable-test-${env.BUILD_ID}"
        NAMESPACE = "chaos-testing"
        DATA_PATH = "/root/milvus_ingest_data/${env.BUILD_ID}"
    }

    stages {
        stage('Validate Parameters') {
            when {
                expression { params.use_existing_instance == true }
            }
            steps {
                container('main') {
                    script {
                        if (!params.milvus_uri || params.milvus_uri.trim().isEmpty()) {
                            error("Milvus URI is required when using existing instance")
                        }
                        if (!params.minio_endpoint || params.minio_endpoint.trim().isEmpty()) {
                            error("MinIO endpoint is required when using existing instance")
                        }
                        
                        echo "Using existing Milvus instance:"
                        echo "  Milvus URI: ${params.milvus_uri}"
                        echo "  MinIO Endpoint: ${params.minio_endpoint}"
                        echo "  MinIO Bucket: ${params.minio_bucket}"
                    }
                }
            }
        }
        
        stage('Install Dependencies') {
            steps {
                container('main') {
                    script {
                        sh "pip install pdm"
                        sh "pip install uv"
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
                            // Use provided URIs
                            milvusUri = params.milvus_uri
                            minioEndpoint = params.minio_endpoint
                            
                            if (!milvusUri || !minioEndpoint) {
                                error("Milvus URI and MinIO endpoint must be provided when using existing instance")
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
                        
                        # Import data to Milvus via MinIO from current workspace
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
                            ${uploadFlags}
                        
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
                            milvusUri = params.milvus_uri
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
    }

    post {
        always {
            echo 'Upload logs and cleanup'
            container('main') {
                script {
                    if (params.use_existing_instance == false) {
                        echo "Get pod status"
                        sh "kubectl get pods -o wide|grep ${env.RELEASE_NAME} || true"
                        
                        // Collect logs using kubectl
                        sh """
                        mkdir -p k8s_log/${env.RELEASE_NAME}
                        kubectl logs -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --all-containers=true --tail=-1 > k8s_log/${env.RELEASE_NAME}/milvus-logs.txt || true
                        kubectl describe pods -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} > k8s_log/${env.RELEASE_NAME}/pod-descriptions.txt || true
                        """

                        // Archive logs
                        sh "tar -zcvf artifacts-${env.RELEASE_NAME}-server-logs.tar.gz k8s_log/ --remove-files || true"

                        archiveArtifacts artifacts: "artifacts-${env.RELEASE_NAME}-server-logs.tar.gz", allowEmptyArchive: true
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

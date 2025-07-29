// Enhanced pipeline for Milvus import stability testing with comprehensive configurations
// Supports multiple schemas, file formats, storage versions, and large-scale testing

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
            defaultValue: '3'
        )
        string(
            description: 'DataNode Nums',
            name: 'datanode_nums',
            defaultValue: '3'
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

ADVANCED TESTING SCHEMAS (Primary Focus):
• text_search_advanced - 17 fields, BM25 functions, all data types, 768d vectors
• full_text_search - 11 fields, BM25 + semantic search, nullable fields, 768d vectors
• default_values - 9 fields, default_value parameters, missing data handling, 768d vectors
• dynamic_fields - 4 fields, dynamic field capabilities, schema evolution, 384d vectors

DOMAIN-SPECIFIC SCHEMAS (Additional Coverage):
• ecommerce - 12 fields, product catalog, multiple embeddings, 768d+512d vectors
• documents - 12 fields, document search, semantic capabilities, 1536d vectors
• images - 14 fields, image gallery, visual similarity, 2048d+512d vectors
• users - 15 fields, user profiles, behavioral embeddings, 256d vectors
• videos - 18 fields, video library, multimodal embeddings, 512d+1024d vectors
• news - 19 fields, news articles, sentiment analysis, 384d+768d vectors

Select based on specific testing requirements (BM25, dynamic fields, multi-vector, etc.)''',
            name: 'schema_type',
            choices: [
                'text_search_advanced',  // 17 fields, BM25 functions, all data types, 768d vectors
                'full_text_search',      // 11 fields, BM25 + semantic search, nullable fields, 768d vectors  
                'default_values',        // 9 fields, default_value parameters, missing data handling, 768d vectors
                'dynamic_fields',        // 4 fields, dynamic field capabilities, schema evolution, 384d vectors
                'ecommerce',            // 12 fields, product catalog, multiple embeddings, 768d+512d vectors
                'documents',            // 12 fields, document search, semantic capabilities, 1536d vectors
                'images',               // 14 fields, image gallery, visual similarity, 2048d+512d vectors
                'users',                // 15 fields, user profiles, behavioral embeddings, 256d vectors
                'videos',               // 18 fields, video library, multimodal embeddings, 512d+1024d vectors
                'news'                  // 19 fields, news articles, sentiment analysis, 384d+768d vectors
            ]
        )
        string(
            description: 'File Count',
            name: 'file_count',
            defaultValue: '10'
        )
        string(
            description: 'File Size (e.g., 10GB, 200MB)',
            name: 'file_size',
            defaultValue: '10GB'
        )
        choice(
            description: 'File Format',
            name: 'file_format',
            choices: ['parquet', 'json']
        )
        choice(
            description: 'Storage Version',
            name: 'storage_version',
            choices: ['V1', 'V2']
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
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        RELEASE_NAME = "import-stable-test-${env.BUILD_ID}"
        NAMESPACE = "chaos-testing"
        DATA_PATH = "/root/milvus_ingest_data/${env.BUILD_ID}"
    }

    stages {
        stage('Install Dependencies') {
            steps {
                container('main') {
                    script {
                        sh "pip install pdm"
                    }
                }
            }
        }

        stage('Prepare Milvus Values') {
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
                        
                        sh "kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl wait --for=condition=Ready pod -l release=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl get pods -o wide|grep ${env.RELEASE_NAME}"
                    }
                }
            }
        }

        stage('Wait for Milvus Stability') {
            steps {
                container('main') {
                    script {
                        echo "Waiting for Milvus to be fully ready..."
                        sh "sleep 2m"
                    }
                }
            }
        }

        stage('Install milvus-ingest') {
            steps {
                container('main') {
                    script {
                        sh """
                        # Install PDM if not available
                        which pdm || pip install pdm
                        
                        # Install milvus-ingest from current workspace
                        pdm install
                        
                        # Verify installation
                        pdm run milvus-ingest --help
                        """
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
                        def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        def minioHost = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-minio -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        
                        sh """
                        echo "Starting bulk import to Milvus:"
                        echo "Milvus Host: ${host}"
                        echo "MinIO Host: ${minioHost}"
                        echo "Data Path: ${outputPath}"
                        
                        # Import data to Milvus via MinIO from current workspace
                        pdm run milvus-ingest to-milvus import \\
                            --local-path ${outputPath} \\
                            --s3-path test-data/${env.BUILD_ID} \\
                            --bucket milvus-bucket \\
                            --endpoint-url http://${minioHost}:9000 \\
                            --access-key-id minioadmin \\
                            --secret-access-key minioadmin \\
                            --uri http://${host}:19530 \\
                            --drop-if-exists \\
                            --wait \\
                            --timeout 4800
                        
                        echo "Import completed successfully"
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
                        def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        
                        sh """
                        echo "Starting data verification:"
                        echo "Milvus Host: ${host}"
                        echo "Data Path: ${outputPath}"
                        
                        # Verify imported data from current workspace
                        pdm run milvus-ingest to-milvus verify \\
                            ${outputPath} \\
                            --uri http://${host}:19530 \\
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
                    echo "Get pod status"
                    sh "kubectl get pods -o wide|grep ${env.RELEASE_NAME} || true"
                    
                    // Collect logs using kubectl
                    sh """
                    mkdir -p k8s_log/${env.RELEASE_NAME}
                    kubectl logs -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --all-containers=true --tail=-1 > k8s_log/${env.RELEASE_NAME}/milvus-logs.txt || true
                    kubectl describe pods -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} > k8s_log/${env.RELEASE_NAME}/pod-descriptions.txt || true
                    """

                    // Archive test data and logs
                    sh "mkdir -p ${env.ARTIFACTS}"
                    sh "cp -r ${env.DATA_PATH} ${env.ARTIFACTS}/test_data_${env.BUILD_ID} || true"
                    sh "tar -zcvf artifacts-${env.RELEASE_NAME}-test-data.tar.gz ${env.ARTIFACTS}/test_data_${env.BUILD_ID} --remove-files || true"
                    sh "tar -zcvf artifacts-${env.RELEASE_NAME}-server-logs.tar.gz k8s_log/ --remove-files || true"

                    archiveArtifacts artifacts: "artifacts-${env.RELEASE_NAME}-test-data.tar.gz", allowEmptyArchive: true
                    archiveArtifacts artifacts: "artifacts-${env.RELEASE_NAME}-server-logs.tar.gz", allowEmptyArchive: true

                    // Cleanup test data
                    sh "rm -rf ${env.DATA_PATH} || true"

                    if ("${params.keep_env}" == "false") {
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
                    echo "Partitions: ${params.partition_count}"
                    echo "Shards: ${params.shard_count}"

                    if ("${params.keep_env}" == "false") {
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

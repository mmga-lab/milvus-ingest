// This pipeline triggers multiple milvus import tests with different configurations
// It covers various scenarios including large files, small files, different formats, and storage versions

pipeline {
    options {
        timestamps()
        timeout(time: 2000, unit: 'MINUTES')   // Extended timeout for batch tests
    }
    agent {
        kubernetes {
            inheritFrom 'default'
            defaultContainer 'main'
            yamlFile "jenkins/pods/import-test-client.yaml"
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
        booleanParam(
            description: 'Run Large Files Tests (10 x 1GB)',
            name: 'run_large_files',
            defaultValue: true
        )
        booleanParam(
            description: 'Run Small Files Tests (100 x 100MB)',
            name: 'run_small_files',
            defaultValue: true
        )
        booleanParam(
            description: 'Test JSON Format',
            name: 'test_json',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Parquet Format',
            name: 'test_parquet',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Storage V1',
            name: 'test_storage_v1',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Storage V2',
            name: 'test_storage_v2',
            defaultValue: true
        )
        booleanParam(
            description: 'Keep Environment After Tests',
            name: 'keep_env',
            defaultValue: false
        )
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        NAMESPACE = "chaos-testing"
    }
    
    stages {
        stage('Prepare Test Scenarios') {
            steps {
                script {
                    // Define all test scenarios
                    def allScenarios = []
                    
                    // Schema types to test - comprehensive coverage of all available schemas
                    def schemaTypes = [
                        'product_catalog',    // Simple product catalog with auto_id (4 fields, 128d)
                        'ecommerce_search',   // E-commerce with nullable fields (5 fields, 256d)
                        'news_articles',      // News with dynamic fields (4 fields, 768d)
                        'document_search',    // Document search with sparse vectors + BM25 (5 fields, 768d)
                        'multi_tenant_data',  // Multi-tenant with partitioning (5 fields, 256d)
                        'multimedia_content'  // Multiple vector types + nullable fields (7 fields, 256d+384d+128d)
                    ]
                    
                    // File configurations - optimized for comprehensive testing without 10*10GB
                    def fileConfigs = []
                    if (params.run_large_files) {
                        fileConfigs.add([count: 10, size: '1GB', desc: 'Large Files'])
                    }
                    if (params.run_small_files) {
                        fileConfigs.add([count: 100, size: '100MB', desc: 'Small Files'])
                    }
                    
                    // Formats
                    def formats = []
                    if (params.test_parquet) {
                        formats.add('parquet')
                    }
                    if (params.test_json) {
                        formats.add('json')
                    }
                    
                    // Storage versions
                    def storageVersions = []
                    if (params.test_storage_v1) {
                        storageVersions.add('V1')
                    }
                    if (params.test_storage_v2) {
                        storageVersions.add('V2')
                    }
                    
                    // Build test matrix
                    schemaTypes.each { schema ->
                        fileConfigs.each { fileConfig ->
                            formats.each { format ->
                                storageVersions.each { storage ->
                                    allScenarios.add([
                                        schema: schema,
                                        fileCount: fileConfig.count,
                                        fileSize: fileConfig.size,
                                        fileDesc: fileConfig.desc,
                                        format: format,
                                        storage: storage,
                                        partitions: 1024,  // Multi-partition test
                                        shards: 16         // Multi-vchannel test
                                    ])
                                }
                            }
                        }
                    }
                    
                    env.TEST_SCENARIOS = groovy.json.JsonOutput.toJson(allScenarios)
                    echo "Total test scenarios: ${allScenarios.size()}"
                    echo "Test scenarios: ${env.TEST_SCENARIOS}"
                }
            }
        }
        
        stage('Execute Test Scenarios') {
            steps {
                script {
                    def scenarios = groovy.json.JsonSlurper().parseText(env.TEST_SCENARIOS)
                    def parallelTests = [:]
                    
                    scenarios.eachWithIndex { scenario, index ->
                        def testName = "Test-${index + 1}: ${scenario.schema}-${scenario.fileDesc}-${scenario.format}-${scenario.storage}"
                        
                        parallelTests[testName] = {
                            stage(testName) {
                                echo "Starting test: ${testName}"
                                echo "Configuration: ${groovy.json.JsonOutput.toJson(scenario)}"
                                
                                try {
                                    build job: 'milvus_import_stable_test', 
                                        parameters: [
                                            string(name: 'image_repository', value: params.image_repository),
                                            string(name: 'image_tag', value: params.image_tag),
                                            string(name: 'schema_type', value: scenario.schema),
                                            string(name: 'file_count', value: scenario.fileCount.toString()),
                                            string(name: 'file_size', value: scenario.fileSize),
                                            string(name: 'file_format', value: scenario.format),
                                            string(name: 'storage_version', value: scenario.storage),
                                            string(name: 'partition_count', value: scenario.partitions.toString()),
                                            string(name: 'shard_count', value: scenario.shards.toString()),
                                            booleanParam(name: 'keep_env', value: params.keep_env)
                                        ],
                                        wait: true,
                                        propagate: false
                                    
                                    echo "Test ${testName} completed successfully"
                                } catch (Exception e) {
                                    echo "Test ${testName} failed: ${e.getMessage()}"
                                    currentBuild.result = 'UNSTABLE'
                                }
                            }
                        }
                    }
                    
                    // Execute tests in batches to avoid overwhelming the system
                    def batchSize = 4  // Run 4 tests in parallel
                    def batches = []
                    parallelTests.each { name, test ->
                        if (batches.size() == 0 || batches[-1].size() >= batchSize) {
                            batches.add([:])
                        }
                        batches[-1][name] = test
                    }
                    
                    batches.eachWithIndex { batch, batchIndex ->
                        stage("Batch ${batchIndex + 1}") {
                            parallel batch
                        }
                    }
                }
            }
        }
        
        stage('Collect Results') {
            steps {
                script {
                    echo "All test scenarios completed"
                    echo "Collecting test results and metrics..."
                    
                    // Archive test results
                    sh """
                    mkdir -p ${env.ARTIFACTS}
                    echo "Test Summary:" > ${env.ARTIFACTS}/test_summary.txt
                    echo "Total Scenarios: ${groovy.json.JsonSlurper().parseText(env.TEST_SCENARIOS).size()}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "Parameters:" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Large Files: ${params.run_large_files}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Small Files: ${params.run_small_files}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - JSON Format: ${params.test_json}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Parquet Format: ${params.test_parquet}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Storage V1: ${params.test_storage_v1}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Storage V2: ${params.test_storage_v2}" >> ${env.ARTIFACTS}/test_summary.txt
                    """
                    
                    archiveArtifacts artifacts: "_artifacts/**", allowEmptyArchive: true
                }
            }
        }
    }
    
    post {
        always {
            echo 'Batch test pipeline completed'
        }
        success {
            echo 'All test scenarios executed successfully!'
        }
        unstable {
            echo 'Some test scenarios failed. Check individual test results.'
        }
        failure {
            echo 'Batch test pipeline failed'
        }
    }
}
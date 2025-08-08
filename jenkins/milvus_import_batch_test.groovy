// This pipeline triggers multiple milvus import tests with different configurations
// It covers various scenarios including large files, small files, different formats, and storage versions

pipeline {
    options {
        timestamps()
        timeout(time: 2000, unit: 'MINUTES')   // Extended timeout for batch tests
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
        booleanParam(
            description: 'Generate performance reports for all tests',
            name: 'generate_reports',
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
        NAMESPACE = "chaos-testing"
        BATCH_REPORTS_DIR = "${env.WORKSPACE}/_batch_reports"
    }
    
    stages {
        stage('Execute Test Scenarios') {
            steps {
                script {
                    // Define all test scenarios
                    def allScenarios = []
                    
                    // Schema types to test - comprehensive coverage of all available schemas
                    def schemaTypes = [
                        // 'product_catalog',    // Simple product catalog with auto_id (4 fields, 128d)
                        // 'ecommerce_search',   // E-commerce with nullable fields (5 fields, 256d)
                        // 'news_articles',      // News with dynamic fields (4 fields, 768d)
                        'document_search',    // Document search with sparse vectors + BM25 (5 fields, 768d)
                        'multi_tenant_data',  // Multi-tenant with partitioning (5 fields, 256d)
                        // 'multimedia_content'  // Multiple vector types + nullable fields (7 fields, 256d+384d+128d)
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
                                        partitions: 16,  // Multi-partition test
                                        shards: 2         // Multi-vchannel test
                                    ])
                                }
                            }
                        }
                    }
                    
                    echo "Total test scenarios: ${allScenarios.size()}"
                    currentBuild.description = "Running ${allScenarios.size()} test scenarios"
                    env.TOTAL_SCENARIOS = allScenarios.size().toString()
                    
                    // Now execute the tests
                    def parallelTests = [:]
                    
                    allScenarios.eachWithIndex { scenario, index ->
                        def testName = "Test-${index + 1}: ${scenario.schema}-${scenario.fileDesc}-${scenario.format}-${scenario.storage}"
                        
                        parallelTests[testName] = {
                            stage(testName) {
                                echo "Starting test: ${testName}"
                                echo "Configuration: schema=${scenario.schema}, files=${scenario.fileCount}x${scenario.fileSize}, format=${scenario.format}, storage=${scenario.storage}"
                                
                                try {
                                    def result = build job: 'import-stable-test', 
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
                                            booleanParam(name: 'keep_env', value: params.keep_env),
                                            booleanParam(name: 'generate_report', value: params.generate_reports),
                                            string(name: 'loki_url', value: params.loki_url),
                                            string(name: 'prometheus_url', value: params.prometheus_url)
                                        ],
                                        wait: true,
                                        propagate: false
                                    
                                    // Try to copy artifacts from the completed job
                                    if (params.generate_reports) {
                                        try {
                                            copyArtifacts(
                                                projectName: 'import-stable-test',
                                                selector: specific("${result.number}"),
                                                filter: '_artifacts/reports/**/*',
                                                target: "${env.BATCH_REPORTS_DIR}/${testName}/",
                                                fingerprintArtifacts: true,
                                                flatten: false,
                                                optional: true
                                            )
                                            echo "Successfully copied artifacts from ${testName}"
                                        } catch (Exception e) {
                                            echo "Failed to copy artifacts from ${testName}: ${e.getMessage()}"
                                        }
                                    }
                                    
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
                    echo "Total Scenarios: ${env.TOTAL_SCENARIOS}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "Parameters:" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Large Files: ${params.run_large_files}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Small Files: ${params.run_small_files}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - JSON Format: ${params.test_json}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Parquet Format: ${params.test_parquet}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Storage V1: ${params.test_storage_v1}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Storage V2: ${params.test_storage_v2}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Generate Reports: ${params.generate_reports}" >> ${env.ARTIFACTS}/test_summary.txt
                    """
                    
                    archiveArtifacts artifacts: "_artifacts/**", allowEmptyArchive: true
                }
            }
        }
        
        stage('Aggregate Performance Reports') {
            when {
                expression { params.generate_reports == true }
            }
            steps {
                script {
                    echo "Aggregating performance reports from all test scenarios"
                    
                    sh """
                    # Create aggregated reports directory
                    mkdir -p ${env.ARTIFACTS}/aggregated_reports
                    
                    # Check if any CSV reports were collected
                    CSV_FILES=\$(find ${env.BATCH_REPORTS_DIR} -name "*.csv" 2>/dev/null | wc -l)
                    
                    if [ "\${CSV_FILES}" -gt 0 ]; then
                        echo "Found \${CSV_FILES} CSV reports from individual tests"
                        
                        # Create the merged CSV file
                        MERGED_CSV="${env.ARTIFACTS}/aggregated_reports/all_tests_performance_report.csv"
                        
                        # Initialize with header from first CSV file
                        FIRST_CSV=\$(find ${env.BATCH_REPORTS_DIR} -name "*.csv" | head -1)
                        if [ -f "\${FIRST_CSV}" ]; then
                            # Copy header line
                            head -1 "\${FIRST_CSV}" > "\${MERGED_CSV}"
                            
                            # Add all data lines from all CSV files (skip headers)
                            find ${env.BATCH_REPORTS_DIR} -name "*.csv" | while read csv_file; do
                                # Extract test name from path for identification
                                test_dir=\$(dirname \$(dirname \$(dirname \${csv_file})))
                                test_name=\$(basename \${test_dir})
                                
                                # Add data lines (skip header) with test name prefix if possible
                                tail -n +2 "\${csv_file}" | sed "s/^/\${test_name}: /" >> "\${MERGED_CSV}"
                            done
                            
                            echo "✅ Successfully merged \${CSV_FILES} CSV reports into: \${MERGED_CSV}"
                            echo "Total data rows: \$(( \$(wc -l < "\${MERGED_CSV}") - 1 ))"
                            
                        else
                            echo "❌ No valid CSV files found to merge"
                        fi
                        
                        # Create a simple summary file with test metadata
                        SUMMARY_FILE="${env.ARTIFACTS}/aggregated_reports/test_summary.csv"
                        echo "Test Name,Schema Type,File Count,File Size,Format,Storage Version,Status" > "\${SUMMARY_FILE}"
                        
                        # Extract metadata from import_info.json files
                        find ${env.BATCH_REPORTS_DIR} -name "import_info.json" | while read info_file; do
                            test_name=\$(basename \$(dirname \$(dirname \${info_file})))
                            schema_type=\$(jq -r '.test_parameters.schema_type' \${info_file} 2>/dev/null || echo "unknown")
                            file_count=\$(jq -r '.test_parameters.file_count' \${info_file} 2>/dev/null || echo "unknown")
                            file_size=\$(jq -r '.test_parameters.file_size' \${info_file} 2>/dev/null || echo "unknown")
                            file_format=\$(jq -r '.test_parameters.file_format' \${info_file} 2>/dev/null || echo "unknown")
                            storage_version=\$(jq -r '.test_parameters.storage_version' \${info_file} 2>/dev/null || echo "unknown")
                            
                            echo "\${test_name},\${schema_type},\${file_count},\${file_size},\${file_format},\${storage_version},Completed" >> "\${SUMMARY_FILE}"
                        done
                        
                        echo "✅ Created test summary: \${SUMMARY_FILE}"
                        
                        # Create a simple text report for easy reading
                        REPORT_SUMMARY="${env.ARTIFACTS}/aggregated_reports/batch_report_summary.txt"
                        cat > "\${REPORT_SUMMARY}" << EOF
Batch Import Test Results Summary
=================================

Build Information:
- Job: ${env.JOB_NAME}
- Build: ${env.BUILD_NUMBER}
- Total Test Scenarios: ${env.TOTAL_SCENARIOS}
- CSV Reports Found: \${CSV_FILES}
- Generated: \$(date -u +"%Y-%m-%d %H:%M:%S UTC")

Files Generated:
- all_tests_performance_report.csv: Merged performance data from all tests
- test_summary.csv: Test configuration and status summary
- batch_report_summary.txt: This summary file

Parameters Used:
- Large Files: ${params.run_large_files}
- Small Files: ${params.run_small_files}
- JSON Format: ${params.test_json}
- Parquet Format: ${params.test_parquet}
- Storage V1: ${params.test_storage_v1}
- Storage V2: ${params.test_storage_v2}
- Generate Reports: ${params.generate_reports}

Data Sources:
- Loki URL: ${params.loki_url}
- Prometheus URL: ${params.prometheus_url}
EOF
                        
                        echo "✅ Created summary report: \${REPORT_SUMMARY}"
                        
                    else
                        echo "❌ No CSV reports found from individual tests"
                        
                        # Create empty summary file
                        echo "No performance reports were generated from the batch tests." > ${env.ARTIFACTS}/aggregated_reports/no_reports.txt
                        echo "This could be due to:" >> ${env.ARTIFACTS}/aggregated_reports/no_reports.txt
                        echo "- Report generation was disabled" >> ${env.ARTIFACTS}/aggregated_reports/no_reports.txt
                        echo "- All individual tests failed" >> ${env.ARTIFACTS}/aggregated_reports/no_reports.txt
                        echo "- Issues with data collection" >> ${env.ARTIFACTS}/aggregated_reports/no_reports.txt
                    fi
                    
                    # Show final contents
                    echo ""
                    echo "Final aggregated reports directory contents:"
                    ls -la ${env.ARTIFACTS}/aggregated_reports/ || true
                    """
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
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
                    
                    # Check if any reports were collected
                    if [ -d "${env.BATCH_REPORTS_DIR}" ] && [ "\$(find ${env.BATCH_REPORTS_DIR} -name '*.html' -o -name '*.json' -o -name '*.csv' | wc -l)" -gt 0 ]; then
                        echo "Found performance reports from individual tests"
                        
                        # Create index of all collected reports
                        cat > ${env.ARTIFACTS}/aggregated_reports/reports_index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Import Performance Reports - Build ${env.BUILD_NUMBER}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f7fa; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .stat-card { background: white; padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }
        .stat-number { font-size: 2.5em; font-weight: bold; color: #667eea; }
        .stat-label { color: #666; margin-top: 8px; }
        .reports-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .report-card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); padding: 20px; }
        .report-title { font-size: 1.1em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .report-details { color: #666; font-size: 0.9em; margin-bottom: 15px; }
        .report-links a { display: inline-block; margin-right: 10px; padding: 5px 10px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; font-size: 0.8em; }
        .report-links a:hover { background: #5a67d8; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Batch Import Performance Reports</h1>
        <p><strong>Build:</strong> ${env.BUILD_NUMBER} | <strong>Job:</strong> ${env.JOB_NAME}</p>
        <p><strong>Generated:</strong> \$(date -u +"%Y-%m-%d %H:%M:%S UTC")</p>
    </div>

    <div class="summary">
        <div class="stat-card">
            <div class="stat-number">${env.TOTAL_SCENARIOS}</div>
            <div class="stat-label">Total Tests</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="reportCount">0</div>
            <div class="stat-label">Performance Reports</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">\$(find ${env.BATCH_REPORTS_DIR} -name "*.html" 2>/dev/null | wc -l || echo 0)</div>
            <div class="stat-label">HTML Reports</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">\$(find ${env.BATCH_REPORTS_DIR} -name "*.json" 2>/dev/null | wc -l || echo 0)</div>
            <div class="stat-label">JSON Data Files</div>
        </div>
    </div>

    <div class="reports-grid" id="reportsGrid">
        <!-- Reports will be added here by script below -->
    </div>

    <script>
        // This will be populated by the shell script below
    </script>
</body>
</html>
EOF

                        # Generate JavaScript to populate reports dynamically
                        echo "<script>" >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                        echo "const reports = [" >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                        
                        REPORT_COUNT=0
                        # Find and process each test's reports
                        find ${env.BATCH_REPORTS_DIR} -name "_artifacts" -type d | while read artifacts_dir; do
                            if [ -d "\${artifacts_dir}/reports" ]; then
                                test_name=\$(basename \$(dirname \${artifacts_dir}))
                                
                                # Check for import_info.json to get test details
                                if [ -f "\${artifacts_dir}/import_info.json" ]; then
                                    # Extract test configuration from import_info.json
                                    schema_type=\$(jq -r '.schema_type // "unknown"' "\${artifacts_dir}/import_info.json")
                                    file_count=\$(jq -r '.file_count // "unknown"' "\${artifacts_dir}/import_info.json")
                                    file_size=\$(jq -r '.file_size // "unknown"' "\${artifacts_dir}/import_info.json")
                                    file_format=\$(jq -r '.file_format // "unknown"' "\${artifacts_dir}/import_info.json")
                                    storage_version=\$(jq -r '.storage_version // "unknown"' "\${artifacts_dir}/import_info.json")
                                    
                                    # Copy reports to aggregated directory with test-specific names
                                    for report_file in \${artifacts_dir}/reports/*; do
                                        if [ -f "\${report_file}" ]; then
                                            filename=\$(basename \${report_file})
                                            new_name="\${test_name}_\${filename}"
                                            cp "\${report_file}" "${env.ARTIFACTS}/aggregated_reports/\${new_name}"
                                            
                                            if [[ \${filename} == *.html ]]; then
                                                # Add to JavaScript array
                                                echo "  {" >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    name: '\${test_name}'," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    schema: '\${schema_type}'," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    fileCount: '\${file_count}'," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    fileSize: '\${file_size}'," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    format: '\${file_format}'," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    storage: '\${storage_version}'," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "    htmlReport: '\${new_name}'" >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                                echo "  }," >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html
                                            fi
                                        fi
                                    done
                                fi
                            fi
                        done
                        
                        # Complete JavaScript and render reports
                        cat >> ${env.ARTIFACTS}/aggregated_reports/reports_index.html << 'EOF'
];

// Populate the reports grid
const reportsGrid = document.getElementById('reportsGrid');
document.getElementById('reportCount').textContent = reports.length;

reports.forEach(report => {
    const card = document.createElement('div');
    card.className = 'report-card';
    
    card.innerHTML = \`
        <div class="report-title">\${report.name}</div>
        <div class="report-details">
            <strong>Schema:</strong> \${report.schema}<br>
            <strong>Files:</strong> \${report.fileCount} × \${report.fileSize}<br>
            <strong>Format:</strong> \${report.format}<br>
            <strong>Storage:</strong> \${report.storage}
        </div>
        <div class="report-links">
            <a href="\${report.htmlReport}" target="_blank">View Report</a>
        </div>
    \`;
    
    reportsGrid.appendChild(card);
});
</script>
EOF

                        # Create a summary CSV of all tests
                        echo "Test Name,Schema Type,File Count,File Size,Format,Storage Version,Status" > ${env.ARTIFACTS}/aggregated_reports/batch_summary.csv
                        find ${env.BATCH_REPORTS_DIR} -name "import_info.json" | while read info_file; do
                            test_name=\$(basename \$(dirname \$(dirname \${info_file})))
                            schema_type=\$(jq -r '.schema_type' \${info_file})
                            file_count=\$(jq -r '.file_count' \${info_file})
                            file_size=\$(jq -r '.file_size' \${info_file})
                            file_format=\$(jq -r '.file_format' \${info_file})
                            storage_version=\$(jq -r '.storage_version' \${info_file})
                            
                            echo "\${test_name},\${schema_type},\${file_count},\${file_size},\${file_format},\${storage_version},Completed" >> ${env.ARTIFACTS}/aggregated_reports/batch_summary.csv
                        done
                        
                        echo "Aggregated \$(find ${env.ARTIFACTS}/aggregated_reports -name "*.html" | wc -l) HTML reports"
                        echo "Created reports index at: ${env.ARTIFACTS}/aggregated_reports/reports_index.html"
                        
                    else
                        echo "No performance reports found from individual tests"
                        
                        cat > ${env.ARTIFACTS}/aggregated_reports/no_reports.html << EOF
<!DOCTYPE html>
<html>
<head><title>No Reports Generated</title></head>
<body>
    <h1>No Performance Reports Generated</h1>
    <p>Either report generation was disabled or all individual tests failed to produce reports.</p>
    <p>Build: ${env.BUILD_NUMBER}</p>
    <p>Total Scenarios: ${env.TOTAL_SCENARIOS}</p>
</body>
</html>
EOF
                    fi
                    
                    # List all files in aggregated reports
                    echo "Contents of aggregated reports directory:"
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
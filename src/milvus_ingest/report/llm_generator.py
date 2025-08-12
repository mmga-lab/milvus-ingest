"""LLM context generator for import analysis - generates llm.txt format."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..logging_config import get_logger
from .loki_collector import LokiDataCollector
from .models import ReportConfig
from .prometheus_collector import PrometheusCollector


class LLMGenerator:
    """Generate LLM-friendly context documentation from import test data."""

    def __init__(self, config: ReportConfig):
        """Initialize the LLM generator."""
        self.config = config
        self.logger = get_logger(__name__)
        self.loki_collector = LokiDataCollector(config)
        self.prometheus_collector = PrometheusCollector(config)

    def generate_llm_context(
        self,
        job_ids: list[str] | None = None,
        collection_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        output_file: str = "/tmp/test_context.llm.txt",
        test_scenario: str | None = None,
        notes: str | None = None,
        release_name: str | None = None,
        milvus_namespace: str | None = None,
        import_info_file: str | None = None,
    ) -> dict[str, Any]:
        """Generate LLM context document.

        Args:
            job_ids: List of job IDs for the import
            collection_name: Collection name
            start_time: Start time for analysis
            end_time: End time for analysis
            output_file: Output file path
            test_scenario: Test scenario description
            notes: Additional notes
            release_name: Milvus release name
            milvus_namespace: Milvus namespace
            import_info_file: Path to import_info.json

        Returns:
            Dictionary with generation summary
        """
        self.logger.info(f"Generating LLM context document for jobs: {job_ids}")

        # Collect all raw data
        raw_data = {
            "metadata": {
                "job_ids": job_ids,
                "collection_name": collection_name,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "test_scenario": test_scenario,
                "notes": notes,
                "release_name": release_name,
                "milvus_namespace": milvus_namespace,
            },
            "import_info": {},
            "loki_logs": [],
            "prometheus_metrics": {},
        }

        # Load import_info.json if provided
        if import_info_file:
            try:
                with open(import_info_file) as f:
                    raw_data["import_info"] = json.load(f)
                    self.logger.info(f"Loaded import info from {import_info_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load import metadata: {e}")

        # Collect raw Loki logs
        self.logger.info("Collecting Loki logs...")
        if job_ids:
            for job_id in job_ids:
                timing_data = self.loki_collector.collect_import_timing_data(
                    job_id=job_id,
                    collection_name=collection_name,
                    start_time=start_time,
                    end_time=end_time,
                )
                for data in timing_data:
                    raw_data["loki_logs"].append({
                        "timestamp": data.timestamp.isoformat() if data.timestamp else None,
                        "job_id": data.job_id,
                        "collection_name": data.collection_name,
                        "import_phase": data.import_phase,
                        "time_cost": data.time_cost,
                        "time_unit": data.time_unit,
                        "message": data.message,
                    })
        else:
            timing_data = self.loki_collector.collect_import_timing_data(
                collection_name=collection_name,
                start_time=start_time,
                end_time=end_time,
            )
            for data in timing_data:
                raw_data["loki_logs"].append({
                    "timestamp": data.timestamp.isoformat() if data.timestamp else None,
                    "job_id": data.job_id,
                    "collection_name": data.collection_name,
                    "import_phase": data.import_phase,
                    "time_cost": data.time_cost,
                    "time_unit": data.time_unit,
                    "message": data.message,
                })

        # Collect raw Prometheus metrics (direct API responses)
        if release_name and milvus_namespace and start_time and end_time:
            self.logger.info("Collecting raw Prometheus API responses...")
            try:
                # Store raw Prometheus query results
                raw_data["prometheus_metrics"]["raw_queries"] = {}
                
                # Define queries to execute
                queries = {
                    # DataNode/Standalone memory usage
                    "datanode_memory": f'container_memory_working_set_bytes{{cluster="", namespace="{milvus_namespace}", pod=~"{release_name}-milvus-.*", container!="", image!=""}}',
                    
                    # DataNode/Standalone CPU usage
                    "datanode_cpu": f'rate(container_cpu_usage_seconds_total{{cluster="", namespace="{milvus_namespace}", pod=~"{release_name}-milvus-.*", container!="", image!=""}}[5m])',
                    
                    # MinIO container filesystem IOPS (since MinIO metrics may not be available)
                    "minio_read_iops": f'rate(container_fs_reads_total{{container!="", cluster="", namespace="{milvus_namespace}", pod=~"{release_name}-minio-.*"}}[5m])',
                    "minio_write_iops": f'rate(container_fs_writes_total{{container!="", cluster="", namespace="{milvus_namespace}", pod=~"{release_name}-minio-.*"}}[5m])',
                    
                    # MinIO container filesystem throughput  
                    "minio_read_bytes": f'rate(container_fs_reads_bytes_total{{container!="", cluster="", namespace="{milvus_namespace}", pod=~"{release_name}-minio-.*"}}[5m])',
                    "minio_write_bytes": f'rate(container_fs_writes_bytes_total{{container!="", cluster="", namespace="{milvus_namespace}", pod=~"{release_name}-minio-.*"}}[5m])',
                    
                    # Binlog metrics
                    "binlog_count": f'milvus_datacoord_segment_binlog_file_count{{app_kubernetes_io_instance="{release_name}"}}',
                    "binlog_size": f'milvus_datacoord_segment_binlog_file_size{{app_kubernetes_io_instance="{release_name}"}}',
                }
                
                # Execute each query and store raw response
                for query_name, query_string in queries.items():
                    self.logger.info(f"Executing Prometheus query: {query_name}")
                    try:
                        # Use range query to get time series data
                        response = self.prometheus_collector.query_prometheus_range(
                            query_string, start_time, end_time
                        )
                        raw_data["prometheus_metrics"]["raw_queries"][query_name] = {
                            "query": query_string,
                            "response": response
                        }
                    except Exception as query_error:
                        self.logger.warning(f"Failed to execute query {query_name}: {query_error}")
                        raw_data["prometheus_metrics"]["raw_queries"][query_name] = {
                            "query": query_string,
                            "error": str(query_error)
                        }
                        
                # Also store query metadata
                raw_data["prometheus_metrics"]["query_metadata"] = {
                    "release_name": release_name,
                    "namespace": milvus_namespace,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "prometheus_url": self.prometheus_collector.prometheus_url
                }
                
            except Exception as e:
                self.logger.error(f"Error collecting Prometheus metrics: {e}")
                raw_data["prometheus_metrics"]["error"] = str(e)

        # Write raw data to file
        self._write_raw_document(raw_data, output_file)

        # Return summary
        return {
            "jobs_analyzed": len(job_ids) if job_ids else 0,
            "total_logs": len(raw_data["loki_logs"]),
            "output_file": output_file,
            "raw_data": raw_data,
        }

    def _write_raw_document(self, raw_data: dict[str, Any], output_file: str):
        """Write the raw data document."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Milvus Import Test Raw Data\n\n")
            f.write("This document contains raw, unprocessed data from the Milvus import test.\n\n")
            
            # Metadata section
            f.write("## Test Metadata\n\n")
            f.write("```json\n")
            f.write(json.dumps(raw_data["metadata"], indent=2, ensure_ascii=False))
            f.write("\n```\n\n")
            
            # Import info section
            if raw_data["import_info"]:
                f.write("## Import Info (from import_info.json)\n\n")
                f.write("```json\n")
                f.write(json.dumps(raw_data["import_info"], indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
            
            # Loki logs section
            f.write("## Loki Logs (Raw)\n\n")
            f.write(f"Total log entries: {len(raw_data['loki_logs'])}\n\n")
            f.write("```json\n")
            f.write(json.dumps(raw_data["loki_logs"], indent=2, ensure_ascii=False, default=str))
            f.write("\n```\n\n")
            
            # Prometheus metrics section
            if raw_data["prometheus_metrics"]:
                f.write("## Prometheus Metrics (Raw)\n\n")
                f.write("```json\n")
                f.write(json.dumps(raw_data["prometheus_metrics"], indent=2, ensure_ascii=False, default=str))
                f.write("\n```\n\n")
            
            # Query information
            f.write("## Data Collection Information\n\n")
            f.write(f"- Loki URL: {self.config.loki_url}\n")
            f.write(f"- Prometheus URL: {self.config.prometheus_url}\n")
            f.write(f"- Pod Pattern: {self.config.pod_pattern}\n")
            f.write(f"- Namespace: {self.config.namespace}\n")
            f.write(f"- Max Log Entries: {self.config.max_log_entries}\n")
            f.write(f"- Timeout: {self.config.timeout_seconds} seconds\n")
            f.write("\n")

        self.logger.info(f"Raw data document written to {output_path}")
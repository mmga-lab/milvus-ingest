"""Prometheus collector for raw metrics data collection only."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

from ..logging_config import get_logger
from .models import ReportConfig


class PrometheusCollector:
    """Simple Prometheus collector that only fetches raw data without analysis."""

    def __init__(self, config: ReportConfig):
        """Initialize the Prometheus collector.

        Args:
            config: Report configuration containing Prometheus URL and other settings
        """
        self.config = config
        self.prometheus_url = config.prometheus_url.rstrip("/")
        self.logger = get_logger(__name__)

    def query_prometheus(self, query: str) -> dict:
        """Execute a Prometheus instant query and return raw response."""
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def query_prometheus_range(
        self, query: str, start_time: datetime, end_time: datetime, step: str = "30s"
    ) -> dict:
        """Execute a Prometheus range query and return raw response."""
        params = {
            "query": query,
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
            "step": step,
        }
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params=params,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def collect_raw_metrics(
        self,
        release_name: str,
        namespace: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Collect raw Prometheus metrics without processing.

        Returns all raw query responses for downstream analysis.
        """
        raw_metrics = {
            "raw_queries": {},
            "metadata": {
                "release_name": release_name,
                "namespace": namespace,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "query_timestamp": datetime.now().isoformat(),
            },
        }

        # Define all queries to execute
        milvus_pod_pattern = f"{release_name}-milvus-.*"
        minio_pod_pattern = f"{release_name}-minio-.*"

        queries = {
            # Milvus components memory usage
            "milvus_memory": f'container_memory_working_set_bytes{{cluster="", namespace="{namespace}", pod=~"{milvus_pod_pattern}", container!="", image!=""}}',
            # Milvus components CPU usage
            "milvus_cpu": f'rate(container_cpu_usage_seconds_total{{cluster="", namespace="{namespace}", pod=~"{milvus_pod_pattern}", container!="", image!=""}}[5m])',
            # MinIO IOPS
            "minio_read_iops": f'rate(container_fs_reads_total{{container!="", cluster="", namespace="{namespace}", pod=~"{minio_pod_pattern}"}}[5m])',
            "minio_write_iops": f'rate(container_fs_writes_total{{container!="", cluster="", namespace="{namespace}", pod=~"{minio_pod_pattern}"}}[5m])',
            # MinIO throughput
            "minio_read_bytes": f'rate(container_fs_reads_bytes_total{{container!="", cluster="", namespace="{namespace}", pod=~"{minio_pod_pattern}"}}[5m])',
            "minio_write_bytes": f'rate(container_fs_writes_bytes_total{{container!="", cluster="", namespace="{namespace}", pod=~"{minio_pod_pattern}"}}[5m])',
            # Binlog metrics
            "binlog_count": f'sum(milvus_datacoord_segment_binlog_file_count{{app_kubernetes_io_instance="{release_name}"}})',
            "binlog_size": f'sum(milvus_datacoord_stored_binlog_size{{app_kubernetes_io_instance="{release_name}"}})',
            # Pod status
            "pod_status": f'up{{namespace="{namespace}", pod=~"{milvus_pod_pattern}"}}',
        }

        # Execute all queries and store raw responses
        for query_name, query_string in queries.items():
            try:
                self.logger.debug(f"Executing query '{query_name}': {query_string}")

                if start_time and end_time:
                    # Range query
                    result = self.query_prometheus_range(
                        query_string, start_time, end_time
                    )
                else:
                    # Instant query
                    result = self.query_prometheus(query_string)

                raw_metrics["raw_queries"][query_name] = {
                    "query": query_string,
                    "response": result,
                    "query_type": "range" if (start_time and end_time) else "instant",
                }

            except Exception as e:
                self.logger.warning(f"Failed to execute query '{query_name}': {e}")
                raw_metrics["raw_queries"][query_name] = {
                    "query": query_string,
                    "error": str(e),
                    "query_type": "range" if (start_time and end_time) else "instant",
                }

        return raw_metrics

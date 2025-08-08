"""Prometheus collector for Milvus import performance metrics with time range support."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..logging_config import get_logger
from .models import ReportConfig


class PrometheusCollector:
    """Prometheus collector with time range support and min/max/avg calculations."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the Prometheus collector.
        
        Args:
            config: Report configuration containing Prometheus URL and other settings
        """
        self.config = config
        self.prometheus_url = config.prometheus_url.rstrip('/')
        self.logger = get_logger(__name__)
    
    def query_prometheus(self, query: str) -> dict:
        """Execute a Prometheus instant query."""
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def query_prometheus_range(self, query: str, start_time: datetime, end_time: datetime, step: str = "30s") -> dict:
        """Execute a Prometheus range query."""
        params = {
            "query": query,
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
            "step": step
        }
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params=params,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def calculate_stats(self, values: list) -> dict:
        """Calculate min, max, average from a list of values."""
        if not values:
            return {"min": 0, "max": 0, "avg": 0}
        
        values = [float(v) for v in values if v is not None]
        if not values:
            return {"min": 0, "max": 0, "avg": 0}
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values)
        }

    def collect_datanode_metrics(
        self, 
        release_name: str, 
        namespace: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get DataNode memory and CPU metrics with optional time range support."""
        metrics = {}
        
        # DataNode Memory (in GB)
        memory_query = f'sum(container_memory_working_set_bytes{{cluster="", namespace="{namespace}", pod=~"{release_name}-milvus-datanode-.*", container!="", image!=""}}) by (container)'
        
        if start_time and end_time:
            try:
                result = self.query_prometheus_range(memory_query, start_time, end_time)
                if result['data']['result']:
                    all_values = []
                    for series in result['data']['result']:
                        for timestamp, value in series['values']:
                            all_values.append(float(value))
                    
                    if all_values:
                        all_values_gb = [v / (1024 * 1024 * 1024) for v in all_values]
                        memory_stats = self.calculate_stats(all_values_gb)
                        metrics['datanode_memory_gb'] = memory_stats
                        metrics['datanode_container'] = result['data']['result'][0]['metric'].get('container', 'unknown')
                    else:
                        metrics['datanode_memory_gb'] = {"min": 0, "max": 0, "avg": 0}
                else:
                    metrics['datanode_memory_gb'] = {"min": 0, "max": 0, "avg": 0}
            except Exception as e:
                self.logger.warning(f"Error querying DataNode memory range: {e}")
                metrics['datanode_memory_gb'] = {"min": 0, "max": 0, "avg": 0}
        else:
            try:
                result = self.query_prometheus(memory_query)
                if result['data']['result']:
                    total_memory_bytes = sum(float(r['value'][1]) for r in result['data']['result'])
                    metrics['datanode_memory_gb'] = total_memory_bytes / (1024 * 1024 * 1024)
                    metrics['datanode_container'] = result['data']['result'][0]['metric'].get('container', 'unknown')
                else:
                    metrics['datanode_memory_gb'] = 0
            except Exception as e:
                self.logger.warning(f"Error querying DataNode memory: {e}")
                metrics['datanode_memory_gb'] = 0
        
        # DataNode CPU (cores)
        cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{release_name}-milvus-datanode-.*"}}[5m]))'
        
        if start_time and end_time:
            try:
                result = self.query_prometheus_range(cpu_query, start_time, end_time)
                if result['data']['result']:
                    all_values = []
                    for series in result['data']['result']:
                        for timestamp, value in series['values']:
                            all_values.append(float(value))
                    
                    cpu_stats = self.calculate_stats(all_values)
                    metrics['datanode_cpu_cores'] = cpu_stats
                else:
                    metrics['datanode_cpu_cores'] = {"min": 0, "max": 0, "avg": 0}
            except Exception as e:
                self.logger.warning(f"Error querying DataNode CPU range: {e}")
                metrics['datanode_cpu_cores'] = {"min": 0, "max": 0, "avg": 0}
        else:
            try:
                result = self.query_prometheus(cpu_query)
                if result['data']['result']:
                    total_cpu = sum(float(r['value'][1]) for r in result['data']['result'])
                    metrics['datanode_cpu_cores'] = total_cpu
                else:
                    metrics['datanode_cpu_cores'] = 0
            except Exception as e:
                self.logger.warning(f"Error querying DataNode CPU: {e}")
                metrics['datanode_cpu_cores'] = 0
        
        return metrics

    def collect_s3_minio_metrics(
        self, 
        release_name: str, 
        namespace: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get S3/MinIO IOPS and throughput metrics with optional time range support."""
        metrics = {}
        
        # S3/MinIO IOPS using filesystem I/O operations
        reads_query = f'sum(rate(container_fs_reads_total{{container!="", cluster="", namespace="{namespace}", pod=~".*minio.*"}}[5m]))'
        writes_query = f'sum(rate(container_fs_writes_total{{container!="", cluster="", namespace="{namespace}", pod=~".*minio.*"}}[5m]))'
        
        if start_time and end_time:
            try:
                total_iops_values = []
                
                # Get reads over time
                result = self.query_prometheus_range(reads_query, start_time, end_time)
                reads_values = []
                if result['data']['result']:
                    for series in result['data']['result']:
                        for timestamp, value in series['values']:
                            reads_values.append(float(value))
                
                # Get writes over time
                result = self.query_prometheus_range(writes_query, start_time, end_time)
                writes_values = []
                if result['data']['result']:
                    for series in result['data']['result']:
                        for timestamp, value in series['values']:
                            writes_values.append(float(value))
                
                # Combine reads and writes
                max_len = max(len(reads_values), len(writes_values))
                for i in range(max_len):
                    reads = reads_values[i] if i < len(reads_values) else 0
                    writes = writes_values[i] if i < len(writes_values) else 0
                    total_iops_values.append(reads + writes)
                
                iops_stats = self.calculate_stats(total_iops_values)
                metrics['s3_minio_iops'] = iops_stats
            except Exception as e:
                self.logger.warning(f"Error querying S3/MinIO IOPS range: {e}")
                metrics['s3_minio_iops'] = {"min": 0, "max": 0, "avg": 0}
        else:
            try:
                total_iops = 0
                
                # Get reads
                result = self.query_prometheus(reads_query)
                if result['data']['result']:
                    total_iops += float(result['data']['result'][0]['value'][1])
                
                # Get writes
                result = self.query_prometheus(writes_query)
                if result['data']['result']:
                    total_iops += float(result['data']['result'][0]['value'][1])
                
                metrics['s3_minio_iops'] = total_iops
            except Exception as e:
                self.logger.warning(f"Error querying S3/MinIO IOPS: {e}")
                # Fallback to S3 API requests
                try:
                    iops_query = 'sum(rate(minio_s3_requests_total[5m]))'
                    result = self.query_prometheus(iops_query)
                    if result['data']['result']:
                        metrics['s3_minio_iops'] = float(result['data']['result'][0]['value'][1])
                    else:
                        metrics['s3_minio_iops'] = 0
                except:
                    metrics['s3_minio_iops'] = 0
        
        # S3/MinIO Throughput (bytes/sec -> MB/sec)
        throughput_queries = [
            'sum(rate(minio_s3_traffic_sent_bytes[5m])) + sum(rate(minio_s3_traffic_received_bytes[5m]))',
            'sum(rate(minio_inter_node_traffic_sent_bytes[5m])) + sum(rate(minio_inter_node_traffic_received_bytes[5m]))',
            'sum(rate(minio_bucket_traffic_sent_bytes[5m])) + sum(rate(minio_bucket_traffic_received_bytes[5m]))'
        ]
        
        for query in throughput_queries:
            try:
                if start_time and end_time:
                    result = self.query_prometheus_range(query, start_time, end_time)
                    if result['data']['result']:
                        all_values = []
                        for series in result['data']['result']:
                            for timestamp, value in series['values']:
                                bytes_per_sec = float(value)
                                all_values.append(bytes_per_sec / (1024 * 1024))  # Convert to MB/s
                        
                        throughput_stats = self.calculate_stats(all_values)
                        metrics['s3_minio_throughput_mbps'] = throughput_stats
                        break
                else:
                    result = self.query_prometheus(query)
                    if result['data']['result']:
                        bytes_per_sec = float(result['data']['result'][0]['value'][1])
                        metrics['s3_minio_throughput_mbps'] = bytes_per_sec / (1024 * 1024)  # Convert to MB/s
                        break
            except Exception:
                continue
        
        if 's3_minio_throughput_mbps' not in metrics:
            if start_time and end_time:
                metrics['s3_minio_throughput_mbps'] = {"min": 0, "max": 0, "avg": 0}
            else:
                metrics['s3_minio_throughput_mbps'] = 0
        
        return metrics

    def collect_binlog_metrics(self, release_name: str) -> Dict[str, Any]:
        """Get Milvus binlog count and size metrics."""
        metrics = {}
        
        # Binlog file count
        count_query = f'sum(milvus_datacoord_segment_binlog_file_count{{app_kubernetes_io_instance="{release_name}"}})'
        try:
            result = self.query_prometheus(count_query)
            if result['data']['result']:
                metrics['binlog_count'] = int(float(result['data']['result'][0]['value'][1]))
            else:
                metrics['binlog_count'] = 0
        except Exception as e:
            self.logger.warning(f"Error querying binlog count: {e}")
            metrics['binlog_count'] = 0
        
        # Binlog total size (bytes -> GB)
        size_query = f'sum(milvus_datacoord_stored_binlog_size{{app_kubernetes_io_instance="{release_name}"}})/1024/1024/1024'
        try:
            result = self.query_prometheus(size_query)
            if result['data']['result']:
                metrics['binlog_size_gb'] = float(result['data']['result'][0]['value'][1])
            else:
                metrics['binlog_size_gb'] = 0
        except Exception as e:
            self.logger.warning(f"Error querying binlog size: {e}")
            metrics['binlog_size_gb'] = 0
        
        return metrics
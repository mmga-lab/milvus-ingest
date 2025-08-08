"""Prometheus collector for Milvus import performance metrics with time range support."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..logging_config import get_logger
from .models import ReportConfig


class PrometheusCollector:
    """Prometheus collector with time range support and min/max/avg calculations.
    
    Supports both cluster and standalone deployment modes for Milvus.
    """
    
    def __init__(self, config: ReportConfig):
        """Initialize the Prometheus collector.
        
        Args:
            config: Report configuration containing Prometheus URL and other settings
        """
        self.config = config
        self.prometheus_url = config.prometheus_url.rstrip('/')
        self.logger = get_logger(__name__)
    
    def detect_deployment_mode(self, release_name: str, namespace: str) -> str:
        """Auto-detect deployment mode by checking which pods exist.
        
        Args:
            release_name: Helm release name
            namespace: Kubernetes namespace
            
        Returns:
            "standalone" or "cluster" based on detected pods
        """
        try:
            # Check for standalone pod
            standalone_query = f'up{{namespace="{namespace}", pod=~"{release_name}-milvus-standalone-.*"}}'
            standalone_result = self.query_prometheus(standalone_query)
            
            if standalone_result['data']['result']:
                self.logger.info(f"Detected standalone deployment mode for {release_name}")
                return "standalone"
            
            # Check for cluster mode pods (datanode)
            cluster_query = f'up{{namespace="{namespace}", pod=~"{release_name}-milvus-datanode-.*"}}'
            cluster_result = self.query_prometheus(cluster_query)
            
            if cluster_result['data']['result']:
                self.logger.info(f"Detected cluster deployment mode for {release_name}")
                return "cluster"
            
            # Default to cluster mode if detection fails
            self.logger.warning(f"Could not detect deployment mode for {release_name}, defaulting to cluster")
            return "cluster"
            
        except Exception as e:
            self.logger.warning(f"Error detecting deployment mode: {e}, defaulting to cluster")
            return "cluster"
    
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
        """Get Milvus compute nodes (DataNode/Standalone) memory and CPU metrics for each pod separately.
        
        Automatically detects deployment mode and queries appropriate pods:
        - Cluster mode: milvus-datanode pods
        - Standalone mode: milvus-standalone pods
        """
        metrics = {
            "datanode_pods": [],
            "summary": {}
        }
        
        # Auto-detect or use configured deployment mode
        deployment_mode = self.config.deployment_mode
        if deployment_mode == "auto":
            deployment_mode = self.detect_deployment_mode(release_name, namespace)
        
        # Determine pod pattern based on deployment mode
        if deployment_mode == "standalone":
            pod_pattern = f"{release_name}-milvus-standalone-.*"
            self.logger.info(f"Using standalone mode - querying pods: {pod_pattern}")
        else:
            pod_pattern = f"{release_name}-milvus-datanode-.*"
            self.logger.info(f"Using cluster mode - querying pods: {pod_pattern}")
        
        # DataNode Memory (in GB) - per pod
        memory_query = f'container_memory_working_set_bytes{{cluster="", namespace="{namespace}", pod=~"{pod_pattern}", container!="", image!=""}}'
        
        try:
            # Use time range queries if available, otherwise single point
            if start_time and end_time:
                memory_result = self.query_prometheus_range(memory_query, start_time, end_time)
                cpu_query = f'rate(container_cpu_usage_seconds_total{{cluster="", namespace="{namespace}", pod=~"{pod_pattern}", container!="", image!=""}}[5m])'
                cpu_result = self.query_prometheus_range(cpu_query, start_time, end_time)
                use_time_range = True
            else:
                memory_result = self.query_prometheus(memory_query)
                cpu_query = f'rate(container_cpu_usage_seconds_total{{cluster="", namespace="{namespace}", pod=~"{pod_pattern}", container!="", image!=""}}[5m])'
                cpu_result = self.query_prometheus(cpu_query)
                use_time_range = False
            
            # Group by pod
            pod_metrics = {}
            
            # Process memory data
            if memory_result['data']['result']:
                for series in memory_result['data']['result']:
                    pod_name = series['metric'].get('pod', 'unknown')
                    container = series['metric'].get('container', 'unknown')
                    
                    if pod_name not in pod_metrics:
                        pod_metrics[pod_name] = {'pod_name': pod_name, 'containers': {}}
                    
                    if use_time_range:
                        # Extract all values over time and calculate stats
                        memory_values_gb = []
                        for timestamp, value in series['values']:
                            memory_gb = float(value) / (1024 * 1024 * 1024)
                            memory_values_gb.append(memory_gb)
                        
                        memory_stats = self.calculate_stats(memory_values_gb) if memory_values_gb else {'min': 0, 'max': 0, 'avg': 0}
                        pod_metrics[pod_name]['containers'][container] = {
                            'memory_gb': memory_stats,
                            'cpu_cores': {'min': 0, 'max': 0, 'avg': 0}
                        }
                    else:
                        # Single point in time
                        memory_gb = float(series['value'][1]) / (1024 * 1024 * 1024)
                        pod_metrics[pod_name]['containers'][container] = {
                            'memory_gb': round(memory_gb, 3),
                            'cpu_cores': 0
                        }
            
            # Process CPU data  
            if cpu_result['data']['result']:
                for series in cpu_result['data']['result']:
                    pod_name = series['metric'].get('pod', 'unknown')
                    container = series['metric'].get('container', 'unknown')
                    
                    if pod_name in pod_metrics and container in pod_metrics[pod_name]['containers']:
                        if use_time_range:
                            # Extract all values over time and calculate stats
                            cpu_values = []
                            for timestamp, value in series['values']:
                                cpu_cores = float(value)
                                cpu_values.append(cpu_cores)
                            
                            cpu_stats = self.calculate_stats(cpu_values) if cpu_values else {'min': 0, 'max': 0, 'avg': 0}
                            pod_metrics[pod_name]['containers'][container]['cpu_cores'] = cpu_stats
                        else:
                            # Single point in time
                            cpu_cores = float(series['value'][1])
                            pod_metrics[pod_name]['containers'][container]['cpu_cores'] = round(cpu_cores, 4)
            
            # Convert to list format and calculate summary
            all_memory_stats = {'min': [], 'max': [], 'avg': []}
            all_cpu_stats = {'min': [], 'max': [], 'avg': []}
            
            for pod_name, pod_data in pod_metrics.items():
                pod_info = {
                    'pod_name': pod_name,
                    'containers': [],
                    'use_time_range': use_time_range
                }
                
                for container, container_data in pod_data['containers'].items():
                    pod_info['containers'].append({
                        'container_name': container,
                        'memory_gb': container_data['memory_gb'],
                        'cpu_cores': container_data['cpu_cores']
                    })
                    
                    # Collect values for summary statistics
                    if use_time_range:
                        if isinstance(container_data['memory_gb'], dict):
                            all_memory_stats['min'].append(container_data['memory_gb']['min'])
                            all_memory_stats['max'].append(container_data['memory_gb']['max'])
                            all_memory_stats['avg'].append(container_data['memory_gb']['avg'])
                        if isinstance(container_data['cpu_cores'], dict):
                            all_cpu_stats['min'].append(container_data['cpu_cores']['min'])
                            all_cpu_stats['max'].append(container_data['cpu_cores']['max'])
                            all_cpu_stats['avg'].append(container_data['cpu_cores']['avg'])
                    else:
                        all_memory_stats['avg'].append(container_data['memory_gb'])
                        all_cpu_stats['avg'].append(container_data['cpu_cores'])
                
                metrics['datanode_pods'].append(pod_info)
            
            # Calculate summary statistics
            if use_time_range:
                if all_memory_stats['avg']:
                    metrics['summary']['memory_gb'] = {
                        'total_avg': round(sum(all_memory_stats['avg']), 3),
                        'min': round(min(all_memory_stats['min']), 3),
                        'max': round(max(all_memory_stats['max']), 3),
                        'avg': round(sum(all_memory_stats['avg']) / len(all_memory_stats['avg']), 3)
                    }
                
                if all_cpu_stats['avg']:
                    metrics['summary']['cpu_cores'] = {
                        'total_avg': round(sum(all_cpu_stats['avg']), 4),
                        'min': round(min(all_cpu_stats['min']), 4),
                        'max': round(max(all_cpu_stats['max']), 4),
                        'avg': round(sum(all_cpu_stats['avg']) / len(all_cpu_stats['avg']), 4)
                    }
            else:
                if all_memory_stats['avg']:
                    metrics['summary']['total_memory_gb'] = round(sum(all_memory_stats['avg']), 3)
                    metrics['summary']['avg_memory_gb'] = round(sum(all_memory_stats['avg']) / len(all_memory_stats['avg']), 3)
                
                if all_cpu_stats['avg']:
                    metrics['summary']['total_cpu_cores'] = round(sum(all_cpu_stats['avg']), 4)
                    metrics['summary']['avg_cpu_cores'] = round(sum(all_cpu_stats['avg']) / len(all_cpu_stats['avg']), 4)
                
        except Exception as e:
            self.logger.warning(f"Error collecting DataNode metrics: {e}")
            metrics = {"datanode_pods": [], "summary": {}}
        
        return metrics

    def collect_s3_minio_metrics(
        self, 
        release_name: str, 
        namespace: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get S3/MinIO IOPS and throughput metrics for each pod separately."""
        metrics = {
            "minio_pods": [],
            "summary": {}
        }
        
        # S3/MinIO IOPS using filesystem I/O operations - per pod
        reads_query = f'rate(container_fs_reads_total{{container!="", cluster="", namespace="{namespace}", pod=~"{release_name}-minio-.*"}}[5m])'
        writes_query = f'rate(container_fs_writes_total{{container!="", cluster="", namespace="{namespace}", pod=~"{release_name}-minio-.*"}}[5m])'
        
        # Throughput queries - per pod
        read_bytes_query = f'rate(container_fs_reads_bytes_total{{container!="", cluster="", namespace="{namespace}", pod=~"{release_name}-minio-.*"}}[5m])'
        write_bytes_query = f'rate(container_fs_writes_bytes_total{{container!="", cluster="", namespace="{namespace}", pod=~"{release_name}-minio-.*"}}[5m])'
        
        try:
            # Use time range queries if available, otherwise single point
            if start_time and end_time:
                reads_result = self.query_prometheus_range(reads_query, start_time, end_time)
                writes_result = self.query_prometheus_range(writes_query, start_time, end_time)
                read_bytes_result = self.query_prometheus_range(read_bytes_query, start_time, end_time)
                write_bytes_result = self.query_prometheus_range(write_bytes_query, start_time, end_time)
                use_time_range = True
            else:
                reads_result = self.query_prometheus(reads_query)
                writes_result = self.query_prometheus(writes_query)
                read_bytes_result = self.query_prometheus(read_bytes_query)
                write_bytes_result = self.query_prometheus(write_bytes_query)
                use_time_range = False
            
            # Group by pod
            pod_metrics = {}
            
            # Process reads IOPS
            if reads_result['data']['result']:
                for series in reads_result['data']['result']:
                    pod_name = series['metric'].get('pod', 'unknown')
                    container = series['metric'].get('container', 'unknown')
                    
                    if pod_name not in pod_metrics:
                        pod_metrics[pod_name] = {'pod_name': pod_name, 'containers': {}}
                    
                    if container not in pod_metrics[pod_name]['containers']:
                        if use_time_range:
                            pod_metrics[pod_name]['containers'][container] = {
                                'reads_per_sec': {'min': 0, 'max': 0, 'avg': 0}, 
                                'writes_per_sec': {'min': 0, 'max': 0, 'avg': 0},
                                'read_mbps': {'min': 0, 'max': 0, 'avg': 0}, 
                                'write_mbps': {'min': 0, 'max': 0, 'avg': 0}
                            }
                        else:
                            pod_metrics[pod_name]['containers'][container] = {
                                'reads_per_sec': 0, 'writes_per_sec': 0,
                                'read_mbps': 0, 'write_mbps': 0
                            }
                    
                    if use_time_range:
                        reads_values = [float(v) for t, v in series['values']]
                        reads_stats = self.calculate_stats(reads_values) if reads_values else {'min': 0, 'max': 0, 'avg': 0}
                        pod_metrics[pod_name]['containers'][container]['reads_per_sec'] = reads_stats
                    else:
                        reads_per_sec = float(series['value'][1])
                        pod_metrics[pod_name]['containers'][container]['reads_per_sec'] = round(reads_per_sec, 2)
            
            # Process writes IOPS
            if writes_result['data']['result']:
                for series in writes_result['data']['result']:
                    pod_name = series['metric'].get('pod', 'unknown')
                    container = series['metric'].get('container', 'unknown')
                    
                    if pod_name in pod_metrics and container in pod_metrics[pod_name]['containers']:
                        if use_time_range:
                            writes_values = [float(v) for t, v in series['values']]
                            writes_stats = self.calculate_stats(writes_values) if writes_values else {'min': 0, 'max': 0, 'avg': 0}
                            pod_metrics[pod_name]['containers'][container]['writes_per_sec'] = writes_stats
                        else:
                            writes_per_sec = float(series['value'][1])
                            pod_metrics[pod_name]['containers'][container]['writes_per_sec'] = round(writes_per_sec, 2)
            
            # Process read throughput
            if read_bytes_result['data']['result']:
                for series in read_bytes_result['data']['result']:
                    pod_name = series['metric'].get('pod', 'unknown')
                    container = series['metric'].get('container', 'unknown')
                    
                    if pod_name in pod_metrics and container in pod_metrics[pod_name]['containers']:
                        if use_time_range:
                            read_mbps_values = [float(v) / (1024 * 1024) for t, v in series['values']]
                            read_mbps_stats = self.calculate_stats(read_mbps_values) if read_mbps_values else {'min': 0, 'max': 0, 'avg': 0}
                            pod_metrics[pod_name]['containers'][container]['read_mbps'] = read_mbps_stats
                        else:
                            read_mbps = float(series['value'][1]) / (1024 * 1024)
                            pod_metrics[pod_name]['containers'][container]['read_mbps'] = round(read_mbps, 2)
            
            # Process write throughput
            if write_bytes_result['data']['result']:
                for series in write_bytes_result['data']['result']:
                    pod_name = series['metric'].get('pod', 'unknown')
                    container = series['metric'].get('container', 'unknown')
                    
                    if pod_name in pod_metrics and container in pod_metrics[pod_name]['containers']:
                        if use_time_range:
                            write_mbps_values = [float(v) / (1024 * 1024) for t, v in series['values']]
                            write_mbps_stats = self.calculate_stats(write_mbps_values) if write_mbps_values else {'min': 0, 'max': 0, 'avg': 0}
                            pod_metrics[pod_name]['containers'][container]['write_mbps'] = write_mbps_stats
                        else:
                            write_mbps = float(series['value'][1]) / (1024 * 1024)
                            pod_metrics[pod_name]['containers'][container]['write_mbps'] = round(write_mbps, 2)
            
            # Convert to list format and calculate summary
            all_iops_stats = {'min': [], 'max': [], 'avg': []}
            all_throughput_stats = {'min': [], 'max': [], 'avg': []}
            
            for pod_name, pod_data in pod_metrics.items():
                pod_info = {
                    'pod_name': pod_name,
                    'containers': [],
                    'use_time_range': use_time_range
                }
                
                for container, container_data in pod_data['containers'].items():
                    if use_time_range:
                        # Calculate totals from stats
                        total_iops_stats = {
                            'min': container_data['reads_per_sec']['min'] + container_data['writes_per_sec']['min'],
                            'max': container_data['reads_per_sec']['max'] + container_data['writes_per_sec']['max'],
                            'avg': container_data['reads_per_sec']['avg'] + container_data['writes_per_sec']['avg']
                        }
                        total_throughput_stats = {
                            'min': container_data['read_mbps']['min'] + container_data['write_mbps']['min'],
                            'max': container_data['read_mbps']['max'] + container_data['write_mbps']['max'],
                            'avg': container_data['read_mbps']['avg'] + container_data['write_mbps']['avg']
                        }
                        
                        pod_info['containers'].append({
                            'container_name': container,
                            'reads_per_sec': container_data['reads_per_sec'],
                            'writes_per_sec': container_data['writes_per_sec'],
                            'total_iops': total_iops_stats,
                            'read_mbps': container_data['read_mbps'],
                            'write_mbps': container_data['write_mbps'],
                            'total_throughput_mbps': total_throughput_stats
                        })
                        
                        all_iops_stats['min'].append(total_iops_stats['min'])
                        all_iops_stats['max'].append(total_iops_stats['max'])
                        all_iops_stats['avg'].append(total_iops_stats['avg'])
                        all_throughput_stats['min'].append(total_throughput_stats['min'])
                        all_throughput_stats['max'].append(total_throughput_stats['max'])
                        all_throughput_stats['avg'].append(total_throughput_stats['avg'])
                    else:
                        # Single values
                        total_iops = container_data['reads_per_sec'] + container_data['writes_per_sec']
                        total_throughput = container_data['read_mbps'] + container_data['write_mbps']
                        
                        pod_info['containers'].append({
                            'container_name': container,
                            'reads_per_sec': container_data['reads_per_sec'],
                            'writes_per_sec': container_data['writes_per_sec'],
                            'total_iops': round(total_iops, 2),
                            'read_mbps': container_data['read_mbps'],
                            'write_mbps': container_data['write_mbps'],
                            'total_throughput_mbps': round(total_throughput, 2)
                        })
                        
                        all_iops_stats['avg'].append(total_iops)
                        all_throughput_stats['avg'].append(total_throughput)
                
                metrics['minio_pods'].append(pod_info)
            
            # Calculate summary statistics
            if use_time_range:
                if all_iops_stats['avg']:
                    metrics['summary']['iops'] = {
                        'total_avg': round(sum(all_iops_stats['avg']), 2),
                        'min': round(min(all_iops_stats['min']), 2),
                        'max': round(max(all_iops_stats['max']), 2),
                        'avg': round(sum(all_iops_stats['avg']) / len(all_iops_stats['avg']), 2)
                    }
                
                if all_throughput_stats['avg']:
                    metrics['summary']['throughput_mbps'] = {
                        'total_avg': round(sum(all_throughput_stats['avg']), 2),
                        'min': round(min(all_throughput_stats['min']), 2),
                        'max': round(max(all_throughput_stats['max']), 2),
                        'avg': round(sum(all_throughput_stats['avg']) / len(all_throughput_stats['avg']), 2)
                    }
            else:
                if all_iops_stats['avg']:
                    metrics['summary']['total_iops'] = round(sum(all_iops_stats['avg']), 2)
                    metrics['summary']['avg_iops'] = round(sum(all_iops_stats['avg']) / len(all_iops_stats['avg']), 2)
                
                if all_throughput_stats['avg']:
                    metrics['summary']['total_throughput_mbps'] = round(sum(all_throughput_stats['avg']), 2)
                    metrics['summary']['avg_throughput_mbps'] = round(sum(all_throughput_stats['avg']) / len(all_throughput_stats['avg']), 2)
                
        except Exception as e:
            self.logger.warning(f"Error collecting S3/MinIO metrics: {e}")
            metrics = {"minio_pods": [], "summary": {}}
        
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
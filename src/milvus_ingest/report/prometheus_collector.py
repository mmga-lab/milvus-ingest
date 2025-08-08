"""Prometheus data collector for Milvus import performance metrics."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from ..logging_config import get_logger
from .models import PrometheusMetricData, SystemMetrics, MilvusClusterMetrics, ReportConfig


class PrometheusDataCollector:
    """Collector for Prometheus metrics related to Milvus imports."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the Prometheus collector.
        
        Args:
            config: Report configuration containing Prometheus URL and other settings
        """
        self.config = config
        self.prometheus_url = config.prometheus_url.rstrip('/')
        self.logger = get_logger(__name__)
        
        # Cache for available metrics
        self._available_metrics: Optional[List[str]] = None
    
    def collect_import_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        step: str = '30s',
    ) -> List[PrometheusMetricData]:
        """Collect Milvus import-related metrics.
        
        Args:
            start_time: Start time for query range
            end_time: End time for query range  
            step: Query step size (e.g., '30s', '1m')
            
        Returns:
            List of Prometheus metric data points
        """
        # Use config defaults if not provided
        start_time = start_time or self.config.start_time
        end_time = end_time or self.config.end_time
        
        # Set default time range if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=self.config.duration_hours)
        
        self.logger.info(
            "Collecting Prometheus import metrics",
            extra={
                "start_time": start_time,
                "end_time": end_time,
                "step": step,
            }
        )
        
        # Define import-related metrics to collect
        import_metrics = [
            # Import job metrics
            "milvus_datacoord_import_jobs",
            "milvus_datacoord_import_tasks",
            "milvus_datacoord_import_job_latency_sum",
            "milvus_datacoord_import_task_latency_sum",
            "milvus_datacoord_bulk_insert_vectors_count",
            
            # Data ingestion metrics
            "milvus_proxy_insert_vectors_count",
            "milvus_datanode_consume_msg_count", 
            "milvus_datanode_consume_bytes_count",
            "milvus_datanode_flushed_data_rows",
            "milvus_datanode_flushed_data_size",
            
            # Storage growth metrics
            "milvus_datacoord_stored_rows_num",
            "milvus_datacoord_stored_binlog_size",
            "milvus_datacoord_stored_index_files_size",
            "milvus_datacoord_segment_num",
            "milvus_datacoord_collection_num",
            
            # Performance metrics
            "milvus_proxy_insert_latency_sum",
            "milvus_proxy_search_latency_sum",
            "milvus_datanode_save_latency_sum",
            "milvus_datacoord_compaction_latency_sum",
        ]
        
        all_metrics = []
        available_metrics = self._get_available_metrics()
        
        for metric_name in import_metrics:
            if metric_name in available_metrics:
                try:
                    metrics = self._query_metric_range(
                        metric_name, start_time, end_time, step
                    )
                    all_metrics.extend(metrics)
                    self.logger.debug(f"Collected {len(metrics)} data points for {metric_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to collect metric {metric_name}: {e}")
            else:
                self.logger.debug(f"Metric {metric_name} not available in Prometheus")
        
        self.logger.info(f"Collected {len(all_metrics)} total metric data points")
        return all_metrics
    
    def collect_system_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        step: str = '30s',
    ) -> List[SystemMetrics]:
        """Collect system resource metrics during import operations.
        
        Args:
            start_time: Start time for query range
            end_time: End time for query range
            step: Query step size
            
        Returns:
            List of system metrics data points
        """
        # Use config defaults if not provided
        start_time = start_time or self.config.start_time
        end_time = end_time or self.config.end_time
        
        # Set default time range if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=self.config.duration_hours)
        
        self.logger.info("Collecting system resource metrics")
        
        # System resource queries
        system_queries = {
            'cpu_usage': '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
            'memory_usage_percent': '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
            'memory_usage_bytes': 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes',
            'disk_io_read': 'rate(node_disk_read_bytes_total[5m])',
            'disk_io_write': 'rate(node_disk_written_bytes_total[5m])',
            'network_rx': 'rate(node_network_receive_bytes_total[5m])',
            'network_tx': 'rate(node_network_transmit_bytes_total[5m])',
        }
        
        # Collect data for each timestamp
        timestamps = self._get_time_range(start_time, end_time, step)
        system_metrics = []
        
        for timestamp in timestamps:
            metrics_data = {
                'timestamp': timestamp,
                'cpu_usage_percent': None,
                'memory_usage_bytes': None,
                'memory_usage_percent': None,
                'disk_io_read_bytes': None,
                'disk_io_write_bytes': None,
                'network_rx_bytes': None,
                'network_tx_bytes': None,
            }
            
            for metric_key, query in system_queries.items():
                try:
                    result = self._query_instant(query, timestamp)
                    if result:
                        # Take the first result or average if multiple
                        value = self._extract_single_value(result)
                        if metric_key == 'cpu_usage':
                            metrics_data['cpu_usage_percent'] = value
                        elif metric_key == 'memory_usage_percent':
                            metrics_data['memory_usage_percent'] = value
                        elif metric_key == 'memory_usage_bytes':
                            metrics_data['memory_usage_bytes'] = int(value) if value else None
                        elif metric_key == 'disk_io_read':
                            metrics_data['disk_io_read_bytes'] = int(value) if value else None
                        elif metric_key == 'disk_io_write':
                            metrics_data['disk_io_write_bytes'] = int(value) if value else None
                        elif metric_key == 'network_rx':
                            metrics_data['network_rx_bytes'] = int(value) if value else None
                        elif metric_key == 'network_tx':
                            metrics_data['network_tx_bytes'] = int(value) if value else None
                            
                except Exception as e:
                    self.logger.warning(f"Failed to collect {metric_key}: {e}")
            
            system_metrics.append(SystemMetrics(**metrics_data))
        
        return system_metrics
    
    def collect_milvus_cluster_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        step: str = '30s',
    ) -> List[MilvusClusterMetrics]:
        """Collect Milvus cluster-level metrics.
        
        Args:
            start_time: Start time for query range
            end_time: End time for query range
            step: Query step size
            
        Returns:
            List of Milvus cluster metrics
        """
        # Use config defaults if not provided
        start_time = start_time or self.config.start_time
        end_time = end_time or self.config.end_time
        
        # Set default time range if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=self.config.duration_hours)
        
        self.logger.info("Collecting Milvus cluster metrics")
        
        # Milvus aggregate queries
        milvus_queries = {
            'import_jobs_total': 'sum(milvus_datacoord_import_jobs)',
            'import_tasks_total': 'sum(milvus_datacoord_import_tasks)',
            'insert_vectors_count': 'sum(milvus_proxy_insert_vectors_count)',
            'stored_binlog_size_bytes': 'sum(milvus_datacoord_stored_binlog_size)',
            'stored_rows_total': 'sum(milvus_datacoord_stored_rows_num)',
            'segment_count': 'sum(milvus_datacoord_segment_num)',
            'collection_count': 'sum(milvus_datacoord_collection_num)',
            'import_job_latency_ms': 'avg(rate(milvus_datacoord_import_job_latency_sum[5m]))',
            'import_task_latency_ms': 'avg(rate(milvus_datacoord_import_task_latency_sum[5m]))',
        }
        
        # Collect data for each timestamp
        timestamps = self._get_time_range(start_time, end_time, step)
        cluster_metrics = []
        
        for timestamp in timestamps:
            metrics_data = {
                'timestamp': timestamp,
                'import_jobs_total': 0,
                'import_tasks_total': 0,
                'insert_vectors_count': 0,
                'stored_binlog_size_bytes': 0,
                'stored_rows_total': 0,
                'segment_count': 0,
                'collection_count': 0,
                'import_job_latency_ms': None,
                'import_task_latency_ms': None,
            }
            
            for metric_key, query in milvus_queries.items():
                try:
                    result = self._query_instant(query, timestamp)
                    if result:
                        value = self._extract_single_value(result)
                        if value is not None:
                            if metric_key.endswith('_ms'):
                                metrics_data[metric_key] = float(value) * 1000  # Convert to ms
                            elif metric_key in ['import_jobs_total', 'import_tasks_total', 'insert_vectors_count', 
                                              'stored_rows_total', 'segment_count', 'collection_count']:
                                metrics_data[metric_key] = int(value)
                            else:
                                metrics_data[metric_key] = int(value)
                                
                except Exception as e:
                    self.logger.warning(f"Failed to collect {metric_key}: {e}")
            
            cluster_metrics.append(MilvusClusterMetrics(**metrics_data))
        
        return cluster_metrics
    
    def _get_available_metrics(self) -> List[str]:
        """Get list of available metrics from Prometheus."""
        if self._available_metrics is not None:
            return self._available_metrics
        
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/label/__name__/values",
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success':
                self._available_metrics = data['data']
                return self._available_metrics
            else:
                raise Exception(f"Failed to get metrics: {data}")
                
        except Exception as e:
            self.logger.error(f"Failed to get available metrics: {e}")
            return []
    
    def _query_metric_range(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        step: str = '30s',
    ) -> List[PrometheusMetricData]:
        """Query metric data over a time range."""
        params = {
            'query': metric_name,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step,
        }
        
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success':
                return self._parse_prometheus_data(data['data']['result'], metric_name)
            else:
                raise Exception(f"Prometheus query failed: {data}")
                
        except Exception as e:
            raise Exception(f"Failed to query {metric_name}: {e}")
    
    def _query_instant(self, query: str, time: datetime) -> Optional[List[Dict]]:
        """Query Prometheus for instant value at specific time."""
        params = {
            'query': query,
            'time': time.timestamp(),
        }
        
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success':
                return data['data']['result']
            else:
                return None
                
        except Exception:
            return None
    
    def _parse_prometheus_data(
        self, 
        results: List[Dict], 
        metric_name: str
    ) -> List[PrometheusMetricData]:
        """Parse Prometheus query results into PrometheusMetricData objects."""
        parsed_data = []
        
        for result in results:
            labels = {k: v for k, v in result['metric'].items() if k != '__name__'}
            
            # Handle different result types
            if 'values' in result:
                # Range query result
                for timestamp_val, value in result['values']:
                    timestamp = datetime.fromtimestamp(float(timestamp_val))
                    parsed_data.append(PrometheusMetricData(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        value=float(value),
                        labels=labels,
                    ))
            elif 'value' in result:
                # Instant query result
                timestamp_val, value = result['value']
                timestamp = datetime.fromtimestamp(float(timestamp_val))
                parsed_data.append(PrometheusMetricData(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=float(value),
                    labels=labels,
                ))
        
        return parsed_data
    
    def _extract_single_value(self, results: List[Dict]) -> Optional[float]:
        """Extract a single aggregated value from Prometheus results."""
        if not results:
            return None
        
        # If multiple results, take the first one or average them
        if len(results) == 1:
            return float(results[0]['value'][1])
        else:
            # Average multiple values
            values = [float(r['value'][1]) for r in results]
            return sum(values) / len(values)
    
    def _get_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        step: str
    ) -> List[datetime]:
        """Generate list of timestamps for the given range and step."""
        # Convert step to seconds
        step_seconds = self._parse_step_to_seconds(step)
        
        timestamps = []
        current = start_time
        
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(seconds=step_seconds)
        
        return timestamps
    
    def _parse_step_to_seconds(self, step: str) -> int:
        """Parse step string (e.g., '30s', '1m', '5m') to seconds."""
        if step.endswith('s'):
            return int(step[:-1])
        elif step.endswith('m'):
            return int(step[:-1]) * 60
        elif step.endswith('h'):
            return int(step[:-1]) * 3600
        else:
            # Default to seconds
            return int(step)
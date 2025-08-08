"""Report generator for import analysis with Prometheus metrics."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ..logging_config import get_logger
from .models import ReportConfig
from .loki_collector import LokiDataCollector
from .prometheus_collector import PrometheusCollector


class ReportGenerator:
    """Report generator for import performance analysis with Prometheus metrics."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the report generator."""
        self.config = config
        self.logger = get_logger(__name__)
        self.loki_collector = LokiDataCollector(config)
        self.prometheus_collector = PrometheusCollector(config)
    
    def generate_report(
        self,
        job_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_file: str = "/tmp/import_summary.csv",
        test_scenario: Optional[str] = None,
        notes: Optional[str] = None,
        release_name: Optional[str] = None,
        milvus_namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate import performance report.
        
        Args:
            job_ids: List of job IDs for a single import (can be multiple jobs)
            collection_name: Collection name
            start_time: Start time for analysis
            end_time: End time for analysis
            output_file: Output CSV file path
            
        Returns:
            Dictionary with import summary
        """
        self.logger.info(f"Generating import performance report for jobs: {job_ids}")
        
        # Collect timing data from Loki
        all_timing_data = []
        job_summaries = {}
        
        if job_ids:
            # Collect data for specific job IDs
            for job_id in job_ids:
                self.logger.info(f"Collecting data for job ID: {job_id}")
                timing_data = self.loki_collector.collect_import_timing_data(
                    job_id=job_id,
                    collection_name=collection_name,
                    start_time=start_time,
                    end_time=end_time,
                )
                all_timing_data.extend(timing_data)
                
                # Summarize per job
                job_summaries[job_id] = self._summarize_job(timing_data)
        else:
            # Collect all import data in time range
            timing_data = self.loki_collector.collect_import_timing_data(
                collection_name=collection_name,
                start_time=start_time,
                end_time=end_time,
            )
            all_timing_data = timing_data
            
            # Group by job ID
            jobs_data = defaultdict(list)
            for data in timing_data:
                if data.job_id:
                    jobs_data[data.job_id].append(data)
            
            for job_id, job_data in jobs_data.items():
                job_summaries[job_id] = self._summarize_job(job_data)
        
        # Extract additional information from logs
        import_info = self._extract_import_info(all_timing_data)
        
        # Collect Prometheus metrics if release_name and namespace provided
        if release_name and milvus_namespace and self.config.prometheus_url:
            try:
                self.logger.info(f"Collecting Prometheus metrics for {release_name} in namespace {milvus_namespace}")
                
                # Collect DataNode metrics
                datanode_metrics = self.prometheus_collector.collect_datanode_metrics(
                    release_name=release_name,
                    namespace=milvus_namespace,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Collect S3/MinIO metrics
                s3_metrics = self.prometheus_collector.collect_s3_minio_metrics(
                    release_name=release_name,
                    namespace=milvus_namespace,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Collect binlog metrics
                binlog_metrics = self.prometheus_collector.collect_binlog_metrics(
                    release_name=release_name
                )
                
                # Add metrics to import_info
                import_info["prometheus_metrics"] = {
                    "datanode": datanode_metrics,
                    "s3_minio": s3_metrics,
                    "binlog": binlog_metrics,
                    "time_range_used": start_time and end_time
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to collect Prometheus metrics: {e}")
                import_info["prometheus_metrics"] = {}
        
        # Add user-provided test scenario and notes
        if test_scenario:
            import_info['test_scenario'] = test_scenario
        if notes:
            import_info['notes'] = notes
        
        # Generate CSV report
        self._write_csv_report(job_summaries, import_info, output_file)
        
        return {
            "jobs_analyzed": len(job_summaries),
            "total_import_time": sum(s.get("total_time", 0) for s in job_summaries.values()),
            "output_file": output_file,
            "job_summaries": job_summaries,
            "import_info": import_info
        }
    
    def _summarize_job(self, timing_data: List) -> Dict[str, Any]:
        """Summarize timing data for a single job."""
        if not timing_data:
            return {}
        
        summary = {
            "job_id": None,
            "collection_name": None,
            "start_time": None,
            "end_time": None,
            "phases": {},
            "total_time": 0
        }
        
        # Extract basic info
        for data in timing_data:
            if data.job_id:
                summary["job_id"] = data.job_id
            if data.collection_name:
                summary["collection_name"] = data.collection_name
            
            # Track phase timings
            if data.import_phase and data.time_cost:
                phase = data.import_phase
                # Convert to seconds for consistency
                time_in_seconds = self._convert_to_seconds(data.time_cost, data.time_unit)
                summary["phases"][phase] = time_in_seconds
                
                # Track start/end times
                if not summary["start_time"] or data.timestamp < summary["start_time"]:
                    summary["start_time"] = data.timestamp
                if not summary["end_time"] or data.timestamp > summary["end_time"]:
                    summary["end_time"] = data.timestamp
        
        # Calculate total time from phases
        if "all_completed" in summary["phases"]:
            summary["total_time"] = summary["phases"]["all_completed"]
        elif summary["phases"]:
            summary["total_time"] = sum(summary["phases"].values())
        
        return summary
    
    def _convert_to_seconds(self, time_value: float, time_unit: str) -> float:
        """Convert time value to seconds."""
        if not time_value:
            return 0.0
            
        if time_unit == "s":
            return time_value
        elif time_unit == "ms":
            return time_value / 1000.0
        elif time_unit == "µs" or time_unit == "us":
            return time_value / 1_000_000.0
        elif time_unit == "ns":
            return time_value / 1_000_000_000.0
        else:
            # Assume seconds if unknown
            return time_value
    
    def _extract_import_info(self, timing_data: List) -> Dict[str, Any]:
        """Extract detailed import information from log messages."""
        info = {
            "test_scenario": "",  # 测试场景名称
            "total_rows": 0,      # 总行数
            "file_type": "",      # 文件类型 (parquet, json, etc.)
            "import_result": "success",  # 导入结果
            "collection_schema": {},
            "file_info": {
                "file_count": 0,
                "file_sizes": [],
                "total_size": 0,
                "binlog_count": 0,    # Binlog文件数量
                "binlog_size": 0,     # Binlog总大小
            },
            "resource_settings": {
                "datanode_cpu": "",   # DataNode CPU设置
                "datanode_memory": "", # DataNode内存设置
            },
            "storage_metrics": {
                "s3_minio_iops": 0,       # S3/MinIO IOPS
                "s3_minio_throughput": 0, # S3/MinIO吞吐量 (MB/s)
            },
            "notes": ""  # 备注信息
        }
        
        # Parse log messages for additional information
        for data in timing_data:
            if not data.message:
                continue
                
            message = data.message
            
            # Extract total rows
            rows_match = re.search(r'(?:total[_\s]*)?rows?[:\s=]+(\d+)', message, re.IGNORECASE)
            if rows_match:
                info["total_rows"] = int(rows_match.group(1))
            
            # Extract file type
            if "parquet" in message.lower():
                info["file_type"] = "parquet"
            elif "json" in message.lower():
                info["file_type"] = "json"
            elif "csv" in message.lower():
                info["file_type"] = "csv"
            elif "numpy" in message.lower() or ".npy" in message.lower():
                info["file_type"] = "numpy"
            
            # Extract file information
            file_match = re.search(r'file[_\s]*count[:\s=]+(\d+)', message, re.IGNORECASE)
            if file_match:
                info["file_info"]["file_count"] = int(file_match.group(1))
            
            size_match = re.search(r'file[_\s]*size[:\s=]+([\d.]+)\s*([KMGT]?B)', message, re.IGNORECASE)
            if size_match:
                size = float(size_match.group(1))
                unit = size_match.group(2)
                size_bytes = self._convert_to_bytes(size, unit)
                info["file_info"]["file_sizes"].append(size_bytes)
                info["file_info"]["total_size"] += size_bytes
            
            # Extract binlog information
            binlog_count_match = re.search(r'binlog[_\s]*count[:\s=]+(\d+)', message, re.IGNORECASE)
            if binlog_count_match:
                info["file_info"]["binlog_count"] = int(binlog_count_match.group(1))
            
            binlog_size_match = re.search(r'binlog[_\s]*size[:\s=]+([\d.]+)\s*([KMGT]?B)', message, re.IGNORECASE)
            if binlog_size_match:
                size = float(binlog_size_match.group(1))
                unit = binlog_size_match.group(2)
                info["file_info"]["binlog_size"] = self._convert_to_bytes(size, unit)
            
            # Extract DataNode resource settings
            if "datanode" in message.lower():
                cpu_match = re.search(r'cpu[:\s=]+([\d.]+)', message, re.IGNORECASE)
                if cpu_match:
                    info["resource_settings"]["datanode_cpu"] = cpu_match.group(1)
                
                mem_match = re.search(r'memory[:\s=]+([\d.]+)\s*([KMGT]?[iB]+)?', message, re.IGNORECASE)
                if mem_match:
                    info["resource_settings"]["datanode_memory"] = f"{mem_match.group(1)}{mem_match.group(2) or ''}"
            
            # Extract S3/MinIO metrics
            iops_match = re.search(r'iops[:\s=]+([\d.]+)', message, re.IGNORECASE)
            if iops_match:
                info["storage_metrics"]["s3_minio_iops"] = float(iops_match.group(1))
            
            throughput_match = re.search(r'throughput[:\s=]+([\d.]+)\s*([KMGT]?B/s)?', message, re.IGNORECASE)
            if throughput_match:
                throughput = float(throughput_match.group(1))
                unit = throughput_match.group(2) or "MB/s"
                # Convert to MB/s for consistency
                if "GB/s" in unit:
                    throughput *= 1024
                elif "KB/s" in unit:
                    throughput /= 1024
                info["storage_metrics"]["s3_minio_throughput"] = throughput
            
            # Check for errors or failures
            if "error" in message.lower() or "failed" in message.lower():
                if "import" in message.lower():
                    info["import_result"] = "failed"
            
            # Try to extract collection schema info
            schema_match = re.search(r'schema[:\s]+({.*?})', message, re.IGNORECASE)
            if schema_match:
                try:
                    import json
                    info["collection_schema"] = json.loads(schema_match.group(1))
                except:
                    pass
        
        return info
    
    def _convert_to_bytes(self, size: float, unit: str) -> float:
        """Convert size to bytes."""
        unit = unit.upper()
        if unit == "B":
            return size
        elif unit == "KB":
            return size * 1024
        elif unit == "MB":
            return size * 1024 * 1024
        elif unit == "GB":
            return size * 1024 * 1024 * 1024
        elif unit == "TB":
            return size * 1024 * 1024 * 1024 * 1024
        else:
            return size
    
    def _write_csv_report(self, job_summaries: Dict, import_info: Dict, output_file: str):
        """Write CSV report with performance metrics."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get Prometheus metrics if available
        prometheus_metrics = import_info.get('prometheus_metrics', {})
        datanode_metrics = prometheus_metrics.get('datanode', {})
        s3_metrics = prometheus_metrics.get('s3_minio', {})
        binlog_metrics = prometheus_metrics.get('binlog', {})
        is_time_range = prometheus_metrics.get('time_range_used', False)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'Test Scenario',         # 测试场景
                'Job ID',
                'Collection Name',
                'Total Rows',            # 总行数
                'File Type',             # 文件类型
                'Import Result',         # 导入结果
                'Total Time (s)',        # 总耗时
                'Pending Time (s)',      # 各阶段耗时
                'Pre-Import Time (s)',
                'Import Time (s)',
                'Stats Time (s)',
                'Build Index Time (s)',
                'L0 Import Time (s)',
                'DataNode Memory',       # DataNode内存
                'DataNode CPU',          # DataNode CPU
                'S3/MinIO IOPS',        # S3/MinIO IOPS
                'S3/MinIO Throughput (MB/s)',  # S3/MinIO吞吐量
                'File Count',
                'Total File Size (GB)',
                'Binlog Count',          # Binlog数量
                'Binlog Size (GB)',      # Binlog大小
                'Notes'                  # 备注
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for job_id, summary in job_summaries.items():
                # Format DataNode memory
                datanode_memory = self._format_prometheus_metric(
                    datanode_metrics.get('datanode_memory_gb', 0), 
                    is_time_range, 
                    unit='GB'
                )
                
                # Format DataNode CPU
                datanode_cpu = self._format_prometheus_metric(
                    datanode_metrics.get('datanode_cpu_cores', 0), 
                    is_time_range, 
                    unit='cores'
                )
                
                # Format S3/MinIO IOPS
                s3_iops = self._format_prometheus_metric(
                    s3_metrics.get('s3_minio_iops', 0), 
                    is_time_range, 
                    unit='req/s'
                )
                
                # Format S3/MinIO throughput
                s3_throughput = self._format_prometheus_metric(
                    s3_metrics.get('s3_minio_throughput_mbps', 0), 
                    is_time_range, 
                    unit='MB/s'
                )
                
                row = {
                    'Test Scenario': import_info.get('test_scenario', ''),
                    'Job ID': job_id,
                    'Collection Name': summary.get('collection_name', ''),
                    'Total Rows': import_info.get('total_rows', 0),
                    'File Type': import_info.get('file_type', ''),
                    'Import Result': import_info.get('import_result', 'success'),
                    'Total Time (s)': f"{summary.get('total_time', 0):.2f}",
                    'Pending Time (s)': f"{summary.get('phases', {}).get('start_to_execute', 0):.2f}",
                    'Pre-Import Time (s)': f"{summary.get('phases', {}).get('preimport_done', 0):.2f}",
                    'Import Time (s)': f"{summary.get('phases', {}).get('import_done', 0):.2f}",
                    'Stats Time (s)': f"{summary.get('phases', {}).get('stats_done', 0):.2f}",
                    'Build Index Time (s)': f"{summary.get('phases', {}).get('build_index_done', 0):.2f}",
                    'L0 Import Time (s)': f"{summary.get('phases', {}).get('l0_import_done', 0):.6f}",
                    'DataNode Memory': datanode_memory,
                    'DataNode CPU': datanode_cpu,
                    'S3/MinIO IOPS': s3_iops,
                    'S3/MinIO Throughput (MB/s)': s3_throughput,
                    'File Count': import_info['file_info']['file_count'],
                    'Total File Size (GB)': f"{import_info['file_info']['total_size'] / (1024**3):.2f}",
                    'Binlog Count': binlog_metrics.get('binlog_count', import_info['file_info'].get('binlog_count', 0)),
                    'Binlog Size (GB)': f"{binlog_metrics.get('binlog_size_gb', import_info['file_info'].get('binlog_size', 0) / (1024**3)):.2f}",
                    'Notes': import_info.get('notes', '')
                }
                writer.writerow(row)
        
        self.logger.info(f"CSV report written to {output_path}")
    
    def _format_prometheus_metric(self, metric_value, is_time_range: bool, unit: str = '') -> str:
        """Format Prometheus metric values for CSV output."""
        if not metric_value:
            return ""
        
        if is_time_range and isinstance(metric_value, dict):
            # Time range format: "min/avg/max unit"
            min_val = metric_value.get('min', 0)
            avg_val = metric_value.get('avg', 0)
            max_val = metric_value.get('max', 0)
            
            if unit:
                return f"{min_val:.2f}/{avg_val:.2f}/{max_val:.2f} {unit}"
            else:
                return f"{min_val:.2f}/{avg_val:.2f}/{max_val:.2f}"
        else:
            # Single value format
            if isinstance(metric_value, (int, float)):
                if unit:
                    return f"{metric_value:.2f} {unit}"
                else:
                    return f"{metric_value:.2f}"
            else:
                return str(metric_value)
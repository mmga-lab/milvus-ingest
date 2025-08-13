"""Report generator for import analysis with Prometheus metrics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..logging_config import get_logger
from .loki_collector import LokiDataCollector
from .models import ReportConfig
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
        job_ids: list[str] | None = None,
        collection_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        output_file: str = "/tmp/import_analysis.md",
        output_format: str = "analysis",
        test_scenario: str | None = None,
        notes: str | None = None,
        release_name: str | None = None,
        milvus_namespace: str | None = None,
        import_info_file: str | None = None,
        glm_api_key: str | None = None,
        glm_model: str = "glm-4-flash",
    ) -> dict[str, Any]:
        """Generate import performance report.

        Args:
            job_ids: List of job IDs for a single import (can be multiple jobs)
            collection_name: Collection name
            start_time: Start time for analysis
            end_time: End time for analysis
            output_file: Output file path
            output_format: Output format ('analysis' for GLM analysis or 'raw' for raw data)
            test_scenario: Test scenario description
            notes: Additional notes
            release_name: Milvus release name
            milvus_namespace: Milvus namespace
            import_info_file: Path to import_info.json file
            glm_api_key: GLM API key for analysis
            glm_model: GLM model to use (default: glm-4-flash)

        Returns:
            Dictionary with import summary
        """
        self.logger.info(f"Generating import performance report for jobs: {job_ids}")

        # Collect timing data from Loki
        all_timing_data = []
        job_summaries = {}

        if job_ids:
            # Collect raw logs for specific job IDs
            for job_id in job_ids:
                self.logger.info(f"Collecting logs for job ID: {job_id}")
                raw_logs = self.loki_collector.collect_raw_logs(
                    job_id=job_id,
                    collection_name=collection_name,
                    start_time=start_time,
                    end_time=end_time,
                )
                all_timing_data.append(raw_logs)
        else:
            # Collect all logs in time range
            raw_logs = self.loki_collector.collect_raw_logs(
                collection_name=collection_name,
                start_time=start_time,
                end_time=end_time,
            )
            all_timing_data = [raw_logs]

        # Initialize import_info with basic structure
        import_info = {}

        # Load additional metadata from import_info.json if provided
        import_metadata = {}
        if import_info_file:
            import_metadata = self._load_import_metadata(import_info_file)
            if import_metadata:
                # Merge metadata into import_info
                import_info.update(import_metadata)

                # If job_ids were not explicitly provided, extract them from import_info.json
                if not job_ids and "job_ids" in import_metadata:
                    job_ids = import_metadata["job_ids"]
                    self.logger.info(f"Using job IDs from import_info.json: {job_ids}")

                    # Re-collect data for the specific job IDs from import_info.json
                    all_timing_data = []
                    job_summaries = {}

                    for job_id in job_ids:
                        self.logger.info(
                            f"Collecting logs for job ID from import_info.json: {job_id}"
                        )
                        raw_logs = self.loki_collector.collect_raw_logs(
                            job_id=job_id,
                            collection_name=collection_name,
                            start_time=start_time,
                            end_time=end_time,
                        )
                        all_timing_data.append(raw_logs)

                    # Keep import_info as is since GLM will process raw data directly

        # Collect Prometheus metrics if release_name and namespace provided
        if release_name and milvus_namespace and self.config.prometheus_url:
            try:
                self.logger.info(
                    f"Collecting Prometheus raw metrics for {release_name} in namespace {milvus_namespace}"
                )

                # Collect raw Prometheus metrics
                prometheus_metrics = self.prometheus_collector.collect_raw_metrics(
                    release_name=release_name,
                    namespace=milvus_namespace,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Add metrics to import_info
                import_info["prometheus_metrics"] = prometheus_metrics

            except Exception as e:
                self.logger.warning(f"Failed to collect Prometheus metrics: {e}")
                import_info["prometheus_metrics"] = {}

        # Add user-provided test scenario and notes
        if test_scenario:
            import_info["test_scenario"] = test_scenario
        if notes:
            import_info["notes"] = notes

        # Generate report based on format (removed CSV, now use ReportAnalyzer)
        from .llm_generator import ReportAnalyzer
        analyzer = ReportAnalyzer(self.config, glm_api_key=glm_api_key, glm_model=glm_model)
        
        return analyzer.generate_report(
            job_ids=job_ids,
            collection_name=collection_name,
            start_time=start_time,
            end_time=end_time,
            output_file=output_file,
            output_format=output_format,
            test_scenario=test_scenario,
            notes=notes,
            release_name=release_name,
            milvus_namespace=milvus_namespace,
            import_info_file=import_info_file,
        )

    def _summarize_job(self, timing_data: list) -> dict[str, Any]:
        """Summarize timing data for a single job."""
        if not timing_data:
            return {}

        summary = {
            "job_id": None,
            "collection_name": None,
            "start_time": None,
            "end_time": None,
            "phases": {},
            "total_time": 0,
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
                time_in_seconds = self._convert_to_seconds(
                    data.time_cost, data.time_unit
                )
                summary["phases"][phase] = time_in_seconds

                # Track start/end times
                if not summary["start_time"] or data.timestamp < summary["start_time"]:
                    summary["start_time"] = data.timestamp
                if not summary["end_time"] or data.timestamp > summary["end_time"]:
                    summary["end_time"] = data.timestamp

        # Calculate total time from phases
        # "all_completed" represents the total import time from start to finish
        if "all_completed" in summary["phases"]:
            summary["total_time"] = summary["phases"]["all_completed"]
        elif summary["phases"]:
            # Fallback: sum all phases if no "all_completed" phase found
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

    def _extract_import_info(self, timing_data: list) -> dict[str, Any]:
        """Extract detailed import information from log messages."""
        info = {
            "test_scenario": "",  # 测试场景名称
            "total_rows": 0,  # 总行数
            "file_type": "",  # 文件类型 (parquet, json, etc.)
            "import_result": "success",  # 导入结果
            "collection_schema": {},
            "file_info": {
                "file_count": 0,
                "file_sizes": [],
                "total_size": 0,
                "binlog_count": 0,  # Binlog文件数量
                "binlog_size": 0,  # Binlog总大小
            },
            "resource_settings": {
                "datanode_cpu": "",  # DataNode CPU设置
                "datanode_memory": "",  # DataNode内存设置
            },
            "storage_metrics": {
                "s3_minio_iops": 0,  # S3/MinIO IOPS
                "s3_minio_throughput": 0,  # S3/MinIO吞吐量 (MB/s)
            },
            "notes": "",  # 备注信息
        }

        # Parse log messages for additional information
        for data in timing_data:
            if not data.message:
                continue

            message = data.message

            # Extract total rows
            rows_match = re.search(
                r"(?:total[_\s]*)?rows?[:\s=]+(\d+)", message, re.IGNORECASE
            )
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
            file_match = re.search(
                r"file[_\s]*count[:\s=]+(\d+)", message, re.IGNORECASE
            )
            if file_match:
                info["file_info"]["file_count"] = int(file_match.group(1))

            size_match = re.search(
                r"file[_\s]*size[:\s=]+([\d.]+)\s*([KMGT]?B)", message, re.IGNORECASE
            )
            if size_match:
                size = float(size_match.group(1))
                unit = size_match.group(2)
                size_bytes = self._convert_to_bytes(size, unit)
                info["file_info"]["file_sizes"].append(size_bytes)
                info["file_info"]["total_size"] += size_bytes

            # Extract binlog information
            binlog_count_match = re.search(
                r"binlog[_\s]*count[:\s=]+(\d+)", message, re.IGNORECASE
            )
            if binlog_count_match:
                info["file_info"]["binlog_count"] = int(binlog_count_match.group(1))

            binlog_size_match = re.search(
                r"binlog[_\s]*size[:\s=]+([\d.]+)\s*([KMGT]?B)", message, re.IGNORECASE
            )
            if binlog_size_match:
                size = float(binlog_size_match.group(1))
                unit = binlog_size_match.group(2)
                info["file_info"]["binlog_size"] = self._convert_to_bytes(size, unit)

            # Extract DataNode resource settings
            if "datanode" in message.lower():
                cpu_match = re.search(r"cpu[:\s=]+([\d.]+)", message, re.IGNORECASE)
                if cpu_match:
                    info["resource_settings"]["datanode_cpu"] = cpu_match.group(1)

                mem_match = re.search(
                    r"memory[:\s=]+([\d.]+)\s*([KMGT]?[iB]+)?", message, re.IGNORECASE
                )
                if mem_match:
                    info["resource_settings"]["datanode_memory"] = (
                        f"{mem_match.group(1)}{mem_match.group(2) or ''}"
                    )

            # Extract S3/MinIO metrics
            iops_match = re.search(r"iops[:\s=]+([\d.]+)", message, re.IGNORECASE)
            if iops_match:
                info["storage_metrics"]["s3_minio_iops"] = float(iops_match.group(1))

            throughput_match = re.search(
                r"throughput[:\s=]+([\d.]+)\s*([KMGT]?B/s)?", message, re.IGNORECASE
            )
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
            if (
                "error" in message.lower() or "failed" in message.lower()
            ) and "import" in message.lower():
                info["import_result"] = "failed"

            # Try to extract collection schema info
            schema_match = re.search(r"schema[:\s]+({.*?})", message, re.IGNORECASE)
            if schema_match:
                try:
                    info["collection_schema"] = json.loads(schema_match.group(1))
                except json.JSONDecodeError:
                    pass

        return info

    def _load_import_metadata(self, import_info_file: str) -> dict[str, Any]:
        """Load metadata from import_info.json file."""
        try:
            with open(import_info_file) as f:
                metadata = json.load(f)

            # Extract relevant fields for the report
            extracted_info = {}

            # Extract job_ids directly from import_info.json
            if "job_ids" in metadata:
                extracted_info["job_ids"] = metadata["job_ids"]

            # Extract total_rows directly from import_info.json (if available)
            if "total_rows" in metadata:
                extracted_info["total_rows"] = metadata["total_rows"]

            # Extract file_info directly from import_info.json (if available)
            if "file_info" in metadata:
                file_info = metadata["file_info"]
                extracted_info["file_info"] = {
                    "file_count": file_info.get(
                        "file_count", metadata.get("total_files", 0)
                    ),
                    "total_size": file_info.get("total_size_bytes", 0),
                    "file_sizes": file_info.get("file_sizes", []),
                    "binlog_count": 0,
                    "binlog_size": 0,
                }
            else:
                # Fallback to basic file count from import_info.json
                extracted_info["file_info"] = {
                    "file_count": metadata.get("total_files", 0),
                    "total_size": 0,
                    "file_sizes": [],
                    "binlog_count": 0,
                    "binlog_size": 0,
                }

            if "file_types" in metadata:
                extracted_info["file_type"] = ", ".join(metadata["file_types"])

            # Schema information - use collection name as schema info
            if "collection_name" in metadata:
                extracted_info["collection_schema"] = {
                    "name": metadata["collection_name"]
                }
                # Also preserve collection_name for backward compatibility
                extracted_info["collection_name"] = metadata["collection_name"]

            self.logger.info(
                f"Loaded import metadata from import_info.json: {extracted_info}"
            )
            return extracted_info

        except Exception as e:
            self.logger.warning(
                f"Failed to load import metadata from {import_info_file}: {e}"
            )
            return {}

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


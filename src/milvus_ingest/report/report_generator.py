"""Report generator for import analysis with Prometheus metrics."""

from __future__ import annotations

import csv
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
        output_file: str = "/tmp/import_summary.csv",
        output_format: str = "csv",
        test_scenario: str | None = None,
        notes: str | None = None,
        release_name: str | None = None,
        milvus_namespace: str | None = None,
        import_info_file: str | None = None,
    ) -> dict[str, Any]:
        """Generate import performance report.

        Args:
            job_ids: List of job IDs for a single import (can be multiple jobs)
            collection_name: Collection name
            start_time: Start time for analysis
            end_time: End time for analysis
            output_file: Output file path
            output_format: Output format ('csv' or 'llm')

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
                            f"Collecting data for job ID from import_info.json: {job_id}"
                        )
                        timing_data = self.loki_collector.collect_import_timing_data(
                            job_id=job_id,
                            collection_name=collection_name,
                            start_time=start_time,
                            end_time=end_time,
                        )
                        all_timing_data.extend(timing_data)

                        # Summarize per job
                        job_summaries[job_id] = self._summarize_job(timing_data)

                    # Re-extract import info with the filtered data, but preserve important metadata
                    log_extracted_info = self._extract_import_info(all_timing_data)

                    # Preserve important data from import_info.json/meta.json that shouldn't be overwritten by log parsing
                    preserved_fields = [
                        "total_rows",
                        "file_info",
                        "collection_schema",
                        "collection_name",
                        "file_type",
                    ]
                    temp_preserved = {}
                    for field in preserved_fields:
                        if field in import_info and import_info[field]:
                            # For file_info, preserve non-zero values
                            if field == "file_info":
                                if (
                                    import_info[field].get("total_size", 0) > 0
                                    or import_info[field].get("file_count", 0) > 0
                                ):
                                    temp_preserved[field] = import_info[field]
                            # For total_rows, preserve non-zero values
                            elif field == "total_rows":
                                if import_info[field] > 0:
                                    temp_preserved[field] = import_info[field]
                            # For other fields, always preserve if they exist
                            else:
                                temp_preserved[field] = import_info[field]

                    # Update with log extracted info first
                    import_info.update(log_extracted_info)

                    # Then restore preserved metadata
                    import_info.update(temp_preserved)

        # Collect Prometheus metrics if release_name and namespace provided
        if release_name and milvus_namespace and self.config.prometheus_url:
            try:
                self.logger.info(
                    f"Collecting Prometheus metrics for {release_name} in namespace {milvus_namespace}"
                )

                # Auto-detect deployment mode if configured to do so
                if self.config.deployment_mode == "auto":
                    detected_mode = self.prometheus_collector.detect_deployment_mode(
                        release_name, milvus_namespace
                    )
                    self.logger.info(f"Auto-detected deployment mode: {detected_mode}")

                # Collect DataNode metrics (works for both cluster and standalone)
                datanode_metrics = self.prometheus_collector.collect_datanode_metrics(
                    release_name=release_name,
                    namespace=milvus_namespace,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Collect S3/MinIO metrics
                s3_metrics = self.prometheus_collector.collect_s3_minio_metrics(
                    release_name=release_name,
                    namespace=milvus_namespace,
                    start_time=start_time,
                    end_time=end_time,
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
                    "time_range_used": start_time and end_time,
                }

            except Exception as e:
                self.logger.warning(f"Failed to collect Prometheus metrics: {e}")
                import_info["prometheus_metrics"] = {}

        # Add user-provided test scenario and notes
        if test_scenario:
            import_info["test_scenario"] = test_scenario
        if notes:
            import_info["notes"] = notes

        # Generate report based on format
        if output_format.lower() == "llm":
            # Use LLM generator for llm.txt format
            from .llm_generator import LLMGenerator
            llm_gen = LLMGenerator(self.config)
            return llm_gen.generate_llm_context(
                job_ids=job_ids,
                collection_name=collection_name,
                start_time=start_time,
                end_time=end_time,
                output_file=output_file,
                test_scenario=test_scenario,
                notes=notes,
                release_name=release_name,
                milvus_namespace=milvus_namespace,
                import_info_file=import_info_file,
            )
        else:
            # Default to CSV format
            self._write_csv_report(job_summaries, import_info, output_file)

            return {
                "jobs_analyzed": len(job_summaries),
                "total_import_time": sum(
                    s.get("total_time", 0) for s in job_summaries.values()
                ),
                "output_file": output_file,
                "job_summaries": job_summaries,
                "import_info": import_info,
            }

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

    def _write_csv_report(
        self, job_summaries: dict, import_info: dict, output_file: str
    ):
        """Write CSV report with performance metrics."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get Prometheus metrics if available
        prometheus_metrics = import_info.get("prometheus_metrics", {})
        datanode_metrics = prometheus_metrics.get("datanode", {})
        s3_metrics = prometheus_metrics.get("s3_minio", {})
        binlog_metrics = prometheus_metrics.get("binlog", {})

        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "Test Scenario",  # 测试场景
                "Job ID",
                "Collection Name",
                "Total Rows",  # 总行数
                "File Type",  # 文件类型
                "Import Result",  # 导入结果
                "Total Time (s)",  # 总耗时
                "Pending Time (s)",  # 各阶段耗时
                "Pre-Import Time (s)",
                "Import Time (s)",
                "Stats Time (s)",
                "Build Index Time (s)",
                "L0 Import Time (s)",
                "Milvus Compute Pods Details",  # Milvus计算节点(DataNode/Standalone)每个pod详情
                "MinIO Pods Details",  # MinIO每个pod详情
                "File Count",
                "Total File Size (GB)",
                "Binlog Count",  # Binlog数量
                "Binlog Size (GB)",  # Binlog大小
                "Notes",  # 备注
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for job_id, summary in job_summaries.items():
                # Format DataNode pod details
                datanode_details = self._format_datanode_details(datanode_metrics)

                # Format MinIO pod details
                minio_details = self._format_minio_details(s3_metrics)

                # Use collection schema name if available, otherwise fall back to collection_name
                collection_display_name = ""
                if (
                    "collection_schema" in import_info
                    and "name" in import_info["collection_schema"]
                ):
                    collection_display_name = import_info["collection_schema"]["name"]
                elif "collection_name" in import_info:
                    collection_display_name = import_info["collection_name"]
                else:
                    collection_display_name = summary.get("collection_name", "")

                row = {
                    "Test Scenario": import_info.get("test_scenario", ""),
                    "Job ID": job_id,
                    "Collection Name": collection_display_name,
                    "Total Rows": import_info.get("total_rows", 0),
                    "File Type": import_info.get("file_type", ""),
                    "Import Result": import_info.get("import_result", "success"),
                    "Total Time (s)": f"{summary.get('total_time', 0):.2f}",
                    "Pending Time (s)": f"{summary.get('phases', {}).get('start_to_execute', 0):.2f}",
                    "Pre-Import Time (s)": f"{summary.get('phases', {}).get('preimport_done', 0):.2f}",
                    "Import Time (s)": f"{summary.get('phases', {}).get('import_done', 0):.2f}",
                    "Stats Time (s)": f"{summary.get('phases', {}).get('stats_done', 0):.2f}",
                    "Build Index Time (s)": f"{summary.get('phases', {}).get('build_index_done', 0):.2f}",
                    "L0 Import Time (s)": f"{summary.get('phases', {}).get('l0_import_done', 0):.6f}",
                    "Milvus Compute Pods Details": datanode_details,
                    "MinIO Pods Details": minio_details,
                    "File Count": import_info["file_info"]["file_count"],
                    "Total File Size (GB)": f"{import_info['file_info']['total_size'] / (1024**3):.2f}",
                    "Binlog Count": binlog_metrics.get(
                        "binlog_count", import_info["file_info"].get("binlog_count", 0)
                    ),
                    "Binlog Size (GB)": f"{binlog_metrics.get('binlog_size_gb', import_info['file_info'].get('binlog_size', 0) / (1024**3)):.2f}",
                    "Notes": import_info.get("notes", ""),
                }
                writer.writerow(row)

        self.logger.info(f"CSV report written to {output_path}")

    def _format_datanode_details(self, datanode_metrics: dict[str, Any]) -> str:
        """Format Milvus compute nodes (DataNode/Standalone) pod details for CSV display."""
        if not datanode_metrics or not datanode_metrics.get("datanode_pods"):
            return ""

        details = []
        use_time_range = datanode_metrics.get("datanode_pods", [{}])[0].get(
            "use_time_range", False
        )

        for pod in datanode_metrics["datanode_pods"]:
            pod_name = pod["pod_name"]
            pod_details = []

            for container in pod["containers"]:
                if use_time_range and isinstance(container["memory_gb"], dict):
                    # Time range format: min/avg/max with clear labels
                    memory_str = f"mem(min:{container['memory_gb']['min']:.2f}/avg:{container['memory_gb']['avg']:.2f}/max:{container['memory_gb']['max']:.2f}GB)"
                    cpu_str = f"cpu(min:{container['cpu_cores']['min']:.3f}/avg:{container['cpu_cores']['avg']:.3f}/max:{container['cpu_cores']['max']:.3f}cores)"
                    container_info = (
                        f"{container['container_name']}: {memory_str}, {cpu_str}"
                    )
                else:
                    # Single value format
                    container_info = f"{container['container_name']}: {container['memory_gb']:.2f}GB, {container['cpu_cores']:.3f}cores"
                pod_details.append(container_info)

            pod_summary = f"{pod_name}[{'; '.join(pod_details)}]"
            details.append(pod_summary)

        # Add summary
        summary = datanode_metrics.get("summary", {})
        if summary:
            if (
                use_time_range
                and "memory_gb" in summary
                and isinstance(summary["memory_gb"], dict)
            ):
                memory_summary = f"mem(min:{summary['memory_gb']['min']:.2f}/avg:{summary['memory_gb']['avg']:.2f}/max:{summary['memory_gb']['max']:.2f}GB)"
                cpu_summary = f"cpu(min:{summary['cpu_cores']['min']:.3f}/avg:{summary['cpu_cores']['avg']:.3f}/max:{summary['cpu_cores']['max']:.3f}cores)"
                summary_str = f"Summary: {memory_summary}, {cpu_summary}"
            else:
                summary_str = f"Summary: {summary.get('total_memory_gb', 0):.2f}GB, {summary.get('total_cpu_cores', 0):.3f}cores"
            details.append(summary_str)

        return " | ".join(details)

    def _format_minio_details(self, s3_metrics: dict[str, Any]) -> str:
        """Format MinIO pod details for CSV display."""
        if not s3_metrics or not s3_metrics.get("minio_pods"):
            return ""

        details = []
        use_time_range = s3_metrics.get("minio_pods", [{}])[0].get(
            "use_time_range", False
        )

        for pod in s3_metrics["minio_pods"]:
            pod_name = pod["pod_name"]
            pod_details = []

            for container in pod["containers"]:
                if use_time_range and isinstance(container["total_iops"], dict):
                    # Time range format: min/avg/max with clear labels
                    iops_str = f"iops(min:{container['total_iops']['min']:.1f}/avg:{container['total_iops']['avg']:.1f}/max:{container['total_iops']['max']:.1f})"
                    throughput_str = f"throughput(min:{container['total_throughput_mbps']['min']:.1f}/avg:{container['total_throughput_mbps']['avg']:.1f}/max:{container['total_throughput_mbps']['max']:.1f}MB/s)"
                    container_info = (
                        f"{container['container_name']}: {iops_str}, {throughput_str}"
                    )
                else:
                    # Single value format
                    container_info = f"{container['container_name']}: {container['total_iops']:.1f}iops, {container['total_throughput_mbps']:.1f}MB/s"
                pod_details.append(container_info)

            pod_summary = f"{pod_name}[{'; '.join(pod_details)}]"
            details.append(pod_summary)

        # Add summary
        summary = s3_metrics.get("summary", {})
        if summary:
            if (
                use_time_range
                and "iops" in summary
                and isinstance(summary["iops"], dict)
            ):
                iops_summary = f"iops(min:{summary['iops']['min']:.1f}/avg:{summary['iops']['avg']:.1f}/max:{summary['iops']['max']:.1f})"
                throughput_summary = f"throughput(min:{summary['throughput_mbps']['min']:.1f}/avg:{summary['throughput_mbps']['avg']:.1f}/max:{summary['throughput_mbps']['max']:.1f}MB/s)"
                summary_str = f"Summary: {iops_summary}, {throughput_summary}"
            else:
                summary_str = f"Summary: {summary.get('total_iops', 0):.1f}iops, {summary.get('total_throughput_mbps', 0):.1f}MB/s"
            details.append(summary_str)

        return " | ".join(details)

    def _format_prometheus_metric(
        self, metric_value, is_time_range: bool, unit: str = ""
    ) -> str:
        """Format Prometheus metric values for CSV output."""
        if not metric_value:
            return ""

        if is_time_range and isinstance(metric_value, dict):
            # Time range format: "min/avg/max unit"
            min_val = metric_value.get("min", 0)
            avg_val = metric_value.get("avg", 0)
            max_val = metric_value.get("max", 0)

            if unit:
                return f"{min_val:.2f}/{avg_val:.2f}/{max_val:.2f} {unit}"
            else:
                return f"{min_val:.2f}/{avg_val:.2f}/{max_val:.2f}"
        else:
            # Single value format
            if isinstance(metric_value, int | float):
                if unit:
                    return f"{metric_value:.2f} {unit}"
                else:
                    return f"{metric_value:.2f}"
            else:
                return str(metric_value)

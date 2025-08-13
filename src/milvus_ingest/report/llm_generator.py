"""Report analyzer for Milvus import performance - supports both raw data export and GLM-powered analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..logging_config import get_logger
from .glm_analyzer import GLMAnalyzer
from .loki_collector import LokiDataCollector
from .models import ReportConfig
from .prometheus_collector import PrometheusCollector
from .report_templates import GLM_ANALYSIS_PROMPT, format_analysis_report


class ReportAnalyzer:
    """Analyze Milvus import performance data with raw export or GLM-powered analysis."""

    def __init__(
        self,
        config: ReportConfig,
        glm_api_key: str | None = None,
        glm_model: str = "glm-4-flash",
    ):
        """Initialize the report analyzer.

        Args:
            config: Report configuration
            glm_api_key: GLM API key for analysis (optional)
            glm_model: GLM model to use (default: glm-4-flash)
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.loki_collector = LokiDataCollector(config)
        self.prometheus_collector = PrometheusCollector(config)

        # Initialize GLM analyzer if API key provided
        self.glm_analyzer = None
        if glm_api_key:
            self.glm_analyzer = GLMAnalyzer(glm_api_key, glm_model)
            self.logger.info(f"GLM analyzer initialized with model: {glm_model}")

    def generate_report(
        self,
        job_ids: list[str] | None = None,
        collection_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        output_file: str = "/tmp/import_report.md",
        output_format: str = "analysis",
        test_scenario: str | None = None,
        notes: str | None = None,
        release_name: str | None = None,
        milvus_namespace: str | None = None,
        import_info_file: str | None = None,
    ) -> dict[str, Any]:
        """Generate performance report.

        Args:
            job_ids: List of job IDs for the import
            collection_name: Collection name
            start_time: Start time for analysis
            end_time: End time for analysis
            output_file: Output file path
            output_format: Output format ('analysis' for GLM analysis, 'raw' for raw data)
            test_scenario: Test scenario description
            notes: Additional notes
            release_name: Milvus release name
            milvus_namespace: Milvus namespace
            import_info_file: Path to import_info.json

        Returns:
            Dictionary with generation summary
        """
        self.logger.info(f"Generating {output_format} report for jobs: {job_ids}")

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
                loki_data = self.loki_collector.collect_raw_logs(
                    job_id=job_id,
                    collection_name=collection_name,
                    start_time=start_time,
                    end_time=end_time,
                )
                raw_data["loki_logs"].extend(loki_data.get("logs", []))
        else:
            loki_data = self.loki_collector.collect_raw_logs(
                collection_name=collection_name,
                start_time=start_time,
                end_time=end_time,
            )
            raw_data["loki_logs"].extend(loki_data.get("logs", []))

        # Collect raw Prometheus metrics
        if release_name and milvus_namespace:
            self.logger.info("Collecting raw Prometheus metrics...")
            try:
                prometheus_data = self.prometheus_collector.collect_raw_metrics(
                    release_name=release_name,
                    namespace=milvus_namespace,
                    start_time=start_time,
                    end_time=end_time,
                )
                raw_data["prometheus_metrics"] = prometheus_data

            except Exception as e:
                self.logger.error(f"Error collecting Prometheus metrics: {e}")
                raw_data["prometheus_metrics"]["error"] = str(e)

        # Generate report based on format
        if output_format.lower() == "analysis" and self.glm_analyzer:
            return self._generate_analysis_report(raw_data, output_file)
        else:
            # Default to raw data export
            return self._generate_raw_report(raw_data, output_file)

    def _generate_analysis_report(
        self, raw_data: dict[str, Any], output_file: str
    ) -> dict[str, Any]:
        """Generate GLM-powered analysis report."""
        try:
            self.logger.info("Starting GLM analysis...")

            # Use raw data directly
            raw_data_str = json.dumps(raw_data, default=str, indent=2)
            self.logger.debug(
                f"Sending raw data to GLM: {len(raw_data_str)} characters"
            )

            # Get analysis from GLM using raw data
            analysis_result = self.glm_analyzer.analyze(raw_data, GLM_ANALYSIS_PROMPT)

            # Format the final report
            formatted_report = format_analysis_report(analysis_result, raw_data)

            # Write analysis report to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_report)

            self.logger.info(f"GLM analysis report written to {output_path}")

            return {
                "jobs_analyzed": len(raw_data["metadata"]["job_ids"])
                if raw_data["metadata"]["job_ids"]
                else 0,
                "total_logs": len(raw_data["loki_logs"]),
                "output_file": output_file,
                "format": "analysis",
                "glm_model": self.glm_analyzer.model,
                "analysis_length": len(analysis_result),
            }

        except Exception as e:
            self.logger.error(f"GLM analysis failed: {e}")
            self.logger.info("Falling back to raw data export...")

            # Fallback to raw data export
            return self._generate_raw_report(
                raw_data, output_file.replace(".md", "_raw.json")
            )

    def _generate_raw_report(
        self, raw_data: dict[str, Any], output_file: str
    ) -> dict[str, Any]:
        """Generate raw data report."""
        # For raw format, export as JSON
        if output_file.endswith(".md"):
            output_file = output_file.replace(".md", ".json")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Raw data exported to {output_path}")

        return {
            "jobs_analyzed": len(raw_data["metadata"]["job_ids"])
            if raw_data["metadata"]["job_ids"]
            else 0,
            "total_logs": len(raw_data["loki_logs"]),
            "output_file": output_file,
            "format": "raw",
        }

    def _write_raw_document(self, raw_data: dict[str, Any], output_file: str):
        """Write the raw data document (legacy method for compatibility)."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Milvus Import Test Raw Data\n\n")
            f.write(
                "This document contains raw, unprocessed data from the Milvus import test.\n\n"
            )

            # Metadata section
            f.write("## Test Metadata\n\n")
            f.write("```json\n")
            f.write(json.dumps(raw_data["metadata"], indent=2, ensure_ascii=False))
            f.write("\n```\n\n")

            # Import info section
            if raw_data["import_info"]:
                f.write("## Import Info (from import_info.json)\n\n")
                f.write("```json\n")
                f.write(
                    json.dumps(raw_data["import_info"], indent=2, ensure_ascii=False)
                )
                f.write("\n```\n\n")

            # Loki logs section
            f.write("## Loki Logs (Raw)\n\n")
            f.write(f"Total log entries: {len(raw_data['loki_logs'])}\n\n")
            f.write("```json\n")
            f.write(
                json.dumps(
                    raw_data["loki_logs"], indent=2, ensure_ascii=False, default=str
                )
            )
            f.write("\n```\n\n")

            # Prometheus metrics section
            if raw_data["prometheus_metrics"]:
                f.write("## Prometheus Metrics (Raw)\n\n")
                f.write("```json\n")
                f.write(
                    json.dumps(
                        raw_data["prometheus_metrics"],
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )
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

    def test_glm_connection(self) -> bool:
        """Test GLM API connection."""
        if not self.glm_analyzer:
            self.logger.warning("GLM analyzer not initialized")
            return False

        return self.glm_analyzer.test_connection()


# Backward compatibility
class LLMGenerator(ReportAnalyzer):
    """Legacy class name for backward compatibility."""

    def generate_llm_context(self, **kwargs) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.generate_report(output_format="raw", **kwargs)

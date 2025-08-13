"""Data models for report generation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ImportTimingData(BaseModel):
    """Import timing information from Loki logs."""

    timestamp: datetime
    job_id: str | None = None
    state: str | None = None
    time_cost: float | None = None
    time_unit: str | None = None
    import_phase: str | None = None
    import_state: str | None = None  # l0Import, buildIndex, etc.
    collection_name: str | None = None
    message: str
    pod: str
    namespace: str
    raw_timestamp: int


class PrometheusMetricData(BaseModel):
    """Prometheus metric data point."""

    timestamp: datetime
    metric_name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)
    unit: str | None = None


class ImportJobSummary(BaseModel):
    """Summary of an import job."""

    job_id: str
    collection_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: str  # completed, failed, running, etc.
    total_duration_ms: float | None = None
    phases: dict[str, float] = Field(default_factory=dict)  # phase -> duration_ms
    errors: list[str] = Field(default_factory=list)


class SystemMetrics(BaseModel):
    """System resource metrics during import."""

    timestamp: datetime
    cpu_usage_percent: float | None = None
    memory_usage_bytes: int | None = None
    memory_usage_percent: float | None = None
    disk_io_read_bytes: int | None = None
    disk_io_write_bytes: int | None = None
    network_rx_bytes: int | None = None
    network_tx_bytes: int | None = None


class MilvusClusterMetrics(BaseModel):
    """Milvus-specific cluster metrics."""

    timestamp: datetime
    import_jobs_total: int = 0
    import_tasks_total: int = 0
    insert_vectors_count: int = 0
    stored_binlog_size_bytes: int = 0
    stored_rows_total: int = 0
    segment_count: int = 0
    collection_count: int = 0
    import_job_latency_ms: float | None = None
    import_task_latency_ms: float | None = None


class ReportData(BaseModel):
    """Complete report data structure."""

    report_id: str
    generated_at: datetime
    time_range_start: datetime
    time_range_end: datetime

    # Data sources
    loki_logs: list[ImportTimingData] = Field(default_factory=list)
    prometheus_metrics: list[PrometheusMetricData] = Field(default_factory=list)

    # Analysis results
    import_jobs: list[ImportJobSummary] = Field(default_factory=list)
    system_metrics: list[SystemMetrics] = Field(default_factory=list)
    milvus_metrics: list[MilvusClusterMetrics] = Field(default_factory=list)

    # Summary statistics
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    average_job_duration_ms: float | None = None
    total_data_imported_gb: float | None = None

    # Performance insights
    bottlenecks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    # Data source URLs
    loki_url: str = "http://10.100.36.154:80"
    prometheus_url: str = "http://10.100.36.157:9090"

    # Query parameters
    job_id_pattern: str | None = None
    pod_pattern: str = ".*"
    collection_name: str | None = None
    namespace: str = "chaos-testing"  # Kubernetes namespace for log queries

    # Deployment mode (cluster, standalone, or auto-detect)
    deployment_mode: str = "auto"  # "auto", "cluster", or "standalone"

    # Time range
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_hours: int = 1

    # Output options
    output_format: str = "html"  # html, json, csv
    output_file: str = "report.html"
    include_raw_data: bool = True

    # Template settings
    template_dir: str | None = None
    chart_library: str = "chartjs"  # chartjs, plotly

    # Query limits
    max_log_entries: int = 20000
    timeout_seconds: int = 30

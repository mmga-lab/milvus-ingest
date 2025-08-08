"""Data models for report generation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImportTimingData(BaseModel):
    """Import timing information from Loki logs."""
    
    timestamp: datetime
    job_id: Optional[str] = None
    state: Optional[str] = None
    time_cost: Optional[float] = None
    time_unit: Optional[str] = None
    import_phase: Optional[str] = None
    import_state: Optional[str] = None  # l0Import, buildIndex, etc.
    collection_name: Optional[str] = None
    message: str
    pod: str
    namespace: str
    raw_timestamp: int


class PrometheusMetricData(BaseModel):
    """Prometheus metric data point."""
    
    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: Optional[str] = None


class ImportJobSummary(BaseModel):
    """Summary of an import job."""
    
    job_id: str
    collection_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str  # completed, failed, running, etc.
    total_duration_ms: Optional[float] = None
    phases: Dict[str, float] = Field(default_factory=dict)  # phase -> duration_ms
    errors: List[str] = Field(default_factory=list)
    
    
class SystemMetrics(BaseModel):
    """System resource metrics during import."""
    
    timestamp: datetime
    cpu_usage_percent: Optional[float] = None
    memory_usage_bytes: Optional[int] = None
    memory_usage_percent: Optional[float] = None
    disk_io_read_bytes: Optional[int] = None
    disk_io_write_bytes: Optional[int] = None
    network_rx_bytes: Optional[int] = None
    network_tx_bytes: Optional[int] = None


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
    import_job_latency_ms: Optional[float] = None
    import_task_latency_ms: Optional[float] = None


class ReportData(BaseModel):
    """Complete report data structure."""
    
    report_id: str
    generated_at: datetime
    time_range_start: datetime
    time_range_end: datetime
    
    # Data sources
    loki_logs: List[ImportTimingData] = Field(default_factory=list)
    prometheus_metrics: List[PrometheusMetricData] = Field(default_factory=list)
    
    # Analysis results
    import_jobs: List[ImportJobSummary] = Field(default_factory=list)
    system_metrics: List[SystemMetrics] = Field(default_factory=list)
    milvus_metrics: List[MilvusClusterMetrics] = Field(default_factory=list)
    
    # Summary statistics
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    average_job_duration_ms: Optional[float] = None
    total_data_imported_gb: Optional[float] = None
    
    # Performance insights
    bottlenecks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    

class ReportConfig(BaseModel):
    """Configuration for report generation."""
    
    # Data source URLs
    loki_url: str = "http://10.100.36.154:80"
    prometheus_url: str = "http://10.100.36.157:9090"
    
    # Query parameters
    job_id_pattern: Optional[str] = None
    pod_pattern: str = ".*"
    collection_name: Optional[str] = None
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_hours: int = 1
    
    # Output options
    output_format: str = "html"  # html, json, csv
    output_file: str = "report.html"
    include_raw_data: bool = True
    
    # Template settings
    template_dir: Optional[str] = None
    chart_library: str = "chartjs"  # chartjs, plotly
    
    # Query limits
    max_log_entries: int = 20000
    timeout_seconds: int = 30
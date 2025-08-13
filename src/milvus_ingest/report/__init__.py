"""Report generation module for Milvus import performance analysis."""

from .loki_collector import LokiDataCollector
from .models import ReportConfig
from .prometheus_collector import PrometheusCollector
from .report_generator import ReportGenerator


def create_config_for_standalone(
    loki_url: str = "http://10.100.36.154:80",
    prometheus_url: str = "http://10.100.36.157:9090",
) -> ReportConfig:
    """Create a ReportConfig optimized for standalone Milvus deployments.

    Args:
        loki_url: Loki service URL
        prometheus_url: Prometheus service URL

    Returns:
        ReportConfig configured for standalone mode
    """
    return ReportConfig(
        loki_url=loki_url,
        prometheus_url=prometheus_url,
        deployment_mode="standalone",
        pod_pattern=".*",
        max_log_entries=20000,
        timeout_seconds=30,
    )


def create_config_for_cluster(
    loki_url: str = "http://10.100.36.154:80",
    prometheus_url: str = "http://10.100.36.157:9090",
) -> ReportConfig:
    """Create a ReportConfig optimized for cluster Milvus deployments.

    Args:
        loki_url: Loki service URL
        prometheus_url: Prometheus service URL

    Returns:
        ReportConfig configured for cluster mode
    """
    return ReportConfig(
        loki_url=loki_url,
        prometheus_url=prometheus_url,
        deployment_mode="cluster",
        pod_pattern=".*",
        max_log_entries=20000,
        timeout_seconds=30,
    )


def create_config_with_auto_detection(
    loki_url: str = "http://10.100.36.154:80",
    prometheus_url: str = "http://10.100.36.157:9090",
) -> ReportConfig:
    """Create a ReportConfig with automatic deployment mode detection.

    This is the recommended approach for most use cases.

    Args:
        loki_url: Loki service URL
        prometheus_url: Prometheus service URL

    Returns:
        ReportConfig configured for auto-detection
    """
    return ReportConfig(
        loki_url=loki_url,
        prometheus_url=prometheus_url,
        deployment_mode="auto",
        pod_pattern=".*",
        max_log_entries=20000,
        timeout_seconds=30,
    )


__all__ = [
    "ReportGenerator",
    "LokiDataCollector",
    "PrometheusCollector",
    "ReportConfig",
    "create_config_for_standalone",
    "create_config_for_cluster",
    "create_config_with_auto_detection",
]

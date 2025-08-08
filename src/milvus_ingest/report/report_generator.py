"""Core report generator for Milvus import performance analysis."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger
from .models import (
    ReportData, 
    ReportConfig, 
    ImportJobSummary,
)
from .loki_collector import LokiDataCollector
from .prometheus_collector import PrometheusDataCollector


class ReportGenerator:
    """Main report generator that orchestrates data collection and report creation."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize data collectors
        self.loki_collector = LokiDataCollector(config)
        self.prometheus_collector = PrometheusDataCollector(config)
    
    def generate_report(
        self,
        job_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ReportData:
        """Generate a comprehensive import performance report.
        
        Args:
            job_id: Specific job ID to analyze
            collection_name: Collection name to filter by
            start_time: Start time for analysis
            end_time: End time for analysis
            
        Returns:
            Complete report data structure
        """
        # Use config defaults if not provided
        job_id = job_id or self.config.job_id_pattern
        collection_name = collection_name or self.config.collection_name
        start_time = start_time or self.config.start_time
        end_time = end_time or self.config.end_time
        
        # Set default time range if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=self.config.duration_hours)
        
        self.logger.info(
            "Generating import performance report",
            extra={
                "job_id": job_id,
                "collection": collection_name,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        
        # Create report data structure
        report_data = ReportData(
            report_id=str(uuid.uuid4())[:8],
            generated_at=datetime.now(),
            time_range_start=start_time,
            time_range_end=end_time,
        )
        
        try:
            # Collect Loki log data
            self.logger.info("Collecting Loki log data...")
            report_data.loki_logs = self.loki_collector.collect_import_timing_data(
                job_id=job_id,
                collection_name=collection_name,
                start_time=start_time,
                end_time=end_time,
            )
            
            # Collect Prometheus metrics
            self.logger.info("Collecting Prometheus metrics...")
            report_data.prometheus_metrics = self.prometheus_collector.collect_import_metrics(
                start_time=start_time,
                end_time=end_time,
            )
            
            # Collect system metrics
            self.logger.info("Collecting system resource metrics...")
            report_data.system_metrics = self.prometheus_collector.collect_system_metrics(
                start_time=start_time,
                end_time=end_time,
            )
            
            # Collect Milvus cluster metrics
            self.logger.info("Collecting Milvus cluster metrics...")
            report_data.milvus_metrics = self.prometheus_collector.collect_milvus_cluster_metrics(
                start_time=start_time,
                end_time=end_time,
            )
            
            # Analyze collected data
            self.logger.info("Analyzing collected data...")
            self._analyze_import_jobs(report_data)
            self._generate_insights(report_data)
            
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            raise
        
        self.logger.info(
            f"Report generation completed",
            extra={
                "report_id": report_data.report_id,
                "loki_logs": len(report_data.loki_logs),
                "prometheus_metrics": len(report_data.prometheus_metrics),
                "import_jobs": len(report_data.import_jobs),
            }
        )
        
        return report_data
    
    def save_report(
        self, 
        report_data: ReportData, 
        output_path: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """Save report to file.
        
        Args:
            report_data: Report data to save
            output_path: Output file path (overrides config)
            output_format: Output format (overrides config)
            
        Returns:
            Path to saved report file
        """
        output_path = output_path or self.config.output_file
        output_format = output_format or self.config.output_format
        
        self.logger.info(f"Saving report to {output_path} in {output_format} format")
        
        if output_format.lower() == 'json':
            return self._save_json_report(report_data, output_path)
        elif output_format.lower() == 'csv':
            return self._save_csv_report(report_data, output_path)
        elif output_format.lower() == 'html':
            return self._save_html_report(report_data, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _analyze_import_jobs(self, report_data: ReportData) -> None:
        """Analyze import jobs from collected data."""
        job_summaries = {}
        
        # Group timing data by job ID
        for timing in report_data.loki_logs:
            if not timing.job_id:
                continue
                
            job_id = timing.job_id
            if job_id not in job_summaries:
                job_summaries[job_id] = ImportJobSummary(
                    job_id=job_id,
                    collection_name=timing.collection_name,
                    status='unknown',
                )
            
            job_summary = job_summaries[job_id]
            
            # Update job details
            if timing.collection_name and not job_summary.collection_name:
                job_summary.collection_name = timing.collection_name
            
            # Track timing phases
            if timing.import_phase == 'start' and not job_summary.start_time:
                job_summary.start_time = timing.timestamp
            elif timing.import_phase in ['completed', 'failed']:
                job_summary.end_time = timing.timestamp
                job_summary.status = timing.import_phase
            
            # Track phase durations
            if timing.time_cost and timing.import_state:
                # Convert to milliseconds
                time_ms = timing.time_cost
                if timing.time_unit == 's':
                    time_ms *= 1000
                elif timing.time_unit in ['us', 'µs']:
                    time_ms /= 1000
                
                job_summary.phases[timing.import_state] = time_ms
            
            # Track errors
            if timing.import_phase == 'failed':
                job_summary.errors.append(timing.message[:200])
        
        # Calculate total durations and set final status
        for job_summary in job_summaries.values():
            if job_summary.start_time and job_summary.end_time:
                duration = job_summary.end_time - job_summary.start_time
                job_summary.total_duration_ms = duration.total_seconds() * 1000
            
            # Set status based on available information
            if job_summary.status == 'unknown':
                if job_summary.errors:
                    job_summary.status = 'failed'
                elif job_summary.phases:
                    job_summary.status = 'completed'
                else:
                    job_summary.status = 'running'
        
        report_data.import_jobs = list(job_summaries.values())
        
        # Calculate summary statistics
        report_data.total_jobs = len(report_data.import_jobs)
        report_data.successful_jobs = len([j for j in report_data.import_jobs if j.status == 'completed'])
        report_data.failed_jobs = len([j for j in report_data.import_jobs if j.status == 'failed'])
        
        # Calculate average job duration
        completed_jobs = [j for j in report_data.import_jobs if j.total_duration_ms]
        if completed_jobs:
            report_data.average_job_duration_ms = sum(j.total_duration_ms for j in completed_jobs) / len(completed_jobs)
    
    def _generate_insights(self, report_data: ReportData) -> None:
        """Generate performance insights and recommendations."""
        insights = []
        recommendations = []
        
        # Analyze job success rate
        if report_data.total_jobs > 0:
            success_rate = report_data.successful_jobs / report_data.total_jobs
            if success_rate < 0.8:
                insights.append(f"Low success rate: {success_rate:.1%} of import jobs completed successfully")
                recommendations.append("Investigate failed jobs and address underlying issues")
        
        # Analyze job durations
        if report_data.import_jobs:
            durations = [j.total_duration_ms for j in report_data.import_jobs if j.total_duration_ms]
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                
                if avg_duration > 300000:  # > 5 minutes
                    insights.append(f"Long average import duration: {avg_duration/1000:.1f}s")
                    recommendations.append("Consider optimizing data format or increasing resources")
                
                if max_duration > 1800000:  # > 30 minutes
                    insights.append(f"Some jobs took very long: {max_duration/1000:.1f}s maximum")
                    recommendations.append("Investigate slowest jobs for bottlenecks")
        
        # Analyze phase timing patterns
        phase_times = {}
        for job in report_data.import_jobs:
            for phase, duration in job.phases.items():
                if phase not in phase_times:
                    phase_times[phase] = []
                phase_times[phase].append(duration)
        
        # Identify bottleneck phases
        for phase, times in phase_times.items():
            if len(times) > 2:  # Need multiple samples
                avg_time = sum(times) / len(times)
                if avg_time > 60000:  # > 1 minute
                    insights.append(f"Slow import phase '{phase}': {avg_time/1000:.1f}s average")
                    if phase == 'buildIndex':
                        recommendations.append("Consider using AUTOINDEX for faster indexing during bulk imports")
                    elif phase == 'l0Import':
                        recommendations.append("Consider optimizing file sizes and batch configurations")
        
        # Analyze system resource usage
        if report_data.system_metrics:
            cpu_usages = [m.cpu_usage_percent for m in report_data.system_metrics if m.cpu_usage_percent]
            memory_usages = [m.memory_usage_percent for m in report_data.system_metrics if m.memory_usage_percent]
            
            if cpu_usages:
                avg_cpu = sum(cpu_usages) / len(cpu_usages)
                
                if avg_cpu > 80:
                    insights.append(f"High CPU usage during imports: {avg_cpu:.1f}% average")
                    recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
                elif avg_cpu < 20:
                    insights.append(f"Low CPU utilization: {avg_cpu:.1f}% average")
                    recommendations.append("CPU resources are underutilized - consider increasing batch sizes")
            
            if memory_usages:
                max_memory = max(memory_usages)
                
                if max_memory > 90:
                    insights.append(f"High memory pressure: {max_memory:.1f}% peak usage")
                    recommendations.append("Monitor for memory leaks or consider increasing memory allocation")
        
        # Analyze data throughput
        if report_data.milvus_metrics:
            row_counts = [m.stored_rows_total for m in report_data.milvus_metrics]
            if len(row_counts) > 1:
                row_growth = row_counts[-1] - row_counts[0]
                time_span_hours = (report_data.time_range_end - report_data.time_range_start).total_seconds() / 3600
                
                if time_span_hours > 0:
                    rows_per_hour = row_growth / time_span_hours
                    if rows_per_hour > 0:
                        insights.append(f"Data ingestion rate: {rows_per_hour:,.0f} rows/hour")
                        
                        if rows_per_hour < 100000:  # < 100K rows/hour
                            recommendations.append("Low ingestion rate - consider optimizing batch sizes and parallelism")
        
        report_data.bottlenecks = insights
        report_data.recommendations = recommendations
    
    def _save_json_report(self, report_data: ReportData, output_path: str) -> str:
        """Save report as JSON file."""
        # Convert to dict for JSON serialization
        report_dict = report_data.model_dump(mode='json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"JSON report saved to {output_path}")
        return output_path
    
    def _save_csv_report(self, report_data: ReportData, output_path: str) -> str:
        """Save report as CSV file."""
        import csv
        
        # Generate CSV with import job summaries
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Job ID', 'Collection', 'Status', 'Start Time', 'End Time',
                'Duration (ms)', 'Phases', 'Errors'
            ])
            
            # Data rows
            for job in report_data.import_jobs:
                phases_str = ';'.join([f"{k}={v:.1f}ms" for k, v in job.phases.items()])
                errors_str = ';'.join(job.errors)
                
                writer.writerow([
                    job.job_id,
                    job.collection_name or '',
                    job.status,
                    job.start_time.isoformat() if job.start_time else '',
                    job.end_time.isoformat() if job.end_time else '',
                    job.total_duration_ms or '',
                    phases_str,
                    errors_str,
                ])
        
        self.logger.info(f"CSV report saved to {output_path}")
        return output_path
    
    def _save_html_report(self, report_data: ReportData, output_path: str) -> str:
        """Save report as HTML file using template."""
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Find template directory
            template_dir = Path(__file__).parent / "templates"
            if self.config.template_dir:
                template_dir = Path(self.config.template_dir)
                
            # Setup Jinja2 environment
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            template = env.get_template("report_template.html")
            
            # Prepare chart data
            chart_data = self._prepare_chart_data(report_data)
            
            # Render template
            html_content = template.render(
                report_data=report_data,
                chart_data=chart_data,
            )
            
        except ImportError:
            self.logger.warning("Jinja2 not available, using simple HTML generator")
            html_content = self._generate_simple_html(report_data)
        except Exception as e:
            self.logger.warning(f"Template rendering failed: {e}, using simple HTML generator")
            html_content = self._generate_simple_html(report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to {output_path}")
        return output_path
    
    def _prepare_chart_data(self, report_data: ReportData) -> dict:
        """Prepare data for Chart.js visualizations."""
        chart_data = {}
        
        # Jobs Timeline Chart
        if report_data.import_jobs:
            jobs_timeline = {
                'datasets': [{
                    'label': 'Job Durations',
                    'data': [],
                    'backgroundColor': 'rgba(102, 126, 234, 0.6)',
                    'borderColor': 'rgba(102, 126, 234, 1)',
                    'pointRadius': 6,
                }]
            }
            
            for job in report_data.import_jobs:
                if job.start_time and job.total_duration_ms:
                    jobs_timeline['datasets'][0]['data'].append({
                        'x': job.start_time.isoformat(),
                        'y': job.total_duration_ms / 1000,  # Convert to seconds
                    })
            
            chart_data['jobs_timeline'] = jobs_timeline
        
        # System Metrics Chart
        if report_data.system_metrics:
            system_metrics = {
                'labels': [m.timestamp.isoformat() for m in report_data.system_metrics],
                'datasets': []
            }
            
            # CPU usage
            cpu_data = [m.cpu_usage_percent or 0 for m in report_data.system_metrics]
            if any(cpu_data):
                system_metrics['datasets'].append({
                    'label': 'CPU Usage (%)',
                    'data': cpu_data,
                    'borderColor': 'rgba(255, 99, 132, 1)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'tension': 0.4,
                })
            
            # Memory usage
            memory_data = [m.memory_usage_percent or 0 for m in report_data.system_metrics]
            if any(memory_data):
                system_metrics['datasets'].append({
                    'label': 'Memory Usage (%)',
                    'data': memory_data,
                    'borderColor': 'rgba(54, 162, 235, 1)',
                    'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                    'tension': 0.4,
                })
            
            chart_data['system_metrics'] = system_metrics
        
        # Phase Duration Chart
        if report_data.import_jobs:
            phase_durations = {}
            for job in report_data.import_jobs:
                for phase, duration in job.phases.items():
                    if phase not in phase_durations:
                        phase_durations[phase] = []
                    phase_durations[phase].append(duration)
            
            if phase_durations:
                phase_chart = {
                    'labels': list(phase_durations.keys()),
                    'datasets': [{
                        'label': 'Average Duration (ms)',
                        'data': [
                            sum(durations) / len(durations) 
                            for durations in phase_durations.values()
                        ],
                        'backgroundColor': [
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(255, 206, 86, 0.6)',
                            'rgba(75, 192, 192, 0.6)',
                            'rgba(153, 102, 255, 0.6)',
                        ],
                        'borderColor': [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                        ],
                        'borderWidth': 1,
                    }]
                }
                chart_data['phase_durations'] = phase_chart
        
        # Data Growth Chart
        if report_data.milvus_metrics:
            data_growth = {
                'labels': [m.timestamp.isoformat() for m in report_data.milvus_metrics],
                'datasets': []
            }
            
            # Stored rows
            rows_data = [m.stored_rows_total for m in report_data.milvus_metrics]
            if any(rows_data):
                data_growth['datasets'].append({
                    'label': 'Stored Rows',
                    'data': rows_data,
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.4,
                    'yAxisID': 'y',
                })
            
            # Storage size (in GB)
            storage_data = [m.stored_binlog_size_bytes / (1024**3) for m in report_data.milvus_metrics]
            if any(storage_data):
                data_growth['datasets'].append({
                    'label': 'Storage Size (GB)',
                    'data': storage_data,
                    'borderColor': 'rgba(153, 102, 255, 1)',
                    'backgroundColor': 'rgba(153, 102, 255, 0.2)',
                    'tension': 0.4,
                    'yAxisID': 'y1',
                })
            
            # Add dual y-axes configuration if we have both datasets
            if len(data_growth['datasets']) > 1:
                data_growth['options'] = {
                    'scales': {
                        'y': {
                            'type': 'linear',
                            'display': True,
                            'position': 'left',
                        },
                        'y1': {
                            'type': 'linear',
                            'display': True,
                            'position': 'right',
                            'grid': {
                                'drawOnChartArea': False,
                            },
                        }
                    }
                }
            
            chart_data['data_growth'] = data_growth
        
        return chart_data
    
    def _generate_simple_html(self, report_data: ReportData) -> str:
        """Generate a simple HTML report (placeholder for template-based version)."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Milvus Import Performance Report - {report_data.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .stat-card {{ background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #2c5aa0; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2c5aa0; border-bottom: 2px solid #2c5aa0; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .running {{ color: #ffc107; }}
        .insight {{ background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .recommendation {{ background: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Milvus Import Performance Report</h1>
        <p><strong>Report ID:</strong> {report_data.report_id}</p>
        <p><strong>Generated:</strong> {report_data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Time Range:</strong> {report_data.time_range_start.strftime('%Y-%m-%d %H:%M:%S')} - {report_data.time_range_end.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="stat-card">
            <div class="stat-number">{report_data.total_jobs}</div>
            <div class="stat-label">Total Jobs</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{report_data.successful_jobs}</div>
            <div class="stat-label">Successful Jobs</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{report_data.failed_jobs}</div>
            <div class="stat-label">Failed Jobs</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{report_data.average_job_duration_ms/1000:.1f}s</div>
            <div class="stat-label">Avg Duration</div>
        </div>
    </div>
"""
        
        # Add import jobs table
        if report_data.import_jobs:
            html += """
    <div class="section">
        <h2>Import Jobs</h2>
        <table>
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>Collection</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Key Phases</th>
                </tr>
            </thead>
            <tbody>
"""
            for job in report_data.import_jobs:
                status_class = job.status
                duration = f"{job.total_duration_ms/1000:.1f}s" if job.total_duration_ms else "N/A"
                phases = ", ".join([f"{k}: {v:.0f}ms" for k, v in sorted(job.phases.items())[:3]])
                
                html += f"""
                <tr>
                    <td>{job.job_id[:12]}...</td>
                    <td>{job.collection_name or 'N/A'}</td>
                    <td class="{status_class}">{job.status.title()}</td>
                    <td>{duration}</td>
                    <td>{phases}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
    </div>
"""
        
        # Add insights and recommendations
        if report_data.bottlenecks or report_data.recommendations:
            html += """
    <div class="section">
        <h2>Analysis & Recommendations</h2>
"""
            for insight in report_data.bottlenecks:
                html += f'<div class="insight">💡 <strong>Insight:</strong> {insight}</div>'
            
            for recommendation in report_data.recommendations:
                html += f'<div class="recommendation">✅ <strong>Recommendation:</strong> {recommendation}</div>'
            
            html += "</div>"
        
        html += """
    <div class="section">
        <h2>Data Sources</h2>
        <p><strong>Loki Logs:</strong> {len(report_data.loki_logs)} entries</p>
        <p><strong>Prometheus Metrics:</strong> {len(report_data.prometheus_metrics)} data points</p>
        <p><strong>System Metrics:</strong> {len(report_data.system_metrics)} data points</p>
        <p><strong>Milvus Metrics:</strong> {len(report_data.milvus_metrics)} data points</p>
    </div>
</body>
</html>
""".format(len=len, report_data=report_data)
        
        return html
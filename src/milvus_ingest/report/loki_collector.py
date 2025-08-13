"""Loki data collector for raw log data collection only."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..logging_config import get_logger
from .models import ReportConfig


class LokiDataCollector:
    """Simple Loki collector that only fetches raw log data without parsing."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the Loki collector.
        
        Args:
            config: Report configuration containing Loki URL and other settings
        """
        self.config = config
        self.loki_url = config.loki_url.rstrip('/')
        self.logger = get_logger(__name__)
    
    def collect_raw_logs(
        self,
        job_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_logs: int = 10000
    ) -> Dict[str, Any]:
        """Collect raw log data from Loki without processing.
        
        Args:
            job_id: Specific job ID to search for
            collection_name: Collection name to filter by
            start_time: Start time for query range
            end_time: End time for query range
            max_logs: Maximum number of logs to retrieve
            
        Returns:
            Dict containing raw log entries and metadata
        """
        # Use defaults if not provided
        start_time = start_time or (datetime.now() - timedelta(hours=1))
        end_time = end_time or datetime.now()
        
        # Build search patterns for different queries
        search_patterns = []
        
        # Primary search patterns based on parameters
        if job_id and collection_name:
            search_patterns.extend([
                f'|~ "{job_id}" |~ "{collection_name}"',
                f'|~ "{job_id}"',
                f'|~ "{collection_name}"'
            ])
        elif job_id:
            search_patterns.extend([
                f'|~ "{job_id}"',
                '|~ "import"'
            ])
        elif collection_name:
            search_patterns.extend([
                f'|~ "{collection_name}"',
                '|~ "import"'
            ])
        else:
            search_patterns.extend([
                '|~ "import"',
                '|~ "job"'
            ])
        
        # Common import-related patterns
        search_patterns.extend([
            '|~ "jobTimeCost"',
            '|~ "time cost"',
            '|~ "import job"',
            '|~ "completed"',
            '|~ "failed"',
            '|~ "error"'
        ])
        
        raw_logs = {
            "logs": [],
            "metadata": {
                "job_id": job_id,
                "collection_name": collection_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "query_timestamp": datetime.now().isoformat(),
                "max_logs": max_logs,
                "total_patterns_tried": 0,
                "successful_queries": 0
            }
        }
        
        # Execute queries for different search patterns
        for i, pattern in enumerate(search_patterns):
            try:
                base_query = f'{{namespace="{self.config.namespace}", pod=~"{self.config.pod_pattern}"}}'
                query = f"{base_query} {pattern}"
                
                self.logger.debug(f"Executing Loki query {i+1}/{len(search_patterns)}: {query}")
                
                logs = self._query_loki(query, start_time, end_time, max_logs)
                if logs:
                    raw_logs["logs"].extend(logs)
                    raw_logs["metadata"]["successful_queries"] += 1
                    self.logger.info(f"Found {len(logs)} logs with pattern: {pattern}")
                    
                    # Stop if we have enough logs
                    if len(raw_logs["logs"]) >= max_logs:
                        break
                
            except Exception as e:
                self.logger.warning(f"Failed to execute query with pattern '{pattern}': {e}")
            
            raw_logs["metadata"]["total_patterns_tried"] += 1
        
        # Remove duplicates based on timestamp and message
        unique_logs = []
        seen_entries = set()
        
        for log in raw_logs["logs"]:
            # Create a unique key based on timestamp and message
            key = (log.get("timestamp", ""), log.get("message", "")[:100])  # First 100 chars of message
            if key not in seen_entries:
                unique_logs.append(log)
                seen_entries.add(key)
        
        raw_logs["logs"] = unique_logs
        raw_logs["metadata"]["unique_logs_count"] = len(unique_logs)
        raw_logs["metadata"]["total_logs_collected"] = len(raw_logs["logs"])
        
        self.logger.info(f"Collected {len(unique_logs)} unique logs from {raw_logs['metadata']['successful_queries']} successful queries")
        
        return raw_logs
    
    def _query_loki(self, query: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> List[Dict[str, Any]]:
        """Execute a LogQL query and return raw log entries."""
        params = {
            "query": query,
            "start": int(start_time.timestamp() * 1_000_000_000),  # Loki expects nanoseconds
            "end": int(end_time.timestamp() * 1_000_000_000),
            "limit": limit,
            "direction": "backward"  # Get newest logs first
        }
        
        try:
            response = requests.get(
                f"{self.loki_url}/loki/api/v1/query_range",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "success":
                self.logger.warning(f"Loki query failed: {data}")
                return []
            
            # Extract raw log entries
            raw_logs = []
            results = data.get("data", {}).get("result", [])
            
            for stream in results:
                stream_labels = stream.get("stream", {})
                values = stream.get("values", [])
                
                for timestamp_ns, log_line in values:
                    # Convert nanosecond timestamp to ISO format
                    timestamp_seconds = int(timestamp_ns) / 1_000_000_000
                    timestamp_iso = datetime.fromtimestamp(timestamp_seconds).isoformat()
                    
                    raw_logs.append({
                        "timestamp": timestamp_iso,
                        "timestamp_ns": timestamp_ns,
                        "message": log_line,
                        "stream_labels": stream_labels
                    })
            
            return raw_logs
            
        except Exception as e:
            self.logger.error(f"Error querying Loki: {e}")
            return []
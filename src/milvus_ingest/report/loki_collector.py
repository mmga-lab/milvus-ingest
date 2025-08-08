"""Loki data collector for import timing analysis."""

from __future__ import annotations

import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..logging_config import get_logger
from .models import ImportTimingData, ReportConfig


class LokiDataCollector:
    """Collector for Loki log data related to Milvus imports."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the Loki collector.
        
        Args:
            config: Report configuration containing Loki URL and other settings
        """
        self.config = config
        self.loki_url = config.loki_url.rstrip('/')
        self.logger = get_logger(__name__)
    
    def collect_import_timing_data(
        self,
        job_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ImportTimingData]:
        """Collect import timing data from Loki.
        
        Args:
            job_id: Specific job ID to search for
            collection_name: Collection name to filter by
            start_time: Start time for query range
            end_time: End time for query range
            
        Returns:
            List of import timing data points
        """
        # Use config defaults if not provided
        job_id = job_id or self.config.job_id_pattern
        start_time = start_time or self.config.start_time
        end_time = end_time or self.config.end_time
        
        # Set default time range if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=self.config.duration_hours)
        
        # Build search patterns based on what we're looking for
        search_patterns = self._build_search_patterns(job_id, collection_name)
        
        self.logger.info(
            "Collecting Loki data",
            extra={
                "job_id": job_id,
                "collection": collection_name,
                "start_time": start_time,
                "end_time": end_time,
                "patterns": search_patterns,
            }
        )
        
        # Try different search patterns to find relevant logs
        all_logs = []
        for patterns in search_patterns:
            try:
                logs = self._query_logs(
                    pod_pattern=self.config.pod_pattern,
                    search_patterns=patterns,
                    start_time=start_time,
                    end_time=end_time,
                    limit=self.config.max_log_entries,
                )
                
                self.logger.debug(f"Found {len(logs)} logs with patterns: {patterns}")
                all_logs.extend(logs)
                
                # If we found logs with specific patterns, prefer those
                if logs and (job_id in str(patterns) or "jobTimeCost" in str(patterns)):
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to query with patterns {patterns}: {e}")
                continue
        
        # Remove duplicates (same timestamp + message)
        unique_logs = []
        seen = set()
        for log in all_logs:
            key = (log['raw_timestamp'], log['message'][:100])  # Use first 100 chars as key
            if key not in seen:
                seen.add(key)
                unique_logs.append(log)
        
        # Extract timing information
        timing_data = self._extract_import_timing(unique_logs)
        
        self.logger.info(f"Collected {len(timing_data)} timing data points from {len(unique_logs)} log entries")
        
        return timing_data
    
    def _build_search_patterns(
        self, 
        job_id: Optional[str], 
        collection_name: Optional[str]
    ) -> List[List[str]]:
        """Build search patterns for LogQL queries."""
        patterns = []
        
        # Most specific patterns first
        if job_id and collection_name:
            patterns.append([job_id, collection_name, "jobTimeCost"])
            patterns.append([job_id, collection_name])
        
        if job_id:
            patterns.append([job_id, "jobTimeCost"])
            patterns.append([job_id, "time cost"])  # Alternative format
            patterns.append([job_id])
        
        if collection_name:
            patterns.append([collection_name, "import"])
            patterns.append([collection_name, "jobTimeCost"])
        
        # General patterns
        patterns.extend([
            ["jobTimeCost"],
            ["time cost"],  # Alternative pattern found in actual logs
            ["import job"],
            ["import"],
        ])
        
        return patterns
    
    def _query_logs(
        self,
        pod_pattern: str,
        search_patterns: List[str],
        start_time: datetime,
        end_time: datetime,
        limit: int = 20000,
    ) -> List[Dict[str, Any]]:
        """Query Loki logs using LogQL."""
        # Build LogQL query
        logql = f'{{pod=~"{pod_pattern}"}}'
        
        # Add search patterns
        for pattern in search_patterns:
            logql += f' |~ "{pattern}"'
        
        # Build request parameters
        params = {
            'query': logql,
            'start': int(start_time.timestamp() * 1000000000),  # nanosecond timestamp
            'end': int(end_time.timestamp() * 1000000000),
            'limit': limit,
            'direction': 'backward'  # newest to oldest
        }
        
        self.logger.debug(f"Executing LogQL query: {logql}")
        
        try:
            response = requests.get(
                f"{self.loki_url}/loki/api/v1/query_range",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success':
                return self._parse_loki_results(data['data']['result'])
            else:
                raise Exception(f"Loki query failed: {data}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Loki request failed: {str(e)}")
    
    def _parse_loki_results(self, results: List[Dict]) -> List[Dict]:
        """Parse Loki query results."""
        parsed_logs = []
        
        for stream in results:
            labels = stream['stream']
            for value in stream['values']:
                timestamp_ns = int(value[0])
                timestamp = datetime.fromtimestamp(timestamp_ns / 1e9)
                log_line = value[1]
                
                parsed_logs.append({
                    'timestamp': timestamp,
                    'labels': labels,
                    'message': log_line,
                    'raw_timestamp': timestamp_ns
                })
        
        # Sort by timestamp (newest to oldest)
        parsed_logs.sort(key=lambda x: x['raw_timestamp'], reverse=True)
        
        return parsed_logs
    
    def _extract_import_timing(self, logs: List[Dict]) -> List[ImportTimingData]:
        """Extract import timing information from log messages."""
        timing_data = []
        
        for log in logs:
            message = log['message']
            
            # Extract job ID
            job_id = self._extract_job_id(message)
            
            # Extract timing information
            time_cost, time_unit, import_state = self._extract_timing_info(message)
            
            # Extract state information
            state = self._extract_state(message)
            
            # Determine import phase
            import_phase = self._determine_import_phase(message)
            
            # Extract collection name
            collection_name = self._extract_collection_name(message)
            
            # Create timing data object
            timing_data.append(ImportTimingData(
                timestamp=log['timestamp'],
                job_id=job_id,
                state=state,
                time_cost=time_cost,
                time_unit=time_unit,
                import_phase=import_phase,
                import_state=import_state,
                collection_name=collection_name,
                message=message,
                pod=log['labels'].get('pod', 'unknown'),
                namespace=log['labels'].get('namespace', 'unknown'),
                raw_timestamp=log['raw_timestamp']
            ))
        
        return timing_data
    
    def _extract_job_id(self, message: str) -> Optional[str]:
        """Extract job ID from log message."""
        patterns = [
            r'\[jobID[:-]?(\w+)\]',
            r'job[_\s]*id[:\s=]+(\w+)',
            r'"jobID"[:\s]*"?(\w+)"?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_timing_info(self, message: str) -> tuple[Optional[float], Optional[str], Optional[str]]:
        """Extract timing information from log message."""
        # Import state patterns (with specific phases)
        import_state_patterns = [
            r'jobTimeCost/(\w+)=(\d+(?:\.\d+)?)(\w+)',     # jobTimeCost/l0Import=98.566µs
            r'\[jobTimeCost/(\w+)=(\d+(?:\.\d+)?)(\w+)\]', # [jobTimeCost/buildIndex=123.45ms]
            r'jobTimeCost-(\w+)=(\d+(?:\.\d+)?)(\w+)',     # jobTimeCost-preImport=456.78s
            r'jobTimeCost:(\w+)=(\d+(?:\.\d+)?)(\w+)',     # jobTimeCost:import=789.01ms
        ]
        
        # General timing patterns
        general_patterns = [
            r'\[jobTimeCost/(\d+(?:\.\d+)?)\s*(\w+)?\]',  # [jobTimeCost/123.45ms]
            r'\[jobTimeCost:(\d+(?:\.\d+)?)\s*(\w+)?\]',  # [jobTimeCost:123.45ms]
            r'\[jobTimeCost-(\d+(?:\.\d+)?)\s*(\w+)?\]',  # [jobTimeCost-123.45ms]
            r'jobTimeCost[/:-]?(\d+(?:\.\d+)?)\s*(\w+)?', # jobTimeCost/123.45ms
            r'"time cost"=(\d+(?:\.\d+)?)(\w+)',          # "time cost"=1.527µs
            r'time[_ ]cost[=:](\d+(?:\.\d+)?)(\w+)',      # time_cost=123ms
        ]
        
        # Try import state patterns first
        for pattern in import_state_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                import_state = match.group(1)
                time_cost = float(match.group(2))
                time_unit = match.group(3) if match.group(3) else 'ms'
                return time_cost, time_unit, import_state
        
        # Try general patterns
        for pattern in general_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                time_cost = float(match.group(1))
                time_unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else 'ms'
                return time_cost, time_unit, None
        
        return None, None, None
    
    def _extract_state(self, message: str) -> Optional[str]:
        """Extract state information from log message."""
        patterns = [
            r'\[state[=-]?(\w+)\]',
            r'state[:\s=]+(\w+)',
            r'"state"[:\s]*"?(\w+)"?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _determine_import_phase(self, message: str) -> Optional[str]:
        """Determine import phase from log message."""
        message_lower = message.lower()
        
        if "import job start" in message_lower:
            return 'start'
        elif "preimport done" in message_lower:
            return 'preImport_done'
        elif "import done" in message_lower and "preimport" not in message_lower:
            return 'import_done'
        elif "build index done" in message_lower:
            return 'build_index_done'
        elif "all completed" in message_lower:
            return 'completed'
        elif "failed" in message_lower or "error" in message_lower:
            return 'failed'
        
        return None
    
    def _extract_collection_name(self, message: str) -> Optional[str]:
        """Extract collection name from log message."""
        patterns = [
            r'collection[:\s]+(\w+)',
            r'"collection"[:\s]*"?(\w+)"?',
            r'coll[:\s=]+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
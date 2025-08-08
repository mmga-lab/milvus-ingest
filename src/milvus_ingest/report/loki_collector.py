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
        
        # Try direct queries for jobTimeCost logs first (most valuable data)
        all_logs = []
        timing_logs_found = False
        
        try:
            if job_id:
                # Query specifically for jobTimeCost logs with the job ID
                timing_logs = self._query_logs_direct(
                    query=f'{{namespace="chaos-testing", pod=~"{self.config.pod_pattern}"}} |~ "jobTimeCost" |~ "{job_id}"',
                    start_time=start_time,
                    end_time=end_time,
                    limit=self.config.max_log_entries,
                )
                self.logger.info(f"Found {len(timing_logs)} jobTimeCost logs for job {job_id}")
                if timing_logs:
                    self.logger.info(f"Sample timing log messages:")
                    for i, log in enumerate(timing_logs[:3]):  # Show first 3 as examples
                        self.logger.info(f"  [{i+1}] {log['message'][:100]}...")
                all_logs.extend(timing_logs)
                if timing_logs:
                    timing_logs_found = True
            
            # If no job-specific timing logs found, get general jobTimeCost logs
            if not timing_logs_found:
                general_timing_logs = self._query_logs_direct(
                    query=f'{{namespace="chaos-testing", pod=~"{self.config.pod_pattern}"}} |~ "jobTimeCost"',
                    start_time=start_time,
                    end_time=end_time,
                    limit=self.config.max_log_entries,
                )
                self.logger.info(f"Found {len(general_timing_logs)} general jobTimeCost logs")
                if general_timing_logs:
                    self.logger.info(f"Sample general timing log messages:")
                    for i, log in enumerate(general_timing_logs[:3]):  # Show first 3 as examples
                        self.logger.info(f"  [{i+1}] {log['message'][:100]}...")
                all_logs.extend(general_timing_logs)
                if general_timing_logs:
                    timing_logs_found = True
                
        except Exception as e:
            self.logger.warning(f"Failed to query jobTimeCost logs: {e}")
        
        # Fallback to pattern-based search if no timing logs found
        if not timing_logs_found:
            self.logger.info("No jobTimeCost logs found, falling back to pattern-based search")
            for patterns in search_patterns:
                try:
                    logs = self._query_logs(
                        pod_pattern=self.config.pod_pattern,
                        search_patterns=patterns,
                        start_time=start_time,
                        end_time=end_time,
                        limit=self.config.max_log_entries,
                    )
                    
                    self.logger.info(f"Found {len(logs)} logs with patterns: {patterns}")
                    if logs:
                        self.logger.info(f"Sample pattern-based log messages:")
                        for i, log in enumerate(logs[:2]):  # Show first 2 as examples
                            self.logger.info(f"  [{i+1}] {log['message'][:100]}...")
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
        
        self.logger.info(f"Collected {len(timing_data)} timing data points from {len(unique_logs)} unique log entries")
        
        # Log summary of timing data found
        if timing_data:
            phases = {}
            for data in timing_data:
                phase = data.import_phase or 'unknown'
                phases[phase] = phases.get(phase, 0) + 1
            
            self.logger.info(f"Timing data summary by phase:")
            for phase, count in phases.items():
                self.logger.info(f"  {phase}: {count} entries")
        else:
            self.logger.warning("No timing data extracted from logs!")
        
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
        
        # General patterns - more specific for import timing logs
        patterns.extend([
            ["jobTimeCost"],
            ["time cost"],  # Alternative pattern found in actual logs
            ["import job start to execute"],
            ["import job preimport done"],
            ["import job stats done"],
            ["import job 10 import done"],
            ["import job all completed"],
            ["import job build index done"],  # Added based on common patterns
            ["import job"],
            ["import"],
        ])
        
        return patterns
    
    def _query_logs_direct(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 20000,
    ) -> List[Dict[str, Any]]:
        """Direct Loki query using provided LogQL string."""
        # Build request parameters
        params = {
            'query': query,
            'start': int(start_time.timestamp() * 1000000000),  # nanosecond timestamp
            'end': int(end_time.timestamp() * 1000000000),
            'limit': limit,
            'direction': 'backward'  # newest to oldest
        }
        
        self.logger.info(f"Executing direct LogQL query: {query}")
        self.logger.info(f"Query parameters: start={start_time}, end={end_time}, limit={limit}")
        
        try:
            response = requests.get(
                f"{self.loki_url}/loki/api/v1/query_range",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            self.logger.info(f"Loki response status: {response.status_code}")
            
            data = response.json()
            self.logger.info(f"Loki response status: {data.get('status', 'unknown')}")
            
            if data.get('status') == 'success':
                result_count = len(data.get('data', {}).get('result', []))
                self.logger.info(f"Loki query returned {result_count} result streams")
            
            return self._parse_loki_response(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Loki request failed: {e}")
            raise
    
    def _parse_loki_response(self, data) -> List[Dict]:
        """Parse Loki response data."""
        if data.get('status') != 'success':
            self.logger.error(f"Loki query failed: {data}")
            return []
        
        return self._parse_loki_results(data['data']['result'])
    
    def _query_logs(
        self,
        pod_pattern: str,
        search_patterns: List[str],
        start_time: datetime,
        end_time: datetime,
        limit: int = 20000,
    ) -> List[Dict[str, Any]]:
        """Query Loki logs using LogQL."""
        # Build LogQL query with namespace filter
        logql = f'{{namespace="chaos-testing", pod=~"{pod_pattern}"}}'
        
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
        
        self.logger.info(f"Executing pattern-based LogQL query: {logql}")
        self.logger.info(f"Query parameters: start={start_time}, end={end_time}, limit={limit}")
        
        try:
            response = requests.get(
                f"{self.loki_url}/loki/api/v1/query_range",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            self.logger.info(f"Pattern-based Loki response status: {response.status_code}")
            
            data = response.json()
            self.logger.info(f"Pattern-based Loki response status: {data.get('status', 'unknown')}")
            
            if data['status'] == 'success':
                result_count = len(data.get('data', {}).get('result', []))
                self.logger.info(f"Pattern-based query returned {result_count} result streams")
                return self._parse_loki_results(data['data']['result'])
            else:
                self.logger.error(f"Pattern-based Loki query failed: {data}")
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
            r'\[jobID:(\d+)\]',  # Added pattern from screenshot: [jobID:459923331454862604]
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
            r'\[jobTimeCost/(\d+(?:\.\d+)?(?:m\d+(?:\.\d+)?)?(?:s|µs|ms))\]',  # [jobTimeCost/1m43.685519992s]
            r'\[jobTimeCost/(\d+(?:\.\d+)?)(\w+)\]',       # [jobTimeCost/98.066s] from screenshot
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
                # Handle different pattern formats
                if pattern == r'\[jobTimeCost/(\d+(?:\.\d+)?(?:m\d+(?:\.\d+)?)?(?:s|µs|ms))\]':
                    # Pattern: [jobTimeCost/1m43.685519992s] - complex time format
                    time_str = match.group(1)
                    time_cost, time_unit = self._parse_complex_time(time_str)
                    import_state = None
                elif pattern == r'\[jobTimeCost/(\d+(?:\.\d+)?)(\w+)\]':
                    # Pattern: [jobTimeCost/98.066s] - no specific import state
                    time_cost = float(match.group(1))
                    time_unit = match.group(2) if match.group(2) else 'ms'
                    import_state = None
                else:
                    # Pattern with import state: jobTimeCost/state=time
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
    
    def _parse_complex_time(self, time_str: str) -> tuple[Optional[float], Optional[str]]:
        """Parse complex time formats like '1m43.685519992s' or '98.066s'."""
        # Handle formats like: 1m43.685519992s, 98.066s, 123ms, 456µs
        if 'm' in time_str and 's' in time_str:
            # Format: 1m43.685519992s
            match = re.match(r'(\d+)m(\d+(?:\.\d+)?)s', time_str)
            if match:
                minutes = float(match.group(1))
                seconds = float(match.group(2))
                total_seconds = minutes * 60 + seconds
                return total_seconds, 's'
        elif time_str.endswith('s'):
            # Format: 98.066s
            return float(time_str[:-1]), 's'
        elif time_str.endswith('ms'):
            # Format: 123ms
            return float(time_str[:-2]), 'ms'
        elif time_str.endswith('µs'):
            # Format: 456µs
            return float(time_str[:-2]), 'µs'
        
        # Fallback: try to extract number and assume seconds
        match = re.search(r'(\d+(?:\.\d+)?)', time_str)
        if match:
            return float(match.group(1)), 's'
        
        return None, None
    
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
        
        # Based on the actual log messages, these are the phases we see:
        if "import job start to execute" in message_lower:
            return 'start_to_execute'
        elif "import job preimport done" in message_lower:
            return 'preimport_done'
        elif "import job import done" in message_lower:
            return 'import_done'
        elif "import job stats done" in message_lower:
            return 'stats_done'
        elif "import job l0 import done" in message_lower:
            return 'l0_import_done'
        elif "import job build index done" in message_lower:
            return 'build_index_done'
        elif "import job all completed" in message_lower:
            return 'all_completed'
        elif "import job start" in message_lower and "execute" not in message_lower:
            return 'start'
        elif "preimport done" in message_lower:
            return 'preimport_done'
        elif "l0 import done" in message_lower:
            return 'l0_import_done'
        elif "import done" in message_lower and "preimport" not in message_lower and "l0" not in message_lower:
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
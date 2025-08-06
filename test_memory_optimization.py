#!/usr/bin/env python3
"""Test script to verify memory optimization for large file generation."""

import subprocess
import time
import psutil
import os
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def test_large_file_generation():
    """Test generation of large files with memory optimization."""
    
    test_configs = [
        {
            "name": "10GB x 10 files with optimization",
            "cmd": [
                "milvus-ingest", "generate",
                "--builtin", "simple",
                "--file-size", "10GB",
                "--file-count", "10",
                "--workers", "2",  # Force 2 workers for 10GB files
                "--out", "test_10gb_optimized"
            ]
        },
        {
            "name": "5GB x 10 files with moderate workers",
            "cmd": [
                "milvus-ingest", "generate",
                "--builtin", "simple",
                "--file-size", "5GB",
                "--file-count", "10",
                "--workers", "4",  # Will be adjusted to 3-4
                "--out", "test_5gb_optimized"
            ]
        },
        {
            "name": "2GB x 10 files with normal workers",
            "cmd": [
                "milvus-ingest", "generate",
                "--builtin", "simple",
                "--file-size", "2GB",
                "--file-count", "10",
                "--out", "test_2gb_normal"
            ]
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Command: {' '.join(config['cmd'])}")
        print(f"{'='*60}")
        
        # Clean up output directory if exists
        output_dir = Path(config['cmd'][-1])
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
        # Monitor memory usage
        start_memory = get_memory_usage()
        print(f"Starting memory usage: {start_memory:.2f} GB")
        
        # Run the command
        start_time = time.time()
        try:
            result = subprocess.run(
                config['cmd'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            elapsed_time = time.time() - start_time
            end_memory = get_memory_usage()
            peak_memory_estimate = end_memory  # Simplified, actual peak might be higher
            
            print(f"\nResults:")
            print(f"  - Execution time: {elapsed_time:.2f} seconds")
            print(f"  - End memory usage: {end_memory:.2f} GB")
            print(f"  - Memory increase: {end_memory - start_memory:.2f} GB")
            print(f"  - Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"  - Error output: {result.stderr[:500]}")
            else:
                # Check output for worker adjustment messages
                if "worker adjustment" in result.stderr.lower() or "limiting workers" in result.stderr.lower():
                    print("\n✅ Worker adjustment detected in output:")
                    for line in result.stderr.split('\n'):
                        if 'worker' in line.lower() and ('adjust' in line.lower() or 'limit' in line.lower()):
                            print(f"    {line}")
                
                # Check for memory warnings
                if "memory" in result.stderr.lower() and "high" in result.stderr.lower():
                    print("\n⚠️ Memory warnings detected:")
                    for line in result.stderr.split('\n'):
                        if 'memory' in line.lower() and ('high' in line.lower() or 'exhaust' in line.lower()):
                            print(f"    {line}")
            
        except subprocess.TimeoutExpired:
            print(f"  - ERROR: Command timed out after 5 minutes")
        except Exception as e:
            print(f"  - ERROR: {str(e)}")
        
        # Clean up
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"  - Cleaned up output directory: {output_dir}")

def test_worker_adjustment_logic():
    """Test the worker adjustment logic directly."""
    print("\n" + "="*60)
    print("Testing Worker Adjustment Logic")
    print("="*60)
    
    import sys
    sys.path.insert(0, '/Users/zilliz/workspace/milvus-ingest')
    
    # Import the specific functions we need
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "optimized_writer", 
        "/Users/zilliz/workspace/milvus-ingest/src/milvus_ingest/optimized_writer.py"
    )
    optimized_writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimized_writer)
    
    determine_generation_strategy = optimized_writer.determine_generation_strategy
    adjust_workers_by_strategy = optimized_writer.adjust_workers_by_strategy
    
    test_cases = [
        ("10GB", 10, 8),  # 10GB files with 8 initial workers
        ("5GB", 10, 8),   # 5GB files with 8 initial workers
        ("2GB", 10, 8),   # 2GB files with 8 initial workers
        ("500MB", 10, 8), # 500MB files with 8 initial workers
    ]
    
    for file_size, file_count, initial_workers in test_cases:
        strategy = determine_generation_strategy(file_size, file_count)
        file_size_bytes = int(strategy.get("file_size_gb", 0) * 1024**3)
        if file_size_bytes == 0:
            # Parse file size if not in GB
            if "MB" in file_size:
                file_size_bytes = int(file_size.replace("MB", "")) * 1024**2
            elif "GB" in file_size:
                file_size_bytes = int(file_size.replace("GB", "")) * 1024**3
        
        adjusted_workers = adjust_workers_by_strategy(
            initial_workers, 
            strategy, 
            file_size_bytes,
            file_count
        )
        
        print(f"\nFile: {file_size} x {file_count}")
        print(f"  Strategy: {strategy['strategy']}")
        print(f"  Memory profile: {strategy['memory_profile']}")
        print(f"  Initial workers: {initial_workers}")
        print(f"  Adjusted workers: {adjusted_workers}")
        print(f"  Max parallel: {strategy.get('max_parallel_files', 'N/A')}")

if __name__ == "__main__":
    print("Memory Optimization Test Suite")
    print("==============================")
    
    # Test worker adjustment logic
    test_worker_adjustment_logic()
    
    # Uncomment to run actual generation tests (requires more time and memory)
    # print("\n\nWould you like to run actual file generation tests? (y/n)")
    # if input().lower() == 'y':
    #     test_large_file_generation()
    
    print("\n✅ Test suite completed!")
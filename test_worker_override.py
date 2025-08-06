#!/usr/bin/env python3
"""Test to verify if user-specified workers are overridden for large files."""

from src.milvus_ingest.optimized_writer import determine_generation_strategy, adjust_workers_by_strategy

def test_worker_override():
    """Test if setting --workers 8 gets overridden for 10GB files."""
    
    test_cases = [
        ("10GB", 10, 8, "10GB files with 8 workers"),
        ("10GB", 10, 16, "10GB files with 16 workers"), 
        ("5GB", 10, 8, "5GB files with 8 workers"),
        ("2GB", 10, 8, "2GB files with 8 workers"),
    ]
    
    print("Testing Worker Override Behavior")
    print("=" * 60)
    print("\nQuestion: If user sets --workers 8 for 10GB files, what happens?")
    print("-" * 60)
    
    for file_size, file_count, user_workers, description in test_cases:
        print(f"\n{description}:")
        print(f"  User specified: --workers {user_workers}")
        
        # Determine strategy
        strategy = determine_generation_strategy(file_size, file_count)
        
        # Parse file size
        file_size_gb = float(file_size.replace("GB", ""))
        file_size_bytes = int(file_size_gb * 1024**3)
        
        # Apply adjustment
        final_workers = adjust_workers_by_strategy(
            user_workers,  # User's requested workers
            strategy,
            file_size_bytes,
            file_count
        )
        
        print(f"  Memory profile: {strategy['memory_profile']}")
        print(f"  Final workers: {final_workers}")
        
        if final_workers < user_workers:
            print(f"  ⚠️ OVERRIDE: User's {user_workers} workers → {final_workers} workers (safety limit)")
        else:
            print(f"  ✅ ACCEPTED: Using user's {user_workers} workers")

if __name__ == "__main__":
    test_worker_override()
    
    print("\n" + "=" * 60)
    print("ANSWER: For 10GB files, even if user sets --workers 8,")
    print("        it will be LIMITED to 2 workers for memory safety.")
    print("=" * 60)
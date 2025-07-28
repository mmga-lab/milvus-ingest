# 性能调优指南

深入了解 milvus-ingest 的性能特点，掌握大规模数据生成的优化技巧，实现最佳的生成效率。

## 🎯 性能目标

### 性能基准
在现代硬件上的典型性能表现：

| 数据规模 | 生成速度 | 内存使用 | 文件输出 | 推荐配置 |
|----------|----------|----------|----------|----------|
| 1万行 | 5,000-10,000 行/秒 | <100MB | <5MB | 默认设置 |
| 10万行 | 7,000-15,000 行/秒 | <500MB | <50MB | 默认设置 |
| 100万行 | 10,000-25,000 行/秒 | <2GB | <500MB | 调优设置 |
| 1000万行 | 15,000-30,000 行/秒 | <4GB | <5GB | 高性能设置 |

### 硬件要求
**最低配置:**
- CPU: 4核心
- 内存: 8GB RAM  
- 存储: 100GB 可用空间

**推荐配置:**
- CPU: 8+核心现代处理器 (Intel i7/AMD Ryzen 7+)
- 内存: 16GB+ RAM
- 存储: SSD 固态硬盘

**高性能配置:**
- CPU: 16+核心处理器 (Xeon/Threadripper)
- 内存: 32GB+ RAM
- 存储: NVMe SSD

## ⚡ 核心优化技术

### 1. 向量化操作 (Vectorization)

工具使用 NumPy 向量化操作实现高性能：

```python
# 传统循环方式 (慢)
for i in range(batch_size):
    vector[i] = generate_single_vector()

# 向量化方式 (快)
vectors = np.random.random((batch_size, dim)).astype(np.float32)
```

**性能提升**: 10-20倍加速

### 2. 批量处理 (Batch Processing)

大批量处理减少函数调用开销：

```bash
# 小批量 (低效)
--batch-size 1000

# 大批量 (高效)
--batch-size 50000    # 默认推荐
--batch-size 100000   # 高性能环境
```

### 3. 内存流式处理 (Streaming)

避免将所有数据加载到内存：

```bash
# 启用文件分割防止内存溢出
--file-size 256MB        # 256MB 每文件
--rows-per-file 1000000 # 100万行每文件
```

### 4. 并行 I/O (Parallel I/O)

使用 PyArrow 和 Snappy 压缩优化写入：

```bash
# 选择最佳格式
--format parquet  # 推荐：最快的写入和读取
--format json     # 较慢但便于调试
```

## 🔧 参数调优指南

### 批处理大小优化

#### 自动检测最佳批处理大小

```bash
# 创建批处理大小测试脚本
cat > test_batch_size.sh << 'EOF'
#!/bin/bash

echo "测试不同批处理大小的性能..."

batch_sizes=(1000 5000 10000 25000 50000 100000)
rows=100000
schema="simple"

for batch_size in "${batch_sizes[@]}"; do
    echo "测试批处理大小: $batch_size"
    
    start_time=$(date +%s.%N)
    
    milvus-ingest generate \
        --builtin $schema \
        --total-rows $rows \
        --batch-size $batch_size \
        --out ./test_batch_${batch_size} \
        --no-progress > /dev/null 2>&1
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    speed=$(echo "scale=0; $rows / $duration" | bc)
    
    echo "  用时: ${duration}秒, 速度: ${speed} 行/秒"
    
    # 清理测试数据
    rm -rf ./test_batch_${batch_size}
done
EOF

chmod +x test_batch_size.sh
./test_batch_size.sh
```

#### 基于内存的批处理配置

```bash
# 内存受限环境 (8GB RAM)
milvus-ingest generate --batch-size 10000

# 标准环境 (16GB RAM)  
milvus-ingest generate --batch-size 50000

# 高内存环境 (32GB+ RAM)
milvus-ingest generate --batch-size 100000
```

### 文件分割策略

#### 基于数据大小的分割

```bash
# 针对不同用途优化文件大小

# 快速本地测试 - 小文件
--file-size 64MB --rows-per-file 100000

# 平衡性能 - 中等文件  
--file-size 256MB --rows-per-file 500000

# 最大吞吐量 - 大文件
--file-size 512MB --rows-per-file 1000000
```

#### 基于下游处理的分割

```bash
# Milvus 直接插入 - 较小文件便于处理
--file-size 128MB --rows-per-file 250000

# S3 批量导入 - 较大文件减少文件数
--file-size 512MB --rows-per-file 1000000

# 分布式处理 - 平衡文件大小
--file-size 256MB --rows-per-file 500000
```

## 📊 不同场景的优化策略

### 高吞吐量场景

目标：最大化数据生成速度

```bash
# 高吞吐量配置
milvus-ingest generate \
    --builtin simple \
    --total-rows 5000000 \
    --batch-size 200000 \
    --file-size 1024MB \
    --rows-per-file 2000000 \
    --format parquet \
    --no-progress \
    --out ./high_throughput_data
```

**优化要点:**
- 使用最大批处理大小
- 大文件减少 I/O 开销
- Parquet 格式最高效
- 禁用进度条减少输出开销

### 内存优化场景

目标：在有限内存下生成大数据集

```bash
# 内存优化配置
milvus-ingest generate \
    --builtin ecommerce \
    --total-rows 10000000 \
    --batch-size 25000 \
    --file-size 128MB \
    --rows-per-file 200000 \
    --format parquet \
    --out ./memory_optimized_data
```

**优化要点:**
- 适中的批处理大小
- 小文件防止内存峰值
- 频繁文件切换释放内存
- 监控内存使用情况

### 存储优化场景

目标：最小化磁盘占用

```bash
# 存储优化配置
milvus-ingest generate \
    --builtin documents \
    --total-rows 2000000 \
    --batch-size 50000 \
    --file-size 512MB \
    --format parquet \
    --out ./storage_optimized_data
```

**优化要点:**
- Parquet 格式压缩率最高
- 较大文件减少元数据开销
- 选择合适的字段类型
- 避免不必要的可空字段

## 🛠️ 实际性能调优案例

### 案例1：电商推荐系统 (1000万商品)

**需求:**
- 1000万商品记录
- 多个向量字段 (768维 + 512维)
- 4小时内完成生成

**硬件环境:**
- 32核 CPU, 64GB RAM
- NVMe SSD 存储

**调优过程:**

```bash
# 第一轮：基础配置测试
time milvus-ingest generate \
    --builtin ecommerce \
    --total-rows 100000 \
    --out ./test_baseline

# 结果：100K行用时45秒，预计1000万行需要75分钟

# 第二轮：增加批处理大小
time milvus-ingest generate \
    --builtin ecommerce \
    --total-rows 100000 \
    --batch-size 100000 \
    --out ./test_batch_optimized

# 结果：100K行用时28秒，预计1000万行需要47分钟

# 第三轮：优化文件分割
time milvus-ingest generate \
    --builtin ecommerce \
    --total-rows 100000 \
    --batch-size 100000 \
    --file-size 512MB \
    --rows-per-file 1000000 \
    --out ./test_file_optimized

# 结果：100K行用时25秒，预计1000万行需要42分钟

# 最终生产配置
time milvus-ingest generate \
    --builtin ecommerce \
    --total-rows 10000000 \
    --batch-size 150000 \
    --file-size 512MB \
    --rows-per-file 1000000 \
    --format parquet \
    --no-progress \
    --out ./ecommerce_10m_final

# 最终结果：38分钟完成，满足4小时要求
```

### 案例2：文档检索系统 (内存受限)

**需求:**
- 500万文档记录
- 768维 BERT 向量
- 8GB 内存限制

**调优策略:**

```bash
# 内存监控脚本
cat > monitor_memory.sh << 'EOF'
#!/bin/bash
while true; do
    memory_usage=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
    echo "$(date) - 内存使用: $memory_usage"
    sleep 30
done
EOF

# 后台运行内存监控
./monitor_memory.sh &
monitor_pid=$!

# 内存优化配置
milvus-ingest generate \
    --builtin documents \
    --total-rows 5000000 \
    --batch-size 15000 \
    --file-size 64MB \
    --rows-per-file 100000 \
    --format parquet \
    --out ./documents_memory_optimized

# 停止监控
kill $monitor_pid
```

**结果:**
- 峰值内存使用: 6.8GB (未超过8GB限制)
- 生成时间: 68分钟
- 输出文件: 50个分片，总大小3.2GB

## 📈 性能监控和分析

### 实时性能监控

```bash
# 创建性能监控脚本
cat > performance_monitor.py << 'EOF'
#!/usr/bin/env python3
"""实时性能监控"""

import psutil
import time
import sys
import json
from datetime import datetime

def monitor_performance(duration_seconds=300):
    """监控系统性能"""
    
    start_time = time.time()
    metrics = []
    
    print(f"开始性能监控 (持续 {duration_seconds} 秒)...")
    print("时间\t\tCPU%\t内存%\t内存GB\t磁盘IO MB/s")
    print("-" * 60)
    
    while time.time() - start_time < duration_seconds:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        if hasattr(monitor_performance, 'prev_disk_io'):
            read_mb = (disk_io.read_bytes - monitor_performance.prev_disk_io.read_bytes) / (1024**2)
            write_mb = (disk_io.write_bytes - monitor_performance.prev_disk_io.write_bytes) / (1024**2)
            io_mb = read_mb + write_mb
        else:
            io_mb = 0
        
        monitor_performance.prev_disk_io = disk_io
        
        # 记录指标
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp}\t{cpu_percent:5.1f}\t{memory_percent:5.1f}\t{memory_gb:5.1f}\t{io_mb:8.1f}")
        
        metrics.append({
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_gb": memory_gb,
            "disk_io_mb_s": io_mb
        })
        
        time.sleep(5)
    
    # 保存指标
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 统计摘要
    avg_cpu = sum(m["cpu_percent"] for m in metrics) / len(metrics)
    max_memory = max(m["memory_gb"] for m in metrics)
    avg_io = sum(m["disk_io_mb_s"] for m in metrics) / len(metrics)
    
    print(f"\n性能摘要:")
    print(f"  平均CPU使用: {avg_cpu:.1f}%")
    print(f"  峰值内存使用: {max_memory:.1f}GB")
    print(f"  平均磁盘IO: {avg_io:.1f}MB/s")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    monitor_performance(duration)
EOF

chmod +x performance_monitor.py
```

### 使用性能监控

```bash
# 后台启动性能监控
python performance_monitor.py 600 &  # 监控10分钟
monitor_pid=$!

# 运行数据生成
milvus-ingest generate \
    --builtin ecommerce \
    --total-rows 1000000 \
    --batch-size 50000 \
    --out ./monitored_generation

# 等待监控完成
wait $monitor_pid

# 查看性能报告
cat performance_metrics.json | jq '.[-10:]'  # 查看最后10个指标
```

### 性能瓶颈诊断

```bash
# 创建瓶颈诊断脚本
cat > diagnose_bottleneck.py << 'EOF'
#!/usr/bin/env python3
"""性能瓶颈诊断"""

import json
import statistics

def analyze_bottleneck(metrics_file="performance_metrics.json"):
    """分析性能瓶颈"""
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    if not metrics:
        print("没有找到性能指标数据")
        return
    
    # 计算统计数据
    cpu_values = [m["cpu_percent"] for m in metrics]
    memory_values = [m["memory_percent"] for m in metrics]
    io_values = [m["disk_io_mb_s"] for m in metrics]
    
    avg_cpu = statistics.mean(cpu_values)
    avg_memory = statistics.mean(memory_values)
    avg_io = statistics.mean(io_values)
    
    max_cpu = max(cpu_values)
    max_memory = max(memory_values)
    max_io = max(io_values)
    
    print("🔍 性能瓶颈分析:")
    print(f"  CPU 使用: 平均 {avg_cpu:.1f}%, 峰值 {max_cpu:.1f}%")
    print(f"  内存使用: 平均 {avg_memory:.1f}%, 峰值 {max_memory:.1f}%")
    print(f"  磁盘IO: 平均 {avg_io:.1f}MB/s, 峰值 {max_io:.1f}MB/s")
    
    # 瓶颈判断
    bottlenecks = []
    
    if max_cpu > 90:
        bottlenecks.append("CPU")
        print(f"  ⚠️  CPU瓶颈: 峰值使用率{max_cpu:.1f}%")
        print(f"    建议: 减少batch_size或使用更多CPU核心")
    
    if max_memory > 85:
        bottlenecks.append("Memory")
        print(f"  ⚠️  内存瓶颈: 峰值使用率{max_memory:.1f}%")
        print(f"    建议: 减少batch_size或启用文件分割")
    
    if avg_io > 100:  # MB/s
        bottlenecks.append("Disk I/O")
        print(f"  ⚠️  磁盘IO瓶颈: 平均{avg_io:.1f}MB/s")
        print(f"    建议: 使用SSD或增大文件大小")
    
    if not bottlenecks:
        print("  ✅ 未发现明显瓶颈，性能表现良好")
        
        # 优化建议
        if avg_cpu < 50:
            print(f"  💡 CPU利用率较低({avg_cpu:.1f}%), 可以增加batch_size")
        
        if avg_memory < 50:
            print(f"  💡 内存利用率较低({avg_memory:.1f}%), 可以增加batch_size")

if __name__ == "__main__":
    analyze_bottleneck()
EOF

chmod +x diagnose_bottleneck.py

# 运行瓶颈诊断
python diagnose_bottleneck.py
```

## 🎯 优化检查清单

### 生成前检查

- [ ] **硬件配置充足** - CPU核心数、内存大小、存储空间
- [ ] **选择合适模式** - 根据业务需求选择最适合的内置模式
- [ ] **估算资源需求** - 预估内存使用和磁盘空间
- [ ] **环境依赖就绪** - Python版本、依赖包、权限设置

### 参数配置检查

- [ ] **批处理大小** - 根据内存大小选择合适的batch_size
- [ ] **文件分割策略** - 根据下游处理选择file_size和rows_per_file
- [ ] **输出格式** - Parquet用于性能，JSON用于调试
- [ ] **进度显示** - 生产环境可以禁用progress提升性能

### 生成中监控

- [ ] **资源使用监控** - CPU、内存、磁盘使用率
- [ ] **生成速度监控** - 行/秒生成速度是否符合预期
- [ ] **错误日志监控** - 是否有警告或错误信息
- [ ] **磁盘空间监控** - 确保有足够空间存储输出

### 生成后验证

- [ ] **数据完整性** - 验证生成行数和文件数量
- [ ] **文件格式** - 确认输出文件可以正常读取
- [ ] **性能指标** - 记录生成时间、速度等关键指标
- [ ] **资源清理** - 清理临时文件和测试数据

## 🚀 进阶优化技巧

### 1. 并行生成 (实验性)

对于超大数据集，可以考虑并行生成：

```bash
# 将大任务分解为多个小任务并行执行
generate_parallel() {
    local total_rows=$1
    local parallel_jobs=$2
    local rows_per_job=$((total_rows / parallel_jobs))
    
    for i in $(seq 0 $((parallel_jobs - 1))); do
        local start_seed=$((42 + i * 1000))
        milvus-ingest generate \
            --builtin ecommerce \
            --total-rows $rows_per_job \
            --seed $start_seed \
            --out ./parallel_job_$i \
            --batch-size 50000 &
    done
    
    wait  # 等待所有任务完成
    
    # 合并结果（需要自定义脚本）
    # merge_datasets ./parallel_job_* ./final_dataset
}

# 使用4个并行任务生成400万行数据
generate_parallel 4000000 4
```

### 2. 内存映射优化

对于超大文件，使用内存映射可以提升性能：

```bash
# 使用更大的系统缓存
echo 3 > /proc/sys/vm/drop_caches  # 清理缓存
echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf  # 增加文件缓存
```

### 3. 存储优化

```bash
# SSD优化
echo 'noop' > /sys/block/nvme0n1/queue/scheduler  # 使用noop调度器

# 文件系统优化 (ext4)
mount -o noatime,data=writeback /dev/nvme0n1 /output_dir
```

## 📊 性能基准对比

### 不同配置的性能对比

| 配置 | 批处理大小 | 文件大小 | 100万行用时 | 峰值内存 | 输出大小 |
|------|-----------|----------|-------------|----------|----------|
| 默认 | 50000 | 256MB | 87秒 | 1.8GB | 485MB |
| 内存优化 | 25000 | 128MB | 105秒 | 1.2GB | 485MB |
| 速度优化 | 100000 | 512MB | 72秒 | 2.8GB | 485MB |
| 平衡 | 75000 | 256MB | 78秒 | 2.1GB | 485MB |

### 硬件影响对比

| 硬件配置 | 生成速度 | 适合规模 | 推荐批处理 |
|----------|----------|----------|------------|
| 4核/8GB | 8,000行/秒 | <100万行 | 25000 |
| 8核/16GB | 15,000行/秒 | <500万行 | 50000 |
| 16核/32GB | 25,000行/秒 | <2000万行 | 100000 |
| 32核/64GB | 35,000行/秒 | 任意规模 | 150000 |

通过系统化的性能调优，你可以将数据生成效率提升2-5倍，在相同硬件条件下处理更大规模的数据集！
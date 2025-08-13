# clean 命令详解

`clean` 命令用于清理工具生成的文件和缓存，帮助释放磁盘空间和重置工作环境。

## 基本语法

```bash
milvus-ingest clean [OPTIONS]
```

## 清理范围

### 默认清理目标
运行 `clean` 命令时，会清理以下内容：

1. **生成的数据文件**
   - 当前目录下所有生成的数据目录
   - 临时文件和中间文件

2. **工作缓存**
   - 生成过程中的临时缓存
   - 性能分析文件（如果存在）

3. **日志文件**
   - 生成过程产生的日志文件
   - 错误日志和调试信息

### 不会清理的内容
- 用户自定义的模式文件 (`~/.milvus-ingest/schemas/`)
- 系统配置文件
- 手动创建的非标准目录

## 选项详解

### --yes
跳过确认提示，自动清理

```bash
# 交互式清理（默认）
milvus-ingest clean

# 自动清理，无需确认
milvus-ingest clean --yes
```

### --target PATH
指定特定的清理目标路径

```bash
# 清理特定目录
milvus-ingest clean --target ./old_data

# 清理多个目录
milvus-ingest clean --target ./data1 --target ./data2
```

### --recursive
递归清理子目录

```bash
# 递归清理当前目录及子目录
milvus-ingest clean --recursive

# 递归清理指定目录
milvus-ingest clean --target ./datasets --recursive
```

### --dry-run
预览清理操作，不实际执行

```bash
# 查看会清理哪些文件
milvus-ingest clean --dry-run

# 预览特定目录的清理
milvus-ingest clean --target ./data --dry-run
```

### --include-cache
包含清理工具缓存目录

```bash
# 清理包括缓存
milvus-ingest clean --include-cache --yes
```

## 使用示例

### 1. 基础清理

```bash
# 交互式清理（推荐）
milvus-ingest clean
```

输出示例：
```
发现以下可清理的内容:

数据目录:
├── ./simple_data/          (45.2 MB, 3 个文件)
├── ./ecommerce_data/       (156.7 MB, 5 个文件)
└── ./test_output/          (12.3 MB, 2 个文件)

临时文件:
├── ./.milvus_temp_12345/   (2.1 MB)
└── ./generation.log        (1.5 MB)

总计: 217.8 MB

确认清理上述文件? [y/N]: y
正在清理...
[SUCCESS] 清理完成，释放空间: 217.8 MB
```

### 2. 自动清理

```bash
# 脚本中使用，跳过确认
milvus-ingest clean --yes
```

### 3. 预览清理内容

```bash
# 查看会清理什么，但不实际清理
milvus-ingest clean --dry-run
```

输出示例：
```
[DRY RUN] 以下文件将被清理:

数据目录:
└── ./simple_data/
    ├── data.parquet         (42.5 MB)
    ├── meta.json           (1.2 KB)
    └── index_suggestions.json (890 B)

临时文件:
└── ./generation.log         (1.5 MB)

[DRY RUN] 总计将释放: 44.0 MB
[DRY RUN] 实际清理请运行: milvus-ingest clean --yes
```

### 4. 清理特定目录

```bash
# 只清理指定的数据目录
milvus-ingest clean --target ./old_experiment_data --yes

# 清理多个特定目录
milvus-ingest clean \
  --target ./test_data_1 \
  --target ./test_data_2 \
  --target ./experiments \
  --yes
```

### 5. 深度清理

```bash
# 包含缓存的完整清理
milvus-ingest clean --include-cache --recursive --yes
```

## 清理策略

### 开发环境清理
```bash
# 每日清理策略（脚本）
#!/bin/bash
echo "执行每日清理..."
milvus-ingest clean --yes
echo "清理完成"
```

### 实验环境清理
```bash
# 实验结束后清理
milvus-ingest clean --target ./experiment_* --recursive --yes
```

### 生产环境清理
```bash
# 谨慎清理，先预览
milvus-ingest clean --dry-run
# 确认后再执行
milvus-ingest clean --yes
```

## 高级用法

### 1. 结合其他命令的工作流

```bash
# 实验工作流：生成-测试-清理
function run_experiment() {
    local schema=$1
    local rows=$2

    echo "开始实验: $schema"

    # 生成数据
    milvus-ingest generate --builtin $schema --total-rows $rows --out ./exp_${schema}

    # 测试导入
    milvus-ingest to-milvus insert ./exp_${schema} --collection-name test_${schema}

    # 清理数据
    milvus-ingest clean --target ./exp_${schema} --yes

    echo "实验完成: $schema"
}

# 批量实验
for schema in simple ecommerce documents; do
    run_experiment $schema 10000
done
```

### 2. 自动化清理脚本

```bash
#!/bin/bash
# 自动清理脚本 - cleanup.sh

set -e

echo "开始自动清理任务..."

# 清理超过7天的数据目录
find . -maxdepth 1 -type d -name "*_data" -mtime +7 -exec echo "发现旧目录: {}" \;
find . -maxdepth 1 -type d -name "*_data" -mtime +7 -exec rm -rf {} \;

# 清理工具生成的文件
milvus-ingest clean --yes

# 清理大于100MB的临时文件
find . -name "*.tmp" -size +100M -delete

echo "自动清理完成"
```

### 3. 安全清理检查

```bash
#!/bin/bash
# 安全清理脚本 - safe_cleanup.sh

# 检查重要文件
if [ -f "./important_data/meta.json" ]; then
    echo "警告: 发现重要数据，跳过清理"
    exit 1
fi

# 预览清理内容
echo "预览清理内容:"
milvus-ingest clean --dry-run

# 确认清理
read -p "确认执行清理? (y/N): " confirm
if [[ $confirm == [yY] ]]; then
    milvus-ingest clean --yes
    echo "清理完成"
else
    echo "清理已取消"
fi
```

## 清理文件类型

### 数据文件
- `*.parquet` - Parquet 数据文件
- `*.json` - JSON 数据文件
- `meta.json` - 元数据文件
- `index_suggestions.json` - 索引建议文件

### 临时文件
- `.milvus_temp_*` - 生成过程临时目录
- `generation.log` - 生成过程日志
- `*.tmp` - 各种临时文件

### 缓存文件
- `~/.milvus-ingest/cache/` - 工具缓存目录
- 性能分析文件
- 编译缓存

## 错误处理

### 常见错误及解决

#### 1. 权限错误
```
错误: [PermissionError] Permission denied: './data/file.parquet'
```

**解决方案:**
```bash
# 检查文件权限
ls -la ./data/

# 修改权限
chmod -R 755 ./data/

# 或使用 sudo (谨慎)
sudo milvus-ingest clean --yes
```

#### 2. 文件被占用
```
错误: [OSError] File is being used by another process
```

**解决方案:**
```bash
# 检查占用进程
lsof +D ./data/

# 停止相关进程后重试
milvus-ingest clean --yes
```

#### 3. 磁盘空间不足
```
错误: [OSError] No space left on device
```

**解决方案:**
```bash
# 强制清理释放空间
milvus-ingest clean --yes --include-cache

# 手动删除大文件
find . -size +1G -type f -delete
```

## 性能和存储

### 清理效果统计
清理操作通常可以释放：

| 数据规模 | 典型文件大小 | 清理后释放空间 |
|----------|--------------|----------------|
| 1万行 | ~5 MB | ~5 MB |
| 10万行 | ~50 MB | ~50 MB |
| 100万行 | ~500 MB | ~500 MB |
| 1000万行 | ~5 GB | ~5 GB |

### 清理速度
- **小文件** (<10 MB): 几乎瞬间完成
- **中等文件** (10 MB - 1 GB): 1-10 秒
- **大文件** (>1 GB): 可能需要数分钟

## 最佳实践

### 1. 定期清理
```bash
# 添加到 crontab 每日清理
# 0 2 * * * cd /path/to/workspace && milvus-ingest clean --yes
```

### 2. 清理前备份重要数据
```bash
# 备份重要实验结果
cp -r ./important_experiment_data ./backups/

# 然后清理
milvus-ingest clean --yes
```

### 3. 使用 .gitignore 避免版本控制
```bash
# 在 .gitignore 中添加
*_data/
*.parquet
generation.log
.milvus_temp_*
```

### 4. 项目结束时完整清理
```bash
# 项目结束清理脚本
milvus-ingest clean --include-cache --recursive --yes
rm -rf ./experiments/
rm -rf ./temp_data/
echo "项目环境已重置"
```

## 相关命令

- [`generate`](generate.md) - 生成需要清理的数据文件
- 所有其他命令都可能产生需要清理的临时文件

---

**提示**: 定期使用 `clean` 命令可以保持工作环境整洁，避免磁盘空间不足的问题。在重要实验前建议先清理环境以确保充足的存储空间。

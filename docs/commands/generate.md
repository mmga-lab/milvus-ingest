# generate 命令详解

`generate` 命令是 milvus-ingest 的核心功能，用于生成高性能的 Milvus 模拟数据。

## 基本语法

```bash
milvus-ingest generate [OPTIONS]
```

## 数据源选项 (二选一，必需)

### --builtin SCHEMA_ID
使用内置或管理的数据模式

```bash
# 使用内置模式
milvus-ingest generate --builtin product_catalog --total-rows 1000

# 使用自定义管理模式  
milvus-ingest generate --builtin my_products --total-rows 1000
```

**可用的内置模式:**
- `product_catalog` - 产品目录 (带auto_id的基础示例)
- `ecommerce_search` - 电商搜索 (包含可空字段的产品搜索)
- `news_articles` - 新闻文章 (支持动态字段的新闻内容)
- `document_search` - 文档搜索 (稀疏向量+BM25的文档检索)
- `multi_tenant_data` - 多租户数据 (带分区的多租户系统)
- `multimedia_content` - 多媒体内容 (多种向量类型的媒体库)

### --schema PATH
使用自定义模式文件 (JSON/YAML)

```bash
milvus-ingest generate --schema my_schema.json --total-rows 1000
milvus-ingest generate --schema my_schema.yaml --total-rows 1000
```

## 基本选项

### --total-rows INTEGER
设置生成的数据行数 (默认: 1000)

```bash
# 小规模测试
milvus-ingest generate --builtin product_catalog --total-rows 1000

# 中等规模
milvus-ingest generate --builtin ecommerce_search --total-rows 100000

# 大规模数据集
milvus-ingest generate --builtin ecommerce_search --total-rows 5000000
```

### --out DIRECTORY  
指定输出目录 (默认: 基于模式名称)

```bash
# 自定义输出目录
milvus-ingest generate --builtin ecommerce --total-rows 10000 --out ./my_data

# 强制覆盖已存在目录
milvus-ingest generate --builtin simple --total-rows 1000 --out ./data --force
```

### --format {parquet,json}
设置输出格式 (默认: parquet)

```bash
# Parquet 格式 (推荐，最快的I/O性能)
milvus-ingest generate --builtin simple --total-rows 10000 --format parquet

# JSON 格式 (标准数组格式 [{}...], 便于调试和与Milvus bulk import兼容)
milvus-ingest generate --builtin simple --total-rows 10000 --format json

# 动态字段示例 (推荐JSON格式便于查看$meta字段内容)
milvus-ingest generate --builtin dynamic_example --total-rows 1000 --format json
```

## 预览和验证选项

### --preview
显示前5行数据，不生成完整数据集

```bash
# 快速预览数据结构
milvus-ingest generate --builtin ecommerce --total-rows 100000 --preview

# 验证自定义模式
milvus-ingest generate --schema my_schema.json --total-rows 1000 --preview
```

### --validate-only
仅验证模式，不生成数据

```bash
# 验证内置模式
milvus-ingest generate --builtin simple --validate-only

# 验证自定义模式
milvus-ingest generate --schema my_schema.json --validate-only
```

## 性能调优选项

### --batch-size INTEGER
设置批处理大小 (默认: 50000)

```bash
# 小内存环境
milvus-ingest generate --builtin simple --total-rows 100000 --batch-size 10000

# 高性能环境 (推荐)
milvus-ingest generate --builtin ecommerce --total-rows 1000000 --batch-size 100000

# 极大数据集
milvus-ingest generate --builtin ecommerce --total-rows 10000000 --batch-size 200000
```

**批处理大小选择指南:**
- **小型数据集** (<10K 行): 使用默认值
- **中型数据集** (10K-100K 行): 25000-50000
- **大型数据集** (100K-1M 行): 50000-100000  
- **超大数据集** (>1M 行): 100000-200000

### --file-size TEXT
设置文件大小限制 (支持单位如 '10GB', '256MB', 默认: 256MB)

```bash
# 较小的文件便于处理
milvus-ingest generate --builtin ecommerce --total-rows 1000000 --file-size 128MB

# 较大的文件减少文件数量
milvus-ingest generate --builtin ecommerce --total-rows 5000000 --file-size 512MB
```

### --rows-per-file INTEGER  
设置单文件最大行数 (默认: 1000000)

```bash
# 更小的文件分片
milvus-ingest generate --builtin ecommerce --total-rows 5000000 --rows-per-file 500000

# 更大的文件分片
milvus-ingest generate --builtin ecommerce --total-rows 10000000 --rows-per-file 2000000
```

### --file-count INTEGER
目标文件数量 (与 --file-size 一起使用时，会覆盖 --total-rows)

```bash
# 生成10个1GB的文件
milvus-ingest generate --builtin ecommerce --file-count 10 --file-size 1GB

# 生成100个200MB的文件进行测试
milvus-ingest generate --builtin documents --file-count 100 --file-size 200MB
```

### --partitions INTEGER
设置 Milvus 分区数量 (需要模式中有分区键字段)

```bash
# 8个分区的数据集
milvus-ingest generate --builtin ecommerce_partitioned --total-rows 1000000 --partitions 8

# 大规模分区测试
milvus-ingest generate --builtin ecommerce_partitioned --total-rows 10000000 --partitions 1024
```

### --shards INTEGER
设置分片数量 (VChannels)，基于主键哈希分布数据

```bash
# 4个分片的数据集
milvus-ingest generate --builtin simple --total-rows 1000000 --shards 4

# 分区与分片组合
milvus-ingest generate --builtin ecommerce_partitioned --total-rows 5000000 --partitions 8 --shards 16
```

### --workers INTEGER
并行工作进程数 (默认: CPU核心数)

```bash
# 使用8个并行进程加速生成
milvus-ingest generate --builtin ecommerce --total-rows 10000000 --workers 8

# 限制资源使用
milvus-ingest generate --builtin documents --total-rows 5000000 --workers 2
```

## 其他选项

### --seed INTEGER
设置随机种子，确保结果可重现

```bash
# 可重现的数据生成
milvus-ingest generate --builtin simple --total-rows 10000 --seed 42

# 每次运行生成相同数据
milvus-ingest generate --builtin ecommerce --total-rows 100000 --seed 123 --out ./reproducible_data
```

### --no-progress
禁用进度条显示

```bash
# 在脚本中使用，避免输出干扰
milvus-ingest generate --builtin simple --total-rows 100000 --no-progress
```


### --force
强制覆盖现有输出目录

```bash
# 强制覆盖已存在的目录
milvus-ingest generate --builtin simple --total-rows 10000 --out ./existing_dir --force
```

## 完整示例

### 1. 快速开始示例

```bash
# 基础预览
milvus-ingest generate --builtin simple --total-rows 1000 --preview

# 小规模测试数据
milvus-ingest generate --builtin ecommerce --total-rows 10000 --out ./test_data
```

### 2. 高性能大规模生成

```bash
# 生成100万行电商数据，优化性能设置
milvus-ingest generate \
  --builtin ecommerce \
  --total-rows 1000000 \
  --batch-size 100000 \
  --file-size 256MB \
  --rows-per-file 500000 \
  --out ./big_ecommerce_data \
  --seed 42
```

### 3. 自定义模式使用

```bash
# 验证自定义模式
milvus-ingest generate --schema ./schemas/my_products.json --validate-only

# 生成自定义数据
milvus-ingest generate \
  --schema ./schemas/my_products.json \
  --total-rows 50000 \
  --format parquet \
  --out ./custom_products \
  --preview
```

### 4. 批量生成不同数据集

```bash
# 生成多个不同类型的数据集
for schema in simple ecommerce documents users; do
  milvus-ingest generate \
    --builtin $schema \
    --total-rows 100000 \
    --out ./datasets/$schema \
    --seed 42
done
```

## 性能优化建议

### 硬件配置
- **CPU**: 4+ 核心现代处理器
- **内存**: 大数据集(>1M行)建议8GB+
- **存储**: SSD 推荐，提升 I/O 性能

### 最佳实践

1. **大数据集使用 Parquet 格式**
   ```bash
   --format parquet  # 最佳 I/O 性能
   ```

2. **调整批处理大小**
   ```bash
   --batch-size 100000  # 大规模数据集
   --batch-size 10000   # 内存受限环境
   ```

3. **启用文件分割**
   ```bash
   --file-size 256MB --rows-per-file 1000000  # 便于处理
   ```

4. **使用随机种子**
   ```bash
   --seed 42  # 确保可重现
   ```

## 输出结构

生成的数据会创建一个目录，包含：

```
output_directory/
├── data.parquet          # 主数据文件(小数据集)
│   或
├── data_part_1.parquet   # 分片数据文件(大数据集)
├── data_part_2.parquet
├── ...
└── meta.json             # 集合元数据
```

`meta.json` 包含：
- 集合配置信息
- 字段定义
- 生成统计
- 索引建议

## 错误处理

### 常见错误及解决方案

1. **内存不足错误**
   ```bash
   # 解决: 减少批处理大小
   --batch-size 10000 --rows-per-file 100000
   ```

2. **模式验证失败**
   ```bash
   # 解决: 先验证模式
   --validate-only
   ```

3. **输出目录已存在**
   ```bash
   # 解决: 使用强制覆盖
   --force
   ```

4. **性能缓慢**
   ```bash
   # 解决: 增加批处理大小
   --batch-size 100000 --format parquet
   ```

## 相关命令

- [`schema list`](schema.md#list) - 查看所有可用模式
- [`schema show`](schema.md#show) - 查看模式详情
- [`to-milvus insert`](to-milvus.md#insert) - 直接导入到 Milvus
- [`upload`](upload.md) - 上传到 S3/MinIO

---

**下一步**: 查看 [模式管理命令](schema.md) 了解如何管理数据模式
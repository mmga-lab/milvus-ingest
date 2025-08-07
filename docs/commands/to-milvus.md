# to-milvus 命令详解

`to-milvus` 命令组提供将生成的数据导入到 Milvus 向量数据库的功能，支持两种导入方式：直接插入 (`insert`) 和批量导入 (`import`)。

## 基本语法

```bash
milvus-ingest to-milvus <SUBCOMMAND> [OPTIONS]
```

## 子命令总览

| 子命令 | 功能 | 适用场景 | 性能特点 |
|--------|------|----------|----------|
| `insert` | 直接插入数据 | 中小规模数据集 (<1M 行) | 同步操作，实时反馈 |
| `import` | 批量导入数据 | 大规模数据集 (>1M 行) | 异步操作，高性能 |
| `verify` | 验证数据完整性 | 导入后的数据验证 | 3级验证，支持查询测试 |

---

## insert - 直接插入数据

直接从本地数据文件（Parquet和JSON格式）读取并插入到 Milvus 集合中，包括自动创建集合和索引。

### 语法
```bash
milvus-ingest to-milvus insert DATA_PATH [OPTIONS]
```

### 必需参数
- `DATA_PATH`: 数据目录路径（由 `generate` 命令生成）

### 连接选项

#### --uri URI
Milvus 服务器地址（默认: http://localhost:19530）

```bash
# 本地 Milvus
milvus-ingest to-milvus insert ./data --uri http://localhost:19530

# 远程 Milvus
milvus-ingest to-milvus insert ./data --uri http://192.168.1.100:19530

# Milvus Cloud
milvus-ingest to-milvus insert ./data --uri https://your-cluster.vectordb.zillizcloud.com
```

#### --token TOKEN
认证令牌（用于 Milvus Cloud 或启用认证的实例）

```bash
# 使用认证令牌
milvus-ingest to-milvus insert ./data \
  --uri https://your-cluster.vectordb.zillizcloud.com \
  --token your_api_token
```

#### --db-name DATABASE
数据库名称（默认: default）

```bash
# 指定数据库
milvus-ingest to-milvus insert ./data \
  --uri http://milvus:19530 \
  --db-name production
```

### 集合配置选项

#### --collection-name NAME
覆盖集合名称（默认使用 meta.json 中的名称）

```bash
# 自定义集合名称
milvus-ingest to-milvus insert ./ecommerce_data \
  --collection-name product_vectors_v2
```

#### --drop-if-exists
如果集合已存在则删除重建

```bash
# 强制重建集合
milvus-ingest to-milvus insert ./data \
  --collection-name test_collection \
  --drop-if-exists
```

### 性能选项

#### --batch-size SIZE
插入批次大小（默认: 1000）

```bash
# 小批次（内存受限环境）
milvus-ingest to-milvus insert ./data --batch-size 500

# 大批次（高性能环境）
milvus-ingest to-milvus insert ./data --batch-size 5000
```

### 完整示例

#### 1. 基础本地插入
```bash
# 生成Parquet格式数据
milvus-ingest generate --builtin simple --total-rows 10000 --out ./simple_data

# 生成JSON格式数据
milvus-ingest generate --builtin simple --total-rows 10000 --format json --out ./simple_json

# 插入到本地 Milvus (自动检测文件格式)
milvus-ingest to-milvus insert ./simple_data
milvus-ingest to-milvus insert ./simple_json
```

#### 2. 远程 Milvus 插入
```bash
# 插入到远程 Milvus
milvus-ingest to-milvus insert ./ecommerce_data \
  --uri http://192.168.1.100:19530 \
  --collection-name products \
  --batch-size 2000
```

#### 3. Milvus Cloud 插入
```bash
# 插入到 Milvus Cloud
milvus-ingest to-milvus insert ./user_data \
  --uri https://your-cluster.vectordb.zillizcloud.com \
  --token your_api_token \
  --db-name production \
  --collection-name user_profiles
```

#### 4. 开发测试流程
```bash
# 开发测试：重复插入相同数据
milvus-ingest to-milvus insert ./test_data \
  --collection-name test_collection \
  --drop-if-exists \
  --batch-size 1000
```

---

## import - 批量导入数据

使用 Milvus 批量导入 API 从 S3/MinIO 导入预上传的数据文件（Parquet和JSON格式），支持高性能异步导入。

### 语法
```bash
milvus-ingest to-milvus import [OPTIONS]
```

### 数据源选项

#### 本地数据（自动上传模式）
```bash
--local-path PATH --s3-path PATH --bucket BUCKET
```

这种模式会自动将本地数据上传到 S3/MinIO，然后执行导入。

```bash
# 自动上传并导入
milvus-ingest to-milvus import \
  --local-path ./ecommerce_data \
  --s3-path datasets/ecommerce/ \
  --bucket milvus-data \
  --endpoint-url http://minio:9000
```

#### 预上传数据模式
直接指定已上传的 S3 文件路径：

```bash
# 从已上传的数据导入
milvus-ingest to-milvus import \
  --files s3://bucket/data/file1.parquet \
  --files s3://bucket/data/file2.parquet \
  --collection-name my_collection
```

### S3/MinIO 连接选项

#### --endpoint-url URL
S3 服务端点（MinIO 或其他 S3 兼容服务）

```bash
# MinIO 本地部署
--endpoint-url http://localhost:9000

# MinIO 远程部署  
--endpoint-url http://minio.company.com:9000

# AWS S3 (可选，不指定则使用默认)
--endpoint-url https://s3.amazonaws.com
```

#### --access-key-id / --secret-access-key
S3 访问凭证

```bash
# 指定凭证
milvus-ingest to-milvus import \
  --local-path ./data \
  --s3-path collections/test/ \
  --bucket test-bucket \
  --endpoint-url http://minio:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin
```

### 集合配置选项

#### --collection-name NAME
指定集合名称（覆盖 meta.json 中的名称）

```bash
# 自定义集合名称
milvus-ingest to-milvus import \
  --local-path ./data \
  --s3-path datasets/products/ \
  --bucket milvus-data \
  --collection-name product_catalog_v2
```

#### --drop-if-exists
删除已存在的集合并重建

```bash
# 强制重建集合
milvus-ingest to-milvus import \
  --local-path ./data \
  --s3-path datasets/test/ \
  --bucket test-bucket \
  --collection-name test_collection \
  --drop-if-exists
```

### 导入控制选项

#### --wait
等待导入完成（默认为异步操作）

```bash
# 等待导入完成
milvus-ingest to-milvus import \
  --local-path ./large_dataset \
  --s3-path datasets/large/ \
  --bucket production-data \
  --wait
```

#### --timeout SECONDS
等待超时时间（与 --wait 配合使用，默认: 300 秒）

```bash
# 设置更长的超时时间
milvus-ingest to-milvus import \
  --local-path ./huge_dataset \
  --s3-path datasets/huge/ \
  --bucket production-data \
  --wait \
  --timeout 1800  # 30 分钟
```

### 完整示例

#### 1. 一步式导入（推荐）
```bash
# 生成Parquet格式数据（默认）
milvus-ingest generate --builtin ecommerce --total-rows 1000000 --out ./ecommerce

# 生成JSON格式数据（与Milvus bulk import兼容）
milvus-ingest generate --builtin ecommerce --total-rows 1000000 --format json --out ./ecommerce_json

# 自动上传并导入（支持两种格式）
milvus-ingest to-milvus import \
  --local-path ./ecommerce \
  --s3-path datasets/ecommerce/ \
  --bucket milvus-data \
  --endpoint-url http://minio:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin \
  --wait
```

#### 2. 分步式导入
```bash
# 步骤1: 生成数据
milvus-ingest generate --builtin documents --total-rows 2000000 --out ./documents

# 步骤2: 上传数据
milvus-ingest upload ./documents s3://milvus-data/documents/ \
  --endpoint-url http://minio:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin

# 步骤3: 导入数据
milvus-ingest to-milvus import \
  --files s3://milvus-data/documents/data.parquet \
  --collection-name document_search \
  --wait \
  --timeout 600
```

#### 3. 生产环境大规模导入
```bash
# 生成大规模数据集
milvus-ingest generate \
  --builtin ecommerce \
  --total-rows 10000000 \
  --file-size 512MB \
  --rows-per-file 1000000 \
  --out ./big_ecommerce

# 高性能导入
milvus-ingest to-milvus import \
  --local-path ./big_ecommerce \
  --s3-path production/ecommerce_v3/ \
  --bucket production-milvus-data \
  --endpoint-url http://minio-cluster:9000 \
  --access-key-id $PROD_ACCESS_KEY \
  --secret-access-key $PROD_SECRET_KEY \
  --collection-name ecommerce_products_v3 \
  --drop-if-exists \
  --wait \
  --timeout 3600  # 1 小时超时
```

---

## 两种导入方式对比

### 何时使用 insert

**适用场景:**
- **小到中等规模数据** (<100万行)
- **开发和测试环境**
- **需要即时反馈的场景**
- **不需要 S3/MinIO 的简单部署**

**优势:**
- 简单易用，一条命令完成
- 实时进度反馈
- 自动集合创建和索引
- 无需额外存储服务

**限制:**
- 性能相对较低
- 内存使用较多
- 不适合超大数据集

### 何时使用 import

**适用场景:**
- **大规模数据集** (>100万行)
- **生产环境批量导入**
- **需要最佳性能的场景**
- **已有 S3/MinIO 基础设施**

**优势:**
- 高性能异步导入
- 支持超大数据集
- 内存使用效率高
- 可以暂停/恢复导入

**限制:**
- 需要 S3/MinIO 服务
- 配置相对复杂
- 异步操作，需要监控

### 性能对比

| 数据规模 | insert 用时 | import 用时 | 推荐方式 |
|----------|-------------|-------------|----------|
| 1万行 | ~30秒 | ~45秒 | insert |
| 10万行 | ~5分钟 | ~3分钟 | insert |
| 100万行 | ~45分钟 | ~8分钟 | import |
| 1000万行 | >4小时 | ~30分钟 | import |

## 集合管理

### 自动集合创建
两种导入方式都会自动根据 `meta.json` 创建集合：

1. **解析字段定义**：从元数据中读取字段类型和配置
2. **创建集合**：使用合适的字段定义创建集合
3. **建立索引**：为向量字段自动创建索引
4. **加载集合**：使集合可用于搜索

### 索引策略
工具会自动为不同维度的向量选择合适的索引：

```bash
# 向量维度 < 1024: 使用 IVF_FLAT
# 向量维度 >= 1024: 使用 IVF_SQ8
# 二进制向量: 使用 BIN_IVF_FLAT
```

### 集合配置示例
```json
// meta.json 中的集合配置
{
  "collection_name": "ecommerce_products",
  "description": "电商产品向量搜索集合",
  "fields": [...],
  "auto_id": true,
  "enable_dynamic_field": false
}
```

## 监控和故障排除

### 插入进度监控
```bash
# insert 命令会显示实时进度
正在插入数据到集合 'test_collection'...
━━━━━━━━━━━━━━━━━━━━ 100% 10000/10000 行 @ 1250 行/秒
[SUCCESS] 成功插入 10000 行数据
[INFO] 集合统计: 10000 条记录
```

### 导入作业监控
```bash
# import 命令的异步监控
[INFO] 导入作业已提交: job_1642234567890
[INFO] 作业状态: Pending -> Importing -> Completed
[SUCCESS] 导入完成: 1000000 行数据已成功导入
```

### 常见错误及解决

#### 1. 连接错误
```
错误: [ConnectionError] Failed to connect to Milvus at http://localhost:19530
```

**解决方案:**
```bash
# 检查 Milvus 服务状态
curl http://localhost:19530/healthz

# 检查 URI 格式
--uri http://localhost:19530  # 正确
--uri localhost:19530         # 错误，缺少协议
```

#### 2. 认证错误
```
错误: [AuthenticationError] Invalid token
```

**解决方案:**
```bash
# 检查令牌有效性
--token your_valid_token

# 确认 Milvus 实例是否需要认证
```

#### 3. 集合已存在
```
错误: [CollectionExistsError] Collection 'test' already exists
```

**解决方案:**
```bash
# 使用不同的集合名称
--collection-name test_v2

# 或删除重建
--drop-if-exists
```

#### 4. S3 访问错误（import 模式）
```
错误: [S3Error] Access denied to bucket 'milvus-data'
```

**解决方案:**
```bash
# 检查 S3 凭证
--access-key-id correct_key --secret-access-key correct_secret

# 检查存储桶权限
# 确保凭证有读写权限
```

## 最佳实践

### 1. 数据准备
```bash
# 使用合适的文件分割
milvus-ingest generate \
  --builtin ecommerce \
  --total-rows 5000000 \
  --file-size 256MB \
  --rows-per-file 500000
```

### 2. 性能优化
```bash
# insert: 调整批次大小
--batch-size 2000  # 适中规模，平衡性能和内存

# import: 确保文件大小合理
# 单文件不超过 16GB
# 建议 256MB-1GB 每文件
```

### 3. 生产部署
```bash
# 使用环境变量管理凭证
export MILVUS_URI=http://milvus-cluster:19530
export MILVUS_TOKEN=your_production_token
export MINIO_ENDPOINT=http://minio-cluster:9000
export MINIO_ACCESS_KEY=prod_access_key
export MINIO_SECRET_KEY=prod_secret_key

# 简化命令
milvus-ingest to-milvus import \
  --local-path ./data \
  --s3-path collections/$(date +%Y%m%d)/ \
  --bucket production-data \
  --wait \
  --timeout 1800
```

### 4. 监控和日志
```bash
# 启用详细日志
export LOGURU_LEVEL=INFO

# 记录操作日志
milvus-ingest to-milvus insert ./data 2>&1 | tee import.log
```

## 相关命令

- [`generate`](generate.md) - 生成需要导入的数据
- [`upload`](upload.md) - 上传数据到 S3/MinIO（用于 import 模式）
- [`schema`](schema.md) - 管理数据模式

---

## verify - 数据完整性验证

验证 Milvus 集合中的数据是否与原始生成的数据匹配，支持多级验证和查询功能测试。

### 语法
```bash
milvus-ingest to-milvus verify DATA_PATH [OPTIONS]
```

### 必需参数
- `DATA_PATH`: 原始数据目录路径（包含 meta.json 的目录）

### 验证级别选项

#### --level {count|scalar|full}
设置验证级别（默认: count）

```bash
# Level 1: 行数验证 + 查询测试（默认，最快）
milvus-ingest to-milvus verify ./data --level count

# Level 2: 标量字段验证 + 查询测试（排除向量字段）
milvus-ingest to-milvus verify ./data --level scalar

# Level 3: 完整验证 + 查询测试（包含向量字段）
milvus-ingest to-milvus verify ./data --level full
```

**验证级别对比:**

| 级别 | 验证内容 | 适用场景 | 验证速度 |
|------|----------|----------|----------|
| `count` | 行数统计 + 查询/搜索测试 | 快速验证导入是否成功 | 最快（几秒） |
| `scalar` | 标量字段数据 + 查询/搜索测试 | 业务数据准确性验证 | 中等（几分钟） |
| `full` | 所有字段数据 + 查询/搜索测试 | 完整数据质量保证 | 较慢（几十分钟） |

### 连接选项

#### --uri URI
Milvus 服务器地址（默认: http://localhost:19530）

```bash
# 验证远程 Milvus 数据
milvus-ingest to-milvus verify ./data \
  --uri http://192.168.1.100:19530 \
  --level scalar
```

#### --token TOKEN
认证令牌

```bash
# Milvus Cloud 验证
milvus-ingest to-milvus verify ./data \
  --uri https://cluster.vectordb.zillizcloud.com \
  --token your_api_token \
  --level full
```

#### --db-name DATABASE
数据库名称（默认: default）

```bash
# 指定数据库验证
milvus-ingest to-milvus verify ./data \
  --db-name production \
  --collection-name products \
  --level scalar
```

### 集合配置选项

#### --collection-name NAME
覆盖集合名称（默认使用 meta.json 中的名称）

```bash
# 验证特定名称的集合
milvus-ingest to-milvus verify ./data \
  --collection-name custom_collection_name \
  --level full
```

### 验证功能特点

#### 1. 渐进式验证
每个级别都包含前一级别的所有验证内容：
- **count**: 基础行数统计
- **scalar**: count + 标量字段值匹配
- **full**: scalar + 向量字段值匹配

#### 2. 查询功能测试
所有验证级别都包含查询功能测试：
- **1000次精确查询测试**: 验证数据检索功能（成功率>95%）
- **1000次向量搜索测试**: 验证相似度搜索功能（成功率>80%）
- **功能验证**: 确保导入的数据不仅存在，还能正常查询

#### 3. 智能字段处理
- **AUTO_ID 跳过**: 自动跳过 auto_id 字段的值验证
- **浮点数容差**: 使用 1e-6 容差进行浮点数比较
- **向量精度处理**: 支持 Float16、BFloat16 等特殊向量类型
- **错误容忍**: 允许最多5%的数据不匹配（浮点精度问题）

### 完整示例

#### 1. 标准验证流程
```bash
# 生成测试数据
milvus-ingest generate --builtin ecommerce_search --total-rows 10000 --out ./test_data

# 导入到 Milvus
milvus-ingest to-milvus insert ./test_data

# 快速验证（推荐作为第一步）
milvus-ingest to-milvus verify ./test_data --level count

# 详细验证（推荐作为第二步）  
milvus-ingest to-milvus verify ./test_data --level scalar

# 完整验证（可选，最全面）
milvus-ingest to-milvus verify ./test_data --level full
```

#### 2. 生产环境验证
```bash
# 生产数据验证流程
milvus-ingest to-milvus verify ./production_data \
  --uri http://prod-milvus:19530 \
  --token $PROD_TOKEN \
  --db-name production \
  --collection-name product_vectors_v2 \
  --level scalar
```

#### 3. 大规模数据验证
```bash
# 大规模数据（建议分步验证）
# 第一步：快速验证行数和基本功能
milvus-ingest to-milvus verify ./large_dataset --level count

# 第二步：如果快速验证通过，再进行详细验证
milvus-ingest to-milvus verify ./large_dataset --level scalar
```

### 输出示例

#### count 级别输出
```
[INFO] 开始数据验证 (level: count)
[INFO] 连接到 Milvus: http://localhost:19530
[INFO] 验证集合: ecommerce_search

╭─────────────── 验证结果总览 ────────────────╮
│ 集合统计                                   │
│ ├─ 原始数据行数: 10,000                    │
│ ├─ Milvus 行数:  10,000                    │
│ └─ 行数匹配: ✅ PASS                       │
│                                           │
│ 查询功能测试                               │
│ ├─ 精确查询测试: 980/1000 (98%) ✅ PASS     │
│ ├─ 向量搜索测试: 845/1000 (84.5%) ✅ PASS   │
│ └─ 功能测试: ✅ PASS                       │
╰───────────────────────────────────────────╯

[SUCCESS] 验证完成: 数据导入成功且功能正常
```

#### scalar 级别输出
```
[INFO] 开始数据验证 (level: scalar)
[INFO] 验证字段: id, product_name, price, category

╭─────────────── 字段验证详情 ────────────────╮
│ Field        │ Type     │ Status          │
├──────────────┼──────────┼─────────────────┤
│ id          │ Int64    │ ✅ PASS (SKIP)   │
│ product_name│ VarChar  │ ✅ PASS (10000) │
│ price       │ Float    │ ✅ PASS (9995)  │
│ category    │ VarChar  │ ✅ PASS (10000) │
╰─────────────────────────────────────────────╯

[SUCCESS] 标量字段验证通过: 29995/30000 (99.98%)
[SUCCESS] 查询功能测试通过
```

#### full 级别输出
```
[INFO] 开始数据验证 (level: full)
[INFO] 包含向量字段验证，这可能需要较长时间...

╭─────────────── 完整验证结果 ────────────────╮
│ 标量字段: ✅ PASS (99.98% 匹配)             │
│ 向量字段: ✅ PASS (99.95% 匹配)             │
│ 查询测试: ✅ PASS (98% 成功率)              │
│ 搜索测试: ✅ PASS (84.5% 成功率)            │
╰───────────────────────────────────────────╯

[SUCCESS] 完整验证通过: 数据质量优秀
```

### 故障排除

#### 1. 连接错误
```
错误: [ConnectionError] 无法连接到 Milvus
```
**解决方案:**
```bash
# 检查 Milvus 服务状态
curl http://localhost:19530/healthz

# 检查 URI 格式
--uri http://localhost:19530  # 正确
--uri localhost:19530         # 错误
```

#### 2. 集合不存在
```
错误: [CollectionNotFound] 集合 'test_collection' 不存在
```
**解决方案:**
```bash
# 检查集合名称
milvus-ingest to-milvus verify ./data --collection-name correct_name

# 或先导入数据
milvus-ingest to-milvus insert ./data
```

#### 3. 数据不匹配
```
警告: 字段 'price' 验证失败 (8950/10000 匹配)
```
**说明:** 这是正常的，通常由浮点精度差异引起。如果匹配率>95%，验证仍会通过。

#### 4. 查询测试失败
```
错误: 查询功能测试失败 (成功率: 45%)
```
**可能原因:**
- 索引未完全构建
- 数据尚未完全加载
- 网络连接问题

**解决方案:**
```bash
# 等待索引构建完成后重试
sleep 60 && milvus-ingest to-milvus verify ./data --level count
```

### 最佳实践

#### 1. 分级验证策略
```bash
# 推荐验证流程
echo "第1步: 快速验证"
milvus-ingest to-milvus verify ./data --level count

echo "第2步: 业务验证"  
milvus-ingest to-milvus verify ./data --level scalar

echo "第3步: 完整验证（可选）"
milvus-ingest to-milvus verify ./data --level full
```

#### 2. 自动化验证
```bash
#!/bin/bash
# 自动化数据导入和验证脚本

DATA_DIR="./generated_data"
COLLECTION_NAME="test_collection"

# 导入数据
echo "导入数据..."
milvus-ingest to-milvus insert $DATA_DIR --collection-name $COLLECTION_NAME

# 等待索引构建
echo "等待索引构建..."
sleep 30

# 执行验证
echo "执行验证..."
milvus-ingest to-milvus verify $DATA_DIR \
  --collection-name $COLLECTION_NAME \
  --level scalar

echo "验证完成"
```

#### 3. 生产环境监控
```bash
# 生产环境定期验证
# 建议加入 crontab
0 2 * * * milvus-ingest to-milvus verify /data/production \
  --uri $PROD_MILVUS_URI \
  --token $PROD_TOKEN \
  --level count \
  >> /var/log/milvus-verify.log 2>&1
```

---

**下一步**: 查看 [完整工作流教程](../tutorials/complete-workflow.md) 了解端到端的数据处理流程
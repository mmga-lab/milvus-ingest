# upload 命令详解

`upload` 命令用于将生成的数据文件上传到 S3 兼容的对象存储服务（如 AWS S3、MinIO 等）。

## 基本语法

```bash
milvus-ingest upload LOCAL_PATH S3_PATH [OPTIONS]
```

## 必需参数

### LOCAL_PATH
本地数据目录路径（由 `generate` 命令生成的目录）

```bash
# 示例路径
./output_data/          # 相对路径
/home/user/data/        # 绝对路径
~/generated_data/       # 用户主目录路径
```

### S3_PATH  
S3 目标路径，格式为 `s3://bucket/prefix/`

```bash
# AWS S3 路径
s3://my-bucket/data/
s3://my-bucket/datasets/ecommerce/
s3://my-bucket/year=2024/month=01/

# MinIO 路径（同样格式）
s3://milvus-data/collections/
s3://test-bucket/experiments/
```

## 存储服务配置

### AWS S3 (默认)
直接使用 AWS S3 服务，使用标准 AWS 凭证：

```bash
# 使用默认 AWS 凭证
milvus-ingest upload ./data s3://my-aws-bucket/data/

# 使用环境变量凭证
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
milvus-ingest upload ./data s3://my-aws-bucket/data/
```

### MinIO 和其他 S3 兼容服务

#### --endpoint-url URL
指定 S3 兼容服务的端点 URL

```bash
# MinIO 本地部署
milvus-ingest upload ./data s3://bucket/data/ \
  --endpoint-url http://localhost:9000

# MinIO 远程部署
milvus-ingest upload ./data s3://bucket/data/ \
  --endpoint-url http://minio.example.com:9000

# 其他 S3 兼容服务
milvus-ingest upload ./data s3://bucket/data/ \
  --endpoint-url https://s3.example.com
```

## 认证选项

### --access-key-id KEY
指定访问密钥 ID

```bash
milvus-ingest upload ./data s3://bucket/data/ \
  --access-key-id AKIAIOSFODNN7EXAMPLE
```

### --secret-access-key SECRET
指定访问密钥

```bash
milvus-ingest upload ./data s3://bucket/data/ \
  --secret-access-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### 组合认证示例
```bash
# MinIO 完整认证
milvus-ingest upload ./ecommerce_data s3://milvus-data/ecommerce/ \
  --endpoint-url http://localhost:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin
```

## 上传选项

### --no-verify-ssl
禁用 SSL 证书验证（用于开发环境）

```bash
# 本地 MinIO 不使用 SSL
milvus-ingest upload ./data s3://bucket/data/ \
  --endpoint-url http://localhost:9000 \
  --no-verify-ssl
```

### --region REGION  
指定 AWS 区域（AWS S3 专用）

```bash
# 指定 AWS 区域
milvus-ingest upload ./data s3://my-bucket/data/ \
  --region us-west-2
```

### --storage-class CLASS
设置 S3 存储类别（AWS S3 专用）

```bash
# 标准存储
milvus-ingest upload ./data s3://bucket/data/ \
  --storage-class STANDARD

# 低频访问存储  
milvus-ingest upload ./data s3://bucket/data/ \
  --storage-class STANDARD_IA

# 归档存储
milvus-ingest upload ./data s3://bucket/data/ \
  --storage-class GLACIER
```

## 完整示例

### 1. AWS S3 上传

```bash
# 生成数据
milvus-ingest generate --builtin ecommerce --total-rows 100000 --out ./ecommerce_data

# 上传到 AWS S3（使用默认凭证）
milvus-ingest upload ./ecommerce_data s3://my-production-bucket/datasets/ecommerce/

# 上传到 AWS S3（指定凭证和区域）
milvus-ingest upload ./ecommerce_data s3://my-production-bucket/datasets/ecommerce/ \
  --access-key-id AKIAIOSFODNN7EXAMPLE \
  --secret-access-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  --region us-east-1 \
  --storage-class STANDARD
```

### 2. MinIO 上传

```bash
# 生成测试数据
milvus-ingest generate --builtin simple --total-rows 10000 --out ./test_data

# 上传到本地 MinIO
milvus-ingest upload ./test_data s3://test-bucket/simple-data/ \
  --endpoint-url http://localhost:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin \
  --no-verify-ssl

# 上传到远程 MinIO
milvus-ingest upload ./test_data s3://production-bucket/datasets/ \
  --endpoint-url https://minio.company.com \
  --access-key-id prod_access_key \
  --secret-access-key prod_secret_key
```

### 3. 批量上传不同数据集

```bash
# 生成多个数据集
schemas=("simple" "ecommerce" "documents" "users")
for schema in "${schemas[@]}"; do
  milvus-ingest generate --builtin $schema --total-rows 50000 --out ./${schema}_data
done

# 批量上传
for schema in "${schemas[@]}"; do
  echo "上传 $schema 数据集..."
  milvus-ingest upload ./${schema}_data s3://datasets/${schema}/ \
    --endpoint-url http://minio:9000 \
    --access-key-id minioadmin \
    --secret-access-key minioadmin
done
```

### 4. 环境变量配置

创建环境配置：
```bash
# .env 文件
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_BUCKET=milvus-data
```

使用环境变量：
```bash
# 加载环境变量
source .env

# 简化上传命令
milvus-ingest upload ./data s3://$MINIO_BUCKET/data/ \
  --endpoint-url $MINIO_ENDPOINT
```

## 上传过程详解

### 上传流程
1. **路径验证**: 检查本地目录和 S3 路径格式
2. **凭证验证**: 验证访问权限和存储桶访问
3. **文件扫描**: 扫描本地目录中的所有文件
4. **分块上传**: 大文件自动分块并行上传
5. **进度显示**: 实时显示上传进度和速度
6. **完整性检查**: 验证上传文件的完整性

### 输出信息

```bash
$ milvus-ingest upload ./data s3://bucket/data/ --endpoint-url http://minio:9000

[INFO] 正在验证本地路径: ./data
[INFO] 发现 3 个文件需要上传 (总大小: 245.7 MB)
[INFO] 正在连接到 MinIO: http://minio:9000
[INFO] 存储桶 'bucket' 已确认可访问

上传进度:
┌─ data.parquet          ━━━━━━━━━━━━━━━━━━━━ 100% 185.2 MB @ 45.3 MB/s
├─ meta.json             ━━━━━━━━━━━━━━━━━━━━ 100%   1.2 KB @ 15.8 KB/s  
└─ index_suggestions.json ━━━━━━━━━━━━━━━━━━━━ 100%   2.3 KB @ 18.2 KB/s

[SUCCESS] 成功上传 3 个文件到 s3://bucket/data/
[INFO] 总用时: 4.2 秒, 平均速度: 58.5 MB/s
[INFO] 上传的文件:
  - s3://bucket/data/data.parquet
  - s3://bucket/data/meta.json  
  - s3://bucket/data/index_suggestions.json
```

## 错误处理

### 常见错误及解决方案

#### 1. 认证失败
```
错误: [AccessDenied] The AWS Access Key Id you provided does not exist in our records
```

**解决方案:**
```bash
# 检查凭证是否正确
--access-key-id your_correct_key --secret-access-key your_correct_secret

# 或设置环境变量
export AWS_ACCESS_KEY_ID=correct_key
export AWS_SECRET_ACCESS_KEY=correct_secret
```

#### 2. 端点连接失败
```
错误: [ConnectionError] Failed to connect to minio:9000
```

**解决方案:**
```bash
# 检查端点 URL 格式
--endpoint-url http://localhost:9000  # 本地 MinIO
--endpoint-url https://minio.company.com  # 远程 MinIO

# 检查网络连接
curl http://localhost:9000/minio/health/live
```

#### 3. 存储桶不存在
```
错误: [NoSuchBucket] The specified bucket does not exist: my-bucket
```

**解决方案:**
```bash
# 检查存储桶名称拼写
s3://correct-bucket-name/data/

# 或先创建存储桶（工具会自动尝试创建）
```

#### 4. SSL 证书问题
```
错误: [SSLError] certificate verify failed
```

**解决方案:**
```bash
# 对于开发环境，禁用 SSL 验证
--no-verify-ssl

# 对于生产环境，确保正确的 SSL 配置
```

#### 5. 文件权限问题
```
错误: [PermissionError] Permission denied: './data/data.parquet'
```

**解决方案:**
```bash
# 检查文件权限
chmod -R 644 ./data/

# 检查目录权限
chmod 755 ./data/
```

## 性能优化

### 1. 网络优化
```bash
# 选择就近的区域（AWS）
--region us-west-1  # 如果你在美国西部

# 使用高带宽网络环境
# 避免在网络高峰期上传大文件
```

### 2. 文件优化
```bash
# 生成时使用 Parquet 格式（更高压缩率）
milvus-ingest generate --format parquet

# 控制文件大小便于传输
milvus-ingest generate --file-size 256MB  # 256MB 每文件
```

### 3. 并发优化
大文件会自动使用分块并发上传，无需额外配置。

## 与其他命令的集成

### 1. 完整数据管道
```bash
# 步骤1: 生成数据
milvus-ingest generate --builtin ecommerce --total-rows 1000000 --out ./ecommerce

# 步骤2: 上传数据
milvus-ingest upload ./ecommerce s3://milvus-data/ecommerce/ \
  --endpoint-url http://minio:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin

# 步骤3: 导入到 Milvus（见 to-milvus 命令）
milvus-ingest to-milvus import ecommerce_products s3://milvus-data/ecommerce/
```

### 2. 脚本化批处理
```bash
#!/bin/bash
# 数据生成和上传脚本

SCHEMAS=("simple" "ecommerce" "documents")
BUCKET="s3://production-data"
ENDPOINT="http://minio:9000"

for schema in "${SCHEMAS[@]}"; do
    echo "处理 $schema..."
    
    # 生成数据
    milvus-ingest generate --builtin $schema --total-rows 100000 --out ./${schema}_data
    
    # 上传数据
    milvus-ingest upload ./${schema}_data ${BUCKET}/${schema}/ \
        --endpoint-url $ENDPOINT \
        --access-key-id $MINIO_ACCESS_KEY \
        --secret-access-key $MINIO_SECRET_KEY
    
    # 清理本地数据
    rm -rf ./${schema}_data
    
    echo "$schema 处理完成"
done
```

## 最佳实践

### 1. 目录结构规划
```bash
# 推荐的 S3 路径结构
s3://bucket/
├── datasets/
│   ├── ecommerce/
│   │   ├── 2024/01/15/    # 按日期组织
│   │   └── latest/        # 最新版本
│   ├── documents/
│   └── users/
├── experiments/           # 实验数据
└── backups/              # 备份数据
```

### 2. 命名约定
```bash
# 清晰的路径命名
s3://milvus-data/collections/product_catalog_v2/
s3://milvus-data/experiments/similarity_test_20240115/
s3://milvus-data/staging/user_profiles/
```

### 3. 安全考虑
```bash
# 生产环境使用 IAM 角色而非硬编码凭证
# 开发环境可以使用 .env 文件但不要提交到版本控制

# 使用最小权限原则
# 只授予必需的存储桶访问权限
```

### 4. 监控和日志
```bash
# 启用详细日志
export LOGURU_LEVEL=INFO

# 记录上传操作
milvus-ingest upload ./data s3://bucket/data/ 2>&1 | tee upload.log
```

## 相关命令

- [`generate`](generate.md) - 生成需要上传的数据
- [`to-milvus import`](to-milvus.md#import) - 从上传的数据导入到 Milvus

---

**下一步**: 查看 [to-milvus 命令详解](to-milvus.md) 了解如何将上传的数据导入到 Milvus
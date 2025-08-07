# 快速开始指南

5 分钟上手 milvus-ingest 工具，从零开始生成高质量的 Milvus 测试数据。

## 📋 前置条件

- Python 3.10+
- PDM（Python 依赖管理器）
- 可选：Docker（用于本地 Milvus 环境）

## 🚀 第一步：安装工具

```bash
# 克隆项目
git clone https://github.com/zilliz/milvus-ingest.git
cd milvus-ingest

# 安装依赖
pdm install

# 验证安装
milvus-ingest --help
```

## 🎯 第二步：生成第一个数据集

### 使用内置模式快速开始

```bash
# 生成简单测试数据并预览
milvus-ingest generate --builtin product_catalog --total-rows 1000 --preview
```

输出示例：
```
Schema: product_catalog
Collection: product_catalog

预览数据 (前5行):
┌────┬─────────────────────────────────┬────────┬──────────────────────────────────┐
│ id │ product_name                   │ price  │ embedding                        │
├────┼─────────────────────────────────┼────────┼──────────────────────────────────┤
│ 1  │ High-quality wireless headphones│ 159.99 │ [0.123, -0.456, 0.789, ...]     │
│ 2  │ Smart home automation device   │ 89.99  │ [-0.234, 0.567, -0.890, ...]    │
│ 3  │ Professional camera lens       │ 299.99 │ [0.345, -0.678, 0.123, ...]     │
│ 4  │ Ergonomic office chair         │ 199.99 │ [-0.456, 0.789, -0.234, ...]    │
│ 5  │ Portable power bank            │ 49.99  │ [0.567, -0.123, 0.456, ...]     │
└────┴─────────────────────────────────┴────────┴──────────────────────────────────┘

字段信息:
- id: Int64 (主键, 自动生成)
- product_name: VarChar (最大长度: 200)  
- price: Float (范围: 9.99-999.99)
- embedding: FloatVector (维度: 128)
```

### 生成实际数据集

```bash
# 生成1万行数据
milvus-ingest generate --builtin product_catalog --total-rows 10000 --out ./my_first_dataset
```

输出：
```
[INFO] 正在生成数据...
生成进度: ━━━━━━━━━━━━━━━━━━━━ 100% 10000/10000 行 @ 2500 行/秒

[SUCCESS] 数据生成完成!
输出目录: ./my_first_dataset
文件大小: 5.2 MB
生成时间: 4.1 秒

生成的文件:
├── data.parquet     # 主数据文件 (5.1 MB)
└── meta.json        # 集合元数据 (1.2 KB)
```

## 🔍 第三步：探索更多数据类型

### 查看所有可用模式

```bash
milvus-ingest schema list
```

输出：
```
Built-in Schemas

╭────────── product_catalog - Product Catalog ─────────╮
│ Simple product catalog for getting started with auto_id │
│ Fields: 4 • Vector dims: 128                          │
│ Use cases: Learning, Basic product search, Auto ID   │
╰──────────────────────────────────────────────────────╯
╭────────── ecommerce_search - E-commerce Search ──────╮
│ E-commerce product search with nullable fields       │
│ Fields: 5 • Vector dims: 256                         │
│ Use cases: Product search, E-commerce, Nullable      │
╰──────────────────────────────────────────────────────╯
╭────────── news_articles - News Articles ─────────────╮
│ News article storage with dynamic fields             │
│ Fields: 4 • Vector dims: 768                         │
│ Use cases: News search, Dynamic schema, Content      │
╰──────────────────────────────────────────────────────╯

以及其他3个模式...
```

### 尝试电商数据模式

```bash
# 先预览电商模式的结构
milvus-ingest schema show ecommerce_search
```

```bash
# 生成电商测试数据
milvus-ingest generate --builtin ecommerce_search --total-rows 5000 --out ./ecommerce_data
```

这会生成包含产品信息、价格、评分、多个向量字段的真实电商数据。

## 🗂️ 第四步：理解输出结构

生成的每个数据集都包含：

```
my_first_dataset/
├── data.parquet      # 主数据文件（Parquet 格式，高性能）
└── meta.json         # 集合元数据
```

### 查看元数据

```bash
cat ./my_first_dataset/meta.json
```

```json
{
  "collection_name": "product_catalog",
  "description": "产品目录模式",
  "fields": [
    {
      "name": "id",
      "type": "Int64", 
      "is_primary": true,
      "auto_id": true
    },
    {
      "name": "product_name",
      "type": "VarChar",
      "max_length": 200
    },
    {
      "name": "price",
      "type": "Float",
      "min": 9.99,
      "max": 999.99
    },
    {
      "name": "embedding", 
      "type": "FloatVector",
      "dim": 128
    }
  ],
  "generation_stats": {
    "total_rows": 10000,
    "file_size_mb": 5.2,
    "generation_time_seconds": 4.1
  }
}
```

## 🎭 第五步：自定义数据模式

### 创建自定义模式文件

```bash
# 创建自定义产品模式
cat > my_products.json << 'EOF'
{
  "collection_name": "my_products",
  "fields": [
    {
      "name": "product_id",
      "type": "Int64",
      "is_primary": true,
      "auto_id": true
    },
    {
      "name": "product_name",
      "type": "VarChar", 
      "max_length": 300
    },
    {
      "name": "price",
      "type": "Float",
      "min": 9.99,
      "max": 999.99
    },
    {
      "name": "description",
      "type": "VarChar",
      "max_length": 1000,
      "nullable": true
    },
    {
      "name": "search_vector",
      "type": "FloatVector",
      "dim": 384
    },
    {
      "name": "tags",
      "type": "Array",
      "element_type": "VarChar",
      "max_capacity": 5,
      "max_length": 50
    }
  ]
}
EOF
```

### 验证并生成自定义数据

```bash
# 验证模式格式
milvus-ingest generate --schema my_products.json --validate-only

# 预览数据
milvus-ingest generate --schema my_products.json --total-rows 100 --preview

# 生成数据集
milvus-ingest generate --schema my_products.json --total-rows 10000 --out ./my_products_data
```

## 📊 第六步：性能测试

### 小规模测试
```bash
# 1万行数据 - 几秒内完成
time milvus-ingest generate --builtin ecommerce --total-rows 10000 --out ./small_test
```

### 中等规模测试
```bash
# 10万行数据 - 约30秒
time milvus-ingest generate --builtin ecommerce --total-rows 100000 --out ./medium_test
```

### 大规模测试（可选）
```bash
# 100万行数据 - 约5分钟
time milvus-ingest generate --builtin ecommerce --total-rows 1000000 --out ./large_test
```

## 🔄 第七步：连接到 Milvus（可选）

如果你有 Milvus 实例，可以直接导入生成的数据：

### 启动本地 Milvus（使用 Docker）

```bash
# 使用项目提供的 Docker Compose
cd deploy/
docker-compose up -d

# 等待服务启动
sleep 30
```

### 导入数据到 Milvus

```bash
# 直接插入小规模数据
milvus-ingest to-milvus insert ./my_first_dataset

# 检查导入结果
curl -X POST "http://localhost:19530/v1/vector/collections/simple_collection/query" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "output_fields": ["id", "text"]
  }'
```

## 🧹 第八步：清理环境

```bash
# 查看生成的文件
ls -la ./

# 清理测试数据（保留重要的）
milvus-ingest clean --dry-run  # 先预览
milvus-ingest clean --yes      # 确认清理
```

## 🎉 完成！你已经学会了：

✅ **安装和基本使用** - 生成第一个数据集  
✅ **探索内置模式** - 了解不同数据类型  
✅ **理解输出格式** - Parquet + 元数据结构  
✅ **创建自定义模式** - 定义自己的数据结构  
✅ **性能基准测试** - 了解工具性能特点  
✅ **Milvus 集成** - 将数据导入向量数据库  
✅ **环境清理** - 保持工作空间整洁  

## 🚀 下一步学习

现在你已经掌握了基础操作，可以继续学习：

- [**性能调优指南**](performance.md) - 生成大规模数据的最佳实践
- [**完整工作流**](complete-workflow.md) - 从生成到生产部署的端到端流程  
- [**自定义模式详解**](custom-schemas.md) - 深入了解模式设计和字段类型
- [**实际应用场景**](../examples/README.md) - 各种业务场景的具体示例

## 💡 快速参考卡片

### 最常用命令
```bash
# 快速预览
milvus-ingest generate --builtin simple --total-rows 1000 --preview

# 生成测试数据  
milvus-ingest generate --builtin <schema> --total-rows <count> --out <dir>

# 查看所有模式
milvus-ingest schema list

# 查看模式详情
milvus-ingest schema show <schema_name>

# 清理环境
milvus-ingest clean --yes
```

### 常用模式推荐
- `product_catalog` - 学习和基础测试（推荐新用户）
- `ecommerce_search` - 电商/推荐系统
- `document_search` - 文档搜索/RAG 应用
- `multi_tenant_data` - 多租户系统

### 性能建议
- 小于10万行：直接生成
- 10万-100万行：使用默认批处理
- 大于100万行：调整 `--batch-size 100000`

---

**🎯 目标达成**: 你现在可以高效地生成各种类型的 Milvus 测试数据！
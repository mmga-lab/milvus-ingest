# JSON格式和动态字段使用指南

本指南详细介绍如何使用 milvus-ingest 的 JSON 格式支持和动态字段功能。

## 📚 JSON格式概述

milvus-ingest 支持两种数据格式：

- **Parquet**: 高性能二进制格式，适合大规模数据生成和分析
- **JSON**: 标准JSON数组格式，易读易调试，与Milvus bulk import完全兼容

## 🎯 JSON格式特点

### 1. 标准数组格式
生成的JSON文件采用标准数组格式，无需额外包装：

```json
[
  {
    "id": 1,
    "name": "Product 1",
    "price": 19.99,
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "$meta": {
      "author": "Alice",
      "views": 1234
    }
  },
  {
    "id": 2,
    "name": "Product 2",
    "price": 29.99,
    "embedding": [0.5, 0.6, 0.7, 0.8],
    "$meta": {
      "author": "Bob",
      "rating": 4.5
    }
  }
]
```

### 2. Milvus兼容性
- ✅ **Direct Insert**: 支持直接插入到Milvus
- ✅ **Bulk Import**: 兼容Milvus批量导入API
- ✅ **Dynamic Fields**: 自动处理 `$meta` 字段中的动态字段
- ✅ **Multi-Format**: 与Parquet格式可以混用

### 3. 格式灵活性
工具可以读取多种JSON格式：
- 标准数组格式: `[{}, {}...]`
- 带rows包装: `{"rows": [{}, {}...]}`
- JSONL格式: 每行一个JSON对象
- 单个对象格式: `{}`

## 📋 基础使用

### 生成JSON数据
```bash
# 生成简单JSON数据
milvus-ingest generate --builtin simple --total-rows 1000 --format json --out json_data

# 查看生成的文件结构
ls json_data/
# 输出: data.json  meta.json

# 预览JSON数据格式
head -1 json_data/data.json | python -m json.tool | head -20
```

### 插入到Milvus
```bash
# 直接插入（自动检测JSON格式）
milvus-ingest to-milvus insert ./json_data --collection-name test_collection

# 批量导入（JSON格式）
milvus-ingest to-milvus import \
  --local-path ./json_data \
  --s3-path json-test/ \
  --bucket test-bucket \
  --endpoint-url http://minio:9000 \
  --wait
```

## 🔧 动态字段功能

### 1. 动态字段概念
动态字段允许在固定schema之外添加额外的字段，提供更大的灵活性：

- **存储方式**: 动态字段存储在Milvus的 `$meta` 字段中
- **数据类型**: 支持String、Int、Float、Bool、Array、JSON
- **出现概率**: 可控制字段在记录中出现的概率
- **值范围**: 支持预定义值列表或数值范围

### 2. 动态字段配置
在schema中添加 `dynamic_fields` 配置：

```json
{
  "collection_name": "dynamic_example",
  "enable_dynamic_field": true,
  "dynamic_fields": [
    {
      "name": "author",
      "type": "String",
      "probability": 0.8,
      "values": ["Alice Johnson", "Bob Smith", "Charlie Brown"]
    },
    {
      "name": "views",
      "type": "Int",
      "probability": 0.9,
      "min": 1,
      "max": 10000
    },
    {
      "name": "rating",
      "type": "Float",
      "probability": 0.7,
      "min": 1.0,
      "max": 5.0
    },
    {
      "name": "tags",
      "type": "Array",
      "probability": 0.6,
      "element_type": "Int",
      "length": 3,
      "min": 1,
      "max": 100
    },
    {
      "name": "metadata",
      "type": "JSON",
      "probability": 0.5
    },
    {
      "name": "status",
      "type": "String",
      "probability": 1.0,
      "values": ["draft", "published", "archived"]
    }
  ],
  "fields": [
    {
      "name": "id",
      "type": "Int64",
      "is_primary": true,
      "auto_id": true
    },
    {
      "name": "title",
      "type": "VarChar",
      "max_length": 200
    },
    {
      "name": "embedding",
      "type": "FloatVector",
      "dim": 384
    }
  ]
}
```

### 3. 动态字段类型详解

#### String类型
```json
{
  "name": "author",
  "type": "String",
  "probability": 0.8,
  "values": ["Alice", "Bob", "Charlie"]  // 预定义值列表
}
```

#### 数值类型（Int/Float）
```json
{
  "name": "score",
  "type": "Float",
  "probability": 0.9,
  "min": 0.0,
  "max": 100.0
}
```

#### Boolean类型
```json
{
  "name": "is_active",
  "type": "Bool",
  "probability": 1.0  // true/false随机生成
}
```

#### Array类型
```json
{
  "name": "tags",
  "type": "Array",
  "probability": 0.6,
  "element_type": "String",  // 元素类型
  "length": 3,               // 数组长度
  "values": ["tech", "ai", "database", "vector"]  // 元素候选值
}
```

#### JSON类型
```json
{
  "name": "metadata",
  "type": "JSON",
  "probability": 0.5  // 自动生成复杂JSON结构
}
```

## 🚀 实战示例

### 示例1：电商产品with动态属性
```bash
# 创建自定义schema
cat > ecommerce_dynamic.json << 'EOF'
{
  "collection_name": "ecommerce_dynamic",
  "enable_dynamic_field": true,
  "dynamic_fields": [
    {
      "name": "brand",
      "type": "String",
      "probability": 0.9,
      "values": ["Apple", "Samsung", "Google", "Microsoft", "Amazon"]
    },
    {
      "name": "discount",
      "type": "Float",
      "probability": 0.3,
      "min": 0.1,
      "max": 0.5
    },
    {
      "name": "reviews_count",
      "type": "Int",
      "probability": 0.8,
      "min": 0,
      "max": 10000
    },
    {
      "name": "features",
      "type": "Array",
      "probability": 0.6,
      "element_type": "String",
      "length": 3,
      "values": ["waterproof", "wireless", "fast-charging", "eco-friendly"]
    }
  ],
  "fields": [
    {
      "name": "id",
      "type": "Int64",
      "is_primary": true,
      "auto_id": true
    },
    {
      "name": "name",
      "type": "VarChar",
      "max_length": 200
    },
    {
      "name": "price",
      "type": "Float",
      "min": 1.0,
      "max": 1000.0
    },
    {
      "name": "embedding",
      "type": "FloatVector",
      "dim": 128
    }
  ]
}
EOF

# 生成带动态字段的JSON数据
milvus-ingest generate \
  --schema ecommerce_dynamic.json \
  --total-rows 5000 \
  --format json \
  --out dynamic_ecommerce

# 查看生成的数据样例
python3 -c "
import json
with open('dynamic_ecommerce/data.json', 'r') as f:
    data = json.load(f)
    print('First record with dynamic fields:')
    print(json.dumps(data[0], indent=2))
"
```

### 示例2：内容管理系统
```bash
# 使用内置的动态字段示例
milvus-ingest generate \
  --builtin dynamic_example \
  --total-rows 1000 \
  --format json \
  --out cms_content

# 插入到Milvus
milvus-ingest to-milvus insert ./cms_content \
  --collection-name cms_articles \
  --drop-if-exists

# 验证动态字段已正确存储
# （需要连接到Milvus进行查询验证）
```

### 示例3：多格式混合工作流
```bash
# 生成Parquet格式（高性能）
milvus-ingest generate \
  --builtin dynamic_example \
  --total-rows 100000 \
  --format parquet \
  --out large_dataset_parquet

# 生成JSON格式（便于调试）
milvus-ingest generate \
  --builtin dynamic_example \
  --total-rows 1000 \
  --format json \
  --out sample_dataset_json

# 两种格式都可以插入到同一个集合
milvus-ingest to-milvus insert ./large_dataset_parquet \
  --collection-name mixed_collection

milvus-ingest to-milvus insert ./sample_dataset_json \
  --collection-name mixed_collection
```

## 🎯 最佳实践

### 1. 格式选择指导
```bash
# 大规模数据生成（>10万行）→ 使用Parquet
milvus-ingest generate --builtin simple --total-rows 1000000 --format parquet

# 小规模调试和验证 → 使用JSON
milvus-ingest generate --builtin simple --total-rows 1000 --format json --preview

# 动态字段调试 → 推荐JSON格式
milvus-ingest generate --builtin dynamic_example --total-rows 100 --format json
```

### 2. 动态字段设计原则
- **probability**: 常用字段设置0.8-1.0，可选字段设置0.3-0.6
- **value范围**: 根据业务需求设置合理的数据范围
- **字段命名**: 使用描述性名称，避免与固定字段冲突
- **类型选择**: 优先使用基础类型（String、Int、Float、Bool）

### 3. 性能优化
```bash
# 大数据集使用大批次
milvus-ingest generate \
  --builtin dynamic_example \
  --total-rows 1000000 \
  --format json \
  --batch-size 50000

# 合理设置文件分割
milvus-ingest generate \
  --builtin dynamic_example \
  --total-rows 5000000 \
  --format json \
  --file-size 256MB \
  --rows-per-file 500000
```

### 4. 数据验证
```bash
# 生成前验证schema
milvus-ingest generate --schema my_schema.json --validate-only

# 小规模预览
milvus-ingest generate --builtin dynamic_example --total-rows 10 --format json --preview

# 检查动态字段内容
python3 -c "
import json
with open('output/data.json', 'r') as f:
    data = json.load(f)
    for i, record in enumerate(data[:3]):
        print(f'Record {i+1} meta fields:', list(record.get('$meta', {}).keys()))
"
```

## 🔧 故障排除

### 1. JSON格式问题
```bash
# 问题: JSON文件无法读取
# 解决: 检查文件格式
head -1 data.json  # 应该以 [ 开头

# 问题: 动态字段未生成
# 解决: 检查schema配置
milvus-ingest generate --schema schema.json --validate-only
```

### 2. Milvus导入问题
```bash
# 问题: bulk import失败
# 解决: 确认JSON格式正确
python3 -c "import json; json.load(open('data.json'))"  # 验证JSON有效性

# 问题: 动态字段丢失
# 解决: 确认集合开启了dynamic field
# 检查meta.json中的 "enable_dynamic_field": true
```

### 3. 性能问题
```bash
# 问题: JSON生成缓慢
# 解决: 增加批次大小
--batch-size 50000

# 问题: 文件过大
# 解决: 启用文件分割
--file-size 256MB --rows-per-file 500000
```

## 📖 相关命令参考

- [`generate`](../commands/generate.md) - 数据生成选项
- [`to-milvus insert`](../commands/to-milvus.md#insert) - 直接插入
- [`to-milvus import`](../commands/to-milvus.md#import) - 批量导入
- [`schema`](../commands/schema.md) - Schema管理

---

**下一步**: 查看[完整工作流教程](complete-workflow.md)了解端到端的数据处理流程

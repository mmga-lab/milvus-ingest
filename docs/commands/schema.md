# schema 命令详解

`schema` 命令组用于管理数据模式，包括查看内置模式、添加自定义模式、以及获取模式帮助信息。

## 基本语法

```bash
milvus-ingest schema <SUBCOMMAND> [OPTIONS]
```

## 子命令总览

| 子命令 | 功能 | 示例 |
|--------|------|------|
| `list` | 列出所有可用模式 | `milvus-ingest schema list` |
| `show` | 显示模式详细信息 | `milvus-ingest schema show simple` |
| `add` | 添加自定义模式 | `milvus-ingest schema add my_schema file.json` |
| `remove` | 删除自定义模式 | `milvus-ingest schema remove my_schema` |
| `help` | 显示模式格式帮助 | `milvus-ingest schema help` |

---

## list - 列出所有模式

列出所有可用的数据模式，包括内置模式和用户自定义模式。

### 语法
```bash
milvus-ingest schema list [OPTIONS]
```

### 选项
- `--verbose`, `-v`: 显示详细信息（描述、标签等）

### 示例

#### 基础列表
```bash
milvus-ingest schema list
```

输出示例：
```
内置模式 (Built-in Schemas):
├── simple              基础示例模式
├── ecommerce           电商产品目录
├── documents           文档搜索模式
├── images              图像库模式
├── users               用户档案模式
├── videos              视频库模式
├── news                新闻文章模式
├── audio_transcripts   音频转录模式
├── ai_conversations    AI对话模式
├── face_recognition    人脸识别模式
├── ecommerce_partitioned 分区电商模式
└── cardinality_demo    基数约束演示

自定义模式 (Custom Schemas):
├── my_products         我的产品模式
└── test_schema         测试模式

总计: 12个内置模式, 2个自定义模式
```

#### 详细列表
```bash
milvus-ingest schema list --verbose
```

输出示例：
```
内置模式 (Built-in Schemas):

┌─ simple
│  描述: 基础示例，包含ID、文本和向量字段
│  字段: 3个 (Int64主键, VarChar, FloatVector)
│  向量维度: 128
│  标签: [基础, 示例, 测试]

┌─ ecommerce
│  描述: 电商产品目录，适用于商品搜索和推荐
│  字段: 8个 (自增主键, 产品信息, 价格, 评分, 向量)
│  向量维度: 768 (搜索) + 512 (图像)
│  标签: [电商, 产品, 搜索, 推荐]

... (更多模式详情)

自定义模式 (Custom Schemas):

┌─ my_products
│  描述: 我的自定义产品模式
│  添加时间: 2024-01-15 10:30:45
│  标签: [自定义, 产品]
│  字段: 5个
```

---

## show - 显示模式详情

显示指定模式的完整结构和字段定义。

### 语法
```bash
milvus-ingest schema show SCHEMA_ID [OPTIONS]
```

### 参数
- `SCHEMA_ID`: 模式标识符（内置模式名或自定义模式名）

### 选项
- `--format {table,json,yaml}`: 输出格式（默认: table）
- `--fields-only`: 仅显示字段信息，不显示元数据

### 示例

#### 显示内置模式
```bash
milvus-ingest schema show simple
```

输出示例：
```
Schema: simple
描述: 基础示例模式，适合初学者和测试使用
集合名称: simple_collection

字段定义:
┌──────────────┬─────────────┬─────────┬──────────┬─────────────────┐
│ 字段名       │ 类型        │ 主键    │ 可空     │ 参数            │
├──────────────┼─────────────┼─────────┼──────────┼─────────────────┤
│ id           │ Int64       │ ✓       │ ✗        │ auto_id=true    │
│ text         │ VarChar     │ ✗       │ ✗        │ max_length=200  │
│ embedding    │ FloatVector │ ✗       │ ✗        │ dim=128         │
└──────────────┴─────────────┴─────────┴──────────┴─────────────────┘

性能指标:
- 预估行大小: ~540 bytes
- 推荐批大小: 50000 rows
- 1M行预估: ~540 MB
```

#### JSON 格式输出
```bash
milvus-ingest schema show ecommerce --format json
```

输出原始JSON结构，适合程序处理：
```json
{
  "collection_name": "ecommerce_products",
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
    }
    // ... 更多字段
  ]
}
```

#### 仅显示字段
```bash
milvus-ingest schema show documents --fields-only
```

---

## add - 添加自定义模式

将自定义模式添加到模式库中，添加后可以像内置模式一样使用。

### 语法
```bash
milvus-ingest schema add SCHEMA_ID FILE_PATH [OPTIONS]
```

### 参数
- `SCHEMA_ID`: 自定义模式的唯一标识符
- `FILE_PATH`: 模式文件路径（JSON 或 YAML）

### 选项
- `--description TEXT`: 模式描述
- `--tags TEXT`: 标签列表（逗号分隔）
- `--overwrite`: 覆盖已存在的同名模式

### 示例

#### 基础添加
```bash
milvus-ingest schema add my_products ./schemas/products.json
```

#### 带描述和标签
```bash
milvus-ingest schema add my_products ./schemas/products.json \
  --description "我的产品目录模式" \
  --tags "产品,电商,自定义"
```

#### 覆盖现有模式
```bash
milvus-ingest schema add my_products ./schemas/products_v2.json \
  --overwrite \
  --description "产品模式v2.0"
```

#### 添加后验证
```bash
# 添加模式
milvus-ingest schema add test_schema ./test.json

# 验证添加成功
milvus-ingest schema show test_schema

# 测试使用
milvus-ingest generate --builtin test_schema --total-rows 10 --preview
```

### 模式文件格式

支持 JSON 和 YAML 格式：

**JSON 示例 (products.json):**
```json
{
  "collection_name": "my_products",
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
      "min": 0.01,
      "max": 9999.99
    },
    {
      "name": "embedding",
      "type": "FloatVector",
      "dim": 384
    }
  ]
}
```

**YAML 示例 (products.yaml):**
```yaml
collection_name: my_products
fields:
  - name: id
    type: Int64
    is_primary: true
    auto_id: true
  - name: name
    type: VarChar
    max_length: 200
  - name: price
    type: Float
    min: 0.01
    max: 9999.99
  - name: embedding
    type: FloatVector
    dim: 384
```

---

## remove - 删除自定义模式

从模式库中删除自定义模式（无法删除内置模式）。

### 语法
```bash
milvus-ingest schema remove SCHEMA_ID [OPTIONS]
```

### 参数
- `SCHEMA_ID`: 要删除的自定义模式标识符

### 选项
- `--yes`: 跳过确认提示

### 示例

#### 交互式删除
```bash
milvus-ingest schema remove my_products
```

输出：
```
确认删除自定义模式 'my_products'? [y/N]: y
模式 'my_products' 已成功删除
```

#### 强制删除
```bash
milvus-ingest schema remove test_schema --yes
```

#### 批量删除
```bash
# 删除多个测试模式
for schema in test1 test2 test3; do
  milvus-ingest schema remove $schema --yes
done
```

---

## help - 模式格式帮助

显示模式文件的格式说明和字段类型参考。

### 语法
```bash
milvus-ingest schema help [OPTIONS]
```

### 选项
- `--format {brief,detailed}`: 帮助详细程度（默认: brief）
- `--field-type TYPE`: 显示特定字段类型的帮助

### 示例

#### 基础帮助
```bash
milvus-ingest schema help
```

输出概要格式说明和字段类型列表。

#### 详细帮助
```bash
milvus-ingest schema help --format detailed
```

输出完整的格式说明、所有字段类型的详细参数、约束规则等。

#### 特定字段类型帮助
```bash
# 向量字段帮助
milvus-ingest schema help --field-type FloatVector

# 数组字段帮助
milvus-ingest schema help --field-type Array
```

---

## 完整工作流示例

### 1. 探索现有模式

```bash
# 查看所有可用模式
milvus-ingest schema list --verbose

# 研究一个内置模式作为参考
milvus-ingest schema show ecommerce --format json > reference.json
```

### 2. 创建自定义模式

```bash
# 获取模式格式帮助
milvus-ingest schema help --format detailed

# 创建模式文件（根据帮助文档）
cat > my_schema.json << EOF
{
  "collection_name": "my_collection",
  "fields": [
    {
      "name": "id",
      "type": "Int64",
      "is_primary": true
    },
    {
      "name": "title",
      "type": "VarChar",
      "max_length": 500
    },
    {
      "name": "vector",
      "type": "FloatVector",
      "dim": 768
    }
  ]
}
EOF

# 添加到模式库
milvus-ingest schema add my_custom my_schema.json \
  --description "我的自定义模式" \
  --tags "测试,自定义"
```

### 3. 验证和使用

```bash
# 验证模式
milvus-ingest schema show my_custom

# 测试生成数据
milvus-ingest generate --builtin my_custom --total-rows 10 --preview

# 生成实际数据
milvus-ingest generate --builtin my_custom --total-rows 10000 --out ./my_data
```

### 4. 管理模式库

```bash
# 查看所有自定义模式
milvus-ingest schema list | grep "自定义模式" -A 10

# 更新模式（先删除再添加）
milvus-ingest schema remove my_custom --yes
milvus-ingest schema add my_custom my_schema_v2.json --overwrite

# 清理测试模式
milvus-ingest schema remove temp_test --yes
```

## 模式存储位置

自定义模式存储在：
```
~/.milvus-ingest/schemas/
├── my_products.json          # 模式定义
├── my_products.meta.json     # 元数据（描述、标签、添加时间）
└── ...
```

## 最佳实践

### 1. 模式设计原则
- **明确的主键**: 每个模式必须有且仅有一个主键字段
- **合理的向量维度**: 根据实际应用选择合适的向量维度
- **适当的字段约束**: 设置合理的 `max_length`、`min`、`max` 等约束

### 2. 命名规范
```bash
# 推荐的模式命名
my_ecommerce    # 业务领域前缀
test_vectors    # 用途前缀
v2_products     # 版本前缀
```

### 3. 版本管理
```bash
# 模式版本化
milvus-ingest schema add products_v1 schema_v1.json
milvus-ingest schema add products_v2 schema_v2.json

# 保留历史版本，方便回滚
```

### 4. 测试流程
```bash
# 标准测试流程
1. 模式验证: schema help
2. 添加模式: schema add
3. 查看结构: schema show
4. 预览数据: generate --preview
5. 小规模测试: generate --total-rows 1000
6. 正式使用: generate --total-rows 100000
```

## 相关命令

- [`generate`](generate.md) - 使用模式生成数据
- [`to-milvus insert`](to-milvus.md#insert) - 将生成的数据导入 Milvus

---

**下一步**: 查看 [生成命令详解](generate.md) 了解如何使用模式生成数据

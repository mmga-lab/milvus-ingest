# Milvus Ingest - 命令行工具文档

这是 milvus-ingest 命令行工具的完整使用文档，涵盖所有命令和选项的详细说明。

## 📚 文档目录

### 核心命令

- [**generate**](commands/generate.md) - 生成模拟数据
- [**schema**](commands/schema.md) - 模式管理 
- [**upload**](commands/upload.md) - 上传到 S3/MinIO
- [**to-milvus**](commands/to-milvus.md) - Milvus 集成 (insert/import/verify)
- [**clean**](commands/clean.md) - 清理生成的文件

### 教程和示例

- [**快速开始**](tutorials/quickstart.md) - 5分钟上手指南
- [**JSON格式指南**](tutorials/json-format-guide.md) - JSON格式和动态字段详解
- [**性能调优**](tutorials/performance.md) - 大规模数据生成优化
- [**完整工作流**](tutorials/complete-workflow.md) - 从生成到 Milvus 导入的完整流程
- [**自定义模式**](tutorials/custom-schemas.md) - 创建和管理自定义数据模式
- [**实际应用场景**](examples/README.md) - 各种使用场景的具体示例

## 🚀 快速参考

### 最常用命令

```bash
# 快速生成数据预览
milvus-ingest generate --builtin product_catalog --total-rows 1000 --preview

# 生成大规模数据集（Parquet格式）
milvus-ingest generate --builtin ecommerce_search --total-rows 1000000 --out ./data

# 生成JSON格式数据（便于调试）
milvus-ingest generate --builtin product_catalog --total-rows 10000 --format json --out ./json_data

# 生成带动态字段的数据
milvus-ingest generate --builtin news_articles --total-rows 5000 --format json --out ./dynamic_data

# 查看所有可用模式
milvus-ingest schema list

# 直接导入到 Milvus（支持Parquet和JSON）
milvus-ingest to-milvus insert ./data
milvus-ingest to-milvus insert ./json_data
```

### 命令结构

```
milvus-ingest <COMMAND> [OPTIONS]

主要命令组：
  generate     数据生成
  schema       模式管理  
  upload       文件上传
  to-milvus    Milvus 集成
  clean        清理工具
```

## 📖 使用模式

### 1. 内置模式 (推荐)
使用预定义的数据模式，适合快速开始和测试：

```bash
milvus-ingest generate --builtin simple --total-rows 10000
```

**可用的内置模式：**
- `product_catalog` - 产品目录（入门推荐）
- `ecommerce_search` - 电商搜索（包含可空字段）
- `news_articles` - 新闻文章（支持动态字段）
- `document_search` - 文档搜索（稀疏向量+BM25）
- `multi_tenant_data` - 多租户数据（分区支持）
- `multimedia_content` - 多媒体内容（多种向量类型）

### 2. 自定义模式
使用 JSON/YAML 文件定义自己的数据结构：

```bash
milvus-ingest generate --schema my_schema.json --total-rows 10000
```

### 3. 模式管理
添加、管理和重用自定义模式：

```bash
# 添加自定义模式
milvus-ingest schema add my_products schema.json

# 像内置模式一样使用
milvus-ingest generate --builtin my_products --total-rows 10000
```

## 🔧 高级功能

### 性能优化
- **大批量处理**: `--batch-size 100000` (默认: 50000)
- **文件分割**: `--file-size 256MB`, `--rows-per-file 1000000`
- **格式选择**: `--format parquet` (最快) 或 `json` (标准数组格式)
- **动态字段**: 支持Milvus动态字段，使用 `$meta` 字段存储

### 集成功能
- **S3/MinIO 上传**: 直接上传生成的数据到云存储
- **Milvus 导入**: 支持直接插入和批量导入两种方式
- **Docker 环境**: 完整的本地测试环境

## 🎯 按场景选择

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| 快速测试 | `generate --builtin simple --preview` | 预览数据结构 |
| 原型开发 | `generate --builtin <type> --total-rows 10000` | 小规模真实数据 |
| 性能测试 | `generate --builtin <type> --total-rows 1000000 --batch-size 100000` | 大规模数据生成 |
| 生产部署 | 完整工作流 (生成→上传→导入) | 参见[完整工作流教程](tutorials/complete-workflow.md) |

## 📋 支持的数据类型

### 标量类型
- **数值**: Int8, Int16, Int32, Int64, Float, Double, Bool
- **文本**: VarChar, String (需要 max_length)  
- **复杂**: JSON, Array

### 向量类型
- **FloatVector**: 32位浮点向量 (需要 dim)
- **BinaryVector**: 二进制向量 (需要 dim)
- **Float16Vector**: 16位浮点向量 (需要 dim)
- **BFloat16Vector**: Brain Float 向量 (需要 dim)
- **SparseFloatVector**: 稀疏浮点向量

## 🚨 重要提示

1. **输出格式**: 工具始终生成目录（包含数据文件和 meta.json），而不是单个文件
2. **性能优化**: 默认针对大规模数据生成进行优化（50K+ 行批处理）
3. **内存管理**: 使用流式写入防止大数据集时内存耗尽
4. **文件分割**: 自动分割大文件（256MB 或 1M 行）以便处理

## 🆘 获取帮助

- 查看具体命令帮助: `milvus-ingest <command> --help`
- 模式格式帮助: `milvus-ingest schema help`
- 问题反馈: [GitHub Issues](https://github.com/zilliz/milvus-ingest/issues)

---

**下一步**: 查看 [快速开始教程](tutorials/quickstart.md) 开始使用！
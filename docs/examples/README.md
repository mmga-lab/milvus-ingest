# 实际应用场景示例

**注意**: 示例代码和schema文件现在位于项目根目录的 `examples/` 文件夹中。

这里提供了各种业务场景下使用 milvus-ingest 的具体示例，帮助你快速应用到实际项目中。

## 📋 场景分类

### 🛒 电商推荐系统
- [商品相似度搜索](ecommerce-similarity.md) - 基于商品特征的相似商品推荐
- [用户行为分析](user-behavior.md) - 用户购买偏好向量化分析
- [个性化推荐](personalized-recommendations.md) - 用户画像与商品匹配

### 📚 知识管理系统  
- [文档语义搜索](document-search.md) - 企业知识库智能检索
- [FAQ 智能问答](faq-chatbot.md) - 客服机器人知识库
- [代码搜索引擎](code-search.md) - 代码片段语义检索

### 🎵 多媒体应用
- [图像相似性搜索](image-similarity.md) - 以图搜图功能实现
- [音频指纹识别](audio-fingerprint.md) - 音乐识别和版权检测
- [视频内容分析](video-analysis.md) - 视频场景和内容理解

### 🤖 AI 应用场景
- [对话系统](conversation-ai.md) - 聊天机器人对话历史分析
- [情感分析](sentiment-analysis.md) - 用户评论情感向量化
- [内容审核](content-moderation.md) - 违规内容自动识别

### 🔐 安全与监控
- [异常检测](anomaly-detection.md) - 系统行为异常识别
- [欺诈检测](fraud-detection.md) - 金融交易风险识别
- [网络安全](cybersecurity.md) - 网络流量异常监控

## 🚀 快速开始

### 选择适合的场景

根据你的业务需求选择最相近的场景：

```bash
# 电商推荐系统
milvus-ingest generate --builtin ecommerce --total-rows 100000 --out ./ecommerce_demo

# 文档搜索系统
milvus-ingest generate --builtin documents --total-rows 50000 --out ./docs_demo

# 多媒体应用
milvus-ingest generate --builtin images --total-rows 30000 --out ./media_demo

# AI 对话系统
milvus-ingest generate --builtin ai_conversations --total-rows 20000 --out ./ai_demo
```

### 使用示例Schema

查看项目根目录的 `examples/` 文件夹获取示例代码和schema文件：

```bash
# 使用示例performance测试schema
milvus-ingest generate --schema examples/schemas/performance_test.json --total-rows 1000

# 验证schema
milvus-ingest generate --schema examples/schemas/performance_test.json --validate-only
```

---

更多详细内容，请查看项目根目录的 [examples/README.md](../../examples/README.md) 文件。
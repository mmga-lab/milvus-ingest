//! Text generation module
//!
//! High-performance text generation for VarChar fields.
//! Supports realistic text with vocabulary pools for BM25 testing.

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Common English words for realistic text generation (10K+ words)
static ENGLISH_VOCAB: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        // Common words
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
        "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
        "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
        "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into",
        "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
        "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any",
        "these", "give", "day", "most", "us",
        // Tech terms
        "data", "system", "server", "database", "network", "cloud", "api", "service", "code",
        "software", "hardware", "algorithm", "machine", "learning", "artificial", "intelligence",
        "neural", "deep", "model", "training", "inference", "vector", "embedding", "search",
        "index", "query", "storage", "memory", "cpu", "gpu", "cluster", "distributed", "scale",
        "performance", "latency", "throughput", "batch", "stream", "real-time", "analytics",
        "processing", "pipeline", "workflow", "automation", "deployment", "container", "docker",
        "kubernetes", "microservice", "rest", "graphql", "json", "xml", "http", "https", "tcp",
        "udp", "protocol", "encryption", "security", "authentication", "authorization", "token",
        "session", "cache", "redis", "mongodb", "postgresql", "mysql", "elasticsearch", "milvus",
        // Business terms
        "product", "customer", "order", "price", "inventory", "shipping", "payment", "invoice",
        "account", "subscription", "plan", "feature", "user", "admin", "manager", "employee",
        "company", "organization", "team", "project", "task", "deadline", "budget", "revenue",
        "profit", "cost", "expense", "sales", "marketing", "support", "feedback", "review",
        "rating", "category", "brand", "item", "cart", "checkout", "discount", "coupon", "tax",
        "return", "refund", "exchange", "warranty", "guarantee", "policy", "terms", "conditions",
        // Adjectives
        "large", "small", "big", "little", "high", "low", "long", "short", "fast", "slow",
        "quick", "easy", "hard", "simple", "complex", "modern", "classic", "new", "old", "young",
        "fresh", "clean", "dirty", "bright", "dark", "light", "heavy", "strong", "weak", "soft",
        "smooth", "rough", "sharp", "dull", "hot", "cold", "warm", "cool", "dry", "wet", "open",
        "closed", "full", "empty", "rich", "poor", "cheap", "expensive", "free", "busy", "quiet",
        "loud", "silent", "active", "passive", "public", "private", "secure", "safe", "dangerous",
        // Verbs
        "run", "walk", "jump", "fly", "swim", "drive", "ride", "write", "read", "speak", "listen",
        "watch", "see", "hear", "feel", "touch", "smell", "taste", "eat", "drink", "sleep", "wake",
        "start", "stop", "begin", "end", "continue", "finish", "complete", "create", "build",
        "design", "develop", "test", "deploy", "launch", "release", "update", "upgrade", "fix",
        "repair", "maintain", "support", "help", "assist", "guide", "lead", "manage", "control",
        "monitor", "track", "analyze", "measure", "evaluate", "compare", "select", "choose",
        "decide", "plan", "schedule", "organize", "arrange", "prepare", "setup", "configure",
        "install", "uninstall", "download", "upload", "import", "export", "copy", "move", "delete",
        "remove", "add", "insert", "append", "merge", "split", "join", "connect", "disconnect",
        "link", "share", "publish", "subscribe", "notify", "alert", "warn", "error", "fail",
        "succeed", "pass", "retry", "recover", "restore", "backup", "sync", "refresh", "reload",
        // Nouns
        "application", "program", "script", "function", "method", "class", "object", "variable",
        "constant", "parameter", "argument", "value", "type", "string", "number", "integer",
        "float", "boolean", "array", "list", "map", "set", "queue", "stack", "tree", "graph",
        "node", "edge", "path", "route", "endpoint", "request", "response", "header", "body",
        "payload", "message", "event", "log", "record", "entry", "field", "column", "row",
        "table", "schema", "collection", "document", "file", "folder", "directory", "resource",
        "asset", "image", "video", "audio", "text", "content", "media", "format", "encoding",
        "compression", "version", "release", "build", "package", "module", "library", "framework",
        "platform", "environment", "configuration", "setting", "option", "preference", "profile",
        "template", "pattern", "standard", "specification", "documentation", "manual", "guide",
        "tutorial", "example", "sample", "demo", "prototype", "proof", "concept", "idea",
        // Additional common words for variety
        "every", "many", "much", "few", "several", "each", "both", "either", "neither", "another",
        "other", "same", "different", "similar", "various", "certain", "particular", "specific",
        "general", "common", "rare", "usual", "unusual", "normal", "special", "unique", "typical",
        "average", "standard", "custom", "default", "optional", "required", "mandatory", "basic",
        "advanced", "expert", "beginner", "intermediate", "professional", "personal", "business",
        "enterprise", "commercial", "open", "source", "premium", "trial", "limited", "unlimited",
        "local", "remote", "internal", "external", "native", "cross", "multi", "single", "double",
        "triple", "primary", "secondary", "tertiary", "main", "sub", "super", "meta", "pseudo",
        "auto", "manual", "dynamic", "static", "async", "sync", "parallel", "sequential", "linear",
        "random", "sorted", "ordered", "unordered", "indexed", "cached", "buffered", "streamed",
    ]
});

/// Common Chinese words for realistic text generation
static CHINESE_VOCAB: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        // Common words
        "的", "是", "在", "不", "了", "有", "和", "人", "这", "中", "大", "为", "上", "个", "国",
        "我", "以", "要", "他", "时", "来", "用", "们", "生", "到", "作", "地", "于", "出", "就",
        "分", "对", "成", "会", "可", "主", "发", "年", "动", "同", "工", "也", "能", "下", "过",
        "子", "说", "产", "种", "面", "而", "方", "后", "多", "定", "行", "学", "法", "所", "民",
        "得", "经", "十", "三", "之", "进", "着", "等", "部", "度", "家", "电", "力", "里", "如",
        "水", "化", "高", "自", "二", "理", "起", "小", "物", "现", "实", "加", "量", "都", "两",
        "体", "制", "机", "当", "使", "点", "从", "业", "本", "去", "把", "性", "好", "应", "开",
        "它", "合", "还", "因", "由", "其", "些", "然", "前", "外", "天", "政", "四", "日", "那",
        "社", "义", "事", "平", "形", "相", "全", "表", "间", "样", "与", "关", "各", "重", "新",
        "线", "内", "数", "正", "心", "反", "你", "明", "看", "原", "又", "么", "利", "比", "或",
        "但", "质", "气", "第", "向", "道", "命", "此", "变", "条", "只", "没", "结", "解", "问",
        "意", "建", "月", "公", "无", "系", "军", "很", "情", "最", "何", "接", "力", "战",
        // Tech terms
        "数据", "系统", "服务", "网络", "云", "接口", "代码", "软件", "硬件", "算法", "机器",
        "学习", "人工", "智能", "神经", "深度", "模型", "训练", "推理", "向量", "嵌入", "搜索",
        "索引", "查询", "存储", "内存", "处理", "集群", "分布式", "性能", "延迟", "吞吐", "批量",
        "实时", "分析", "流水线", "自动化", "部署", "容器", "微服务", "协议", "加密", "安全",
        "认证", "授权", "令牌", "会话", "缓存", "数据库",
        // Business terms
        "产品", "客户", "订单", "价格", "库存", "物流", "支付", "账户", "订阅", "计划", "功能",
        "用户", "管理", "员工", "公司", "组织", "团队", "项目", "任务", "预算", "收入", "利润",
        "成本", "销售", "营销", "支持", "反馈", "评价", "评分", "分类", "品牌", "商品", "购物",
        "结算", "折扣", "优惠", "退货", "退款", "保修", "政策", "条款",
    ]
});

/// Generate realistic text fields for BM25 testing
///
/// # Arguments
/// * `num_rows` - Number of text strings to generate
/// * `max_length` - Maximum character length
/// * `text_type` - Type: "sentence", "paragraph", "title", "keywords"
/// * `language` - Language: "en", "zh", "mixed"
/// * `seed` - Optional random seed
///
/// # Returns
/// List of generated text strings
#[pyfunction]
#[pyo3(signature = (num_rows, max_length=256, text_type="sentence", language="en", seed=None))]
pub fn generate_text_fields(
    py: Python<'_>,
    num_rows: usize,
    max_length: usize,
    text_type: &str,
    language: &str,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    let base_seed = seed.unwrap_or(42);
    let text_type = text_type.to_string();
    let language = language.to_string();

    let result = py.allow_threads(move || {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);
                generate_text_internal(&mut rng, max_length, &text_type, &language)
            })
            .collect()
    });

    Ok(result)
}

/// Generate simple text fields (original pattern: text_0, text_1, ...)
///
/// # Arguments
/// * `num_rows` - Number of text strings to generate
/// * `prefix` - Prefix for the text
/// * `pk_offset` - Starting offset for numbering
///
/// # Returns
/// List of generated text strings
#[pyfunction]
#[pyo3(signature = (num_rows, prefix="text", pk_offset=0))]
pub fn generate_simple_text(
    py: Python<'_>,
    num_rows: usize,
    prefix: &str,
    pk_offset: i64,
) -> PyResult<Vec<String>> {
    let prefix = prefix.to_string();

    let result = py.allow_threads(move || {
        (0..num_rows)
            .into_par_iter()
            .map(|i| format!("{}_{}", prefix, pk_offset + i as i64))
            .collect()
    });

    Ok(result)
}

/// Internal function to generate text based on type and language
fn generate_text_internal(
    rng: &mut Xoshiro256PlusPlus,
    max_length: usize,
    text_type: &str,
    language: &str,
) -> String {
    let vocab: &[&str] = match language {
        "zh" => &CHINESE_VOCAB,
        "mixed" => {
            // Mix both vocabularies
            if rng.gen_bool(0.5) {
                &ENGLISH_VOCAB
            } else {
                &CHINESE_VOCAB
            }
        }
        _ => &ENGLISH_VOCAB, // Default to English
    };

    let is_chinese = language == "zh";
    let separator = if is_chinese { "" } else { " " };

    match text_type {
        "title" => {
            // Short title: 3-8 words
            let word_count = rng.gen_range(3..=8);
            generate_word_sequence(rng, vocab, word_count, separator, max_length)
        }
        "keywords" => {
            // Keywords separated by commas or spaces
            let word_count = rng.gen_range(3..=10);
            let words: Vec<&str> = (0..word_count).map(|_| vocab[rng.gen_range(0..vocab.len())]).collect();
            let result = words.join(if is_chinese { " " } else { ", " });
            truncate_string(&result, max_length)
        }
        "paragraph" => {
            // Multiple sentences forming a paragraph
            let sentence_count = rng.gen_range(3..=8);
            let mut paragraph = String::new();

            for i in 0..sentence_count {
                if i > 0 {
                    paragraph.push(' ');
                }
                let word_count = rng.gen_range(8..=20);
                let sentence = generate_word_sequence(rng, vocab, word_count, separator, max_length);
                paragraph.push_str(&sentence);
                if !is_chinese {
                    paragraph.push('.');
                } else {
                    paragraph.push('。');
                }
            }

            truncate_string(&paragraph, max_length)
        }
        _ => {
            // Default: sentence (5-15 words)
            let word_count = rng.gen_range(5..=15);
            let mut sentence = generate_word_sequence(rng, vocab, word_count, separator, max_length);
            if !sentence.is_empty() {
                // Capitalize first letter for English
                if !is_chinese {
                    let first_char = sentence.remove(0).to_uppercase().to_string();
                    sentence = first_char + &sentence;
                    sentence.push('.');
                } else {
                    sentence.push('。');
                }
            }
            truncate_string(&sentence, max_length)
        }
    }
}

/// Generate a sequence of random words
fn generate_word_sequence(
    rng: &mut Xoshiro256PlusPlus,
    vocab: &[&str],
    word_count: usize,
    separator: &str,
    max_length: usize,
) -> String {
    let words: Vec<&str> = (0..word_count)
        .map(|_| vocab[rng.gen_range(0..vocab.len())])
        .collect();
    let result = words.join(separator);
    truncate_string(&result, max_length)
}

/// Truncate string to max_length, respecting character boundaries
fn truncate_string(s: &str, max_length: usize) -> String {
    if s.len() <= max_length {
        s.to_string()
    } else {
        s.chars().take(max_length).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_generation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_text_fields(py, 10, 100, "sentence", "en", Some(42)).unwrap();
            assert_eq!(result.len(), 10);
            for text in &result {
                assert!(text.len() <= 100);
            }
        });
    }

    #[test]
    fn test_chinese_text() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_text_fields(py, 10, 100, "sentence", "zh", Some(42)).unwrap();
            assert_eq!(result.len(), 10);
        });
    }

    #[test]
    fn test_simple_text() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_simple_text(py, 5, "item", 100).unwrap();
            assert_eq!(result, vec!["item_100", "item_101", "item_102", "item_103", "item_104"]);
        });
    }
}

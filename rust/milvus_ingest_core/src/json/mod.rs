//! JSON field generation module
//!
//! High-performance JSON field generation for various patterns:
//! - ecommerce: Product metadata
//! - event: Event/log data
//! - config: Configuration objects
//! - analytics: Analytics data
//! - document: Document metadata

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde_json::{json, Value};

/// Predefined categories for ecommerce pattern
const CATEGORIES: &[&str] = &[
    "Electronics",
    "Clothing",
    "Home & Garden",
    "Sports",
    "Books",
    "Toys",
    "Health",
    "Automotive",
    "Food",
    "Jewelry",
];

/// Predefined brands for ecommerce pattern
const BRANDS: &[&str] = &[
    "TechCorp",
    "StyleMax",
    "HomeLife",
    "SportPro",
    "ReadMore",
    "PlayTime",
    "WellBeing",
    "AutoParts",
    "FreshFood",
    "GemStone",
];

/// Predefined event types
const EVENT_TYPES: &[&str] = &[
    "click",
    "view",
    "purchase",
    "add_to_cart",
    "search",
    "login",
    "logout",
    "signup",
    "error",
    "page_view",
];

/// Predefined log levels
const LOG_LEVELS: &[&str] = &["debug", "info", "warn", "error", "critical"];

/// Predefined status values
const STATUSES: &[&str] = &["active", "inactive", "pending", "completed", "failed", "cancelled"];

/// Colors for product attributes
const COLORS: &[&str] = &["red", "blue", "green", "black", "white"];

/// Sizes for product attributes
const SIZES: &[&str] = &["S", "M", "L", "XL"];

/// Devices
const DEVICES: &[&str] = &["desktop", "mobile", "tablet"];

/// Compression types
const COMPRESSIONS: &[&str] = &["none", "gzip", "lz4", "zstd"];

/// Metric types
const METRIC_TYPES: &[&str] = &["pageview", "conversion", "revenue", "signup"];

/// Countries
const COUNTRIES: &[&str] = &["US", "CN", "JP", "DE", "GB", "FR", "IN", "BR"];

/// Browsers
const BROWSERS: &[&str] = &["Chrome", "Firefox", "Safari", "Edge"];

/// Sources
const SOURCES: &[&str] = &["google", "facebook", "twitter", "direct", "email"];

/// Languages
const LANGUAGES: &[&str] = &["en", "zh", "ja", "de", "fr", "es"];

/// File types
const FILE_TYPES: &[&str] = &["pdf", "docx", "txt", "md", "html"];

/// Generate JSON fields with various patterns
///
/// # Arguments
/// * `num_rows` - Number of JSON objects to generate
/// * `pattern` - Pattern: "ecommerce", "event", "config", "analytics", "document"
/// * `pk_offset` - Primary key offset for ID generation
/// * `seed` - Optional random seed
///
/// # Returns
/// List of Python dicts representing JSON objects
#[pyfunction]
#[pyo3(signature = (num_rows, pattern="ecommerce", pk_offset=0, seed=None))]
pub fn generate_json_fields(
    py: Python<'_>,
    num_rows: usize,
    pattern: &str,
    pk_offset: i64,
    seed: Option<u64>,
) -> PyResult<Vec<Py<PyDict>>> {
    let base_seed = seed.unwrap_or(42);
    let pattern = pattern.to_string();

    // Generate JSON values in parallel
    let json_values: Vec<Value> = py.allow_threads(|| {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);
                let pk = pk_offset + row_idx as i64;
                generate_json_value(&mut rng, &pattern, pk)
            })
            .collect()
    });

    // Convert to Python dicts
    json_values
        .into_iter()
        .map(|v| json_to_pydict(py, &v))
        .collect()
}

/// Generate a JSON value based on pattern
fn generate_json_value(rng: &mut Xoshiro256PlusPlus, pattern: &str, pk: i64) -> Value {
    match pattern {
        "ecommerce" => generate_ecommerce_json(rng, pk),
        "event" => generate_event_json(rng, pk),
        "config" => generate_config_json(rng, pk),
        "analytics" => generate_analytics_json(rng, pk),
        "document" => generate_document_json(rng, pk),
        _ => generate_ecommerce_json(rng, pk), // Default to ecommerce
    }
}

/// Generate ecommerce product metadata
fn generate_ecommerce_json(rng: &mut Xoshiro256PlusPlus, pk: i64) -> Value {
    let category = CATEGORIES[rng.gen_range(0..CATEGORIES.len())];
    let brand = BRANDS[rng.gen_range(0..BRANDS.len())];
    let price: f64 = (rng.gen_range(100..100000) as f64) / 100.0;
    let rating: f64 = (rng.gen_range(10..50) as f64) / 10.0;
    let review_count: i32 = rng.gen_range(0..10000);
    let in_stock: bool = rng.gen_bool(0.8);
    let discount_percent: i32 = rng.gen_range(0..30);
    let stock_quantity: i32 = if in_stock { rng.gen_range(1..1000) } else { 0 };
    let color = COLORS[rng.gen_range(0..COLORS.len())];
    let size = SIZES[rng.gen_range(0..SIZES.len())];
    let weight_kg: f64 = (rng.gen_range(10..10000) as f64) / 100.0;

    json!({
        "product_id": pk,
        "category": category,
        "brand": brand,
        "price": price,
        "original_price": price * 1.2,
        "discount_percent": discount_percent,
        "rating": rating,
        "review_count": review_count,
        "in_stock": in_stock,
        "stock_quantity": stock_quantity,
        "tags": generate_tags(rng, 3),
        "attributes": {
            "color": color,
            "size": size,
            "weight_kg": weight_kg
        }
    })
}

/// Generate event/log data
fn generate_event_json(rng: &mut Xoshiro256PlusPlus, pk: i64) -> Value {
    let event_type = EVENT_TYPES[rng.gen_range(0..EVENT_TYPES.len())];
    let level = LOG_LEVELS[rng.gen_range(0..LOG_LEVELS.len())];
    let timestamp: i64 = 1700000000000i64 + rng.gen_range(0..86400000);
    let user_id: i64 = rng.gen_range(1..1000000);
    let session_id = format!("sess_{}", rng.gen::<u32>());
    let ip_address = format!(
        "{}.{}.{}.{}",
        rng.gen_range(1..255),
        rng.gen_range(0..255),
        rng.gen_range(0..255),
        rng.gen_range(1..255)
    );
    let response_time_ms: i32 = rng.gen_range(10..5000);
    let success: bool = rng.gen_bool(0.95);
    let page = format!("/page/{}", rng.gen_range(1..100));
    let referrer = if rng.gen_bool(0.7) {
        "https://example.com"
    } else {
        ""
    };
    let device = DEVICES[rng.gen_range(0..DEVICES.len())];

    json!({
        "event_id": pk,
        "event_type": event_type,
        "level": level,
        "timestamp": timestamp,
        "user_id": user_id,
        "session_id": session_id,
        "ip_address": ip_address,
        "user_agent": "Mozilla/5.0 (compatible; TestBot/1.0)",
        "response_time_ms": response_time_ms,
        "success": success,
        "metadata": {
            "page": page,
            "referrer": referrer,
            "device": device
        }
    })
}

/// Generate configuration objects
fn generate_config_json(rng: &mut Xoshiro256PlusPlus, pk: i64) -> Value {
    let version = format!(
        "{}.{}.{}",
        rng.gen_range(1..10),
        rng.gen_range(0..20),
        rng.gen_range(0..100)
    );
    let enabled: bool = rng.gen_bool(0.8);
    let priority: i32 = rng.gen_range(1..10);
    let max_connections: i32 = rng.gen_range(10..1000);
    let timeout_ms: i32 = rng.gen_range(1000..30000);
    let retry_count: i32 = rng.gen_range(1..5);
    let buffer_size: i64 = 1024 * (1i64 << rng.gen_range(0..10));
    let compression = COMPRESSIONS[rng.gen_range(0..COMPRESSIONS.len())];
    let log_level = LOG_LEVELS[rng.gen_range(0..LOG_LEVELS.len())];
    let created_at: i64 = 1700000000i64 + rng.gen_range(0..86400);
    let updated_at: i64 = 1700000000i64 + rng.gen_range(86400..172800);

    json!({
        "config_id": pk,
        "name": format!("config_{}", pk),
        "version": version,
        "enabled": enabled,
        "priority": priority,
        "settings": {
            "max_connections": max_connections,
            "timeout_ms": timeout_ms,
            "retry_count": retry_count,
            "buffer_size": buffer_size,
            "compression": compression,
            "log_level": log_level
        },
        "tags": generate_tags(rng, 2),
        "created_at": created_at,
        "updated_at": updated_at
    })
}

/// Generate analytics data
fn generate_analytics_json(rng: &mut Xoshiro256PlusPlus, pk: i64) -> Value {
    let metric_type = METRIC_TYPES[rng.gen_range(0..METRIC_TYPES.len())];
    let date = format!(
        "2024-{:02}-{:02}",
        rng.gen_range(1..13),
        rng.gen_range(1..29)
    );
    let hour: i32 = rng.gen_range(0..24);
    let country = COUNTRIES[rng.gen_range(0..COUNTRIES.len())];
    let device_type = DEVICES[rng.gen_range(0..DEVICES.len())];
    let browser = BROWSERS[rng.gen_range(0..BROWSERS.len())];
    let views: i32 = rng.gen_range(0..10000);
    let clicks: i32 = rng.gen_range(0..1000);
    let conversions: i32 = rng.gen_range(0..100);
    let revenue: f64 = (rng.gen_range(0..100000) as f64) / 100.0;
    let bounce_rate: f64 = (rng.gen_range(0..100) as f64) / 100.0;
    let avg_session_duration: i32 = rng.gen_range(0..1800);
    let campaign = if rng.gen_bool(0.3) {
        format!("campaign_{}", rng.gen_range(1..20))
    } else {
        "organic".to_string()
    };
    let source = SOURCES[rng.gen_range(0..SOURCES.len())];

    json!({
        "record_id": pk,
        "metric_type": metric_type,
        "date": date,
        "hour": hour,
        "country": country,
        "device_type": device_type,
        "browser": browser,
        "metrics": {
            "views": views,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": revenue,
            "bounce_rate": bounce_rate,
            "avg_session_duration": avg_session_duration
        },
        "dimensions": {
            "campaign": campaign,
            "source": source
        }
    })
}

/// Generate document metadata
fn generate_document_json(rng: &mut Xoshiro256PlusPlus, pk: i64) -> Value {
    let status = STATUSES[rng.gen_range(0..STATUSES.len())];
    let author_id: i64 = rng.gen_range(1..10000);
    let version: i32 = rng.gen_range(1..100);
    let word_count: i32 = rng.gen_range(100..50000);
    let language = LANGUAGES[rng.gen_range(0..LANGUAGES.len())];
    let file_type = FILE_TYPES[rng.gen_range(0..FILE_TYPES.len())];
    let file_size_bytes: i64 = rng.gen_range(1024..10485760);
    let created_at: i64 = 1700000000i64 + rng.gen_range(0..86400);
    let modified_at: i64 = 1700000000i64 + rng.gen_range(86400..172800);
    let is_public: bool = rng.gen_bool(0.3);

    json!({
        "doc_id": pk,
        "title": format!("Document {}", pk),
        "author_id": author_id,
        "status": status,
        "version": version,
        "word_count": word_count,
        "language": language,
        "file_type": file_type,
        "file_size_bytes": file_size_bytes,
        "permissions": {
            "read": generate_permission_list(rng),
            "write": generate_permission_list(rng),
            "admin": generate_permission_list(rng)
        },
        "tags": generate_tags(rng, 4),
        "created_at": created_at,
        "modified_at": modified_at,
        "is_public": is_public
    })
}

/// Generate random tags
fn generate_tags(rng: &mut Xoshiro256PlusPlus, max_count: usize) -> Vec<String> {
    let count = rng.gen_range(1..=max_count);
    let all_tags = [
        "featured",
        "sale",
        "new",
        "popular",
        "limited",
        "premium",
        "basic",
        "pro",
        "enterprise",
        "beta",
    ];
    let mut tags = Vec::with_capacity(count);
    let mut indices: Vec<usize> = (0..all_tags.len()).collect();
    indices.shuffle(rng);
    for i in 0..count {
        tags.push(all_tags[indices[i]].to_string());
    }
    tags
}

/// Generate permission list (user/group IDs)
fn generate_permission_list(rng: &mut Xoshiro256PlusPlus) -> Vec<String> {
    let count = rng.gen_range(0..5);
    (0..count)
        .map(|_| {
            if rng.gen_bool(0.5) {
                format!("user_{}", rng.gen_range(1..1000))
            } else {
                format!("group_{}", rng.gen_range(1..100))
            }
        })
        .collect()
}

/// Convert serde_json Value to Python dict
fn json_to_pydict(py: Python<'_>, value: &Value) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    if let Value::Object(map) = value {
        for (key, val) in map {
            let py_val = json_value_to_py(py, val)?;
            dict.set_item(key, py_val)?;
        }
    }

    Ok(dict.into())
}

/// Convert serde_json Value to Python object
fn json_value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Array(arr) => {
            let list: Vec<PyObject> = arr
                .iter()
                .map(|v| json_value_to_py(py, v))
                .collect::<PyResult<_>>()?;
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        Value::Object(_) => {
            let dict = json_to_pydict(py, value)?;
            Ok(dict.into_any())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_generation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_json_fields(py, 10, "ecommerce", 0, Some(42)).unwrap();
            assert_eq!(result.len(), 10);
        });
    }

    #[test]
    fn test_all_patterns() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            for pattern in &["ecommerce", "event", "config", "analytics", "document"] {
                let result = generate_json_fields(py, 5, pattern, 0, Some(42)).unwrap();
                assert_eq!(result.len(), 5);
            }
        });
    }
}

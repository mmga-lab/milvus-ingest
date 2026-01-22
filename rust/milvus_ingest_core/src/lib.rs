//! High-performance Rust core for milvus-ingest data generation
//!
//! This crate provides SIMD-accelerated and multi-threaded implementations
//! for generating various Milvus data types.

use pyo3::prelude::*;

mod array;
mod json;
mod text;
mod vectors;

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Vector generation functions
    m.add_function(wrap_pyfunction!(vectors::float16::generate_float16_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(
        vectors::float16::generate_bfloat16_vectors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vectors::float16::generate_normalized_float16_vectors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(vectors::sparse::generate_sparse_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(vectors::sparse::generate_sparse_vectors_nnz, m)?)?;
    m.add_function(wrap_pyfunction!(vectors::float::generate_float_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(
        vectors::float::generate_normalized_float_vectors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(vectors::float::generate_float_vectors_range, m)?)?;

    // Text generation functions
    m.add_function(wrap_pyfunction!(text::generate_text_fields, m)?)?;
    m.add_function(wrap_pyfunction!(text::generate_simple_text, m)?)?;

    // JSON generation functions
    m.add_function(wrap_pyfunction!(json::generate_json_fields, m)?)?;

    // Array generation functions
    m.add_function(wrap_pyfunction!(array::generate_varchar_array, m)?)?;
    m.add_function(wrap_pyfunction!(array::generate_int_array, m)?)?;
    m.add_function(wrap_pyfunction!(array::generate_float_array, m)?)?;

    // Numeric generation functions
    m.add_function(wrap_pyfunction!(array::generate_int64_batch, m)?)?;
    m.add_function(wrap_pyfunction!(array::generate_float64_batch, m)?)?;
    m.add_function(wrap_pyfunction!(array::generate_float32_batch_2d, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_features, m)?)?;

    Ok(())
}

/// Get the version of milvus_ingest_core
#[pyfunction]
fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get CPU features available for SIMD
#[pyfunction]
fn get_cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if is_x86_feature_detected!("avx") {
            features.push("avx");
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2");
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1");
        }
        if is_x86_feature_detected!("fma") {
            features.push("fma");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        features.push("neon");
    }

    if features.is_empty() {
        features.push("scalar");
    }

    features
}

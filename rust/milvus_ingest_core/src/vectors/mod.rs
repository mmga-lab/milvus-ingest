//! Vector generation modules
//!
//! Provides high-performance implementations for generating various vector types:
//! - Float16/BFloat16 vectors (SIMD + multi-threaded)
//! - Sparse vectors (parallel generation)
//! - Float vectors with normalization

pub mod float;
pub mod float16;
pub mod sparse;

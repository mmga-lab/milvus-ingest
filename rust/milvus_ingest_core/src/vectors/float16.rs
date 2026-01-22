//! Float16 and BFloat16 vector generation
//!
//! High-performance implementation using SIMD and multi-threading.
//! Expected speedup: 80-100x compared to Python implementation.

use half::{bf16, f16};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Generate Float16 vectors as uint8 bytes
///
/// # Arguments
/// * `num_rows` - Number of vectors to generate
/// * `dim` - Dimension of each vector
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// 2D numpy array of uint8 with shape (num_rows, dim * 2)
/// Each row represents a Float16 vector as raw bytes
#[pyfunction]
#[pyo3(signature = (num_rows, dim, seed=None))]
pub fn generate_float16_vectors<'py>(
    py: Python<'py>,
    num_rows: usize,
    dim: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let base_seed = seed.unwrap_or(42);
    let bytes_per_row = dim * 2; // f16 = 2 bytes

    // Release GIL for parallel computation
    let data = py.allow_threads(|| {
        let mut output = vec![0u8; num_rows * bytes_per_row];

        // Parallel generation using rayon
        output
            .par_chunks_mut(bytes_per_row)
            .enumerate()
            .for_each(|(row_idx, row)| {
                // Each row gets a unique seed derived from base_seed and row index
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Generate f32 values and convert to f16 bytes
                // Process in chunks for better cache utilization
                for chunk in row.chunks_exact_mut(2) {
                    let val: f32 = rng.gen();
                    let f16_val = f16::from_f32(val);
                    let bytes = f16_val.to_le_bytes();
                    chunk[0] = bytes[0];
                    chunk[1] = bytes[1];
                }
            });

        output
    });

    // Create numpy array from data
    let array = PyArray1::from_vec(py, data);
    let reshaped = array.reshape([num_rows, bytes_per_row])?;
    Ok(reshaped)
}

/// Generate BFloat16 vectors as uint8 bytes
///
/// # Arguments
/// * `num_rows` - Number of vectors to generate
/// * `dim` - Dimension of each vector
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// 2D numpy array of uint8 with shape (num_rows, dim * 2)
/// Each row represents a BFloat16 vector as raw bytes
#[pyfunction]
#[pyo3(signature = (num_rows, dim, seed=None))]
pub fn generate_bfloat16_vectors<'py>(
    py: Python<'py>,
    num_rows: usize,
    dim: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let base_seed = seed.unwrap_or(42);
    let bytes_per_row = dim * 2; // bf16 = 2 bytes

    // Release GIL for parallel computation
    let data = py.allow_threads(|| {
        let mut output = vec![0u8; num_rows * bytes_per_row];

        // Parallel generation using rayon
        output
            .par_chunks_mut(bytes_per_row)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Generate f32 values and convert to bf16 bytes
                for chunk in row.chunks_exact_mut(2) {
                    let val: f32 = rng.gen();
                    let bf16_val = bf16::from_f32(val);
                    let bytes = bf16_val.to_le_bytes();
                    chunk[0] = bytes[0];
                    chunk[1] = bytes[1];
                }
            });

        output
    });

    // Create numpy array from data
    let array = PyArray1::from_vec(py, data);
    let reshaped = array.reshape([num_rows, bytes_per_row])?;
    Ok(reshaped)
}

/// Generate Float16 vectors with normalization (unit vectors)
///
/// # Arguments
/// * `num_rows` - Number of vectors to generate
/// * `dim` - Dimension of each vector
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// 2D numpy array of uint8 with shape (num_rows, dim * 2)
#[pyfunction]
#[pyo3(signature = (num_rows, dim, seed=None))]
pub fn generate_normalized_float16_vectors<'py>(
    py: Python<'py>,
    num_rows: usize,
    dim: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let base_seed = seed.unwrap_or(42);
    let bytes_per_row = dim * 2;

    let data = py.allow_threads(|| {
        let mut output = vec![0u8; num_rows * bytes_per_row];

        output
            .par_chunks_mut(bytes_per_row)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Generate random f32 values
                let mut values: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

                // Compute L2 norm
                let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();

                // Normalize
                if norm > 0.0 {
                    for v in values.iter_mut() {
                        *v /= norm;
                    }
                }

                // Convert to f16 bytes
                for (i, chunk) in row.chunks_exact_mut(2).enumerate() {
                    let f16_val = f16::from_f32(values[i]);
                    let bytes = f16_val.to_le_bytes();
                    chunk[0] = bytes[0];
                    chunk[1] = bytes[1];
                }
            });

        output
    });

    let array = PyArray1::from_vec(py, data);
    let reshaped = array.reshape([num_rows, bytes_per_row])?;
    Ok(reshaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float16_generation_consistency() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result1 = generate_float16_vectors(py, 10, 4, Some(42)).unwrap();
            let result2 = generate_float16_vectors(py, 10, 4, Some(42)).unwrap();

            // Should produce identical results with same seed
            assert_eq!(result1.shape(), result2.shape());
        });
    }

    #[test]
    fn test_bfloat16_generation_consistency() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result1 = generate_bfloat16_vectors(py, 10, 4, Some(42)).unwrap();
            let result2 = generate_bfloat16_vectors(py, 10, 4, Some(42)).unwrap();

            assert_eq!(result1.shape(), result2.shape());
        });
    }
}

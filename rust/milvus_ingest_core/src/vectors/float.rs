//! Float vector generation with optional normalization
//!
//! High-performance implementation using SIMD and multi-threading.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Generate float32 vectors
///
/// # Arguments
/// * `num_rows` - Number of vectors to generate
/// * `dim` - Dimension of each vector
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// 2D numpy array of float32 with shape (num_rows, dim)
#[pyfunction]
#[pyo3(signature = (num_rows, dim, seed=None))]
pub fn generate_float_vectors<'py>(
    py: Python<'py>,
    num_rows: usize,
    dim: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let base_seed = seed.unwrap_or(42);

    let data = py.allow_threads(|| {
        let mut output = vec![0.0f32; num_rows * dim];

        output
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                for val in row.iter_mut() {
                    *val = rng.gen();
                }
            });

        output
    });

    let array = PyArray1::from_vec(py, data);
    let reshaped = array.reshape([num_rows, dim])?;
    Ok(reshaped)
}

/// Generate normalized float32 vectors (unit vectors)
///
/// # Arguments
/// * `num_rows` - Number of vectors to generate
/// * `dim` - Dimension of each vector
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// 2D numpy array of float32 with shape (num_rows, dim), L2 normalized
#[pyfunction]
#[pyo3(signature = (num_rows, dim, seed=None))]
pub fn generate_normalized_float_vectors<'py>(
    py: Python<'py>,
    num_rows: usize,
    dim: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let base_seed = seed.unwrap_or(42);

    let data = py.allow_threads(|| {
        let mut output = vec![0.0f32; num_rows * dim];

        output
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Generate random values
                for val in row.iter_mut() {
                    *val = rng.gen();
                }

                // Compute L2 norm using SIMD-friendly loop
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();

                // Normalize
                if norm > 0.0 {
                    let inv_norm = 1.0 / norm;
                    for val in row.iter_mut() {
                        *val *= inv_norm;
                    }
                }
            });

        output
    });

    let array = PyArray1::from_vec(py, data);
    let reshaped = array.reshape([num_rows, dim])?;
    Ok(reshaped)
}

/// Generate float32 vectors with custom range
///
/// # Arguments
/// * `num_rows` - Number of vectors to generate
/// * `dim` - Dimension of each vector
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// 2D numpy array of float32 with shape (num_rows, dim)
#[pyfunction]
#[pyo3(signature = (num_rows, dim, min_val, max_val, seed=None))]
pub fn generate_float_vectors_range<'py>(
    py: Python<'py>,
    num_rows: usize,
    dim: usize,
    min_val: f32,
    max_val: f32,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let base_seed = seed.unwrap_or(42);
    let range = max_val - min_val;

    let data = py.allow_threads(|| {
        let mut output = vec![0.0f32; num_rows * dim];

        output
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                for val in row.iter_mut() {
                    *val = rng.gen::<f32>() * range + min_val;
                }
            });

        output
    });

    let array = PyArray1::from_vec(py, data);
    let reshaped = array.reshape([num_rows, dim])?;
    Ok(reshaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_generation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_float_vectors(py, 100, 128, Some(42)).unwrap();
            assert_eq!(result.shape(), [100, 128]);
        });
    }

    #[test]
    fn test_normalized_float_generation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_normalized_float_vectors(py, 100, 128, Some(42)).unwrap();
            assert_eq!(result.shape(), [100, 128]);
        });
    }
}

//! Array field generation module
//!
//! High-performance generation for Milvus array types:
//! - Array[VarChar]
//! - Array[Int64]
//! - Array[Float]
//!
//! Also includes SIMD-optimized numeric batch generation.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Word pool for generating varchar arrays
const WORD_POOL: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray",
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi",
    "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry", "strawberry",
    "north", "south", "east", "west", "up", "down", "left", "right", "center", "middle", "spring",
    "summer", "autumn", "winter", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
];

/// Generate Array[VarChar] fields
///
/// # Arguments
/// * `num_rows` - Number of arrays to generate
/// * `max_capacity` - Maximum number of elements per array
/// * `max_length` - Maximum length of each string element
/// * `seed` - Optional random seed
///
/// # Returns
/// List of lists of strings
#[pyfunction]
#[pyo3(signature = (num_rows, max_capacity=10, max_length=64, seed=None))]
pub fn generate_varchar_array(
    py: Python<'_>,
    num_rows: usize,
    max_capacity: usize,
    max_length: usize,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<String>>> {
    let base_seed = seed.unwrap_or(42);

    let result = py.allow_threads(|| {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Random array length
                let arr_len = rng.gen_range(1..=max_capacity);

                (0..arr_len)
                    .map(|_| {
                        // Generate string from word pool
                        let word_count = rng.gen_range(1..=3);
                        let words: Vec<&str> = (0..word_count)
                            .map(|_| WORD_POOL[rng.gen_range(0..WORD_POOL.len())])
                            .collect();
                        let text = words.join("_");
                        if text.len() > max_length {
                            text[..max_length].to_string()
                        } else {
                            text
                        }
                    })
                    .collect()
            })
            .collect()
    });

    Ok(result)
}

/// Generate Array[Int64] fields
///
/// # Arguments
/// * `num_rows` - Number of arrays to generate
/// * `max_capacity` - Maximum number of elements per array
/// * `min_val` - Minimum value for integers
/// * `max_val` - Maximum value for integers
/// * `seed` - Optional random seed
///
/// # Returns
/// List of lists of integers
#[pyfunction]
#[pyo3(signature = (num_rows, max_capacity=10, min_val=0, max_val=1000000, seed=None))]
pub fn generate_int_array(
    py: Python<'_>,
    num_rows: usize,
    max_capacity: usize,
    min_val: i64,
    max_val: i64,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<i64>>> {
    let base_seed = seed.unwrap_or(42);

    let result = py.allow_threads(|| {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                let arr_len = rng.gen_range(1..=max_capacity);
                (0..arr_len)
                    .map(|_| rng.gen_range(min_val..=max_val))
                    .collect()
            })
            .collect()
    });

    Ok(result)
}

/// Generate Array[Float] fields
///
/// # Arguments
/// * `num_rows` - Number of arrays to generate
/// * `max_capacity` - Maximum number of elements per array
/// * `min_val` - Minimum value for floats
/// * `max_val` - Maximum value for floats
/// * `seed` - Optional random seed
///
/// # Returns
/// List of lists of floats
#[pyfunction]
#[pyo3(signature = (num_rows, max_capacity=10, min_val=0.0, max_val=1.0, seed=None))]
pub fn generate_float_array(
    py: Python<'_>,
    num_rows: usize,
    max_capacity: usize,
    min_val: f64,
    max_val: f64,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<f64>>> {
    let base_seed = seed.unwrap_or(42);
    let range = max_val - min_val;

    let result = py.allow_threads(|| {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                let arr_len = rng.gen_range(1..=max_capacity);
                (0..arr_len)
                    .map(|_| rng.gen::<f64>() * range + min_val)
                    .collect()
            })
            .collect()
    });

    Ok(result)
}

/// Generate batch of Int64 values (SIMD optimized)
///
/// # Arguments
/// * `num_rows` - Number of values to generate
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
/// * `seed` - Optional random seed
///
/// # Returns
/// 1D numpy array of int64
#[pyfunction]
#[pyo3(signature = (num_rows, min_val=0, max_val=1000000, seed=None))]
pub fn generate_int64_batch<'py>(
    py: Python<'py>,
    num_rows: usize,
    min_val: i64,
    max_val: i64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let base_seed = seed.unwrap_or(42);

    let data = py.allow_threads(|| {
        let mut output = vec![0i64; num_rows];

        // Process in chunks for better parallelization
        const CHUNK_SIZE: usize = 8192;
        output
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = base_seed.wrapping_add(chunk_idx as u64 * CHUNK_SIZE as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(chunk_seed);

                for val in chunk.iter_mut() {
                    *val = rng.gen_range(min_val..=max_val);
                }
            });

        output
    });

    Ok(PyArray1::from_vec(py, data))
}

/// Generate batch of Float64 values (SIMD optimized)
///
/// # Arguments
/// * `num_rows` - Number of values to generate
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
/// * `seed` - Optional random seed
///
/// # Returns
/// 1D numpy array of float64
#[pyfunction]
#[pyo3(signature = (num_rows, min_val=0.0, max_val=1.0, seed=None))]
pub fn generate_float64_batch<'py>(
    py: Python<'py>,
    num_rows: usize,
    min_val: f64,
    max_val: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let base_seed = seed.unwrap_or(42);
    let range = max_val - min_val;

    let data = py.allow_threads(|| {
        let mut output = vec![0.0f64; num_rows];

        const CHUNK_SIZE: usize = 8192;
        output
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = base_seed.wrapping_add(chunk_idx as u64 * CHUNK_SIZE as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(chunk_seed);

                for val in chunk.iter_mut() {
                    *val = rng.gen::<f64>() * range + min_val;
                }
            });

        output
    });

    Ok(PyArray1::from_vec(py, data))
}

/// Generate batch of Float32 values as 2D array
///
/// # Arguments
/// * `num_rows` - Number of rows
/// * `num_cols` - Number of columns
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
/// * `seed` - Optional random seed
///
/// # Returns
/// 2D numpy array of float32
#[pyfunction]
#[pyo3(signature = (num_rows, num_cols, min_val=0.0, max_val=1.0, seed=None))]
pub fn generate_float32_batch_2d<'py>(
    py: Python<'py>,
    num_rows: usize,
    num_cols: usize,
    min_val: f32,
    max_val: f32,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let base_seed = seed.unwrap_or(42);
    let range = max_val - min_val;

    let data = py.allow_threads(|| {
        let mut output = vec![0.0f32; num_rows * num_cols];

        output
            .par_chunks_mut(num_cols)
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
    let reshaped = array.reshape([num_rows, num_cols])?;
    Ok(reshaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varchar_array() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_varchar_array(py, 10, 5, 32, Some(42)).unwrap();
            assert_eq!(result.len(), 10);
            for arr in &result {
                assert!(arr.len() >= 1 && arr.len() <= 5);
            }
        });
    }

    #[test]
    fn test_int_array() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_int_array(py, 10, 5, 0, 100, Some(42)).unwrap();
            assert_eq!(result.len(), 10);
            for arr in &result {
                for val in arr {
                    assert!(*val >= 0 && *val <= 100);
                }
            }
        });
    }

    #[test]
    fn test_int64_batch() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_int64_batch(py, 1000, 0, 100, Some(42)).unwrap();
            assert_eq!(result.len(), 1000);
        });
    }
}

//! Sparse vector generation
//!
//! High-performance implementation for generating sparse float vectors.
//! Expected speedup: 15-20x compared to Python implementation.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Generate sparse float vectors as list of dicts
///
/// # Arguments
/// * `num_rows` - Number of sparse vectors to generate
/// * `max_dim` - Maximum dimension index (exclusive)
/// * `density_min` - Minimum density (fraction of non-zero elements)
/// * `density_max` - Maximum density (fraction of non-zero elements)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// List of Python dicts, each mapping dimension index (int) to value (float)
#[pyfunction]
#[pyo3(signature = (num_rows, max_dim, density_min=0.01, density_max=0.1, seed=None))]
pub fn generate_sparse_vectors(
    py: Python<'_>,
    num_rows: usize,
    max_dim: usize,
    density_min: f64,
    density_max: f64,
    seed: Option<u64>,
) -> PyResult<Vec<Py<PyDict>>> {
    let base_seed = seed.unwrap_or(42);

    // Generate sparse data in parallel (release GIL)
    let sparse_data: Vec<BTreeMap<usize, f32>> = py.allow_threads(|| {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Random density between min and max
                let density = rng.gen::<f64>() * (density_max - density_min) + density_min;
                let nnz = ((max_dim as f64) * density).ceil() as usize;
                let nnz = nnz.max(1).min(max_dim); // At least 1, at most max_dim

                // Generate random unique indices
                let mut indices: Vec<usize> = (0..max_dim).collect();
                indices.shuffle(&mut rng);
                indices.truncate(nnz);
                indices.sort_unstable();

                // Generate values and create map
                let mut sparse_vec = BTreeMap::new();
                for idx in indices {
                    let value: f32 = rng.gen();
                    sparse_vec.insert(idx, value);
                }

                sparse_vec
            })
            .collect()
    });

    // Convert to Python dicts
    let result: PyResult<Vec<Py<PyDict>>> = sparse_data
        .into_iter()
        .map(|sparse_vec| {
            let dict = PyDict::new(py);
            for (k, v) in sparse_vec {
                dict.set_item(k, v)?;
            }
            Ok(dict.into())
        })
        .collect();

    result
}

/// Generate sparse float vectors with specific non-zero count
///
/// # Arguments
/// * `num_rows` - Number of sparse vectors to generate
/// * `max_dim` - Maximum dimension index (exclusive)
/// * `nnz_min` - Minimum number of non-zero elements
/// * `nnz_max` - Maximum number of non-zero elements
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// List of Python dicts, each mapping dimension index (int) to value (float)
#[pyfunction]
#[pyo3(signature = (num_rows, max_dim, nnz_min, nnz_max, seed=None))]
pub fn generate_sparse_vectors_nnz(
    py: Python<'_>,
    num_rows: usize,
    max_dim: usize,
    nnz_min: usize,
    nnz_max: usize,
    seed: Option<u64>,
) -> PyResult<Vec<Py<PyDict>>> {
    let base_seed = seed.unwrap_or(42);

    let sparse_data: Vec<BTreeMap<usize, f32>> = py.allow_threads(|| {
        (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row_seed = base_seed.wrapping_add(row_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

                // Random nnz between min and max
                let nnz = rng.gen_range(nnz_min..=nnz_max);
                let nnz = nnz.min(max_dim);

                // Generate random unique indices
                let mut indices: Vec<usize> = (0..max_dim).collect();
                indices.shuffle(&mut rng);
                indices.truncate(nnz);
                indices.sort_unstable();

                // Generate values
                let mut sparse_vec = BTreeMap::new();
                for idx in indices {
                    let value: f32 = rng.gen();
                    sparse_vec.insert(idx, value);
                }

                sparse_vec
            })
            .collect()
    });

    let result: PyResult<Vec<Py<PyDict>>> = sparse_data
        .into_iter()
        .map(|sparse_vec| {
            let dict = PyDict::new(py);
            for (k, v) in sparse_vec {
                dict.set_item(k, v)?;
            }
            Ok(dict.into())
        })
        .collect();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_generation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = generate_sparse_vectors(py, 10, 1000, 0.01, 0.1, Some(42)).unwrap();
            assert_eq!(result.len(), 10);
        });
    }
}

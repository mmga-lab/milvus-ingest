//! Faker data generation module - ported from synth-gen concepts
//!
//! This module provides high-performance fake data generation using the `fake` crate,
//! inspired by the synth project's generator patterns.

use chrono::{FixedOffset, TimeZone};
use fake::faker::address::en as address_en;
use fake::faker::company::en as company_en;
use fake::faker::internet::en as internet_en;
use fake::faker::lorem::en as lorem_en;
use fake::faker::name::en as name_en;
use fake::faker::phone_number::en as phone_en;
use fake::Fake;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Supported faker types
#[derive(Debug, Clone, Copy)]
pub enum FakerType {
    Name,
    FirstName,
    LastName,
    Email,
    Username,
    Phone,
    Address,
    City,
    Country,
    Company,
    CompanyName,
    Industry,
    JobTitle,
    Url,
    DomainName,
    Ipv4,
    UserAgent,
    Uuid,
}

impl FakerType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "name" | "fullname" | "full_name" => Some(FakerType::Name),
            "firstname" | "first_name" => Some(FakerType::FirstName),
            "lastname" | "last_name" => Some(FakerType::LastName),
            "email" => Some(FakerType::Email),
            "username" | "user_name" => Some(FakerType::Username),
            "phone" | "phonenumber" | "phone_number" => Some(FakerType::Phone),
            "address" | "streetaddress" | "street_address" => Some(FakerType::Address),
            "city" | "cityname" | "city_name" => Some(FakerType::City),
            "country" | "countryname" | "country_name" => Some(FakerType::Country),
            "company" | "companyname" | "company_name" => Some(FakerType::CompanyName),
            "industry" => Some(FakerType::Industry),
            "jobtitle" | "job_title" | "job" => Some(FakerType::JobTitle),
            "url" | "website" => Some(FakerType::Url),
            "domain" | "domainname" | "domain_name" => Some(FakerType::DomainName),
            "ip" | "ipv4" | "ip_address" => Some(FakerType::Ipv4),
            "useragent" | "user_agent" => Some(FakerType::UserAgent),
            "uuid" | "guid" => Some(FakerType::Uuid),
            _ => None,
        }
    }
}

/// Generate a single fake value based on type
fn generate_single_faker_value<R: rand::Rng>(rng: &mut R, faker_type: FakerType) -> String {
    match faker_type {
        FakerType::Name => name_en::Name().fake_with_rng(rng),
        FakerType::FirstName => name_en::FirstName().fake_with_rng(rng),
        FakerType::LastName => name_en::LastName().fake_with_rng(rng),
        FakerType::Email => internet_en::SafeEmail().fake_with_rng(rng),
        FakerType::Username => internet_en::Username().fake_with_rng(rng),
        FakerType::Phone => phone_en::PhoneNumber().fake_with_rng(rng),
        // SecondaryAddress provides apartment/suite style addresses
        FakerType::Address => address_en::SecondaryAddress().fake_with_rng(rng),
        FakerType::City => address_en::CityName().fake_with_rng(rng),
        FakerType::Country => address_en::CountryName().fake_with_rng(rng),
        FakerType::CompanyName => company_en::CompanyName().fake_with_rng(rng),
        FakerType::Company => company_en::CompanyName().fake_with_rng(rng),
        FakerType::Industry => company_en::Industry().fake_with_rng(rng),
        FakerType::JobTitle => company_en::Profession().fake_with_rng(rng),
        FakerType::Url => format!(
            "https://{}/{}",
            internet_en::DomainSuffix().fake_with_rng::<String, _>(rng),
            internet_en::Username().fake_with_rng::<String, _>(rng)
        ),
        FakerType::DomainName => internet_en::DomainSuffix().fake_with_rng(rng),
        FakerType::Ipv4 => internet_en::IPv4().fake_with_rng(rng),
        FakerType::UserAgent => internet_en::UserAgent().fake_with_rng(rng),
        FakerType::Uuid => uuid::Uuid::new_v4().to_string(),
    }
}

/// Generate fake strings using the fake crate.
///
/// Args:
///     count: Number of strings to generate
///     faker_type: Type of fake data ("name", "email", "address", "phone", "company", etc.)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of generated fake data
#[pyfunction]
#[pyo3(signature = (count, faker_type, seed=None))]
pub fn generate_faker_strings(
    py: Python<'_>,
    count: usize,
    faker_type: &str,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    let ftype = FakerType::from_str(faker_type).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown faker type: '{}'. Supported types: name, firstname, lastname, email, \
             username, phone, address, city, country, company, industry, jobtitle, url, \
             domain, ipv4, useragent, uuid",
            faker_type
        ))
    })?;

    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        // Parallel generation with per-thread RNG
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                generate_single_faker_value(&mut rng, ftype)
            })
            .collect()
    });

    Ok(results)
}

/// Generate categorical values from a weighted distribution.
///
/// This is inspired by synth's categorical generator, which allows specifying
/// categories with associated weights/frequencies.
///
/// Args:
///     count: Number of values to generate
///     categories: List of category strings
///     weights: List of weights (will be normalized to probabilities)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of selected categories
#[pyfunction]
#[pyo3(signature = (count, categories, weights, seed=None))]
pub fn generate_categorical(
    py: Python<'_>,
    count: usize,
    categories: Vec<String>,
    weights: Vec<f64>,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    if categories.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Categories cannot be empty",
        ));
    }

    if categories.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Categories and weights must have the same length ({} vs {})",
            categories.len(),
            weights.len()
        )));
    }

    // Normalize weights to cumulative distribution
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Total weight must be positive",
        ));
    }

    let mut cumulative: Vec<f64> = Vec::with_capacity(weights.len());
    let mut sum = 0.0;
    for w in &weights {
        sum += w / total_weight;
        cumulative.push(sum);
    }

    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let r: f64 = rand::Rng::gen(&mut rng);

                // Binary search for the category
                let idx = cumulative
                    .iter()
                    .position(|&c| r < c)
                    .unwrap_or(categories.len() - 1);

                categories[idx].clone()
            })
            .collect()
    });

    Ok(results)
}

/// Generate range numbers with optional step.
///
/// Inspired by synth's number range generator.
///
/// Args:
///     count: Number of values to generate
///     low: Minimum value (inclusive)
///     high: Maximum value (exclusive)
///     step: Optional step size (if provided, values will be multiples of step)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<f64> of generated numbers
#[pyfunction]
#[pyo3(signature = (count, low, high, step=None, seed=None))]
pub fn generate_range_numbers(
    py: Python<'_>,
    count: usize,
    low: f64,
    high: f64,
    step: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Vec<f64>> {
    if low >= high {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "low must be less than high",
        ));
    }

    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let raw: f64 = rand::Rng::gen_range(&mut rng, low..high);

                // Apply step if specified
                if let Some(s) = step {
                    if s > 0.0 {
                        let steps = ((raw - low) / s).floor();
                        low + steps * s
                    } else {
                        raw
                    }
                } else {
                    raw
                }
            })
            .collect()
    });

    Ok(results)
}

/// Generate integers in a range with optional step.
///
/// Args:
///     count: Number of values to generate
///     low: Minimum value (inclusive)
///     high: Maximum value (exclusive)
///     step: Optional step size
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<i64> of generated integers
#[pyfunction]
#[pyo3(signature = (count, low, high, step=None, seed=None))]
pub fn generate_range_integers(
    py: Python<'_>,
    count: usize,
    low: i64,
    high: i64,
    step: Option<i64>,
    seed: Option<u64>,
) -> PyResult<Vec<i64>> {
    if low >= high {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "low must be less than high",
        ));
    }

    let base_seed = seed.unwrap_or(42);
    let step_val = step.unwrap_or(1).max(1);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let num_steps = ((high - low) / step_val) as u64;
                let step_idx = rand::Rng::gen_range(&mut rng, 0..num_steps.max(1));
                low + (step_idx as i64) * step_val
            })
            .collect()
    });

    Ok(results)
}

/// Generate boolean values with specified true probability.
///
/// Inspired by synth's frequency-based boolean generator.
///
/// Args:
///     count: Number of values to generate
///     true_probability: Probability of generating true (0.0 to 1.0)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<bool> of generated booleans
#[pyfunction]
#[pyo3(signature = (count, true_probability=0.5, seed=None))]
pub fn generate_frequency_bool(
    py: Python<'_>,
    count: usize,
    true_probability: f64,
    seed: Option<u64>,
) -> PyResult<Vec<bool>> {
    if !(0.0..=1.0).contains(&true_probability) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "true_probability must be between 0.0 and 1.0",
        ));
    }

    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let r: f64 = rand::Rng::gen(&mut rng);
                r < true_probability
            })
            .collect()
    });

    Ok(results)
}

/// Generate date strings in ISO format (YYYY-MM-DD).
///
/// Args:
///     count: Number of dates to generate
///     start_year: Start year (inclusive)
///     end_year: End year (inclusive)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of date strings
#[pyfunction]
#[pyo3(signature = (count, start_year=2020, end_year=2024, seed=None))]
pub fn generate_date_strings(
    py: Python<'_>,
    count: usize,
    start_year: i32,
    end_year: i32,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    if start_year > end_year {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_year must be <= end_year",
        ));
    }

    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let year = rand::Rng::gen_range(&mut rng, start_year..=end_year);
                let month = rand::Rng::gen_range(&mut rng, 1..=12);
                let max_day = match month {
                    2 => {
                        if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                            29
                        } else {
                            28
                        }
                    }
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                let day = rand::Rng::gen_range(&mut rng, 1..=max_day);
                format!("{:04}-{:02}-{:02}", year, month, day)
            })
            .collect()
    });

    Ok(results)
}

/// Generate datetime strings in ISO format (YYYY-MM-DDTHH:MM:SS).
///
/// Args:
///     count: Number of datetimes to generate
///     start_year: Start year (inclusive)
///     end_year: End year (inclusive)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of datetime strings
#[pyfunction]
#[pyo3(signature = (count, start_year=2020, end_year=2024, seed=None))]
pub fn generate_datetime_strings(
    py: Python<'_>,
    count: usize,
    start_year: i32,
    end_year: i32,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    if start_year > end_year {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_year must be <= end_year",
        ));
    }

    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let year = rand::Rng::gen_range(&mut rng, start_year..=end_year);
                let month = rand::Rng::gen_range(&mut rng, 1..=12);
                let max_day = match month {
                    2 => {
                        if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                            29
                        } else {
                            28
                        }
                    }
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                let day = rand::Rng::gen_range(&mut rng, 1..=max_day);
                let hour = rand::Rng::gen_range(&mut rng, 0..24);
                let minute = rand::Rng::gen_range(&mut rng, 0..60);
                let second = rand::Rng::gen_range(&mut rng, 0..60);
                format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                    year, month, day, hour, minute, second
                )
            })
            .collect()
    });

    Ok(results)
}

/// Generate TIMESTAMPTZ strings in ISO 8601 format with timezone.
///
/// Output format: YYYY-MM-DDTHH:MM:SS+HH:MM (e.g., "2025-05-01T23:59:59+08:00")
/// Compatible with Milvus TIMESTAMPTZ field type (Milvus 2.6.6+).
///
/// Args:
///     count: Number of timestamps to generate
///     start_year: Start year (inclusive)
///     end_year: End year (inclusive)
///     timezone_offset_hours: Fixed timezone offset in hours (e.g., 8 for +08:00, -5 for -05:00).
///                            If None, generates random offsets from common timezones.
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of TIMESTAMPTZ strings
#[pyfunction]
#[pyo3(signature = (count, start_year=2020, end_year=2024, timezone_offset_hours=None, seed=None))]
pub fn generate_timestamptz_strings(
    py: Python<'_>,
    count: usize,
    start_year: i32,
    end_year: i32,
    timezone_offset_hours: Option<i32>,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    if start_year > end_year {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_year must be <= end_year",
        ));
    }

    let base_seed = seed.unwrap_or(42);

    // Common timezone offsets in hours (supports half-hour offsets via special handling)
    let tz_offset_hours: Vec<i32> = vec![
        -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, 0, 1, 2, 3, 5, 8, 9, 10, 12,
    ];

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));

                // Generate date/time components
                let year = rand::Rng::gen_range(&mut rng, start_year..=end_year);
                let month = rand::Rng::gen_range(&mut rng, 1u32..=12);
                let max_day = match month {
                    2 => {
                        if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                            29
                        } else {
                            28
                        }
                    }
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                let day = rand::Rng::gen_range(&mut rng, 1u32..=max_day);
                let hour = rand::Rng::gen_range(&mut rng, 0u32..24);
                let minute = rand::Rng::gen_range(&mut rng, 0u32..60);
                let second = rand::Rng::gen_range(&mut rng, 0u32..60);

                // Get timezone offset
                let offset_hours = match timezone_offset_hours {
                    Some(h) => h,
                    None => {
                        let idx = rand::Rng::gen_range(&mut rng, 0..tz_offset_hours.len());
                        tz_offset_hours[idx]
                    }
                };

                // Create FixedOffset and DateTime using chrono
                let offset_secs = offset_hours * 3600;
                let tz = FixedOffset::east_opt(offset_secs).unwrap_or(FixedOffset::east_opt(0).unwrap());
                let dt = tz
                    .with_ymd_and_hms(year, month, day, hour, minute, second)
                    .single()
                    .unwrap_or_else(|| tz.with_ymd_and_hms(year, month, 1, 0, 0, 0).unwrap());

                // Format as ISO 8601 with timezone (RFC 3339)
                dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
            })
            .collect()
    });

    Ok(results)
}

// =============================================================================
// Geometry (WKT) Generation Functions
// =============================================================================

/// Generate WKT POINT strings.
///
/// Output format: POINT (longitude latitude)
/// Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).
///
/// Args:
///     count: Number of points to generate
///     lon_min: Minimum longitude (-180 to 180)
///     lon_max: Maximum longitude
///     lat_min: Minimum latitude (-90 to 90)
///     lat_max: Maximum latitude
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of WKT POINT strings
#[pyfunction]
#[pyo3(signature = (count, lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0, seed=None))]
pub fn generate_wkt_points(
    py: Python<'_>,
    count: usize,
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
    seed: Option<u64>,
) -> Vec<String> {
    let base_seed = seed.unwrap_or(42);

    py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let lon = rand::Rng::gen_range(&mut rng, lon_min..lon_max);
                let lat = rand::Rng::gen_range(&mut rng, lat_min..lat_max);
                format!("POINT ({:.6} {:.6})", lon, lat)
            })
            .collect()
    })
}

/// Generate WKT LINESTRING strings.
///
/// Output format: LINESTRING (x1 y1, x2 y2, ...)
/// Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).
///
/// Args:
///     count: Number of linestrings to generate
///     points_min: Minimum number of points per linestring
///     points_max: Maximum number of points per linestring
///     lon_min: Minimum longitude
///     lon_max: Maximum longitude
///     lat_min: Minimum latitude
///     lat_max: Maximum latitude
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of WKT LINESTRING strings
#[pyfunction]
#[pyo3(signature = (count, points_min=2, points_max=5, lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0, seed=None))]
pub fn generate_wkt_linestrings(
    py: Python<'_>,
    count: usize,
    points_min: usize,
    points_max: usize,
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
    seed: Option<u64>,
) -> Vec<String> {
    let base_seed = seed.unwrap_or(42);
    let points_min = points_min.max(2); // At least 2 points

    py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let num_points = rand::Rng::gen_range(&mut rng, points_min..=points_max);

                let points: Vec<String> = (0..num_points)
                    .map(|_| {
                        let lon = rand::Rng::gen_range(&mut rng, lon_min..lon_max);
                        let lat = rand::Rng::gen_range(&mut rng, lat_min..lat_max);
                        format!("{:.6} {:.6}", lon, lat)
                    })
                    .collect();

                format!("LINESTRING ({})", points.join(", "))
            })
            .collect()
    })
}

/// Generate WKT POLYGON strings.
///
/// Output format: POLYGON ((x1 y1, x2 y2, ..., x1 y1))
/// Generates simple convex polygons (no holes).
/// Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).
///
/// Args:
///     count: Number of polygons to generate
///     vertices_min: Minimum number of vertices (3+)
///     vertices_max: Maximum number of vertices
///     center_lon: Center longitude for polygon generation
///     center_lat: Center latitude for polygon generation
///     radius: Approximate radius in degrees
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of WKT POLYGON strings
#[pyfunction]
#[pyo3(signature = (count, vertices_min=4, vertices_max=8, center_lon=0.0, center_lat=0.0, radius=1.0, seed=None))]
pub fn generate_wkt_polygons(
    py: Python<'_>,
    count: usize,
    vertices_min: usize,
    vertices_max: usize,
    center_lon: f64,
    center_lat: f64,
    radius: f64,
    seed: Option<u64>,
) -> Vec<String> {
    let base_seed = seed.unwrap_or(42);
    let vertices_min = vertices_min.max(3); // At least 3 vertices

    py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                let num_vertices = rand::Rng::gen_range(&mut rng, vertices_min..=vertices_max);

                // Generate random center offset
                let cx = center_lon + rand::Rng::gen_range(&mut rng, -radius..radius);
                let cy = center_lat + rand::Rng::gen_range(&mut rng, -radius..radius);

                // Generate vertices in a circular pattern (convex polygon)
                let mut angles: Vec<f64> = (0..num_vertices)
                    .map(|_| rand::Rng::gen_range(&mut rng, 0.0..std::f64::consts::TAU))
                    .collect();
                angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mut points: Vec<String> = angles
                    .iter()
                    .map(|&angle| {
                        let r = radius * (0.5 + rand::Rng::gen_range(&mut rng, 0.0..0.5));
                        let lon = cx + r * angle.cos();
                        let lat = cy + r * angle.sin();
                        format!("{:.6} {:.6}", lon, lat)
                    })
                    .collect();

                // Close the ring
                if let Some(first) = points.first().cloned() {
                    points.push(first);
                }

                format!("POLYGON (({}))", points.join(", "))
            })
            .collect()
    })
}

/// Generate WKT geometry strings of mixed types.
///
/// Generates a mix of POINT, LINESTRING, and POLYGON geometries.
/// Compatible with Milvus GEOMETRY field type (Milvus 2.6.4+).
///
/// Args:
///     count: Number of geometries to generate
///     geometry_type: Type of geometry ("point", "linestring", "polygon", "mixed")
///     lon_min: Minimum longitude
///     lon_max: Maximum longitude
///     lat_min: Minimum latitude
///     lat_max: Maximum latitude
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of WKT geometry strings
#[pyfunction]
#[pyo3(signature = (count, geometry_type="point", lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0, seed=None))]
pub fn generate_wkt_geometries(
    py: Python<'_>,
    count: usize,
    geometry_type: &str,
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
    seed: Option<u64>,
) -> Vec<String> {
    let base_seed = seed.unwrap_or(42);
    let geom_type = geometry_type.to_lowercase();

    py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));

                let selected_type = if geom_type == "mixed" {
                    match rand::Rng::gen_range(&mut rng, 0..3) {
                        0 => "point",
                        1 => "linestring",
                        _ => "polygon",
                    }
                } else {
                    geom_type.as_str()
                };

                match selected_type {
                    "point" => {
                        let lon = rand::Rng::gen_range(&mut rng, lon_min..lon_max);
                        let lat = rand::Rng::gen_range(&mut rng, lat_min..lat_max);
                        format!("POINT ({:.6} {:.6})", lon, lat)
                    }
                    "linestring" => {
                        let num_points = rand::Rng::gen_range(&mut rng, 2..=5);
                        let points: Vec<String> = (0..num_points)
                            .map(|_| {
                                let lon = rand::Rng::gen_range(&mut rng, lon_min..lon_max);
                                let lat = rand::Rng::gen_range(&mut rng, lat_min..lat_max);
                                format!("{:.6} {:.6}", lon, lat)
                            })
                            .collect();
                        format!("LINESTRING ({})", points.join(", "))
                    }
                    "polygon" | _ => {
                        let num_vertices = rand::Rng::gen_range(&mut rng, 4..=6);
                        let cx = rand::Rng::gen_range(&mut rng, lon_min..lon_max);
                        let cy = rand::Rng::gen_range(&mut rng, lat_min..lat_max);
                        let radius = (lon_max - lon_min).min(lat_max - lat_min) * 0.1;

                        let mut angles: Vec<f64> = (0..num_vertices)
                            .map(|_| rand::Rng::gen_range(&mut rng, 0.0..std::f64::consts::TAU))
                            .collect();
                        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let mut points: Vec<String> = angles
                            .iter()
                            .map(|&angle| {
                                let r = radius * (0.5 + rand::Rng::gen_range(&mut rng, 0.0..0.5));
                                let lon = cx + r * angle.cos();
                                let lat = cy + r * angle.sin();
                                format!("{:.6} {:.6}", lon, lat)
                            })
                            .collect();

                        if let Some(first) = points.first().cloned() {
                            points.push(first);
                        }
                        format!("POLYGON (({}))", points.join(", "))
                    }
                }
            })
            .collect()
    })
}

/// Generate unique IDs with prefix.
///
/// Args:
///     count: Number of IDs to generate
///     prefix: Prefix for the ID
///     start_index: Starting index
///
/// Returns:
///     Vec<String> of generated IDs
#[pyfunction]
#[pyo3(signature = (count, prefix="id", start_index=0))]
pub fn generate_sequential_ids(
    py: Python<'_>,
    count: usize,
    prefix: &str,
    start_index: usize,
) -> Vec<String> {
    py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| format!("{}_{}", prefix, start_index + i))
            .collect()
    })
}

// =============================================================================
// Lorem Text Generation Functions
// =============================================================================

/// Generate Lorem Ipsum words.
///
/// Args:
///     count: Number of strings to generate
///     words_per_item: Number of words per item (default 1)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of lorem words
#[pyfunction]
#[pyo3(signature = (count, words_per_item=1, seed=None))]
pub fn generate_lorem_words(
    py: Python<'_>,
    count: usize,
    words_per_item: usize,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    let base_seed = seed.unwrap_or(42);

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                if words_per_item == 1 {
                    lorem_en::Word().fake_with_rng(&mut rng)
                } else {
                    let words: Vec<String> = lorem_en::Words(words_per_item..words_per_item + 1)
                        .fake_with_rng(&mut rng);
                    words.join(" ")
                }
            })
            .collect()
    });

    Ok(results)
}

/// Generate Lorem Ipsum sentences.
///
/// Args:
///     count: Number of sentences to generate
///     word_count_min: Minimum words per sentence (default 4)
///     word_count_max: Maximum words per sentence (default 10)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of lorem sentences
#[pyfunction]
#[pyo3(signature = (count, word_count_min=4, word_count_max=10, seed=None))]
pub fn generate_lorem_sentences(
    py: Python<'_>,
    count: usize,
    word_count_min: usize,
    word_count_max: usize,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    let base_seed = seed.unwrap_or(42);
    let word_range = word_count_min..word_count_max + 1;

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                lorem_en::Sentence(word_range.clone()).fake_with_rng(&mut rng)
            })
            .collect()
    });

    Ok(results)
}

/// Generate Lorem Ipsum paragraphs.
///
/// Args:
///     count: Number of paragraphs to generate
///     sentence_count_min: Minimum sentences per paragraph (default 3)
///     sentence_count_max: Maximum sentences per paragraph (default 6)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of lorem paragraphs
#[pyfunction]
#[pyo3(signature = (count, sentence_count_min=3, sentence_count_max=6, seed=None))]
pub fn generate_lorem_paragraphs(
    py: Python<'_>,
    count: usize,
    sentence_count_min: usize,
    sentence_count_max: usize,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    let base_seed = seed.unwrap_or(42);
    let sentence_range = sentence_count_min..sentence_count_max + 1;

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                lorem_en::Paragraph(sentence_range.clone()).fake_with_rng(&mut rng)
            })
            .collect()
    });

    Ok(results)
}

/// Generate Lorem Ipsum text with configurable length.
///
/// This is a flexible text generator that produces realistic-looking text.
///
/// Args:
///     count: Number of text items to generate
///     text_type: Type of text - "word", "sentence", "paragraph", "text" (default: "sentence")
///     min_units: Minimum units (words/sentences depending on type, default 4)
///     max_units: Maximum units (default 10)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Vec<String> of generated text
#[pyfunction]
#[pyo3(signature = (count, text_type="sentence", min_units=4, max_units=10, seed=None))]
pub fn generate_lorem_text(
    py: Python<'_>,
    count: usize,
    text_type: &str,
    min_units: usize,
    max_units: usize,
    seed: Option<u64>,
) -> PyResult<Vec<String>> {
    let base_seed = seed.unwrap_or(42);
    let text_type = text_type.to_lowercase();
    let unit_range = min_units..max_units + 1;

    let results = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(i as u64));
                match text_type.as_str() {
                    "word" | "words" => {
                        let words: Vec<String> =
                            lorem_en::Words(unit_range.clone()).fake_with_rng(&mut rng);
                        words.join(" ")
                    }
                    "paragraph" | "paragraphs" => {
                        lorem_en::Paragraph(unit_range.clone()).fake_with_rng(&mut rng)
                    }
                    "text" => {
                        // Generate multiple paragraphs for longer text
                        let paragraphs: Vec<String> =
                            lorem_en::Paragraphs(unit_range.clone()).fake_with_rng(&mut rng);
                        paragraphs.join("\n\n")
                    }
                    _ => {
                        // Default: sentence
                        lorem_en::Sentence(unit_range.clone()).fake_with_rng(&mut rng)
                    }
                }
            })
            .collect()
    });

    Ok(results)
}

/// List all available faker types.
#[pyfunction]
pub fn list_faker_types() -> Vec<&'static str> {
    vec![
        // Person
        "name",
        "firstname",
        "lastname",
        // Internet
        "email",
        "username",
        "phone",
        "url",
        "domain",
        "ipv4",
        "useragent",
        // Location
        "address",
        "city",
        "country",
        // Business
        "company",
        "industry",
        "jobtitle",
        // ID
        "uuid",
        // Lorem (use generate_lorem_* functions for more control)
        "word",
        "sentence",
        "paragraph",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faker_type_parsing() {
        assert!(FakerType::from_str("name").is_some());
        assert!(FakerType::from_str("NAME").is_some());
        assert!(FakerType::from_str("email").is_some());
        assert!(FakerType::from_str("unknown_type").is_none());
    }

    #[test]
    fn test_generate_single_faker_value() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let name = generate_single_faker_value(&mut rng, FakerType::Name);
        assert!(!name.is_empty());

        let email = generate_single_faker_value(&mut rng, FakerType::Email);
        assert!(email.contains('@'));
    }

    #[test]
    fn test_categorical_weights() {
        // Test with extreme weights
        let categories = vec!["A".to_string(), "B".to_string()];
        let weights = vec![100.0, 0.0];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut cumulative: Vec<f64> = Vec::new();
        let total: f64 = weights.iter().sum();
        let mut sum = 0.0;
        for w in &weights {
            sum += w / total;
            cumulative.push(sum);
        }

        // With weight 100:0, all results should be "A"
        for _ in 0..100 {
            let r: f64 = rand::Rng::gen(&mut rng);
            let idx = cumulative
                .iter()
                .position(|&c| r < c)
                .unwrap_or(categories.len() - 1);
            assert_eq!(categories[idx], "A");
        }
    }

    #[test]
    fn test_range_with_step() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let low = 0.0;
        let high = 100.0;
        let step = 10.0;

        for _ in 0..100 {
            let raw: f64 = rand::Rng::gen_range(&mut rng, low..high);
            let stepped = if step > 0.0 {
                let steps = ((raw - low) / step).floor();
                low + steps * step
            } else {
                raw
            };

            // Verify it's a multiple of step
            assert!((stepped % step).abs() < 1e-10);
        }
    }
}

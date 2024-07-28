use std::{collections::HashSet};

use pyo3::prelude::*;
use ndarray::Array2;

/// Compute the Jaccard similarity between two sets.
fn tag_similarity_kernel(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    intersection / union
}

/// Compute the Jaccard similarity matrix between two vecs of sets.
fn tag_similarity_matrix(a: &Vec<HashSet<String>>, b: &Vec<HashSet<String>>) -> Array2<f64> {
    let mut matrix = Array2::zeros((a.len(), b.len()));
    for (i, a_tag) in a.iter().enumerate() {
        for (j, b_tag) in b.iter().enumerate() {
            matrix[[i, j]] = tag_similarity_kernel(a_tag, b_tag);
        }
    }
    matrix
}

/// Compute the sum of the diagonal of a matrix.
fn diag_sum(matrix: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..matrix.shape()[0] {
        sum += matrix[[i, i]];
    }
    sum
}

/// Compute mmd or kernel distance between two arrays of sets of tags
#[pyfunction]
fn kernel_tag_distance(a: Vec<HashSet<String>>, b: Vec<HashSet<String>>) -> PyResult<f64> {
    let m = a.len() as f64;
    let n = b.len() as f64;
    let kxx = tag_similarity_matrix(&a, &a);
    let kyy = tag_similarity_matrix(&b, &b);
    let kxy = tag_similarity_matrix(&a, &b);
    let kxx_sum = kxx.sum() - diag_sum(&kxx);
    let kyy_sum = kyy.sum() - diag_sum(&kyy);
    let kxy_sum = kxy.sum();
    let term1 = kxx_sum / m / (m-1.0);
    let term2 = kyy_sum / n / (n-1.0);
    let term3 = kxy_sum * 2.0 / m / n;
    Ok(term1 + term2 - term3)
}

/// A Python module implemented in Rust.
#[pymodule]
fn ktdo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kernel_tag_distance, m)?)?;
    Ok(())
}

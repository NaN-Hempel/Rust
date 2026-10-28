// The linear algebra comes from nalgebra; this file fixes the types the network works with.
// A column vector is a matrix with one column. Data is stored one sample per column, so a layer
// handles a whole batch in one matrix product, Z = W·X, instead of z = W·x for every sample.

use nalgebra::DMatrix;
use rand::RngExt;

pub type Matrix = DMatrix<f64>;

// A matrix with every element drawn uniformly from [-limit, limit).
pub fn random(rows: usize, cols: usize, limit: f64, rng: &mut impl RngExt) -> Matrix {
    Matrix::from_fn(rows, cols, |_, _| rng.random_range(-limit..limit))
}

// All ones. A row of ones, 1ᵀ, copies a column to every column of a batch (b·1ᵀ), and a column
// of ones, 1, sums a matrix's columns (A·1).
pub fn ones(rows: usize, cols: usize) -> Matrix {
    Matrix::from_element(rows, cols, 1.0)
}

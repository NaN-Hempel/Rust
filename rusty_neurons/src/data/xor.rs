// XOR: the output is 1 when exactly one of the two inputs is 1. No straight line separates the
// 1s from the 0s, so a network can only learn it with a hidden layer.

use nalgebra::dmatrix;

use crate::code::math::Matrix;

// The four cases, one per column: inputs is 2×4 and targets is 1×4.
pub fn xor() -> (Matrix, Matrix) {
    let inputs = dmatrix![0.0, 0.0, 1.0, 1.0;
                          0.0, 1.0, 0.0, 1.0];
    let targets = dmatrix![0.0, 1.0, 1.0, 0.0];
    (inputs, targets)
}

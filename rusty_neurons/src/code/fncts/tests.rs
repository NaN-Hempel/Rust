// Tests for fncts.rs.

use approx::assert_abs_diff_eq;
use nalgebra::dmatrix;

use super::*;

#[test]
fn derivatives_match_finite_differences() {
    // Points away from 0, where ReLU has its kink.
    let z = dmatrix![-2.0, -0.5, 0.3, 1.7];
    let h = 1e-6;
    for activation in [Activation::Identity, Activation::Sigmoid, Activation::Tanh, Activation::Relu] {
        let numeric = (activation.apply(&z.add_scalar(h)) - activation.apply(&z.add_scalar(-h))) / (2.0 * h);
        assert_abs_diff_eq!(numeric, activation.derivative(&z), epsilon = 1e-8);
    }
}

#[test]
fn mse_of_a_perfect_prediction_is_zero() {
    let y = dmatrix![0.0, 1.0, 1.0, 0.0];
    assert_eq!(mse(&y, &y), 0.0);
    // Each of the 4 predictions off by 0.5: (4 · 0.25) / (2 · 4).
    assert_eq!(mse(&y.add_scalar(0.5), &y), 0.125);
}

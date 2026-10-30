// Tests for network.rs.

use approx::assert_abs_diff_eq;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::*;
use crate::data::xor::xor;

// Layer l's weights, or its bias.
fn parameter(network: &mut Network, l: usize, bias: bool) -> &mut Matrix {
    let layer = &mut network.layers[l];
    if bias { &mut layer.bias } else { &mut layer.weights }
}

// Checks backpropagation against brute force: nudge each weight and bias by ±h, and see how
// the loss changes.
#[test]
fn gradients_match_finite_differences() {
    let mut rng = StdRng::seed_from_u64(7);
    let input = random(3, 5, 1.0, &mut rng);
    let target = random(2, 5, 1.0, &mut rng);
    let activations = [Activation::Tanh, Activation::Relu, Activation::Sigmoid];
    let mut network = Network::new(&[3, 4, 4, 2], &activations, &mut rng);
    // Non-zero bias, so its gradients get tested properly too.
    for layer in &mut network.layers {
        layer.bias = random(layer.bias.nrows(), 1, 0.5, &mut rng);
    }

    let (_, gradients) = network.gradients(&input, &target);
    let h = 1e-6;
    for (l, gradient) in gradients.iter().enumerate() {
        for (bias, analytic) in [(false, &gradient.weights), (true, &gradient.bias)] {
            let numeric = Matrix::from_fn(analytic.nrows(), analytic.ncols(), |i, j| {
                let original = parameter(&mut network, l, bias)[(i, j)];
                parameter(&mut network, l, bias)[(i, j)] = original + h;
                let above = network.loss(&input, &target);
                parameter(&mut network, l, bias)[(i, j)] = original - h;
                let below = network.loss(&input, &target);
                parameter(&mut network, l, bias)[(i, j)] = original;
                (above - below) / (2.0 * h)
            });
            assert_abs_diff_eq!(numeric, *analytic, epsilon = 1e-8);
        }
    }
}

#[test]
fn learns_xor() {
    let (input, target) = xor();
    let mut network = Network::new(&[2, 4, 1], &[Activation::Tanh, Activation::Sigmoid], &mut StdRng::seed_from_u64(42));
    for _ in 0..5000 {
        network.train_step(&input, &target, 1.0);
    }
    assert_abs_diff_eq!(network.predict(&input), target, epsilon = 0.1);
}

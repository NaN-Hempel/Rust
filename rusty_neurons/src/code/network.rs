// A fully connected feed-forward neural network, written as matrix operations.
//
// Each layer turns its input A₀ (n inputs × m samples, one sample per column) into its output A:
//     Z = W·A₀ + b·1ᵀ      W: weights, b: bias, 1ᵀ: a row of m ones, which copies b to every sample
//     A = f(Z)             f: the activation, element by element
//
// Training is gradient descent: nudge every weight against the gradient of the loss L.
// Backpropagation finds those gradients from the last layer back to the first, with the chain
// rule written in matrices:
//     δ      = ∂L/∂A ⊙ f′(Z)   how L changes with each element of Z
//     ∂L/∂W  = δ·A₀ᵀ
//     ∂L/∂b  = δ·1             δ summed over the samples
//     ∂L/∂A₀ = Wᵀ·δ            handed back to the layer before, as its ∂L/∂A

use rand::RngExt;
use crate::code::fncts::{mse, mse_gradient, Activation};
use crate::code::math::{ones, random, Matrix};

#[derive(Clone)]
pub struct Layer {
    pub weights: Matrix, // outputs × inputs
    pub bias: Matrix,    // outputs × 1
    pub activation: Activation,
}

impl Layer {
    // Random weights and zero bias.
    pub fn new(inputs: usize, outputs: usize, activation: Activation, rng: &mut impl RngExt) -> Layer {
        Layer {
            weights: random(outputs, inputs, activation.init_limit(inputs, outputs), rng),
            bias: Matrix::zeros(outputs, 1),
            activation,
        }
    }
}

// ∂L/∂W and ∂L/∂b for one layer, the same shapes as its weights and bias.
pub struct Gradient {
    pub weights: Matrix,
    pub bias: Matrix,
}

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    // sizes lists the neurons per layer, inputs first: [2, 4, 1] is 2 inputs, a hidden layer of
    // 4 neurons and 1 output. activations has one entry for each layer after the inputs.
    pub fn new(sizes: &[usize], activations: &[Activation], rng: &mut impl RngExt) -> Network {
        assert_eq!(activations.len(), sizes.len() - 1, "need one activation per layer after the inputs");
        let layers = sizes.windows(2).zip(activations).map(|(size, &activation)| Layer::new(size[0], size[1], activation, rng)).collect();
        Network { layers }
    }

    // The network's output for each sample (column) of input.
    pub fn predict(&self, input: &Matrix) -> Matrix {
        let (_, mut outputs) = self.forward(input);
        outputs.pop().unwrap()
    }

    // The output for a single sample, as a plain list of numbers.
    pub fn decide(&self, input: &[f64]) -> Vec<f64> {
        self.predict(&Matrix::from_column_slice(input.len(), 1, input)).as_slice().to_vec()
    }

    // Number of weights and biases in all the layers together.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.weights.len() + layer.bias.len()).sum()
    }

    // Every weight and bias in one list, layer by layer (each layer's weights, then its bias).
    // For training without gradients, as in evolution, where this list is the genome.
    pub fn params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.param_count());
        for layer in &self.layers {
            params.extend_from_slice(layer.weights.as_slice());
            params.extend_from_slice(layer.bias.as_slice());
        }
        params
    }

    // The reverse of params(): sets every weight and bias from one list, in the same order.
    pub fn set_params(&mut self, params: &[f64]) {
        assert_eq!(params.len(), self.param_count(), "wrong number of parameters");
        let mut rest = params;
        for layer in &mut self.layers {
            for matrix in [&mut layer.weights, &mut layer.bias] {
                let (mine, others) = rest.split_at(matrix.len());
                matrix.copy_from_slice(mine);
                rest = others;
            }
        }
    }

    pub fn loss(&self, input: &Matrix, target: &Matrix) -> f64 {
        mse(&self.predict(input), target)
    }

    // Runs the input through every layer, keeping what backpropagation needs: each layer's Z,
    // and the outputs, where outputs[0] is the input itself and outputs[l + 1] is layer l's output.
    fn forward(&self, input: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let ones = ones(1, input.ncols()); // 1ᵀ
        let mut zs = Vec::with_capacity(self.layers.len());
        let mut outputs = vec![input.clone()];
        for layer in &self.layers {
            let z = &layer.weights * outputs.last().unwrap() + &layer.bias * &ones;
            outputs.push(layer.activation.apply(&z));
            zs.push(z);
        }
        (zs, outputs)
    }

    // Backpropagation: from the zs and outputs of a forward pass and grad, the gradient of the
    // loss with respect to the network's output (∂L/∂A of the last layer), works back through the
    // layers to the gradient of every layer's weights and bias. It doesn't depend on which loss is used.
    pub fn backwards_pass(&self, zs: &[Matrix], outputs: &[Matrix], mut grad: Matrix) -> Vec<Gradient> {
        let ones = ones(outputs[0].ncols(), 1); // 1
        let mut gradients = Vec::with_capacity(self.layers.len());
        for (l, layer) in self.layers.iter().enumerate().rev() {
            let delta = grad.component_mul(&layer.activation.derivative(&zs[l])); // ⊙
            gradients.push(Gradient { weights: &delta * &outputs[l].transpose(), bias: &delta * &ones });
            grad = &layer.weights.transpose() * &delta;
        }
        gradients.reverse(); // they were found last layer first
        gradients
    }

    // The loss, and its gradient with respect to every layer's weights and bias.
    pub fn gradients(&self, input: &Matrix, target: &Matrix) -> (f64, Vec<Gradient>) {
        let (zs, outputs) = self.forward(input);
        let prediction = outputs.last().unwrap();
        let gradients = self.backwards_pass(&zs, &outputs, mse_gradient(prediction, target));
        (mse(prediction, target), gradients)
    }

    // One step of gradient descent: W ← W − η·∂L/∂W and b ← b − η·∂L/∂b, η being the learning rate.
    pub fn descend(&mut self, gradients: &[Gradient], learning_rate: f64) {
        for (layer, gradient) in self.layers.iter_mut().zip(gradients) {
            layer.weights -= &gradient.weights * learning_rate;
            layer.bias -= &gradient.bias * learning_rate;
        }
    }

    // One training step on a batch. Returns the loss before the step.
    pub fn train_step(&mut self, input: &Matrix, target: &Matrix, learning_rate: f64) -> f64 {
        let (loss, gradients) = self.gradients(input, target);
        self.descend(&gradients, learning_rate);
        loss
    }
}

#[cfg(test)]
mod tests;

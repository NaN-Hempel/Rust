// Activation functions and the loss function, with the derivatives backpropagation needs.
// An activation acts on each element on its own. It is the one part of the network that isn't
// linear algebra, and without it any stack of layers would collapse into a single matrix.

use crate::code::math::Matrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Identity, // f(z) = z
    Sigmoid,  // f(z) = 1 / (1 + e⁻ᶻ), squashes into (0, 1)
    Tanh,     // f(z) = tanh(z), squashes into (−1, 1)
    Relu,     // f(z) = max(0, z)
}

impl Activation {
    // f(Z), element by element.
    pub fn apply(self, z: &Matrix) -> Matrix {
        z.map(|z| self.value(z))
    }

    // f′(Z), element by element.
    pub fn derivative(self, z: &Matrix) -> Matrix {
        z.map(|z| self.slope(z))
    }

    fn value(self, z: f64) -> f64 {
        match self {
            Activation::Identity => z,
            Activation::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Activation::Tanh => z.tanh(),
            Activation::Relu => z.max(0.0),
        }
    }

    fn slope(self, z: f64) -> f64 {
        match self {
            Activation::Identity => 1.0,
            Activation::Sigmoid => {
                let s = self.value(z);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - z.tanh().powi(2),
            Activation::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    // Starting weights are drawn from ±limit. This limit keeps the signal from growing or fading
    // from layer to layer: Glorot's for squashing functions, He's for ReLU.
    pub fn init_limit(self, inputs: usize, outputs: usize) -> f64 {
        match self {
            Activation::Relu => (6.0 / inputs as f64).sqrt(),
            _ => (6.0 / (inputs + outputs) as f64).sqrt(),
        }
    }
}

// Mean squared error over a batch of m samples, one per column: L = ‖A − Y‖² / 2m.
pub fn mse(prediction: &Matrix, target: &Matrix) -> f64 {
    (prediction - target).norm_squared() / (2.0 * prediction.ncols() as f64)
}

// Its gradient with respect to the prediction: ∂L/∂A = (A − Y) / m.
pub fn mse_gradient(prediction: &Matrix, target: &Matrix) -> Matrix {
    (prediction - target) * (1.0 / prediction.ncols() as f64)
}

#[cfg(test)]
mod tests;

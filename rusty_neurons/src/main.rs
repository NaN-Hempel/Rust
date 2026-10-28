mod code;
mod data;

use rand::SeedableRng;
use rand::rngs::StdRng;

use code::fncts::Activation;
use code::network::Network;

const EPOCHS: usize = 5000;
const LEARNING_RATE: f64 = 1.0;
const SEED: u64 = 42; // another seed gives other starting weights

fn main() {
    let (inputs, targets) = data::xor::xor();

    // 2 inputs, a hidden layer of 4 neurons, 1 output.
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut network = Network::new(&[2, 4, 1], &[Activation::Tanh, Activation::Sigmoid], &mut rng);

    for epoch in 0..EPOCHS {
        let loss = network.train_step(&inputs, &targets, LEARNING_RATE);
        if epoch % 500 == 0 {
            println!("epoch {epoch:>5}   loss {loss:.6}");
        }
    }
    println!("final        loss {:.6}", network.loss(&inputs, &targets));

    let prediction = network.predict(&inputs);
    println!("\nx₁  x₂   target   prediction");
    for j in 0..inputs.ncols() {
        println!("{}   {}    {}        {:.3}", inputs[(0, j)], inputs[(1, j)], targets[(0, j)], prediction[(0, j)]);
    }

    print!("\nHidden layer weights W₁:{:.4}", network.layers[0].weights);
    print!("Hidden layer bias b₁:{:.4}", network.layers[0].bias);
}

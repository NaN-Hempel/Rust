// Evolution: a population of networks flies, the best are kept, and the rest are replaced by
// mutated copies of good ones. Repeated over many generations, the flights get better, with no
// gradients at all: only the score of each whole flight.
//
// A network's genome is every one of its weights and biases in one list (Network::params).

use std::fs;
use std::io;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rusty_neurons::code::network::Network;

use crate::probe::ephemeris::Ephemeris;
use crate::probe::flight::{fly, new_brain, Flight, SHAPE};

const POPULATION: usize = 100;
// The best this many are kept unchanged into the next generation, so the best never gets worse.
const ELITES: usize = 10;
// Saved with the best network, to be shown with it: this many others, picked at random from the
// population, to see what the rest of it does.
const SAVED_RANDOM: usize = 10;
// Parents are picked by tournament: the best of this many, chosen at random.
const TOURNAMENT: usize = 3;
// Each child's genes are its parent's plus noise with this standard deviation, which shrinks
// every generation (by SIGMA_DECAY, down to MIN_SIGMA): big jumps to explore at first, then
// small ones to fine-tune.
const START_SIGMA: f64 = 0.1;
const SIGMA_DECAY: f64 = 0.99;
const MIN_SIGMA: f64 = 0.01;
const SEED: u64 = 2026;

// Runs `generations` generations, saving the best network and SAVED_RANDOM random others to path,
// best first, after each. Returns the best.
pub fn evolve(ephemeris: &Ephemeris, generations: usize, path: &str) -> Network {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut template = new_brain(&mut rng);
    let mut population: Vec<Vec<f64>> = (0..POPULATION).map(|_| new_brain(&mut rng).params()).collect();
    let mut sigma = START_SIGMA;

    println!("gen    best  (closest km   km/s   Δv km/s  launch day  km/s)   median");
    for generation in 0..generations {
        let flights = evaluate(&population, &template, ephemeris);
        let mut order: Vec<usize> = (0..POPULATION).collect();
        order.sort_by(|&a, &b| flights[a].score.total_cmp(&flights[b].score));
        let best = &flights[order[0]];
        println!(
            "{:>3} {:>7.3}  ({:>11.0} {:>6.2} {:>9.2}  {}){}   {:.3}",
            generation,
            best.score,
            best.distance / 1000.0,
            best.speed / 1000.0,
            best.delta_v / 1000.0,
            best.launch.map_or("   not launched".to_string(), |(t, speed)| format!("{:>10.0} {:>5.2}", t / 86_400.0, speed / 1000.0)),
            best.destroyed.as_ref().map_or(String::new(), |b| format!(" destroyed near {}", b)),
            flights[order[POPULATION / 2]].score
        );
        // The random picks get their own random numbers, so evolution goes the same with or without them.
        let mut picker = StdRng::seed_from_u64(SEED + 1 + generation as u64);
        let mut others: Vec<usize> = order[1..].to_vec();
        for k in 0..SAVED_RANDOM.min(others.len()) {
            let j = picker.random_range(k..others.len());
            others.swap(k, j);
        }
        let saved: Vec<Network> = std::iter::once(order[0])
            .chain(others.into_iter().take(SAVED_RANDOM))
            .map(|i| {
                let mut brain = template.clone();
                brain.set_params(&population[i]);
                brain
            })
            .collect();
        save(&saved, path).unwrap_or_else(|e| eprintln!("couldn't save {}: {}", path, e));
        template.clone_from(&saved[0]);

        let mut next: Vec<Vec<f64>> = order[..ELITES].iter().map(|&i| population[i].clone()).collect();
        while next.len() < POPULATION {
            let parent = (0..TOURNAMENT).map(|_| rng.random_range(0..POPULATION)).min_by(|&a, &b| flights[a].score.total_cmp(&flights[b].score)).unwrap();
            next.push(population[parent].iter().map(|&gene| gene + sigma * gaussian(&mut rng)).collect());
        }
        population = next;
        sigma = (sigma * SIGMA_DECAY).max(MIN_SIGMA);
    }
    template
}

// Flies every genome, spread over all the processor's cores. Flights are independent, and the
// ephemeris is only read, so they can share it.
fn evaluate(population: &[Vec<f64>], template: &Network, ephemeris: &Ephemeris) -> Vec<Flight> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = population.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let workers: Vec<_> = population
            .chunks(chunk)
            .map(|genomes| {
                scope.spawn(move || {
                    let mut brain = template.clone();
                    genomes.iter().map(|genome| { brain.set_params(genome); fly(&brain, ephemeris, false) }).collect::<Vec<_>>()
                })
            })
            .collect();
        workers.into_iter().flat_map(|worker| worker.join().expect("a flight panicked")).collect()
    })
}

// A number from the standard normal distribution (mean 0, standard deviation 1), by the
// Box-Muller transform of two uniform ones.
fn gaussian(rng: &mut impl RngExt) -> f64 {
    let u: f64 = 1.0 - rng.random::<f64>(); // in (0, 1], so its log is finite
    let v: f64 = rng.random();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

// Saves networks (best first) as text: a comment, the layer sizes, then for each network a comment with its
// number and every weight and bias, one per line.
pub fn save(brains: &[Network], path: &str) -> io::Result<()> {
    let sizes: Vec<String> = SHAPE.iter().map(|s| s.to_string()).collect();
    let mut text = format!("# probe networks, best first: layer sizes, then each one's weights and biases (see Network::params)\n{}\n", sizes.join(" "));
    for (n, brain) in brains.iter().enumerate() {
        text += &format!("# {}\n", n + 1);
        for p in brain.params() {
            text += &format!("{:e}\n", p);
        }
    }
    fs::write(path, text)
}

// Loads the networks saved by save().
pub fn load(path: &str) -> io::Result<Vec<Network>> {
    let text = fs::read_to_string(path)?;
    let invalid = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {}", path, why));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let sizes: Vec<usize> = lines.next().unwrap_or("").split_whitespace().filter_map(|s| s.parse().ok()).collect();
    if sizes != SHAPE {
        return Err(invalid(&format!("layer sizes are {:?}, but the probe's network is {:?}", sizes, SHAPE)));
    }
    let params: Vec<f64> = lines.map(|l| l.trim().parse().map_err(|_| invalid(&format!("not a number: {}", l)))).collect::<Result<_, _>>()?;
    let template = new_brain(&mut StdRng::seed_from_u64(0));
    let count = template.param_count();
    if params.is_empty() || params.len() % count != 0 {
        return Err(invalid(&format!("{} parameters, not a whole number of networks of {}", params.len(), count)));
    }
    Ok(params
        .chunks(count)
        .map(|genome| {
            let mut brain = template.clone();
            brain.set_params(genome);
            brain
        })
        .collect())
}

#[cfg(test)]
mod tests;

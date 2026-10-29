// Tests for evolve.rs.

use super::*;

#[test]
fn save_and_load_round_trip() {
    let mut rng = StdRng::seed_from_u64(4);
    let brains: Vec<Network> = (0..3).map(|_| new_brain(&mut rng)).collect();
    let path = std::env::temp_dir().join(format!("sol_rust_probe_{}.txt", std::process::id()));
    let path = path.to_str().unwrap();
    save(&brains, path).unwrap();
    let loaded = load(path).unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(loaded.len(), 3);
    for (a, b) in loaded.iter().zip(&brains) {
        assert_eq!(a.params(), b.params());
    }
}

#[test]
fn gaussian_has_mean_0_and_deviation_1() {
    let mut rng = StdRng::seed_from_u64(1);
    let samples: Vec<f64> = (0..100_000).map(|_| gaussian(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let deviation = (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    assert!(mean.abs() < 0.01, "mean {}", mean);
    assert!((deviation - 1.0).abs() < 0.01, "standard deviation {}", deviation);
}

// Basic sums on audio: making a test tone, measuring how loud a signal is, and converting
// between plain gains and decibels. Samples are f64, with full scale at -1.0 and 1.0.

use std::f64::consts::PI;

#[cfg(test)]
mod tests;

// A sine wave of `frequency` Hz and peak `amplitude`, lasting `seconds`, at `sample_rate` samples
// per second.
pub fn sine(frequency: f64, amplitude: f64, seconds: f64, sample_rate: u32) -> Vec<f64> {
    let n = (seconds * sample_rate as f64).round() as usize;
    let step = 2.0 * PI * frequency / sample_rate as f64; // phase advance per sample (rad)
    (0..n).map(|i| amplitude * (step * i as f64).sin()).collect()
}

// The largest absolute sample value (0 for no samples).
pub fn peak(samples: &[f64]) -> f64 {
    samples.iter().fold(0.0, |max, s| max.max(s.abs()))
}

// Root mean square: the "average" loudness. For a sine of peak A, it is A / √2.
pub fn rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt()
}

// A gain as decibels: 1.0 is 0 dB, 0.5 about -6 dB. Silence (0) is -infinity.
pub fn gain_to_db(gain: f64) -> f64 {
    20.0 * gain.log10()
}

// The reverse of gain_to_db.
pub fn db_to_gain(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

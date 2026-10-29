// Where the planets are during a probe's flight, worked out once beforehand.
//
// A probe pulls on nothing, so the planets move the same whatever it does. Every flight in an
// evolution run (thousands of them) can then share one recording of the planets, and each only
// has to integrate its own probe.

use crate::code::math::Vec3;
use crate::code::physics::G;
use crate::code::sim::{self, TEST_PARTICLE_MASS};
use crate::data::planets::Planet;

// The bodies whose gravity the probe feels: the planets out to Saturn, the Moon, and Jupiter's
// four big moons, which matter once a probe gets to Jupiter. The rest pull on it far less.
pub const BODIES: [&str; 12] = ["Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Io", "Europa", "Ganymede", "Callisto", "Saturn"];

pub struct Ephemeris {
    pub names: Vec<String>,
    // G times each body's mass, and its radius in metres.
    pub mu: Vec<f64>,
    pub radius: Vec<f64>,
    // Seconds between samples.
    pub interval: f64,
    // Every body's position and velocity at every sample: sample k is states[k * count..(k + 1) * count].
    states: Vec<(Vec3, Vec3)>,
}

impl Ephemeris {
    // Simulates planets (all of them, so the recorded bodies move as they do in the window) in
    // steps of dt seconds for `duration` seconds, recording BODIES every `interval` seconds.
    pub fn record(planets: Vec<Planet>, duration: f64, interval: f64, dt: f64) -> Ephemeris {
        let index: Vec<usize> = BODIES
            .iter()
            .map(|&name| planets.iter().position(|p| p.name == name).unwrap_or_else(|| panic!("no {} in the data", name)))
            .collect();
        // The simulation writes massive bodies' states back after every step; test particles'
        // only when synced.
        assert!(index.iter().all(|&i| planets[i].mass >= TEST_PARTICLE_MASS), "the recorded bodies must be massive");
        let names = index.iter().map(|&i| planets[i].name.clone()).collect();
        let mu = index.iter().map(|&i| G * planets[i].mass).collect();
        let radius = index.iter().map(|&i| planets[i].radius * 1000.0).collect();

        let steps_per_sample = (interval / dt).round() as usize;
        let samples = (duration / interval).ceil() as usize + 1;
        let mut simulation = sim::new_simulation(planets, 0, dt);
        let mut states = Vec::with_capacity(samples * index.len());
        for k in 0..samples {
            if k > 0 {
                for _ in 0..steps_per_sample {
                    sim::update_simulation(&mut simulation, dt);
                }
            }
            states.extend(index.iter().map(|&i| (simulation.planets[i].position, simulation.planets[i].velocity)));
        }
        Ephemeris { names, mu, radius, interval: steps_per_sample as f64 * dt, states }
    }

    pub fn count(&self) -> usize {
        self.mu.len()
    }

    pub fn samples(&self) -> usize {
        self.states.len() / self.count()
    }

    // Index of the body called name.
    pub fn body(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).unwrap_or_else(|| panic!("{} is not in the ephemeris", name))
    }

    // Every body's position and velocity at sample k.
    pub fn sample(&self, k: usize) -> &[(Vec3, Vec3)] {
        &self.states[k * self.count()..(k + 1) * self.count()]
    }

    // Every body's position at time t (seconds, within the recording), into out.
    // Between samples it uses cubic Hermite interpolation: the cubic that has the right position
    // and velocity at both ends. Its error grows with the fourth power of the interval, so hourly
    // samples are accurate to metres, even for the Moon.
    pub fn positions_at(&self, t: f64, out: &mut [Vec3]) {
        let (k, s) = self.locate(t);
        let (before, after) = (self.sample(k), self.sample(k + 1));
        let h = self.interval;
        for (x, (&(p0, v0), &(p1, v1))) in out.iter_mut().zip(before.iter().zip(after)) {
            *x = hermite(p0, v0, p1, v1, h, s);
        }
    }

    // The sample just before t, and how far t is towards the next one (0 to 1).
    fn locate(&self, t: f64) -> (usize, f64) {
        let f = (t / self.interval).clamp(0.0, (self.samples() - 1) as f64);
        let k = (f.floor() as usize).min(self.samples() - 2);
        (k, f - k as f64)
    }
}

// The cubic through position p0 (velocity v0) at s = 0 and p1 (v1) at s = 1, at s, where the
// whole interval is h seconds long.
pub fn hermite(p0: Vec3, v0: Vec3, p1: Vec3, v1: Vec3, h: f64, s: f64) -> Vec3 {
    let (s2, s3) = (s * s, s * s * s);
    p0 * (2.0 * s3 - 3.0 * s2 + 1.0) + v0 * (h * (s3 - 2.0 * s2 + s)) + p1 * (3.0 * s2 - 2.0 * s3) + v1 * (h * (s3 - s2))
}

#[cfg(test)]
mod tests;

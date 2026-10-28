// Contains the test particles, each with its own time step.
//
// Test particles are pulled by the massive bodies but pull on nothing (see sim.rs), so the
// massive bodies can be stepped first, by themselves, and each particle afterwards over a longer
// step of its own, using the massive bodies' positions interpolated from a short history.
//
// Each particle's step is a power of two times the base step: as long as it can be while staying
// accurate, up to MAX_LEVEL. A particle only steps when the number of base steps taken is a
// multiple of its step ("block time steps"), so all particles with the same step share the
// interpolated positions, and the history only needs to reach back MAX_LEVEL steps.
//
// Base steps can be shorter than the simulation's dt (when time runs slowly), so step lengths are
// chosen as if every base step were the full dt. A particle's step can then only turn out shorter
// in time than it needs to be, never longer.

use crate::code::math::{Vec3, DRIFT, KICK, PI};
use crate::code::physics::pull;
use crate::data::planets::Planet;

// Longest particle step, in base steps. With 5-minute base steps this is about 3.6 days.
pub const MAX_LEVEL: u64 = 1024;
// Number of different step lengths: 1, 2, 4, ... MAX_LEVEL base steps.
const LEVELS: usize = MAX_LEVEL.trailing_zeros() as usize + 1;
// A particle takes at least this many steps per orbit around whatever pulls on it hardest.
const STEPS_PER_ORBIT: f64 = 200.0;
// Fractions of a step at which Yoshida's three kicks happen: the sum of the drifts before each.
const KICK_TIME: [f64; 3] = [DRIFT[0], DRIFT[0] + DRIFT[1], DRIFT[0] + DRIFT[1] + DRIFT[2]];

pub struct Particles {
    // Index into planets of each particle.
    pub index: Vec<usize>,
    // Each particle's state at its last step, and the acceleration there (used to predict where
    // it is in between steps). These, not planets, are the particles' true state.
    position: Vec<Vec3>,
    velocity: Vec<Vec3>,
    acc: Vec<Vec3>,
    last_time: Vec<f64>,
    // Particles (indices into the vectors above) by step length: buckets[k] steps every 2^k base steps.
    buckets: Vec<Vec<usize>>,
    history: History,
    // Base steps taken, the simulated time, and the longest a base step can be.
    step: u64,
    time: f64,
    max_dt: f64,
    // Massive bodies' interpolated positions at each of the three kicks; reused.
    at_kick: [Vec<Vec3>; 3],
}

impl Particles {
    // index: which planets are particles. mu, positions and velocities are the massive bodies'.
    // max_dt is the longest base step advance() will be given.
    pub fn new(index: Vec<usize>, planets: &[Planet], mu: &[f64], positions: &[Vec3], velocities: &[Vec3], dt: f64) -> Self {
        let count = index.len();
        let mut particles = Particles {
            position: index.iter().map(|&i| planets[i].position).collect(),
            velocity: index.iter().map(|&i| planets[i].velocity).collect(),
            acc: vec![Vec3::ZERO; count],
            last_time: vec![0.0; count],
            buckets: vec![Vec::new(); LEVELS],
            history: History::new(mu.len()),
            step: 0,
            time: 0.0,
            max_dt: dt,
            at_kick: [vec![Vec3::ZERO; mu.len()], vec![Vec3::ZERO; mu.len()], vec![Vec3::ZERO; mu.len()]],
            index,
        };
        particles.history.record(0, 0.0, positions, velocities);
        for k in 0..count {
            particles.acc[k] = pull(mu, positions, particles.position[k]);
            let level = level_for(mu, positions, particles.position[k], dt, 0);
            particles.buckets[level].push(k);
        }
        particles
    }

    // Called after the massive bodies have taken a base step of dt (at most max_dt), ending at
    // positions and velocities. Steps every particle that is due, and writes its new state to planets.
    pub fn advance(&mut self, mu: &[f64], positions: &[Vec3], velocities: &[Vec3], dt: f64, planets: &mut [Planet]) {
        self.step += 1;
        self.time += dt;
        self.history.record(self.step, self.time, positions, velocities);

        // (particle, new level) for particles whose step length changes; applied after the loop,
        // so no particle steps twice.
        let mut moves = Vec::new();
        for level in 0..LEVELS {
            let length = 1u64 << level;
            if self.step % length != 0 {
                break; // longer steps aren't due either
            }
            if self.buckets[level].is_empty() {
                continue;
            }
            let from = self.step - length;
            let start = self.history.time(from);
            let h = self.time - start;
            for (kick, out) in self.at_kick.iter_mut().enumerate() {
                self.history.positions_at(from, self.step, start + KICK_TIME[kick] * h, out);
            }

            for &k in &self.buckets[level] {
                // One Yoshida step of length h, kicking with the interpolated positions.
                let (mut x, mut v) = (self.position[k], self.velocity[k]);
                for stage in 0..4 {
                    x += v * (DRIFT[stage] * h);
                    if stage < 3 {
                        v += pull(mu, &self.at_kick[stage], x) * (KICK[stage] * h);
                    }
                }
                self.position[k] = x;
                self.velocity[k] = v;
                self.acc[k] = pull(mu, positions, x);
                self.last_time[k] = self.time;
                planets[self.index[k]].position = x;
                planets[self.index[k]].velocity = v;

                let new_level = level_for(mu, positions, x, self.max_dt, self.step);
                if new_level != level {
                    moves.push((k, level, new_level));
                }
            }
        }
        for (k, from, to) in moves {
            self.buckets[from].retain(|&p| p != k);
            self.buckets[to].push(k);
        }
    }

    // Writes every particle's predicted position and velocity now into planets. Particles only
    // step every so often, so between steps planets would otherwise show where they last were.
    pub fn sync(&self, planets: &mut [Planet]) {
        for (k, &i) in self.index.iter().enumerate() {
            let t = self.time - self.last_time[k];
            planets[i].position = self.position[k] + self.velocity[k] * t + self.acc[k] * (t * t / 2.0);
            planets[i].velocity = self.velocity[k] + self.acc[k] * t;
        }
    }

    // Base steps taken, and the simulated time.
    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    // Whether every particle has just stepped (every MAX_LEVEL base steps, all step lengths come
    // due together). Then nothing from before is needed to carry on: the bodies' positions and
    // velocities and each particle's step length are enough to restore the simulation exactly.
    pub fn at_sync_point(&self) -> bool {
        self.step % MAX_LEVEL == 0
    }

    // Fills out with each particle's bucket (log2 of its step length), in index order.
    pub fn levels_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.resize(self.index.len(), 0);
        for (level, bucket) in self.buckets.iter().enumerate() {
            for &k in bucket {
                out[k] = level as u8;
            }
        }
    }

    // Puts the particles back as they were at a sync point: at base step `step` and simulated
    // time `time`, in the buckets `levels`, with their states as in planets. mu, positions and
    // velocities are the massive bodies' at that moment. Each particle steps on its own, so the
    // order within the rebuilt buckets doesn't matter.
    pub fn restore(&mut self, step: u64, time: f64, levels: &[u8], mu: &[f64], positions: &[Vec3], velocities: &[Vec3], planets: &[Planet]) {
        self.step = step;
        self.time = time;
        self.history.record(step, time, positions, velocities);
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        for (k, &i) in self.index.iter().enumerate() {
            self.position[k] = planets[i].position;
            self.velocity[k] = planets[i].velocity;
            self.acc[k] = pull(mu, positions, self.position[k]);
            self.last_time[k] = time;
            self.buckets[levels[k] as usize].push(k);
        }
    }
}

// The bucket (log2 of the step length in base steps) for a particle at x, at base step `step`,
// with base steps of up to max_dt seconds.
// The step is the longest power of two that gives STEPS_PER_ORBIT steps per orbit around the
// body pulling on it hardest, and that `step` is a multiple of, so the particle's steps stay in
// line with the other particles of the same length.
fn level_for(mu: &[f64], positions: &[Vec3], x: Vec3, max_dt: f64, step: u64) -> usize {
    // sqrt(r³ / mu) is an orbit's period divided by 2π; the shortest over all massive bodies counts.
    let shortest = mu
        .iter()
        .zip(positions)
        .map(|(&m, &p)| ((p - x).norm_squared().powf(1.5) / m).sqrt())
        .fold(f64::INFINITY, f64::min);
    let wanted = 2.0 * PI * shortest / STEPS_PER_ORBIT / max_dt;
    let mut level = (wanted.max(1.0).log2().floor() as usize).min(LEVELS - 1);
    while step % (1u64 << level) != 0 {
        level -= 1;
    }
    level
}

// The massive bodies' positions and velocities at the end of each of the last MAX_LEVEL base
// steps, in a ring buffer indexed by step number.
struct History {
    massive: usize,
    times: Vec<f64>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
}

const SLOTS: u64 = MAX_LEVEL + 1;

impl History {
    fn new(massive: usize) -> Self {
        let slots = SLOTS as usize;
        History { massive, times: vec![0.0; slots], positions: vec![Vec3::ZERO; slots * massive], velocities: vec![Vec3::ZERO; slots * massive] }
    }

    fn slot(&self, step: u64) -> usize {
        (step % SLOTS) as usize
    }

    fn time(&self, step: u64) -> f64 {
        self.times[self.slot(step)]
    }

    fn record(&mut self, step: u64, time: f64, positions: &[Vec3], velocities: &[Vec3]) {
        let s = self.slot(step);
        self.times[s] = time;
        let range = s * self.massive..(s + 1) * self.massive;
        self.positions[range.clone()].copy_from_slice(positions);
        self.velocities[range].copy_from_slice(velocities);
    }

    // Fills out with the massive bodies' positions at `time`, which lies between steps `from` and
    // `to`, by cubic Hermite interpolation between the recorded steps either side of it.
    fn positions_at(&self, from: u64, to: u64, time: f64, out: &mut [Vec3]) {
        // Binary search for the last recorded step at or before time.
        let (mut lo, mut hi) = (from, to);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.time(mid) <= time { lo = mid } else { hi = mid }
        }
        let (a, b) = (self.slot(lo), self.slot(lo + 1));
        let h = self.times[b] - self.times[a];
        let u = (time - self.times[a]) / h;
        let (u2, u3) = (u * u, u * u * u);
        // Hermite basis functions, for the start and end positions and (scaled by h) velocities.
        let (p0, v0, p1, v1) = (2.0 * u3 - 3.0 * u2 + 1.0, (u3 - 2.0 * u2 + u) * h, 3.0 * u2 - 2.0 * u3, (u3 - u2) * h);
        let n = self.massive;
        for j in 0..n {
            out[j] = self.positions[a * n + j] * p0 + self.velocities[a * n + j] * v0 + self.positions[b * n + j] * p1 + self.velocities[b * n + j] * v1;
        }
    }
}

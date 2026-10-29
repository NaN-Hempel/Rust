// Contains functions for handling the simulation of the solar system.

use crate::code::physics::*;
use crate::code::math::*;
use crate::code::particles::Particles;
use crate::data::planets::Planet;

// Bodies lighter than this (kg) are test particles: gravity moves them, but they don't pull on
// anything. In the data, everything below it is an asteroid, comet or spacecraft (Bennu, the
// heaviest, is 7e13 kg) and everything above it a moon or bigger (Deimos, the lightest, 1.8e18 kg).
pub const TEST_PARTICLE_MASS: f64 = 1.0e16;

// Represents the state of the simulation, including positions, velocities, and masses of celestial bodies.
// The massive bodies all take the same step, dt; the test particles take steps of their own
// (see particles.rs).
pub struct Simulation {
    pub planets: Vec<Planet>,
    // Steps for run_simulation(), which only the tests use; the window steps every frame instead.
    #[cfg_attr(not(test), allow(dead_code))]
    pub n: usize,
    pub dt: f64,
    // Indices into planets of the massive bodies. The buffers below are in this order.
    massive: Vec<usize>,
    // G times each massive body's mass, computed once; masses don't change.
    mu: Vec<f64>,
    // Reused every step, so stepping allocates nothing.
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    acc: Vec<Vec3>,
    pub particles: Particles,
    // Bodies that follow a path worked out beforehand (like a probe flown by probe::flight)
    // instead of being moved by gravity: each one's index into planets, and its position and
    // velocity at a given time. They pull on nothing.
    scripted: Vec<(usize, Box<dyn Fn(f64) -> (Vec3, Vec3)>)>,
}

pub fn new_simulation(planets: Vec<Planet>, n: usize, dt: f64) -> Simulation {
    with_test_particle_mass(planets, n, dt, TEST_PARTICLE_MASS)
}

// As new_simulation, with a different cut-off for test particles (0 makes every body massive).
pub fn with_test_particle_mass(planets: Vec<Planet>, n: usize, dt: f64, threshold: f64) -> Simulation {
    let (massive, light): (Vec<usize>, Vec<usize>) = (0..planets.len()).partition(|&i| planets[i].mass >= threshold);
    let mu: Vec<f64> = massive.iter().map(|&i| G * planets[i].mass).collect();
    let positions: Vec<Vec3> = massive.iter().map(|&i| planets[i].position).collect();
    let velocities: Vec<Vec3> = massive.iter().map(|&i| planets[i].velocity).collect();
    Simulation {
        particles: Particles::new(light, &planets, &mu, &positions, &velocities, dt),
        acc: vec![Vec3::ZERO; massive.len()],
        scripted: Vec::new(),
        massive,
        mu,
        positions,
        velocities,
        planets,
        n,
        dt,
    }
}

impl Simulation {
    // Number of bodies whose gravity is simulated; the rest are test particles.
    pub fn massive_count(&self) -> usize {
        self.massive.len()
    }

    // Adds body, to follow track (its position and velocity at each simulated time) rather than gravity.
    pub fn add_scripted(&mut self, mut body: Planet, track: impl Fn(f64) -> (Vec3, Vec3) + 'static) {
        (body.position, body.velocity) = track(self.time());
        self.scripted.push((self.planets.len(), Box::new(track)));
        self.planets.push(body);
    }

    // Brings the test particles in planets up to the current time. They only step every so often,
    // so call this before reading their positions (for example, before drawing a frame).
    pub fn sync(&mut self) {
        self.particles.sync(&mut self.planets);
    }

    // Simulated seconds so far.
    pub fn time(&self) -> f64 {
        self.particles.time()
    }

    // Whether a snapshot can be taken now (see Particles::at_sync_point).
    pub fn at_sync_point(&self) -> bool {
        self.particles.at_sync_point()
    }

    // Saves the simulation into `snapshot`, reusing its memory. Only at a sync point.
    pub fn snapshot_into(&self, snapshot: &mut Snapshot) {
        debug_assert!(self.at_sync_point(), "snapshots can only be taken at sync points");
        snapshot.states.clear();
        snapshot.states.extend(self.planets.iter().map(|p| (p.position, p.velocity)));
        self.particles.levels_into(&mut snapshot.levels);
        snapshot.step = self.particles.step();
        snapshot.time = self.time();
    }

    // Puts the simulation back exactly as it was when `snapshot` was taken.
    pub fn restore(&mut self, snapshot: &Snapshot) {
        for (planet, &(position, velocity)) in self.planets.iter_mut().zip(&snapshot.states) {
            planet.position = position;
            planet.velocity = velocity;
        }
        for (k, &i) in self.massive.iter().enumerate() {
            self.positions[k] = self.planets[i].position;
            self.velocities[k] = self.planets[i].velocity;
        }
        self.particles.restore(snapshot.step, snapshot.time, &snapshot.levels, &self.mu, &self.positions, &self.velocities, &self.planets);
    }
}

// A saved moment of a simulation, for jumping back in time: every body's position and velocity
// and each test particle's step length, taken at a sync point, where that is all it takes to
// carry on exactly. About 100 bytes per body, rather than a copy of the whole simulation.
#[derive(Clone, Default)]
pub struct Snapshot {
    states: Vec<(Vec3, Vec3)>,
    levels: Vec<u8>,
    step: u64,
    pub time: f64,
}

// Advances the massive bodies by dt seconds, using a fourth-order Yoshida step, and steps any
// test particles that are due. Call sync() before reading the test particles' positions.
// dt can be anything up to the simulation's own dt, which is the longest step that is still
// accurate; shorter steps are just as accurate, only slower.
pub fn update_simulation(sim: &mut Simulation, dt: f64) {
    debug_assert!(dt <= sim.dt * (1.0 + 1e-9), "step of {} s is longer than the simulation's {} s", dt, sim.dt);
    for (k, &i) in sim.massive.iter().enumerate() {
        sim.positions[k] = sim.planets[i].position;
        sim.velocities[k] = sim.planets[i].velocity;
    }

    let mu = &sim.mu;
    yoshida4_step(&mut sim.positions, &mut sim.velocities, dt, &mut sim.acc, |x, out| accelerations(mu, x, out));

    for (k, &i) in sim.massive.iter().enumerate() {
        sim.planets[i].position = sim.positions[k];
        sim.planets[i].velocity = sim.velocities[k];
    }
    sim.particles.advance(&sim.mu, &sim.positions, &sim.velocities, dt, &mut sim.planets);
    let time = sim.time();
    for (i, track) in &sim.scripted {
        (sim.planets[*i].position, sim.planets[*i].velocity) = track(time);
    }
}

// Advances the simulation n steps of dt seconds each. Only the tests use it.
#[cfg_attr(not(test), allow(dead_code))]
pub fn run_simulation(sim: &mut Simulation) {
    for _ in 0..sim.n {
        update_simulation(sim, sim.dt);
    }
    sim.sync();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::particles::MAX_LEVEL;
    use crate::data::planets::{get_planets, jupiter_trojans, AU};
    use crate::code::physics::orbital_state;

    #[test]
    fn earth_returns_after_one_year() {
        let dt = 3600.0;
        let steps = (365.256 * 24.0) as usize; // one sidereal year in hours
        let mut sim = new_simulation(get_planets(), steps, dt);
        let start = sim.planets.iter().find(|p| p.name == "Earth").unwrap().position;
        run_simulation(&mut sim);
        let end = sim.planets.iter().find(|p| p.name == "Earth").unwrap().position;
        assert!((end - start).norm() / AU < 0.01, "Earth is {} AU from its start", (end - start).norm() / AU);
    }

    // Kinetic plus potential energy of the whole system, in joules.
    fn total_energy(planets: &[Planet]) -> f64 {
        let kinetic: f64 = planets.iter().map(|p| 0.5 * p.mass * p.velocity.norm().powi(2)).sum();
        let mut potential = 0.0;
        for i in 0..planets.len() {
            for j in (i + 1)..planets.len() {
                potential -= G * planets[i].mass * planets[j].mass / (planets[j].position - planets[i].position).norm();
            }
        }
        kinetic + potential
    }

    fn distance_km(planets: &[Planet], a: &str, b: &str) -> f64 {
        let find = |name: &str| planets.iter().find(|p| p.name == name).unwrap().position;
        (find(a) - find(b)).norm() / 1000.0
    }

    // One year in 5-minute steps, checking the energy and two moons' distances once a day.
    #[test]
    fn energy_is_conserved_and_moons_stay_in_orbit() {
        let mut sim = new_simulation(get_planets(), 288, 300.0);
        let start = total_energy(&sim.planets);
        let mut max_drift: f64 = 0.0;
        for _ in 0..365 {
            run_simulation(&mut sim);
            max_drift = max_drift.max(((total_energy(&sim.planets) - start) / start).abs());
            let phobos = distance_km(&sim.planets, "Phobos", "Mars");
            let moon = distance_km(&sim.planets, "Moon", "Earth");
            assert!((9_200.0..9_550.0).contains(&phobos), "Phobos is {} km from Mars", phobos);
            assert!((355_000.0..410_000.0).contains(&moon), "the Moon is {} km from Earth", moon);
        }
        println!("largest relative energy change: {:e}", max_drift);
        assert!(max_drift < ENERGY_TOLERANCE, "energy changed by {:e}", max_drift);
    }
    // Measured at about 2e-13 with Yoshida (RK4 gave 7e-13); this leaves a wide margin.
    const ENERGY_TOLERANCE: f64 = 1e-11;

    #[test]
    fn splits_bodies_at_the_mass_gap() {
        let sim = new_simulation(get_planets(), 0, 300.0);
        let light: Vec<&str> = sim.particles.index.iter().map(|&i| sim.planets[i].name.as_str()).collect();
        assert_eq!(light, ["Zoozve", "Apophis", "Bennu", "1P/Halley", "JWST", "Voyager 1", "Voyager 2", "New Horizons"]);
    }

    // Compared with simulating every body's gravity at the base step, after a year: the massive
    // bodies hardly notice that the particles no longer pull on them, and the particles, despite
    // their longer steps, end up close to where they would have been.
    // JWST is left out: at L2 it drifts away exponentially (50 million km in a year, in both
    // simulations), so any difference at all grows with it.
    #[test]
    fn test_particles_match_full_simulation() {
        let mut fast = new_simulation(get_planets(), 288, 300.0);
        let mut full = with_test_particle_mass(get_planets(), 288, 300.0, 0.0);
        for _ in 0..365 {
            run_simulation(&mut fast);
            run_simulation(&mut full);
        }
        for (i, (a, b)) in fast.planets.iter().zip(&full.planets).enumerate() {
            let off = (a.position - b.position).norm();
            if !fast.particles.index.contains(&i) {
                assert!(off < 1000.0, "{} is {} m off", a.name, off);
            } else if a.name != "JWST" {
                let relative = off / b.position.norm();
                assert!(relative < PARTICLE_TOLERANCE, "{} is {} km off ({:e} of its distance)", a.name, off / 1000.0, relative);
            }
        }
    }
    // Measured: massive bodies within 32 m; particles within 3.1e-6 of their distance from the
    // Sun (Apophis, about 390 km at 0.8 AU).
    const PARTICLE_TOLERANCE: f64 = 2.0e-5;

    // An asteroid belt of `count` test particles between 2.2 and 3.3 AU, at evenly spread angles.
    fn asteroid_belt(count: usize) -> Vec<Planet> {
        let sun_mass = get_planets()[0].mass;
        (0..count)
            .map(|k| {
                let f = k as f64 / count as f64;
                let (p, v) = orbital_state((2.2 + 1.1 * f) * AU, 0.1, 5.0, sun_mass);
                let (sin, cos) = (2.0 * PI * f * 37.0).sin_cos(); // 37 turns, so neighbours aren't adjacent
                let turn = |a: Vec3| Vec3::new(a.x * cos - a.y * sin, a.x * sin + a.y * cos, a.z);
                Planet::new(&format!("Asteroid {}", k), "Asteroid", 1.0e12, 1.0, 0.0, Some(0), turn(p), turn(v))
            })
            .collect()
    }

    // The Trojans stay around L4 and L5 for a whole Jupiter orbit (12 years, in hourly steps).
    // Measured: they swing between about 31° and 90° from Jupiter, over 48 years.
    #[test]
    fn trojans_stay_near_l4_and_l5() {
        let mut bodies = get_planets();
        let trojans = jupiter_trojans(&bodies, 200);
        bodies.extend(trojans);
        let mut sim = new_simulation(bodies, 24, 3600.0);
        for _ in 0..12 * 365 {
            run_simulation(&mut sim);
            let sun = sim.planets[0].position;
            let jupiter = sim.planets.iter().find(|p| p.name == "Jupiter").unwrap();
            let (rj, normal) = (jupiter.position - sun, (jupiter.position - sun).cross(jupiter.velocity - sim.planets[0].velocity).normalised());
            for p in sim.planets.iter().filter(|p| p.body_type == "Jupiter trojan") {
                // Angle ahead of Jupiter, seen from the Sun.
                let r = p.position - sun;
                let angle = normal.dot(rj.cross(r)).atan2(rj.dot(r)).to_degrees();
                let ahead = if p.name.contains("L4") { angle } else { -angle };
                assert!((20.0..120.0).contains(&ahead), "{} is {:.1}° from Jupiter", p.name, angle);
            }
        }
    }

    // Mixing shorter steps in, as slow time speeds do, gives the same orbits as full-length steps:
    // 100 days, each as 200 steps of 300 s and 264 of 100 s, against 288 steps of 300 s.
    // The test particles' history then has steps of different lengths in it.
    #[test]
    fn shorter_steps_give_the_same_result() {
        let mut fixed = new_simulation(get_planets(), 288, 300.0);
        let mut mixed = new_simulation(get_planets(), 0, 300.0);
        for _ in 0..100 {
            run_simulation(&mut fixed);
            for _ in 0..200 {
                update_simulation(&mut mixed, 300.0);
            }
            for _ in 0..264 {
                update_simulation(&mut mixed, 100.0);
            }
        }
        mixed.sync();
        let sun = fixed.planets[0].position;
        for (i, (a, b)) in mixed.planets.iter().zip(&fixed.planets).enumerate().skip(1) {
            // JWST drifts away from L2 exponentially, so any difference at all grows (see above).
            if a.name == "JWST" {
                continue;
            }
            let relative = (a.position - b.position).norm() / (b.position - sun).norm();
            let tolerance = if fixed.particles.index.contains(&i) { PARTICLE_STEP_TOLERANCE } else { MASSIVE_STEP_TOLERANCE };
            assert!(relative < tolerance, "{} differs by {:e} of its distance from the Sun", a.name, relative);
        }
    }
    // Measured: at most 4.5e-7 for the massive bodies (Phobos, about 100 km: its 7.7-hour orbit
    // gains most from the shorter steps) and 4.1e-7 for the test particles (Zoozve), whose own
    // longer steps get shorter in time too. Both are the full-length steps' small error, not a
    // difference in the orbits.
    const MASSIVE_STEP_TOLERANCE: f64 = 5.0e-6;
    const PARTICLE_STEP_TOLERANCE: f64 = 5.0e-6;

    // Restoring a snapshot and running on gives exactly the same result, to the last bit, as
    // never having stopped: through particles changing step length, sync points and steps of
    // different lengths.
    #[test]
    fn snapshots_restore_exactly() {
        let mut bodies = get_planets();
        let trojans = jupiter_trojans(&bodies, 50);
        bodies.extend(trojans);
        let mut sim = new_simulation(bodies, 0, 300.0);
        for _ in 0..2 * MAX_LEVEL {
            update_simulation(&mut sim, 300.0);
        }
        assert!(sim.at_sync_point());
        let mut snapshot = Snapshot::default();
        sim.snapshot_into(&mut snapshot);
        let run_on = |sim: &mut Simulation| {
            for k in 0..5000 {
                update_simulation(sim, if k % 3 == 0 { 120.0 } else { 300.0 });
            }
            sim.sync();
            sim.planets.iter().map(|p| (p.position, p.velocity)).collect::<Vec<_>>()
        };
        let first = run_on(&mut sim);
        sim.restore(&snapshot);
        assert_eq!(first, run_on(&mut sim));
    }

    fn time_per_frame(mut sim: Simulation) -> f64 {
        // Long enough to include several of the particles' longest steps, which come in bursts.
        let steps = 8_192;
        let start = std::time::Instant::now();
        for _ in 0..steps {
            update_simulation(&mut sim, 300.0);
        }
        start.elapsed().as_secs_f64() * 1000.0 / steps as f64 * 288.0
    }

    // Run with: cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_physics() {
        println!("physics, ms per frame (288 steps):");
        println!("{:>7} {:>13} {:>16}", "bodies", "all massive", "test particles");
        for extra in [0, 100, 1_000, 10_000] {
            let bodies = || { let mut b = get_planets(); b.extend(asteroid_belt(extra)); b };
            let count = bodies().len();
            // The all-massive version grows with the square of the count, so it is skipped when it would take minutes.
            let full = if count <= 1_100 { format!("{:.2}", time_per_frame(with_test_particle_mass(bodies(), 0, 300.0, 0.0))) } else { "-".into() };
            let fast = time_per_frame(new_simulation(bodies(), 0, 300.0));
            println!("{:>7} {:>13} {:>16.2}", count, full, fast);
        }
    }
}

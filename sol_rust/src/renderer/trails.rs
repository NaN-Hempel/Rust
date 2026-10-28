// Keeps the recent positions of each body, so its orbit can be drawn behind it.
// Positions are stored relative to the body's parent, so a moon's trail is a loop around its
// planet rather than a squiggle along the planet's orbit.
// Points are recorded at even intervals of simulated time, and each trail covers about one orbit,
// so trails look the same however fast time runs and however long the simulation's steps are.

use std::collections::VecDeque;

use crate::code::math::{Vec3, PI};
use crate::code::physics::G;
use crate::data::planets::Planet;

// About this many points are kept for one orbit, whatever its period. Fewer if the simulation's
// steps are longer than the interval between points (as for Phobos, at the fastest speeds).
const POINTS_PER_ORBIT: f64 = 500.0;
// Bodies that aren't in orbit (escaping probes) record a point a day, and keep this many.
const MAX_POINTS: usize = 12_000;
const DAY: f64 = 86_400.0;
// Jupiter trojans come by the hundred, so they get short tails, this fraction of an orbit long
// (about 7 months), instead of whole orbits.
pub const TROJAN_TAIL: f64 = 0.05;

pub struct Trails {
    // Each body's recorded positions.
    history: Vec<Trail>,
    // For each body: seconds between points, and how far back its trail reaches (0 for no trail).
    interval: Vec<f64>,
    span: Vec<f64>,
    // Simulated seconds so far, and when each body records its next point.
    time: f64,
    next: Vec<f64>,
}

impl Trails {
    // Each orbit's period comes from the body's energy relative to its parent:
    // a = -μ / 2ε (vis-viva), then T = 2π·sqrt(a³ / μ) (Kepler's third law), with μ = G·M.
    pub fn new(planets: &[Planet]) -> Self {
        let (interval, span) = planets
            .iter()
            .map(|p| {
                let Some(parent) = p.parent else {
                    return (0.0, 0.0); // the Sun
                };
                let mu = G * planets[parent].mass;
                let r = (p.position - planets[parent].position).norm();
                let v = (p.velocity - planets[parent].velocity).norm();
                let energy = v * v / 2.0 - mu / r;
                if energy >= 0.0 {
                    return (DAY, DAY * MAX_POINTS as f64); // not in orbit
                }
                let a = -mu / (2.0 * energy);
                let period = 2.0 * PI * (a.powi(3) / mu).sqrt();
                let span = if p.body_type == "Jupiter trojan" { period * TROJAN_TAIL } else { period };
                (period / POINTS_PER_ORBIT, span)
            })
            .unzip();
        Trails { history: (0..planets.len()).map(|_| Trail::default()).collect(), interval, span, time: 0.0, next: vec![0.0; planets.len()] }
    }

    // Moves the trails' clock on by dt seconds, adds the current position of every body that is
    // due, and drops points older than its trail's span. Call it after every simulation step,
    // with that step's length.
    pub fn record(&mut self, planets: &[Planet], dt: f64) {
        self.time += dt;
        for (i, planet) in planets.iter().enumerate() {
            if self.span[i] == 0.0 || self.time < self.next[i] {
                continue;
            }
            // One interval after the last point was due, so the spacing averages out to the
            // interval even when it isn't a whole number of steps; but not in the past, or a body
            // whose steps are longer than its interval would fall ever further behind.
            self.next[i] = (self.next[i] + self.interval[i]).max(self.time);
            let origin = planet.parent.map_or(Vec3::ZERO, |p| planets[p].position);
            let trail = &mut self.history[i];
            trail.push(self.time, planet.position - origin);
            trail.drop_before(self.time - self.span[i]);
        }
    }

    // Sets the trails' clock back to `time` (seconds), dropping points recorded after it, for
    // when the simulation jumps back. Points older than that are kept.
    pub fn rewind(&mut self, time: f64) {
        for trail in &mut self.history {
            trail.drop_after(time);
        }
        self.time = time;
        self.next.fill(time);
    }

    // Body i's recorded positions, oldest first, placed around its parent's current position,
    // and ending at the body itself so the trail doesn't stop short of it.
    pub fn points<'a>(&'a self, i: usize, planets: &[Planet]) -> impl Iterator<Item = Vec3> + 'a {
        let origin = planets[i].parent.map_or(Vec3::ZERO, |p| planets[p].position);
        let points = &self.history[i].points;
        let current = (!points.is_empty()).then_some(planets[i].position);
        points.iter().map(move |p| Vec3::new(p.offset[0] as f64, p.offset[1] as f64, p.offset[2] as f64) + origin).chain(current)
    }
}

// One body's trail: its points, oldest first, and the simulated times of the oldest and newest.
// To take half the memory, points are stored in single precision. A position relative to the
// parent is then accurate to about 1 part in 17 million: a few hundred km even 45 AU from the
// Sun, far below a pixel at any zoom that shows more than one of a trail's points. Each point's time
// is kept as the gap since the point before it, which single precision holds to well under a
// second, as the gaps are no longer than the interval between points.
#[derive(Default)]
struct Trail {
    points: VecDeque<Point>,
    first: f64,
    last: f64,
}

#[derive(Clone, Copy)]
struct Point {
    gap: f32,
    offset: [f32; 3],
}

impl Trail {
    fn push(&mut self, time: f64, offset: Vec3) {
        if self.points.is_empty() {
            self.first = time;
            self.last = time;
        }
        self.points.push_back(Point { gap: (time - self.last) as f32, offset: [offset.x as f32, offset.y as f32, offset.z as f32] });
        self.last = time;
    }

    // Drops points recorded before `time`.
    fn drop_before(&mut self, time: f64) {
        while self.first < time && self.points.pop_front().is_some() {
            match self.points.front() {
                Some(next) => self.first += next.gap as f64,
                None => break,
            }
        }
    }

    // Drops points recorded after `time`.
    fn drop_after(&mut self, time: f64) {
        while self.last > time {
            let Some(point) = self.points.pop_back() else { break };
            self.last -= point.gap as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::sim;
    use crate::data::planets::get_planets;

    // Simulates `days` days in steps of dt seconds (dt is also the simulation's longest step),
    // then returns body `name`'s trail as (number of points, how many days back the oldest is).
    fn trail_after(name: &str, days: f64, dt: f64) -> (usize, f64) {
        let mut s = sim::new_simulation(get_planets(), 0, dt);
        let mut trails = Trails::new(&s.planets);
        for _ in 0..(days * DAY / dt).round() as usize {
            sim::update_simulation(&mut s, dt);
            trails.record(&s.planets, dt);
        }
        let i = s.planets.iter().position(|p| p.name == name).unwrap();
        let trail = &trails.history[i];
        (trail.points.len(), (trails.time - trail.first) / DAY)
    }

    // Points 45 AU out (a Kuiper belt orbit) come back within a few hundred km, and their times
    // are still exact enough to trim the right points off either end.
    #[test]
    fn compact_points_are_accurate() {
        let (radius, gap) = (45.0 * 1.495_978_707e11, 1.9e7); // a point every 0.6 years
        let position = |k: usize| {
            let angle = k as f64 * 0.0126;
            Vec3::new(radius * angle.cos(), radius * angle.sin(), 1.0e11)
        };
        let mut trail = Trail::default();
        for k in 0..1000 {
            trail.push(k as f64 * gap, position(k));
        }
        for (k, p) in trail.points.iter().enumerate() {
            let stored = Vec3::new(p.offset[0] as f64, p.offset[1] as f64, p.offset[2] as f64);
            assert!((stored - position(k)).norm() < 400_000.0, "point {} is {} km off", k, (stored - position(k)).norm() / 1000.0);
        }
        trail.drop_after(500.0 * gap + 1.0);
        assert_eq!(trail.points.len(), 501);
        assert!((trail.last - 500.0 * gap).abs() < 1.0);
        trail.drop_before(100.0 * gap - 1.0);
        assert_eq!(trail.points.len(), 401);
        assert!((trail.first - 100.0 * gap).abs() < 1.0);
    }

    // Earth's trail covers one year in about 500 points, whatever the step length.
    #[test]
    fn trail_covers_one_orbit_whatever_the_step() {
        for dt in [3600.0, 1800.0] {
            let (points, days) = trail_after("Earth", 400.0, dt);
            assert!((499..=501).contains(&points), "Earth has {} points with {} s steps", points, dt);
            assert!((days - 365.25).abs() < 1.0, "Earth's trail covers {} days with {} s steps", days, dt);
        }
    }

    // Phobos orbits every 7.65 hours: faster than 500 points can be recorded with 5-minute steps,
    // so it gets a point every step instead, still covering one orbit.
    #[test]
    fn fast_moons_get_a_point_every_step() {
        for (dt, expected) in [(300.0, 92), (60.0, 459)] {
            let (points, days) = trail_after("Phobos", 1.0, dt);
            assert!(points.abs_diff(expected) <= 2, "Phobos has {} points with {} s steps", points, dt);
            assert!((days * 24.0 - 7.65).abs() < 0.1, "Phobos's trail covers {} hours with {} s steps", days * 24.0, dt);
        }
    }
}

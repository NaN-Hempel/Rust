// One flight: a probe steered by a neural network, from Earth towards the TARGET planet.
//
// Every few hours the network is shown where the probe is, where the target is and how much fuel
// is left, and answers with a thrust direction, a throttle, and whether to launch. Until it
// launches, the probe waits on Earth; the launch sends it off in the direction it chooses (see
// launch). After that, the probe coasts under gravity with the engine as set. The flight gets a
// score (lower is better) for how close to the target it got, and how slowly it was moving
// relative to it there: arriving, not flying past.

use rand::RngExt;
use rusty_neurons::code::fncts::Activation;
use rusty_neurons::code::network::Network;

use crate::code::math::Vec3;
use crate::code::physics::pull;
use crate::data::planets::AU;
use crate::probe::ephemeris::Ephemeris;

// Where the probes are trying to go: any body in ephemeris::BODIES.
pub const TARGET: &str = "Jupiter";

// The engine: an ion drive with a high specific impulse, so little thrust but a lot of change of
// velocity from its fuel; twice the thrust of NASA's Dawn (about 1e-4 m/s² and 11 km/s). At full
// throttle it adds 17 m/s a day and takes about 2.4 years to use all its fuel. Spiralling out
// from Earth's orbit to Jupiter's on the engine alone would take about 16.7 km/s (the difference
// in their orbital speeds), so it needs the launch, or a planet's help, as well.
pub const MAX_THRUST: f64 = 2.0e-4; // m/s²
pub const DELTA_V: f64 = 15_000.0; // m/s, the fuel, as the total change of velocity it can give
// The network decides every this many samples (hours).
pub const DECIDE_EVERY: usize = 6;
// The rocket that launches the probe leaves it this fast relative to Earth, at full throttle, once
// clear of Earth's gravity (m/s). The quickest way to Jupiter on the launch alone, a Hohmann
// transfer, would need 8.8 km/s; to Mars, 2.9 km/s.
pub const LAUNCH_SPEED: f64 = 4000.0;
// How far from Earth the launch leaves it (m): about where the Sun's pull takes over from Earth's.
// The climb out from Earth is skipped; it starts here with the launch speed, as if it had made it.
const ESCAPE_DISTANCE: f64 = 1.0e9;
// A probe that comes closer to a body's centre than this many of its radii is destroyed: for
// Jupiter, within one radius of its cloud tops, where its radiation is fiercest; for the others,
// near enough to grazing the surface (or the atmosphere) to count as hitting it. Flybys further
// out than this are what gravity assists are made of, and are integrated accurately however fast
// (see substeps).
pub const MIN_RADII: f64 = 2.0;
// Added to the score of a probe destroyed that way: as bad as getting 10 times less close.
const DESTROYED_PENALTY: f64 = 1.0;
// Steps are kept shorter than this fraction of sqrt(r³/μ) for the nearest pull (about 1/2π of an
// orbit at that distance), so close passes of a planet are integrated accurately.
const STEP_FRACTION: f64 = 0.02;
const MAX_SUBSTEPS: usize = 512;
// A flight ends early if the probe goes further from the Sun than this, or closer.
const FURTHEST: f64 = 10.0 * AU;
const NEAREST: f64 = 0.1 * AU;
// Speed relative to the target counts as the distance it would carry the probe in this time (s).
const SPEED_TIME: f64 = 10.0 * 86_400.0;
// The gap between the probe's and the target's distances from the Sun counts down to this (m):
// crossing the target's orbit somewhere else is a good start, but no substitute for meeting it.
const GAP_FLOOR: f64 = 1.0e10;
// Score per km/s of fuel used, and of launch speed. Small: much more and leaving at all costs
// more than the gap term (see fly) can win back, so the best probe goes nowhere.
const FUEL_WEIGHT: f64 = 0.005;

// The network's shape: what it is shown (see inputs), one hidden layer, and its answer.
pub const SHAPE: [usize; 3] = [16 + NEIGHBOUR_INPUTS * NEIGHBOURS, 16, 5];
// Besides the Sun and the target, the network is shown this many other bodies: whichever pull on
// the probe hardest at the time (see neighbours), each as NEIGHBOUR_INPUTS numbers.
const NEIGHBOURS: usize = 2;
const NEIGHBOUR_INPUTS: usize = 8;
const ACTIVATIONS: [Activation; 2] = [Activation::Tanh, Activation::Tanh];

// A network of the right shape, with random weights.
pub fn new_brain(rng: &mut impl RngExt) -> Network {
    Network::new(&SHAPE, &ACTIVATIONS, rng)
}

pub struct Flight {
    // The probe's position and velocity at every sample until the flight ended (only if asked for).
    pub path: Vec<(Vec3, Vec3)>,
    // Lower is better: see fly. Infinite if it never launched.
    pub score: f64,
    // At the best moment (the one with the lowest score): seconds since the start, distance from
    // the target (m) and speed relative to it (m/s).
    pub best_time: f64,
    pub distance: f64,
    pub speed: f64,
    // Fuel used (m/s), and the body it came too close to (see MIN_RADII), if it did.
    pub delta_v: f64,
    pub destroyed: Option<String>,
    // The smallest gap between its distance from the Sun and the target's, after launch (m).
    pub gap: f64,
    // When it launched (seconds since the start) and how fast, relative to Earth (m/s), if it did.
    pub launch: Option<(f64, f64)>,
}

// Flies the probe with `brain` at the controls for as long as the ephemeris lasts.
// keep_path records the path, for replaying; evolution doesn't need it.
pub fn fly(brain: &Network, ephemeris: &Ephemeris, keep_path: bool) -> Flight {
    let (sun, earth, target) = (ephemeris.body("Sun"), ephemeris.body("Earth"), ephemeris.body(TARGET));
    let (mut x, mut v) = ephemeris.sample(0)[earth];
    let mut flight = Flight { path: Vec::new(), score: f64::INFINITY, best_time: 0.0, distance: f64::INFINITY, speed: 0.0, delta_v: 0.0, destroyed: None, launch: None, gap: f64::INFINITY };
    let mut thrust = Vec3::ZERO;
    let mut bodies = vec![Vec3::ZERO; ephemeris.count()];
    let h = ephemeris.interval;
    let last = ephemeris.samples() - 1;

    for k in 0..=last {
        let states = ephemeris.sample(k);
        if flight.launch.is_none() {
            (x, v) = states[earth]; // still waiting on Earth
        }
        if keep_path {
            flight.path.push((x, v));
        }
        let (to_target, relative) = (states[target].0 - x, states[target].1 - v);
        // Only moments after launch count: waiting on Earth isn't getting anywhere, even when
        // Earth happens to pass close to the target.
        let score = score_at(to_target.norm(), relative.norm());
        if flight.launch.is_some() {
            if score < flight.score {
                (flight.score, flight.best_time, flight.distance, flight.speed) = (score, k as f64 * h, to_target.norm(), relative.norm());
            }
            let gap = ((x - states[sun].0).norm() - (states[target].0 - states[sun].0).norm()).abs();
            flight.gap = flight.gap.min(gap);
        }
        let from_sun = (x - states[sun].0).norm();
        if k == last || flight.destroyed.is_some() || !(NEAREST..FURTHEST).contains(&from_sun) {
            break;
        }
        if k % DECIDE_EVERY == 0 {
            let fuel_left = 1.0 - flight.delta_v / DELTA_V;
            let launched = flight.launch.is_some();
            let mut input = inputs(x - states[sun].0, v - states[sun].1, to_target, relative, fuel_left, k as f64 / last as f64, launched).to_vec();
            input.extend(neighbours(ephemeris, states, x, v, &[sun, target]));
            let output = brain.decide(&input);
            if !launched {
                if let Some((position, velocity, speed)) = launch(&output, states[earth]) {
                    (x, v) = (position, velocity);
                    flight.launch = Some((k as f64 * h, speed));
                }
            }
            thrust = steer(&output);
        }
        if flight.launch.is_none() {
            continue;
        }

        // The thrust is used over the whole sample; if that would take more fuel than is left,
        // it is turned down to what is left.
        let left = DELTA_V - flight.delta_v;
        if thrust.norm() * h > left {
            thrust = thrust * (left / (thrust.norm() * h));
        }
        flight.delta_v += thrust.norm() * h;

        // Kick-drift-kick (leapfrog) substeps, enough for the nearest pull.
        let n = substeps(ephemeris, states, x, v, h);
        let dt = h / n as f64;
        let t0 = k as f64 * h;
        let mut a = pull(&ephemeris.mu, &positions(states, &mut bodies), x) + thrust;
        for j in 0..n {
            v += a * (dt / 2.0);
            x += v * dt;
            ephemeris.positions_at(t0 + (j + 1) as f64 * dt, &mut bodies);
            a = pull(&ephemeris.mu, &bodies, x) + thrust;
            v += a * (dt / 2.0);
            if let Some(i) = too_close(ephemeris, &bodies, x) {
                flight.destroyed = Some(ephemeris.names[i].clone());
                break;
            }
        }
    }
    // The score: half the best moment's, and half for the smallest gap between the probe's and the
    // target's distances from the Sun, in the same digits of km. The gap rewards climbing towards
    // the target's orbit long before it gets anywhere near the target itself; meeting the target
    // makes both small. Then a little for the fuel and launch speed used.
    let launch_speed = flight.launch.map_or(0.0, |(_, speed)| speed);
    let gap_score = (flight.gap.max(GAP_FLOOR) / 1000.0).log10();
    flight.score = (flight.score + gap_score) / 2.0 + FUEL_WEIGHT * (flight.delta_v + launch_speed) / 1000.0;
    // Being destroyed costs as well, or the best moment before it would count as much as any.
    if flight.destroyed.is_some() {
        flight.score += DESTROYED_PENALTY;
    }
    flight
}

// How good a moment is, distance d (m) from the target at relative speed (m/s): the number of
// digits in the km it is from the target, or would be in SPEED_TIME at that speed. Far away the
// distance counts most, so matching the target's speed there is no use; close by, the speed:
// arriving, not flying past.
fn score_at(d: f64, speed: f64) -> f64 {
    ((d + speed * SPEED_TIME) / 1000.0).max(1.0).log10()
}

// What the network is shown, each scaled to be about 1 or less: the probe's position (AU) and
// velocity (per 30 km/s, about Earth's speed) relative to the Sun; the direction to the target,
// and how far (the digits in the distance in km, over 8); its velocity relative to the probe (per
// 10 km/s); the fractions of fuel and time left; and whether it has launched (1) or not (0).
fn inputs(x: Vec3, v: Vec3, to_target: Vec3, relative: Vec3, fuel_left: f64, time_gone: f64, launched: bool) -> [f64; 16] {
    let (x, v, dir, rel) = (x / AU, v / 30_000.0, to_target.normalised(), relative / 10_000.0);
    let digits = (to_target.norm() / 1000.0).max(1.0).log10() / 8.0;
    [x.x, x.y, x.z, v.x, v.y, v.z, dir.x, dir.y, dir.z, digits, rel.x, rel.y, rel.z, fuel_left, 1.0 - time_gone, if launched { 1.0 } else { 0.0 }]
}

// What the network is shown about the NEIGHBOURS bodies, other than those in `skip`, that pull
// hardest on the probe (at x, moving at v): near Earth, Earth and the Moon; near Jupiter, its
// moons. Strongest first, for each: the direction to it, and how far (as in inputs); its velocity
// relative to the probe (per 10 km/s); and how hard it pulls, as the digits of its pull in m/s²,
// plus 4, over 4 (0 for 1e-4 m/s², the most the engine can manage, and 1 for 1 m/s²).
// Waiting on Earth, the probe is at Earth's centre: Earth then counts as pulling as hard as it
// does at its surface, and as no direction.
fn neighbours(ephemeris: &Ephemeris, states: &[(Vec3, Vec3)], x: Vec3, v: Vec3, skip: &[usize]) -> Vec<f64> {
    let mut input = Vec::with_capacity(NEIGHBOURS * NEIGHBOUR_INPUTS);
    for i in strongest(ephemeris, states, x, skip) {
        let (to, relative) = (states[i].0 - x, (states[i].1 - v) / 10_000.0);
        let dir = if to.norm() > ephemeris.radius[i] { to.normalised() } else { Vec3::ZERO };
        let digits = (to.norm() / 1000.0).max(1.0).log10() / 8.0;
        let strength = (pull_of(ephemeris, states, x, i).log10() + 4.0) / 4.0;
        input.extend([dir.x, dir.y, dir.z, digits, relative.x, relative.y, relative.z, strength]);
    }
    input
}

// The NEIGHBOURS bodies, other than those in skip, that pull hardest on a probe at x, strongest first.
fn strongest(ephemeris: &Ephemeris, states: &[(Vec3, Vec3)], x: Vec3, skip: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..states.len()).filter(|i| !skip.contains(i)).collect();
    order.sort_by(|&a, &b| pull_of(ephemeris, states, x, b).total_cmp(&pull_of(ephemeris, states, x, a)));
    order.truncate(NEIGHBOURS);
    order
}

// How hard body i pulls on a probe at x (m/s²), as at its surface if x is inside it.
fn pull_of(ephemeris: &Ephemeris, states: &[(Vec3, Vec3)], x: Vec3, i: usize) -> f64 {
    ephemeris.mu[i] / (states[i].0 - x).norm_squared().max(ephemeris.radius[i].powi(2))
}

// Whether the network's answer, while waiting on Earth (position and velocity `earth`), is to
// launch: its fifth output above 0. If so, the probe's position and velocity just after, and its
// launch speed. It leaves in the direction of the first three outputs, at LAUNCH_SPEED times the
// throttle (the fourth output, as in steer).
fn launch(output: &[f64], (position, velocity): (Vec3, Vec3)) -> Option<(Vec3, Vec3, f64)> {
    let direction = Vec3::new(output[0], output[1], output[2]);
    if output[4] <= 0.0 || direction.norm() < 1e-9 {
        return None;
    }
    let (direction, speed) = (direction.normalised(), LAUNCH_SPEED * output[3].max(0.0));
    Some((position + direction * ESCAPE_DISTANCE, velocity + direction * speed, speed))
}

// The thrust (m/s²) the network's answer asks for: its first three outputs are a direction, and
// the fourth the throttle, where anything below 0 is off.
fn steer(output: &[f64]) -> Vec3 {
    let direction = Vec3::new(output[0], output[1], output[2]);
    let throttle = output[3].max(0.0);
    if throttle == 0.0 || direction.norm() < 1e-9 {
        return Vec3::ZERO;
    }
    direction.normalised() * (MAX_THRUST * throttle)
}

// Substeps needed in a sample of h seconds starting at x, for the pull of whichever body needs most.
// That is for the closest the probe (at x, moving at v) gets to each body during the sample,
// not where it starts: a fast probe can start an hour far from a planet and pass it within the
// hour. The closest is found along a straight line at their relative velocity; gravity only
// bends the path towards the body, which the margin in STEP_FRACTION covers. Nothing gets
// closer than MIN_RADII and survives, so that is as close as it needs to plan for.
fn substeps(ephemeris: &Ephemeris, states: &[(Vec3, Vec3)], x: Vec3, v: Vec3, h: f64) -> usize {
    let shortest = states
        .iter()
        .zip(ephemeris.mu.iter().zip(&ephemeris.radius))
        .map(|(&(p, u), (&mu, &radius))| {
            let (d, w) = (x - p, v - u);
            let t = (-d.dot(w) / w.norm_squared().max(1e-30)).clamp(0.0, h); // when it is closest
            let closest = (d + w * t).norm().max(MIN_RADII * radius);
            (closest.powi(3) / mu).sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    ((h / (STEP_FRACTION * shortest)).ceil() as usize).clamp(1, MAX_SUBSTEPS)
}

// The body that x is too close to (see MIN_RADII), if any; bodies are their positions.
fn too_close(ephemeris: &Ephemeris, bodies: &[Vec3], x: Vec3) -> Option<usize> {
    (0..bodies.len()).find(|&i| (x - bodies[i]).norm() < MIN_RADII * ephemeris.radius[i])
}

// Fills out with the positions in states, and returns it.
fn positions<'a>(states: &[(Vec3, Vec3)], out: &'a mut [Vec3]) -> &'a [Vec3] {
    for (x, &(p, _)) in out.iter_mut().zip(states) {
        *x = p;
    }
    out
}

#[cfg(test)]
mod tests;

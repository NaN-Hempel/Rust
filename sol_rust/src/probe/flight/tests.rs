// Tests for flight.rs.

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::*;
use crate::data::planets::get_planets;

fn ephemeris(days: f64) -> Ephemeris {
    Ephemeris::record(get_planets(), days * 86_400.0, 3600.0, 300.0)
}

// A network with every weight zero answers 0 to everything: throttle off, and no launch.
fn idle_brain() -> Network {
    let mut brain = new_brain(&mut StdRng::seed_from_u64(0));
    brain.set_params(&vec![0.0; brain.param_count()]);
    brain
}

// The direction of Earth's motion around the Sun at the start.
fn prograde(ephemeris: &Ephemeris) -> Vec3 {
    let states = ephemeris.sample(0);
    (states[ephemeris.body("Earth")].1 - states[ephemeris.body("Sun")].1).normalised()
}

// A network that launches at its first decision, along `direction`, at nearly full throttle.
fn launching_brain(direction: Vec3) -> Network {
    let mut brain = idle_brain();
    let bias = &mut brain.layers.last_mut().unwrap().bias;
    (bias[(0, 0)], bias[(1, 0)], bias[(2, 0)]) = (direction.x, direction.y, direction.z);
    bias[(3, 0)] = 3.0; // throttle tanh(3) = 0.995
    bias[(4, 0)] = 1.0; // launch
    brain
}

// A network that never says launch keeps the probe on Earth, using no fuel.
#[test]
fn idle_probe_waits_on_earth() {
    let ephemeris = ephemeris(30.0);
    let flight = fly(&idle_brain(), &ephemeris, true);
    assert!(flight.launch.is_none());
    assert_eq!(flight.delta_v, 0.0);
    let earth = ephemeris.body("Earth");
    for (k, state) in flight.path.iter().enumerate() {
        assert_eq!(*state, ephemeris.sample(k)[earth]);
    }
}

// Launching prograde, along Earth's motion, it leaves at once at LAUNCH_SPEED times the throttle,
// ESCAPE_DISTANCE from Earth, and climbs onto a wider orbit than Earth's.
#[test]
fn prograde_launch_raises_the_orbit() {
    let ephemeris = ephemeris(120.0);
    let flight = fly(&launching_brain(prograde(&ephemeris)), &ephemeris, true);
    let (time, speed) = flight.launch.expect("it should have launched");
    assert_eq!(time, 0.0);
    assert!((speed - LAUNCH_SPEED * 3.0f64.tanh()).abs() < 1e-9, "launched at {} m/s", speed);
    let (sun, earth) = (ephemeris.body("Sun"), ephemeris.body("Earth"));
    // path[0] is before the launch; path[1] an hour after.
    let (x, _) = flight.path[1];
    let from_earth = (x - ephemeris.sample(1)[earth].0).norm();
    assert!((from_earth - ESCAPE_DISTANCE).abs() < 0.05 * ESCAPE_DISTANCE, "{} km from Earth", from_earth / 1000.0);
    let last = flight.path.len() - 1;
    let (r_probe, r_earth) = ((flight.path[last].0 - ephemeris.sample(last)[sun].0).norm(), (ephemeris.sample(last)[earth].0 - ephemeris.sample(last)[sun].0).norm());
    println!("after 120 days: probe {:.4} AU from the Sun, Earth {:.4} AU", r_probe / AU, r_earth / AU);
    assert!(r_probe > r_earth, "a prograde launch should take it further from the Sun than Earth");
}

// After a launch, the engine runs on the same outputs (prograde, nearly full throttle) until the
// fuel runs out, then the probe coasts. Halving the step hardly changes where it is after a year:
// the integration is accurate. Measured: about 160 km.
#[test]
fn flight_is_step_independent() {
    let end = |interval: f64| {
        let ephemeris = Ephemeris::record(get_planets(), 365.25 * 86_400.0, interval, 300.0);
        let flight = fly(&launching_brain(prograde(&ephemeris)), &ephemeris, true);
        assert!(flight.destroyed.is_none());
        flight.path.last().unwrap().0
    };
    let off = (end(3600.0) - end(1800.0)).norm();
    println!("half the step moves it {:.0} km", off / 1000.0);
    assert!(off < STEP_TOLERANCE, "half the step moves it {} km", off / 1000.0);
}
const STEP_TOLERANCE: f64 = 1000.0e3; // m

// The same network always flies the same flight, to the last bit.
#[test]
fn flights_are_repeatable() {
    let ephemeris = ephemeris(60.0);
    let brain = new_brain(&mut StdRng::seed_from_u64(9));
    let (a, b) = (fly(&brain, &ephemeris, true), fly(&brain, &ephemeris, true));
    assert_eq!(a.path, b.path);
    assert_eq!(a.score, b.score);
}

// Fuel runs out: a network that thrusts all the time after launch uses exactly DELTA_V and no more.
#[test]
fn fuel_runs_out() {
    // Long enough to burn all the fuel at the throttle used, tanh(3), with some to spare.
    let ephemeris = ephemeris(1.1 * DELTA_V / (MAX_THRUST * 3.0f64.tanh()) / 86_400.0);
    let flight = fly(&launching_brain(prograde(&ephemeris)), &ephemeris, false);
    assert!((flight.delta_v - DELTA_V).abs() < 1e-6, "used {} m/s", flight.delta_v);
}

// Closer than MIN_RADII Jupiter radii is too close; just outside is fine.
#[test]
fn too_close_means_within_min_radii() {
    let ephemeris = ephemeris(1.0);
    let mut bodies = vec![Vec3::ZERO; ephemeris.count()];
    ephemeris.positions_at(0.0, &mut bodies);
    let jupiter = ephemeris.body("Jupiter");
    let at = |radii: f64| bodies[jupiter] + Vec3::new(0.0, 0.0, radii * ephemeris.radius[jupiter]);
    assert_eq!(too_close(&ephemeris, &bodies, at(MIN_RADII * 0.99)), Some(jupiter));
    assert_eq!(too_close(&ephemeris, &bodies, at(MIN_RADII * 1.01)), None);
}

// A probe heading straight for Jupiter fast enough to pass it within the hour gets the substeps
// for that close pass, not for where it starts; one heading away doesn't need them. Measured: 39 and 13.
#[test]
fn substeps_plan_for_the_closest_approach() {
    let ephemeris = ephemeris(1.0);
    let states = ephemeris.sample(0);
    let (p, u) = states[ephemeris.body("Jupiter")];
    let x = p + Vec3::new(3.0e8, 0.0, 0.0); // 300,000 km out
    let toward = substeps(&ephemeris, states, x, u + Vec3::new(-60_000.0, 0.0, 0.0), 3600.0);
    let away = substeps(&ephemeris, states, x, u + Vec3::new(60_000.0, 0.0, 0.0), 3600.0);
    println!("substeps: {} heading in, {} heading out", toward, away);
    assert!(toward > 2 * away, "{} substeps heading in, {} heading out", toward, away);
}

// Near Jupiter, the bodies pulling hardest (besides the Sun and Jupiter) are its moons; waiting on
// Earth, they are Earth and the Moon, with nothing undefined.
#[test]
fn neighbours_are_the_strongest_pulls() {
    let ephemeris = ephemeris(1.0);
    let states = ephemeris.sample(0);
    let (sun, jupiter, earth) = (ephemeris.body("Sun"), ephemeris.body("Jupiter"), ephemeris.body("Earth"));
    // Strongest pulls first; each neighbour's last input is its strength.
    let strengths = |input: &[f64]| (0..NEIGHBOURS).map(|n| input[n * NEIGHBOUR_INPUTS + NEIGHBOUR_INPUTS - 1]).collect::<Vec<_>>();
    let picked = |x: Vec3, skip: &[usize]| strongest(&ephemeris, states, x, skip).iter().map(|&i| ephemeris.names[i].clone()).collect::<Vec<_>>();

    let near_jupiter = states[jupiter].0 + Vec3::new(5.0e8, 0.0, 0.0);
    let input = neighbours(&ephemeris, states, near_jupiter, states[jupiter].1, &[sun, jupiter]);
    assert_eq!(input.len(), NEIGHBOURS * NEIGHBOUR_INPUTS);
    let names = picked(near_jupiter, &[sun, jupiter]);
    println!("near Jupiter: {:?}, strengths {:?}", names, strengths(&input));
    assert!(names.iter().all(|n| ["Io", "Europa", "Ganymede", "Callisto"].contains(&n.as_str())), "{:?}", names);
    assert!(strengths(&input)[0] >= strengths(&input)[1]);

    let (x, v) = states[earth];
    let input = neighbours(&ephemeris, states, x, v, &[sun, jupiter]);
    assert!(input.iter().all(|i| i.is_finite()), "{:?}", input);
    assert_eq!(picked(x, &[sun, jupiter]), ["Earth", "Moon"]);
}

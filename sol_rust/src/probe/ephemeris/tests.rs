// Tests for ephemeris.rs.

use super::*;
use crate::data::planets::get_planets;

// Its samples are the simulation's own states, at the right times.
#[test]
fn samples_match_the_simulation() {
    let ephemeris = Ephemeris::record(get_planets(), 10.0 * 86_400.0, 3600.0, 300.0);
    assert_eq!(ephemeris.samples(), 241);
    let mut simulation = sim::new_simulation(get_planets(), 0, 300.0);
    for _ in 0..12 * 100 {
        sim::update_simulation(&mut simulation, 300.0);
    }
    let mars = ephemeris.body("Mars");
    let truth = simulation.planets.iter().find(|p| p.name == "Mars").unwrap();
    assert_eq!(ephemeris.sample(100)[mars], (truth.position, truth.velocity));
}

// Halfway between the hourly samples the probe uses, the interpolated positions are close to
// half-hourly samples there. Io, on its 1.8-day orbit, is the hardest. Measured: 543 m.
#[test]
fn interpolation_is_accurate() {
    let half_hourly = Ephemeris::record(get_planets(), 30.0 * 86_400.0, 1800.0, 300.0);
    let hourly = Ephemeris::record(get_planets(), 30.0 * 86_400.0, 3600.0, 300.0);
    let mut positions = vec![Vec3::ZERO; hourly.count()];
    let mut worst = (0.0, "");
    for k in (1..half_hourly.samples() - 1).step_by(2) {
        hourly.positions_at(k as f64 * 1800.0, &mut positions);
        for (i, &(truth, _)) in half_hourly.sample(k).iter().enumerate() {
            let off = (positions[i] - truth).norm();
            if off > worst.0 {
                worst = (off, hourly.names[i].as_str());
            }
        }
    }
    println!("worst interpolation error: {:.0} m ({})", worst.0, worst.1);
    assert!(worst.0 < INTERPOLATION_TOLERANCE, "interpolation is {} m off for {}", worst.0, worst.1);
}
const INTERPOLATION_TOLERANCE: f64 = 1000.0;

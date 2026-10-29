// A spacecraft steered by a small neural network (from rusty_neurons), which learns to fly from
// Earth to another planet (flight::TARGET) by evolution rather than backpropagation: there is no "right answer" to compare
// each thrust with, only how well the whole flight went.
//
// ephemeris.rs records the planets once, flight.rs flies one probe through them, and evolve.rs
// breeds the networks that fly best.

pub mod ephemeris;
pub mod evolve;
pub mod flight;

const DAY: f64 = 86_400.0;
// How long a mission may take: long enough for a launch window (to Jupiter, one every 13 months),
// whatever the starting date, and for the slow engine to spiral out and slow down there, with
// time left to try more than one way of getting there (a direct transfer takes 2.7 years), or
// to come back round for another go.
pub const MISSION: f64 = 16.0 * 365.25 * DAY;
// Seconds between the ephemeris's samples, which is also the probe's step away from planets.
pub const SAMPLE: f64 = 3600.0;
// The step the planets are simulated with, as in main.rs (short enough for Phobos).
pub const SIM_STEP: f64 = DAY / 288.0;

use crate::code::math::Vec3;
use crate::data::planets::Planet;
use flight::Flight;

// A probe as a body called name for the window, and its track: where it is at each time along the flight's
// path, between samples by the same interpolation as ephemeris.rs. After the path ends (when the
// mission is over, or it was destroyed) it stays where it was.
pub fn body(name: &str, flight: Flight, interval: f64) -> (Planet, impl Fn(f64) -> (Vec3, Vec3) + use<>) {
    let path = flight.path;
    let (position, velocity) = path[0];
    let probe = Planet::new(name, "Probe", 1000.0, 0.005, 0.0, Some(0), position, velocity);
    let track = move |t: f64| {
        let f = (t / interval).max(0.0);
        let k = f.floor() as usize;
        if k + 1 >= path.len() {
            return *path.last().unwrap();
        }
        let ((p0, v0), (p1, v1), s) = (path[k], path[k + 1], f - k as f64);
        (ephemeris::hermite(p0, v0, p1, v1, interval, s), v0 * (1.0 - s) + v1 * s)
    };
    (probe, track)
}

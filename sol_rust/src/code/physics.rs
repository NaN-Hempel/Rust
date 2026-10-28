// Contains functions for handling the physics of the solar system simulation.

pub const G: f64 = 6.674_30e-11;

pub use crate::code::math::*;

// Fills out with the gravitational acceleration of every body due to all the others.
// mu holds each body's gravitational parameter, G times its mass. Working with accelerations
// rather than forces means bodies with (almost) no mass, like spacecraft, work too.
pub fn accelerations(mu: &[f64], positions: &[Vec3], out: &mut [Vec3]) {
    out.fill(Vec3::ZERO);
    let n = positions.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = positions[j] - positions[i];
            let inv_r3 = inverse_cube(d);
            out[i] += d * (mu[j] * inv_r3);
            out[j] -= d * (mu[i] * inv_r3);
        }
    }
}

// Acceleration at x due to bodies with gravitational parameters mu at positions.
// Used for test particles, which are pulled by the massive bodies but pull on nothing.
pub fn pull(mu: &[f64], positions: &[Vec3], x: Vec3) -> Vec3 {
    let mut a = Vec3::ZERO;
    for (&m, &p) in mu.iter().zip(positions) {
        let d = p - x;
        a += d * (m * inverse_cube(d));
    }
    a
}

// 1 / |d|³, so that a = mu * d / r³, with one square root and one division.
fn inverse_cube(d: Vec3) -> f64 {
    let r_squared = d.norm_squared();
    1.0 / (r_squared * r_squared.sqrt())
}

// Infers the position (m) and velocity (m/s) of a body orbiting central_mass (kg), from its
// semi-major axis (m), eccentricity and inclination (deg).
// The ascending node, argument of periapsis and anomaly aren't known, so they are taken as 0:
// the body starts at perihelion on the +x axis, its orbit tilted about the x-axis.
pub fn orbital_state(semi_major_axis: f64, eccentricity: f64, inclination_deg: f64, central_mass: f64) -> (Vec3, Vec3) {
    if semi_major_axis == 0.0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let elements = Orbit { semi_major_axis, eccentricity, inclination: deg_to_rad(inclination_deg), node: 0.0, periapsis: 0.0, mean_anomaly: 0.0 };
    elements.state(G * central_mass)
}

// The six classical orbital elements. Distances in metres, angles in radians.
pub struct Orbit {
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    // Tilt of the orbit from the x-y plane.
    pub inclination: f64,
    // Longitude of the ascending node: direction, from the x-axis, where the orbit rises through the x-y plane.
    pub node: f64,
    // Argument of periapsis: angle from the ascending node to the closest point, along the orbit.
    pub periapsis: f64,
    // Where the body is: the fraction of an orbit since periapsis, as an angle (2π is a whole orbit).
    pub mean_anomaly: f64,
}

impl Orbit {
    // Position and velocity relative to the central body, whose gravitational parameter is mu (G·M).
    pub fn state(&self, mu: f64) -> (Vec3, Vec3) {
        let (a, e) = (self.semi_major_axis, self.eccentricity);
        // Solve Kepler's equation, M = E - e·sin E, for the eccentric anomaly E (Newton's method).
        let mut ecc_anomaly = self.mean_anomaly;
        for _ in 0..30 {
            let step = (ecc_anomaly - e * ecc_anomaly.sin() - self.mean_anomaly) / (1.0 - e * ecc_anomaly.cos());
            ecc_anomaly -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        // The true anomaly: the actual angle from periapsis.
        let nu = 2.0 * ((1.0 + e).sqrt() * (ecc_anomaly / 2.0).sin()).atan2((1.0 - e).sqrt() * (ecc_anomaly / 2.0).cos());

        // In the orbit's own plane, with periapsis along +x.
        let semi_latus_rectum = a * (1.0 - e * e);
        let r = semi_latus_rectum / (1.0 + e * nu.cos());
        let speed = (mu / semi_latus_rectum).sqrt();
        let position = Vec3::new(r * nu.cos(), r * nu.sin(), 0.0);
        let velocity = Vec3::new(-speed * nu.sin(), speed * (e + nu.cos()), 0.0);

        // Turn by the argument of periapsis, tilt about the x-axis, then turn to the node.
        let (z_axis, x_axis) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
        let orient = |v: Vec3| v.rotated(z_axis, self.periapsis).rotated(x_axis, self.inclination).rotated(z_axis, self.node);
        (orient(position), orient(velocity))
    }
}

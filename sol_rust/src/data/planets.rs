// Contains data structures and functions related to planets in the solar system.

use serde::Deserialize;

use crate::code::math::{deg_to_rad, Vec3, PI};
use crate::code::physics::{orbital_state, Orbit};

pub const AU: f64 = 1.495_978_707e11; // metres
const EARTH_MASS: f64 = 5.972_17e24; // kg

// Initial values, embedded at compile time so the working directory doesn't matter.
const INIT_VALUES: &str = include_str!("init_vals/init_values.csv");

// A body in the simulation.
// mass in kg, radius and semi_major_axis in km, position in m, velocity in m/s.
// parent is the index of the body it orbits (the Sun for planets, the planet for moons);
// semi_major_axis is its orbit around that body, or 0 if it isn't in orbit.
#[derive(Clone)]
pub struct Planet {
    pub name: String,
    pub body_type: String,
    pub mass: f64,
    pub radius: f64,
    pub semi_major_axis: f64,
    pub parent: Option<usize>,
    pub position: Vec3,
    pub velocity: Vec3,
}

impl Planet {
    pub fn new(name: &str, body_type: &str, mass: f64, radius: f64, semi_major_axis: f64, parent: Option<usize>, position: Vec3, velocity: Vec3) -> Self {
        Self {
            name: name.to_string(),
            body_type: body_type.to_string(),
            mass,
            radius,
            semi_major_axis,
            parent,
            position,
            velocity,
        }
    }
}

// One row of init_values.csv; only the columns we use.
#[derive(Deserialize)]
struct Record {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Type")]
    body_type: String,
    #[serde(rename = "Parent")]
    parent: String,
    #[serde(rename = "Mean radius (km)")]
    radius: f64,
    #[serde(rename = "Mass (Earth = 1)")]
    mass: f64,
    #[serde(rename = "Semi-major axis (AU)")]
    semi_major_axis_au: Option<f64>,
    #[serde(rename = "Semi-major axis (10^3 km)")]
    semi_major_axis_km: Option<f64>,
    #[serde(rename = "Orbital velocity (km/s)")]
    orbital_velocity: Option<f64>,
    #[serde(rename = "Orbital eccentricity")]
    eccentricity: Option<f64>,
    #[serde(rename = "Orbital inclination (deg)")]
    inclination: Option<f64>,
}

// Returns every body that has enough data to place it, in the order they appear in the CSV.
// Bodies without it are skipped (JUICE and Europa Clipper).
pub fn get_planets() -> Vec<Planet> {
    let mut bodies: Vec<Planet> = Vec::new();
    for record in csv::Reader::from_reader(INIT_VALUES.as_bytes()).deserialize::<Record>() {
        let r = record.expect("invalid row in init_values.csv");
        if let Some((parent, a, position, velocity)) = initial_state(&r, &bodies) {
            bodies.push(Planet::new(&r.name, &r.body_type, r.mass * EARTH_MASS, r.radius, a / 1000.0, parent, position, velocity));
        }
    }
    bodies
}

// `count` made-up Trojan asteroids, as test particles: half around Jupiter's L4 point, 60° ahead
// of it in its orbit, and half around L5, 60° behind. Each starts with Jupiter's position and
// velocity relative to the Sun, turned by about 60° in Jupiter's orbital plane; the Sun, Jupiter
// and an L4 or L5 point form an equilateral triangle that keeps its shape as Jupiter orbits.
//
// Like the real swarms, they form loose clouds: dense near the point and thinning out, spread
// along the orbit, with a range of eccentricities and inclinations. Each property is drawn from a
// bell curve (with the standard deviation and cap below), using a seeded random number generator,
// so every run starts the same.
pub fn jupiter_trojans(bodies: &[Planet], count: usize) -> Vec<Planet> {
    let sun_index = find(bodies, "Sun").expect("no Sun in the data");
    let jupiter = &bodies[find(bodies, "Jupiter").expect("no Jupiter in the data")];
    let sun = &bodies[sun_index];
    let (r, v) = (jupiter.position - sun.position, jupiter.velocity - sun.velocity);
    let normal = r.cross(v).normalised();
    let mut rng = Rng(0x7_2_0_7_A_5);

    (0..count)
        .map(|k| {
            let (point, side) = if k % 2 == 0 { ("L4", 1.0) } else { ("L5", -1.0) };
            // Along the orbit (deg from the point), distance from the Sun (fraction), eccentricity
            // and inclination (deg). Distance varies least: it sets how far a Trojan swings, and
            // too much would knock it out of the cloud.
            let along = rng.normal(9.0, 25.0);
            let scale = 1.0 + rng.normal(0.003, 0.01);
            let eccentricity = rng.normal(0.04, 0.12);
            let inclination = deg_to_rad(rng.normal(10.0, 30.0).abs());
            let node = 2.0 * PI * rng.uniform();

            // Turn within Jupiter's orbital plane to the point, plus the offset along the orbit.
            let angle = deg_to_rad(side * 60.0 + along);
            let (r, v) = (r.rotated(normal, angle), v.rotated(normal, angle));
            // Some velocity towards or away from the Sun makes the orbit eccentric. Keeping the
            // speed the same keeps the semi-major axis (and so the orbital period) the same.
            let v = (v + r.normalised() * (eccentricity * v.norm())).normalised() * v.norm();
            // Tilt the orbit about a line through the Sun in Jupiter's plane, in a random direction,
            // so the Trojans start at different heights above and below the plane.
            let axis = r.normalised().rotated(normal, node);
            let (r, v) = (r.rotated(axis, inclination), v.rotated(axis, inclination));
            // Moving the particle out by `scale` and slowing it by sqrt(scale) keeps its orbit the same shape.
            let (r, v) = (r * scale, v / scale.sqrt());
            Planet::new(
                &format!("Trojan {} {}", point, k / 2 + 1),
                "Jupiter trojan",
                1.0e12,
                1.0,
                jupiter.semi_major_axis * scale,
                Some(sun_index),
                sun.position + r,
                sun.velocity + v,
            )
        })
        .collect()
}

// `count` made-up Kuiper belt objects, as test particles orbiting the Sun, spread over their orbits
// in random directions. Roughly following the real belt:
// - about 70% classical objects, 42 to 48 AU from the Sun: two thirds of them "cold", on nearly
//   flat, nearly circular orbits, the rest "hot", more tilted and eccentric;
// - about 30% plutinos, around 39.4 AU, where they go round twice for every three orbits of
//   Neptune, like Pluto.
// Eccentricities are kept low enough that none comes closer to the Sun than about 31 AU, outside
// Neptune's orbit. (Real plutinos can come closer because their timing keeps them away from
// Neptune; these aren't placed with that timing.)
pub fn kuiper_belt(bodies: &[Planet], count: usize) -> Vec<Planet> {
    let sun_index = find(bodies, "Sun").expect("no Sun in the data");
    let sun = &bodies[sun_index];
    let mu = crate::code::physics::G * sun.mass;
    let mut rng = Rng(0x4B_55_49_50_45_52);

    (0..count)
        .map(|k| {
            let (kind, a, eccentricity, inclination) = if rng.uniform() < 0.3 {
                ("Plutino", 39.4 + rng.normal(0.2, 0.5), rng.normal(0.07, 0.2).abs(), rng.normal(10.0, 25.0).abs())
            } else if rng.uniform() < 2.0 / 3.0 {
                ("Classical", 42.0 + 6.0 * rng.uniform(), rng.normal(0.03, 0.1).abs(), rng.normal(2.0, 6.0).abs())
            } else {
                ("Classical", 42.0 + 6.0 * rng.uniform(), rng.normal(0.08, 0.25).abs(), rng.normal(12.0, 30.0).abs())
            };
            let orbit = Orbit {
                semi_major_axis: a * AU,
                eccentricity,
                inclination: deg_to_rad(inclination),
                node: 2.0 * PI * rng.uniform(),
                periapsis: 2.0 * PI * rng.uniform(),
                mean_anomaly: 2.0 * PI * rng.uniform(),
            };
            let (r, v) = orbit.state(mu);
            Planet::new(
                &format!("{} KBO {}", kind, k + 1),
                "Kuiper belt object",
                1.0e12,
                1.0,
                a * AU / 1000.0,
                Some(sun_index),
                sun.position + r,
                sun.velocity + v,
            )
        })
        .collect()
}

// A small seeded pseudo-random number generator (SplitMix64), so made-up bodies are the same every run.
struct Rng(u64);

impl Rng {
    // A number in [0, 1).
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    // A number from a bell curve around 0 with standard deviation sigma, capped at ±cap
    // (Box-Muller transform).
    fn normal(&mut self, sigma: f64, cap: f64) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        (z * sigma).clamp(-cap, cap)
    }
}

fn find(bodies: &[Planet], name: &str) -> Option<usize> {
    bodies.iter().position(|b| b.name == name)
}

// Works out a body's parent, semi-major axis (m), position and velocity from its row.
// Its parent (and, for JWST, Earth) must come earlier in the CSV.
fn initial_state(r: &Record, bodies: &[Planet]) -> Option<(Option<usize>, f64, Vec3, Vec3)> {
    let au = r.semi_major_axis_au.unwrap_or(0.0) * AU;
    let km = r.semi_major_axis_km.unwrap_or(0.0) * 1.0e6; // 10^3 km to m
    let sun = find(bodies, "Sun");

    match r.body_type.as_str() {
        "Star" => Some((None, 0.0, Vec3::ZERO, Vec3::ZERO)),

        // Approximation: at the Sun-Earth L2 point, straight out from the Sun beyond Earth,
        // moving with Earth. L2 is unstable, so it drifts away after a while.
        "Space telescope" => {
            let earth = &bodies[find(bodies, "Earth")?];
            if km == 0.0 {
                return None;
            }
            let scale = 1.0 + km / earth.position.norm();
            Some((sun, 0.0, earth.position * scale, earth.velocity * scale))
        }

        // Approximation: at its distance from the Sun (AU column), flying straight outwards
        // at its speed (orbital velocity column), in its rough real direction.
        "Interstellar probe" => {
            let direction = probe_direction(&r.name)?;
            let speed = r.orbital_velocity.filter(|&v| v > 0.0)? * 1000.0;
            if au == 0.0 {
                return None;
            }
            Some((sun, 0.0, direction * au, direction * speed))
        }

        // Anything else orbits the Sun if it has a semi-major axis in AU (planets, dwarf planets,
        // asteroids, comets, Zoozve), or else its parent at its semi-major axis in km (moons).
        _ => {
            let (central, a) = if au > 0.0 {
                (sun?, au)
            } else if km > 0.0 {
                (find(bodies, &r.parent)?, km)
            } else {
                return None;
            };
            let c = &bodies[central];
            // The two bodies orbit each other, so both masses count (this matters for the Moon).
            let mass = c.mass + r.mass * EARTH_MASS;
            let (position, velocity) = orbital_state(a, r.eccentricity.unwrap_or(0.0), r.inclination.unwrap_or(0.0), mass);
            Some((Some(central), a, c.position + position, c.velocity + velocity))
        }
    }
}

// Rough ecliptic longitude and latitude (deg) of each probe's direction from the Sun.
// The planets' starting angles are made up anyway, so these only need to be roughly right.
fn probe_direction(name: &str) -> Option<Vec3> {
    let (longitude, latitude) = match name {
        "Voyager 1" => (255.0, 35.0),
        "Voyager 2" => (290.0, -33.0),
        "New Horizons" => (285.0, 2.0),
        _ => return None,
    };
    let (lon, lat) = (deg_to_rad(longitude), deg_to_rad(latitude));
    Some(Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_body<'a>(bodies: &'a [Planet], name: &str) -> &'a Planet {
        bodies.iter().find(|p| p.name == name).unwrap()
    }

    #[test]
    fn sun_is_at_rest_at_origin() {
        let bodies = get_planets();
        let sun = find_body(&bodies, "Sun");
        assert_eq!((sun.position, sun.velocity), (Vec3::ZERO, Vec3::ZERO));
    }

    #[test]
    fn earth_starts_at_perihelion() {
        let bodies = get_planets();
        let earth = find_body(&bodies, "Earth");
        let (r, v) = (earth.position.norm(), earth.velocity.norm());
        assert!((r / AU - 0.983).abs() < 0.001, "r = {} AU", r / AU);
        assert!((v / 1000.0 - 30.3).abs() < 0.1, "v = {} km/s", v / 1000.0);
    }

    #[test]
    fn mercury_orbit_is_inclined() {
        let bodies = get_planets();
        let mercury = find_body(&bodies, "Mercury");
        let v = mercury.velocity.norm();
        assert!((mercury.velocity.z - v * deg_to_rad(7.0).sin()).abs() < 1e-6);
        assert_eq!(mercury.position.z, 0.0);
    }

    #[test]
    fn moon_orbits_earth() {
        let bodies = get_planets();
        let (earth, moon) = (find_body(&bodies, "Earth"), find_body(&bodies, "Moon"));
        assert_eq!(moon.parent, find(&bodies, "Earth"));
        // Perigee distance and speed relative to Earth
        let r = (moon.position - earth.position).norm() / 1000.0;
        let v = (moon.velocity - earth.velocity).norm() / 1000.0;
        assert!((r - 363_300.0).abs() < 1000.0, "r = {} km", r);
        assert!((v - 1.08).abs() < 0.02, "v = {} km/s", v);
    }

    #[test]
    fn every_loaded_body_has_a_finite_state() {
        for b in get_planets() {
            let values = [b.position.x, b.position.y, b.position.z, b.velocity.x, b.velocity.y, b.velocity.z];
            assert!(values.iter().all(|x| x.is_finite()), "{} has a non-finite state", b.name);
        }
    }

    #[test]
    fn spacecraft_are_placed() {
        let bodies = get_planets();
        // JWST about 1.5 million km beyond Earth, Voyager 1 at 174.8 AU moving at 17 km/s.
        let (earth, jwst) = (find_body(&bodies, "Earth"), find_body(&bodies, "JWST"));
        assert!(((jwst.position - earth.position).norm() / 1.0e9 - 1.5).abs() < 0.01);
        let voyager = find_body(&bodies, "Voyager 1");
        assert!((voyager.position.norm() / AU - 174.8).abs() < 0.01);
        assert!((voyager.velocity.norm() / 1000.0 - 17.0).abs() < 0.01);
    }

    // Every Kuiper belt object starts with the speed its orbit needs (vis-viva equation), and
    // none comes closer to the Sun than Neptune's orbit (30 AU).
    #[test]
    fn kuiper_belt_orbits_are_consistent() {
        let bodies = get_planets();
        let sun = &bodies[0];
        let mu = crate::code::physics::G * sun.mass;
        for p in kuiper_belt(&bodies, 600) {
            let (r, v) = (p.position - sun.position, p.velocity - sun.velocity);
            let a = p.semi_major_axis * 1000.0;
            let expected = mu * (2.0 / r.norm() - 1.0 / a);
            assert!((v.norm_squared() / expected - 1.0).abs() < 1e-12, "{} has the wrong speed", p.name);
            // Eccentricity vector, pointing to perihelion, with length e.
            let e = (r * (v.norm_squared() - mu / r.norm()) - v * r.dot(v)) / mu;
            let perihelion = a * (1.0 - e.norm()) / AU;
            assert!(perihelion > 31.0, "{} comes within {:.1} AU of the Sun", p.name, perihelion);
        }
    }

    // A body placed a quarter of an orbit past periapsis on a tilted, eccentric orbit is where
    // Kepler's equation says, at the right height for its inclination.
    #[test]
    fn orbit_state_follows_keplers_equation() {
        let mu = 1.327e20;
        let orbit = Orbit { semi_major_axis: AU, eccentricity: 0.3, inclination: deg_to_rad(20.0), node: 0.0, periapsis: 0.0, mean_anomaly: PI / 2.0 };
        let (r, v) = orbit.state(mu);
        // Solve M = E - e sin E by repeated substitution, independently of Orbit::state.
        let mut e_anomaly = PI / 2.0;
        for _ in 0..100 {
            e_anomaly = PI / 2.0 + 0.3 * e_anomaly.sin();
        }
        let distance = AU * (1.0 - 0.3 * e_anomaly.cos());
        assert!((r.norm() / distance - 1.0).abs() < 1e-12);
        assert!((v.norm_squared() / (mu * (2.0 / r.norm() - 1.0 / AU)) - 1.0).abs() < 1e-12);
        // Node along +x, so the height above the plane is r·sin(i)·sin(true anomaly), with y·tan(i) = z.
        assert!((r.z - r.y * deg_to_rad(20.0).tan()).abs() < 1.0);
        assert!(r.z > 0.0);
    }
}

// Draws Saturn's rings: the C, B and A rings, with the Cassini Division between B and A.
// They lie in Saturn's equator, tilted from its orbit by its axial tilt in a direction fixed in
// space, so over Saturn's 29-year orbit they open and close as seen from the Sun, like the real ones.
// Planet dots are drawn bigger than to scale, so the rings are sized to Saturn's dot, keeping
// their real proportions to the planet.

use crate::code::math::{deg_to_rad, Vec3};
use crate::data::planets::Planet;
use crate::renderer::draw::{Canvas, Projector};

// Saturn's axial tilt in degrees, as in solar.csv.
const SATURN_TILT: f64 = 26.73;
// The rings as (inner radius, outer radius, opacity), radii in Saturn radii.
// The gap between the B and A rings is the Cassini Division.
const RINGS: [(f64, f64, f64); 3] = [
    (1.239, 1.527, 0.25), // C ring: faint
    (1.527, 1.951, 0.8),  // B ring: the brightest
    (2.027, 2.269, 0.55), // A ring
];
const RING_COLOUR: u32 = 0x00_D8_C8_A0;

// Which half of the rings to draw: the half beyond the planet goes behind its dot, the half on
// the viewer's side in front of it.
#[derive(Clone, Copy, PartialEq)]
pub enum Half {
    Far,
    Near,
}

// The direction of Saturn's north pole: the normal of its orbit around the Sun, tilted by
// SATURN_TILT about a direction fixed in space (the x-axis, flattened into the orbit's plane).
pub fn saturn_pole(saturn: &Planet, sun: &Planet) -> Vec3 {
    let normal = (saturn.position - sun.position).cross(saturn.velocity - sun.velocity).normalised();
    let x_axis = Vec3::new(1.0, 0.0, 0.0);
    let axis = (x_axis - normal * x_axis.dot(normal)).normalised();
    normal.rotated(axis, deg_to_rad(SATURN_TILT))
}

// Draws one half of the rings around a planet at `centre` whose pole points along `pole`, with
// the planet `radius` pixels across the middle.
pub fn draw(canvas: &mut Canvas, projector: &Projector, centre: Vec3, pole: Vec3, radius: f64, half: Half) {
    // Two directions at right angles in the ring plane, each one planet radius long.
    let reference = if pole.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let a = pole.cross(reference).normalised();
    let b = pole.cross(a);
    let length = radius / projector.scale();
    // How each moves on screen (pixels) and in depth.
    let (cx, cy, cd) = projector.project(centre);
    let moved = |v: Vec3| {
        let (x, y, d) = projector.project(centre + v * length);
        (x - cx, y - cy, d - cd)
    };
    let ((ax, ay, ad), (bx, by, bd)) = (moved(a), moved(b));

    // A point u·a + v·b in the ring plane lands at u·(ax, ay) + v·(bx, by) on screen; inverting
    // that gives, for each pixel, where in the ring plane it shows. Seen edge-on there is nothing
    // to draw.
    let det = ax * by - bx * ay;
    if det.abs() < 1e-3 * radius * radius {
        return;
    }
    let reach = RINGS[RINGS.len() - 1].1 * radius;
    let (x0, x1) = (((cx - reach).floor() as i64).max(0), ((cx + reach).ceil() as i64).min(canvas.width - 1));
    let (y0, y1) = (((cy - reach).floor() as i64).max(0), ((cy + reach).ceil() as i64).min(canvas.height - 1));
    for py in y0..=y1 {
        for px in x0..=x1 {
            let (dx, dy) = (px as f64 + 0.5 - cx, py as f64 + 0.5 - cy);
            let u = (dx * by - dy * bx) / det;
            let v = (ax * dy - ay * dx) / det;
            // Nearer the viewer than the planet's centre, or further.
            let near = u * ad + v * bd > 0.0;
            if near != (half == Half::Near) {
                continue;
            }
            let distance = (u * u + v * v).sqrt();
            if let Some(&(_, _, opacity)) = RINGS.iter().find(|&&(inner, outer, _)| distance >= inner && distance < outer) {
                canvas.blend(px, py, RING_COLOUR, opacity);
            }
        }
    }
}

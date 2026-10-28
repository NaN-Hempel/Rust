// Contains the camera, a canvas with drawing primitives, and the function that draws a frame.
// Pixels are u32 in 0x00RRGGBB format, row by row from the top left.

use crate::code::math::{Vec3, PI};
use crate::data::planets::Planet;
use crate::renderer::rings;
use crate::renderer::text;
use crate::renderer::trails::Trails;

const BACKGROUND: u32 = 0x00_00_00_08;
const LABEL_BACKGROUND: u32 = 0x00_20_20_28;
const LABEL_TEXT: u32 = 0x00_FF_FF_FF;

// Orthographic camera looking at target.
// yaw spins the view around the z-axis; pitch tilts it, from 0 (top-down) to ±90° (edge-on).
pub struct Camera {
    pub target: Vec3,
    // Distance in metres from the centre of the window to the edge of its shorter side.
    pub view_radius: f64,
    pub yaw: f64,
    pub pitch: f64,
}

impl Camera {
    // A top-down camera on the origin, zoomed out far enough to show every planet (and the
    // Kuiper belt, if there is one).
    pub fn fit(planets: &[Planet]) -> Self {
        let furthest = planets
            .iter()
            .filter(|p| matches!(p.body_type.as_str(), "Planet" | "Gas giant" | "Ice giant" | "Kuiper belt object"))
            .map(|p| p.position.norm())
            .fold(0.0, f64::max);
        Camera { target: Vec3::ZERO, view_radius: furthest * 1.1, yaw: 0.0, pitch: 0.0 }
    }

    // Zooms to show the moons of body i, if it has any. Following it is up to the caller,
    // which sets target every frame.
    pub fn focus(&mut self, i: usize, planets: &[Planet]) {
        let furthest_moon = planets
            .iter()
            .filter(|p| p.parent == Some(i) && planets[i].parent.is_some())
            .map(|p| (p.position - planets[i].position).norm())
            .fold(0.0, f64::max);
        if furthest_moon > 0.0 {
            self.view_radius = furthest_moon * 1.3;
        }
    }

    // factor > 1 zooms in, < 1 zooms out.
    pub fn zoom(&mut self, factor: f64) {
        self.view_radius /= factor;
    }

    pub fn rotate(&mut self, yaw: f64, pitch: f64) {
        self.yaw += yaw;
        self.pitch = (self.pitch + pitch).clamp(-PI / 2.0, PI / 2.0);
    }

    // The projection for a window of this size, which converts positions to pixels.
    // Build it once per frame and use it for every point.
    pub fn projector(&self, width: i64, height: i64) -> Projector {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Projector {
            target: self.target,
            sin_yaw,
            cos_yaw,
            sin_pitch,
            cos_pitch,
            scale: width.min(height) as f64 / 2.0 / self.view_radius,
            centre: (width as f64 / 2.0, height as f64 / 2.0),
        }
    }
}

// A camera's projection with its sines, cosines and scale worked out in advance.
pub struct Projector {
    target: Vec3,
    sin_yaw: f64,
    cos_yaw: f64,
    sin_pitch: f64,
    cos_pitch: f64,
    scale: f64,
    centre: (f64, f64),
}

impl Projector {
    // Converts a position in metres to pixel coordinates (y points down) and a depth,
    // which is larger the closer the point is to the viewer.
    pub fn project(&self, p: Vec3) -> (f64, f64, f64) {
        let p = p - self.target;
        let x = p.x * self.cos_yaw - p.y * self.sin_yaw;
        let y = p.x * self.sin_yaw + p.y * self.cos_yaw;
        let up = y * self.cos_pitch + p.z * self.sin_pitch;
        let depth = p.z * self.cos_pitch - y * self.sin_pitch;
        (self.centre.0 + x * self.scale, self.centre.1 - up * self.scale, depth)
    }

    // Pixels per metre.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

// A trail in screen coordinates, with the box around it for quick hover checks.
struct Path {
    points: Vec<(f64, f64)>,
    min: (f64, f64),
    max: (f64, f64),
}

impl Path {
    // Projects points, leaving out any closer than a pixel to the last one kept,
    // except the final point so the trail still reaches its body.
    fn project(points: impl Iterator<Item = Vec3>, projector: &Projector) -> Path {
        let mut path = Path { points: Vec::new(), min: (f64::INFINITY, f64::INFINITY), max: (f64::NEG_INFINITY, f64::NEG_INFINITY) };
        let mut skipped = None;
        for p in points {
            let (x, y, _) = projector.project(p);
            if let Some(&(lx, ly)) = path.points.last() {
                if (x - lx).abs() < 1.0 && (y - ly).abs() < 1.0 {
                    skipped = Some((x, y));
                    continue;
                }
            }
            path.add((x, y));
            skipped = None;
        }
        if let Some(point) = skipped {
            path.add(point);
        }
        path
    }

    fn add(&mut self, (x, y): (f64, f64)) {
        self.points.push((x, y));
        self.min = (self.min.0.min(x), self.min.1.min(y));
        self.max = (self.max.0.max(x), self.max.1.max(y));
    }

    // Whether (x, y) is within margin of the path's box.
    fn near(&self, (x, y): (f64, f64), margin: f64) -> bool {
        x >= self.min.0 - margin && x <= self.max.0 + margin && y >= self.min.1 - margin && y <= self.max.1 + margin
    }
}

// A pixel buffer with its size. Drawing outside it is clipped.
pub struct Canvas<'a> {
    pixels: &'a mut [u32],
    pub width: i64,
    pub height: i64,
}

impl<'a> Canvas<'a> {
    pub fn new(pixels: &'a mut [u32], width: u32, height: u32) -> Self {
        Canvas { pixels, width: width as i64, height: height as i64 }
    }

    pub fn clear(&mut self, colour: u32) {
        self.pixels.fill(colour);
    }

    pub fn fill_rect(&mut self, x: i64, y: i64, w: i64, h: i64, colour: u32) {
        for py in y.max(0)..(y + h).min(self.height) {
            for px in x.max(0)..(x + w).min(self.width) {
                self.pixels[(py * self.width + px) as usize] = colour;
            }
        }
    }

    // A solid, anti-aliased filled circle.
    pub fn fill_circle(&mut self, cx: f64, cy: f64, r: f64, colour: u32) {
        self.blend_circle(cx, cy, r, colour, 1.0);
    }

    // A see-through filled circle: like fill_circle(), but blended with what is already there.
    // Anti-aliased: each pixel is blended in proportion to how much of it the circle covers,
    // measured at 4×4 points inside it. Without this, a dot smaller than a couple of pixels
    // flickers between 1 and 4 pixels as it moves across the pixel grid.
    pub fn blend_circle(&mut self, cx: f64, cy: f64, r: f64, colour: u32, alpha: f64) {
        const SAMPLES: i64 = 4;
        for py in (cy - r).floor() as i64..=(cy + r).ceil() as i64 {
            for px in (cx - r).floor() as i64..=(cx + r).ceil() as i64 {
                let mut inside = 0;
                for sy in 0..SAMPLES {
                    for sx in 0..SAMPLES {
                        let dx = px as f64 + (sx as f64 + 0.5) / SAMPLES as f64 - cx;
                        let dy = py as f64 + (sy as f64 + 0.5) / SAMPLES as f64 - cy;
                        if dx * dx + dy * dy <= r * r {
                            inside += 1;
                        }
                    }
                }
                if inside > 0 {
                    self.blend(px, py, colour, alpha * inside as f64 / (SAMPLES * SAMPLES) as f64);
                }
            }
        }
    }

    // A see-through rectangle: like fill_rect(), but blended with what is already there.
    pub fn blend_rect(&mut self, x: i64, y: i64, w: i64, h: i64, colour: u32, alpha: f64) {
        for py in y..y + h {
            for px in x..x + w {
                self.blend(px, py, colour, alpha);
            }
        }
    }

    // Mixes colour into the pixel at (x, y): alpha 0 leaves it unchanged, 1 replaces it.
    pub fn blend(&mut self, x: i64, y: i64, colour: u32, alpha: f64) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            let pixel = &mut self.pixels[(y * self.width + x) as usize];
            let mix = |shift: u32| {
                let (old, new) = (((*pixel >> shift) & 0xFF) as f64, ((colour >> shift) & 0xFF) as f64);
                ((old + (new - old) * alpha) as u32) << shift
            };
            *pixel = mix(16) | mix(8) | mix(0);
        }
    }

    // A see-through line: like line(), but blended with what is already there, so overlapping
    // lines show through each other. The start point is left out, so lines joined end to end
    // don't blend their shared point twice.
    pub fn blend_line(&mut self, (x0, y0): (f64, f64), (x1, y1): (f64, f64), colour: u32, alpha: f64) {
        let (w, h) = (self.width as f64, self.height as f64);
        if (x0 < 0.0 && x1 < 0.0) || (y0 < 0.0 && y1 < 0.0) || (x0 >= w && x1 >= w) || (y0 >= h && y1 >= h) {
            return;
        }
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil().min(4.0 * (w + h)).max(1.0);
        for i in 1..=steps as i64 {
            let t = i as f64 / steps;
            self.blend((x0 + (x1 - x0) * t) as i64, (y0 + (y1 - y0) * t) as i64, colour, alpha);
        }
    }
}

// Draws trails, bodies (far to near) and, if the cursor is over a body or its orbit, its name.
// Moons are left out while they would be drawn inside their planet's dot.
// ui_scale is the window's pixels per point, so dots and text keep their size on HiDPI screens.
// Returns the index of the body under the cursor, if any.
pub fn draw_frame(canvas: &mut Canvas, camera: &Camera, planets: &[Planet], trails: &Trails, cursor: Option<(f64, f64)>, ui_scale: i64) -> Option<usize> {
    canvas.clear(BACKGROUND);
    let projector = camera.projector(canvas.width, canvas.height);
    let slack = 4.0 * ui_scale as f64;

    // (screen x, screen y, depth, dot radius) of every body.
    let screen: Vec<(f64, f64, f64, f64)> = planets
        .iter()
        .map(|p| {
            let (x, y, depth) = projector.project(p.position);
            let radius = if is_swarm(p) { SWARM_DOT_RADIUS } else { dot_radius(p.radius) };
            // Zoomed in far enough, a body is drawn at its true size, if that is bigger.
            let true_radius = p.radius * 1000.0 * projector.scale();
            (x, y, depth, (radius * ui_scale as f64).max(true_radius))
        })
        .collect();
    let visible: Vec<bool> = planets
        .iter()
        .enumerate()
        .map(|(i, p)| match p.parent {
            // A moon: its parent itself has a parent.
            Some(parent) if planets[parent].parent.is_some() => {
                let (x, y, _, _) = screen[i];
                let (px, py, _, pr) = screen[parent];
                (x - px).hypot(y - py) > pr + 2.0 * ui_scale as f64
            }
            _ => true,
        })
        .collect();

    // Each visible trail in screen coordinates.
    let paths: Vec<Path> = (0..planets.len())
        .map(|i| {
            let points = trails.points(i, planets).take(if visible[i] { usize::MAX } else { 0 });
            Path::project(points, &projector)
        })
        .collect();

    // Visible bodies, sorted far to near.
    let mut order: Vec<usize> = (0..planets.len()).filter(|&i| visible[i]).collect();
    order.sort_by(|&a, &b| screen[a].2.total_cmp(&screen[b].2));

    // The body under the cursor (the biggest, if dots overlap, with a few pixels of slack),
    // or else the body whose orbit passes closest to the cursor, if one is within the slack.
    let hovered = cursor.and_then(|(cx, cy)| {
        let on_dot = order
            .iter()
            .copied()
            .filter(|&i| { let (x, y, _, r) = screen[i]; (x - cx).powi(2) + (y - cy).powi(2) <= (r + slack).powi(2) })
            .max_by(|&a, &b| planets[a].radius.total_cmp(&planets[b].radius));
        on_dot.or_else(|| {
            paths
                .iter()
                .enumerate()
                .filter(|(_, path)| path.near((cx, cy), slack))
                .map(|(i, path)| (i, distance_to_path((cx, cy), &path.points)))
                .filter(|&(_, d)| d <= slack)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        })
    });

    for (i, path) in paths.iter().enumerate() {
        // Fade in from faint (oldest) to stronger (newest); tails fade in from nothing. The hovered
        // orbit is drawn stronger, and at full opacity even if its body's trails are normally fainter.
        let (boost, opacity) = if hovered == Some(i) { (2.0, 1.0) } else { (1.0, trail_opacity(&planets[i])) };
        let faintest = if has_tail(&planets[i]) { 0.0 } else { 0.1 };
        let c = colour(&planets[i]);
        for (k, pair) in path.points.windows(2).enumerate() {
            let age = k as f64 / path.points.len() as f64;
            canvas.blend_line(pair[0], pair[1], c, ((faintest + (0.5 - faintest) * age) * boost).min(1.0) * opacity);
        }
    }

    // Saturn's rings, if Saturn is there: the far half behind its dot, the near half in front.
    let saturn = planets.iter().position(|p| p.name == "Saturn");
    let pole = saturn.and_then(|i| planets[i].parent.map(|sun| rings::saturn_pole(&planets[i], &planets[sun])));
    for &i in &order {
        let (x, y, _, r) = screen[i];
        let ring_pole = pole.filter(|_| Some(i) == saturn);
        if let Some(pole) = ring_pole {
            rings::draw(canvas, &projector, planets[i].position, pole, r, rings::Half::Far);
        }
        if is_swarm(&planets[i]) {
            canvas.blend_circle(x, y, r, colour(&planets[i]), SWARM_DOT_OPACITY);
        } else {
            canvas.fill_circle(x, y, r, colour(&planets[i]));
        }
        if let Some(pole) = ring_pole {
            rings::draw(canvas, &projector, planets[i].position, pole, r, rings::Half::Near);
        }
    }

    if let Some(i) = hovered {
        let (x, y, _, r) = screen[i];
        draw_label(canvas, x + r + slack, y, &planets[i].name, ui_scale);
    }
    hovered
}

// Shortest distance in pixels from p to a polyline.
fn distance_to_path(p: (f64, f64), path: &[(f64, f64)]) -> f64 {
    path.windows(2).map(|s| distance_to_segment(p, s[0], s[1])).fold(f64::INFINITY, f64::min)
}

fn distance_to_segment((px, py): (f64, f64), (ax, ay): (f64, f64), (bx, by): (f64, f64)) -> f64 {
    let (dx, dy) = (bx - ax, by - ay);
    let length_squared = dx * dx + dy * dy;
    // Fraction along the segment of the point closest to p, clamped to the segment's ends.
    let t = if length_squared == 0.0 { 0.0 } else { (((px - ax) * dx + (py - ay) * dy) / length_squared).clamp(0.0, 1.0) };
    ((px - ax - t * dx).powi(2) + (py - ay - t * dy).powi(2)).sqrt()
}

// A panel's space around its text, and the height of each line including the gap below it, in pixels.
fn panel_padding(scale: i64) -> i64 {
    6 * scale
}

fn panel_line_height(scale: i64) -> i64 {
    text::height(scale) + 4 * scale
}

// Width and height in pixels of a panel showing these lines.
pub fn panel_size<S: AsRef<str>>(lines: &[S], scale: i64) -> (i64, i64) {
    let padding = panel_padding(scale);
    let width = lines.iter().map(|l| text::width(l.as_ref(), scale)).max().unwrap_or(0) + 2 * padding;
    (width, lines.len() as i64 * panel_line_height(scale) - 4 * scale + 2 * padding)
}

// Lines of text in a box with its top-left corner at (x, y).
// Returns the box as (x, y, width, height) in pixels.
pub fn draw_panel<S: AsRef<str>>(canvas: &mut Canvas, x: i64, y: i64, lines: &[S], scale: i64) -> (i64, i64, i64, i64) {
    let padding = panel_padding(scale);
    let (width, height) = panel_size(lines, scale);
    canvas.fill_rect(x, y, width, height, LABEL_BACKGROUND);
    for (i, line) in lines.iter().enumerate() {
        text::draw(canvas, x + padding, y + padding + i as i64 * panel_line_height(scale), line.as_ref(), LABEL_TEXT, scale);
    }
    (x, y, width, height)
}

// Where columns from..to of a line of a panel drawn at (x, y) are, as (x, y, width, height) in
// pixels: the whole height of the line, for clicking on.
pub fn panel_text_rect(x: i64, y: i64, line: usize, from: usize, to: usize, scale: i64) -> (i64, i64, i64, i64) {
    let glyph = text::width(" ", scale);
    let top = y + panel_padding(scale) + line as i64 * panel_line_height(scale) - 2 * scale;
    (x + panel_padding(scale) + from as i64 * glyph, top, (to - from) as i64 * glyph, panel_line_height(scale))
}

// Name in a box, its left edge at x and vertically centred on y.
fn draw_label(canvas: &mut Canvas, x: f64, y: f64, name: &str, scale: i64) {
    let padding = 3 * scale;
    let (text_w, text_h) = (text::width(name, scale), text::height(scale));
    let (left, top) = (x as i64, y as i64 - text_h / 2 - padding);
    canvas.fill_rect(left, top, text_w + 2 * padding, text_h + 2 * padding, LABEL_BACKGROUND);
    text::draw(canvas, left + padding, top + padding, name, LABEL_TEXT, scale);
}

// Dot size in points, growing with the log of the radius (km) so the Sun doesn't swamp everything.
fn dot_radius(radius_km: f64) -> f64 {
    ((radius_km.log10() - 2.5) * 2.0).max(1.5)
}

// The planets and the Moon have their own colours; everything else is coloured by type.
fn colour(planet: &Planet) -> u32 {
    match planet.name.as_str() {
        "Sun" => return 0x00_FF_D2_4A,
        "Mercury" => return 0x00_A0_A0_A0,
        "Venus" => return 0x00_E8_D4_A0,
        "Earth" => return 0x00_4A_90_E2,
        "Mars" => return 0x00_D0_5A_3A,
        "Jupiter" => return 0x00_D8_B0_80,
        "Saturn" => return 0x00_E8_D0_8A,
        "Uranus" => return 0x00_9F_E0_E8,
        "Neptune" => return 0x00_4C_6E_E8,
        "Moon" => return 0x00_D8_D8_D8,
        _ => {}
    }
    let body_type = planet.body_type.as_str();
    if body_type.starts_with("Moon") {
        0x00_B8_B0_A8
    } else if body_type == "Dwarf planet" {
        0x00_C8_A8_88
    } else if body_type == "Jupiter trojan" {
        0x00_E0_A0_60
    } else if body_type == "Kuiper belt object" {
        0x00_9A_B4_D0
    } else if body_type.contains("Asteroid") || body_type.starts_with("Quasi-satellite") {
        0x00_98_88_78
    } else if body_type.contains("comet") {
        0x00_9A_D8_FF
    } else {
        0x00_7C_E8_8A // spacecraft
    }
}

// Made-up bodies that come in swarms of hundreds (Jupiter trojans, the Kuiper belt). They are
// drawn smaller and fainter, dots and trails, or they would hide everything else.
fn is_swarm(planet: &Planet) -> bool {
    matches!(planet.body_type.as_str(), "Jupiter trojan" | "Kuiper belt object")
}

// Swarm bodies' dot radius in points, and how solid their dots and trails are (0 to 1).
const SWARM_DOT_RADIUS: f64 = 0.8;
const SWARM_DOT_OPACITY: f64 = 0.55;
const SWARM_TRAIL_OPACITY: f64 = 0.2;

// Bodies whose trail is a short tail (see trails.rs) rather than a whole orbit.
fn has_tail(planet: &Planet) -> bool {
    planet.body_type == "Jupiter trojan"
}

// How solid a body's trail is drawn, from 0 (invisible) to 1. Tails are short enough to be drawn
// as strongly as a planet's trail.
fn trail_opacity(planet: &Planet) -> f64 {
    if has_tail(planet) { 1.0 } else if is_swarm(planet) { SWARM_TRAIL_OPACITY } else { 1.0 }
}

#[cfg(test)]
mod benches {
    use super::*;
    use crate::code::sim;
    use crate::data::planets::get_planets;

    // Run with: cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_render() {
        let dt = 300.0;
        let mut s = sim::new_simulation(get_planets(), 0, dt);
        let mut trails = Trails::new(&s.planets);
        for _ in 0..365 * 288 {
            sim::update_simulation(&mut s, dt);
            trails.record(&s.planets, dt);
        }
        s.sync();
        let jupiter = s.planets.iter().position(|p| p.name == "Jupiter").unwrap();
        let (w, h) = (1600u32, 1000u32);
        let mut pixels = vec![0u32; (w * h) as usize];
        for view in ["system", "Jupiter"] {
            let mut camera = Camera::fit(&s.planets);
            if view == "Jupiter" {
                camera.target = s.planets[jupiter].position;
                camera.focus(jupiter, &s.planets);
            }
            // The cursor sits in empty space, so every orbit is checked for hovering.
            let cursor = Some((w as f64 * 0.9, h as f64 * 0.9));
            let frames = 100;
            let start = std::time::Instant::now();
            for _ in 0..frames {
                let mut canvas = Canvas::new(&mut pixels, w, h);
                draw_frame(&mut canvas, &camera, &s.planets, &trails, cursor, 1);
            }
            println!("render ({}): {:.2} ms per frame", view, start.elapsed().as_secs_f64() * 1000.0 / frames as f64);
        }
    }
}

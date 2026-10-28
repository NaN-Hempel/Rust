// Opens a window and runs the simulation in it, drawing one frame about every 1/60 s.
// Controls:
//   drag with the left mouse button: rotate the view
//   scroll wheel, or + / -: zoom
//   click a body: follow it (and zoom to its moons, if it has any)
//   up and down arrows (or the ⏩ and ⏪ buttons): speed time up or slow it down
//   right and left arrows: jump forwards or back in time, by a second's worth at the current speed
//   Space (or the ‖ button): pause
//   R: reset the view    Escape: quit
//   H, or clicking the list: show or hide the list of controls
//   P, or clicking the panel: show or hide the performance panel
//   T, or clicking the panel (not its buttons): show or hide the time panel
//   hover over a body or its orbit to see its name

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::{Duration, Instant};

use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::code::sim::{self, Simulation};
use crate::renderer::draw::{self, Camera, Canvas};
use crate::renderer::stats::Stats;
use crate::renderer::time::{Button, TimeControl, BUTTONS};
use crate::renderer::trails::Trails;

const FRAME_TIME: Duration = Duration::from_micros(16_667);
// Longest the physics may take in one frame. At high speeds it can need more; the frame then
// does as many steps as fit in this time, so the window stays responsive, and the time panel
// shows that the simulation is running slower than asked.
const PHYSICS_BUDGET: Duration = Duration::from_millis(12);
// Radians of rotation per pixel dragged.
const DRAG_SPEED: f64 = 0.005;
// The simulation is saved this often (in frames), and before each jump, so it can jump back:
// time can't run backwards, so a jump back restores the newest snapshot from before the moment
// it jumps to, and works forward from there. The oldest snapshots are dropped beyond the limit,
// which is as far back as it can jump.
const SNAPSHOT_EVERY: u32 = 60;
const MAX_SNAPSHOTS: usize = 20;
// A press and release closer together than this (in pixels) is a click rather than a drag.
const CLICK_DISTANCE: f64 = 4.0;

// Shown in the top-left corner; H toggles between the two.
const CONTROLS: &[&str] = &[
    "Controls",
    "",
    "Drag               rotate",
    "Scroll / + -       zoom",
    "Click a body       follow it",
    "Hover              show name",
    "R                  reset view",
    "Down / Up  or ⏪ ⏩  slower / faster",
    "Left / Right       jump back / ahead",
    "Space  or  ‖       pause",
    "H / click here     hide controls",
    "P / click panel    performance",
    "T / click panel    time panel",
    "Esc                quit",
];
// Shown instead of each panel while it is hidden; clicking one shows the panel again.
const CONTROLS_HINT: &[&str] = &["H: controls"];
const STATS_HINT: &[&str] = &["P: performance"];
const TIME_HINT: &[&str] = &["T: time"];
// Distance of the panels from the window's corners, in points.
const PANEL_MARGIN: i64 = 10;
// How much the button under the cursor is lightened.
const BUTTON_HIGHLIGHT: f64 = 0.15;

// A rectangle in window pixels: (x, y, width, height).
type Rect = (i64, i64, i64, i64);

struct App {
    simulation: Simulation,
    camera: Camera,
    trails: Trails,
    time: TimeControl,
    // Simulated seconds so far, and the steps taken in the last frame that wasn't paused: (number, length).
    sim_time: f64,
    last_steps: (usize, f64),
    // Saved (simulated time, simulation) pairs, oldest first, and frames since the last was saved.
    snapshots: Vec<(f64, Simulation)>,
    frames_since_snapshot: u32,
    show_controls: bool,
    // Where the panels (or their hints) and the time panel's buttons were drawn last frame.
    panel: Rect,
    stats_panel: Rect,
    time_panel: Rect,
    show_time: bool,
    buttons: [(Button, Rect); 3],
    // Frame timings, shown in the performance panel in the top-right corner.
    stats: Stats,
    show_stats: bool,
    next_frame: Instant,
    // Cursor position in window pixels, or None when it's outside the window.
    cursor: Option<(f64, f64)>,
    // Where the left button went down, while it is held.
    press: Option<(f64, f64)>,
    // The body under the cursor in the last frame, and the body the camera follows.
    hovered: Option<usize>,
    following: Option<usize>,
    // Created in resumed(), once the event loop is running.
    window: Option<Rc<Window>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
}

impl App {
    fn redraw(&mut self) {
        let physics_start = Instant::now();
        if !self.time.paused {
            let (steps, dt) = self.time.steps(FRAME_TIME.as_secs_f64(), self.simulation.dt);
            let mut taken = 0;
            while taken < steps {
                sim::update_simulation(&mut self.simulation, dt);
                self.trails.record(&self.simulation.planets, dt);
                taken += 1;
                if physics_start.elapsed() > PHYSICS_BUDGET {
                    break;
                }
            }
            self.time.taken(taken, steps);
            self.sim_time += dt * taken as f64;
            self.last_steps = (taken, dt);
            self.simulation.sync();
            self.frames_since_snapshot += 1;
            if self.frames_since_snapshot >= SNAPSHOT_EVERY {
                self.snapshot();
            }
        }
        if let Some(i) = self.following {
            self.camera.target = self.simulation.planets[i].position;
        }
        let physics = physics_start.elapsed();

        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return;
        };
        let size = window.inner_size();
        let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return; // minimised
        };
        surface.resize(width, height).expect("failed to resize surface");
        let ui_scale = window.scale_factor().round().max(1.0) as i64;

        let render_start = Instant::now();
        let mut buffer = surface.buffer_mut().expect("failed to get pixel buffer");
        let mut canvas = Canvas::new(&mut buffer, size.width, size.height);
        self.hovered = draw::draw_frame(&mut canvas, &self.camera, &self.simulation.planets, &self.trails, self.cursor, ui_scale);
        let margin = PANEL_MARGIN * ui_scale;
        let panel = if self.show_controls { CONTROLS } else { CONTROLS_HINT };
        self.panel = draw::draw_panel(&mut canvas, margin, margin, panel, ui_scale);
        let lines = if self.show_stats { stats_lines(&self.stats, self.last_steps, &self.simulation) } else { STATS_HINT.iter().map(|l| l.to_string()).collect() };
        let x = canvas.width - draw::panel_size(&lines, ui_scale).0 - margin;
        self.stats_panel = draw::draw_panel(&mut canvas, x, margin, &lines, ui_scale);

        // The time panel, in the bottom-left corner, with its buttons on the last line.
        let lines = if self.show_time { self.time.panel_lines(self.sim_time).to_vec() } else { TIME_HINT.iter().map(|l| l.to_string()).collect() };
        let y = canvas.height - draw::panel_size(&lines, ui_scale).1 - margin;
        self.time_panel = draw::draw_panel(&mut canvas, margin, y, &lines, ui_scale);
        self.buttons = BUTTONS.map(|(button, from, to)| {
            let rect = if self.show_time { draw::panel_text_rect(margin, y, lines.len() - 1, from, to, ui_scale) } else { (0, 0, 0, 0) };
            (button, rect)
        });
        if let Some((_, (x, y, w, h))) = self.cursor.and_then(|c| button_at(&self.buttons, c)) {
            canvas.blend_rect(x, y, w, h, 0x00_FF_FF_FF, BUTTON_HIGHLIGHT);
        }
        let render = render_start.elapsed();

        let present_start = Instant::now();
        buffer.present().expect("failed to present frame");
        self.stats.record(physics, render, present_start.elapsed());
    }

    fn snapshot(&mut self) {
        if self.snapshots.len() == MAX_SNAPSHOTS {
            self.snapshots.remove(0);
        }
        self.snapshots.push((self.sim_time, self.simulation.clone()));
        self.frames_since_snapshot = 0;
    }

    // Takes `steps` steps of dt seconds without drawing them, recording trails as it goes.
    fn run_unseen(&mut self, steps: usize, dt: f64) {
        for _ in 0..steps {
            sim::update_simulation(&mut self.simulation, dt);
            self.trails.record(&self.simulation.planets, dt);
        }
        self.sim_time += dt * steps as f64;
        self.simulation.sync();
    }

    // Jumps a second's worth ahead at the current speed, computing every step.
    fn jump_ahead(&mut self) {
        self.snapshot();
        let (steps, dt) = self.time.jump(self.simulation.dt);
        self.run_unseen(steps, dt);
    }

    // Jumps a second's worth back at the current speed, or to the oldest snapshot if that is
    // further back than the snapshots go.
    fn jump_back(&mut self) {
        let (steps, dt) = self.time.jump(self.simulation.dt);
        let target = (self.sim_time - steps as f64 * dt).max(0.0);
        // Snapshots after the target belong to the future being left; drop them.
        let Some(k) = self.snapshots.iter().rposition(|&(t, _)| t <= target).or((!self.snapshots.is_empty()).then_some(0)) else {
            return;
        };
        self.snapshots.truncate(k + 1);
        let (time, simulation) = self.snapshots[k].clone();
        self.simulation = simulation;
        self.sim_time = time;
        self.trails.rewind(time);
        let remaining = target - time;
        if remaining > 0.0 {
            let steps = (remaining / self.simulation.dt).ceil();
            self.run_unseen(steps as usize, remaining / steps);
        }
        self.simulation.sync();
    }

    fn reset_view(&mut self) {
        self.camera = Camera::fit(&self.simulation.planets);
        self.following = None;
    }

    // A click: on a time panel button, presses it; on a body, follows it; on a panel (or the hint
    // left by a hidden one), with no body under the cursor, shows or hides it.
    fn click(&mut self, point: (f64, f64)) {
        if let Some((button, _)) = button_at(&self.buttons, point) {
            self.time.press(button);
        } else if let Some(i) = self.hovered {
            self.following = Some(i);
            self.camera.focus(i, &self.simulation.planets);
        } else if inside(self.panel, point) {
            self.show_controls = !self.show_controls;
        } else if inside(self.stats_panel, point) {
            self.show_stats = !self.show_stats;
        } else if inside(self.time_panel, point) {
            self.show_time = !self.show_time;
        }
    }
}

// The time panel button at a point in the window, if there is one, and where it is.
fn button_at(buttons: &[(Button, Rect)], point: (f64, f64)) -> Option<(Button, Rect)> {
    buttons.iter().copied().find(|&(_, rect)| inside(rect, point))
}

// A count with commas between thousands: 1633 as "1,633".
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn inside((x, y, w, h): Rect, (px, py): (f64, f64)) -> bool {
    px >= x as f64 && px < (x + w) as f64 && py >= y as f64 && py < (y + h) as f64
}

// The lines of the performance panel.
fn stats_lines(stats: &Stats, (steps, dt): (usize, f64), simulation: &Simulation) -> Vec<String> {
    let budget = FRAME_TIME.as_secs_f64() * 1000.0;
    let step = if dt >= 1.0 { format!("{:.0} s", dt) } else { format!("{:.1} ms", dt * 1000.0) };
    vec![
        format!("Performance ({} build)", if cfg!(debug_assertions) { "debug" } else { "release" }),
        String::new(),
        format!("FPS        {:>7.1}", stats.fps),
        format!("Frame      {:>7.2} ms  ({:.0}% of {:.1} ms)", stats.frame_ms(), stats.frame_ms() / budget * 100.0, budget),
        format!("  Physics  {:>7.2} ms", stats.physics_ms),
        format!("  Render   {:>7.2} ms", stats.render_ms),
        format!("  Present  {:>7.2} ms", stats.present_ms),
        String::new(),
        format!("Steps      {:>7} per frame, {} each", steps, step),
        format!("Bodies     {:>7}  ({} massive)", simulation.planets.len(), simulation.massive_count()),
    ]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // The body count tells apart windows started with different numbers of test particles.
        let title = format!("Solar system ({} bodies)", thousands(self.simulation.planets.len()));
        let attributes = Window::default_attributes().with_title(title);
        let window = Rc::new(event_loop.create_window(attributes).expect("failed to create window"));
        let context = Context::new(window.clone()).expect("failed to create graphics context");
        self.surface = Some(Surface::new(&context, window.clone()).expect("failed to create surface"));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => self.time.press(Button::Pause),
                    Key::Named(NamedKey::ArrowUp) => self.time.press(Button::Faster),
                    Key::Named(NamedKey::ArrowDown) => self.time.press(Button::Slower),
                    Key::Named(NamedKey::ArrowRight) => self.jump_ahead(),
                    Key::Named(NamedKey::ArrowLeft) => self.jump_back(),
                    Key::Character("+") | Key::Character("=") => self.camera.zoom(1.25),
                    Key::Character("-") => self.camera.zoom(0.8),
                    Key::Character("r") | Key::Character("R") => self.reset_view(),
                    Key::Character("h") | Key::Character("H") => self.show_controls = !self.show_controls,
                    Key::Character("p") | Key::Character("P") => self.show_stats = !self.show_stats,
                    Key::Character("t") | Key::Character("T") => self.show_time = !self.show_time,
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => match state {
                ElementState::Pressed => self.press = self.cursor,
                ElementState::Released => {
                    if let (Some((px, py)), Some((cx, cy))) = (self.press, self.cursor) {
                        if (cx - px).hypot(cy - py) < CLICK_DISTANCE {
                            self.click((cx, cy));
                        }
                    }
                    self.press = None;
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some(_), Some((x, y))) = (self.press, self.cursor) {
                    self.camera.rotate((position.x - x) * DRAG_SPEED, (position.y - y) * DRAG_SPEED);
                }
                self.cursor = Some((position.x, position.y));
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                self.camera.zoom(1.1_f64.powf(lines));
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    // Asks for a redraw once per frame, and sleeps until then.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if now >= self.next_frame {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            self.next_frame = now + FRAME_TIME;
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

// Opens the window and runs until it is closed. The simulation's dt is the longest step it
// takes; slower speeds take shorter ones.
pub fn run(simulation: Simulation) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App {
        camera: Camera::fit(&simulation.planets),
        trails: Trails::new(&simulation.planets),
        time: TimeControl::new(),
        sim_time: 0.0,
        last_steps: (0, simulation.dt),
        snapshots: vec![(0.0, simulation.clone())],
        frames_since_snapshot: 0,
        simulation,
        show_controls: true,
        panel: (0, 0, 0, 0),
        stats_panel: (0, 0, 0, 0),
        time_panel: (0, 0, 0, 0),
        show_time: true,
        buttons: BUTTONS.map(|(button, _, _)| (button, (0, 0, 0, 0))),
        stats: Stats::new(),
        show_stats: true,
        next_frame: Instant::now(),
        cursor: None,
        press: None,
        hovered: None,
        following: None,
        window: None,
        surface: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

// Measures how long each part of a frame takes, averaged over short intervals so the numbers
// are readable while they change.

use std::time::{Duration, Instant};

// How often the averages are updated.
const INTERVAL: Duration = Duration::from_millis(500);

pub struct Stats {
    interval_start: Instant,
    frames: u32,
    physics: Duration,
    render: Duration,
    present: Duration,
    // Averages over the last finished interval; times in milliseconds.
    pub fps: f64,
    pub physics_ms: f64,
    pub render_ms: f64,
    pub present_ms: f64,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            interval_start: Instant::now(),
            frames: 0,
            physics: Duration::ZERO,
            render: Duration::ZERO,
            present: Duration::ZERO,
            fps: 0.0,
            physics_ms: 0.0,
            render_ms: 0.0,
            present_ms: 0.0,
        }
    }

    // Adds one frame's timings, and updates the averages once an interval has passed.
    pub fn record(&mut self, physics: Duration, render: Duration, present: Duration) {
        self.frames += 1;
        self.physics += physics;
        self.render += render;
        self.present += present;

        let elapsed = self.interval_start.elapsed();
        if elapsed >= INTERVAL {
            let frames = self.frames as f64;
            self.fps = frames / elapsed.as_secs_f64();
            self.physics_ms = self.physics.as_secs_f64() * 1000.0 / frames;
            self.render_ms = self.render.as_secs_f64() * 1000.0 / frames;
            self.present_ms = self.present.as_secs_f64() * 1000.0 / frames;
            *self = Stats { fps: self.fps, physics_ms: self.physics_ms, render_ms: self.render_ms, present_ms: self.present_ms, ..Stats::new() };
        }
    }

    pub fn frame_ms(&self) -> f64 {
        self.physics_ms + self.render_ms + self.present_ms
    }
}

// Controls how fast simulated time runs: a list of speeds to step through, and pausing.
// Every frame, the window advances the simulation by the speed times the frame's length, in
// steps no longer than the simulation's own dt: slow speeds take one short step per frame, so
// motion stays smooth, and fast speeds take many full-length steps.
// The time panel shows the simulated date: as if the simulation began when the program started,
// in local time.

use chrono::{DateTime, Local, TimeDelta};

const MINUTE: f64 = 60.0;
const HOUR: f64 = 3600.0;
const DAY: f64 = 86_400.0;
const YEAR: f64 = 365.25 * DAY;
const MONTH: f64 = YEAR / 12.0;

// Simulated seconds per real second, and how each speed is shown.
const SPEEDS: &[(f64, &str)] = &[
    (1.0, "real time"),
    (MINUTE, "1 minute/s"),
    (10.0 * MINUTE, "10 minutes/s"),
    (HOUR, "1 hour/s"),
    (6.0 * HOUR, "6 hours/s"),
    (DAY, "1 day/s"),
    (7.0 * DAY, "1 week/s"),
    (MONTH, "1 month/s"),
    (2.0 * MONTH, "2 months/s"),
    (6.0 * MONTH, "6 months/s"),
    (YEAR, "1 year/s"),
    (2.0 * YEAR, "2 years/s"),
];
// About a day per frame at 60 frames a second.
const DEFAULT_SPEED: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Button {
    Slower,
    Pause,
    Faster,
}

// The buttons on the time panel's last line, with the columns of text each covers (from, to):
// three each, with the symbol in the middle.
pub const BUTTONS: [(Button, usize, usize); 3] = [(Button::Slower, 0, 3), (Button::Pause, 3, 6), (Button::Faster, 6, 9)];

pub struct TimeControl {
    // Index into SPEEDS.
    speed: usize,
    pub paused: bool,
    // The fraction of each frame's steps the physics has had time for, smoothed over a few
    // frames: 1 normally, less when the speed is more than the computer can keep up with.
    keeping_up: f64,
    // The real date and time the simulation counts on from.
    start: DateTime<Local>,
}

impl TimeControl {
    // Starts the simulated calendar now.
    pub fn new() -> Self {
        TimeControl { speed: DEFAULT_SPEED, paused: false, keeping_up: 1.0, start: Local::now() }
    }

    pub fn press(&mut self, button: Button) {
        match button {
            Button::Slower => self.set_speed(self.speed.saturating_sub(1)),
            Button::Faster => self.set_speed((self.speed + 1).min(SPEEDS.len() - 1)),
            Button::Pause => self.paused = !self.paused,
        }
    }

    fn set_speed(&mut self, speed: usize) {
        self.speed = speed;
        self.keeping_up = 1.0;
    }

    // The simulated time for a frame of `frame` real seconds, as a number of steps and their
    // length, no step longer than max_dt.
    pub fn steps(&self, frame: f64, max_dt: f64) -> (usize, f64) {
        let total = SPEEDS[self.speed].0 * frame;
        let steps = (total / max_dt).ceil().max(1.0);
        (steps as usize, total / steps)
    }

    // A jump through time: one second's worth at the current speed, as a number of steps and
    // their length, no step longer than max_dt.
    pub fn jump(&self, max_dt: f64) -> (usize, f64) {
        self.steps(1.0, max_dt)
    }

    // Tells the control how many of a frame's steps were taken, out of how many were wanted.
    pub fn taken(&mut self, taken: usize, wanted: usize) {
        self.keeping_up += 0.2 * (taken as f64 / wanted as f64 - self.keeping_up);
    }

    // The date and time `elapsed` simulated seconds after the start, in local time (with summer
    // time where it applies on that date).
    fn date(&self, elapsed: f64) -> String {
        TimeDelta::try_milliseconds((elapsed * 1000.0) as i64)
            .and_then(|delta| self.start.checked_add_signed(delta))
            .map_or_else(|| "beyond the calendar".to_string(), |date| date.with_timezone(&Local).format("%-d %b %Y  %H:%M:%S").to_string())
    }

    // The time panel's text, given the simulated seconds so far. The buttons are on the last line.
    pub fn panel_lines(&self, elapsed: f64) -> [String; 2] {
        let status = if self.paused {
            ", paused".to_string()
        } else if self.keeping_up < 0.95 {
            format!(", running at {}/s", duration(SPEEDS[self.speed].0 * self.keeping_up))
        } else {
            String::new()
        };
        [
            format!("{}{}", self.date(elapsed), status),
            format!(" ⏪  {}  ⏩   {}", if self.paused { '▶' } else { '‖' }, SPEEDS[self.speed].1),
        ]
    }
}

// A length of simulated time, in a unit that suits it.
fn duration(seconds: f64) -> String {
    if seconds < 2.0 * MINUTE {
        format!("{:.0} seconds", seconds)
    } else if seconds < 2.0 * HOUR {
        format!("{:.0} minutes", seconds / MINUTE)
    } else if seconds < 2.0 * DAY {
        format!("{:.1} hours", seconds / HOUR)
    } else if seconds < YEAR {
        format!("{:.1} days", seconds / DAY)
    } else {
        format!("{:.2} years", seconds / YEAR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // At every speed, a frame's steps add up to the right amount of time, and none is longer
    // than the simulation allows.
    #[test]
    fn frames_split_into_short_enough_steps() {
        let mut time = TimeControl::new();
        for speed in 0..SPEEDS.len() {
            time.set_speed(speed);
            let (steps, dt) = time.steps(1.0 / 60.0, 300.0);
            assert!(dt <= 300.0 * (1.0 + 1e-12), "{}: steps of {} s", SPEEDS[speed].1, dt);
            assert!((steps as f64 * dt / (SPEEDS[speed].0 / 60.0) - 1.0).abs() < 1e-12, "{}: wrong total", SPEEDS[speed].1);
        }
    }

    #[test]
    fn speed_stops_at_both_ends() {
        let mut time = TimeControl::new();
        for _ in 0..50 {
            time.press(Button::Faster);
        }
        assert_eq!(time.speed, SPEEDS.len() - 1);
        for _ in 0..50 {
            time.press(Button::Slower);
        }
        assert_eq!(time.speed, 0);
    }

    // The date counts on from the start by the simulated time.
    #[test]
    fn date_counts_on_from_the_start() {
        use chrono::TimeZone;
        let start = Local.with_ymd_and_hms(2026, 6, 10, 12, 0, 0).unwrap();
        let time = TimeControl { start, ..TimeControl::new() };
        assert_eq!(time.date(0.0), "10 Jun 2026  12:00:00");
        assert_eq!(time.date(DAY + 90.0), "11 Jun 2026  12:01:30");
        assert_eq!(time.date(30.0 * DAY), "10 Jul 2026  12:00:00");
    }

    // The default is about a day per frame, the speed the simulation ran at before it had controls.
    #[test]
    fn default_speed_is_about_a_day_per_frame() {
        let per_frame = SPEEDS[DEFAULT_SPEED].0 / 60.0;
        assert!((per_frame / DAY - 1.0).abs() < 0.02, "{} days per frame", per_frame / DAY);
    }
}

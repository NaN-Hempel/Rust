// Solar System Simulation by Nanne Hempel, as a learning project.

mod code;
mod data;
mod renderer;
use code::sim;
use data::planets;

const DAY: f64 = 86_400.0; // seconds

// Made-up test particles added by default; --trojans N and --kuiper N change them (0 for none).
const DEFAULT_TROJANS: usize = 500;
const DEFAULT_KUIPER: usize = 1500;

// The number given after `flag` on the command line (for example --trojans 200), or `default`
// without it.
fn count_option(flag: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == flag) {
        Some(i) => args.get(i + 1).and_then(|n| n.parse().ok()).unwrap_or_else(|| {
            eprintln!("usage: sol_rust [--trojans N] [--kuiper N]");
            std::process::exit(2);
        }),
        None => default,
    }
}

fn main() {
    let mut plan: Vec<planets::Planet> = planets::get_planets();
    let trojans = planets::jupiter_trojans(&plan, count_option("--trojans", DEFAULT_TROJANS));
    plan.extend(trojans);
    let kuiper = planets::kuiper_belt(&plan, count_option("--kuiper", DEFAULT_KUIPER));
    plan.extend(kuiper);

    // Steps of five minutes, short enough for Phobos's 7.7-hour orbit.
    // n is only used by run_simulation; the renderer steps every frame instead.
    let dt = DAY / 288.0;
    let simulation = sim::new_simulation(plan, 0, dt);
    renderer::window::run(simulation);
}

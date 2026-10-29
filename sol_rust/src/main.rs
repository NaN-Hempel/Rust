// Solar System Simulation by Nanne Hempel, as a learning project.

mod code;
mod data;
mod probe;
mod renderer;
use code::sim;
use data::planets;

const DAY: f64 = 86_400.0; // seconds

// Made-up test particles added by default; --trojans N and --kuiper N change them (0 for none).
const DEFAULT_TROJANS: usize = 500;
const DEFAULT_KUIPER: usize = 3500;

// The number given after `flag` on the command line (for example --trojans 200), or `default`
// without it.
fn count_option(flag: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == flag) {
        Some(i) => args.get(i + 1).and_then(|n| n.parse().ok()).unwrap_or_else(|| {
            usage();
        }),
        None => default,
    }
}

// The argument after `flag`, or `default` if there is none (or the next one is another flag).
// None if `flag` wasn't given.
fn text_option(flag: &str, default: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let i = args.iter().position(|a| a == flag)?;
    Some(args.get(i + 1).filter(|a| !a.starts_with("--")).cloned().unwrap_or_else(|| default.to_string()))
}

fn usage() -> ! {
    eprintln!("usage: sol_rust [--trojans N] [--kuiper N] [--probe [FILE]]");
    eprintln!("       sol_rust --train-probe [GENERATIONS] [--probe FILE]");
    std::process::exit(2);
}

// Where the probe's network is saved, by default.
const PROBE_FILE: &str = "probe_best.txt";
const DEFAULT_GENERATIONS: &str = "100";

fn main() {
    // --train-probe: evolve a network to fly a probe to flight::TARGET, without opening the window.
    if let Some(generations) = text_option("--train-probe", DEFAULT_GENERATIONS) {
        let generations: usize = generations.parse().unwrap_or_else(|_| usage());
        let path = text_option("--probe", PROBE_FILE).unwrap_or(PROBE_FILE.to_string());
        println!("recording the planets for {:.0} days...", probe::MISSION / DAY);
        let ephemeris = probe::ephemeris::Ephemeris::record(planets::get_planets(), probe::MISSION, probe::SAMPLE, probe::SIM_STEP);
        probe::evolve::evolve(&ephemeris, generations, &path);
        println!("the best network and some random others saved to {}", path);
        return;
    }

    let mut plan: Vec<planets::Planet> = planets::get_planets();
    let trojans = planets::jupiter_trojans(&plan, count_option("--trojans", DEFAULT_TROJANS));
    plan.extend(trojans);
    let kuiper = planets::kuiper_belt(&plan, count_option("--kuiper", DEFAULT_KUIPER));
    plan.extend(kuiper);

    // Steps of five minutes, short enough for Phobos's 7.7-hour orbit.
    // n is only used by run_simulation; the renderer steps every frame instead.
    let dt = DAY / 288.0;
    let mut simulation = sim::new_simulation(plan, 0, dt);

    // --probe: fly the saved networks' probes, and show them with the planets.
    if let Some(path) = text_option("--probe", PROBE_FILE) {
        let brains = probe::evolve::load(&path).unwrap_or_else(|e| {
            eprintln!("couldn't load the probes' networks: {}", e);
            std::process::exit(1);
        });
        let ephemeris = probe::ephemeris::Ephemeris::record(planets::get_planets(), probe::MISSION, probe::SAMPLE, probe::SIM_STEP);
        for (n, brain) in brains.iter().enumerate() {
            // The first is the best; the rest are random others from the population.
            let name = if n == 0 { "Probe 1 (best)".to_string() } else { format!("Probe {}", n + 1) };
            let flight = probe::flight::fly(brain, &ephemeris, true);
            let launch = match flight.launch {
                Some((t, speed)) => format!("launched on day {:.0} at {:.2} km/s", t / DAY, speed / 1000.0),
                None => "never launched".to_string(),
            };
            println!(
                "{}: {}; closest to {} {:.0} km at {:.2} km/s, on day {:.0}, using {:.2} km/s of fuel{}",
                name,
                launch,
                probe::flight::TARGET,
                flight.distance / 1000.0,
                flight.speed / 1000.0,
                flight.best_time / DAY,
                flight.delta_v / 1000.0,
                flight.destroyed.as_ref().map_or(String::new(), |b| format!("; destroyed near {}", b))
            );
            let (body, track) = probe::body(&name, flight, ephemeris.interval);
            simulation.add_scripted(body, track);
        }
    }
    renderer::window::run(simulation);
}

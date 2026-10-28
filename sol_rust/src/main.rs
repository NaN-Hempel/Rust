// Solar System Simulation by Nanne Hempel, as a learning project.

mod code;
mod data;
mod renderer;
use code::sim;
use data::planets;

const DAY: f64 = 86_400.0; // seconds

fn print_bodies(planets: &[planets::Planet]) {
    for p in planets {
        println!(
            "{:<14} r = {:>8.3} AU  v = {:>6.2} km/s  position (AU) = ({:>7.3}, {:>7.3}, {:>7.3})",
            p.name,
            p.position.norm() / planets::AU,
            p.velocity.norm() / 1000.0,
            p.position.x / planets::AU,
            p.position.y / planets::AU,
            p.position.z / planets::AU,
        );
    }
}

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
    for p in &plan {
        println!("{}: mass {:.3e} kg, radius {} km, semi-major axis {:.0} km", p.name, p.mass, p.radius, p.semi_major_axis);
    }
    let listed = plan.len();
    let trojans = count_option("--trojans", DEFAULT_TROJANS);
    if trojans > 0 {
        let added = planets::jupiter_trojans(&plan, trojans);
        plan.extend(added);
        println!("\nAdded {} test particles around Jupiter's L4 and L5 points.", trojans);
    }
    let kuiper = count_option("--kuiper", DEFAULT_KUIPER);
    if kuiper > 0 {
        let added = planets::kuiper_belt(&plan, kuiper);
        plan.extend(added);
        println!("Added {} test particles in the Kuiper belt.", kuiper);
    }

    // Steps of five minutes, short enough for Phobos's 7.7-hour orbit.
    // n is only used by run_simulation; the renderer steps every frame instead.
    let dt = DAY / 288.0;
    let simulation = sim::new_simulation(plan, 0, dt);

    println!("\nStart:");
    print_bodies(&simulation.planets[..listed]);

    renderer::window::run(simulation);
}

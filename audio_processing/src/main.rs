// Audio processing by Nanne Hempel, as a learning project.
// For now: makes a test tone, measures it, and saves it as tone.wav.

mod math;
mod wav;

const SAMPLE_RATE: u32 = 44_100; // samples per second, as on a CD

fn main() {
    // Two seconds of concert A (440 Hz), at -6 dB so it isn't too loud.
    let tone = math::sine(440.0, math::db_to_gain(-6.0), 2.0, SAMPLE_RATE);
    println!(
        "{} samples: peak {:.1} dB, RMS {:.1} dB",
        tone.len(),
        math::gain_to_db(math::peak(&tone)),
        math::gain_to_db(math::rms(&tone))
    );

    let path = "tone.wav";
    match wav::write(path, &tone, SAMPLE_RATE) {
        Ok(()) => println!("saved to {}", path),
        Err(e) => eprintln!("couldn't save {}: {}", path, e),
    }
}

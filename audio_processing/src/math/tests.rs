use super::*;

#[test]
fn sine_has_the_right_length_and_peak() {
    let tone = sine(440.0, 0.5, 1.0, 44_100);
    assert_eq!(tone.len(), 44_100);
    assert_eq!(tone[0], 0.0);
    assert!((peak(&tone) - 0.5).abs() < 1e-3);
}

#[test]
fn rms_of_a_sine_is_peak_over_root_two() {
    // 100 Hz at 44.1 kHz is exactly 441 samples a cycle, so one second is whole cycles.
    let tone = sine(100.0, 0.8, 1.0, 44_100);
    assert!((rms(&tone) - 0.8 / 2.0_f64.sqrt()).abs() < 1e-9);
}

#[test]
fn silence_measures_zero() {
    assert_eq!(peak(&[]), 0.0);
    assert_eq!(rms(&[]), 0.0);
    assert_eq!(rms(&[0.0; 10]), 0.0);
    assert_eq!(gain_to_db(0.0), f64::NEG_INFINITY);
}

#[test]
fn decibels_round_trip() {
    assert_eq!(gain_to_db(1.0), 0.0);
    assert!((gain_to_db(0.5) + 6.0206).abs() < 1e-4);
    for db in [-60.0, -6.0, 0.0, 3.0] {
        assert!((gain_to_db(db_to_gain(db)) - db).abs() < 1e-12);
    }
}

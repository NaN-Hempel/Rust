use super::*;

#[test]
fn fill_copies_each_sample_to_every_channel() {
    let mut buffer = [0.0f32; 6];
    let mut position = 0;
    fill(&mut buffer, 2, &[0.1, 0.2, 0.3], &mut position);
    assert_eq!(buffer, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
    assert_eq!(position, 3);
}

#[test]
fn fill_carries_on_where_it_left_off() {
    let samples = [0.1, 0.2, 0.3, 0.4];
    let mut position = 0;
    let mut buffer = [0.0f32; 2];
    fill(&mut buffer, 1, &samples, &mut position);
    fill(&mut buffer, 1, &samples, &mut position);
    assert_eq!(buffer, [0.3, 0.4]);
}

#[test]
fn fill_is_silent_after_the_end() {
    let mut buffer = [1.0f32; 4];
    let mut position = 0;
    fill(&mut buffer, 1, &[0.5], &mut position);
    assert_eq!(buffer, [0.5, 0.0, 0.0, 0.0]);
    assert_eq!(position, 1);
}

#[test]
fn fill_converts_to_the_device_sample_type() {
    let mut buffer = [0i16; 2];
    let mut position = 0;
    fill(&mut buffer, 1, &[1.0, -1.0], &mut position);
    // cpal maps -1.0 to i16::MIN, one lower than wav.rs's -i16::MAX.
    assert_eq!(buffer, [i16::MAX, i16::MIN]);
}

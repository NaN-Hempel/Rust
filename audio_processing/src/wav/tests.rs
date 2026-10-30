use super::*;

fn u32_at(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap())
}

#[test]
fn header_describes_the_samples() {
    let bytes = encode(&[0.0; 100], 44_100);
    assert_eq!(bytes.len(), 44 + 200);
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(u32_at(&bytes, 4), bytes.len() as u32 - 8);
    assert_eq!(&bytes[8..16], b"WAVEfmt ");
    assert_eq!(u32_at(&bytes, 24), 44_100); // sample rate
    assert_eq!(u32_at(&bytes, 28), 88_200); // bytes per second
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(u32_at(&bytes, 40), 200);
}

#[test]
fn samples_are_scaled_and_clipped() {
    assert_eq!(to_i16(0.0), 0);
    assert_eq!(to_i16(1.0), i16::MAX);
    assert_eq!(to_i16(-1.0), -i16::MAX);
    assert_eq!(to_i16(0.5), 16_384);
    assert_eq!(to_i16(3.0), i16::MAX);
    assert_eq!(to_i16(-3.0), -i16::MAX);
}

#[test]
fn samples_follow_the_header_little_endian() {
    let bytes = encode(&[1.0, -1.0], 8_000);
    assert_eq!(&bytes[44..], &[0xFF, 0x7F, 0x01, 0x80]);
}

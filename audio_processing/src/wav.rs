// Writing samples as a WAV file: mono, 16-bit PCM, the simplest kind there is. A WAV file is a
// 44-byte header describing the format, followed by the samples as little-endian integers.

use std::io;
use std::path::Path;

#[cfg(test)]
mod tests;

const BITS_PER_SAMPLE: u16 = 16;
const CHANNELS: u16 = 1;

// The whole file as bytes. Samples outside -1.0..=1.0 are clipped to full scale.
pub fn encode(samples: &[f64], sample_rate: u32) -> Vec<u8> {
    let block_align = CHANNELS * BITS_PER_SAMPLE / 8; // bytes per sample, over all channels
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (samples.len() * block_align as usize) as u32;

    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    // RIFF header: the file's size after these first 8 bytes.
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    // The format chunk: 16 bytes, format 1 meaning uncompressed PCM.
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&CHANNELS.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    // The data chunk: the samples themselves.
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        bytes.extend_from_slice(&to_i16(s).to_le_bytes());
    }
    bytes
}

pub fn write(path: impl AsRef<Path>, samples: &[f64], sample_rate: u32) -> io::Result<()> {
    std::fs::write(path, encode(samples, sample_rate))
}

// A sample as a 16-bit integer, with 1.0 as i16::MAX.
fn to_i16(sample: f64) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f64).round() as i16
}

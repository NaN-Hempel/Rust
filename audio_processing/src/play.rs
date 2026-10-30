// Playing samples through the speakers, with cpal.
//
// cpal doesn't take the samples all at once: it runs a callback on its own audio thread, which
// asks for a small buffer at a time, as fast as the speakers use them up. That callback has to be
// quick (no allocating, locking or printing), or the sound drops out. So the samples are got
// ready first, and the callback only copies them across.

use std::error::Error;
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};

#[cfg(test)]
mod tests;

// Plays mono samples (-1.0 to 1.0) at sample_rate on the default output device, and returns
// once they have all been played. The same sound goes to every channel (both speakers).
pub fn play(samples: &[f64], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let device = cpal::default_host().default_output_device().ok_or("no output device found")?;
    // A config the device supports at our sample rate, f32 if it has one.
    let supported = device
        .supported_output_configs()?
        .filter_map(|range| range.try_with_sample_rate(sample_rate))
        .max_by_key(|config| config.sample_format() == SampleFormat::F32)
        .ok_or_else(|| format!("the output device can't play at {} Hz", sample_rate))?;
    let config = supported.config();

    let samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();
    // The audio thread tells this one when it's finished, or went wrong.
    let (done, finished) = mpsc::sync_channel(1);
    let stream = match supported.sample_format() {
        SampleFormat::F32 => stream::<f32>(&device, config, samples, done)?,
        SampleFormat::F64 => stream::<f64>(&device, config, samples, done)?,
        SampleFormat::I16 => stream::<i16>(&device, config, samples, done)?,
        SampleFormat::I32 => stream::<i32>(&device, config, samples, done)?,
        SampleFormat::U16 => stream::<u16>(&device, config, samples, done)?,
        other => return Err(format!("unsupported sample format {}", other).into()),
    };
    stream.play()?;
    finished.recv()??;
    Ok(())
}

// An output stream that plays samples, in the device's sample type T, then sends Ok on done.
fn stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    samples: Vec<f32>,
    done: mpsc::SyncSender<Result<(), cpal::Error>>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut position = 0;
    let errors = done.clone();
    device.build_output_stream(
        config,
        move |buffer: &mut [T], _| {
            // Asked for more after the last sample: the buffer with it in has gone to the speakers.
            if position >= samples.len() {
                let _ = done.try_send(Ok(())); // try_send never blocks the audio thread
            }
            fill(buffer, channels, &samples, &mut position);
        },
        move |e| {
            let _ = errors.try_send(Err(e));
        },
        None,
    )
}

// Fills buffer, which holds frames of `channels` samples each (left, right, left, ...), with the
// samples from position on, the same in every channel, and moves position past them. Once the
// samples run out, the rest is silence.
fn fill<T>(buffer: &mut [T], channels: usize, samples: &[f32], position: &mut usize)
where
    T: SizedSample + FromSample<f32>,
{
    for frame in buffer.chunks_mut(channels) {
        let sample = samples.get(*position).copied().unwrap_or(0.0);
        *position = (*position + 1).min(samples.len());
        for out in frame {
            *out = T::from_sample(sample);
        }
    }
}

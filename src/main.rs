use hound::{WavReader, WavSpec, WavWriter};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

struct Event {
    frequency_ratio: f32,
    length_ratio: f32,
}

fn get_samples_for_event(event: &Event, input_samples: &[f32]) -> Vec<f32> {
    let num_samples = (input_samples.len() as f32 * event.length_ratio).ceil() as usize;

    process_chunk_with_pitch_and_time_shift(
        input_samples.to_vec(),
        event.frequency_ratio,
        event.length_ratio,
        num_samples,
    )
}

fn process_chunk_with_pitch_and_time_shift(
    mut chunk: Vec<f32>,
    frequency_ratio: f32,
    length_ratio: f32,
    num_samples: usize,
) -> Vec<f32> {
    let fft_size = chunk.len();

    // Compute FFT
    let mut output: Vec<Complex<f32>> = chunk.into_iter().map(|s| Complex::new(s, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(output.len());
    fft.process(&mut output);

    // Apply frequency and length ratios
    let mut shifted_output = vec![Complex::zero(); output.len()];
    for (i, &value) in output.iter().enumerate() {
        let shifted_index = (i as f32 * frequency_ratio).round() as usize;
        if shifted_index < shifted_output.len() {
            shifted_output[shifted_index] = value;
        }
    }

    // Compute IFFT
    let mut ifft_output = shifted_output;
    let ifft = planner.plan_fft_inverse(ifft_output.len());
    ifft.process(&mut ifft_output);

    // Normalize and truncate chunk
    let mut output_chunk: Vec<f32> = ifft_output
        .into_iter()
        .map(|c| c.re / fft_size as f32)
        .collect();
    output_chunk.truncate(num_samples);

    output_chunk
}

fn main() {
    let mut reader = WavReader::open("input.wav").unwrap();
    let spec = reader.spec();
    let input_samples: Vec<_> = reader.samples::<f32>().map(|s| s.unwrap()).collect();

    let events = vec![
        Event {
            frequency_ratio: 1.0,
            length_ratio: 1.0,
        },
        Event {
            frequency_ratio: 1.5,
            length_ratio: 0.7,
        },
        // More events...
    ];

    let mut writer = WavWriter::create("output.wav", spec).unwrap();

    for event in &events {
        let event_samples = get_samples_for_event(event, &input_samples);

        for sample in event_samples {
            writer.write_sample(sample).unwrap();
        }
    }
}

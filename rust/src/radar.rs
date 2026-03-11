use num_complex::Complex32;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use std::f64::consts::PI;

// ── Internal helpers (return Vec<Complex32>) ───────────────────────

fn pulse_train_samples(
    pulse: &[Complex32],
    pri_samples: usize,
    num_pulses: usize,
    stagger: &[i64],
) -> Vec<Complex32> {
    let total_len = pri_samples * num_pulses;
    let mut output = vec![Complex32::new(0.0, 0.0); total_len];

    for p in 0..num_pulses {
        let base_offset = p * pri_samples;
        let stagger_offset = if stagger.is_empty() {
            0i64
        } else {
            stagger[p % stagger.len()]
        };
        let start = (base_offset as i64 + stagger_offset).max(0) as usize;

        for (i, &sample) in pulse.iter().enumerate() {
            let idx = start + i;
            if idx < total_len {
                output[idx] = sample;
            }
        }
    }

    output
}

fn fmcw_sweep_samples(
    num_samples: usize,
    bandwidth: f64,
    fs: f64,
    sweep_type: &str,
) -> Result<Vec<Complex32>, String> {
    match sweep_type {
        "sawtooth" => {
            let f0 = -bandwidth / 2.0;
            let chirp_rate = bandwidth / (num_samples as f64 / fs);
            let samples: Vec<Complex32> = (0..num_samples)
                .map(|i| {
                    let t = i as f64 / fs;
                    let phase = 2.0 * PI * (f0 * t + 0.5 * chirp_rate * t * t);
                    Complex32::new(phase.cos() as f32, phase.sin() as f32)
                })
                .collect();
            Ok(samples)
        }
        "triangle" => {
            let half = num_samples / 2;
            let remainder = num_samples - half;
            let f0 = -bandwidth / 2.0;
            let f1 = bandwidth / 2.0;

            let mut samples = Vec::with_capacity(num_samples);
            let up_duration = half as f64 / fs;
            let chirp_rate_up = (f1 - f0) / up_duration;

            // Up-sweep: f0 -> f1
            for i in 0..half {
                let t = i as f64 / fs;
                let phase = 2.0 * PI * (f0 * t + 0.5 * chirp_rate_up * t * t);
                samples.push(Complex32::new(phase.cos() as f32, phase.sin() as f32));
            }

            // Phase at end of up-sweep for continuity
            let t_end_up = half as f64 / fs;
            let end_phase = 2.0 * PI * (f0 * t_end_up + 0.5 * chirp_rate_up * t_end_up * t_end_up);

            // Down-sweep: f1 -> f0
            let down_duration = remainder as f64 / fs;
            let chirp_rate_down = (f0 - f1) / down_duration;
            for i in 0..remainder {
                let t = i as f64 / fs;
                let phase = end_phase + 2.0 * PI * (f1 * t + 0.5 * chirp_rate_down * t * t);
                samples.push(Complex32::new(phase.cos() as f32, phase.sin() as f32));
            }

            Ok(samples)
        }
        _ => Err(format!(
            "Unknown sweep_type '{}'. Use 'sawtooth' or 'triangle'.",
            sweep_type
        )),
    }
}

fn stepped_frequency_samples(
    num_steps: usize,
    samples_per_step: usize,
    freq_step: f64,
    fs: f64,
) -> Vec<Complex32> {
    let total_len = num_steps * samples_per_step;
    let mut samples = Vec::with_capacity(total_len);
    let mut phase_accum = 0.0f64;

    for step in 0..num_steps {
        let freq = step as f64 * freq_step;
        let phase_inc = 2.0 * PI * freq / fs;

        for _ in 0..samples_per_step {
            samples.push(Complex32::new(
                phase_accum.cos() as f32,
                phase_accum.sin() as f32,
            ));
            phase_accum += phase_inc;
        }
    }

    samples
}

fn nlfm_sweep_samples(
    num_samples: usize,
    fs: f64,
    bandwidth: f64,
    sweep_type: &str,
) -> Result<Vec<Complex32>, String> {
    match sweep_type {
        "tandem_hooked" => {
            let mut samples = Vec::with_capacity(num_samples);
            let mut phase_accum = 0.0f64;
            let norm = (3.0f64).tanh();

            for i in 0..num_samples {
                let t_norm = i as f64 / num_samples as f64;
                let x = 6.0 * (t_norm - 0.5);
                let freq_norm = x.tanh() / norm;
                let freq = freq_norm * bandwidth / 2.0;

                let phase_inc = 2.0 * PI * freq / fs;
                phase_accum += phase_inc;
                samples.push(Complex32::new(
                    phase_accum.cos() as f32,
                    phase_accum.sin() as f32,
                ));
            }

            Ok(samples)
        }
        "s_curve" => {
            let mut samples = Vec::with_capacity(num_samples);
            let mut phase_accum = 0.0f64;
            let norm = (2.0f64).tanh();

            for i in 0..num_samples {
                let t_norm = i as f64 / num_samples as f64;
                let x = 4.0 * (t_norm - 0.5);
                let freq_norm = x.tanh() / norm;
                let freq = freq_norm * bandwidth / 2.0;

                let phase_inc = 2.0 * PI * freq / fs;
                phase_accum += phase_inc;
                samples.push(Complex32::new(
                    phase_accum.cos() as f32,
                    phase_accum.sin() as f32,
                ));
            }

            Ok(samples)
        }
        _ => Err(format!(
            "Unknown sweep_type '{}'. Use 'tandem_hooked' or 's_curve'.",
            sweep_type
        )),
    }
}

// ── PyO3-exported functions ────────────────────────────────────────

/// Place a pulse at PRI intervals with optional stagger offsets.
/// Total output length = pri_samples * num_pulses.
/// stagger: per-pulse timing offsets in samples; cycled if shorter than num_pulses.
#[pyfunction]
pub fn generate_pulse_train<'py>(
    py: Python<'py>,
    pulse: PyReadonlyArray1<Complex32>,
    pri_samples: usize,
    num_pulses: usize,
    stagger: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let pulse_data = pulse.as_slice()?;
    let stagger_data = stagger.as_slice()?;
    let output = pulse_train_samples(pulse_data, pri_samples, num_pulses, stagger_data);
    Ok(Array1::from_vec(output).into_pyarray(py))
}

/// Generate an FMCW sweep signal.
/// sweep_type: "sawtooth" or "triangle"
#[pyfunction]
pub fn generate_fmcw_sweep<'py>(
    py: Python<'py>,
    num_samples: usize,
    bandwidth: f64,
    fs: f64,
    sweep_type: &str,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let samples = fmcw_sweep_samples(num_samples, bandwidth, fs, sweep_type)
        .map_err(PyValueError::new_err)?;
    Ok(Array1::from_vec(samples).into_pyarray(py))
}

/// Generate a stepped-frequency waveform with phase-continuous CW tones.
#[pyfunction]
pub fn generate_stepped_frequency<'py>(
    py: Python<'py>,
    num_steps: usize,
    samples_per_step: usize,
    freq_step: f64,
    fs: f64,
) -> Bound<'py, PyArray1<Complex32>> {
    let samples = stepped_frequency_samples(num_steps, samples_per_step, freq_step, fs);
    Array1::from_vec(samples).into_pyarray(py)
}

/// Generate a nonlinear FM sweep signal.
/// sweep_type: "tandem_hooked" or "s_curve"
#[pyfunction]
pub fn generate_nlfm_sweep<'py>(
    py: Python<'py>,
    num_samples: usize,
    fs: f64,
    bandwidth: f64,
    sweep_type: &str,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let samples = nlfm_sweep_samples(num_samples, fs, bandwidth, sweep_type)
        .map_err(PyValueError::new_err)?;
    Ok(Array1::from_vec(samples).into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Pulse train tests ---

    #[test]
    fn pulse_train_output_length() {
        let pulse = vec![Complex32::new(1.0, 0.0); 32];
        let result = pulse_train_samples(&pulse, 128, 4, &[]);
        assert_eq!(result.len(), 128 * 4);
    }

    #[test]
    fn pulse_train_pulse_placement() {
        let pulse = vec![Complex32::new(1.0, 0.0); 4];
        let result = pulse_train_samples(&pulse, 16, 3, &[]);

        // Check pulse at start of each PRI
        for p in 0..3 {
            let base = p * 16;
            assert!((result[base].re - 1.0).abs() < 1e-5);
            // Gap samples should be zero
            assert!(result[base + 4].re.abs() < 1e-5);
        }
    }

    #[test]
    fn pulse_train_with_stagger() {
        let pulse = vec![Complex32::new(1.0, 0.0); 2];
        let result = pulse_train_samples(&pulse, 16, 3, &[0, 2, 4]);

        // Pulse 0 at offset 0
        assert!((result[0].re - 1.0).abs() < 1e-5);
        // Pulse 1 at offset 16+2=18
        assert!((result[18].re - 1.0).abs() < 1e-5);
        // Pulse 2 at offset 32+4=36
        assert!((result[36].re - 1.0).abs() < 1e-5);
    }

    // --- FMCW tests ---

    #[test]
    fn fmcw_sawtooth_length() {
        let result = fmcw_sweep_samples(256, 100_000.0, 1_000_000.0, "sawtooth").unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn fmcw_triangle_length() {
        let result = fmcw_sweep_samples(256, 100_000.0, 1_000_000.0, "triangle").unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn fmcw_unit_magnitude() {
        let result = fmcw_sweep_samples(256, 100_000.0, 1_000_000.0, "sawtooth").unwrap();
        for s in &result {
            let mag = (s.re * s.re + s.im * s.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-4, "Magnitude {} != 1.0", mag);
        }
    }

    #[test]
    fn fmcw_unknown_type_errors() {
        let result = fmcw_sweep_samples(256, 100_000.0, 1_000_000.0, "unknown");
        assert!(result.is_err());
    }

    // --- Stepped frequency tests ---

    #[test]
    fn stepped_frequency_length() {
        let result = stepped_frequency_samples(8, 64, 10_000.0, 1_000_000.0);
        assert_eq!(result.len(), 8 * 64);
    }

    #[test]
    fn stepped_frequency_unit_magnitude() {
        let result = stepped_frequency_samples(4, 32, 5_000.0, 1_000_000.0);
        for s in &result {
            let mag = (s.re * s.re + s.im * s.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-4, "Magnitude {} != 1.0", mag);
        }
    }

    // --- NLFM tests ---

    #[test]
    fn nlfm_tandem_hooked_length() {
        let result = nlfm_sweep_samples(256, 1_000_000.0, 100_000.0, "tandem_hooked").unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn nlfm_s_curve_length() {
        let result = nlfm_sweep_samples(256, 1_000_000.0, 100_000.0, "s_curve").unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn nlfm_unit_magnitude() {
        let result = nlfm_sweep_samples(256, 1_000_000.0, 100_000.0, "s_curve").unwrap();
        for s in &result {
            let mag = (s.re * s.re + s.im * s.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-4, "Magnitude {} != 1.0", mag);
        }
    }

    #[test]
    fn nlfm_unknown_type_errors() {
        let result = nlfm_sweep_samples(256, 1_000_000.0, 100_000.0, "unknown");
        assert!(result.is_err());
    }
}

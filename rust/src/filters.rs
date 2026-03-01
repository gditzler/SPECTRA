use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use num_complex::Complex32;
use pyo3::prelude::*;

/// Generate root-raised-cosine filter taps.
fn rrc_taps(rolloff: f32, span: usize, sps: usize) -> Vec<f32> {
    let num_taps = span * sps + 1;
    let half = (num_taps / 2) as f32;
    let sps_f = sps as f32;
    let mut taps: Vec<f32> = (0..num_taps)
        .map(|i| {
            let t = (i as f32 - half) / sps_f;
            if t.abs() < 1e-12 {
                (1.0 - rolloff + 4.0 * rolloff / std::f32::consts::PI) / sps_f.sqrt()
            } else if (t.abs() - 1.0 / (4.0 * rolloff)).abs() < 1e-12 && rolloff > 0.0 {
                let sqrt2 = std::f32::consts::SQRT_2;
                rolloff / (sqrt2 * sps_f.sqrt())
                    * ((1.0 + 2.0 / std::f32::consts::PI) * (std::f32::consts::PI / (4.0 * rolloff)).sin()
                        + (1.0 - 2.0 / std::f32::consts::PI) * (std::f32::consts::PI / (4.0 * rolloff)).cos())
            } else {
                let pi_t = std::f32::consts::PI * t;
                let num = (pi_t * (1.0 - rolloff)).sin()
                    + 4.0 * rolloff * t * (pi_t * (1.0 + rolloff)).cos();
                let den = pi_t * (1.0 - (4.0 * rolloff * t).powi(2));
                if den.abs() < 1e-12 {
                    0.0
                } else {
                    num / (den * sps_f.sqrt())
                }
            }
        })
        .collect();

    // Normalize to unit energy
    let energy: f32 = taps.iter().map(|t| t * t).sum();
    if energy > 0.0 {
        let norm = energy.sqrt();
        for t in &mut taps {
            *t /= norm;
        }
    }
    taps
}

/// Apply RRC pulse-shaping filter: upsample by sps, then convolve with RRC taps.
#[pyfunction]
pub fn apply_rrc_filter<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<'py, Complex32>,
    rolloff: f32,
    span: usize,
    sps: usize,
) -> Bound<'py, PyArray1<Complex32>> {
    let symbols = symbols.as_array();
    let taps = rrc_taps(rolloff, span, sps);

    // Upsample: insert sps-1 zeros between each symbol
    let upsampled_len = symbols.len() * sps;
    let mut upsampled = vec![Complex32::new(0.0, 0.0); upsampled_len];
    for (i, &s) in symbols.iter().enumerate() {
        upsampled[i * sps] = s;
    }

    // Convolve
    let output_len = upsampled_len + taps.len() - 1;
    let output = Array1::from_shape_fn(output_len, |n| {
        let mut sum = Complex32::new(0.0, 0.0);
        for (k, &tap) in taps.iter().enumerate() {
            if n >= k && (n - k) < upsampled_len {
                sum += upsampled[n - k] * tap;
            }
        }
        sum
    });
    output.into_pyarray(py)
}

use num_complex::Complex32;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
                    * ((1.0 + 2.0 / std::f32::consts::PI)
                        * (std::f32::consts::PI / (4.0 * rolloff)).sin()
                        + (1.0 - 2.0 / std::f32::consts::PI)
                            * (std::f32::consts::PI / (4.0 * rolloff)).cos())
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

/// Expose RRC filter taps to Python for caching.
#[pyfunction]
pub fn rrc_taps_py<'py>(
    py: Python<'py>,
    rolloff: f32,
    span: usize,
    sps: usize,
) -> Bound<'py, PyArray1<f32>> {
    let taps = rrc_taps(rolloff, span, sps);
    Array1::from_vec(taps).into_pyarray(py)
}

/// Upsample symbols by sps (zero-insertion) then convolve with filter taps.
fn upsample_convolve(symbols: &[Complex32], taps: &[f32], sps: usize) -> Array1<Complex32> {
    let upsampled_len = symbols.len() * sps;
    let mut upsampled = vec![Complex32::new(0.0, 0.0); upsampled_len];
    for (i, &s) in symbols.iter().enumerate() {
        upsampled[i * sps] = s;
    }

    let output_len = upsampled_len + taps.len() - 1;
    Array1::from_shape_fn(output_len, |n| {
        let mut sum = Complex32::new(0.0, 0.0);
        for (k, &tap) in taps.iter().enumerate() {
            if n >= k && (n - k) < upsampled_len {
                sum += upsampled[n - k] * tap;
            }
        }
        sum
    })
}

/// Apply pulse-shaping with pre-computed filter taps: upsample then convolve.
#[pyfunction]
pub fn apply_rrc_filter_with_taps<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<'py, Complex32>,
    taps: PyReadonlyArray1<'py, f32>,
    sps: usize,
) -> Bound<'py, PyArray1<Complex32>> {
    let symbols = symbols.as_array();
    let taps = taps.as_array();
    upsample_convolve(symbols.as_slice().unwrap(), taps.as_slice().unwrap(), sps).into_pyarray(py)
}

/// Generate Gaussian filter taps for GFSK/GMSK pulse shaping.
/// Taps sum to 1.0, symmetric, peak at center.
#[pyfunction]
pub fn gaussian_taps<'py>(
    py: Python<'py>,
    bt: f32,
    span: usize,
    sps: usize,
) -> Bound<'py, PyArray1<f32>> {
    let half = (span * sps / 2) as i32;
    let sps_f = sps as f32;
    let mut taps: Vec<f32> = (-half..=half)
        .map(|i| {
            let t = i as f32 / sps_f;
            let pi = std::f32::consts::PI;
            let ln2 = 2.0_f32.ln();
            (2.0 * pi / ln2).sqrt() * bt * (-2.0 * (pi * bt * t).powi(2) / ln2).exp()
        })
        .collect();
    // Normalize so taps sum to 1.0
    let sum: f32 = taps.iter().sum();
    if sum > 0.0 {
        for t in &mut taps {
            *t /= sum;
        }
    }
    Array1::from_vec(taps).into_pyarray(py)
}

/// Windowed-sinc (Blackman) FIR lowpass filter design.
/// cutoff is normalized frequency in (0, 1) where 1 = Nyquist.
#[pyfunction]
pub fn lowpass_taps<'py>(
    py: Python<'py>,
    num_taps: usize,
    cutoff: f32,
) -> Bound<'py, PyArray1<f32>> {
    let pi = std::f32::consts::PI;
    let m = (num_taps - 1) as f32;
    let mut taps: Vec<f32> = (0..num_taps)
        .map(|i| {
            let n = i as f32;
            // Sinc component
            let sinc = if (n - m / 2.0).abs() < 1e-12 {
                cutoff
            } else {
                let x = n - m / 2.0;
                (pi * cutoff * x).sin() / (pi * x)
            };
            // Blackman window
            let w = 0.42 - 0.5 * (2.0 * pi * n / m).cos() + 0.08 * (4.0 * pi * n / m).cos();
            sinc * w
        })
        .collect();
    // Normalize to unit DC gain
    let sum: f32 = taps.iter().sum();
    if sum.abs() > 0.0 {
        for t in &mut taps {
            *t /= sum;
        }
    }
    Array1::from_vec(taps).into_pyarray(py)
}

/// General complex convolution: convolve a complex signal with real filter taps.
#[pyfunction]
pub fn convolve_complex<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, Complex32>,
    taps: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<Complex32>> {
    let signal = signal.as_array();
    let taps = taps.as_array();
    let sig_len = signal.len();
    let tap_len = taps.len();
    let output_len = sig_len + tap_len - 1;
    let output = Array1::from_shape_fn(output_len, |n| {
        let mut sum = Complex32::new(0.0, 0.0);
        for (k, &tap) in taps.iter().enumerate() {
            if n >= k && (n - k) < sig_len {
                sum += signal[n - k] * tap;
            }
        }
        sum
    });
    output.into_pyarray(py)
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
    upsample_convolve(symbols.as_slice().unwrap(), &taps, sps).into_pyarray(py)
}

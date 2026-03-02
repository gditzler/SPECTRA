use num_complex::Complex32;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// Generate a linear frequency modulated (chirp) signal.
#[pyfunction]
pub fn generate_chirp<'py>(
    py: Python<'py>,
    duration: f64,
    fs: f64,
    f0: f64,
    f1: f64,
) -> Bound<'py, PyArray1<Complex32>> {
    let num_samples = (duration * fs) as usize;
    let chirp_rate = (f1 - f0) / duration;
    let signal = Array1::from_shape_fn(num_samples, |i| {
        let t = i as f64 / fs;
        let phase = 2.0 * std::f64::consts::PI * (f0 * t + 0.5 * chirp_rate * t * t);
        Complex32::new(phase.cos() as f32, phase.sin() as f32)
    });
    signal.into_pyarray(py)
}

/// Generate a complex sinusoidal tone.
#[pyfunction]
pub fn generate_tone<'py>(
    py: Python<'py>,
    frequency: f64,
    duration: f64,
    fs: f64,
) -> Bound<'py, PyArray1<Complex32>> {
    let num_samples = (duration * fs) as usize;
    let signal = Array1::from_shape_fn(num_samples, |i| {
        let t = i as f64 / fs;
        let phase = 2.0 * std::f64::consts::PI * frequency * t;
        Complex32::new(phase.cos() as f32, phase.sin() as f32)
    });
    signal.into_pyarray(py)
}

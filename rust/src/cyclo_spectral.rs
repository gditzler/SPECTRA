use num_complex::Complex32;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Apply a real-valued window to complex samples element-wise.
pub(crate) fn apply_window(samples: &[Complex32], window: &[f32]) -> Vec<Complex32> {
    samples
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| Complex32::new(s.re * w, s.im * w))
        .collect()
}

/// Hann window of given size.
pub(crate) fn hann_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    let n_minus_1 = (size - 1) as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_minus_1).cos()))
        .collect()
}

/// Sliding windowed FFT channelizer.
///
/// Returns `frames[t][k]` where `t` is the time frame index and `k` is the
/// frequency bin.  Each frame is windowed with a Hann window before the FFT.
pub(crate) fn channelize_frames(
    samples: &[Complex32],
    nfft: usize,
    hop: usize,
) -> Vec<Vec<Complex32>> {
    if samples.len() < nfft || nfft == 0 || hop == 0 {
        return Vec::new();
    }

    let window = hann_window(nfft);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);

    let n_frames = (samples.len() - nfft) / hop + 1;
    let mut frames = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
        let start = t * hop;
        let mut buffer = apply_window(&samples[start..start + nfft], &window);
        fft.process(&mut buffer);
        frames.push(buffer);
    }

    frames
}

/// fftshift along the row (first) axis of a 2-D array.
pub(crate) fn fftshift_rows(arr: &Array2<Complex32>) -> Array2<Complex32> {
    let (nrows, ncols) = arr.dim();
    let half = nrows / 2;
    let mut shifted = Array2::<Complex32>::zeros((nrows, ncols));
    for r in 0..nrows {
        let r_new = (r + half) % nrows;
        for c in 0..ncols {
            shifted[[r_new, c]] = arr[[r, c]];
        }
    }
    shifted
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Compute the Spectral Correlation Density (SCD) via the Strip Spectral
/// Correlation Algorithm (SSCA).
///
/// The channelizer applies sliding `nfft`-point Hann-windowed FFTs with the
/// given hop size.  For each cyclic-frequency offset `d` (in frequency bins),
/// the cross-spectral product `X[t,k] * conj(X[t, (k-d) mod nfft])` is
/// accumulated over all time frames and normalised.
///
/// Returns a 2-D complex array `[nfft, n_alpha]`.
/// - Frequency axis (rows): DC-centred via fftshift.
/// - Cyclic-frequency axis (cols): centred at `alpha = 0` (column `n_alpha/2`).
#[pyfunction]
pub fn compute_scd_ssca<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    n_alpha: usize,
    hop: usize,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();

    let frames = channelize_frames(&samples, nfft, hop);
    let n_frames = frames.len();

    if n_frames == 0 {
        return Array2::<Complex32>::zeros((nfft, n_alpha)).into_pyarray(py);
    }

    let half_alpha = n_alpha as isize / 2;
    let inv_n = 1.0 / n_frames as f32;
    let nfft_i = nfft as isize;

    let mut scd = Array2::<Complex32>::zeros((nfft, n_alpha));

    for ai in 0..n_alpha {
        let d = ai as isize - half_alpha; // cyclic-freq offset in bins
        for k in 0..nfft {
            let k2 = ((k as isize - d).rem_euclid(nfft_i)) as usize;
            let mut acc = Complex32::new(0.0, 0.0);
            for frame in &frames {
                acc += frame[k] * frame[k2].conj();
            }
            scd[[k, ai]] = Complex32::new(acc.re * inv_n, acc.im * inv_n);
        }
    }

    // DC-centre along frequency axis
    fftshift_rows(&scd).into_pyarray(py)
}

/// Power Spectral Density via Welch's averaged-periodogram method.
///
/// Segments the input into overlapping frames, applies a Hann window, computes
/// the FFT, averages the squared magnitudes, and normalises by the window
/// power.
///
/// Returns a 1-D real array `[nfft]` in linear power units, DC-centred.
#[pyfunction]
pub fn compute_psd_welch<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    overlap: usize,
) -> Bound<'py, PyArray1<f32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();
    let n = samples.len();

    if n == 0 || nfft == 0 {
        return Array1::<f32>::zeros(nfft).into_pyarray(py);
    }

    let window = hann_window(nfft);
    let window_power: f32 = window.iter().map(|w| w * w).sum();

    let step = if overlap < nfft { nfft - overlap } else { 1 };

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);

    // Use f64 accumulation to reduce round-off
    let mut psd = vec![0.0f64; nfft];
    let mut n_segments = 0usize;

    let mut start = 0;
    while start + nfft <= n {
        let mut buffer = apply_window(&samples[start..start + nfft], &window);
        fft.process(&mut buffer);

        for (k, val) in buffer.iter().enumerate() {
            psd[k] += (val.re * val.re + val.im * val.im) as f64;
        }
        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return Array1::<f32>::zeros(nfft).into_pyarray(py);
    }

    let norm = (n_segments as f64) * (window_power as f64);
    for val in &mut psd {
        *val /= norm;
    }

    // fftshift
    let half = nfft / 2;
    let mut shifted = vec![0.0f32; nfft];
    for k in 0..nfft {
        shifted[(k + half) % nfft] = psd[k] as f32;
    }

    Array1::from_vec(shifted).into_pyarray(py)
}

/// Compute the Spectral Correlation Density (SCD) via the FFT Accumulation
/// Method (FAM).
///
/// Like SSCA, the FAM channelises the input with sliding Hann-windowed FFTs.
/// It then applies a second FFT along the time axis for each frequency channel
/// before forming cross-spectral products in the transformed domain.  By
/// Parseval's theorem the result converges to the same estimate as SSCA, but
/// FAM can be more efficient when the number of time frames is large.
///
/// Returns a 2-D complex array `[nfft_chan, nfft_fft]`.
/// - Frequency axis (rows): DC-centred via fftshift.
/// - Cyclic-frequency axis (cols): centred at `alpha = 0`.
#[pyfunction]
pub fn compute_scd_fam<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft_chan: usize,
    nfft_fft: usize,
    hop: usize,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();

    let frames = channelize_frames(&samples, nfft_chan, hop);
    let n_frames = frames.len();

    if n_frames == 0 {
        return Array2::<Complex32>::zeros((nfft_chan, nfft_fft)).into_pyarray(py);
    }

    // Pad to next power of two for efficient FFT along time axis.
    let p = n_frames.next_power_of_two().max(nfft_fft);

    let mut planner = FftPlanner::new();
    let fft_time = planner.plan_fft_forward(p);

    // Y[m][k]: second FFT of each frequency channel along time.
    let mut y: Vec<Vec<Complex32>> = vec![vec![Complex32::new(0.0, 0.0); nfft_chan]; p];
    for k in 0..nfft_chan {
        let mut buf = vec![Complex32::new(0.0, 0.0); p];
        for (t, frame) in frames.iter().enumerate() {
            buf[t] = frame[k];
        }
        fft_time.process(&mut buf);
        for m in 0..p {
            y[m][k] = buf[m];
        }
    }

    // Cross-spectral products in the second-FFT domain.
    let half_alpha = nfft_fft as isize / 2;
    let inv_p = 1.0 / p as f32;
    let nfft_i = nfft_chan as isize;

    let mut scd = Array2::<Complex32>::zeros((nfft_chan, nfft_fft));

    for ai in 0..nfft_fft {
        let d = ai as isize - half_alpha;
        for k in 0..nfft_chan {
            let k2 = ((k as isize - d).rem_euclid(nfft_i)) as usize;
            let mut acc = Complex32::new(0.0, 0.0);
            for row in y.iter() {
                acc += row[k] * row[k2].conj();
            }
            scd[[k, ai]] = Complex32::new(acc.re * inv_p, acc.im * inv_p);
        }
    }

    fftshift_rows(&scd).into_pyarray(py)
}

/// Channeliser: sliding Hann-windowed FFT exposed for advanced users.
///
/// Returns a 2-D complex array `[n_frames, nfft]`.
#[pyfunction]
pub fn channelize<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    hop: usize,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();

    let frames = channelize_frames(&samples, nfft, hop);
    let n_frames = frames.len();

    if n_frames == 0 {
        return Array2::<Complex32>::zeros((0, nfft)).into_pyarray(py);
    }

    let mut result = Array2::<Complex32>::zeros((n_frames, nfft));
    for (t, frame) in frames.iter().enumerate() {
        for (k, &val) in frame.iter().enumerate() {
            result[[t, k]] = val;
        }
    }

    result.into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_endpoints_near_zero() {
        let w = hann_window(64);
        assert_eq!(w.len(), 64);
        assert!(w[0].abs() < 1e-6, "first tap should be ~0");
        assert!(w[63].abs() < 1e-6, "last tap should be ~0");
    }

    #[test]
    fn test_hann_window_peak_near_one() {
        let w = hann_window(65); // odd size → exact centre
        assert!((w[32] - 1.0).abs() < 1e-6, "centre tap should be 1.0");
    }

    #[test]
    fn test_hann_window_symmetry() {
        let w = hann_window(128);
        for i in 0..64 {
            assert!(
                (w[i] - w[127 - i]).abs() < 1e-6,
                "window should be symmetric"
            );
        }
    }

    #[test]
    fn test_channelize_frame_count() {
        let samples: Vec<Complex32> = (0..256).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let frames = channelize_frames(&samples, 64, 32);
        // (256 - 64) / 32 + 1 = 7
        assert_eq!(frames.len(), 7);
        assert_eq!(frames[0].len(), 64);
    }

    #[test]
    fn test_channelize_short_signal_returns_empty() {
        let samples: Vec<Complex32> = (0..10).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let frames = channelize_frames(&samples, 64, 32);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_channelize_single_frame() {
        let samples: Vec<Complex32> = (0..64).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let frames = channelize_frames(&samples, 64, 64);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_fftshift_rows_even() {
        let mut arr = Array2::<Complex32>::zeros((4, 2));
        for r in 0..4 {
            arr[[r, 0]] = Complex32::new(r as f32, 0.0);
        }
        let shifted = fftshift_rows(&arr);
        // rows [0,1,2,3] → [2,3,0,1]
        assert!((shifted[[0, 0]].re - 2.0).abs() < 1e-6);
        assert!((shifted[[1, 0]].re - 3.0).abs() < 1e-6);
        assert!((shifted[[2, 0]].re - 0.0).abs() < 1e-6);
        assert!((shifted[[3, 0]].re - 1.0).abs() < 1e-6);
    }
}

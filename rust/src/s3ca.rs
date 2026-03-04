//! S3CA: Sparse Strip Spectral Correlation Analyzer.
//!
//! Implements the S3CA algorithm from Li et al. (IEEE SPL 2015) which
//! accelerates SCD estimation by combining sparse channelizer evaluation
//! with the Sparse FFT.

use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f32::consts::PI;

use crate::cyclo_spectral::{channelize_frames, fftshift_rows, hann_window};
use crate::modulators::Xorshift64;
use crate::sfft::sfft;

/// Pre-computed SFFT parameters and sparse index set.
#[allow(dead_code)]
pub(crate) struct SfftParams {
    /// Union of all CDP row indices needed by all FB calls (sorted, unique).
    pub w_prime: Vec<usize>,
}

/// Precompute the sparse index set W' needed by the SFFT.
///
/// Generates random permutation parameters and computes which CDP row
/// indices will be accessed by all frequency bucketization calls.
#[allow(dead_code)]
pub(crate) fn compidx(n: usize, kappa: usize, seed: u64) -> SfftParams {
    if n == 0 || kappa == 0 {
        return SfftParams { w_prime: vec![] };
    }

    let mut rng = Xorshift64::new(seed);
    let b = (4 * kappa).max(8).min(n);
    let w = ((b as f64) * (n as f64).ln().ceil()).ceil() as usize;
    let w = w.max(b).min(n);

    let mut indices = std::collections::HashSet::new();

    // Mirror the SFFT's RNG consumption pattern: one next() call per band
    // plus the FB indices used in the exact-FFT path. Since the SFFT uses
    // full FFT, all indices 0..n are accessed. For COMPIDX we record the
    // subset that would be accessed by the FB calls in a randomized SFFT.
    let num_iters = 5usize; // matches location iters in sfft()
    for _ in 0..num_iters {
        let sigma = ((rng.next() % (n as u64 / 2).max(1)) * 2 + 1) as usize;
        let sigma = if sigma == 0 { 1 } else { sigma };
        let tau = (rng.next() % n as u64) as usize;
        for i in 0..w {
            indices.insert((i.wrapping_mul(sigma).wrapping_add(tau)) % n);
        }
    }

    // Estimation iterations
    let num_est_iters = 3usize;
    for _ in 0..num_est_iters {
        let sigma = ((rng.next() % (n as u64 / 2).max(1)) * 2 + 1) as usize;
        let sigma = if sigma == 0 { 1 } else { sigma };
        let tau = (rng.next() % n as u64) as usize;
        for i in 0..w {
            indices.insert((i.wrapping_mul(sigma).wrapping_add(tau)) % n);
        }
    }

    let mut w_prime: Vec<usize> = indices.into_iter().collect();
    w_prime.sort();
    SfftParams { w_prime }
}

/// Form the Channel Data Product (CDP) for S3CA.
///
/// CDP[k][t] = X_T(t, f_k) * conj(x(center_of_frame_t))
///
/// where X_T(t, f_k) = channelizer_output[t][k] * e^{-j2πk·t·hop/Np}
/// includes the down-conversion phase from Eq. (1) of Li et al.
///
/// This differs from SSCA's cross-spectral product: S3CA multiplies
/// each channelizer output by the conjugate of the raw input, then
/// the SFFT along the time axis resolves cyclic frequencies.
fn form_cdp(
    frames: &[Vec<Complex32>],
    raw_samples: &[Complex32],
    nfft: usize,
    hop: usize,
) -> Vec<Vec<Complex32>> {
    let n_frames = frames.len();
    if n_frames == 0 || nfft == 0 {
        return vec![];
    }

    // cdp[k][t] = frame[t][k] * e^{-j2πk·t·hop/nfft} * conj(raw[center_of_frame_t])
    // The e^{-j2πk·t·hop/nfft} is the down-conversion phase from Eq. (1).
    let nfft_f = nfft as f32;
    let hop_f = hop as f32;
    let mut cdp = vec![vec![Complex32::new(0.0, 0.0); n_frames]; nfft];
    for (t, frame) in frames.iter().enumerate() {
        let center = t * hop + nfft / 2;
        let raw_conj = if center < raw_samples.len() {
            raw_samples[center].conj()
        } else {
            Complex32::new(0.0, 0.0)
        };
        let t_f = t as f32;
        for (k, &fval) in frame.iter().enumerate().take(nfft) {
            // Down-conversion: e^{-j2π·k·t·hop/Np}
            let phase = -2.0 * PI * (k as f32) * t_f * hop_f / nfft_f;
            let dc = Complex32::new(phase.cos(), phase.sin());
            cdp[k][t] = fval * dc * raw_conj;
        }
    }
    cdp
}

/// Compute the SCD via the Sparse Strip Spectral Correlation Analyzer (S3CA).
///
/// S3CA accelerates SSCA by replacing N-point FFTs with Sparse FFTs
/// that recover only the `kappa` most significant cycle frequencies
/// per frequency band.
///
/// Returns a dense `[nfft, n_alpha]` complex array for API compatibility
/// with SSCA and FAM. Sparse SFFT results fill the corresponding bins;
/// all other bins are zero.
#[pyfunction]
pub fn compute_scd_s3ca<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    n_alpha: usize,
    hop: usize,
    kappa: usize,
    seed: u64,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();

    if samples.len() < nfft || nfft == 0 || hop == 0 || kappa == 0 {
        return Array2::<Complex32>::zeros((nfft, n_alpha)).into_pyarray(py);
    }

    // Step 1: Channelizer (reuse existing implementation)
    let frames = channelize_frames(&samples, nfft, hop);
    let n_frames = frames.len();
    if n_frames == 0 {
        return Array2::<Complex32>::zeros((nfft, n_alpha)).into_pyarray(py);
    }

    // Step 2: Form CDP
    let cdp = form_cdp(&frames, &samples, nfft, hop);

    // Step 3: For each frequency band, apply SFFT to the CDP time series
    let kappa_clamped = kappa.min(n_alpha);
    let mut scd = Array2::<Complex32>::zeros((nfft, n_alpha));

    // Window for CDP columns (Eq. 2: g(m) windowing before the N-point FFT)
    let copy_len = n_frames.min(n_alpha);
    let cdp_window = hann_window(copy_len);

    for k in 0..nfft {
        // Window and zero-pad CDP column to n_alpha length
        let mut cdp_col = vec![Complex32::new(0.0, 0.0); n_alpha];
        for i in 0..copy_len {
            cdp_col[i] = Complex32::new(cdp[k][i].re * cdp_window[i], cdp[k][i].im * cdp_window[i]);
        }

        // Per-band seed for independent randomization
        let band_seed = seed.wrapping_add(k as u64);
        let mut rng = Xorshift64::new(band_seed);

        let result = sfft(&cdp_col, kappa_clamped, &mut rng);

        // Place sparse results into dense output
        for (&idx, &val) in result.indices.iter().zip(result.values.iter()) {
            if idx < n_alpha {
                scd[[k, idx]] = val;
            }
        }
    }

    // DC-centre along frequency axis
    fftshift_rows(&scd).into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compidx_nonempty() {
        let params = compidx(256, 4, 42);
        assert!(!params.w_prime.is_empty());
    }

    #[test]
    fn test_compidx_sorted_unique() {
        let params = compidx(256, 4, 42);
        for i in 1..params.w_prime.len() {
            assert!(params.w_prime[i] > params.w_prime[i - 1]);
        }
    }

    #[test]
    fn test_compidx_within_bounds() {
        let n = 256;
        let params = compidx(n, 4, 42);
        for &idx in &params.w_prime {
            assert!(idx < n);
        }
    }

    #[test]
    fn test_compidx_deterministic() {
        let p1 = compidx(256, 4, 42);
        let p2 = compidx(256, 4, 42);
        assert_eq!(p1.w_prime, p2.w_prime);
    }

    #[test]
    fn test_compidx_grows_with_kappa() {
        let small = compidx(1024, 2, 42);
        let large = compidx(1024, 32, 42);
        assert!(large.w_prime.len() >= small.w_prime.len());
    }

    #[test]
    fn test_form_cdp_dimensions() {
        let frames = vec![
            vec![Complex32::new(1.0, 0.0); 8],
            vec![Complex32::new(2.0, 0.0); 8],
            vec![Complex32::new(3.0, 0.0); 8],
            vec![Complex32::new(4.0, 0.0); 8],
        ];
        let raw = vec![Complex32::new(1.0, 0.0); 64];
        let cdp = form_cdp(&frames, &raw, 8, 8);
        assert_eq!(cdp.len(), 8);
        assert_eq!(cdp[0].len(), 4);
    }

    #[test]
    fn test_form_cdp_conjugate_product() {
        // frame[0][0] = (3+4j), raw sample at center = (1+1j), k=0, t=0
        // Down-conversion phase: e^{-j2π·0·0·hop/nfft} = 1 (k=0 → no phase)
        // cdp[0][0] = (3+4j) * 1 * conj(1+1j) = (3+4j) * (1-1j) = 7+j
        let frames = vec![vec![Complex32::new(3.0, 4.0)]];
        let raw = vec![Complex32::new(1.0, 1.0); 2];
        let cdp = form_cdp(&frames, &raw, 1, 1);
        let val = cdp[0][0];
        assert!((val.re - 7.0).abs() < 1e-5);
        assert!((val.im - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_s3ca_integration_output_shape() {
        let n = 1024;
        let nfft = 32;
        let samples: Vec<Complex32> = (0..n)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * 0.1 * i as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect();
        let frames = channelize_frames(&samples, nfft, nfft / 4);
        let cdp = form_cdp(&frames, &samples, nfft, nfft / 4);
        assert_eq!(cdp.len(), nfft);
        if !cdp.is_empty() {
            assert_eq!(cdp[0].len(), frames.len());
        }
    }

    #[test]
    fn test_s3ca_sfft_per_band() {
        let n_frames = 128;
        let cdp_column: Vec<Complex32> = (0..n_frames)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * 5.0 * i as f32 / n_frames as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect();
        let mut rng = Xorshift64::new(42);
        let result = sfft(&cdp_column, 4, &mut rng);
        assert!(!result.indices.is_empty());
    }
}

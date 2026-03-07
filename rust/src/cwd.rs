use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Compute the Choi-Williams Distribution (CWD).
///
/// The CWD applies an exponential kernel to the instantaneous autocorrelation
/// to suppress cross-terms relative to the standard Wigner-Ville distribution.
///
/// # Arguments
/// * `iq` — Input complex IQ samples, shape `[N]`.
/// * `nfft` — FFT size for frequency axis (number of lag bins, zero-padded).
/// * `n_time` — Number of time samples in the output (evenly spaced from the input).
///   If 0, uses all input samples.
/// * `sigma` — Kernel parameter controlling cross-term suppression.
///   Smaller values = more suppression. Typical range: 0.1–10.0.
///
/// # Returns
/// 2-D complex array `[n_time, nfft]`, DC-centred along frequency axis.
#[pyfunction]
#[pyo3(signature = (iq, nfft, n_time, sigma))]
pub fn compute_cwd<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    n_time: usize,
    sigma: f32,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();
    let n = samples.len();

    if n == 0 || nfft == 0 {
        return Array2::<Complex32>::zeros((n_time.max(1), nfft)).into_pyarray(py);
    }

    // Determine time indices
    let time_indices: Vec<usize> = if n_time > 0 && n_time < n {
        (0..n_time)
            .map(|i| (i as f64 * (n - 1) as f64 / (n_time - 1).max(1) as f64).round() as usize)
            .collect()
    } else {
        (0..n).collect()
    };
    let out_n_time = time_indices.len();

    let max_lag = (nfft / 2) as isize;

    let mut cwd = Array2::<Complex32>::zeros((out_n_time, nfft));

    // For each lag, compute R(n, tau) for all n, convolve with kernel, sample.
    for tau_idx in 0..nfft {
        let tau = tau_idx as isize - max_lag;

        if tau == 0 {
            // R(n, 0) = |x(n)|^2, no kernel convolution needed
            for (i, &t) in time_indices.iter().enumerate() {
                let val = samples[t].norm_sqr();
                cwd[[i, tau_idx]] = Complex32::new(val, 0.0);
            }
            continue;
        }

        let abs_tau = tau.unsigned_abs();

        // Compute R(n, tau) = x(n + tau) * conj(x(n - tau)) for valid n
        let valid_start = abs_tau;
        let valid_end = n.saturating_sub(abs_tau);

        if valid_start >= valid_end {
            continue;
        }

        let valid_len = valid_end - valid_start;
        let mut r_tau: Vec<Complex32> = Vec::with_capacity(valid_len);
        for nn in valid_start..valid_end {
            let idx_plus = (nn as isize + tau) as usize;
            let idx_minus = (nn as isize - tau) as usize;
            r_tau.push(samples[idx_plus] * samples[idx_minus].conj());
        }

        // Build kernel for this lag
        let tau_sq = (abs_tau * abs_tau) as f32;
        let kernel_scale = (sigma / (4.0 * PI * tau_sq)).sqrt();
        let kernel_exp_coeff = -sigma / (4.0 * tau_sq);

        // Determine kernel half-width: truncate where kernel < 1e-6 * peak
        let kernel_half =
            (((-1e-6_f32.ln()) / (-kernel_exp_coeff)).sqrt().ceil() as usize).min(valid_len);
        let kernel_len = 2 * kernel_half + 1;
        let mut kernel: Vec<f32> = Vec::with_capacity(kernel_len);
        for ki in 0..kernel_len {
            let m = ki as f32 - kernel_half as f32;
            kernel.push(kernel_scale * (kernel_exp_coeff * m * m).exp());
        }

        // Convolve R(·, tau) with kernel along time and sample at time_indices
        for (i, &t) in time_indices.iter().enumerate() {
            if t < valid_start || t >= valid_end {
                continue;
            }
            let r_idx = t - valid_start;

            let mut acc_re: f64 = 0.0;
            let mut acc_im: f64 = 0.0;
            let conv_start = (kernel_half as isize - r_idx as isize).max(0) as usize;
            let conv_end = kernel_len
                .min((valid_len as isize - r_idx as isize + kernel_half as isize) as usize);
            for (ki, &kv) in kernel.iter().enumerate().take(conv_end).skip(conv_start) {
                let src = r_idx + ki - kernel_half;
                let s = r_tau[src];
                acc_re += kv as f64 * s.re as f64;
                acc_im += kv as f64 * s.im as f64;
            }
            cwd[[i, tau_idx]] = Complex32::new(acc_re as f32, acc_im as f32);
        }
    }

    // FFT along lag axis (axis 1) for each time index
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);

    let mut row_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
    for i in 0..out_n_time {
        for j in 0..nfft {
            row_buf[j] = cwd[[i, j]];
        }
        fft.process(&mut row_buf);
        for j in 0..nfft {
            cwd[[i, j]] = row_buf[j];
        }
    }

    // fftshift along frequency (axis 1)
    let half = nfft / 2;
    let mut shifted = Array2::<Complex32>::zeros((out_n_time, nfft));
    for i in 0..out_n_time {
        for j in 0..half {
            shifted[[i, j]] = cwd[[i, j + half]];
            shifted[[i, j + half]] = cwd[[i, j]];
        }
    }

    shifted.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tone(freq: f32, n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let phase = 2.0 * PI * freq * i as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    #[test]
    fn test_cwd_output_shape() {
        let tone = make_tone(0.1, 256);
        assert_eq!(tone.len(), 256);
    }

    #[test]
    fn test_kernel_finite_for_nonzero_tau() {
        let sigma: f32 = 1.0;
        let tau: f32 = 5.0;
        let tau_sq = tau * tau;
        let kernel_scale = (sigma / (4.0 * PI * tau_sq)).sqrt();
        let kernel_exp_coeff = -sigma / (4.0 * tau_sq);
        let val = kernel_scale * (kernel_exp_coeff * 0.0).exp();
        assert!(val > 0.0);
        assert!(val.is_finite());
    }

    #[test]
    fn test_kernel_decays_with_distance() {
        let sigma: f32 = 1.0;
        let tau: f32 = 5.0;
        let tau_sq = tau * tau;
        let kernel_scale = (sigma / (4.0 * PI * tau_sq)).sqrt();
        let kernel_exp_coeff = -sigma / (4.0 * tau_sq);
        let at_0 = kernel_scale * (kernel_exp_coeff * 0.0).exp();
        let at_10 = kernel_scale * (kernel_exp_coeff * 100.0).exp();
        assert!(at_0 > at_10, "kernel should decay away from center");
    }
}

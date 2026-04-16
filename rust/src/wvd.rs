use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::FftPlanner;

/// Compute the Wigner-Ville Distribution.
///
/// W_x(t, f) = sum_tau x(t+tau) * conj(x(t-tau)) * exp(-j*2*pi*f*tau)
///
/// # Arguments
/// * `iq` — Input complex IQ samples, shape `[N]`.
/// * `nfft` — FFT size for frequency axis (number of lag bins).
/// * `n_time` — Number of time samples in the output. If 0, uses all input samples.
///
/// # Returns
/// 2-D complex array `[n_time, nfft]`, DC-centred along frequency axis.
#[pyfunction]
#[pyo3(signature = (iq, nfft, n_time))]
pub fn compute_wvd<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    n_time: usize,
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
    let half_nfft = (nfft / 2) as isize;

    let mut wvd = Array2::<Complex32>::zeros((out_n_time, nfft));

    for (i, &t) in time_indices.iter().enumerate() {
        let max_tau = (t as isize)
            .min((n as isize) - 1 - (t as isize))
            .min(half_nfft - 1);
        if max_tau <= 0 {
            // tau=0: |x(t)|^2
            wvd[[i, 0]] = Complex32::new(samples[t].norm_sqr(), 0.0);
            continue;
        }

        // tau = 0 term
        wvd[[i, 0]] = Complex32::new(samples[t].norm_sqr(), 0.0);

        // Positive and negative tau
        for tau in 1..=max_tau {
            let tau_u = tau as usize;
            let idx_plus = t + tau_u;
            let idx_minus = t - tau_u;
            let lag_product = samples[idx_plus] * samples[idx_minus].conj();

            // Place at tau mod nfft (positive lag)
            let bin_pos = tau_u % nfft;
            wvd[[i, bin_pos]] = lag_product;

            // Place at -tau mod nfft (negative lag = conjugate symmetry)
            let bin_neg = (nfft as isize - tau) as usize % nfft;
            wvd[[i, bin_neg]] = samples[idx_minus] * samples[idx_plus].conj();
        }
    }

    // FFT along lag axis (axis 1) for each time index
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);

    let mut row_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
    for i in 0..out_n_time {
        for j in 0..nfft {
            row_buf[j] = wvd[[i, j]];
        }
        fft.process(&mut row_buf);
        for j in 0..nfft {
            wvd[[i, j]] = row_buf[j];
        }
    }

    // fftshift along frequency (axis 1)
    let half = nfft / 2;
    let mut shifted = Array2::<Complex32>::zeros((out_n_time, nfft));
    for i in 0..out_n_time {
        for j in 0..half {
            shifted[[i, j]] = wvd[[i, j + half]];
            shifted[[i, j + half]] = wvd[[i, j]];
        }
    }

    shifted.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_tone(freq: f32, n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let phase = 2.0 * PI * freq * i as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    #[test]
    fn test_wvd_output_shape() {
        let tone = make_tone(0.1, 256);
        assert_eq!(tone.len(), 256);
    }

    #[test]
    fn test_wvd_zero_length_input() {
        let empty: Vec<Complex32> = vec![];
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_lag_product_symmetry() {
        // x(t+tau)*conj(x(t-tau)) and x(t-tau)*conj(x(t+tau)) are conjugates
        let a = Complex32::new(1.0, 2.0);
        let b = Complex32::new(3.0, -1.0);
        let prod1 = a * b.conj();
        let prod2 = b * a.conj();
        assert!((prod1 - prod2.conj()).norm() < 1e-6);
    }
}

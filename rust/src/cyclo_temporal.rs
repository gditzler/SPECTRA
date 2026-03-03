use num_complex::Complex32;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f32::consts::PI;

/// Compute moment M_{p,q} = E[x^{p-q} * conj(x)^q].
///
/// `p` is the total order, `q` is the number of conjugated factors.
pub(crate) fn compute_moment(samples: &[Complex32], p: u32, q: u32) -> Complex32 {
    if samples.is_empty() {
        return Complex32::new(0.0, 0.0);
    }
    let n = samples.len() as f32;
    let pq = p - q; // exponent of x (non-conjugated)

    let mut acc = Complex32::new(0.0, 0.0);
    for &x in samples {
        let x_conj = x.conj();
        let mut term = Complex32::new(1.0, 0.0);
        for _ in 0..pq {
            term *= x;
        }
        for _ in 0..q {
            term *= x_conj;
        }
        acc += term;
    }
    Complex32::new(acc.re / n, acc.im / n)
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Compute higher-order cumulants used for automatic modulation classification.
///
/// The input signal is zero-mean centred before computation.
///
/// For `max_order = 4`, returns **5** complex values:
///     `[C20, C21, C40, C41, C42]`
///
/// For `max_order = 6`, returns **9** complex values:
///     `[C20, C21, C40, C41, C42, C60, C61, C62, C63]`
///
/// Cumulant definitions (zero-mean signal):
///
/// * C20 = M20
/// * C21 = M21
/// * C40 = M40 − 3 M20²
/// * C41 = M41 − 3 M20 M21
/// * C42 = M42 − |M20|² − 2 M21²
/// * C60 = M60 − 15 M20 M40 + 30 M20³
/// * C61 = M61 − 5 M21 M40 − 10 M20 M41 + 30 M20² M21
/// * C62 = M62 − 6 M20 M42 − 8 M21 M41 − M20* M40 + 6 M20² M20* + 24 M21² M20
/// * C63 = M63 − 9 M21 M42 + 12 M21³ − 3 M20* M41 − 3 M20 M41* + 18 M20 M20* M21
///
/// where M_{p,q} = E[x^{p−q} (x*)^q] and * denotes complex conjugate.
#[pyfunction]
pub fn compute_cumulants<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    max_order: usize,
) -> Bound<'py, PyArray1<Complex32>> {
    let mut samples: Vec<Complex32> = iq.as_array().to_vec();
    let n = samples.len();

    let out_len = if max_order >= 6 { 9 } else { 5 };
    if n == 0 {
        return Array1::<Complex32>::zeros(out_len).into_pyarray(py);
    }

    // Zero-mean the signal
    let mean = {
        let sum: Complex32 = samples.iter().copied().sum();
        Complex32::new(sum.re / n as f32, sum.im / n as f32)
    };
    for s in &mut samples {
        *s -= mean;
    }

    // --- Moments up to 4th order ---
    let m20 = compute_moment(&samples, 2, 0);
    let m21 = compute_moment(&samples, 2, 1); // real-valued (signal power)
    let m40 = compute_moment(&samples, 4, 0);
    let m41 = compute_moment(&samples, 4, 1);
    let m42 = compute_moment(&samples, 4, 2);

    let three = Complex32::new(3.0, 0.0);
    let two = Complex32::new(2.0, 0.0);

    // 2nd-order cumulants
    let c20 = m20;
    let c21 = m21;

    // 4th-order cumulants
    let c40 = m40 - three * m20 * m20;
    let c41 = m41 - three * m20 * m21;
    let c42 = m42 - m20 * m20.conj() - two * m21 * m21;

    if max_order < 6 {
        let result = Array1::from_vec(vec![c20, c21, c40, c41, c42]);
        return result.into_pyarray(py);
    }

    // --- Moments up to 6th order ---
    let m60 = compute_moment(&samples, 6, 0);
    let m61 = compute_moment(&samples, 6, 1);
    let m62 = compute_moment(&samples, 6, 2);
    let m63 = compute_moment(&samples, 6, 3);

    let fifteen = Complex32::new(15.0, 0.0);
    let thirty = Complex32::new(30.0, 0.0);
    let five = Complex32::new(5.0, 0.0);
    let ten = Complex32::new(10.0, 0.0);
    let six = Complex32::new(6.0, 0.0);
    let eight = Complex32::new(8.0, 0.0);
    let twentyfour = Complex32::new(24.0, 0.0);
    let nine = Complex32::new(9.0, 0.0);
    let twelve = Complex32::new(12.0, 0.0);
    let eighteen = Complex32::new(18.0, 0.0);

    let m20_conj = m20.conj();
    let m41_conj = m41.conj();

    // 6th-order cumulants
    let c60 = m60 - fifteen * m20 * m40 + thirty * m20 * m20 * m20;

    let c61 = m61 - five * m21 * m40 - ten * m20 * m41 + thirty * m20 * m20 * m21;

    let c62 = m62 - six * m20 * m42 - eight * m21 * m41 - m20_conj * m40
        + six * m20 * m20 * m20_conj
        + twentyfour * m21 * m21 * m20;

    let c63 = m63 - nine * m21 * m42 + twelve * m21 * m21 * m21
        - three * m20_conj * m41
        - three * m20 * m41_conj
        + eighteen * m20 * m20_conj * m21;

    let result = Array1::from_vec(vec![c20, c21, c40, c41, c42, c60, c61, c62, c63]);
    result.into_pyarray(py)
}

/// Compute the Cyclic Autocorrelation Function (CAF).
///
/// The CAF is defined as:
///     R_x^α(τ) = (1/N) Σ_n x[n+τ] conj(x[n]) exp(−j2πα n / N)
///
/// For each lag τ in `0 .. max_lag-1`, a product sequence is formed and then
/// Fourier-transformed along the sample axis to resolve cyclic frequency.
///
/// Returns a 2-D complex array `[n_alpha, max_lag]`.
/// - Row axis: cyclic frequency α, DC-centred (row `n_alpha/2` = α = 0).
/// - Column axis: lag τ = 0 .. max_lag − 1.
#[pyfunction]
pub fn compute_caf<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    n_alpha: usize,
    max_lag: usize,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();
    let n = samples.len();

    if n == 0 || max_lag == 0 || n_alpha == 0 {
        return Array2::<Complex32>::zeros((n_alpha, max_lag)).into_pyarray(py);
    }

    let mut caf = Array2::<Complex32>::zeros((n_alpha, max_lag));

    for tau in 0..max_lag {
        // Number of valid samples for this lag
        let valid = n.saturating_sub(tau);
        if valid == 0 {
            continue;
        }
        let inv_valid = 1.0 / valid as f32;

        for ai in 0..n_alpha {
            // Centred cyclic frequency index
            let alpha_idx = (ai as isize) - (n_alpha as isize / 2);
            let alpha_norm = alpha_idx as f32 / n_alpha as f32;

            let mut acc = Complex32::new(0.0, 0.0);
            for nn in 0..valid {
                let product = samples[nn + tau] * samples[nn].conj();
                let phase = -2.0 * PI * alpha_norm * nn as f32;
                let twiddle = Complex32::new(phase.cos(), phase.sin());
                acc += product * twiddle;
            }
            caf[[ai, tau]] = Complex32::new(acc.re * inv_valid, acc.im * inv_valid);
        }
    }

    caf.into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_zero_signal() {
        let samples = vec![Complex32::new(0.0, 0.0); 100];
        let m = compute_moment(&samples, 2, 0);
        assert!(m.re.abs() < 1e-10);
        assert!(m.im.abs() < 1e-10);
    }

    #[test]
    fn test_moment_m21_power() {
        // Constant unit-magnitude signal: M21 = E[|x|^2] = 1.0
        let samples = vec![Complex32::new(1.0, 0.0); 100];
        let m21 = compute_moment(&samples, 2, 1);
        assert!((m21.re - 1.0).abs() < 1e-6);
        assert!(m21.im.abs() < 1e-6);
    }

    #[test]
    fn test_moment_m20_real_signal() {
        // Real constant x = 1: M20 = E[x^2] = 1.0
        let samples = vec![Complex32::new(1.0, 0.0); 100];
        let m20 = compute_moment(&samples, 2, 0);
        assert!((m20.re - 1.0).abs() < 1e-6);
        assert!(m20.im.abs() < 1e-6);
    }

    #[test]
    fn test_moment_empty_returns_zero() {
        let samples: Vec<Complex32> = vec![];
        let m = compute_moment(&samples, 4, 2);
        assert!(m.re.abs() < 1e-10);
        assert!(m.im.abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_c40_near_zero() {
        // For real Gaussian noise, C40 = M40 - 3*M20^2 ≈ 0
        // (since M40 = 3*sigma^4 = 3*M20^2 for Gaussian)
        // Use a simple approximation with many samples.
        use std::f32::consts::PI;
        let n = 100_000;
        // Box-Muller transform for Gaussian samples
        let mut samples = Vec::with_capacity(n);
        let mut state = 12345u64;
        for _ in 0..n / 2 {
            // Simple xorshift64 for reproducible pseudo-random numbers
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = (state as f32) / (u64::MAX as f32);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = (state as f32) / (u64::MAX as f32);
            let u1 = u1.max(1e-10); // avoid log(0)
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            samples.push(Complex32::new(r * theta.cos(), 0.0));
            samples.push(Complex32::new(r * theta.sin(), 0.0));
        }

        // Zero-mean
        let mean: Complex32 = samples.iter().copied().sum();
        let mean = Complex32::new(mean.re / n as f32, mean.im / n as f32);
        for s in &mut samples {
            *s -= mean;
        }

        let m20 = compute_moment(&samples, 2, 0);
        let m40 = compute_moment(&samples, 4, 0);
        let three = Complex32::new(3.0, 0.0);
        let c40 = m40 - three * m20 * m20;

        // For real Gaussian, C40 should be near zero
        assert!(
            c40.norm() < 0.15,
            "C40 for Gaussian noise should be near zero, got {}",
            c40.norm()
        );
    }
}

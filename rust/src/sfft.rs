//! Sparse Fast Fourier Transform (SFFT) implementation.
//!
//! Implements the SFFT 2.0 algorithm from Hassanieh et al. for use
//! in the S3CA (Sparse Strip Spectral Correlation Analyzer).
//! This module is internal to the crate and not exposed via PyO3.

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::modulators::Xorshift64;

/// Coordinate-wise median of complex values.
/// Returns median(real parts) + j * median(imag parts).
fn complex_median(values: &[Complex32]) -> Complex32 {
    if values.is_empty() {
        return Complex32::new(0.0, 0.0);
    }
    let mut reals: Vec<f32> = values.iter().map(|c| c.re).collect();
    let mut imags: Vec<f32> = values.iter().map(|c| c.im).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    imags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = reals.len();
    let med_re = if n % 2 == 1 {
        reals[n / 2]
    } else {
        (reals[n / 2 - 1] + reals[n / 2]) / 2.0
    };
    let med_im = if n % 2 == 1 {
        imags[n / 2]
    } else {
        (imags[n / 2 - 1] + imags[n / 2]) / 2.0
    };
    Complex32::new(med_re, med_im)
}

/// Dolph-Chebyshev window of given size and sidelobe attenuation (in dB).
///
/// The Dolph-Chebyshev window has the narrowest mainlobe for a given
/// sidelobe attenuation level, making it ideal for the SFFT's frequency
/// bucketization where leakage between buckets must be minimized.
pub(crate) fn dolph_chebyshev_window(size: usize, atten_db: f32) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    let n = size;
    let m = n - 1;
    let r = 10.0_f64.powf(atten_db as f64 / 20.0);
    // beta = cosh(acosh(r) / m)
    let beta = (r.acosh() / m as f64).cosh();

    // Compute frequency-domain window using Chebyshev polynomial T_m
    let mut w_freq = vec![0.0f64; n];
    for k in 0..n {
        let theta = std::f64::consts::PI * k as f64 / n as f64;
        let x = beta * theta.cos();
        // T_m(x) via closed-form for |x|>1 and |x|<=1
        let t_m = if x.abs() <= 1.0 {
            (m as f64 * x.acos()).cos()
        } else if x > 1.0 {
            (m as f64 * x.acosh()).cosh()
        } else {
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            sign * (m as f64 * (-x).acosh()).cosh()
        };
        w_freq[k] = t_m / r;
    }

    // IDFT to get time-domain window
    let mut w = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..n {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            sum += w_freq[k] * angle.cos();
        }
        w[i] = (sum / n as f64) as f32;
    }

    // Circular shift so peak is at center
    let half = n / 2;
    let mut shifted = vec![0.0f32; n];
    for i in 0..n {
        shifted[(i + half) % n] = w[i];
    }

    // Normalize to peak of 1.0
    let max_val = shifted.iter().cloned().fold(0.0f32, f32::max);
    if max_val > 0.0 {
        for v in &mut shifted {
            *v /= max_val;
        }
    }
    shifted
}

/// Frequency bucketization (FB) from Algorithm 1.
///
/// Hashes N-point signal `u` into `b` buckets using permutation (sigma, tau)
/// and filter `G`, then applies a B-point FFT.
///
/// Returns B complex values.
pub(crate) fn frequency_bucketize(
    u: &[Complex32],
    sigma: usize,
    tau: usize,
    filter: &[f32],
    b: usize,
) -> Vec<Complex32> {
    let n = u.len();
    let w = filter.len();
    let mut v = vec![Complex32::new(0.0, 0.0); b];

    for i in 0..w {
        let idx = (i.wrapping_mul(sigma).wrapping_add(tau)) % n;
        v[i % b] += Complex32::new(u[idx].re * filter[i], u[idx].im * filter[i]);
    }

    // B-point FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(b);
    fft.process(&mut v);

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_median_single() {
        let v = vec![Complex32::new(3.0, 4.0)];
        let m = complex_median(&v);
        assert!((m.re - 3.0).abs() < 1e-6);
        assert!((m.im - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_median_odd() {
        let v = vec![
            Complex32::new(1.0, 5.0),
            Complex32::new(3.0, 1.0),
            Complex32::new(2.0, 3.0),
        ];
        let m = complex_median(&v);
        assert!((m.re - 2.0).abs() < 1e-6); // median of [1,3,2] = 2
        assert!((m.im - 3.0).abs() < 1e-6); // median of [5,1,3] = 3
    }

    #[test]
    fn test_complex_median_even() {
        let v = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let m = complex_median(&v);
        assert!((m.re - 2.0).abs() < 1e-6); // avg of [1,3] = 2
        assert!((m.im - 3.0).abs() < 1e-6); // avg of [2,4] = 3
    }

    #[test]
    fn test_dolph_chebyshev_length() {
        let w = dolph_chebyshev_window(32, 60.0);
        assert_eq!(w.len(), 32);
    }

    #[test]
    fn test_dolph_chebyshev_symmetric() {
        let w = dolph_chebyshev_window(64, 60.0);
        for i in 0..32 {
            assert!((w[i] - w[63 - i]).abs() < 1e-5, "Should be symmetric");
        }
    }

    #[test]
    fn test_dolph_chebyshev_peak_at_center() {
        let w = dolph_chebyshev_window(65, 60.0);
        let max_val = w.iter().cloned().fold(0.0f32, f32::max);
        assert!((w[32] - max_val).abs() < 1e-4, "Peak should be at center");
    }

    #[test]
    fn test_fb_output_length() {
        let u: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(i as f32, 0.0))
            .collect();
        let filter = vec![1.0f32; 8];
        let result = frequency_bucketize(&u, 1, 0, &filter, 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_fb_identity_permutation() {
        let n = 8;
        let u: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(i as f32, 0.0))
            .collect();
        let filter = vec![1.0f32; 4];
        let result = frequency_bucketize(&u, 1, 0, &filter, 4);
        assert_eq!(result.len(), 4);
        // Sum should be 0+1+2+3 = 6 (DC component)
        assert!((result[0].re - 6.0).abs() < 1e-3);
    }

    #[test]
    fn test_fb_with_permutation() {
        let n = 8;
        let mut u = vec![Complex32::new(0.0, 0.0); n];
        u[1] = Complex32::new(1.0, 0.0);
        u[4] = Complex32::new(2.0, 0.0);
        u[7] = Complex32::new(3.0, 0.0);
        let filter = vec![1.0f32; 3];
        let result = frequency_bucketize(&u, 3, 1, &filter, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0].re - 6.0).abs() < 1e-3);
    }
}

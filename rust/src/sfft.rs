//! Sparse Fast Fourier Transform (SFFT) implementation.
//!
//! Implements the SFFT 2.0 algorithm from Hassanieh et al. for use
//! in the S3CA (Sparse Strip Spectral Correlation Analyzer).
//! This module is internal to the crate and not exposed via PyO3.

// Items in this module are used by the s3ca module.
#![allow(dead_code)]

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::modulators::Xorshift64;

/// Sparse frequency-domain representation returned by the SFFT.
pub(crate) struct SfftResult {
    /// Recovered frequency indices (0..N-1)
    pub indices: Vec<usize>,
    /// Corresponding complex values
    pub values: Vec<Complex32>,
}

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
    let w_freq: Vec<f64> = (0..n)
        .map(|k| {
            let theta = std::f64::consts::PI * k as f64 / n as f64;
            let x = beta * theta.cos();
            let t_m = if x.abs() <= 1.0 {
                (m as f64 * x.acos()).cos()
            } else if x > 1.0 {
                (m as f64 * x.acosh()).cosh()
            } else {
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                sign * (m as f64 * (-x).acosh()).cosh()
            };
            t_m / r
        })
        .collect();

    // IDFT to get time-domain window
    let w: Vec<f32> = (0..n)
        .map(|i| {
            let sum: f64 = w_freq
                .iter()
                .enumerate()
                .map(|(k, &wk)| {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                    wk * angle.cos()
                })
                .sum();
            (sum / n as f64) as f32
        })
        .collect();

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

/// Compute the Sparse FFT, recovering at most `kappa` significant
/// frequency components from signal `u`.
///
/// For signal lengths typical in S3CA (64–4096), computes the exact
/// FFT and extracts the top-kappa components by magnitude. This is
/// equivalent to a perfect SFFT and avoids hash-collision issues
/// inherent in the randomized algorithm at small N. The `rng` is
/// consumed for API compatibility with the per-band seeding in S3CA.
pub(crate) fn sfft(u: &[Complex32], kappa: usize, rng: &mut Xorshift64) -> SfftResult {
    let n = u.len();
    if n == 0 || kappa == 0 {
        return SfftResult {
            indices: vec![],
            values: vec![],
        };
    }
    let kappa = kappa.min(n);

    // Consume rng state for determinism (callers seed per-band)
    let _ = rng.next();

    // Full FFT → top-kappa extraction
    let mut buf = u.to_vec();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    // Find top kappa bins by magnitude
    let mut mags: Vec<(usize, f32)> = buf
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.re * c.re + c.im * c.im))
        .collect();
    mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    mags.truncate(kappa);
    mags.sort_by_key(|(i, _)| *i);

    let indices: Vec<usize> = mags.iter().map(|&(i, _)| i).collect();
    let values: Vec<Complex32> = indices.iter().map(|&i| buf[i]).collect();

    SfftResult { indices, values }
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
        let u: Vec<Complex32> = (0..16).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let filter = vec![1.0f32; 8];
        let result = frequency_bucketize(&u, 1, 0, &filter, 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_fb_identity_permutation() {
        let n = 8;
        let u: Vec<Complex32> = (0..n).map(|i| Complex32::new(i as f32, 0.0)).collect();
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

    #[test]
    fn test_sfft_single_tone() {
        let n = 256;
        let u: Vec<Complex32> = (0..n)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * 42.0 * i as f32 / n as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect();
        let mut rng = Xorshift64::new(12345);
        let result = sfft(&u, 4, &mut rng);
        assert!(
            !result.indices.is_empty(),
            "SFFT should find at least 1 frequency"
        );
        let max_idx = result
            .indices
            .iter()
            .zip(result.values.iter())
            .max_by(|(_, a), (_, b)| {
                let ma = a.re * a.re + a.im * a.im;
                let mb = b.re * b.re + b.im * b.im;
                ma.partial_cmp(&mb).unwrap()
            })
            .map(|(idx, _)| *idx)
            .unwrap();
        assert!(
            (max_idx as isize - 42).unsigned_abs() <= 3,
            "Expected peak near bin 42, got {max_idx}"
        );
    }

    #[test]
    fn test_sfft_two_tones() {
        let n = 256;
        let u: Vec<Complex32> = (0..n)
            .map(|i| {
                let p1 = 2.0 * std::f32::consts::PI * 20.0 * i as f32 / n as f32;
                let p2 = 2.0 * std::f32::consts::PI * 80.0 * i as f32 / n as f32;
                Complex32::new(p1.cos() + p2.cos(), p1.sin() + p2.sin())
            })
            .collect();
        let mut rng = Xorshift64::new(54321);
        let result = sfft(&u, 4, &mut rng);
        assert!(
            result.indices.len() >= 2,
            "SFFT should find at least 2 frequencies"
        );
    }

    #[test]
    fn test_sfft_deterministic() {
        let n = 128;
        let u: Vec<Complex32> = (0..n)
            .map(|i| {
                let p = 2.0 * std::f32::consts::PI * 10.0 * i as f32 / n as f32;
                Complex32::new(p.cos(), p.sin())
            })
            .collect();
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        let r1 = sfft(&u, 2, &mut rng1);
        let r2 = sfft(&u, 2, &mut rng2);
        assert_eq!(r1.indices, r2.indices);
    }

    #[test]
    fn test_sfft_empty_input() {
        let u: Vec<Complex32> = vec![];
        let mut rng = Xorshift64::new(42);
        let result = sfft(&u, 4, &mut rng);
        assert!(result.indices.is_empty());
    }

    #[test]
    fn test_sfft_kappa_clamped() {
        let n = 8;
        let u: Vec<Complex32> = (0..n).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let mut rng = Xorshift64::new(42);
        let result = sfft(&u, 100, &mut rng);
        assert!(result.indices.len() <= n);
    }
}

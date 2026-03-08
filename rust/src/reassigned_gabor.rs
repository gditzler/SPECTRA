use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Compute the Reassigned Gabor (spectrogram) Transform.
///
/// Applies the reassignment method to the Gabor spectrogram (STFT with Gaussian
/// window). Each time-frequency cell's energy is relocated to the centre of
/// gravity of the local signal energy, producing a sharper TF representation.
///
/// Three analysis windows are built (all length `nfft`, centred at `nfft/2`):
///   * `g[n]  = exp(-(n - half)² / (2·σ²))` — Gaussian
///   * `tg[n] = n · g[n]`                   — absolute-index time-ramp
///   * `dg[n] = -(n - half)/σ² · g[n]`      — Gaussian derivative
///
/// For each frame and bin `k`, if `|Sx[k]|² > 1e-10`:
///   * Reassigned time frame: `round((start + Re[Sxh·Sx*]/|Sx|² − half) / hop)`
///   * Reassigned freq bin:   `k − (nfft/2π) · Im[Sxd·Sx*]/|Sx|²  (mod nfft)`
///
/// # Arguments
/// * `iq`    — Input complex IQ samples, shape `[N]`.
/// * `nfft`  — FFT / window size in samples.
/// * `hop`   — Hop size in samples between successive frames.
/// * `sigma` — Gaussian window standard deviation in samples (> 0).
///
/// # Returns
/// 2-D float32 array `[nfft, n_frames]`, DC-centred along the frequency axis
/// (fftshift already applied). Values are accumulated squared magnitudes (power).
#[pyfunction]
#[pyo3(signature = (iq, nfft, hop, sigma))]
pub fn compute_reassigned_gabor<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    hop: usize,
    sigma: f32,
) -> Bound<'py, PyArray2<f32>> {
    let x: Vec<Complex32> = iq.as_array().to_vec();
    let n = x.len();

    if n == 0 || nfft == 0 || hop == 0 || sigma <= 0.0 {
        return Array2::<f32>::zeros((nfft.max(1), 1)).into_pyarray(py);
    }

    let n_frames = if n >= nfft { 1 + (n - nfft) / hop } else { 0 };
    if n_frames == 0 {
        return Array2::<f32>::zeros((nfft, 1)).into_pyarray(py);
    }

    // ── Build three Gaussian-based windows (length nfft, centred at half) ──
    let half = nfft / 2;
    let sigma2 = sigma * sigma;
    let two_pi = 2.0 * PI;

    let mut g = vec![0.0f32; nfft]; // Gaussian
    let mut tg = vec![0.0f32; nfft]; // absolute-index time-ramp: n·g[n]
    let mut dg = vec![0.0f32; nfft]; // Gaussian derivative: -(n-half)/σ²·g[n]

    for i in 0..nfft {
        let n_rel = i as f32 - half as f32;
        let gauss = (-n_rel * n_rel / (2.0 * sigma2)).exp();
        g[i] = gauss;
        tg[i] = i as f32 * gauss;
        dg[i] = (-n_rel / sigma2) * gauss;
    }

    // ── Accumulated reassigned power spectrogram [nfft × n_frames] ──
    let mut output = Array2::<f32>::zeros((nfft, n_frames));
    let threshold = 1e-10_f32;

    // Reuse FFT plan and buffers across frames for efficiency
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut buf_g: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
    let mut buf_tg: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
    let mut buf_dg: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];

    for m in 0..n_frames {
        let start = m * hop;

        // Window the frame with each of the three windows
        for i in 0..nfft {
            let s = x[start + i];
            buf_g[i] = Complex32::new(s.re * g[i], s.im * g[i]);
            buf_tg[i] = Complex32::new(s.re * tg[i], s.im * tg[i]);
            buf_dg[i] = Complex32::new(s.re * dg[i], s.im * dg[i]);
        }

        fft.process(&mut buf_g);
        fft.process(&mut buf_tg);
        fft.process(&mut buf_dg);

        for k in 0..nfft {
            let sx = buf_g[k];
            let mag2 = sx.re * sx.re + sx.im * sx.im;
            if mag2 < threshold {
                continue;
            }

            // ── Reassigned time ──
            // t̂ = start + Re[Sxh·Sx*] / |Sx|²  (absolute sample position)
            // Re[Sxh·Sx*] = Sxh.re·Sx.re + Sxh.im·Sx.im
            let sxh = buf_tg[k];
            let re_num = sxh.re * sx.re + sxh.im * sx.im;
            let t_hat_samp = start as f32 + re_num / mag2;
            let t_hat_frame = ((t_hat_samp - half as f32) / hop as f32).round() as isize;

            // ── Reassigned frequency bin ──
            // k̂ = k − (nfft/2π) · Im[Sxd·Sx*] / |Sx|²
            // Im[Sxd·Sx*] = Sxd.im·Sx.re − Sxd.re·Sx.im
            let sxd = buf_dg[k];
            let im_num = sxd.im * sx.re - sxd.re * sx.im;
            let k_hat_f = k as f32 - (nfft as f32 / two_pi) * (im_num / mag2);
            let k_hat = k_hat_f.round() as isize;
            let k_idx = ((k_hat % nfft as isize) + nfft as isize) as usize % nfft;

            // ── Scatter energy to reassigned coordinates ──
            if t_hat_frame >= 0 && (t_hat_frame as usize) < n_frames {
                output[[k_idx, t_hat_frame as usize]] += mag2;
            }
        }
    }

    // ── fftshift along frequency axis (axis 0): move DC to centre ──
    // Before: [F0, F1, ..., F_{half-1}, F_{half}, ..., F_{nfft-1}]
    //          DC, positive freqs ...  | negative freqs ...
    // After:  [F_{half}, ..., F_{nfft-1}, F0, ..., F_{half-1}]
    //          -fs/2 ...              |        ... +fs/2
    let mut shifted = Array2::<f32>::zeros((nfft, n_frames));
    for k in 0..half {
        for col in 0..n_frames {
            shifted[[k, col]] = output[[k + half, col]];
            shifted[[k + half, col]] = output[[k, col]];
        }
    }

    shifted.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_window(nfft: usize, sigma: f32) -> Vec<f32> {
        let half = nfft / 2;
        let sigma2 = sigma * sigma;
        (0..nfft)
            .map(|i| {
                let n_rel = i as f32 - half as f32;
                (-n_rel * n_rel / (2.0 * sigma2)).exp()
            })
            .collect()
    }

    #[test]
    fn test_gaussian_window_symmetric() {
        // Window centered at half = nfft/2 = 8: w[half-k] == w[half+k]
        let nfft = 16;
        let half = nfft / 2; // = 8
        let w = gaussian_window(nfft, 4.0);
        for k in 1..half {
            assert!(
                (w[half - k] - w[half + k]).abs() < 1e-6,
                "window not symmetric: w[{}]={} != w[{}]={}",
                half - k,
                w[half - k],
                half + k,
                w[half + k]
            );
        }
    }

    #[test]
    fn test_gaussian_window_peak_at_center() {
        let nfft = 32;
        let w = gaussian_window(nfft, 4.0);
        let peak_idx = w
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, nfft / 2);
    }

    #[test]
    fn test_gaussian_window_positive() {
        let w = gaussian_window(64, 8.0);
        assert!(w.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_n_frames_formula() {
        // n=512, nfft=128, hop=64 → 1 + (512-128)/64 = 7
        let n: usize = 512;
        let nfft: usize = 128;
        let hop: usize = 64;
        let n_frames = if n >= nfft { 1 + (n - nfft) / hop } else { 0 };
        assert_eq!(n_frames, 7);
    }

    #[test]
    fn test_fftshift_swaps_halves() {
        // fftshift of [0,1,2,3] along axis 0 with n_frames=1 → [2,3,0,1]
        let nfft = 4;
        let n_frames = 1;
        let mut output = Array2::<f32>::zeros((nfft, n_frames));
        for k in 0..nfft {
            output[[k, 0]] = k as f32;
        }
        let half = nfft / 2;
        let mut shifted = Array2::<f32>::zeros((nfft, n_frames));
        for k in 0..half {
            for col in 0..n_frames {
                shifted[[k, col]] = output[[k + half, col]];
                shifted[[k + half, col]] = output[[k, col]];
            }
        }
        assert_eq!(shifted[[0, 0]], 2.0);
        assert_eq!(shifted[[1, 0]], 3.0);
        assert_eq!(shifted[[2, 0]], 0.0);
        assert_eq!(shifted[[3, 0]], 1.0);
    }
}

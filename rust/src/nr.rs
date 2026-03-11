use num_complex::Complex32;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::FftPlanner;

// ---------------------------------------------------------------------------
// Internal (testable) helpers
// ---------------------------------------------------------------------------

/// IFFT of subcarrier data (zero-padded to fft_size) + cyclic prefix insertion.
/// Returns Vec<Complex32> of length fft_size + cp_length.
fn nr_ofdm_symbol_inner(
    fft_size: usize,
    cp_length: usize,
    subcarrier_data: &[Complex32],
) -> Result<Vec<Complex32>, String> {
    if subcarrier_data.len() > fft_size {
        return Err(format!(
            "subcarrier_data length {} exceeds fft_size {}",
            subcarrier_data.len(),
            fft_size
        ));
    }

    let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); fft_size];
    buffer[..subcarrier_data.len()].copy_from_slice(subcarrier_data);

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut buffer);

    // Normalize (rustfft does not normalize)
    let scale = 1.0 / fft_size as f32;
    for s in buffer.iter_mut() {
        *s *= scale;
    }

    let mut output = Vec::with_capacity(fft_size + cp_length);
    if cp_length > 0 {
        let cp_start = fft_size - cp_length;
        output.extend_from_slice(&buffer[cp_start..]);
    }
    output.extend_from_slice(&buffer);
    Ok(output)
}

/// Generate PSS m-sequence and return 127-length Vec of +/-1 real values.
fn nr_pss_inner(n_id_2: usize) -> Result<Vec<Complex32>, String> {
    if n_id_2 > 2 {
        return Err(format!("n_id_2 must be 0, 1, or 2, got {}", n_id_2));
    }

    // x sequence: x(i+7) = (x(i+4) + x(i)) mod 2, init [1,1,1,0,1,1,0]
    let total_x = 127 + 43 * 2 + 1; // 214 entries covers all needed indices
    let mut x = vec![0u8; total_x];
    x[0] = 1;
    x[1] = 1;
    x[2] = 1;
    x[3] = 0;
    x[4] = 1;
    x[5] = 1;
    x[6] = 0;
    for i in 0..(total_x - 7) {
        x[i + 7] = (x[i + 4] + x[i]) % 2;
    }

    let result: Vec<Complex32> = (0..127)
        .map(|n| {
            let m = (n + 43 * n_id_2) % 127;
            let val = 1.0 - 2.0 * x[m] as f32;
            Complex32::new(val, 0.0)
        })
        .collect();
    Ok(result)
}

/// Generate SSS Gold sequence and return 127-length Vec.
fn nr_sss_inner(n_id_1: usize, n_id_2: usize) -> Result<Vec<Complex32>, String> {
    if n_id_1 > 335 {
        return Err(format!("n_id_1 must be in 0..335, got {}", n_id_1));
    }
    if n_id_2 > 2 {
        return Err(format!("n_id_2 must be 0, 1, or 2, got {}", n_id_2));
    }

    let total_len = 127 + 7;
    // x0: x0(i+7) = (x0(i+4) + x0(i)) mod 2, init [1,0,0,0,0,0,0]
    let mut x0 = vec![0u8; total_len];
    x0[0] = 1;
    for i in 0..(total_len - 7) {
        x0[i + 7] = (x0[i + 4] + x0[i]) % 2;
    }

    // x1: x1(i+7) = (x1(i+1) + x1(i)) mod 2, init [1,0,0,0,0,0,0]
    let mut x1 = vec![0u8; total_len];
    x1[0] = 1;
    for i in 0..(total_len - 7) {
        x1[i + 7] = (x1[i + 1] + x1[i]) % 2;
    }

    let m0 = 15 * (n_id_1 / 112) + 5 * n_id_2;
    let m1 = n_id_1 % 112;

    let result: Vec<Complex32> = (0..127)
        .map(|n| {
            let idx0 = (n + m0) % 127;
            let idx1 = (n + m1) % 127;
            let val = (1.0 - 2.0 * x0[idx0] as f32) * (1.0 - 2.0 * x1[idx1] as f32);
            Complex32::new(val, 0.0)
        })
        .collect();
    Ok(result)
}

/// Generate DMRS QPSK sequence of length num_rbs * 6.
fn nr_dmrs_inner(num_rbs: usize, n_id: usize, slot: usize, symbol_idx: usize) -> Vec<Complex32> {
    let length = num_rbs * 6;
    let n_symb = 14usize;

    let c_init: u32 = {
        let term1 = ((n_symb * slot + symbol_idx + 1) as u64) * (2 * n_id as u64 + 1);
        ((1u64 << 17) * term1 + 2 * n_id as u64) as u32
    };

    let nc = 1600usize;
    let seq_len = 2 * length + nc;

    let mut x1 = vec![0u8; seq_len + 31];
    x1[0] = 1;
    for i in 0..seq_len {
        x1[i + 31] = (x1[i + 3] ^ x1[i]) & 1;
    }

    let mut x2 = vec![0u8; seq_len + 31];
    for (bit, x2_val) in x2.iter_mut().enumerate().take(31) {
        *x2_val = ((c_init >> bit) & 1) as u8;
    }
    for i in 0..seq_len {
        x2[i + 31] = (x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]) & 1;
    }

    let scale = 1.0 / std::f32::consts::SQRT_2;
    (0..length)
        .map(|m| {
            let c_re = (x1[2 * m + nc] ^ x2[2 * m + nc]) as f32;
            let c_im = (x1[2 * m + 1 + nc] ^ x2[2 * m + 1 + nc]) as f32;
            Complex32::new(scale * (1.0 - 2.0 * c_re), scale * (1.0 - 2.0 * c_im))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

/// Generate a 5G NR OFDM symbol: IFFT of subcarrier data + cyclic prefix.
#[pyfunction]
pub fn generate_nr_ofdm_symbol<'py>(
    py: Python<'py>,
    fft_size: usize,
    cp_length: usize,
    subcarrier_data: PyReadonlyArray1<Complex32>,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let data = subcarrier_data.as_slice()?;
    let output = nr_ofdm_symbol_inner(fft_size, cp_length, data)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let arr = Array1::from(output);
    Ok(arr.into_pyarray(py))
}

/// Generate 5G NR PSS per TS 38.211 7.4.2.2.
#[pyfunction]
pub fn generate_nr_pss<'py>(
    py: Python<'py>,
    n_id_2: usize,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let seq = nr_pss_inner(n_id_2).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let arr = Array1::from(seq);
    Ok(arr.into_pyarray(py))
}

/// Generate 5G NR SSS per TS 38.211 7.4.2.3.
#[pyfunction]
pub fn generate_nr_sss<'py>(
    py: Python<'py>,
    n_id_1: usize,
    n_id_2: usize,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let seq =
        nr_sss_inner(n_id_1, n_id_2).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let arr = Array1::from(seq);
    Ok(arr.into_pyarray(py))
}

/// Generate 5G NR DMRS QPSK sequence.
#[pyfunction]
pub fn generate_nr_dmrs<'py>(
    py: Python<'py>,
    num_rbs: usize,
    n_id: usize,
    slot: usize,
    symbol_idx: usize,
    _seed: u64,
) -> Bound<'py, PyArray1<Complex32>> {
    let seq = nr_dmrs_inner(num_rbs, n_id, slot, symbol_idx);
    let arr = Array1::from(seq);
    arr.into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Tests (pure Rust, no PyO3 GIL needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pss_length_and_real() {
        for n_id_2 in 0..3 {
            let seq = nr_pss_inner(n_id_2).unwrap();
            assert_eq!(seq.len(), 127);
            for v in &seq {
                assert_eq!(v.im, 0.0);
                assert!((v.re.abs() - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_pss_invalid_n_id_2() {
        assert!(nr_pss_inner(3).is_err());
    }

    #[test]
    fn test_sss_length() {
        let seq = nr_sss_inner(0, 0).unwrap();
        assert_eq!(seq.len(), 127);
    }

    #[test]
    fn test_sss_valid_range() {
        assert!(nr_sss_inner(335, 2).is_ok());
        assert!(nr_sss_inner(336, 0).is_err());
        assert!(nr_sss_inner(0, 3).is_err());
    }

    #[test]
    fn test_sss_real_pm1() {
        let seq = nr_sss_inner(100, 1).unwrap();
        for v in &seq {
            assert_eq!(v.im, 0.0);
            assert!((v.re.abs() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ofdm_symbol_length() {
        let fft_size = 256;
        let cp_length = 18;
        let data: Vec<Complex32> = (0..64).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let output = nr_ofdm_symbol_inner(fft_size, cp_length, &data).unwrap();
        assert_eq!(output.len(), fft_size + cp_length);
    }

    #[test]
    fn test_ofdm_symbol_data_too_large() {
        let data: Vec<Complex32> = vec![Complex32::new(1.0, 0.0); 300];
        assert!(nr_ofdm_symbol_inner(256, 18, &data).is_err());
    }

    #[test]
    fn test_dmrs_length() {
        let num_rbs = 25;
        let seq = nr_dmrs_inner(num_rbs, 0, 0, 2);
        assert_eq!(seq.len(), num_rbs * 6);
    }

    #[test]
    fn test_dmrs_qpsk_magnitude() {
        let seq = nr_dmrs_inner(10, 42, 3, 5);
        for v in &seq {
            let mag = (v.re * v.re + v.im * v.im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-5,
                "DMRS QPSK magnitude {} != 1.0",
                mag
            );
        }
    }
}

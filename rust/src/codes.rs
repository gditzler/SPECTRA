use num_complex::Complex32;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use std::f64::consts::PI;

// ── Internal helpers (return Vec<Complex32>) ───────────────────────

fn frank_code_chips(order: usize) -> Vec<Complex32> {
    let mut chips = Vec::with_capacity(order * order);
    for i in 0..order {
        for j in 0..order {
            let phase = (2.0 * PI / order as f64) * (i as f64) * (j as f64);
            chips.push(Complex32::new(phase.cos() as f32, phase.sin() as f32));
        }
    }
    chips
}

fn p1_code_chips(order: usize) -> Vec<Complex32> {
    let n = order as f64;
    let mut chips = Vec::with_capacity(order * order);
    for j in 1..=order {
        for i in 1..=order {
            let phase = -(PI / n)
                * (n - (2.0 * j as f64 - 1.0))
                * ((j as f64 - 1.0) * n + (i as f64 - 1.0));
            chips.push(Complex32::new(phase.cos() as f32, phase.sin() as f32));
        }
    }
    chips
}

fn p2_code_chips(order: usize) -> Vec<Complex32> {
    let n = order as f64;
    let mut chips = Vec::with_capacity(order * order);
    for j in 1..=order {
        for i in 1..=order {
            let phase = (PI / (2.0 * n)) * (2.0 * i as f64 - 1.0 - n) * (2.0 * j as f64 - 1.0 - n);
            chips.push(Complex32::new(phase.cos() as f32, phase.sin() as f32));
        }
    }
    chips
}

fn p3_code_chips(length: usize) -> Vec<Complex32> {
    let n = length as f64;
    (1..=length)
        .map(|k| {
            let phase = (PI / n) * ((k - 1) as f64).powi(2);
            Complex32::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect()
}

fn p4_code_chips(length: usize) -> Vec<Complex32> {
    let n = length as f64;
    (1..=length)
        .map(|k| {
            let km1 = (k - 1) as f64;
            let phase = (PI / n) * km1 * km1 - PI * km1;
            Complex32::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect()
}

// ── PyO3-exported functions ────────────────────────────────────────

/// Generate Frank code chips (M^2 complex values on unit circle).
#[pyfunction]
pub fn generate_frank_code<'py>(py: Python<'py>, order: usize) -> Bound<'py, PyArray1<Complex32>> {
    let chips = frank_code_chips(order);
    Array1::from_vec(chips).into_pyarray(py)
}

/// Generate P1 polyphase code chips (N^2 complex values).
#[pyfunction]
pub fn generate_p1_code<'py>(py: Python<'py>, order: usize) -> Bound<'py, PyArray1<Complex32>> {
    let chips = p1_code_chips(order);
    Array1::from_vec(chips).into_pyarray(py)
}

/// Generate P2 polyphase code chips (N^2 complex values, N must be even).
#[pyfunction]
pub fn generate_p2_code<'py>(
    py: Python<'py>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if order % 2 != 0 {
        return Err(PyValueError::new_err("P2 code requires even order"));
    }
    let chips = p2_code_chips(order);
    Ok(Array1::from_vec(chips).into_pyarray(py))
}

/// Generate P3 polyphase code chips (N complex values, arbitrary length).
#[pyfunction]
pub fn generate_p3_code<'py>(py: Python<'py>, length: usize) -> Bound<'py, PyArray1<Complex32>> {
    let chips = p3_code_chips(length);
    Array1::from_vec(chips).into_pyarray(py)
}

/// Generate P4 polyphase code chips (N complex values, arbitrary length).
#[pyfunction]
pub fn generate_p4_code<'py>(py: Python<'py>, length: usize) -> Bound<'py, PyArray1<Complex32>> {
    let chips = p4_code_chips(length);
    Array1::from_vec(chips).into_pyarray(py)
}

// ── Costas sequence ────────────────────────────────────────────────

fn primitive_root(p: usize) -> usize {
    let order = p - 1;
    for g in 2..p {
        let mut val = 1usize;
        let mut is_primitive = true;
        for _ in 1..order {
            val = (val * g) % p;
            if val == 1 {
                is_primitive = false;
                break;
            }
        }
        if is_primitive {
            return g;
        }
    }
    unreachable!("All primes > 2 have primitive roots")
}

/// Generate a Costas frequency-hopping sequence via Welch construction.
/// Returns a list of frequency indices (1-based) of length prime-1.
#[pyfunction]
pub fn generate_costas_sequence(prime: usize) -> PyResult<Vec<usize>> {
    if prime < 3 {
        return Err(PyValueError::new_err("prime must be >= 3"));
    }
    let g = primitive_root(prime);
    let sequence: Vec<usize> = (1..prime)
        .map(|i| {
            let mut val = 1usize;
            for _ in 0..i {
                val = (val * g) % prime;
            }
            val
        })
        .collect();
    Ok(sequence)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Frank code tests ---

    #[test]
    fn frank_code_length() {
        let chips = frank_code_chips(4);
        assert_eq!(chips.len(), 16);
    }

    #[test]
    fn frank_code_unit_magnitude() {
        let chips = frank_code_chips(4);
        for c in &chips {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "Chip magnitude {} != 1.0", mag);
        }
    }

    #[test]
    fn frank_code_first_row_zero_phase() {
        let chips = frank_code_chips(4);
        for c in chips.iter().take(4) {
            assert!((c.re - 1.0).abs() < 1e-5);
            assert!(c.im.abs() < 1e-5);
        }
    }

    // --- P1 code tests ---

    #[test]
    fn p1_code_length() {
        let chips = p1_code_chips(4);
        assert_eq!(chips.len(), 16);
    }

    #[test]
    fn p1_code_unit_magnitude() {
        let chips = p1_code_chips(4);
        for c in &chips {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5);
        }
    }

    // --- P2 code tests ---

    #[test]
    fn p2_code_length() {
        let chips = p2_code_chips(4);
        assert_eq!(chips.len(), 16);
    }

    #[test]
    fn p2_code_unit_magnitude() {
        let chips = p2_code_chips(4);
        for c in &chips {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5);
        }
    }

    // --- P3 code tests ---

    #[test]
    fn p3_code_length() {
        let chips = p3_code_chips(16);
        assert_eq!(chips.len(), 16);
    }

    #[test]
    fn p3_code_unit_magnitude() {
        let chips = p3_code_chips(25);
        for c in &chips {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn p3_first_chip_is_one() {
        let chips = p3_code_chips(16);
        assert!((chips[0].re - 1.0).abs() < 1e-5);
        assert!(chips[0].im.abs() < 1e-5);
    }

    // --- P4 code tests ---

    #[test]
    fn p4_code_length() {
        let chips = p4_code_chips(36);
        assert_eq!(chips.len(), 36);
    }

    #[test]
    fn p4_code_unit_magnitude() {
        let chips = p4_code_chips(36);
        for c in &chips {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn p4_first_chip_is_one() {
        let chips = p4_code_chips(16);
        assert!((chips[0].re - 1.0).abs() < 1e-5);
        assert!(chips[0].im.abs() < 1e-5);
    }

    // --- Costas tests ---

    #[test]
    fn costas_welch_length() {
        let seq = generate_costas_sequence(7).unwrap();
        assert_eq!(seq.len(), 6);
    }

    #[test]
    fn costas_welch_permutation() {
        let seq = generate_costas_sequence(7).unwrap();
        let mut sorted = seq.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 6);
        assert!(sorted.iter().all(|&v| (1..=6).contains(&v)));
    }

    #[test]
    fn costas_small_prime() {
        let seq = generate_costas_sequence(5).unwrap();
        assert_eq!(seq.len(), 4);
    }

    #[test]
    fn costas_invalid_prime() {
        assert!(generate_costas_sequence(2).is_err());
    }
}

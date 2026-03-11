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

// ── Spread spectrum code generators ───────────────────────────────

/// Primitive polynomials as tap positions (1-indexed) for LFSR m-sequence generation.
/// Each entry maps order -> list of tap positions (including the order itself).
fn primitive_poly_taps(order: usize) -> Option<&'static [usize]> {
    match order {
        5 => Some(&[5, 2]),
        6 => Some(&[6, 1]),
        7 => Some(&[7, 1]),
        8 => Some(&[8, 6, 5, 4]),
        9 => Some(&[9, 4]),
        10 => Some(&[10, 3]),
        _ => None,
    }
}

/// Second primitive polynomials for Gold code preferred pairs.
/// Each entry is (order, preferred_pair_idx) -> tap positions for the second polynomial.
fn gold_second_poly_taps(order: usize, preferred_pair_idx: usize) -> Option<&'static [usize]> {
    match (order, preferred_pair_idx) {
        // Order 5 preferred pairs
        (5, 0) => Some(&[5, 4, 3, 2]), // x^5 + x^4 + x^3 + x^2 + 1
        (5, 1) => Some(&[5, 4, 2, 1]), // x^5 + x^4 + x^2 + x + 1
        (5, 2) => Some(&[5, 3, 2, 1]), // x^5 + x^3 + x^2 + x + 1
        (5, 3) => Some(&[5, 4]),       // x^5 + x^4 + 1 (reciprocal pair)
        (5, 4) => Some(&[5, 4, 3, 1]), // x^5 + x^4 + x^3 + x + 1
        (5, 5) => Some(&[5, 3]),       // x^5 + x^3 + 1
        // Order 6 preferred pairs
        (6, 0) => Some(&[6, 5, 2, 1]), // x^6 + x^5 + x^2 + x + 1
        (6, 1) => Some(&[6, 5, 3, 2]), // x^6 + x^5 + x^3 + x^2 + 1
        (6, 2) => Some(&[6, 5]),       // x^6 + x^5 + 1
        // Order 7 preferred pairs
        (7, 0) => Some(&[7, 3]),             // x^7 + x^3 + 1
        (7, 1) => Some(&[7, 3, 2, 1]),       // x^7 + x^3 + x^2 + x + 1
        (7, 2) => Some(&[7, 5, 4, 3, 2, 1]), // x^7 + x^5 + x^4 + x^3 + x^2 + x + 1
        // (7, 3) omitted: x^7+x^4+1 is not a preferred pair
        (7, 4) => Some(&[7, 6, 4, 2]), // x^7 + x^6 + x^4 + x^2 + 1
        (7, 5) => Some(&[7, 6, 5, 4, 2, 1]), // x^7 + x^6 + x^5 + x^4 + x^2 + x + 1
        // Order 8 preferred pairs
        (8, 0) => Some(&[8, 7, 6, 5, 2, 1]), // x^8 + x^7 + x^6 + x^5 + x^2 + x + 1
        (8, 1) => Some(&[8, 7, 6, 1]),       // x^8 + x^7 + x^6 + x + 1
        (8, 2) => Some(&[8, 6, 5, 3]),       // x^8 + x^6 + x^5 + x^3 + 1
        (8, 3) => Some(&[8, 7, 2, 1]),       // x^8 + x^7 + x^2 + x + 1
        // Order 9 preferred pairs
        (9, 0) => Some(&[9, 6, 4, 3]), // x^9 + x^6 + x^4 + x^3 + 1
        (9, 1) => Some(&[9, 8, 4, 1]), // x^9 + x^8 + x^4 + x + 1
        (9, 2) => Some(&[9, 8, 6, 5]), // x^9 + x^8 + x^6 + x^5 + 1
        // Order 10 preferred pairs
        (10, 0) => Some(&[10, 8, 3, 2]), // x^10 + x^8 + x^3 + x^2 + 1
        (10, 1) => Some(&[10, 4, 3, 1]), // x^10 + x^4 + x^3 + x + 1
        (10, 2) => Some(&[10, 9, 4, 1]), // x^10 + x^9 + x^4 + x + 1
        _ => None,
    }
}

/// Generate an m-sequence of length 2^order - 1 using LFSR with given taps.
/// Returns bipolar {-1.0, +1.0} values.
fn generate_msequence(order: usize, taps: &[usize]) -> Vec<f32> {
    let length = (1usize << order) - 1;
    let mut register = vec![1u8; order];
    let mut seq = Vec::with_capacity(length);

    for _ in 0..length {
        seq.push(register[order - 1]);
        let mut feedback = 0u8;
        for &tap in taps {
            feedback ^= register[tap - 1];
        }
        register.rotate_right(1);
        register[0] = feedback;
    }

    // Convert {0, 1} -> {-1, +1}
    seq.into_iter()
        .map(|b| if b == 0 { -1.0f32 } else { 1.0f32 })
        .collect()
}

/// Generate a Gold code of length 2^order - 1.
///
/// Gold codes are formed by XORing two preferred-pair m-sequences.
/// The `preferred_pair_idx` selects which preferred pair to use.
/// Returns bipolar {-1.0, +1.0} values.
#[pyfunction]
pub fn generate_gold_code<'py>(
    py: Python<'py>,
    order: usize,
    preferred_pair_idx: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let taps1 = primitive_poly_taps(order).ok_or_else(|| {
        PyValueError::new_err(format!("Gold code order must be in 5..=10, got {}", order))
    })?;
    let taps2 = gold_second_poly_taps(order, preferred_pair_idx).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid preferred_pair_idx {} for order {}",
            preferred_pair_idx, order
        ))
    })?;

    let m1 = generate_msequence(order, taps1);
    let m2 = generate_msequence(order, taps2);

    // Gold code = m1 * m2 (bipolar XOR is multiplication)
    let gold: Vec<f32> = m1.iter().zip(m2.iter()).map(|(a, b)| a * b).collect();
    Ok(Array1::from_vec(gold).into_pyarray(py))
}

/// Generate a small-set Kasami code of length 2^order - 1.
///
/// Requires even order. Decimates an m-sequence by 2^(order/2)+1 to produce
/// a short code, then XORs with cyclically shifted m-sequence.
/// Returns bipolar {-1.0, +1.0} values.
#[pyfunction]
pub fn generate_kasami_code<'py>(
    py: Python<'py>,
    order: usize,
    shift_idx: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if order % 2 != 0 {
        return Err(PyValueError::new_err("Kasami codes require even order"));
    }
    if !(5..=10).contains(&order) {
        return Err(PyValueError::new_err(format!(
            "Kasami code order must be 6, 8, or 10, got {}",
            order
        )));
    }

    let taps = primitive_poly_taps(order).ok_or_else(|| {
        PyValueError::new_err(format!("No primitive polynomial for order {}", order))
    })?;

    let length = (1usize << order) - 1;
    let m_seq = generate_msequence(order, taps);

    // Decimate by d = 2^(order/2) + 1 to get short code of length 2^(order/2) - 1
    let d = (1usize << (order / 2)) + 1;
    let short_len = (1usize << (order / 2)) - 1;

    let short_code: Vec<f32> = (0..short_len).map(|i| m_seq[(i * d) % length]).collect();

    // Tile short code to length of m-sequence
    let mut short_tiled = Vec::with_capacity(length);
    for i in 0..length {
        short_tiled.push(short_code[i % short_len]);
    }

    // Apply cyclic shift to the tiled short code
    let shift = shift_idx % length;
    let mut shifted = vec![0.0f32; length];
    for i in 0..length {
        shifted[i] = short_tiled[(i + length - shift) % length];
    }

    // Kasami code = m_seq * shifted_short (bipolar XOR)
    let kasami: Vec<f32> = m_seq
        .iter()
        .zip(shifted.iter())
        .map(|(a, b)| a * b)
        .collect();

    Ok(Array1::from_vec(kasami).into_pyarray(py))
}

/// Generate a Walsh-Hadamard code (one row of the Hadamard matrix).
///
/// Uses Sylvester construction: H_1 = [1], H_{n+1} = [[H_n, H_n], [H_n, -H_n]].
/// Returns bipolar {-1.0, +1.0} values of length 2^order.
#[pyfunction]
pub fn generate_walsh_hadamard<'py>(
    py: Python<'py>,
    order: usize,
    code_idx: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let size = 1usize << order;
    if code_idx >= size {
        return Err(PyValueError::new_err(format!(
            "code_idx {} must be < 2^order = {}",
            code_idx, size
        )));
    }

    // Build row `code_idx` of Hadamard matrix efficiently using bit-counting.
    // H[i][j] = (-1)^(popcount(i & j))
    let row: Vec<f32> = (0..size)
        .map(|j| {
            let bits = (code_idx & j).count_ones();
            if bits % 2 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect();

    Ok(Array1::from_vec(row).into_pyarray(py))
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

    // --- Gold code tests ---

    #[test]
    fn gold_msequence_length() {
        let code = generate_msequence(5, primitive_poly_taps(5).unwrap());
        assert_eq!(code.len(), 31);
    }

    #[test]
    fn gold_msequence_bipolar() {
        let code = generate_msequence(5, primitive_poly_taps(5).unwrap());
        for &v in &code {
            assert!(v == 1.0 || v == -1.0, "Expected +/-1, got {}", v);
        }
    }

    #[test]
    fn gold_code_bipolar() {
        let taps1 = primitive_poly_taps(5).unwrap();
        let taps2 = gold_second_poly_taps(5, 0).unwrap();
        let m1 = generate_msequence(5, taps1);
        let m2 = generate_msequence(5, taps2);
        let gold: Vec<f32> = m1.iter().zip(m2.iter()).map(|(a, b)| a * b).collect();
        assert_eq!(gold.len(), 31);
        for &v in &gold {
            assert!(v == 1.0 || v == -1.0, "Expected +/-1, got {}", v);
        }
    }

    #[test]
    fn gold_code_no_poly_for_invalid_order() {
        assert!(primitive_poly_taps(3).is_none());
        assert!(primitive_poly_taps(11).is_none());
    }

    #[test]
    fn gold_code_no_pair_for_invalid_idx() {
        assert!(gold_second_poly_taps(5, 99).is_none());
    }

    // --- Kasami code tests (internal) ---

    #[test]
    fn kasami_code_internal_length() {
        // Kasami requires even order; test with order 6
        let taps = primitive_poly_taps(6).unwrap();
        let length = (1usize << 6) - 1; // 63
        let m_seq = generate_msequence(6, taps);
        assert_eq!(m_seq.len(), length);

        // Decimate
        let d = (1usize << 3) + 1; // 9
        let short_len = (1usize << 3) - 1; // 7
        let short: Vec<f32> = (0..short_len).map(|i| m_seq[(i * d) % length]).collect();
        assert_eq!(short.len(), 7);
        for &v in &short {
            assert!(v == 1.0 || v == -1.0);
        }
    }

    // --- Walsh-Hadamard tests (internal) ---

    #[test]
    fn walsh_row_length() {
        let size = 1usize << 4; // 16
        let idx = 0usize;
        let row: Vec<f32> = (0..size)
            .map(|j| {
                let bits = (idx & j).count_ones();
                if bits % 2 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect();
        assert_eq!(row.len(), 16);
        // Row 0: all +1
        for &v in &row {
            assert!((v - 1.0f32).abs() < 1e-5, "Row 0 should be all +1");
        }
    }

    #[test]
    fn walsh_orthogonality_internal() {
        let size = 1usize << 4;
        let idx0 = 0usize;
        let row0: Vec<f32> = (0..size)
            .map(|j| {
                if (idx0 & j).count_ones() % 2 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect();
        let row1: Vec<f32> = (0..size)
            .map(|j| {
                if (1usize & j).count_ones() % 2 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect();
        let dot: f32 = row0.iter().zip(row1.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() < 1e-5,
            "Walsh rows should be orthogonal, got {}",
            dot
        );
    }

    #[test]
    fn walsh_bipolar_internal() {
        let size = 1usize << 4;
        for idx in 0..size {
            let row: Vec<f32> = (0..size)
                .map(|j| {
                    if (idx & j).count_ones() % 2 == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    }
                })
                .collect();
            for &v in &row {
                assert!(v == 1.0 || v == -1.0, "Expected +/-1, got {}", v);
            }
        }
    }
}

use num_complex::Complex32;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// QPSK constellation: points at pi/4, 3pi/4, -3pi/4, -pi/4
const QPSK_CONSTELLATION: [Complex32; 4] = [
    Complex32::new(
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ), // pi/4
    Complex32::new(
        -std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ), // 3pi/4
    Complex32::new(
        -std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ), // -3pi/4
    Complex32::new(
        std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ), // -pi/4
];

/// Simple seeded PRNG (xorshift64) for deterministic symbol generation.
pub(crate) struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub(crate) fn new(seed: u64) -> Self {
        // splitmix64 to avoid seed 0/1 collision
        let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
        s = (s ^ (s >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94d049bb133111eb);
        s ^= s >> 31;
        Self {
            state: if s == 0 { 1 } else { s },
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

#[pyfunction]
pub fn generate_qpsk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> Bound<'py, PyArray1<Complex32>> {
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() % 4) as usize;
        QPSK_CONSTELLATION[idx]
    });
    symbols.into_pyarray(py)
}

#[pyfunction]
pub fn generate_bpsk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> Bound<'py, PyArray1<Complex32>> {
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        if rng.next() % 2 == 0 {
            Complex32::new(1.0, 0.0)
        } else {
            Complex32::new(-1.0, 0.0)
        }
    });
    symbols.into_pyarray(py)
}

/// 8PSK constellation: 8 equally spaced points on the unit circle.
#[pyfunction]
pub fn generate_8psk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> Bound<'py, PyArray1<Complex32>> {
    let constellation: [Complex32; 8] = std::array::from_fn(|k| {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / 8.0;
        Complex32::new(angle.cos() as f32, angle.sin() as f32)
    });
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() % 8) as usize;
        constellation[idx]
    });
    symbols.into_pyarray(py)
}

/// QAM constellation: square grid of order points, normalized to unit avg power.
/// Order must be a perfect square (16, 64, 256, ...).
#[pyfunction]
pub fn generate_qam_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let side = (order as f64).sqrt() as usize;
    if side * side != order {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "QAM order must be a perfect square (16, 64, 256, ...)",
        ));
    }
    // Build constellation
    let mut constellation = Vec::with_capacity(order);
    for i in 0..side {
        for j in 0..side {
            let re = 2.0 * i as f64 - (side - 1) as f64;
            let im = 2.0 * j as f64 - (side - 1) as f64;
            constellation.push(Complex32::new(re as f32, im as f32));
        }
    }
    // Normalize to unit average power
    let avg_power: f64 = constellation
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) as f64)
        .sum::<f64>()
        / order as f64;
    let scale = 1.0 / avg_power.sqrt() as f32;
    for c in &mut constellation {
        c.re *= scale;
        c.im *= scale;
    }
    // Generate random symbols
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() as usize) % order;
        constellation[idx]
    });
    Ok(symbols.into_pyarray(py))
}

/// Generic M-PSK symbol generator: M equally-spaced points on the unit circle.
#[pyfunction]
pub fn generate_psk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if order < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "PSK order must be at least 2",
        ));
    }
    let constellation: Vec<Complex32> = (0..order)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / order as f64;
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect();
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() as usize) % order;
        constellation[idx]
    });
    Ok(symbols.into_pyarray(py))
}

/// M-ary ASK symbol generator: M amplitude levels on the real axis,
/// normalized to unit average power. OOK is ASK with M=2.
#[pyfunction]
pub fn generate_ask_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if order < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ASK order must be at least 2",
        ));
    }
    // Build amplitude levels: 0, 1, 2, ..., M-1
    let levels: Vec<f64> = (0..order).map(|k| k as f64).collect();
    // Normalize to unit average power: avg(level^2) = 1
    let avg_power: f64 = levels.iter().map(|l| l * l).sum::<f64>() / order as f64;
    let scale = if avg_power > 0.0 {
        1.0 / avg_power.sqrt()
    } else {
        1.0
    };
    let constellation: Vec<Complex32> = levels
        .iter()
        .map(|l| Complex32::new((l * scale) as f32, 0.0))
        .collect();
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() as usize) % order;
        constellation[idx]
    });
    Ok(symbols.into_pyarray(py))
}

/// Generate random FSK frequency symbols as normalized floats in [-1, 1].
/// Returns M-ary frequency values: linspace(-1+1/M, 1-1/M, M).
#[pyfunction]
pub fn generate_fsk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> Bound<'py, PyArray1<f32>> {
    // Build frequency map: M equally spaced values in (-1, 1)
    let freq_map: Vec<f32> = (0..order)
        .map(|k| (-1.0 + (2.0 * k as f64 + 1.0) / order as f64) as f32)
        .collect();
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() as usize) % order;
        freq_map[idx]
    });
    symbols.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift64_deterministic() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn xorshift64_different_seeds_differ() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(99);
        let seq1: Vec<u64> = (0..10).map(|_| rng1.next()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next()).collect();
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn psk8_constellation_points() {
        // Verify 8PSK constellation has 8 unit-magnitude points
        let constellation: [Complex32; 8] = std::array::from_fn(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / 8.0;
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        });
        for c in &constellation {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn qam16_constellation_properties() {
        // 16QAM: 4x4 grid, normalized to unit average power
        let side = 4usize;
        let order = 16usize;
        let mut constellation = Vec::with_capacity(order);
        for i in 0..side {
            for j in 0..side {
                let re = 2.0 * i as f64 - (side - 1) as f64;
                let im = 2.0 * j as f64 - (side - 1) as f64;
                constellation.push(Complex32::new(re as f32, im as f32));
            }
        }
        assert_eq!(constellation.len(), 16);
        let avg_power: f64 = constellation
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) as f64)
            .sum::<f64>()
            / order as f64;
        let scale = 1.0 / avg_power.sqrt() as f32;
        for c in &mut constellation {
            c.re *= scale;
            c.im *= scale;
        }
        // After normalization, average power should be ~1.0
        let norm_power: f64 = constellation
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) as f64)
            .sum::<f64>()
            / order as f64;
        assert!((norm_power - 1.0).abs() < 1e-5);
    }

    #[test]
    fn psk_constellation_unit_magnitude() {
        for order in [4, 8, 16, 32, 64] {
            let constellation: Vec<Complex32> = (0..order)
                .map(|k| {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 / order as f64;
                    Complex32::new(angle.cos() as f32, angle.sin() as f32)
                })
                .collect();
            assert_eq!(constellation.len(), order);
            for c in &constellation {
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                assert!((mag - 1.0).abs() < 1e-5, "order={order}, mag={mag}");
            }
        }
    }

    #[test]
    fn ask_unit_avg_power() {
        for order in [2, 4, 8] {
            let levels: Vec<f64> = (0..order).map(|k| k as f64).collect();
            let avg_power: f64 = levels.iter().map(|l| l * l).sum::<f64>() / order as f64;
            let scale = if avg_power > 0.0 {
                1.0 / avg_power.sqrt()
            } else {
                1.0
            };
            let constellation: Vec<f64> = levels.iter().map(|l| l * scale).collect();
            let norm_power: f64 = constellation.iter().map(|c| c * c).sum::<f64>() / order as f64;
            assert!(
                (norm_power - 1.0).abs() < 1e-5,
                "order={order}, norm_power={norm_power}"
            );
        }
    }

    #[test]
    fn ask_ook_levels() {
        // OOK (order=2): levels should be 0 and some positive value
        let levels: Vec<f64> = (0..2).map(|k| k as f64).collect();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], 0.0);
        assert_eq!(levels[1], 1.0);
    }

    #[test]
    fn fsk_freq_map_values() {
        let order = 4usize;
        let freq_map: Vec<f32> = (0..order)
            .map(|k| (-1.0 + (2.0 * k as f64 + 1.0) / order as f64) as f32)
            .collect();
        assert_eq!(freq_map.len(), 4);
        for &v in &freq_map {
            assert!((-1.0_f32..=1.0_f32).contains(&v));
        }
    }
}

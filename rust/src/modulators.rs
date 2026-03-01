use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use num_complex::Complex32;
use pyo3::prelude::*;

/// QPSK constellation: points at pi/4, 3pi/4, -3pi/4, -pi/4
const QPSK_CONSTELLATION: [Complex32; 4] = [
    Complex32::new(std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),   // pi/4
    Complex32::new(-std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),  // 3pi/4
    Complex32::new(-std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2), // -3pi/4
    Complex32::new(std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),  // -pi/4
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
        Self { state: if s == 0 { 1 } else { s } }
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

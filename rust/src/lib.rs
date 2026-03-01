use pyo3::prelude::*;

mod modulators;

/// SPECTRA Rust backend for high-performance DSP primitives.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(modulators::generate_qpsk_symbols, m)?)?;
    Ok(())
}

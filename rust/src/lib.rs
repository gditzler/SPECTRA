use pyo3::prelude::*;

mod filters;
mod modulators;
mod oscillators;

/// SPECTRA Rust backend for high-performance DSP primitives.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(modulators::generate_qpsk_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(modulators::generate_bpsk_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(filters::apply_rrc_filter, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::generate_chirp, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::generate_tone, m)?)?;
    Ok(())
}

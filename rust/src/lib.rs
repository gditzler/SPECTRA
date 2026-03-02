use pyo3::prelude::*;

mod codes;
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
    // Polyphase codes
    m.add_function(wrap_pyfunction!(codes::generate_frank_code, m)?)?;
    m.add_function(wrap_pyfunction!(codes::generate_p1_code, m)?)?;
    m.add_function(wrap_pyfunction!(codes::generate_p2_code, m)?)?;
    m.add_function(wrap_pyfunction!(codes::generate_p3_code, m)?)?;
    m.add_function(wrap_pyfunction!(codes::generate_p4_code, m)?)?;
    m.add_function(wrap_pyfunction!(codes::generate_costas_sequence, m)?)?;
    // Comms symbol generators
    m.add_function(wrap_pyfunction!(modulators::generate_8psk_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(modulators::generate_qam_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(modulators::generate_fsk_symbols, m)?)?;
    Ok(())
}

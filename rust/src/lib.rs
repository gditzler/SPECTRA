use pyo3::prelude::*;

mod codes;
mod cyclo_spectral;
mod cyclo_temporal;
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
    m.add_function(wrap_pyfunction!(modulators::generate_psk_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(modulators::generate_ask_symbols, m)?)?;
    // Filter utilities
    m.add_function(wrap_pyfunction!(filters::gaussian_taps, m)?)?;
    m.add_function(wrap_pyfunction!(filters::lowpass_taps, m)?)?;
    m.add_function(wrap_pyfunction!(filters::convolve_complex, m)?)?;
    // Cyclostationary signal processing
    m.add_function(wrap_pyfunction!(cyclo_spectral::compute_scd_ssca, m)?)?;
    m.add_function(wrap_pyfunction!(cyclo_spectral::compute_scd_fam, m)?)?;
    m.add_function(wrap_pyfunction!(cyclo_spectral::compute_psd_welch, m)?)?;
    m.add_function(wrap_pyfunction!(cyclo_spectral::channelize, m)?)?;
    m.add_function(wrap_pyfunction!(cyclo_temporal::compute_cumulants, m)?)?;
    m.add_function(wrap_pyfunction!(cyclo_temporal::compute_caf, m)?)?;
    Ok(())
}

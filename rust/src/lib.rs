use pyo3::prelude::*;

mod codes;
mod cwd;
mod cyclo_spectral;
mod cyclo_temporal;
mod filters;
mod modulators;
mod oscillators;
mod reassigned_gabor;
mod s3ca;
mod sfft;

/// SPECTRA Rust backend for high-performance DSP primitives.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(modulators::generate_qpsk_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(modulators::generate_bpsk_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(filters::apply_rrc_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::rrc_taps_py, m)?)?;
    m.add_function(wrap_pyfunction!(filters::apply_rrc_filter_with_taps, m)?)?;
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
    m.add_function(wrap_pyfunction!(s3ca::compute_scd_s3ca, m)?)?;
    m.add_function(wrap_pyfunction!(cwd::compute_cwd, m)?)?;
    m.add_function(wrap_pyfunction!(
        reassigned_gabor::compute_reassigned_gabor,
        m
    )?)?;
    Ok(())
}

def test_import():
    import spectra
    assert hasattr(spectra, "__version__")
    assert spectra.__version__ == "0.1.0"

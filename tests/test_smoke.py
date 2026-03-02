def test_import():
    import spectra
    assert hasattr(spectra, "__version__")
    assert spectra.__version__ == "0.1.0"


def test_radar_waveform_imports():
    from spectra import LFM, FrankCode, P1Code, P2Code, P3Code, P4Code, CostasCode
    assert LFM().label == "LFM"
    assert FrankCode().label == "Frank"
    assert P1Code().label == "P1"
    assert P2Code().label == "P2"
    assert P3Code().label == "P3"
    assert P4Code().label == "P4"
    assert CostasCode().label == "Costas"


def test_comms_waveform_imports():
    from spectra import PSK8, QAM16, QAM64, QAM256, FSK, MSK, GMSK, OFDM
    assert PSK8().label == "8PSK"
    assert QAM16().label == "16QAM"
    assert QAM64().label == "64QAM"
    assert QAM256().label == "256QAM"
    assert FSK().label == "FSK"
    assert MSK().label == "MSK"
    assert GMSK().label == "GMSK"
    assert OFDM().label == "OFDM"

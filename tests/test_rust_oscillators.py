import numpy as np
import numpy.testing as npt


class TestGenerateChirp:
    def test_output_shape_and_dtype(self):
        from spectra._rust import generate_chirp

        fs = 1e6
        duration = 0.001
        signal = generate_chirp(duration=duration, fs=fs, f0=1e3, f1=1e5)
        assert signal.shape == (int(duration * fs),)
        assert signal.dtype == np.complex64

    def test_unit_magnitude(self):
        from spectra._rust import generate_chirp

        signal = generate_chirp(duration=0.01, fs=1e6, f0=0.0, f1=1e5)
        npt.assert_allclose(np.abs(signal), 1.0, atol=1e-5)

    def test_frequency_sweep(self):
        from spectra._rust import generate_chirp

        fs = 1e6
        f0, f1 = 1e3, 1e5
        signal = generate_chirp(duration=0.01, fs=fs, f0=f0, f1=f1)
        phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        # Verify sweep direction: end frequency > start frequency
        start_freq = np.mean(inst_freq[:10])
        end_freq = np.mean(inst_freq[-10:])
        assert end_freq > start_freq, "Chirp should sweep upward"
        # Verify approximate start and end frequencies
        npt.assert_allclose(start_freq, f0, atol=500)
        npt.assert_allclose(end_freq, f1, atol=500)


class TestGenerateTone:
    def test_output_shape_and_dtype(self):
        from spectra._rust import generate_tone

        signal = generate_tone(frequency=1e3, duration=0.01, fs=1e6)
        assert signal.shape == (int(0.01 * 1e6),)
        assert signal.dtype == np.complex64

    def test_unit_magnitude(self):
        from spectra._rust import generate_tone

        signal = generate_tone(frequency=1e3, duration=0.01, fs=1e6)
        npt.assert_allclose(np.abs(signal), 1.0, atol=1e-5)

    def test_correct_frequency(self):
        from spectra._rust import generate_tone

        freq = 1000.0
        fs = 1e6
        signal = generate_tone(frequency=freq, duration=0.01, fs=fs)
        phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        npt.assert_allclose(inst_freq, freq, atol=1.0)

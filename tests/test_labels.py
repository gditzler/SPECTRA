import numpy as np
import torch


class TestSTFT:
    def test_output_shape(self):
        from spectra.transforms.stft import STFT

        stft = STFT(nfft=256, hop_length=64)
        iq = np.random.randn(4096).astype(np.float32) + 1j * np.random.randn(4096).astype(
            np.float32
        )
        iq = iq.astype(np.complex64)
        spectrogram = stft(iq)
        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.ndim == 3  # [channels, freq, time]
        assert spectrogram.shape[0] == 1  # single channel (magnitude)
        assert spectrogram.shape[1] == 256  # nfft freq bins

    def test_output_is_real(self):
        from spectra.transforms.stft import STFT

        stft = STFT(nfft=128, hop_length=32)
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        spectrogram = stft(iq)
        assert spectrogram.dtype == torch.float32


class TestToCoco:
    def test_basic_conversion(self):
        from spectra.scene.labels import STFTParams, to_coco
        from spectra.scene.signal_desc import SignalDescription

        descs = [
            SignalDescription(
                t_start=0.0,
                t_stop=0.0005,
                f_low=-100e3,
                f_high=100e3,
                label="QPSK",
                snr=20.0,
            ),
        ]
        params = STFTParams(
            nfft=256,
            hop_length=64,
            sample_rate=1e6,
            num_samples=1000,
        )
        result = to_coco(descs, params, class_list=["QPSK", "BPSK"])
        assert "boxes" in result
        assert "labels" in result
        assert isinstance(result["boxes"], torch.Tensor)
        assert result["boxes"].shape == (1, 4)
        assert isinstance(result["labels"], torch.Tensor)
        assert result["labels"].shape == (1,)

    def test_multiple_signals(self):
        from spectra.scene.labels import STFTParams, to_coco
        from spectra.scene.signal_desc import SignalDescription

        descs = [
            SignalDescription(0.0, 0.0005, -100e3, 100e3, "QPSK", 20.0),
            SignalDescription(0.0002, 0.0008, -300e3, -100e3, "BPSK", 15.0),
        ]
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        result = to_coco(descs, params, class_list=["QPSK", "BPSK"])
        assert result["boxes"].shape == (2, 4)
        assert result["labels"].shape == (2,)

    def test_boxes_are_valid(self):
        from spectra.scene.labels import STFTParams, to_coco
        from spectra.scene.signal_desc import SignalDescription

        descs = [
            SignalDescription(0.0, 0.0005, -100e3, 100e3, "QPSK", 20.0),
        ]
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        result = to_coco(descs, params, class_list=["QPSK"])
        box = result["boxes"][0]
        # x_min < x_max, y_min < y_max
        assert box[0] < box[2]
        assert box[1] < box[3]
        # All coords non-negative
        assert torch.all(box >= 0)

    def test_class_label_indices(self):
        from spectra.scene.labels import STFTParams, to_coco
        from spectra.scene.signal_desc import SignalDescription

        descs = [
            SignalDescription(0.0, 0.0005, -100e3, 100e3, "BPSK", 20.0),
            SignalDescription(0.0, 0.0005, -300e3, -100e3, "QPSK", 15.0),
        ]
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        class_list = ["QPSK", "BPSK"]
        result = to_coco(descs, params, class_list=class_list)
        assert result["labels"][0].item() == 1  # BPSK is index 1
        assert result["labels"][1].item() == 0  # QPSK is index 0

    def test_empty_descriptions(self):
        from spectra.scene.labels import STFTParams, to_coco

        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        result = to_coco([], params, class_list=["QPSK"])
        assert result["boxes"].shape == (0, 4)
        assert result["labels"].shape == (0,)

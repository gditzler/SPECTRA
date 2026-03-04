import torch

from spectra.transforms.stft import STFT


class Spectrogram(STFT):
    """STFT-based spectrogram with optional dB scaling.

    Returns [1, freq, time] magnitude tensor.
    """

    def __init__(self, nfft: int = 256, hop_length: int = 64, db_scale: bool = False):
        super().__init__(nfft=nfft, hop_length=hop_length)
        self.db_scale = db_scale

    def __call__(self, iq):
        result = super().__call__(iq)
        if self.db_scale:
            result = 20.0 * torch.log10(result + 1e-12)
        return result

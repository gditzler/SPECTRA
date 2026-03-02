import numpy as np
import torch


class Spectrogram:
    """STFT-based spectrogram with optional dB scaling.

    Returns [1, freq, time] magnitude tensor.
    """

    def __init__(self, nfft: int = 256, hop_length: int = 64, db_scale: bool = False):
        self.nfft = nfft
        self.hop_length = hop_length
        self.db_scale = db_scale

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq_tensor = torch.from_numpy(iq)
        window = torch.hann_window(self.nfft)
        stft_result = torch.stft(
            iq_tensor,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=window,
            return_complex=True,
        )
        stft_result = torch.fft.fftshift(stft_result, dim=0)
        magnitude = torch.abs(stft_result)
        if self.db_scale:
            magnitude = 20.0 * torch.log10(magnitude + 1e-12)
        return magnitude.unsqueeze(0).float()

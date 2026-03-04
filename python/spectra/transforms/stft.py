import numpy as np
import torch


class STFT:
    def __init__(self, nfft: int = 256, hop_length: int = 64):
        self.nfft = nfft
        self.hop_length = hop_length
        self._window = torch.hann_window(self.nfft)

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq_tensor = torch.from_numpy(iq)
        stft_result = torch.stft(
            iq_tensor,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=self._window,
            return_complex=True,
        )
        # fftshift along frequency axis so DC is centered
        stft_result = torch.fft.fftshift(stft_result, dim=0)
        magnitude = torch.abs(stft_result)
        # Return as [1, freq, time] for compatibility with image-based models
        return magnitude.unsqueeze(0).float()

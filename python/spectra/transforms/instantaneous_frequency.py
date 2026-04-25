import numpy as np
import torch


class InstantaneousFrequency:
    """Instantaneous frequency of a complex IQ signal via differential phase.

    Computes the sample-by-sample instantaneous frequency using:

        IF[n] = angle(x[n] * conj(x[n-1])) * sample_rate / (2 * pi)

    which is the first difference of the unwrapped phase, scaled to Hz.
    The first output sample is zero (no predecessor available).

    Args:
        sample_rate: Receiver sample rate in Hz. Used to scale the output to Hz.
            If ``None``, output is in radians per sample (unnormalized).
        normalize: If ``True``, divide by ``sample_rate / 2`` so the output
            is in [-1, 1] relative to the Nyquist frequency. Requires
            ``sample_rate`` to be set. Default ``False``.

    Example::

        from spectra import QPSK
        from spectra.transforms.instantaneous_frequency import InstantaneousFrequency

        iq = QPSK().generate(256, sample_rate=1e6, seed=0)
        transform = InstantaneousFrequency(sample_rate=1e6)
        freq = transform(iq)  # Tensor[1, N]
    """

    def __init__(self, sample_rate: float = 1.0, normalize: bool = False):
        self.sample_rate = sample_rate
        self.normalize = normalize

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        """Compute the instantaneous frequency of the input IQ signal.

        Args:
            iq: 1-D complex IQ array.

        Returns:
            ``torch.Tensor`` of shape ``[1, N]`` (``float32``).
        """
        # Differential phase: angle of x[n] * conj(x[n-1])
        diff = iq[1:] * np.conj(iq[:-1])
        inst_freq = np.angle(diff).astype(np.float32)

        # Scale to Hz: radians/sample * (sample_rate / 2*pi)
        inst_freq *= self.sample_rate / (2.0 * np.pi)

        # Prepend a zero so output length matches input length
        inst_freq = np.concatenate([np.zeros(1, dtype=np.float32), inst_freq])

        if self.normalize:
            nyquist = self.sample_rate / 2.0
            inst_freq /= nyquist

        return torch.from_numpy(inst_freq).unsqueeze(0)

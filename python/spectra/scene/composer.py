from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from spectra.scene.signal_desc import SignalDescription
from spectra.utils.dsp import frequency_shift
from spectra.waveforms.base import Waveform


@dataclass
class SceneConfig:
    """Configuration for a wideband multi-signal scene.

    Attributes:
        capture_duration: Total capture duration in seconds.
        capture_bandwidth: Total capture bandwidth in Hz, centered at DC.
        sample_rate: Sample rate in samples per second.
        num_signals: Number of signals to place. Either a fixed int or a
            ``(min, max)`` tuple for uniform random count per scene.
        signal_pool: List of :class:`~spectra.waveforms.base.Waveform` instances
            to draw from uniformly at random.
        snr_range: ``(min_db, max_db)`` per-signal SNR range drawn uniformly.
        allow_overlap: If ``False``, signals are frequency-packed to avoid
            spectral overlap. Default ``True``.

    Example::

        config = SceneConfig(
            capture_duration=0.001,
            capture_bandwidth=10e6,
            sample_rate=20e6,
            num_signals=(1, 5),
            signal_pool=[QPSK(), BPSK(), QAM16()],
            snr_range=(5.0, 20.0),
        )
    """

    capture_duration: float
    capture_bandwidth: float
    sample_rate: float
    num_signals: Union[int, Tuple[int, int]]
    signal_pool: List[Waveform]
    snr_range: Tuple[float, float]
    allow_overlap: bool = True


class Composer:
    def __init__(self, config: SceneConfig):
        self.config = config

    def generate(
        self, seed: int, impairments=None
    ) -> Tuple[np.ndarray, List[SignalDescription]]:
        rng = np.random.default_rng(seed)
        cfg = self.config

        num_capture_samples = int(cfg.capture_duration * cfg.sample_rate)
        composite = np.zeros(num_capture_samples, dtype=np.complex64)
        descriptions: List[SignalDescription] = []

        # Determine number of signals
        if isinstance(cfg.num_signals, tuple):
            n_signals = rng.integers(cfg.num_signals[0], cfg.num_signals[1] + 1)
        else:
            n_signals = cfg.num_signals

        half_bw = cfg.capture_bandwidth / 2.0

        for i in range(n_signals):
            # Pick a waveform from the pool
            waveform = cfg.signal_pool[rng.integers(0, len(cfg.signal_pool))]

            # Determine signal bandwidth
            sig_bw = waveform.bandwidth(cfg.sample_rate)

            # Random center frequency within capture bandwidth
            max_center = half_bw - sig_bw / 2.0
            if max_center <= -half_bw + sig_bw / 2.0:
                f_center = 0.0
            else:
                f_center = rng.uniform(-max_center, max_center)

            # Random SNR
            snr_db = rng.uniform(*cfg.snr_range)
            snr_linear = 10.0 ** (snr_db / 10.0)

            # Determine number of symbols to fill the capture
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = num_capture_samples // sps

            # Generate baseband IQ
            sig_seed = int(rng.integers(0, 2**32))
            iq = waveform.generate(
                num_symbols=num_symbols,
                sample_rate=cfg.sample_rate,
                seed=sig_seed,
            )

            # Truncate or pad to fit capture window
            if len(iq) >= num_capture_samples:
                iq = iq[:num_capture_samples]
            else:
                padded = np.zeros(num_capture_samples, dtype=np.complex64)
                # Random start time
                max_start = num_capture_samples - len(iq)
                start_idx = int(rng.integers(0, max(1, max_start)))
                padded[start_idx : start_idx + len(iq)] = iq
                iq = padded

            # Apply per-signal impairments if provided
            if impairments is not None:
                desc_temp = SignalDescription(
                    t_start=0.0,
                    t_stop=cfg.capture_duration,
                    f_low=f_center - sig_bw / 2.0,
                    f_high=f_center + sig_bw / 2.0,
                    label=waveform.label,
                    snr=snr_db,
                )
                iq, desc_temp = impairments(iq, desc_temp, sample_rate=cfg.sample_rate)

            # Frequency-shift to center frequency
            iq = frequency_shift(iq, f_center, cfg.sample_rate)

            # Scale to target SNR (relative to unit noise)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                iq = iq * np.sqrt(snr_linear / sig_power).astype(np.float32)

            # Find actual time extent (nonzero samples)
            nonzero = np.nonzero(np.abs(iq) > 1e-10)[0]
            if len(nonzero) > 0:
                t_start = nonzero[0] / cfg.sample_rate
                t_stop = (nonzero[-1] + 1) / cfg.sample_rate
            else:
                t_start = 0.0
                t_stop = cfg.capture_duration

            composite += iq

            descriptions.append(
                SignalDescription(
                    t_start=t_start,
                    t_stop=t_stop,
                    f_low=f_center - sig_bw / 2.0,
                    f_high=f_center + sig_bw / 2.0,
                    label=waveform.label,
                    snr=snr_db,
                )
            )

        return composite, descriptions

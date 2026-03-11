"""MIMO channel impairment."""

from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.impairments.tdl_channel import TDLChannel
from spectra.scene.signal_desc import SignalDescription


class MIMOChannel(Transform):
    """MIMO flat-fading or frequency-selective channel.

    Maps N_tx transmit streams to N_rx receive streams through an
    N_rx x N_tx channel matrix. Accepts 1D input (single TX assumed)
    or 2D input (n_tx, N_samples).

    Args:
        n_tx: Number of transmit antennas.
        n_rx: Number of receive antennas.
        channel_type: "flat" (i.i.d. Rayleigh) or "tdl" (per-element TDL).
        tdl_profile: TDL profile name (only for channel_type="tdl").
        spatial_correlation_rx: Optional (n_rx, n_rx) RX correlation matrix.
        spatial_correlation_tx: Optional (n_tx, n_tx) TX correlation matrix.
    """

    def __init__(
        self,
        n_tx: int = 2,
        n_rx: int = 2,
        channel_type: str = "flat",
        tdl_profile: str = "TDL-A",
        spatial_correlation_rx: Optional[np.ndarray] = None,
        spatial_correlation_tx: Optional[np.ndarray] = None,
    ):
        if channel_type not in ("flat", "tdl"):
            raise ValueError(f"Unknown channel_type: {channel_type!r}. Use 'flat' or 'tdl'.")
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.channel_type = channel_type
        self.tdl_profile = tdl_profile
        self.spatial_correlation_rx = spatial_correlation_rx
        self.spatial_correlation_tx = spatial_correlation_tx

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        # Handle 1D input (single TX antenna)
        if iq.ndim == 1:
            iq = iq.reshape(1, -1)
            if self.n_tx != 1:
                # Replicate to n_tx streams (same signal from each TX)
                iq = np.tile(iq, (self.n_tx, 1))

        if iq.shape[0] != self.n_tx:
            raise ValueError(f"Input has {iq.shape[0]} TX streams but n_tx={self.n_tx}")

        n_samples = iq.shape[1]

        if self.channel_type == "flat":
            rx = self._apply_flat(iq, n_samples)
        else:
            rx = self._apply_tdl(iq, desc, **kwargs)

        # Store MIMO metadata
        desc.modulation_params = desc.modulation_params or {}
        desc.modulation_params["mimo"] = {
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "channel_type": self.channel_type,
        }

        return rx.astype(np.complex64), desc

    def _apply_flat(self, iq: np.ndarray, n_samples: int) -> np.ndarray:
        """Apply flat (single-tap) Rayleigh fading."""
        # Generate i.i.d. complex Gaussian channel matrix
        H = (
            np.random.randn(self.n_rx, self.n_tx) + 1j * np.random.randn(self.n_rx, self.n_tx)
        ) / np.sqrt(2.0)

        # Apply spatial correlation if provided
        if self.spatial_correlation_rx is not None:
            # R_rx^(1/2)
            eigvals, eigvecs = np.linalg.eigh(self.spatial_correlation_rx)
            eigvals = np.maximum(eigvals, 0)
            sqrt_rx = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T.conj()
            H = sqrt_rx @ H

        if self.spatial_correlation_tx is not None:
            eigvals, eigvecs = np.linalg.eigh(self.spatial_correlation_tx)
            eigvals = np.maximum(eigvals, 0)
            sqrt_tx = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T.conj()
            H = H @ sqrt_tx.T

        # Output: H @ x, where x is (n_tx, n_samples)
        return H @ iq

    def _apply_tdl(self, iq: np.ndarray, desc: SignalDescription, **kwargs) -> np.ndarray:
        """Apply independent TDL channel per (rx, tx) element."""
        n_samples = iq.shape[1]
        rx = np.zeros((self.n_rx, n_samples), dtype=np.complex128)

        for rx_idx in range(self.n_rx):
            for tx_idx in range(self.n_tx):
                tdl = TDLChannel(profile=self.tdl_profile)
                tx_signal = iq[tx_idx]
                out, _ = tdl(tx_signal, desc, **kwargs)
                rx[rx_idx] += out[:n_samples]

        return rx

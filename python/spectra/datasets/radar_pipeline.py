"""End-to-end radar processing pipeline dataset.

Generates multi-CPI radar scenarios: waveform -> target injection -> clutter ->
matched filter -> MTI -> CFAR -> Kalman tracker, producing training data for
radar ML tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.algorithms.mti import doppler_filter_bank, single_pulse_canceller
from spectra.algorithms.radar import ca_cfar, os_cfar
from spectra.impairments.clutter import RadarClutter
from spectra.targets.rcs import NonFluctuatingRCS, SwerlingRCS
from spectra.tracking.kalman import ConstantVelocityKF, RangeDopplerKF
from spectra.waveforms.base import Waveform


@dataclass
class RadarPipelineTarget:
    """Ground-truth and processing results for one pipeline sample."""

    true_ranges: np.ndarray
    true_velocities: np.ndarray
    rcs_amplitudes: np.ndarray
    detections: List[np.ndarray]
    kf_states: np.ndarray
    num_targets: int
    waveform_label: str
    snr_db: float
    clutter_preset: str
    doppler_detections: Optional[List[np.ndarray]] = None


class RadarPipelineDataset(Dataset):
    """On-the-fly end-to-end radar pipeline dataset.

    Note:
        Trajectory ``range_at()`` returns values in range-bin units (not metres).
        Configure your trajectory ``initial_range`` and ``velocity`` accordingly.
    """

    def __init__(
        self,
        waveform_pool: List[Waveform],
        trajectory_pool: List,
        swerling_cases: Optional[List[int]] = None,
        clutter_presets: Optional[List[RadarClutter]] = None,
        num_range_bins: int = 256,
        sample_rate: float = 1e6,
        carrier_frequency: float = 10e9,
        pri: float = 1e-3,
        snr_range: Tuple[float, float] = (5.0, 25.0),
        num_targets_range: Tuple[int, int] = (1, 3),
        sequence_length: int = 1,
        pulses_per_cpi: int = 16,
        apply_mti: bool = True,
        cfar_type: str = "ca",
        track_doppler: bool = False,
        num_samples: int = 10000,
        seed: int = 0,
    ) -> None:
        self.waveform_pool = waveform_pool
        self.trajectory_pool = trajectory_pool
        self.swerling_cases = swerling_cases if swerling_cases is not None else [0, 1, 2, 3, 4]
        self.clutter_presets = clutter_presets if clutter_presets is not None else []
        self.num_range_bins = num_range_bins
        self.sample_rate = sample_rate
        self.carrier_frequency = carrier_frequency
        self.pri = pri
        self.snr_range = snr_range
        self.num_targets_range = num_targets_range
        self.sequence_length = sequence_length
        self.pulses_per_cpi = pulses_per_cpi
        self.apply_mti = apply_mti
        self.cfar_type = cfar_type
        self.track_doppler = track_doppler
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, RadarPipelineTarget]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        wf_idx = int(rng.integers(0, len(self.waveform_pool)))
        waveform = self.waveform_pool[wf_idx]

        n_targets = int(rng.integers(self.num_targets_range[0], self.num_targets_range[1] + 1))

        trajectories = []
        for _ in range(n_targets):
            traj_template = self.trajectory_pool[int(rng.integers(0, len(self.trajectory_pool)))]
            offset = rng.uniform(-50, 50)
            from spectra.targets.trajectory import ConstantTurnRate, ConstantVelocity
            if isinstance(traj_template, ConstantTurnRate):
                traj = ConstantTurnRate(
                    initial_range=traj_template.initial_range + offset,
                    velocity=traj_template.velocity + rng.uniform(-5, 5),
                    turn_rate=traj_template.turn_rate,
                    dt=traj_template.dt,
                )
            else:
                traj = ConstantVelocity(
                    initial_range=traj_template.initial_range + offset,
                    velocity=traj_template.velocity + rng.uniform(-5, 5),
                    dt=traj_template.dt,
                )
            trajectories.append(traj)

        swerling_case = int(rng.choice(self.swerling_cases))
        if swerling_case == 0:
            rcs_model = NonFluctuatingRCS(sigma=1.0)
        else:
            rcs_model = SwerlingRCS(case=swerling_case, sigma=1.0)

        rcs_amps = rcs_model.amplitudes(
            num_dwells=self.sequence_length,
            num_pulses_per_dwell=self.pulses_per_cpi,
            rng=rng,
        )

        snr_db = float(rng.uniform(self.snr_range[0], self.snr_range[1]))
        snr_linear = 10.0 ** (snr_db / 10.0)

        if self.clutter_presets:
            clutter = self.clutter_presets[int(rng.integers(0, len(self.clutter_presets)))]
            clutter_name = f"cnr_{clutter.cnr:.0f}dB"
        else:
            clutter = None
            clutter_name = "none"

        # Generate a short pulse template for matched filtering.
        # Cap at num_range_bins // 4 so the pulse is much shorter than the
        # range window, giving meaningful range resolution.
        max_pulse_len = max(self.num_range_bins // 16, 8)
        template = waveform.generate(
            num_symbols=1, sample_rate=self.sample_rate, seed=int(rng.integers(0, 2**32))
        )
        template = template[:max_pulse_len]
        if len(template) < max_pulse_len:
            padded = np.zeros(max_pulse_len, dtype=np.complex64)
            padded[: len(template)] = template
            template = padded
        template_len = len(template)

        range_profiles = []
        all_detections = []
        all_doppler_detections = []
        true_ranges = np.zeros((self.sequence_length, n_targets))
        true_velocities = np.zeros((self.sequence_length, n_targets))
        all_rcs_amps = np.zeros((self.sequence_length, n_targets))

        wavelength = 3e8 / self.carrier_frequency

        for frame in range(self.sequence_length):
            pulse_matrix = np.zeros(
                (self.pulses_per_cpi, self.num_range_bins), dtype=complex
            )

            for k, traj in enumerate(trajectories):
                true_state = traj.state_at(frame)
                true_range = true_state[0]
                true_vel = true_state[1]
                true_ranges[frame, k] = true_range
                true_velocities[frame, k] = true_vel

                range_bin = int(np.clip(round(true_range), 0, self.num_range_bins - 1))

                f_d = 2.0 * true_vel / wavelength if wavelength > 0 else 0.0

                for n in range(self.pulses_per_cpi):
                    amp = rcs_amps[frame, n] * np.sqrt(snr_linear)
                    doppler_phase = np.exp(1j * 2 * np.pi * f_d * n * self.pri)
                    # Center the template on range_bin so the MF peak
                    # (with mode="same") aligns with the true range.
                    half = template_len // 2
                    start = max(range_bin - half, 0)
                    end = min(range_bin - half + template_len, self.num_range_bins)
                    t_start = start - (range_bin - half)
                    t_end = t_start + (end - start)
                    if end > start:
                        pulse_matrix[n, start:end] += (
                            amp * doppler_phase * template[t_start:t_end]
                        )
                    all_rcs_amps[frame, k] = float(rcs_amps[frame, 0])

            noise = np.sqrt(0.5) * (
                rng.standard_normal(pulse_matrix.shape)
                + 1j * rng.standard_normal(pulse_matrix.shape)
            )
            pulse_matrix = pulse_matrix + noise

            if clutter is not None:
                pulse_matrix = clutter(pulse_matrix, rng)

            h = np.conj(template[::-1])
            mf_matrix = np.zeros_like(pulse_matrix)
            for n in range(self.pulses_per_cpi):
                mf_matrix[n] = np.convolve(pulse_matrix[n], h, mode="same")

            if self.apply_mti and self.pulses_per_cpi > 1:
                mf_matrix = single_pulse_canceller(mf_matrix)

            rdm = doppler_filter_bank(mf_matrix, window="hann")

            range_profile = np.max(rdm, axis=0)

            if self.cfar_type == "os":
                det_mask = os_cfar(range_profile, guard_cells=2, training_cells=8)
            else:
                det_mask = ca_cfar(range_profile, guard_cells=2, training_cells=8)

            det_bins = np.where(det_mask)[0]
            all_detections.append(det_bins)

            if self.track_doppler and len(det_bins) > 0:
                N_doppler = rdm.shape[0]
                raw_doppler = np.array(
                    [np.argmax(rdm[:, rb]) for rb in det_bins], dtype=int
                )
                centered_doppler = np.where(
                    raw_doppler < N_doppler // 2, raw_doppler, raw_doppler - N_doppler
                )
                all_doppler_detections.append(centered_doppler)
            elif self.track_doppler:
                all_doppler_detections.append(np.array([], dtype=int))

            range_profiles.append(range_profile)

        state_dim = 2
        kf_states = np.zeros((self.sequence_length, n_targets, state_dim))

        for k in range(n_targets):
            if self.track_doppler:
                kf = RangeDopplerKF(
                    dt=self.pri * self.pulses_per_cpi,
                    wavelength=wavelength,
                    pri=self.pri,
                    pulses_per_cpi=self.pulses_per_cpi,
                    process_noise_std=1.0,
                    range_noise_std=5.0,
                    doppler_noise_std=2.0,
                    x0=np.array([true_ranges[0, k], true_velocities[0, k]]),
                )
            else:
                kf = ConstantVelocityKF(
                    dt=self.pri * self.pulses_per_cpi,
                    process_noise_std=1.0,
                    measurement_noise_std=5.0,
                    x0=np.array([true_ranges[0, k], true_velocities[0, k]]),
                )

            for frame in range(self.sequence_length):
                predicted = kf.predict()
                dets_range = all_detections[frame]

                if len(dets_range) > 0:
                    if self.track_doppler:
                        dets_doppler = all_doppler_detections[frame]
                        H = kf.measurement_matrix
                        R = kf.measurement_noise
                        pred_z = H @ predicted
                        S = H @ kf.covariance @ H.T + R
                        S_inv = np.linalg.inv(S)
                        best_dist = float("inf")
                        best_idx = -1
                        for j in range(len(dets_range)):
                            z_j = np.array([float(dets_range[j]), float(dets_doppler[j])])
                            innov = z_j - pred_z
                            d = float(innov @ S_inv @ innov)
                            if d < best_dist:
                                best_dist = d
                                best_idx = j
                        gate_threshold = 9.21  # chi-squared, 2 DoF, 99%
                        if best_dist < gate_threshold and best_idx >= 0:
                            z = np.array([
                                float(dets_range[best_idx]),
                                float(dets_doppler[best_idx]),
                            ])
                            kf.update(z)
                    else:
                        pred_range = predicted[0]
                        nearest_idx = np.argmin(np.abs(dets_range - pred_range))
                        nearest_det = float(dets_range[nearest_idx])
                        gate = 20.0
                        if abs(nearest_det - pred_range) < gate:
                            kf.update(np.array([nearest_det]))

                kf_states[frame, k] = kf.state

        profiles = np.stack(range_profiles)

        profiles_db = 10.0 * np.log10(profiles + 1e-30)
        p_min = profiles_db.min()
        p_max = profiles_db.max()
        if p_max > p_min:
            profiles_norm = (profiles_db - p_min) / (p_max - p_min)
        else:
            profiles_norm = np.zeros_like(profiles_db)

        tensor = torch.from_numpy(profiles_norm.astype(np.float32))

        target = RadarPipelineTarget(
            true_ranges=true_ranges,
            true_velocities=true_velocities,
            rcs_amplitudes=all_rcs_amps,
            detections=all_detections,
            kf_states=kf_states,
            num_targets=n_targets,
            waveform_label=waveform.label,
            snr_db=snr_db,
            clutter_preset=clutter_name,
            doppler_detections=all_doppler_detections if self.track_doppler else None,
        )
        return tensor, target

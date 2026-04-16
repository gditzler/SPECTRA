# examples/18_link_simulator.py
"""Example 18 — Link-Level Simulator: BER/SER/PER Curves
=========================================================
Level: Advanced

This example shows how to:
  1. Use the CoherentReceiver to demodulate noiseless and noisy waveforms
  2. Compute BER, SER, and PER with the new metric functions
  3. Run a LinkSimulator sweep to generate BER vs. Eb/N0 curves
  4. Compare simulated BPSK BER against the theoretical curve
  5. Compare BER curves across modulation schemes (BPSK, QPSK, QAM16)

Run:
    python examples/18_link_simulator.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from spectra.waveforms import BPSK, QPSK, QAM16
from spectra.receivers import CoherentReceiver
from spectra.link import LinkSimulator, LinkResults
from spectra.metrics import bit_error_rate, symbol_error_rate, packet_error_rate
from spectra._rust import (
    generate_bpsk_symbols_with_indices,
    generate_qpsk_symbols_with_indices,
    apply_rrc_filter_with_taps,
    get_bpsk_constellation,
    get_qpsk_constellation,
    get_qam_constellation,
)
from spectra.receivers.coherent import constellation_to_bits
from spectra.utils.rrc_cache import cached_rrc_taps

from plot_helpers import savefig, OUTPUT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

SEED         = 42
NUM_SYMBOLS  = 10000
PACKET_LEN   = 200
EB_N0_RANGE  = np.arange(0, 14, 1.0)

# ── 1. Constellation and Receiver Demo ────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (label, get_const) in zip(axes, [
    ("BPSK", get_bpsk_constellation),
    ("QPSK", get_qpsk_constellation),
    ("16QAM", lambda: get_qam_constellation(16)),
]):
    const = np.array(get_const())
    ax.scatter(const.real, const.imag, s=80, c="steelblue", zorder=5)
    for i, pt in enumerate(const):
        ax.annotate(str(i), (pt.real, pt.imag), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="gray")
    ax.set_title(f"{label} Constellation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

plt.tight_layout()
savefig("18_constellations.png")
print("1. Constellations plotted")

# ── 2. Noiseless Demodulation ─────────────────────────────────────────────────

wf = BPSK(samples_per_symbol=8)
symbols, tx_indices = generate_bpsk_symbols_with_indices(500, SEED)
taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

rx = CoherentReceiver(wf)
rx_indices, rx_bits = rx.demodulate(tx_iq)

# Compare within valid region (trim filter transients)
trim = wf.filter_span
tx_arr = np.array(tx_indices)
valid_tx = tx_arr[trim:trim + len(rx_indices) - trim]
valid_rx = rx_indices[trim:trim + len(valid_tx)]

ber_noiseless = bit_error_rate(
    constellation_to_bits(valid_tx, 2),
    constellation_to_bits(valid_rx, 2),
)
print(f"2. Noiseless BPSK demodulation BER: {ber_noiseless:.6f}")

# ── 3. BER/SER/PER Metric Demo ───────────────────────────────────────────────

# Simulate at a single Eb/N0 point to show metrics
sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8),
                    num_symbols=5000, packet_length=PACKET_LEN, seed=SEED)
result_single = sim.run(np.array([4.0]))

print(f"3. BPSK at Eb/N0 = 4 dB:")
print(f"   BER = {result_single.ber[0]:.4e}")
print(f"   SER = {result_single.ser[0]:.4e}")
print(f"   PER = {result_single.per[0]:.4e} (packet_length={PACKET_LEN})")

# ── 4. BPSK BER vs. Theoretical ──────────────────────────────────────────────

sim_bpsk = LinkSimulator(waveform=BPSK(samples_per_symbol=8),
                         num_symbols=NUM_SYMBOLS, seed=SEED)
results_bpsk = sim_bpsk.run(EB_N0_RANGE)

theory = results_bpsk.theoretical_ber()

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(results_bpsk.eb_n0_db, results_bpsk.ber, "o-", color="steelblue",
            label="Simulated BER", markersize=5)
if theory is not None:
    ax.semilogy(results_bpsk.eb_n0_db, theory, "k--", label="Theoretical BER", linewidth=1.5)
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("Bit Error Rate")
ax.set_title(f"BPSK BER — Simulated vs. Theoretical ({NUM_SYMBOLS} symbols)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(bottom=1e-5)
plt.tight_layout()
savefig("18_bpsk_ber_vs_theory.png")
print("4. BPSK BER vs. theoretical plotted")

# ── 5. Multi-Modulation BER Comparison ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
colors = {"BPSK": "steelblue", "QPSK": "coral", "16QAM": "seagreen"}

for wf_cls, label in [(BPSK, "BPSK"), (QPSK, "QPSK"), (QAM16, "16QAM")]:
    sim = LinkSimulator(waveform=wf_cls(samples_per_symbol=8),
                        num_symbols=NUM_SYMBOLS, seed=SEED)
    results = sim.run(EB_N0_RANGE)
    # Replace zero BER with NaN so they don't appear on log plot
    ber_plot = results.ber.copy()
    ber_plot[ber_plot == 0] = np.nan
    ax.semilogy(results.eb_n0_db, ber_plot, "o-", color=colors[label],
                label=label, markersize=5)

ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("Bit Error Rate")
ax.set_title(f"BER Comparison — BPSK vs. QPSK vs. 16QAM ({NUM_SYMBOLS} symbols)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(bottom=1e-5)
plt.tight_layout()
savefig("18_ber_comparison.png")
print("5. Multi-modulation BER comparison plotted")

# ── 6. SER and PER Curves ────────────────────────────────────────────────────

sim_full = LinkSimulator(waveform=QPSK(samples_per_symbol=8),
                         num_symbols=NUM_SYMBOLS, packet_length=PACKET_LEN, seed=SEED)
results_full = sim_full.run(EB_N0_RANGE)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SER
ser_plot = results_full.ser.copy()
ser_plot[ser_plot == 0] = np.nan
axes[0].semilogy(results_full.eb_n0_db, ser_plot, "s-", color="coral", markersize=5)
axes[0].set_xlabel("Eb/N0 (dB)")
axes[0].set_ylabel("Symbol Error Rate")
axes[0].set_title("QPSK Symbol Error Rate")
axes[0].grid(True, which="both", alpha=0.3)

# PER
per_plot = results_full.per.copy()
per_plot[per_plot == 0] = np.nan
axes[1].semilogy(results_full.eb_n0_db, per_plot, "D-", color="seagreen", markersize=5)
axes[1].set_xlabel("Eb/N0 (dB)")
axes[1].set_ylabel("Packet Error Rate")
axes[1].set_title(f"QPSK Packet Error Rate (L={PACKET_LEN} bits)")
axes[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
savefig("18_ser_per_curves.png")
print("6. SER and PER curves plotted")

print(f"\nAll figures saved to {OUTPUT_DIR}")

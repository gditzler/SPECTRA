# python/spectra/studio/export_tab.py
"""Export tab for SPECTRA Studio."""

from __future__ import annotations

import os
import numpy as np
import gradio as gr

from spectra.cli.config_builder import build_config, serialize_config


def _export_single(iq_data, meta, output_path, fmt, center_freq):
    """Export current signal as a single file."""
    if iq_data is None:
        return "No signal generated. Go to Generate tab first."
    sr = meta.get("sample_rate", 1e6)
    label = meta.get("waveform_label", "unknown")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "SigMF":
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
        writer = SigMFWriter(output_path, sample_rate=sr, center_frequency=center_freq)
        writer.write(iq_data)
        return f"Exported SigMF to {output_path}.sigmf-meta ({len(iq_data)} samples)"
    elif fmt == "NumPy":
        np.save(f"{output_path}.npy", iq_data)
        return f"Exported NumPy to {output_path}.npy"
    elif fmt == "Raw IQ":
        iq_data.astype(np.complex64).tofile(f"{output_path}.cf32")
        return f"Exported raw IQ to {output_path}.cf32"
    return "Unknown format"


def _save_config_only(config_state):
    """Serialize config to YAML string for download."""
    if not config_state:
        return "No config available. Generate a signal first."
    return serialize_config(config_state)


def build_export_tab(iq_state, meta_state, config_state):
    """Build the Export tab UI."""
    mode = gr.Radio(["Single file", "Dataset"], value="Single file", label="Export Mode")
    fmt = gr.Dropdown(choices=["SigMF", "NumPy", "Raw IQ"], value="SigMF", label="Format")
    output_path = gr.Textbox(value="./spectra_output/signal", label="Output Path (base name)")
    center_freq = gr.Number(value=0.0, label="Center Frequency (Hz)")

    # Dataset mode controls (hidden by default)
    with gr.Group(visible=False) as dataset_group:
        n_train = gr.Number(value=50000, label="Train Samples", precision=0)
        n_val = gr.Number(value=10000, label="Val Samples", precision=0)
        n_test = gr.Number(value=10000, label="Test Samples", precision=0)
        snr_min = gr.Slider(-10, 40, value=-10, label="SNR Min (dB)")
        snr_max = gr.Slider(-10, 40, value=30, label="SNR Max (dB)")

    mode.change(lambda m: gr.update(visible=(m == "Dataset")), inputs=[mode], outputs=[dataset_group])

    export_btn = gr.Button("Export", variant="primary")
    save_yaml_btn = gr.Button("Save Config Only", variant="secondary")
    status = gr.Textbox(label="Status", interactive=False)
    yaml_output = gr.Code(label="YAML Config", language="yaml", visible=False)

    export_btn.click(
        _export_single,
        inputs=[iq_state, meta_state, output_path, fmt, center_freq],
        outputs=[status],
    )

    def save_yaml(cfg):
        yaml_str = _save_config_only(cfg)
        return gr.update(visible=True, value=yaml_str)

    save_yaml_btn.click(save_yaml, inputs=[config_state], outputs=[yaml_output])

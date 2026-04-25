# python/spectra/studio/visualize_tab.py
"""Visualize tab for SPECTRA Studio."""

from __future__ import annotations

import gradio as gr

from spectra.studio.plotting import (
    plot_ambiguity,
    plot_constellation,
    plot_eye,
    plot_fft,
    plot_iq,
    plot_scd,
    plot_waterfall,
)


def _load_file(file_obj):
    """Load IQ from uploaded file, return (iq, sample_rate)."""
    if file_obj is None:
        return None, 1e6
    from spectra.utils.file_handlers import SigMFReader, get_reader
    path = file_obj.name
    if path.endswith(".sigmf-meta"):
        reader = SigMFReader()
    else:
        reader = get_reader(path)
    iq, meta = reader.read(path)
    sr = meta.sample_rate or 1e6
    return iq, sr


def build_visualize_tab(iq_state, meta_state):
    """Build the Visualize tab UI."""
    source = gr.Radio(
        ["Use generated signal", "Load file"],
        value="Use generated signal",
        label="Data Source",
    )
    file_upload = gr.File(
        label="Upload IQ file",
        file_types=[".sigmf-meta", ".cf32", ".npy", ".raw"],
        visible=False,
    )
    sr_input = gr.Number(value=1e6, label="Sample Rate (Hz)")

    source.change(
        lambda s: gr.update(visible=(s == "Load file")),
        inputs=[source],
        outputs=[file_upload],
    )

    with gr.Tabs():
        for name, plot_fn, extra_inputs in [
            ("IQ", plot_iq, []),
            ("FFT", plot_fft, []),
            ("Waterfall", plot_waterfall, []),
            ("Constellation", plot_constellation, []),
            ("SCD", plot_scd, []),
            ("Ambiguity", plot_ambiguity, []),
            ("Eye Diagram", plot_eye, []),
        ]:
            with gr.Tab(name):
                plot_output = gr.Plot(label=name)
                sps_input = gr.Number(
                    value=8,
                    label="Samples per Symbol",
                    visible=(name == "Eye Diagram"),
                )
                plot_btn = gr.Button(f"Plot {name}")

                def make_callback(fn, tab_name):
                    def cb(iq_data, meta, src, file_obj, sr, sps):
                        if src == "Load file":
                            iq, sr_loaded = _load_file(file_obj)
                            if iq is None:
                                return None
                            sr = sr_loaded
                        else:
                            iq = iq_data
                        if iq is None:
                            return None
                        if tab_name == "Eye Diagram":
                            return fn(iq, samples_per_symbol=int(sps))
                        elif tab_name in ("Constellation", "Ambiguity"):
                            return fn(iq)
                        else:
                            return fn(iq, sample_rate=sr)
                    return cb

                plot_btn.click(
                    make_callback(plot_fn, name),
                    inputs=[iq_state, meta_state, source, file_upload, sr_input, sps_input],
                    outputs=[plot_output],
                )

# python/spectra/studio/app.py
"""SPECTRA Studio — Gradio application assembly."""

from __future__ import annotations

import gradio as gr
import numpy as np

from spectra.studio.theme import spectra_theme


def create_app(dark: bool = False) -> gr.Blocks:
    """Create and return the SPECTRA Studio Gradio app."""
    theme = spectra_theme(dark=dark)
    css = ".gradio-container { max-width: 1200px; margin: auto; }"

    # Store theme/css for launch() (Gradio 6.0+ moved these from constructor)
    with gr.Blocks(title="SPECTRA Studio") as app:
        app._spectra_theme = theme
        app._spectra_css = css
        gr.Markdown("# SPECTRA Studio\nInteractive RF waveform generation, visualization, and export.")

        # Shared state
        iq_state = gr.State(value=None)
        meta_state = gr.State(value={})
        config_state = gr.State(value={})

        with gr.Tabs():
            with gr.Tab("Generate"):
                from spectra.studio.generate_tab import build_generate_tab
                build_generate_tab(iq_state, meta_state, config_state)

            with gr.Tab("Visualize"):
                from spectra.studio.visualize_tab import build_visualize_tab
                build_visualize_tab(iq_state, meta_state)

            with gr.Tab("Export"):
                from spectra.studio.export_tab import build_export_tab
                build_export_tab(iq_state, meta_state, config_state)

    return app

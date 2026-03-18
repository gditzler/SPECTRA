"""Custom Gradio theme for SPECTRA Studio."""

from __future__ import annotations

try:
    import gradio as gr

    def spectra_theme(dark: bool = False) -> gr.themes.Base:
        """Create the SPECTRA Studio theme."""
        theme = gr.themes.Soft(
            primary_hue=gr.themes.Color(
                c50="#e8f0fe", c100="#c5d9f7", c200="#9ebfef",
                c300="#7aa5e7", c400="#5a8edf", c500="#4a90d9",
                c600="#3b73ae", c700="#2d5783", c800="#1e3a58",
                c900="#101e2d", c950="#0a1119",
            ),
            secondary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.slate,
        ).set(
            body_background_fill_dark="#0d1117",
            block_background_fill_dark="#1a1a2e",
            button_primary_background_fill="#4a90d9",
            button_primary_background_fill_hover="#5a9ee3",
            button_secondary_background_fill="#50b87a",
            button_secondary_background_fill_hover="#5fc888",
        )
        return theme

except ImportError:
    def spectra_theme(dark: bool = False):
        raise ImportError("gradio is required for theming. Install with: pip install spectra[ui]")

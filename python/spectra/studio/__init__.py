"""SPECTRA Studio — interactive RF dataset generation and visualization."""


def launch(port: int = 7860, share: bool = False, dark: bool = False) -> None:
    """Launch the SPECTRA Studio Gradio app.

    Args:
        port: Local port for the Gradio server.
        share: Create a public Gradio share link.
        dark: Start in dark mode.
    """
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise ImportError(
            "SPECTRA Studio requires gradio. Install with: pip install spectra[ui]"
        )

    from spectra.studio.app import create_app

    app = create_app(dark=dark)
    app.launch(server_port=port, share=share)

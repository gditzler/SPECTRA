# python/spectra/studio/generate_tab.py
"""Generate tab for SPECTRA Studio."""

from __future__ import annotations

import numpy as np
import gradio as gr

from spectra.cli.config_builder import (
    IMPAIRMENT_PRESETS,
    get_waveform_registry,
    get_impairment_registry,
    build_config,
)
from spectra.studio.params import get_all_categories, get_waveforms_for_category, get_waveform_params
from spectra.studio.plotting import plot_constellation, plot_fft


def _generate_signal(category, waveform_name, sample_rate, num_samples, snr_db, seed,
                     preset, wideband, capture_bw, num_signals, *param_values):
    """Callback: generate IQ, return (iq, meta, config, constellation_fig, psd_fig)."""
    try:
        return _generate_signal_inner(
            category, waveform_name, sample_rate, num_samples, snr_db, seed,
            preset, wideband, capture_bw, num_signals, *param_values,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return error state so Gradio doesn't hang
        return None, {"error": str(e)}, {}, None, None


def _generate_signal_inner(category, waveform_name, sample_rate, num_samples, snr_db, seed,
                           preset, wideband, capture_bw, num_signals, *param_values):
    """Inner generation logic."""
    registry = get_waveform_registry()
    cls = registry.get(waveform_name)
    if cls is None:
        return None, {}, {}, None, None

    # Build kwargs from dynamic params
    params = get_waveform_params(waveform_name)
    kwargs = {}
    for i, p in enumerate(params[:10]):  # max 10 param slots
        if i < len(param_values) and param_values[i] is not None:
            val = param_values[i]
            if p["type"] == "int":
                val = int(val)
            elif p["type"] == "float":
                val = float(val)
            kwargs[p["name"]] = val

    waveform = cls(**kwargs)

    if wideband:
        from spectra.scene.composer import Composer, SceneConfig
        # Ensure capture_bw doesn't exceed sample_rate (Nyquist)
        capture_bw = min(float(capture_bw), float(sample_rate) * 0.9)
        # Cap preview duration to keep generation fast
        max_preview_samples = min(int(num_samples), 32768)
        config = SceneConfig(
            capture_duration=max_preview_samples / sample_rate,
            capture_bandwidth=capture_bw,
            sample_rate=sample_rate,
            num_signals=int(num_signals),
            signal_pool=[waveform],
            snr_range=(snr_db, snr_db + 10),
        )
        composer = Composer(config)
        iq, descs = composer.generate(seed=int(seed))
        label = f"Scene ({len(descs)} signals)"
    else:
        sps = getattr(waveform, "samples_per_symbol", 8)
        n_samples = min(int(num_samples), 32768)  # cap preview for speed
        n_sym = max(1, n_samples // sps)
        iq = waveform.generate(num_symbols=n_sym, sample_rate=float(sample_rate), seed=int(seed))
        iq = iq[:n_samples]
        label = waveform.label

    # Apply impairments
    if preset != "clean" and preset in IMPAIRMENT_PRESETS:
        from spectra.impairments import Compose, AWGN
        from spectra.scene.signal_desc import SignalDescription
        imp_reg = get_impairment_registry()
        chain = []
        for entry in IMPAIRMENT_PRESETS[preset]:
            imp_cls = imp_reg.get(entry["type"])
            if imp_cls:
                chain.append(imp_cls(**entry.get("params", {})))
        if chain:
            desc = SignalDescription(0, len(iq)/sample_rate, -sample_rate/2, sample_rate/2, label, snr_db)
            iq, _ = Compose(chain)(iq, desc, sample_rate=sample_rate)

    meta = {"waveform_label": label, "sample_rate": sample_rate, "num_samples": len(iq), "snr_db": snr_db}
    fig_const = plot_constellation(iq)
    fig_psd = plot_fft(iq, sample_rate=sample_rate)
    return iq, meta, {}, fig_const, fig_psd


def build_generate_tab(iq_state, meta_state, config_state):
    """Build the Generate tab UI components and wire callbacks."""
    categories = get_all_categories()

    with gr.Row():
        with gr.Column(scale=2):
            category = gr.Dropdown(choices=categories, value=categories[0], label="Category")
            waveform = gr.Dropdown(choices=get_waveforms_for_category(categories[0]), label="Waveform")
            sample_rate = gr.Number(value=1e6, label="Sample Rate (Hz)")
            num_samples = gr.Number(value=1024, label="Number of IQ Samples", precision=0)
            snr_db = gr.Slider(-10, 40, value=20, label="SNR (dB)")
            seed = gr.Number(value=42, label="Seed", precision=0)
            preset = gr.Dropdown(choices=["clean", "mild", "realistic"], value="clean", label="Impairment Preset")

            wideband = gr.Checkbox(value=False, label="Wideband Scene Mode")
            capture_bw = gr.Number(value=800e3, label="Capture Bandwidth (Hz)", visible=False)
            num_signals = gr.Number(value=3, label="Number of Signals", visible=False, precision=0)
            wideband.change(lambda w: (gr.update(visible=w), gr.update(visible=w)),
                           inputs=[wideband], outputs=[capture_bw, num_signals])

            # Dynamic param slots (10 generic components)
            param_slots = []
            for i in range(10):
                slot = gr.Number(value=0, label=f"Param {i}", visible=False)
                param_slots.append(slot)

            def update_waveform_list(cat):
                wfs = get_waveforms_for_category(cat)
                return gr.update(choices=wfs, value=wfs[0] if wfs else None)

            def update_params(wf_name):
                params = get_waveform_params(wf_name)
                updates = []
                for i in range(10):
                    if i < len(params):
                        p = params[i]
                        updates.append(gr.update(visible=True, label=p["label"],
                                                  value=p["default"] if p["default"] is not None else 0))
                    else:
                        updates.append(gr.update(visible=False))
                return updates

            category.change(update_waveform_list, inputs=[category], outputs=[waveform])
            waveform.change(update_params, inputs=[waveform], outputs=param_slots)

            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            const_plot = gr.Plot(label="Constellation")
            psd_plot = gr.Plot(label="Power Spectral Density")
            info = gr.Markdown("*Click Generate to preview*")

    gen_btn.click(
        _generate_signal,
        inputs=[category, waveform, sample_rate, num_samples, snr_db, seed,
                preset, wideband, capture_bw, num_signals] + param_slots,
        outputs=[iq_state, meta_state, config_state, const_plot, psd_plot],
    )

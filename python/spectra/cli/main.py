# python/spectra/cli/main.py
"""Unified CLI for SPECTRA: studio, generate, viz, build."""

from __future__ import annotations

import argparse
import sys


def _cmd_studio(args: argparse.Namespace) -> None:
    """Launch SPECTRA Studio (Gradio UI)."""
    from spectra.studio import launch

    launch(port=args.port, share=args.share, dark=args.dark)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Headless batch generation from YAML config."""
    from spectra.benchmarks.loader import load_benchmark
    from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

    dataset = load_benchmark(args.config, split=args.split)

    if isinstance(dataset, tuple):
        # "all" split returns a 3-tuple
        for split_name, ds in zip(["train", "val", "test"], dataset):
            output_dir = f"{args.output}/{split_name}"
            print(f"Exporting {split_name} split ({len(ds)} samples) to {output_dir}...")
            SigMFWriter.write_from_dataset(
                ds, output_dir, sample_rate=args.sample_rate or 1e6
            )
    else:
        print(f"Exporting {args.split} split ({len(dataset)} samples) to {args.output}...")
        SigMFWriter.write_from_dataset(
            dataset, args.output, sample_rate=args.sample_rate or 1e6
        )
    print("Done.")


def _cmd_viz(args: argparse.Namespace) -> None:
    """Quick visualization of an IQ file."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectra.utils.file_handlers import get_reader

    reader = get_reader(args.file)
    iq, metadata = reader.read(args.file)
    sr = metadata.sample_rate or 1e6

    from spectra.studio.plotting import plot_constellation, plot_fft, plot_iq, plot_waterfall

    plot_funcs = {
        "fft": lambda: plot_fft(iq, sample_rate=sr, dark=False),
        "iq": lambda: plot_iq(iq, sample_rate=sr, dark=False),
        "waterfall": lambda: plot_waterfall(iq, sample_rate=sr, dark=False),
        "constellation": lambda: plot_constellation(iq, dark=False),
    }

    fig = plot_funcs[args.plot]()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


def _cmd_build(args: argparse.Namespace) -> None:
    """Run the interactive signal builder wizard."""
    from spectra.cli.signal_builder import main as builder_main

    builder_main()


def main() -> None:
    """SPECTRA CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spectra",
        description="SPECTRA — RF waveform generation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # studio
    sp_studio = subparsers.add_parser("studio", help="Launch SPECTRA Studio UI")
    sp_studio.add_argument("--port", type=int, default=7860)
    sp_studio.add_argument("--share", action="store_true")
    sp_studio.add_argument("--dark", action="store_true")

    # generate
    sp_gen = subparsers.add_parser("generate", help="Headless batch generation")
    sp_gen.add_argument("--config", required=True, help="YAML config path")
    sp_gen.add_argument("--output", default="./spectra_output", help="Output directory")
    sp_gen.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    sp_gen.add_argument("--sample-rate", type=float, default=None)

    # viz (requires spectra[ui] for plotting functions)
    sp_viz = subparsers.add_parser("viz", help="Quick IQ file visualization")
    sp_viz.add_argument("file", help="Path to IQ file (.sigmf-meta, .cf32, .npy)")
    sp_viz.add_argument(
        "--plot", default="fft",
        choices=["fft", "iq", "waterfall", "constellation"],
    )
    sp_viz.add_argument("--save", default=None, help="Save plot to file instead of showing")

    # build
    subparsers.add_parser("build", help="Interactive signal builder wizard")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "studio": _cmd_studio,
        "generate": _cmd_generate,
        "viz": _cmd_viz,
        "build": _cmd_build,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()

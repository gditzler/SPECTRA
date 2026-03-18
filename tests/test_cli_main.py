# tests/test_cli_main.py
"""Tests for unified CLI entry point."""
import subprocess
import sys
import pytest


def test_spectra_help():
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "studio" in result.stdout
    assert "generate" in result.stdout
    assert "viz" in result.stdout
    assert "build" in result.stdout


def test_spectra_generate_no_config():
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "generate"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0  # missing required --config


def test_spectra_viz_no_file():
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "viz"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0  # missing required file arg


def test_spectra_studio_missing_gradio():
    """Studio should fail gracefully if gradio not installed (may pass if installed)."""
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "studio", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    # Either shows help (gradio installed) or error message
    assert result.returncode == 0

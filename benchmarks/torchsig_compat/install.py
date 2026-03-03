"""Install TorchSig from GitHub into the current environment."""
import subprocess
import sys
import tempfile
from pathlib import Path


def is_torchsig_installed():
    """Check if TorchSig is importable."""
    try:
        import torchsig  # noqa: F401
        return True
    except ImportError:
        return False


def install_torchsig(branch="main"):
    """Clone and pip-install TorchSig from GitHub."""
    if is_torchsig_installed():
        print("TorchSig is already installed.")
        return

    repo_url = "https://github.com/TorchDSP/torchsig.git"
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "torchsig"
        print(f"Cloning TorchSig ({branch}) ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--branch", branch,
             repo_url, str(clone_dir)],
            stdout=subprocess.DEVNULL,
        )
        print("Installing TorchSig ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(clone_dir)],
            stdout=subprocess.DEVNULL,
        )

    print("TorchSig installed successfully.")


if __name__ == "__main__":
    install_torchsig()

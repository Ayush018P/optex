"""Download Optiver data via Kaggle API if available."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download() -> None:
    dataset = "optiver-realized-volatility-prediction"
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", dataset, "-p", str(DATA_DIR)], check=True)
    except Exception as exc:  # pragma: no cover
        print(f"Kaggle download failed: {exc}. Using sample data instead.")


if __name__ == "__main__":
    download()

"""Render docs/assets/src/pipeline.html into docs/assets/pipeline.png using headless Chrome.

Usage:
    .venv/bin/python docs/assets/src/build_pipeline.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HTML_PATH = ROOT / "docs" / "assets" / "src" / "pipeline.html"
OUT_PATH = ROOT / "docs" / "assets" / "pipeline.png"


def find_chrome() -> str:
    """Find a Chrome or Chromium executable."""
    env = os.environ.get("CHROME_BIN")
    if env and Path(env).is_file():
        return env

    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    for candidate in candidates:
        if candidate.startswith("/"):
            if Path(candidate).is_file():
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found

    msg = "Could not find a Chrome or Chromium executable. Set CHROME_BIN."
    raise FileNotFoundError(msg)


def build() -> None:
    chrome = find_chrome()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_png = Path(tmp.name)

    try:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--force-device-scale-factor=2",
                "--window-size=1420,650",
                f"--screenshot={tmp_png}",
                HTML_PATH.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )

        im = Image.open(tmp_png).convert("RGB")
        arr = np.array(im)
        bg_rgb = arr[-1, -1]
        diff = np.max(np.abs(arr.astype(int) - bg_rgb.astype(int)), axis=2)
        ys, _ = np.where(diff > 2)
        y_max = min(im.height, int(ys.max()) + 40)
        im_cropped = im.crop((0, 0, im.width, y_max))

        im_q = im_cropped.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        im_q.save(OUT_PATH, optimize=True)
        kb = OUT_PATH.stat().st_size / 1024
        print(f"Rendered {OUT_PATH.name}: {im_cropped.size} -> {kb:.1f} KB")
    finally:
        if tmp_png.exists():
            tmp_png.unlink()


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""Build VibeMouse into a single-file Windows EXE."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _ROOT / "build"
_ICO_PATH = _BUILD_DIR / "vibemouse.ico"
_SPEC_PATH = _ROOT / "vibemouse.spec"


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def _check_dependencies() -> None:
    missing: list[str] = []
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        missing.append('pyinstaller>=6.0')
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        missing.append('sherpa-onnx')
    if missing:
        print("Missing build dependencies:")
        for dep in missing:
            print(f'  pip install "{dep}"')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model check
# ---------------------------------------------------------------------------


def _check_model() -> None:
    """Check whether the ASR model is already downloaded."""
    model_dir = Path.home() / ".cache" / "vibemouse" / "models"
    model_name = "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16"
    model_path = model_dir / model_name

    print("=== Model status ===")
    if (model_path / "tokens.txt").exists() and list(model_path.glob("encoder*.onnx")):
        size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        print(f"  Model found: {model_path}")
        print(f"  Size: {size_mb / (1024 * 1024):.0f} MB")
    else:
        print(f"  Model not found at: {model_path}")
        print("  Model will be downloaded automatically on first launch (~1.3 GB)")
    print()


# ---------------------------------------------------------------------------
# ICO generation
# ---------------------------------------------------------------------------


def _generate_ico() -> None:
    """Generate a .ico file for the EXE using Pillow."""
    from PIL import Image, ImageDraw

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 48, 64, 128, 256]
    images: list[Image.Image] = []
    for size in sizes:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        margin = max(1, size // 16)
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill="#22c55e",
            outline="white",
            width=max(1, size // 32),
        )
        images.append(img)

    images[0].save(
        str(_ICO_PATH),
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"Icon generated: {_ICO_PATH}")


# ---------------------------------------------------------------------------
# PyInstaller
# ---------------------------------------------------------------------------


def _run_pyinstaller() -> None:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(_SPEC_PATH),
        "--distpath",
        str(_ROOT / "dist"),
        "--workpath",
        str(_BUILD_DIR / "pyinstaller"),
        "--noconfirm",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print("PyInstaller build failed.")
        sys.exit(result.returncode)

    exe_path = _ROOT / "dist" / "VibeMouse.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\nBuild succeeded: {exe_path} ({size_mb:.1f} MB)")
    else:
        print("Build completed but EXE not found at expected path.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 50)
    print("  VibeMouse Build")
    print("=" * 50)
    print()

    _check_dependencies()
    _check_model()
    _generate_ico()

    print("=== Building EXE ===")
    _run_pyinstaller()

    print()
    print("=" * 50)
    print("  Done!  dist/VibeMouse.exe")
    print("=" * 50)


if __name__ == "__main__":
    main()

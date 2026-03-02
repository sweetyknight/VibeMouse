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


def _check_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Install it with:")
        print('  pip install "pyinstaller>=6.0"')
        sys.exit(1)


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

    # Save as multi-size ICO
    images[0].save(
        str(_ICO_PATH),
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"Icon generated: {_ICO_PATH}")


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
        print(f"Build succeeded: {exe_path} ({size_mb:.1f} MB)")
    else:
        print("Build completed but EXE not found at expected path.")
        sys.exit(1)


def main() -> None:
    _check_pyinstaller()
    _generate_ico()
    _run_pyinstaller()


if __name__ == "__main__":
    main()

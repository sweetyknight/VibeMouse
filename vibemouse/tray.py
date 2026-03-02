"""System tray icon for VibeMouse on Windows."""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pystray import Icon, MenuItem

# Guard: tray module is Windows-only
if sys.platform != "win32":
    raise ImportError("vibemouse.tray is only supported on Windows")

import winreg  # noqa: E402 (windows-only import after platform guard)

import pystray  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from vibemouse.app import VoiceMouseApp  # noqa: E402
from vibemouse.config import load_config  # noqa: E402

_APP_NAME = "VibeMouse"
_REGISTRY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_ICON_SIZE = 64


def _make_icon(color: str) -> Image.Image:
    """Generate a simple circular tray icon with the given fill color."""
    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse(
        [margin, margin, _ICON_SIZE - margin, _ICON_SIZE - margin],
        fill=color,
        outline="white",
        width=2,
    )
    return img


_COLOR_READY = "#22c55e"  # green
_COLOR_RECORDING = "#ef4444"  # red
_COLOR_STREAMING = "#3b82f6"  # blue (streaming recognition)
_COLOR_BUSY = "#f59e0b"  # amber (processing)

# Pre-generate icons once at import time to avoid repeated PIL allocations
# on every status change callback.
_ICON_READY = _make_icon(_COLOR_READY)
_ICON_RECORDING = _make_icon(_COLOR_RECORDING)
_ICON_STREAMING = _make_icon(_COLOR_STREAMING)


def _is_autostart_enabled() -> bool:
    """Check whether the autostart registry entry exists."""
    try:
        with winreg.OpenKey(_REGISTRY_KEY_HANDLE(), _REGISTRY_KEY, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, _APP_NAME)
            return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _REGISTRY_KEY_HANDLE() -> int:
    return winreg.HKEY_CURRENT_USER


def _set_autostart(enabled: bool) -> None:
    """Write or remove the autostart registry entry."""
    try:
        with winreg.OpenKey(
            _REGISTRY_KEY_HANDLE(), _REGISTRY_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            if enabled:
                exe_path = sys.executable
                # If running as a frozen exe, use that path directly
                if getattr(sys, "frozen", False):
                    exe_path = sys.executable
                else:
                    exe_path = f'"{sys.executable}" -m vibemouse'
                winreg.SetValueEx(key, _APP_NAME, 0, winreg.REG_SZ, exe_path)
            else:
                try:
                    winreg.DeleteValue(key, _APP_NAME)
                except FileNotFoundError:
                    pass
    except OSError as exc:
        print(f"Failed to update autostart registry: {exc}")


class VibeTray:
    """Manages the system tray icon and VoiceMouseApp lifecycle."""

    def __init__(self) -> None:
        self._app: VoiceMouseApp | None = None
        self._app_thread: threading.Thread | None = None
        self._icon: Icon | None = None
        self._current_state: str = "ready"

    def run(self) -> None:
        """Start the tray icon (blocking — runs the pystray message loop)."""
        self._icon = pystray.Icon(
            _APP_NAME,
            icon=_ICON_READY,
            title=f"{_APP_NAME} — Ready",
            menu=self._build_menu(),
        )
        # Start the app in a background thread
        self._app_thread = threading.Thread(target=self._run_app, daemon=True)
        self._app_thread.start()
        # Blocking: runs Win32 message loop
        self._icon.run()

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                lambda _text: self._status_label(),
                action=None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Auto-start with Windows",
                action=self._toggle_autostart,
                checked=lambda _item: _is_autostart_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", action=self._on_exit),
        )

    def _status_label(self) -> str:
        labels = {
            "ready": "Ready",
            "recording_start": "Recording...",
            "recording_stop": "Processing...",
            "streaming": "Streaming...",
            "transcribed": "Ready",
            "error": "Error — check logs",
        }
        return labels.get(self._current_state, "Ready")

    def _run_app(self) -> None:
        """Run VoiceMouseApp in a background thread."""
        try:
            config = load_config()
            self._app = VoiceMouseApp(config, on_status_change=self._on_app_status)
            self._app.run()
        except Exception as exc:
            print(f"VoiceMouseApp crashed: {exc}")
            if self._icon is not None:
                self._icon.stop()

    def _on_app_status(self, event: str, detail: str) -> None:
        """Callback from VoiceMouseApp — update tray icon appearance."""
        self._current_state = event
        if self._icon is None:
            return

        if event == "recording_start":
            self._icon.icon = _ICON_RECORDING
            self._icon.title = f"{_APP_NAME} — Recording"
        elif event == "streaming":
            self._icon.icon = _ICON_STREAMING
            self._icon.title = f"{_APP_NAME} — Streaming"
        elif event in ("ready", "transcribed", "recording_stop"):
            self._icon.icon = _ICON_READY
            self._icon.title = f"{_APP_NAME} — Ready"
        elif event == "error":
            self._icon.icon = _ICON_READY
            self._icon.title = f"{_APP_NAME} — Error"

        # Force menu refresh so status label updates
        self._icon.update_menu()

    def _toggle_autostart(self, icon: Icon, item: MenuItem) -> None:
        _set_autostart(not _is_autostart_enabled())

    def _on_exit(self, icon: Icon, item: MenuItem) -> None:
        if self._app is not None:
            self._app.request_stop()
        icon.stop()

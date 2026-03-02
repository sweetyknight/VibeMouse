from __future__ import annotations

import os
import sys

from vibemouse.app import VoiceMouseApp
from vibemouse.config import load_config

_MUTEX_NAME = "VibeMouse_SingleInstance"


def _use_tray() -> bool:
    """Decide whether to launch in tray mode."""
    if sys.platform != "win32":
        return False
    if os.getenv("VIBEMOUSE_NO_TRAY", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return True


def _acquire_single_instance_lock() -> bool:
    """Try to acquire a system-wide named mutex.

    Returns True if this is the only instance, False if another is already
    running.  The mutex handle is intentionally kept alive (not closed) for
    the lifetime of the process.
    """
    import ctypes

    _ERROR_ALREADY_EXISTS = 183
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, _MUTEX_NAME)  # type: ignore[attr-defined]
    if ctypes.windll.kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:  # type: ignore[attr-defined]
        # Another instance already holds the mutex — close our duplicate
        # handle and signal the caller.
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return False
    # Store handle on the module to prevent GC from releasing it.
    _acquire_single_instance_lock._handle = handle  # type: ignore[attr-defined]
    return True


def main() -> None:
    if _use_tray():
        if not _acquire_single_instance_lock():
            print("VibeMouse is already running.")
            return

        from vibemouse.tray import VibeTray

        tray = VibeTray()
        tray.run()
    else:
        config = load_config()
        app = VoiceMouseApp(config)
        app.run()


if __name__ == "__main__":
    main()

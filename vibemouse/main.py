from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from vibemouse.app import VoiceMouseApp
from vibemouse.config import load_config

_MUTEX_NAME = "VibeMouse_SingleInstance"

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging.

    When running as a frozen (PyInstaller) windowed app, stdout/stderr are
    not visible.  Route all log output (and captured prints) to a file so
    that errors can be diagnosed after the fact.
    """
    log_dir = Path.home() / ".cache" / "vibemouse"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vibemouse.log"

    handlers: list[logging.Handler] = [
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]

    # Keep console output when a terminal is available.
    if not getattr(sys, "frozen", False) or sys.stdout is not None:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    # Redirect print() to the log file when running as frozen windowed app
    # so that existing print()-based diagnostics are not lost.
    if getattr(sys, "frozen", False):
        try:
            log_fh = open(log_file, "a", encoding="utf-8")  # noqa: SIM115
            if sys.stdout is None or getattr(sys.stdout, "write", None) is None:
                sys.stdout = log_fh
            if sys.stderr is None or getattr(sys.stderr, "write", None) is None:
                sys.stderr = log_fh
        except Exception:
            pass


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
    _setup_logging()

    try:
        _main_inner()
    except Exception:
        logger.exception("VibeMouse fatal error")
        raise


def _main_inner() -> None:
    if _use_tray():
        if not _acquire_single_instance_lock():
            logger.info("VibeMouse is already running.")
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

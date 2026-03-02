from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

_IS_WINDOWS: bool = sys.platform == "win32"


class _KeyboardController(Protocol):
    def press(self, key: object) -> None: ...

    def release(self, key: object) -> None: ...

    def type(self, text: str) -> None: ...


# ---------------------------------------------------------------------------
# Win32 direct Unicode input — bypasses the IME entirely
# ---------------------------------------------------------------------------

if _IS_WINDOWS:
    import ctypes
    import ctypes.wintypes as _wt

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", _wt.WORD),
            ("wScan", _wt.WORD),
            ("dwFlags", _wt.DWORD),
            ("time", _wt.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    # MOUSEINPUT is the largest union member — needed for correct sizeof(INPUT).
    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", _wt.DWORD),
            ("dwFlags", _wt.DWORD),
            ("time", _wt.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _InputUnion(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", _wt.DWORD), ("u", _InputUnion)]

    _SIZEOF_INPUT: int = ctypes.sizeof(_INPUT)
    _SendInput = ctypes.windll.user32.SendInput  # type: ignore[attr-defined]

    # Virtual-key code for the Backspace key.
    _VK_BACK: int = 0x08

    def _send_unicode_string(text: str) -> None:
        """Type *text* via Win32 SendInput with KEYEVENTF_UNICODE.

        Each character is sent as a Unicode scan-code, completely bypassing
        the input method editor (IME) so that already-transcribed text is
        injected verbatim without opening a composition window.
        """
        _KBD: int = 1       # INPUT_KEYBOARD
        _UNI: int = 0x0004  # KEYEVENTF_UNICODE
        _UP: int = 0x0002   # KEYEVENTF_KEYUP

        scan_codes: list[int] = []
        for ch in text:
            cp = ord(ch)
            if cp <= 0xFFFF:
                scan_codes.append(cp)
            else:
                # UTF-16 surrogate pair for supplementary characters (e.g. emoji)
                scan_codes.append(0xD800 + ((cp - 0x10000) >> 10))
                scan_codes.append(0xDC00 + ((cp - 0x10000) & 0x3FF))

        if not scan_codes:
            return

        n = len(scan_codes) * 2  # key-down + key-up per code
        arr = (_INPUT * n)()
        idx = 0
        for sc in scan_codes:
            arr[idx].type = _KBD
            arr[idx].u.ki.wScan = sc
            arr[idx].u.ki.dwFlags = _UNI
            idx += 1
            arr[idx].type = _KBD
            arr[idx].u.ki.wScan = sc
            arr[idx].u.ki.dwFlags = _UNI | _UP
            idx += 1

        _SendInput(n, arr, _SIZEOF_INPUT)

    def _send_backspaces(count: int) -> None:
        """Send *count* Backspace key events in a single SendInput call.

        Much faster than pressing/releasing one key at a time through pynput
        because all events are injected in one kernel transition.
        """
        if count <= 0:
            return

        _KBD: int = 1       # INPUT_KEYBOARD
        _UP: int = 0x0002   # KEYEVENTF_KEYUP

        n = count * 2  # key-down + key-up per backspace
        arr = (_INPUT * n)()
        idx = 0
        for _ in range(count):
            arr[idx].type = _KBD
            arr[idx].u.ki.wVk = _VK_BACK
            idx += 1
            arr[idx].type = _KBD
            arr[idx].u.ki.wVk = _VK_BACK
            arr[idx].u.ki.dwFlags = _UP
            idx += 1

        _SendInput(n, arr, _SIZEOF_INPUT)


class StreamingTextOutput:
    """Incrementally types streaming recognition results with correction support.

    Uses a diff algorithm to compute the minimal set of backspaces and new
    keystrokes required to update the on-screen text when the recogniser
    revises its hypothesis.
    """

    def __init__(
        self,
        keyboard: _KeyboardController,
        backspace_key: object,
        *,
        keystroke_delay_s: float = 0.01,
        type_fn: Callable[[str], None] | None = None,
        backspace_fn: Callable[[int], None] | None = None,
    ) -> None:
        self._kb = keyboard
        self._backspace_key = backspace_key
        self._delay_s = keystroke_delay_s
        self._type_fn: Callable[[str], None] = type_fn if type_fn is not None else keyboard.type
        self._backspace_fn: Callable[[int], None] | None = backspace_fn
        self._lock = threading.Lock()
        self._current_text: str = ""

    @property
    def current_text(self) -> str:
        with self._lock:
            return self._current_text

    def update(self, new_text: str) -> None:
        """Update displayed text — backspace changed chars, then type new ones."""
        with self._lock:
            old = self._current_text
            if new_text == old:
                return

            prefix_len = _common_prefix_length(old, new_text)
            backspace_count = len(old) - prefix_len
            suffix = new_text[prefix_len:]

            self._backspace_n(backspace_count)
            if suffix:
                self._type_text(suffix)

            self._current_text = new_text

    def finalize(self) -> str:
        """Return the final text and reset internal state.

        The text already visible on screen is left untouched.
        """
        with self._lock:
            result = self._current_text
            self._current_text = ""
            return result

    def cancel(self) -> None:
        """Erase all typed text from screen and reset state."""
        with self._lock:
            if self._current_text:
                self._backspace_n(len(self._current_text))
            self._current_text = ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _backspace_n(self, count: int) -> None:
        if count <= 0:
            return
        if self._backspace_fn is not None:
            self._backspace_fn(count)
            return
        for _ in range(count):
            self._kb.press(self._backspace_key)
            self._kb.release(self._backspace_key)
            if self._delay_s > 0:
                time.sleep(self._delay_s)

    def _type_text(self, text: str) -> None:
        self._type_fn(text)
        if self._delay_s > 0:
            time.sleep(self._delay_s)


def _common_prefix_length(a: str, b: str) -> int:
    """Return the length of the longest common prefix of *a* and *b*."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit

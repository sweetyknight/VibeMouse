from __future__ import annotations

import importlib
import subprocess
from typing import Protocol, cast

import pyperclip


class TextOutput:
    def __init__(self) -> None:
        try:
            keyboard_module = importlib.import_module("pynput.keyboard")
        except Exception as error:
            raise RuntimeError(
                f"Failed to load keyboard control dependencies: {error}"
            ) from error

        controller_ctor = cast(
            _ControllerCtor,
            getattr(cast(object, keyboard_module), "Controller"),
        )
        key_holder = cast(
            _KeyHolder,
            getattr(cast(object, keyboard_module), "Key"),
        )
        self._kb: _KeyboardController = controller_ctor()
        self._enter_key: object = key_holder.enter

    def send_enter(self) -> None:
        self._kb.press(self._enter_key)
        self._kb.release(self._enter_key)

    def inject_or_clipboard(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return "empty"

        if self._is_text_input_focused():
            self._kb.type(normalized)
            return "typed"

        pyperclip.copy(normalized)
        return "clipboard"

    def _is_text_input_focused(self) -> bool:
        script = (
            "import gi\n"
            "gi.require_version('Atspi', '2.0')\n"
            "from gi.repository import Atspi\n"
            "obj = Atspi.get_desktop(0).get_focus()\n"
            "editable = False\n"
            "role = ''\n"
            "if obj is not None:\n"
            "    role = obj.get_role_name().lower()\n"
            "    attrs = obj.get_attributes() or []\n"
            "    for it in attrs:\n"
            "        s = str(it).lower()\n"
            "        if s == 'editable:true' or s.endswith(':editable:true'):\n"
            "            editable = True\n"
            "            break\n"
            "roles = {'text', 'entry', 'password text', 'terminal', 'paragraph', 'document text', 'document web'}\n"
            "print('1' if editable or role in roles else '0')\n"
        )

        proc = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "1"


class _KeyboardController(Protocol):
    def press(self, key: object) -> None: ...

    def release(self, key: object) -> None: ...

    def type(self, text: str) -> None: ...


class _ControllerCtor(Protocol):
    def __call__(self) -> _KeyboardController: ...


class _KeyHolder(Protocol):
    enter: object

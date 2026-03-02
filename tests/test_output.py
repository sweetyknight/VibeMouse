from __future__ import annotations

import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vibemouse.output import TextOutput


class _FakeKeyboardController:
    def __init__(self, *, fail_on_press: bool = False) -> None:
        self.events: list[tuple[str, object]] = []
        self._fail_on_press: bool = fail_on_press

    def press(self, key: object) -> None:
        if self._fail_on_press:
            raise RuntimeError("press failed")
        self.events.append(("press", key))

    def release(self, key: object) -> None:
        self.events.append(("release", key))

    def type(self, text: str) -> None:
        self.events.append(("type", text))


class TextOutputTests(unittest.TestCase):
    @staticmethod
    def _make_subject() -> TextOutput:
        return object.__new__(TextOutput)

    @staticmethod
    def _bind_keyboard(subject: TextOutput, keyboard: _FakeKeyboardController) -> None:
        setattr(subject, "_kb", keyboard)
        setattr(subject, "_ctrl_key", "CTRL")
        setattr(subject, "_shift_key", "SHIFT")
        setattr(subject, "_enter_key", "ENTER")
        setattr(subject, "_atspi", None)
        setattr(subject, "_hyprland_session", False)

    def test_send_enter_uses_enter_mode(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        with patch("vibemouse.output.time.sleep"):
            subject.send_enter(mode="enter")

        self.assertEqual(keyboard.events, [("press", "ENTER"), ("release", "ENTER")])

    def test_send_enter_supports_none_mode(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        subject.send_enter(mode="none")

        self.assertEqual(keyboard.events, [])

    def test_send_enter_prefers_atspi_when_available(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        class _FakeKeySynthType:
            PRESSRELEASE: object = object()

        class _FakeAtspi:
            KeySynthType: type[_FakeKeySynthType] = _FakeKeySynthType

            @staticmethod
            def generate_keyboard_event(
                keyval: int,
                keystring: str | None,
                synth_type: object,
            ) -> bool:
                _ = keyval
                _ = keystring
                _ = synth_type
                return True

        setattr(subject, "_atspi", _FakeAtspi())

        subject.send_enter(mode="enter")

        self.assertEqual(keyboard.events, [])

    def test_send_enter_prefers_hyprland_sendshortcut_when_available(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)
        setattr(subject, "_hyprland_session", True)

        with patch(
            "vibemouse.output.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok\n"),
        ) as run_mock:
            subject.send_enter(mode="enter")

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(keyboard.events, [])

    def test_send_enter_supports_ctrl_enter(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        with patch("vibemouse.output.time.sleep"):
            subject.send_enter(mode="ctrl_enter")

        self.assertEqual(
            keyboard.events,
            [
                ("press", "CTRL"),
                ("press", "ENTER"),
                ("release", "ENTER"),
                ("release", "CTRL"),
            ],
        )

    def test_send_enter_supports_shift_enter(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        with patch("vibemouse.output.time.sleep"):
            subject.send_enter(mode="shift_enter")

        self.assertEqual(
            keyboard.events,
            [
                ("press", "SHIFT"),
                ("press", "ENTER"),
                ("release", "ENTER"),
                ("release", "SHIFT"),
            ],
        )

    def test_send_enter_rejects_unknown_mode(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        with self.assertRaisesRegex(ValueError, "Unsupported enter mode"):
            subject.send_enter(mode="meta_enter")

    def test_keyboard_property_exposes_controller(self) -> None:
        subject = self._make_subject()
        keyboard = _FakeKeyboardController()
        self._bind_keyboard(subject, keyboard)

        self.assertIs(subject.keyboard, keyboard)

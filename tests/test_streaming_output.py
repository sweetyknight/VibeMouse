from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from vibemouse.streaming_output import StreamingTextOutput, _common_prefix_length


class _FakeKeyboardController:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def press(self, key: object) -> None:
        self.events.append(("press", key))

    def release(self, key: object) -> None:
        self.events.append(("release", key))

    def type(self, text: str) -> None:
        self.events.append(("type", text))


_BS = "BACKSPACE"


class CommonPrefixLengthTests(unittest.TestCase):
    def test_identical_strings(self) -> None:
        self.assertEqual(_common_prefix_length("abc", "abc"), 3)

    def test_no_common_prefix(self) -> None:
        self.assertEqual(_common_prefix_length("abc", "xyz"), 0)

    def test_partial_common_prefix(self) -> None:
        self.assertEqual(_common_prefix_length("abc", "abd"), 2)

    def test_empty_strings(self) -> None:
        self.assertEqual(_common_prefix_length("", ""), 0)
        self.assertEqual(_common_prefix_length("", "abc"), 0)
        self.assertEqual(_common_prefix_length("abc", ""), 0)

    def test_one_is_prefix_of_other(self) -> None:
        self.assertEqual(_common_prefix_length("ab", "abcd"), 2)
        self.assertEqual(_common_prefix_length("abcd", "ab"), 2)


@patch("vibemouse.streaming_output._IS_WINDOWS", False)
class StreamingTextOutputTests(unittest.TestCase):
    def _make(self) -> tuple[StreamingTextOutput, _FakeKeyboardController]:
        kb = _FakeKeyboardController()
        out = StreamingTextOutput(kb, backspace_key=_BS, keystroke_delay_s=0)
        return out, kb

    def test_initial_update_types_full_text(self) -> None:
        out, kb = self._make()
        out.update("hello")
        typed = "".join(ev[1] for ev in kb.events if ev[0] == "type")
        self.assertEqual(typed, "hello")

    def test_appending_types_only_suffix(self) -> None:
        out, kb = self._make()
        out.update("hel")
        kb.events.clear()

        out.update("hello")
        typed = "".join(ev[1] for ev in kb.events if ev[0] == "type")
        backspaces = [ev for ev in kb.events if ev == ("press", _BS)]
        self.assertEqual(typed, "lo")
        self.assertEqual(len(backspaces), 0)

    def test_correction_backspaces_and_retypes(self) -> None:
        out, kb = self._make()
        out.update("hello war")
        kb.events.clear()

        # Common prefix = "hello w" (7 chars)
        # Backspace "ar" (2 chars), then type "orld"
        out.update("hello world")
        backspaces = [ev for ev in kb.events if ev == ("press", _BS)]
        typed = "".join(ev[1] for ev in kb.events if ev[0] == "type")
        self.assertEqual(len(backspaces), 2)
        self.assertEqual(typed, "orld")

    def test_duplicate_update_is_noop(self) -> None:
        out, kb = self._make()
        out.update("hello")
        kb.events.clear()

        out.update("hello")
        self.assertEqual(kb.events, [])

    def test_finalize_returns_text_and_resets(self) -> None:
        out, kb = self._make()
        out.update("hello world")

        result = out.finalize()
        self.assertEqual(result, "hello world")
        self.assertEqual(out.current_text, "")

    def test_cancel_backspaces_all_text(self) -> None:
        out, kb = self._make()
        out.update("hello")
        kb.events.clear()

        out.cancel()
        backspaces = [ev for ev in kb.events if ev == ("press", _BS)]
        self.assertEqual(len(backspaces), 5)
        self.assertEqual(out.current_text, "")

    def test_cancel_on_empty_is_noop(self) -> None:
        out, kb = self._make()
        out.cancel()
        self.assertEqual(kb.events, [])

    def test_current_text_property(self) -> None:
        out, _kb = self._make()
        self.assertEqual(out.current_text, "")
        out.update("abc")
        self.assertEqual(out.current_text, "abc")

    def test_complete_replacement(self) -> None:
        out, kb = self._make()
        out.update("abc")
        kb.events.clear()

        out.update("xyz")
        backspaces = [ev for ev in kb.events if ev == ("press", _BS)]
        typed = "".join(ev[1] for ev in kb.events if ev[0] == "type")
        self.assertEqual(len(backspaces), 3)
        self.assertEqual(typed, "xyz")

    def test_thread_safety(self) -> None:
        out, _kb = self._make()
        errors: list[Exception] = []

        def worker(text: str) -> None:
            try:
                for i in range(50):
                    out.update(f"{text}-{i}")
            except Exception as err:
                errors.append(err)

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertIsInstance(out.current_text, str)

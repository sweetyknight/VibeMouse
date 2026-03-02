from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from vibemouse.config import AppConfig
from vibemouse.streaming_output import StreamingTextOutput
from vibemouse.transcriber import StreamingResult


def _default_config(**overrides: object) -> AppConfig:
    defaults = dict(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        pre_buffer_seconds=0.0,
        sherpa_model_dir=Path("/tmp/models"),
        sherpa_num_threads=2,
        asr_backend="vad_offline",
        vad_min_silence_duration=0.25,
        vad_min_speech_duration=0.25,
        vad_threshold=0.5,
        offline_model_name="sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16",
        recording_mode="hold",
        button_debounce_ms=150,
        front_button="x1",
        rear_button="x2",
        enter_mode="enter",
        auto_paste=True,
    )
    defaults.update(overrides)
    return AppConfig(**defaults)  # type: ignore[arg-type]


class _FakeKeyboardController:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def press(self, key: object) -> None:
        self.events.append(("press", key))

    def release(self, key: object) -> None:
        self.events.append(("release", key))

    def type(self, text: str) -> None:
        self.events.append(("type", text))


class _FakeStreamingSession:
    """Minimal fake for StreamingSession."""

    def __init__(self, *, stop_result: str = "", stop_error: Exception | None = None) -> None:
        self._stop_result = stop_result
        self._stop_error = stop_error
        self.stopped = False

    def feed_audio(self, chunk: object) -> None:
        pass

    def stop(self) -> str:
        self.stopped = True
        if self._stop_error is not None:
            raise self._stop_error
        return self._stop_result


class StartStreamingCleanupTests(unittest.TestCase):
    """Verify _start_streaming cleans up session when recorder.start fails."""

    def _make_app(self) -> object:
        """Build a VoiceMouseApp with all heavy dependencies stubbed out."""
        from vibemouse.app import VoiceMouseApp

        app = object.__new__(VoiceMouseApp)
        app._config = _default_config()
        app._recorder = MagicMock()
        app._recorder.is_recording = False
        app._transcriber = MagicMock()
        app._output = MagicMock()
        app._on_status_change = None
        app._stop_event = threading.Event()
        app._workers_lock = threading.Lock()
        app._workers = set()
        app._session = None
        app._recording_mode = "hold"
        app._mode_lock = threading.Lock()
        app._finalizing = threading.Event()

        kb = _FakeKeyboardController()
        app._streaming_output = StreamingTextOutput(kb, backspace_key="BS", keystroke_delay_s=0)

        return app

    def test_session_stopped_when_recorder_start_fails(self) -> None:
        from vibemouse.app import VoiceMouseApp

        app = self._make_app()

        fake_session = _FakeStreamingSession()
        app._transcriber.start_session.return_value = fake_session
        app._recorder.start.side_effect = RuntimeError("No microphone")

        app._start_streaming()

        self.assertTrue(fake_session.stopped)
        self.assertIsNone(app._session)

    def test_session_set_only_after_successful_recorder_start(self) -> None:
        from vibemouse.app import VoiceMouseApp

        app = self._make_app()

        fake_session = _FakeStreamingSession()
        app._transcriber.start_session.return_value = fake_session

        app._start_streaming()

        self.assertIs(app._session, fake_session)

    def test_streaming_output_reset_before_new_session(self) -> None:
        """A leftover streaming output from a previous session must be cleared."""
        from vibemouse.app import VoiceMouseApp

        app = self._make_app()
        # Simulate leftover state from a previous session
        app._streaming_output.update("leftover text")

        fake_session = _FakeStreamingSession()
        app._transcriber.start_session.return_value = fake_session

        app._start_streaming()

        self.assertEqual(app._streaming_output.current_text, "")


class FinalizeStreamingTests(unittest.TestCase):
    """Verify _finalize_streaming always resets streaming output."""

    def _make_app(self) -> object:
        from vibemouse.app import VoiceMouseApp

        app = object.__new__(VoiceMouseApp)
        app._config = _default_config()
        app._recorder = MagicMock()
        app._on_status_change = None
        app._workers_lock = threading.Lock()
        app._workers = set()
        app._finalizing = threading.Event()

        kb = _FakeKeyboardController()
        app._streaming_output = StreamingTextOutput(kb, backspace_key="BS", keystroke_delay_s=0)

        return app

    @patch("vibemouse.app.time.sleep")
    def test_streaming_output_finalized_on_success(self, _mock_sleep: object) -> None:
        from vibemouse.app import VoiceMouseApp

        app = self._make_app()
        app._streaming_output.update("recognized text")

        session = _FakeStreamingSession(stop_result="recognized text")
        app._finalize_streaming(session)

        self.assertEqual(app._streaming_output.current_text, "")
        app._recorder.cancel.assert_called_once()

    @patch("vibemouse.app.time.sleep")
    def test_streaming_output_finalized_on_error(self, _mock_sleep: object) -> None:
        from vibemouse.app import VoiceMouseApp

        app = self._make_app()
        app._streaming_output.update("partial text")

        session = _FakeStreamingSession(stop_error=RuntimeError("decode failed"))
        app._finalize_streaming(session)

        self.assertEqual(app._streaming_output.current_text, "")
        app._recorder.cancel.assert_called_once()


class FrontButtonPressAndHoldTests(unittest.TestCase):
    """Verify press-and-hold behavior for the front button."""

    def _make_app(self) -> object:
        from vibemouse.app import VoiceMouseApp

        app = object.__new__(VoiceMouseApp)
        app._config = _default_config()
        app._recorder = MagicMock()
        app._recorder.is_recording = False
        app._transcriber = MagicMock()
        app._output = MagicMock()
        app._on_status_change = None
        app._stop_event = threading.Event()
        app._workers_lock = threading.Lock()
        app._workers = set()
        app._session = None
        app._recording_mode = "hold"
        app._mode_lock = threading.Lock()
        app._finalizing = threading.Event()

        kb = _FakeKeyboardController()
        app._streaming_output = StreamingTextOutput(kb, backspace_key="BS", keystroke_delay_s=0)

        return app

    def test_press_starts_recording_when_not_recording(self) -> None:
        app = self._make_app()
        fake_session = _FakeStreamingSession()
        app._transcriber.start_session.return_value = fake_session
        app._recorder.is_recording = False

        app._on_front_press()

        app._recorder.start.assert_called_once()
        self.assertIs(app._session, fake_session)

    def test_press_is_noop_when_already_recording(self) -> None:
        app = self._make_app()
        app._recorder.is_recording = True

        app._on_front_press()

        app._transcriber.start_session.assert_not_called()

    @patch("vibemouse.app.time.sleep")
    def test_release_stops_recording_when_recording(self, _mock_sleep: object) -> None:
        app = self._make_app()
        app._recorder.is_recording = True
        fake_session = _FakeStreamingSession()
        app._session = fake_session

        app._on_front_release()

        # Session is cleared immediately on release.
        self.assertIsNone(app._session)

        # cancel() is called in the worker thread — wait for it.
        with app._workers_lock:
            workers = list(app._workers)
        for worker in workers:
            worker.join(timeout=5)

        app._recorder.cancel.assert_called_once()

    def test_release_is_noop_when_not_recording(self) -> None:
        app = self._make_app()
        app._recorder.is_recording = False

        app._on_front_release()

        app._recorder.cancel.assert_not_called()


class ToggleModeTests(unittest.TestCase):
    """Verify toggle recording mode behavior."""

    def _make_app(self) -> object:
        from vibemouse.app import VoiceMouseApp

        app = object.__new__(VoiceMouseApp)
        app._config = _default_config(recording_mode="toggle")
        app._recorder = MagicMock()
        app._recorder.is_recording = False
        app._transcriber = MagicMock()
        app._output = MagicMock()
        app._on_status_change = None
        app._stop_event = threading.Event()
        app._workers_lock = threading.Lock()
        app._workers = set()
        app._session = None
        app._recording_mode = "toggle"
        app._mode_lock = threading.Lock()
        app._finalizing = threading.Event()

        kb = _FakeKeyboardController()
        app._streaming_output = StreamingTextOutput(kb, backspace_key="BS", keystroke_delay_s=0)

        return app

    def test_first_press_starts_recording(self) -> None:
        app = self._make_app()
        fake_session = _FakeStreamingSession()
        app._transcriber.start_session.return_value = fake_session
        app._recorder.is_recording = False

        app._on_front_press()

        app._recorder.start.assert_called_once()

    @patch("vibemouse.app.time.sleep")
    def test_second_press_stops_recording(self, _mock_sleep: object) -> None:
        app = self._make_app()
        app._recorder.is_recording = True
        fake_session = _FakeStreamingSession()
        app._session = fake_session

        app._on_front_press()

        self.assertIsNone(app._session)

    def test_release_is_ignored(self) -> None:
        app = self._make_app()
        app._recorder.is_recording = True

        app._on_front_release()

        app._recorder.cancel.assert_not_called()

    def test_press_ignored_during_finalization(self) -> None:
        app = self._make_app()
        app._finalizing.set()
        app._recorder.is_recording = False

        app._on_front_press()

        app._transcriber.start_session.assert_not_called()


class RecordingModePropertyTests(unittest.TestCase):
    """Verify recording_mode getter/setter."""

    def _make_app(self) -> object:
        from vibemouse.app import VoiceMouseApp

        app = object.__new__(VoiceMouseApp)
        app._config = _default_config()
        app._recording_mode = "hold"
        app._mode_lock = threading.Lock()
        app._on_status_change = MagicMock()
        return app

    def test_getter_returns_current_mode(self) -> None:
        app = self._make_app()
        self.assertEqual(app.recording_mode, "hold")

    def test_setter_changes_mode(self) -> None:
        app = self._make_app()
        app.set_recording_mode("toggle")
        self.assertEqual(app.recording_mode, "toggle")

    def test_setter_rejects_invalid_mode(self) -> None:
        app = self._make_app()
        with self.assertRaises(ValueError):
            app.set_recording_mode("push")

    def test_setter_fires_status_callback(self) -> None:
        app = self._make_app()
        app.set_recording_mode("toggle")
        app._on_status_change.assert_called_once_with("mode_change", "toggle")

    def test_setter_noop_when_same_mode(self) -> None:
        app = self._make_app()
        app.set_recording_mode("hold")
        app._on_status_change.assert_not_called()

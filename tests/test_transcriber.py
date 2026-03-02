from __future__ import annotations

import queue
import threading
import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from vibemouse.transcriber import StreamingResult, StreamingSession

AudioFrame = NDArray[np.float32]


class _FakeOnlineStream:
    def __init__(self) -> None:
        self.waveforms: list[AudioFrame] = []

    def accept_waveform(self, sample_rate: int, waveform: AudioFrame) -> None:
        self.waveforms.append(waveform)


class _FakeOnlineRecognizer:
    """Controllable fake recognizer for unit tests."""

    def __init__(self) -> None:
        self._results: list[str] = []
        self._result_index: int = 0
        self._is_endpoint_flags: list[bool] = []
        self._endpoint_index: int = 0
        self._ready_counts: list[int] = []
        self._ready_index: int = 0
        self._ready_remaining: int = 0
        self.reset_count: int = 0

    def set_results(self, results: list[str]) -> None:
        self._results = results
        self._result_index = 0

    def set_ready_counts(self, counts: list[int]) -> None:
        self._ready_counts = counts
        self._ready_index = 0
        self._ready_remaining = 0

    def set_endpoint_flags(self, flags: list[bool]) -> None:
        self._is_endpoint_flags = flags
        self._endpoint_index = 0

    def create_stream(self) -> _FakeOnlineStream:
        return _FakeOnlineStream()

    def is_ready(self, stream: object) -> bool:
        if self._ready_remaining > 0:
            self._ready_remaining -= 1
            return True
        if self._ready_index < len(self._ready_counts):
            count = self._ready_counts[self._ready_index]
            self._ready_index += 1
            self._ready_remaining = count - 1 if count > 0 else 0
            return count > 0
        return False

    def decode_stream(self, stream: object) -> None:
        pass

    def get_result(self, stream: object) -> str:
        if self._result_index < len(self._results):
            result = self._results[self._result_index]
            self._result_index += 1
            return result
        return ""

    def is_endpoint(self, stream: object) -> bool:
        if self._endpoint_index < len(self._is_endpoint_flags):
            flag = self._is_endpoint_flags[self._endpoint_index]
            self._endpoint_index += 1
            return flag
        return False

    def reset(self, stream: object) -> None:
        self.reset_count += 1


class StreamingSessionTests(unittest.TestCase):
    def test_callback_exception_does_not_crash_decode_loop(self) -> None:
        """on_result exceptions must not prevent final text from being set."""
        recognizer = _FakeOnlineRecognizer()
        recognizer.set_ready_counts([1])
        recognizer.set_results(["hello"])
        recognizer.set_endpoint_flags([False])

        call_count = 0

        def bad_callback(result: StreamingResult) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        session = StreamingSession(
            recognizer=cast(object, recognizer),  # type: ignore[arg-type]
            sample_rate=16000,
            on_result=bad_callback,
        )

        chunk = np.zeros(1600, dtype=np.float32)
        session.feed_audio(chunk.reshape(-1, 1))

        import time
        time.sleep(0.3)

        final = session.stop()
        self.assertEqual(final, "hello")
        self.assertGreaterEqual(call_count, 1)

    def test_empty_audio_returns_empty_text(self) -> None:
        recognizer = _FakeOnlineRecognizer()

        results: list[StreamingResult] = []
        session = StreamingSession(
            recognizer=cast(object, recognizer),  # type: ignore[arg-type]
            sample_rate=16000,
            on_result=results.append,
        )

        final = session.stop()
        self.assertEqual(final, "")
        self.assertEqual(results, [])

    def test_endpoint_resets_and_confirms_segment(self) -> None:
        recognizer = _FakeOnlineRecognizer()
        recognizer.set_ready_counts([1, 1])
        recognizer.set_results(["first", "second"])
        recognizer.set_endpoint_flags([True, False])

        results: list[StreamingResult] = []
        session = StreamingSession(
            recognizer=cast(object, recognizer),  # type: ignore[arg-type]
            sample_rate=16000,
            on_result=results.append,
        )

        chunk = np.zeros(1600, dtype=np.float32)
        session.feed_audio(chunk.reshape(-1, 1))

        import time
        time.sleep(0.2)

        session.feed_audio(chunk.reshape(-1, 1))
        time.sleep(0.2)

        final = session.stop()
        self.assertEqual(final, "firstsecond")
        self.assertEqual(recognizer.reset_count, 1)

    def test_feed_audio_after_stop_is_ignored(self) -> None:
        recognizer = _FakeOnlineRecognizer()

        session = StreamingSession(
            recognizer=cast(object, recognizer),  # type: ignore[arg-type]
            sample_rate=16000,
            on_result=lambda r: None,
        )

        session.stop()

        # Should not raise
        chunk = np.zeros(1600, dtype=np.float32)
        session.feed_audio(chunk.reshape(-1, 1))

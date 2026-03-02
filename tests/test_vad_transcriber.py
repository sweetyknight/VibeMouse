from __future__ import annotations

import threading
import time
import unittest

import numpy as np
from numpy.typing import NDArray

from vibemouse.transcriber import StreamingResult
from vibemouse.vad_transcriber import VadOfflineSession


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSpeechSegment:
    def __init__(self, samples: NDArray[np.float32]) -> None:
        self.samples = samples


class _FakeVad:
    """Controllable fake VAD that emits segments on demand."""

    def __init__(self) -> None:
        self._segments: list[_FakeSpeechSegment] = []
        self._accept_count: int = 0
        # Map of accept_waveform call index → segment to emit.
        self._schedule: dict[int, _FakeSpeechSegment] = {}
        self._flush_segments: list[_FakeSpeechSegment] = []

    def schedule_segment(
        self, at_accept_count: int, segment: _FakeSpeechSegment
    ) -> None:
        self._schedule[at_accept_count] = segment

    def schedule_flush_segment(self, segment: _FakeSpeechSegment) -> None:
        self._flush_segments.append(segment)

    def accept_waveform(self, samples: NDArray[np.float32]) -> None:
        self._accept_count += 1
        if self._accept_count in self._schedule:
            self._segments.append(self._schedule[self._accept_count])

    def empty(self) -> bool:
        return len(self._segments) == 0

    @property
    def front(self) -> _FakeSpeechSegment:
        return self._segments[0]

    def pop(self) -> None:
        self._segments.pop(0)

    def flush(self) -> None:
        self._segments.extend(self._flush_segments)
        self._flush_segments.clear()


class _FakeOfflineResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeOfflineStream:
    def __init__(self, text: str) -> None:
        self._result = _FakeOfflineResult(text)

    def accept_waveform(self, sample_rate: int, samples: NDArray[np.float32]) -> None:
        pass

    @property
    def result(self) -> _FakeOfflineResult:
        return self._result


class _FakeOfflineRecognizer:
    def __init__(self, results: list[str]) -> None:
        self._results = list(results)
        self._index = 0

    def create_stream(self) -> _FakeOfflineStream:
        text = self._results[self._index] if self._index < len(self._results) else ""
        self._index += 1
        return _FakeOfflineStream(text)

    def decode_stream(self, stream: _FakeOfflineStream) -> None:
        pass


# ---------------------------------------------------------------------------
# Helper to build a session with fakes
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_WINDOW = 512
# Must exceed _MIN_SEGMENT_SAMPLES (8000) in vad_transcriber.py.
_SEGMENT_SAMPLES = 10000


def _make_audio(num_windows: int) -> NDArray[np.float32]:
    """Create a float32 audio chunk that is exactly *num_windows* VAD windows."""
    return np.zeros(num_windows * _WINDOW, dtype=np.float32)


def _make_session(
    vad: _FakeVad,
    recognizer: _FakeOfflineRecognizer,
    results: list[StreamingResult] | None = None,
) -> VadOfflineSession:
    collected = results if results is not None else []

    def on_result(r: StreamingResult) -> None:
        collected.append(r)

    return VadOfflineSession(
        vad=vad,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
        sample_rate=_SAMPLE_RATE,
        on_result=on_result,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SingleSegmentTests(unittest.TestCase):
    def test_single_segment_recognized(self) -> None:
        vad = _FakeVad()
        segment = _FakeSpeechSegment(np.zeros(_SEGMENT_SAMPLES, dtype=np.float32))
        vad.schedule_segment(at_accept_count=2, segment=segment)

        recognizer = _FakeOfflineRecognizer(["hello world"])
        results: list[StreamingResult] = []
        session = _make_session(vad, recognizer, results)

        session.feed_audio(_make_audio(3))
        time.sleep(0.3)
        final = session.stop()

        self.assertEqual(final, "hello world")

    def test_short_segment_ignored(self) -> None:
        vad = _FakeVad()
        segment = _FakeSpeechSegment(np.zeros(100, dtype=np.float32))
        vad.schedule_segment(at_accept_count=1, segment=segment)

        recognizer = _FakeOfflineRecognizer(["  "])  # whitespace-only
        results: list[StreamingResult] = []
        session = _make_session(vad, recognizer, results)

        session.feed_audio(_make_audio(2))
        time.sleep(0.3)
        final = session.stop()

        self.assertEqual(final, "")


class MultiSegmentTests(unittest.TestCase):
    def test_multiple_segments_concatenated(self) -> None:
        vad = _FakeVad()
        seg1 = _FakeSpeechSegment(np.zeros(_SEGMENT_SAMPLES, dtype=np.float32))
        seg2 = _FakeSpeechSegment(np.zeros(_SEGMENT_SAMPLES, dtype=np.float32))
        vad.schedule_segment(at_accept_count=1, segment=seg1)
        vad.schedule_segment(at_accept_count=3, segment=seg2)

        recognizer = _FakeOfflineRecognizer(["hello", "world"])
        results: list[StreamingResult] = []
        session = _make_session(vad, recognizer, results)

        session.feed_audio(_make_audio(4))
        time.sleep(0.3)
        final = session.stop()

        self.assertEqual(final, "helloworld")


class FlushOnStopTests(unittest.TestCase):
    def test_flush_produces_final_segment(self) -> None:
        vad = _FakeVad()
        flush_seg = _FakeSpeechSegment(np.zeros(_SEGMENT_SAMPLES, dtype=np.float32))
        vad.schedule_flush_segment(flush_seg)

        recognizer = _FakeOfflineRecognizer(["flushed text"])
        results: list[StreamingResult] = []
        session = _make_session(vad, recognizer, results)

        # Feed audio that does not trigger a segment, then stop.
        session.feed_audio(_make_audio(1))
        time.sleep(0.2)
        final = session.stop()

        self.assertEqual(final, "flushed text")


class NoAudioTests(unittest.TestCase):
    def test_no_audio_returns_empty(self) -> None:
        vad = _FakeVad()
        recognizer = _FakeOfflineRecognizer([])
        session = _make_session(vad, recognizer)

        final = session.stop()

        self.assertEqual(final, "")


class CallbackErrorTests(unittest.TestCase):
    def test_callback_exception_does_not_crash(self) -> None:
        vad = _FakeVad()
        segment = _FakeSpeechSegment(np.zeros(_SEGMENT_SAMPLES, dtype=np.float32))
        vad.schedule_segment(at_accept_count=1, segment=segment)

        recognizer = _FakeOfflineRecognizer(["text"])

        def bad_callback(r: StreamingResult) -> None:
            raise RuntimeError("boom")

        session = VadOfflineSession(
            vad=vad,  # type: ignore[arg-type]
            recognizer=recognizer,  # type: ignore[arg-type]
            sample_rate=_SAMPLE_RATE,
            on_result=bad_callback,
        )

        session.feed_audio(_make_audio(2))
        time.sleep(0.3)
        final = session.stop()

        # Session should still produce a result despite callback errors.
        self.assertEqual(final, "text")


class FeedAfterStopTests(unittest.TestCase):
    def test_feed_audio_after_stop_is_ignored(self) -> None:
        vad = _FakeVad()
        recognizer = _FakeOfflineRecognizer([])
        session = _make_session(vad, recognizer)

        session.stop()
        # Should not raise.
        session.feed_audio(_make_audio(1))

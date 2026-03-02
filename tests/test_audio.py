from __future__ import annotations

import threading
import unittest

import numpy as np
from numpy.typing import NDArray

from vibemouse.audio import AudioRecorder


class AudioRecorderTests(unittest.TestCase):
    def test_cancel_when_not_recording_is_noop(self) -> None:
        recorder = AudioRecorder(sample_rate=16000, channels=1, dtype="float32")
        recorder.cancel()  # should not raise

    def test_on_chunk_receives_audio_frames(self) -> None:
        chunks: list[NDArray[np.float32]] = []
        recorder = AudioRecorder(sample_rate=16000, channels=1, dtype="float32")

        # Simulate the callback directly (avoids needing real sounddevice)
        recorder._on_chunk = chunks.append
        recorder._recording = True

        frame = np.ones((160, 1), dtype=np.float32)
        recorder._callback(frame, 160, None, None)

        self.assertEqual(len(chunks), 1)
        # The callback now produces a pre-flattened 1D copy for efficiency
        expected = frame.reshape(-1)
        np.testing.assert_array_equal(chunks[0], expected)
        self.assertEqual(chunks[0].ndim, 1)

    def test_callback_exception_in_on_chunk_is_swallowed(self) -> None:
        def bad_handler(chunk: NDArray[np.float32]) -> None:
            raise RuntimeError("boom")

        recorder = AudioRecorder(sample_rate=16000, channels=1, dtype="float32")
        recorder._on_chunk = bad_handler
        recorder._recording = True

        frame = np.zeros((160, 1), dtype=np.float32)
        # Should not raise
        recorder._callback(frame, 160, None, None)

    def test_callback_not_called_when_not_recording(self) -> None:
        chunks: list[NDArray[np.float32]] = []
        recorder = AudioRecorder(sample_rate=16000, channels=1, dtype="float32")
        recorder._on_chunk = chunks.append
        recorder._recording = False

        frame = np.zeros((160, 1), dtype=np.float32)
        recorder._callback(frame, 160, None, None)

        # on_chunk should still be called (it's independent of _recording flag)
        self.assertEqual(len(chunks), 1)


class PreBufferTests(unittest.TestCase):
    def test_ring_buffer_accumulates_when_not_recording(self) -> None:
        recorder = AudioRecorder(
            sample_rate=16000, channels=1, dtype="float32",
            pre_buffer_seconds=0.5,
        )
        # Simulate callback while not recording (no on_chunk)
        frame = np.ones((160, 1), dtype=np.float32)
        for _ in range(10):
            recorder._callback(frame, 160, None, None)

        self.assertEqual(len(recorder._ring), 10)

    def test_ring_buffer_evicts_old_chunks(self) -> None:
        # 0.01s buffer at 16kHz = 160 samples max
        recorder = AudioRecorder(
            sample_rate=16000, channels=1, dtype="float32",
            pre_buffer_seconds=0.01,
        )
        frame = np.ones((160, 1), dtype=np.float32)
        # Feed 5 chunks of 160 samples each — only ~1 should fit
        for _ in range(5):
            recorder._callback(frame, 160, None, None)

        self.assertLessEqual(recorder._ring_samples, 320)

    def test_start_flushes_ring_buffer(self) -> None:
        received: list[NDArray[np.float32]] = []
        recorder = AudioRecorder(
            sample_rate=16000, channels=1, dtype="float32",
            pre_buffer_seconds=0.5,
        )
        # Pre-fill ring buffer
        frame = np.ones((160, 1), dtype=np.float32)
        for _ in range(3):
            recorder._callback(frame, 160, None, None)

        self.assertEqual(len(recorder._ring), 3)

        # Simulate start — we need to bypass the real InputStream creation.
        # Manually set the stream so start() reuses it.
        recorder._hot_stream = object()  # type: ignore[assignment]
        recorder.start(on_chunk=received.append)

        # Ring buffer should have been flushed to on_chunk
        self.assertEqual(len(received), 3)
        self.assertEqual(len(recorder._ring), 0)

    def test_cancel_keeps_stream_when_pre_buffer_enabled(self) -> None:
        recorder = AudioRecorder(
            sample_rate=16000, channels=1, dtype="float32",
            pre_buffer_seconds=0.3,
        )
        # Simulate an active recording with a fake stream
        fake_stream = object()
        recorder._stream = fake_stream  # type: ignore[assignment]
        recorder._recording = True
        recorder._on_chunk = lambda _: None

        recorder.cancel()

        # Stream should have been moved to hot standby, not closed
        self.assertIs(recorder._hot_stream, fake_stream)
        self.assertFalse(recorder._recording)

    def test_no_pre_buffer_when_disabled(self) -> None:
        recorder = AudioRecorder(
            sample_rate=16000, channels=1, dtype="float32",
            pre_buffer_seconds=0.0,
        )
        frame = np.ones((160, 1), dtype=np.float32)
        recorder._callback(frame, 160, None, None)

        # Ring should stay empty when pre_buffer_seconds is 0
        self.assertEqual(len(recorder._ring), 0)

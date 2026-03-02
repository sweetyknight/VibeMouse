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

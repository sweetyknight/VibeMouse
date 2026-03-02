from __future__ import annotations

import collections
import importlib
import logging
import threading
from collections.abc import Callable
from typing import Protocol, cast

from vibemouse.transcriber import AudioFrame

logger = logging.getLogger(__name__)


class _AudioStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _SoundDeviceModule(Protocol):
    def InputStream(
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        callback: Callable[[AudioFrame, int, object, object], None],
    ) -> _AudioStream: ...


class AudioRecorder:
    """Records audio from the default input device.

    When *pre_buffer_seconds* > 0, the microphone stream stays open between
    recordings so that a small ring-buffer of recent audio is always
    available.  On the next ``start()`` call the buffered audio is flushed
    to ``on_chunk`` immediately, eliminating the "first syllable lost"
    problem common with push-to-talk VAD pipelines.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        dtype: str,
        *,
        pre_buffer_seconds: float = 0.0,
    ) -> None:
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._dtype: str = dtype
        self._sd: _SoundDeviceModule | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stream: _AudioStream | None = None
        self._recording: bool = False
        self._on_chunk: Callable[[AudioFrame], None] | None = None

        # Pre-buffer: keep the last N chunks so the beginning of speech
        # that arrives before the button press is not lost.
        self._pre_buffer_seconds: float = pre_buffer_seconds
        self._ring: collections.deque[AudioFrame] = collections.deque()
        self._ring_samples: int = 0
        self._ring_max_samples: int = int(sample_rate * pre_buffer_seconds)
        self._hot_stream: _AudioStream | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def ensure_hot(self) -> None:
        """Start the microphone stream for pre-buffering (no-op if already running)."""
        if self._pre_buffer_seconds <= 0:
            return
        self._ensure_audio_module()
        with self._lock:
            if self._hot_stream is not None:
                return
            if self._sd is None:
                raise RuntimeError("Audio input module not initialized")
            logger.info(
                "Opening hot mic stream (rate=%d, ch=%d, dtype=%s)",
                self._sample_rate, self._channels, self._dtype,
            )
            stream = self._sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype,
                callback=self._callback,
            )
            stream.start()
            logger.info("Hot mic stream started")
            self._hot_stream = stream

    def start(
        self, *, on_chunk: Callable[[AudioFrame], None] | None = None
    ) -> None:
        self._ensure_audio_module()
        with self._lock:
            if self._recording:
                return
            self._on_chunk = on_chunk

            # If a hot stream is already running, reuse it.
            if self._hot_stream is not None:
                self._stream = self._hot_stream
                self._hot_stream = None
            else:
                if self._sd is None:
                    raise RuntimeError("Audio input module not initialized")
                stream = self._sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype=self._dtype,
                    callback=self._callback,
                )
                stream.start()
                self._stream = stream

            self._recording = True

            # Flush the pre-buffer to the consumer before any new audio arrives.
            buffered = list(self._ring)
            self._ring.clear()
            self._ring_samples = 0

        # Deliver buffered chunks outside the lock.
        if on_chunk is not None:
            for chunk in buffered:
                try:
                    on_chunk(chunk)
                except Exception:
                    pass

    def cancel(self) -> None:
        with self._lock:
            if not self._recording:
                self._on_chunk = None
                return
            stream = self._stream
            self._stream = None
            self._recording = False
            self._on_chunk = None

            # If pre-buffering is enabled, keep the stream running.
            if self._pre_buffer_seconds > 0 and stream is not None:
                self._hot_stream = stream
                stream = None  # do NOT close it

        if stream is not None:
            stream.stop()
            stream.close()

    def shutdown(self) -> None:
        """Close the microphone stream completely (including hot-standby)."""
        self.cancel()
        with self._lock:
            hot = self._hot_stream
            self._hot_stream = None
            self._ring.clear()
            self._ring_samples = 0
        if hot is not None:
            hot.stop()
            hot.close()

    def _callback(
        self, indata: AudioFrame, frames: int, time_data: object, status: object
    ) -> None:
        del frames, time_data, status
        chunk = indata.reshape(-1).copy()

        on_chunk = self._on_chunk
        if on_chunk is not None:
            try:
                on_chunk(chunk)
            except Exception:
                pass
        elif self._ring_max_samples > 0:
            # Not recording — accumulate in the ring buffer.
            self._ring.append(chunk)
            self._ring_samples += len(chunk)
            while self._ring_samples > self._ring_max_samples and self._ring:
                evicted = self._ring.popleft()
                self._ring_samples -= len(evicted)

    def _ensure_audio_module(self) -> None:
        if self._sd is not None:
            return
        try:
            sounddevice_module = importlib.import_module("sounddevice")
        except Exception as error:
            logger.error("Failed to import sounddevice: %s", error, exc_info=True)
            raise RuntimeError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from error

        logger.info(
            "sounddevice loaded (PortAudio %s)",
            getattr(sounddevice_module, "get_portaudio_version", lambda: ("?",))(),
        )
        self._sd = cast(_SoundDeviceModule, sounddevice_module)

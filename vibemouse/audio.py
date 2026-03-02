from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


AudioFrame = NDArray[np.float32]


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
    def __init__(
        self, sample_rate: int, channels: int, dtype: str
    ) -> None:
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._dtype: str = dtype
        self._sd: _SoundDeviceModule | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stream: _AudioStream | None = None
        self._recording: bool = False
        self._on_chunk: Callable[[AudioFrame], None] | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(
        self, *, on_chunk: Callable[[AudioFrame], None] | None = None
    ) -> None:
        self._ensure_audio_module()
        with self._lock:
            if self._recording:
                return
            self._on_chunk = on_chunk
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

    def cancel(self) -> None:
        with self._lock:
            if not self._recording:
                self._on_chunk = None
                return
            stream = self._stream
            self._stream = None
            self._recording = False
            self._on_chunk = None

        if stream is not None:
            stream.stop()
            stream.close()

    def _callback(
        self, indata: AudioFrame, frames: int, time_data: object, status: object
    ) -> None:
        del frames, time_data, status
        on_chunk = self._on_chunk
        if on_chunk is not None:
            try:
                # Produce a pre-flattened 1D float32 copy in a single allocation.
                # sounddevice gives us (frames, channels) — reshape(-1) creates a
                # 1D view, then .copy() materialises a contiguous 1D array.
                on_chunk(indata.reshape(-1).copy())
            except Exception:
                pass

    def _ensure_audio_module(self) -> None:
        if self._sd is not None:
            return
        try:
            sounddevice_module = importlib.import_module("sounddevice")
        except Exception as error:
            raise RuntimeError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from error

        self._sd = cast(_SoundDeviceModule, cast(object, sounddevice_module))

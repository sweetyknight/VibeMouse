from __future__ import annotations

import importlib
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from vibemouse.config import AppConfig
from vibemouse.model_manager import SherpaModelPaths, resolve_model

AudioFrame = NDArray[np.float32]

_SENTINEL = object()


@dataclass(frozen=True)
class StreamingResult:
    """Immutable snapshot of a streaming recognition result."""

    text: str
    is_final: bool


StreamingCallback = Callable[[StreamingResult], None]


# ---------------------------------------------------------------------------
# Protocol types for the sherpa-onnx external dependency
# ---------------------------------------------------------------------------


class _OnlineStream(Protocol):
    def accept_waveform(self, sample_rate: int, waveform: AudioFrame) -> None: ...


class _OnlineRecognizer(Protocol):
    def create_stream(self) -> _OnlineStream: ...

    def is_ready(self, stream: _OnlineStream) -> bool: ...

    def decode_stream(self, stream: _OnlineStream) -> None: ...

    def get_result(self, stream: _OnlineStream) -> str: ...

    def is_endpoint(self, stream: _OnlineStream) -> bool: ...

    def reset(self, stream: _OnlineStream) -> None: ...


# ---------------------------------------------------------------------------
# StreamingTranscriber — manages the sherpa-onnx recogniser lifecycle
# ---------------------------------------------------------------------------


class StreamingTranscriber:
    """Wraps a sherpa-onnx ``OnlineRecognizer`` for streaming ASR."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._recognizer: _OnlineRecognizer | None = None
        self._load_lock = threading.Lock()

    def ensure_loaded(self) -> None:
        """Eagerly load the model.  Safe to call more than once."""
        if self._recognizer is not None:
            return
        with self._load_lock:
            if self._recognizer is not None:
                return
            # Verify the package is installed before downloading the model.
            try:
                importlib.import_module("sherpa_onnx")
            except ImportError as error:
                raise RuntimeError(
                    "sherpa-onnx is not installed. Install with: pip install sherpa-onnx"
                ) from error
            paths = resolve_model(self._config.sherpa_model_dir)
            self._recognizer = self._create_recognizer(paths)

    def start_session(self, on_result: StreamingCallback) -> StreamingSession:
        """Create a new streaming session backed by its own decode thread."""
        self.ensure_loaded()
        if self._recognizer is None:
            raise RuntimeError("sherpa-onnx recognizer failed to initialize")
        return StreamingSession(
            recognizer=self._recognizer,
            sample_rate=self._config.sample_rate,
            on_result=on_result,
        )

    def _create_recognizer(self, paths: SherpaModelPaths) -> _OnlineRecognizer:
        try:
            sherpa = importlib.import_module("sherpa_onnx")
        except ImportError as error:
            raise RuntimeError(
                "sherpa-onnx is not installed. Install with: pip install sherpa-onnx"
            ) from error

        from_paraformer = getattr(
            getattr(sherpa, "OnlineRecognizer"), "from_paraformer", None
        )
        if from_paraformer is None:
            raise RuntimeError(
                "sherpa_onnx.OnlineRecognizer.from_paraformer not found"
            )

        return cast(
            _OnlineRecognizer,
            from_paraformer(
                tokens=str(paths.tokens),
                encoder=str(paths.encoder),
                decoder=str(paths.decoder),
                num_threads=self._config.sherpa_num_threads,
                sample_rate=self._config.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=300,
                decoding_method="greedy_search",
                provider="cpu",
            ),
        )


# ---------------------------------------------------------------------------
# StreamingSession — one recording→text lifecycle
# ---------------------------------------------------------------------------


class StreamingSession:
    """Owns an audio queue, a decode thread, and emits incremental results."""

    def __init__(
        self,
        recognizer: _OnlineRecognizer,
        sample_rate: int,
        on_result: StreamingCallback,
    ) -> None:
        self._recognizer = recognizer
        self._sample_rate = sample_rate
        self._on_result = on_result
        # Cap at ~200 chunks (~0.8 s of 16 kHz mono audio at 64-frame blocks)
        # to bound memory if decoding falls behind.
        self._audio_queue: queue.Queue[AudioFrame | object] = queue.Queue(maxsize=200)
        self._last_text: str = ""
        self._stopped = threading.Event()
        self._final_text: str = ""
        self._stream = recognizer.create_stream()
        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True
        )
        self._decode_thread.start()

    def feed_audio(self, chunk: AudioFrame) -> None:
        """Called from the audio-callback thread.  Must be non-blocking."""
        if not self._stopped.is_set():
            # chunk is already 1D float32 from AudioRecorder — enqueue directly.
            # Use put_nowait so the real-time audio callback never blocks; if the
            # queue is full (decoder falling behind), we silently drop the chunk.
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def stop(self) -> str:
        """Signal the decode thread to finish and return the final text."""
        self._audio_queue.put(_SENTINEL)
        self._stopped.set()
        self._decode_thread.join(timeout=5.0)
        return self._final_text

    # ------------------------------------------------------------------
    # Decode loop (runs on its own daemon thread)
    # ------------------------------------------------------------------

    def _decode_loop(self) -> None:
        stream = self._stream
        confirmed_parts: list[str] = []
        current_segment = ""

        try:
            while True:
                # Block for the first chunk, then drain all available chunks
                # and concatenate them into a single waveform to minimise the
                # number of accept_waveform / decode_stream round-trips.
                try:
                    first = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if first is _SENTINEL:
                    break

                chunks: list[AudioFrame] = [cast(AudioFrame, first)]
                sentinel_seen = False
                while True:
                    try:
                        item = self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _SENTINEL:
                        sentinel_seen = True
                        break
                    chunks.append(cast(AudioFrame, item))

                combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                stream.accept_waveform(self._sample_rate, combined)

                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)

                result = self._recognizer.get_result(stream)
                current_segment = result.strip() if result else ""

                full_text = "".join(confirmed_parts) + current_segment
                if full_text != self._last_text:
                    self._last_text = full_text
                    try:
                        self._on_result(StreamingResult(text=full_text, is_final=False))
                    except Exception:
                        pass

                if self._recognizer.is_endpoint(stream):
                    if current_segment:
                        confirmed_parts.append(current_segment)
                        current_segment = ""
                    self._recognizer.reset(stream)

                if sentinel_seen:
                    break
        except Exception as error:
            print(f"Decode loop error: {error}")

        # Final pass — always attempt to produce a result even after errors
        self._final_text = "".join(confirmed_parts) + current_segment
        if self._final_text != self._last_text:
            try:
                self._on_result(StreamingResult(text=self._final_text, is_final=True))
            except Exception:
                pass

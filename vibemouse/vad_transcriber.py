"""VAD + offline recognizer transcriber (e.g. FireRedASR).

Instead of streaming partial results character-by-character, this module
uses a Voice Activity Detector (Silero VAD) to find sentence boundaries
and then runs a high-accuracy offline recognizer on each complete speech
segment.  The result is "say a sentence → whole sentence appears".
"""

from __future__ import annotations

import importlib
import logging
import queue
import threading
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from vibemouse.config import AppConfig
from vibemouse.model_manager import (
    SherpaModelPaths,
    resolve_offline_model,
    resolve_vad_model,
)
from vibemouse.transcriber import AudioFrame, StreamingCallback, StreamingResult

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Silero VAD default window size at 16 kHz.
_VAD_WINDOW_SIZE = 512

# Minimum samples for the offline recognizer's encoder convolutions.
# Segments shorter than this are padded to avoid "Invalid input shape" errors.
# 0.5 s × 16 kHz = 8000 samples.
_MIN_SEGMENT_SAMPLES = 8000

# Silence appended before VAD flush so it detects the speech→silence boundary
# naturally instead of relying on forced emission.  10 VAD windows ≈ 0.32 s.
_TAIL_SILENCE_WINDOWS = 10

# Trailing silence appended to each segment BEFORE feeding the offline
# recognizer.  Convolutional encoders (like FireRedASR) need right-side
# context; without this padding the last 1–2 characters are often dropped.
# 0.3 s × 16 kHz = 4800 samples.
_RECOGNIZER_TAIL_PADDING = 4800


# ---------------------------------------------------------------------------
# Protocol types for sherpa-onnx external dependency
# ---------------------------------------------------------------------------


class _SpeechSegment(Protocol):
    @property
    def samples(self) -> NDArray[np.float32]: ...


class _VoiceActivityDetector(Protocol):
    def accept_waveform(self, samples: NDArray[np.float32]) -> None: ...

    def empty(self) -> bool: ...

    @property
    def front(self) -> _SpeechSegment: ...

    def pop(self) -> None: ...

    def flush(self) -> None: ...


class _OfflineResult(Protocol):
    @property
    def text(self) -> str: ...


class _OfflineStream(Protocol):
    def accept_waveform(self, sample_rate: int, samples: NDArray[np.float32]) -> None: ...

    @property
    def result(self) -> _OfflineResult: ...


class _OfflineRecognizer(Protocol):
    def create_stream(self) -> _OfflineStream: ...

    def decode_stream(self, stream: _OfflineStream) -> None: ...


# ---------------------------------------------------------------------------
# VadOfflineTranscriber — manages VAD + OfflineRecognizer lifecycle
# ---------------------------------------------------------------------------


class VadOfflineTranscriber:
    """Wraps a sherpa-onnx ``VoiceActivityDetector`` + ``OfflineRecognizer``."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._recognizer: _OfflineRecognizer | None = None
        self._vad_model_path: str | None = None
        self._sherpa: object | None = None
        self._load_lock = threading.Lock()

    def ensure_loaded(self) -> None:
        """Eagerly load both VAD and ASR models.  Safe to call more than once."""
        if self._recognizer is not None:
            return
        with self._load_lock:
            if self._recognizer is not None:
                return

            try:
                self._sherpa = importlib.import_module("sherpa_onnx")
            except ImportError as error:
                raise RuntimeError(
                    "sherpa-onnx is not installed. Install with: pip install sherpa-onnx"
                ) from error

            vad_path = resolve_vad_model(self._config.sherpa_model_dir)
            self._vad_model_path = str(vad_path)

            asr_paths = resolve_offline_model(
                self._config.sherpa_model_dir,
                self._config.offline_model_name,
            )
            self._recognizer = self._create_recognizer(asr_paths)

    def start_session(self, on_result: StreamingCallback) -> VadOfflineSession:
        """Create a new session backed by its own decode thread."""
        self.ensure_loaded()
        if self._recognizer is None or self._vad_model_path is None:
            raise RuntimeError("VAD/ASR models failed to initialize")

        vad = self._create_vad()
        return VadOfflineSession(
            vad=vad,
            recognizer=self._recognizer,
            sample_rate=self._config.sample_rate,
            on_result=on_result,
        )

    def _create_recognizer(self, paths: SherpaModelPaths) -> _OfflineRecognizer:
        sherpa = self._sherpa
        from_fire_red_asr = getattr(
            getattr(sherpa, "OfflineRecognizer"), "from_fire_red_asr", None
        )
        if from_fire_red_asr is None:
            raise RuntimeError(
                "sherpa_onnx.OfflineRecognizer.from_fire_red_asr not found. "
                "Upgrade sherpa-onnx to a version that includes FireRedASR support."
            )
        return cast(
            _OfflineRecognizer,
            from_fire_red_asr(
                encoder=str(paths.encoder),
                decoder=str(paths.decoder),
                tokens=str(paths.tokens),
                num_threads=self._config.sherpa_num_threads,
                decoding_method="greedy_search",
                provider="cpu",
            ),
        )

    def _create_vad(self) -> _VoiceActivityDetector:
        """Create a fresh VAD instance (one per session, since VAD is stateful)."""
        sherpa = self._sherpa

        vad_config = sherpa.VadModelConfig()
        vad_config.silero_vad.model = self._vad_model_path
        vad_config.silero_vad.min_silence_duration = self._config.vad_min_silence_duration
        vad_config.silero_vad.min_speech_duration = self._config.vad_min_speech_duration
        vad_config.silero_vad.threshold = self._config.vad_threshold
        vad_config.silero_vad.window_size = _VAD_WINDOW_SIZE
        vad_config.sample_rate = self._config.sample_rate
        vad_config.num_threads = 1  # VAD is lightweight
        vad_config.provider = "cpu"

        return cast(
            _VoiceActivityDetector,
            sherpa.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100),
        )


# ---------------------------------------------------------------------------
# VadOfflineSession — one recording → text lifecycle
# ---------------------------------------------------------------------------


class VadOfflineSession:
    """Owns an audio queue, a decode thread with VAD + offline recognition."""

    def __init__(
        self,
        vad: _VoiceActivityDetector,
        recognizer: _OfflineRecognizer,
        sample_rate: int,
        on_result: StreamingCallback,
    ) -> None:
        self._vad = vad
        self._recognizer = recognizer
        self._sample_rate = sample_rate
        self._on_result = on_result

        self._audio_queue: queue.Queue[AudioFrame | object] = queue.Queue(maxsize=200)
        self._last_text: str = ""
        self._stopped = threading.Event()
        self._final_text: str = ""

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True
        )
        self._decode_thread.start()

    def feed_audio(self, chunk: AudioFrame) -> None:
        """Called from the audio-callback thread.  Must be non-blocking."""
        if not self._stopped.is_set():
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def stop(self) -> str:
        """Signal the decode thread to finish and return the final text."""
        # Place the sentinel first so the decode loop processes all queued
        # audio chunks before exiting.  Use a timeout to avoid blocking
        # forever if the queue is full.
        try:
            self._audio_queue.put(_SENTINEL, timeout=5.0)
        except queue.Full:
            pass
        # Prevent further feed_audio calls and serve as a backup exit
        # condition for the decode loop if the sentinel was not placed.
        self._stopped.set()
        # Longer timeout than streaming — offline recognition can be slow.
        self._decode_thread.join(timeout=10.0)
        return self._final_text

    # ------------------------------------------------------------------
    # Decode loop (runs on its own daemon thread)
    # ------------------------------------------------------------------

    def _decode_loop(self) -> None:  # noqa: C901 — unavoidable complexity
        vad = self._vad
        recognizer = self._recognizer
        confirmed_parts: list[str] = []
        leftover: AudioFrame | None = None

        try:
            while True:
                # Block for the first chunk, then drain all available.
                try:
                    first = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stopped.is_set():
                        break
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

                # Prepend leftover from previous iteration.
                if leftover is not None:
                    chunks.insert(0, leftover)
                    leftover = None

                combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

                # Feed VAD in window_size-aligned chunks.
                offset = 0
                while offset + _VAD_WINDOW_SIZE <= len(combined):
                    window = combined[offset : offset + _VAD_WINDOW_SIZE]
                    vad.accept_waveform(window)
                    offset += _VAD_WINDOW_SIZE

                # Save remainder for next iteration.
                if offset < len(combined):
                    leftover = combined[offset:]

                # Recognize any complete speech segments.
                self._drain_vad(vad, recognizer, confirmed_parts)

                if sentinel_seen:
                    break

        except Exception as error:
            logger.exception("VAD decode loop error: %s", error)

        # Flush remaining audio through VAD.
        self._flush_vad(vad, leftover)
        self._drain_vad(vad, recognizer, confirmed_parts)

        self._final_text = "".join(confirmed_parts)
        if self._final_text != self._last_text:
            try:
                self._on_result(
                    StreamingResult(text=self._final_text, is_final=True)
                )
            except Exception:
                pass

    def _drain_vad(
        self,
        vad: _VoiceActivityDetector,
        recognizer: _OfflineRecognizer,
        confirmed_parts: list[str],
    ) -> None:
        """Recognize all speech segments currently available in the VAD."""
        while not vad.empty():
            # Copy samples BEFORE pop() — the C++ binding returns a view
            # into an internal buffer that is invalidated by pop().
            samples = np.array(vad.front.samples)
            vad.pop()

            text = self._recognize_segment(recognizer, samples)
            if text:
                confirmed_parts.append(text)
                full_text = "".join(confirmed_parts)
                if full_text != self._last_text:
                    self._last_text = full_text
                    try:
                        self._on_result(
                            StreamingResult(text=full_text, is_final=False)
                        )
                    except Exception:
                        pass

    @staticmethod
    def _flush_vad(
        vad: _VoiceActivityDetector,
        leftover: AudioFrame | None,
    ) -> None:
        """Pad leftover samples and flush the VAD to emit any trailing speech."""
        try:
            if leftover is not None and len(leftover) > 0:
                if len(leftover) < _VAD_WINDOW_SIZE:
                    padded = np.zeros(_VAD_WINDOW_SIZE, dtype=np.float32)
                    padded[: len(leftover)] = leftover
                    vad.accept_waveform(padded)
                else:
                    # Shouldn't happen, but handle gracefully.
                    vad.accept_waveform(leftover)

            # Feed trailing silence so the encoder has right-side context for
            # the final speech frames.  This mimics the natural "speech → silence"
            # pattern that the VAD sees during normal mid-recording emission.
            silence_window = np.zeros(_VAD_WINDOW_SIZE, dtype=np.float32)
            for _ in range(_TAIL_SILENCE_WINDOWS):
                vad.accept_waveform(silence_window)

            vad.flush()
        except Exception:
            logger.exception("VAD flush error")

    def _recognize_segment(
        self,
        recognizer: _OfflineRecognizer,
        samples: NDArray[np.float32],
    ) -> str:
        """Run offline recognition on a speech segment.  Returns stripped text."""
        if len(samples) == 0:
            return ""

        # Ensure minimum length for the encoder's input shape requirement.
        if len(samples) < _MIN_SEGMENT_SAMPLES:
            padded = np.zeros(_MIN_SEGMENT_SAMPLES, dtype=np.float32)
            padded[: len(samples)] = samples
            samples = padded

        # Append trailing silence so the encoder's convolutional layers have
        # right-side context for the final speech frames.  VAD segments end
        # right at the speech boundary — without this the last 1–2 chars of
        # fast speech are frequently dropped.
        samples = np.concatenate(
            [samples, np.zeros(_RECOGNIZER_TAIL_PADDING, dtype=np.float32)]
        )

        stream = recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, samples)
        recognizer.decode_stream(stream)
        result_text = stream.result.text
        return result_text.strip() if result_text else ""

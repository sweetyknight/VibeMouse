from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _read_button(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in {"x1", "x2"}:
        raise ValueError(f"{name} must be either 'x1' or 'x2', got {value!r}")
    return value


def _require_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _require_non_negative(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a float, got {raw!r}") from error


def _read_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {options}; got {value!r}")
    return value


@dataclass(frozen=True)
class AppConfig:
    # Audio recording
    sample_rate: int
    channels: int
    dtype: str
    pre_buffer_seconds: float

    # Sherpa-onnx common
    sherpa_model_dir: Path
    sherpa_num_threads: int

    # ASR backend
    asr_backend: str  # "vad_offline"

    # VAD settings
    vad_min_silence_duration: float
    vad_min_speech_duration: float
    vad_threshold: float

    # Offline model name (used when asr_backend == "vad_offline")
    offline_model_name: str

    # Mouse buttons
    button_debounce_ms: int
    front_button: str
    rear_button: str

    # Output
    enter_mode: str
    auto_paste: bool


def load_config() -> AppConfig:
    sample_rate = _require_positive(
        "VIBEMOUSE_SAMPLE_RATE", _read_int("VIBEMOUSE_SAMPLE_RATE", 16000)
    )
    channels = _require_positive(
        "VIBEMOUSE_CHANNELS", _read_int("VIBEMOUSE_CHANNELS", 1)
    )

    sherpa_model_dir = Path(
        os.getenv(
            "VIBEMOUSE_SHERPA_MODEL_DIR",
            str(Path.home() / ".cache" / "vibemouse" / "models"),
        )
    )
    sherpa_num_threads = _require_positive(
        "VIBEMOUSE_SHERPA_NUM_THREADS",
        _read_int("VIBEMOUSE_SHERPA_NUM_THREADS", 2),
    )

    front_button = _read_button("VIBEMOUSE_FRONT_BUTTON", "x1")
    rear_button = _read_button("VIBEMOUSE_REAR_BUTTON", "x2")
    if front_button == rear_button:
        raise ValueError("VIBEMOUSE_FRONT_BUTTON and VIBEMOUSE_REAR_BUTTON must differ")

    button_debounce_ms = _require_non_negative(
        "VIBEMOUSE_BUTTON_DEBOUNCE_MS",
        _read_int("VIBEMOUSE_BUTTON_DEBOUNCE_MS", 150),
    )
    enter_mode = _read_choice(
        "VIBEMOUSE_ENTER_MODE",
        "enter",
        {"enter", "ctrl_enter", "shift_enter", "none"},
    )

    asr_backend = _read_choice(
        "VIBEMOUSE_ASR_BACKEND", "vad_offline", {"vad_offline"}
    )
    vad_min_silence_duration = _read_float("VIBEMOUSE_VAD_MIN_SILENCE_DURATION", 0.25)
    vad_min_speech_duration = _read_float("VIBEMOUSE_VAD_MIN_SPEECH_DURATION", 0.25)
    vad_threshold = _read_float("VIBEMOUSE_VAD_THRESHOLD", 0.5)
    offline_model_name = os.getenv(
        "VIBEMOUSE_OFFLINE_MODEL_NAME",
        "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16",
    )

    pre_buffer_seconds = _read_float("VIBEMOUSE_PRE_BUFFER_SECONDS", 0.5)

    return AppConfig(
        sample_rate=sample_rate,
        channels=channels,
        dtype=os.getenv("VIBEMOUSE_DTYPE", "float32"),
        pre_buffer_seconds=pre_buffer_seconds,
        sherpa_model_dir=sherpa_model_dir,
        sherpa_num_threads=sherpa_num_threads,
        asr_backend=asr_backend,
        vad_min_silence_duration=vad_min_silence_duration,
        vad_min_speech_duration=vad_min_speech_duration,
        vad_threshold=vad_threshold,
        offline_model_name=offline_model_name,
        button_debounce_ms=button_debounce_ms,
        enter_mode=enter_mode,
        auto_paste=_read_bool("VIBEMOUSE_AUTO_PASTE", True),
        front_button=front_button,
        rear_button=rear_button,
    )

from __future__ import annotations

import os
import tempfile
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
    return int(raw.strip())


@dataclass(frozen=True)
class AppConfig:
    sample_rate: int
    channels: int
    dtype: str
    transcriber_backend: str
    model_name: str
    device: str
    language: str
    use_itn: bool
    enable_vad: bool
    vad_max_single_segment_ms: int
    merge_vad: bool
    merge_length_s: int
    fallback_to_cpu: bool
    front_button: str
    rear_button: str
    temp_dir: Path


def load_config() -> AppConfig:
    temp_dir = Path(
        os.getenv("VIBEMOUSE_TEMP_DIR", str(Path(tempfile.gettempdir()) / "vibemouse"))
    )
    return AppConfig(
        sample_rate=_read_int("VIBEMOUSE_SAMPLE_RATE", 16000),
        channels=_read_int("VIBEMOUSE_CHANNELS", 1),
        dtype=os.getenv("VIBEMOUSE_DTYPE", "float32"),
        transcriber_backend=os.getenv("VIBEMOUSE_BACKEND", "auto").strip().lower(),
        model_name=os.getenv("VIBEMOUSE_MODEL", "iic/SenseVoiceSmall"),
        device=os.getenv("VIBEMOUSE_DEVICE", "cpu"),
        language=os.getenv("VIBEMOUSE_LANGUAGE", "auto"),
        use_itn=_read_bool("VIBEMOUSE_USE_ITN", True),
        enable_vad=_read_bool("VIBEMOUSE_ENABLE_VAD", True),
        vad_max_single_segment_ms=_read_int("VIBEMOUSE_VAD_MAX_SEGMENT_MS", 30000),
        merge_vad=_read_bool("VIBEMOUSE_MERGE_VAD", True),
        merge_length_s=_read_int("VIBEMOUSE_MERGE_LENGTH_S", 15),
        fallback_to_cpu=_read_bool("VIBEMOUSE_FALLBACK_CPU", True),
        front_button=os.getenv("VIBEMOUSE_FRONT_BUTTON", "x1").strip().lower(),
        rear_button=os.getenv("VIBEMOUSE_REAR_BUTTON", "x2").strip().lower(),
        temp_dir=temp_dir,
    )

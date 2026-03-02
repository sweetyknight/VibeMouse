from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
_MODEL_URL_TEMPLATE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "{model_name}.tar.bz2"
)


@dataclass(frozen=True)
class SherpaModelPaths:
    """Resolved paths for the three required model files."""

    tokens: Path
    encoder: Path
    decoder: Path


def resolve_model(
    model_dir: Path,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> SherpaModelPaths:
    """Ensure the model is downloaded and return paths to its files."""
    model_path = model_dir / model_name
    if not _model_files_present(model_path):
        _download_and_extract(model_dir, model_name)
    return _build_paths(model_path)


def _model_files_present(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    has_tokens = (model_path / "tokens.txt").exists()
    has_encoder = bool(list(model_path.glob("encoder*.onnx")))
    return has_tokens and has_encoder


def _download_and_extract(model_dir: Path, model_name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    url = _MODEL_URL_TEMPLATE.format(model_name=model_name)
    archive_path = model_dir / f"{model_name}.tar.bz2"

    logger.info("Downloading sherpa-onnx model from %s", url)
    print(f"Downloading sherpa-onnx model: {model_name} ...")
    try:
        urlretrieve(url, str(archive_path))
    except Exception as error:
        raise RuntimeError(
            f"Failed to download sherpa-onnx model from {url}: {error}"
        ) from error

    logger.info("Extracting model to %s", model_dir)
    print(f"Extracting model to {model_dir} ...")
    try:
        with tarfile.open(str(archive_path), "r:bz2") as tar:
            tar.extractall(path=str(model_dir), filter="data")
    except Exception as error:
        raise RuntimeError(
            f"Failed to extract model archive: {error}"
        ) from error
    finally:
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass

    print(f"Model ready: {model_dir / model_name}")


def _build_paths(model_path: Path) -> SherpaModelPaths:
    tokens = model_path / "tokens.txt"
    if not tokens.exists():
        raise RuntimeError(f"tokens.txt not found in {model_path}")

    encoder = _find_onnx(model_path, "encoder")
    decoder = _find_onnx(model_path, "decoder")

    return SherpaModelPaths(tokens=tokens, encoder=encoder, decoder=decoder)


def _find_onnx(model_path: Path, prefix: str) -> Path:
    """Find an ONNX file matching *prefix*, preferring int8 quantised."""
    for pattern in [f"{prefix}*.int8.onnx", f"{prefix}*.onnx"]:
        candidates = sorted(model_path.glob(pattern))
        if candidates:
            return candidates[0]
    raise RuntimeError(f"No {prefix}*.onnx file found in {model_path}")

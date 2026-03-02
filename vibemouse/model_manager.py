from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_SIZE = 256 * 1024  # 256 KB per read
_DOWNLOAD_TIMEOUT = 30  # seconds per network operation

_DEFAULT_OFFLINE_MODEL_NAME = "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16"
_MODEL_URL_TEMPLATE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "{model_name}.tar.bz2"
)
_VAD_MODEL_FILENAME = "silero_vad.onnx"
_VAD_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)

_PUNCTUATION_MODEL_URL_TEMPLATE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/"
    "{model_name}.tar.bz2"
)


@dataclass(frozen=True)
class SherpaModelPaths:
    """Resolved paths for the three required model files."""

    tokens: Path
    encoder: Path
    decoder: Path


def resolve_vad_model(model_dir: Path) -> Path:
    """Ensure silero_vad.onnx is downloaded and return its path."""
    vad_path = model_dir / _VAD_MODEL_FILENAME
    if vad_path.exists():
        return vad_path
    _download_single_file(model_dir, _VAD_MODEL_URL, _VAD_MODEL_FILENAME)
    return vad_path


def resolve_offline_model(
    model_dir: Path,
    model_name: str = _DEFAULT_OFFLINE_MODEL_NAME,
) -> SherpaModelPaths:
    """Ensure the offline ASR model is downloaded and return paths."""
    model_path = model_dir / model_name
    if not _model_files_present(model_path):
        _download_and_extract(model_dir, model_name)
    return _build_paths(model_path)


def resolve_punctuation_model(
    model_dir: Path,
    model_name: str,
) -> Path:
    """Ensure the punctuation model is downloaded and return the .onnx path."""
    model_path = model_dir / model_name
    if not model_path.exists():
        _download_and_extract_punctuation(model_dir, model_name)
    return _find_punct_onnx(model_path)


def _download_and_extract_punctuation(model_dir: Path, model_name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    url = _PUNCTUATION_MODEL_URL_TEMPLATE.format(model_name=model_name)
    archive_path = model_dir / f"{model_name}.tar.bz2"
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")

    logger.info("Downloading punctuation model from %s", url)
    print(f"Downloading punctuation model: {model_name} ...")
    try:
        _resumable_download(url, part_path)
    except Exception as error:
        _safe_unlink(part_path)
        raise RuntimeError(
            f"Failed to download punctuation model from {url}: {error}"
        ) from error

    part_path.rename(archive_path)

    logger.info("Extracting punctuation model to %s", model_dir)
    print(f"Extracting punctuation model to {model_dir} ...")
    extract_target = model_dir / model_name
    try:
        with tarfile.open(str(archive_path), "r:bz2") as tar:
            tar.extractall(path=str(model_dir), filter="data")
    except Exception as error:
        _safe_rmtree(extract_target)
        raise RuntimeError(
            f"Failed to extract punctuation model archive: {error}"
        ) from error
    finally:
        _safe_unlink(archive_path)

    print(f"Punctuation model ready: {extract_target}")


def _find_punct_onnx(model_path: Path) -> Path:
    """Find the punctuation .onnx file, preferring int8 quantised."""
    for pattern in ["model.int8.onnx", "model.onnx"]:
        candidate = model_path / pattern
        if candidate.exists():
            return candidate
    # Fallback: search recursively.
    for pattern in ["**/model.int8.onnx", "**/model.onnx"]:
        candidates = sorted(model_path.glob(pattern))
        if candidates:
            return candidates[0]
    raise RuntimeError(f"No punctuation model.onnx found in {model_path}")


def _download_single_file(model_dir: Path, url: str, filename: str) -> None:
    """Download a single file (not an archive) into *model_dir*.

    Uses a ``.part`` temporary file so a partial download is never mistaken
    for a complete one.  On failure the partial file is removed.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / filename
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s from %s", filename, url)
    print(f"Downloading {filename} ...")
    try:
        _resumable_download(url, part)
    except Exception as error:
        _safe_unlink(part)
        raise RuntimeError(
            f"Failed to download {filename} from {url}: {error}"
        ) from error
    part.rename(dest)
    print(f"Downloaded: {dest}")


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
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")

    logger.info("Downloading sherpa-onnx model from %s", url)
    print(f"Downloading sherpa-onnx model: {model_name} ...")
    try:
        _resumable_download(url, part_path)
    except Exception as error:
        _safe_unlink(part_path)
        raise RuntimeError(
            f"Failed to download sherpa-onnx model from {url}: {error}"
        ) from error

    # Rename .part → final archive name only after download completes.
    part_path.rename(archive_path)

    logger.info("Extracting model to %s", model_dir)
    print(f"Extracting model to {model_dir} ...")
    extract_target = model_dir / model_name
    try:
        with tarfile.open(str(archive_path), "r:bz2") as tar:
            tar.extractall(path=str(model_dir), filter="data")
    except Exception as error:
        # Clean up partially extracted directory.
        _safe_rmtree(extract_target)
        raise RuntimeError(
            f"Failed to extract model archive: {error}"
        ) from error
    finally:
        _safe_unlink(archive_path)

    print(f"Model ready: {extract_target}")


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


# ---------------------------------------------------------------------------
# Resumable download with progress display
# ---------------------------------------------------------------------------


def _resumable_download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with HTTP Range resume support.

    If *dest* already contains a partial download, an HTTP ``Range`` header is
    sent to resume from where it left off.  Servers that do not support range
    requests (return 200 instead of 206) cause the file to be re-downloaded
    from scratch.
    """
    existing_size = dest.stat().st_size if dest.exists() else 0

    headers: dict[str, str] = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info("Resuming download from byte %d", existing_size)
        print(f"Resuming download from {_format_size(existing_size)} ...")

    request = Request(url, headers=headers)
    response = urlopen(request, timeout=_DOWNLOAD_TIMEOUT)  # noqa: S310

    # If server responded 200 (not 206), it does not honour Range —
    # start from scratch.
    if response.status == 200 and existing_size > 0:
        logger.info("Server does not support Range; restarting download")
        existing_size = 0

    content_length = response.headers.get("Content-Length")
    total_size = int(content_length) + existing_size if content_length else None

    mode = "ab" if existing_size > 0 and response.status == 206 else "wb"
    downloaded = existing_size if mode == "ab" else 0
    last_pct = -1

    with open(dest, mode) as fh:
        while True:
            chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total_size and total_size > 0:
                pct = int(downloaded * 100 / total_size)
                if pct != last_pct:
                    print(
                        f"\r  {_format_size(downloaded)} / "
                        f"{_format_size(total_size)} ({pct}%)",
                        end="",
                        flush=True,
                    )
                    last_pct = pct
    # End the progress line.
    print()


def _format_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def _safe_unlink(path: Path) -> None:
    """Remove a file, ignoring errors if it does not exist."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree, ignoring errors."""
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass

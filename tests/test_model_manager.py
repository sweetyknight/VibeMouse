from __future__ import annotations

import tarfile
import tempfile
import unittest
from pathlib import Path

from vibemouse.model_manager import (
    SherpaModelPaths,
    _build_paths,
    _find_onnx,
    _model_files_present,
)


class ModelFilesPresenceTests(unittest.TestCase):
    def test_missing_directory_returns_false(self) -> None:
        self.assertFalse(_model_files_present(Path("/nonexistent/path")))

    def test_empty_directory_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_model_files_present(Path(tmp)))

    def test_tokens_only_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tokens.txt").touch()
            self.assertFalse(_model_files_present(Path(tmp)))

    def test_tokens_and_encoder_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tokens.txt").touch()
            (Path(tmp) / "encoder.onnx").touch()
            self.assertTrue(_model_files_present(Path(tmp)))


class FindOnnxTests(unittest.TestCase):
    def test_prefers_int8_quantized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "encoder.onnx").touch()
            (p / "encoder.int8.onnx").touch()
            result = _find_onnx(p, "encoder")
            self.assertEqual(result.name, "encoder.int8.onnx")

    def test_falls_back_to_plain_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "encoder.onnx").touch()
            result = _find_onnx(p, "encoder")
            self.assertEqual(result.name, "encoder.onnx")

    def test_raises_when_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No encoder"):
                _find_onnx(Path(tmp), "encoder")


class BuildPathsTests(unittest.TestCase):
    def test_builds_correct_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "tokens.txt").touch()
            (p / "encoder.int8.onnx").touch()
            (p / "decoder.int8.onnx").touch()

            paths = _build_paths(p)
            self.assertEqual(paths.tokens.name, "tokens.txt")
            self.assertEqual(paths.encoder.name, "encoder.int8.onnx")
            self.assertEqual(paths.decoder.name, "decoder.int8.onnx")

    def test_raises_when_tokens_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "tokens.txt"):
                _build_paths(Path(tmp))

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from vibemouse.config import load_config

# On Windows, Path.home() needs USERPROFILE; on Unix it needs HOME.
_HOME_VARS = {
    k: v
    for k, v in os.environ.items()
    if k in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
}


def _env(**overrides: str) -> dict[str, str]:
    """Return a clean env dict with only home vars + explicit overrides."""
    merged = dict(_HOME_VARS)
    merged.update(overrides)
    return merged


class LoadConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            config = load_config()

        self.assertTrue(config.auto_paste)
        self.assertEqual(config.enter_mode, "enter")
        self.assertEqual(config.button_debounce_ms, 150)
        self.assertEqual(config.front_button, "x1")
        self.assertEqual(config.rear_button, "x2")
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.channels, 1)
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.sherpa_num_threads, 2)
        self.assertEqual(
            config.sherpa_model_dir,
            Path.home() / ".cache" / "vibemouse" / "models",
        )

    def test_auto_paste_can_be_disabled(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_AUTO_PASTE="false"), clear=True):
            config = load_config()

        self.assertFalse(config.auto_paste)

    def test_enter_mode_can_be_configured(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_ENTER_MODE="ctrl_enter"), clear=True):
            config = load_config()

        self.assertEqual(config.enter_mode, "ctrl_enter")

    def test_enter_mode_supports_none(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_ENTER_MODE="none"), clear=True):
            config = load_config()

        self.assertEqual(config.enter_mode, "none")

    def test_invalid_enter_mode_is_rejected(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_ENTER_MODE="meta_enter"), clear=True):
            with self.assertRaisesRegex(
                ValueError, "VIBEMOUSE_ENTER_MODE must be one of"
            ):
                _ = load_config()

    def test_negative_debounce_is_rejected(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_BUTTON_DEBOUNCE_MS="-1"), clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "VIBEMOUSE_BUTTON_DEBOUNCE_MS must be a non-negative integer",
            ):
                _ = load_config()

    def test_invalid_integer_reports_variable_name(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_SAMPLE_RATE="abc"), clear=True):
            with self.assertRaisesRegex(
                ValueError, "VIBEMOUSE_SAMPLE_RATE must be an integer"
            ):
                _ = load_config()

    def test_invalid_button_value_is_rejected(self) -> None:
        with patch.dict(os.environ, _env(VIBEMOUSE_FRONT_BUTTON="x3"), clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "VIBEMOUSE_FRONT_BUTTON must be either 'x1' or 'x2'",
            ):
                _ = load_config()

    def test_same_front_and_rear_buttons_are_rejected(self) -> None:
        with patch.dict(
            os.environ,
            _env(VIBEMOUSE_FRONT_BUTTON="x1", VIBEMOUSE_REAR_BUTTON="x1"),
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VIBEMOUSE_FRONT_BUTTON and VIBEMOUSE_REAR_BUTTON must differ",
            ):
                _ = load_config()

    def test_sherpa_model_dir_can_be_configured(self) -> None:
        with patch.dict(
            os.environ,
            _env(VIBEMOUSE_SHERPA_MODEL_DIR="/tmp/my-models"),
            clear=True,
        ):
            config = load_config()

        self.assertEqual(config.sherpa_model_dir, Path("/tmp/my-models"))

    def test_sherpa_num_threads_can_be_configured(self) -> None:
        with patch.dict(
            os.environ,
            _env(VIBEMOUSE_SHERPA_NUM_THREADS="4"),
            clear=True,
        ):
            config = load_config()

        self.assertEqual(config.sherpa_num_threads, 4)

    def test_zero_sherpa_num_threads_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            _env(VIBEMOUSE_SHERPA_NUM_THREADS="0"),
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VIBEMOUSE_SHERPA_NUM_THREADS must be a positive integer",
            ):
                _ = load_config()

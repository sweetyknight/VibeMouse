from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AudioFrame = NDArray[np.float32]


@dataclass(frozen=True)
class StreamingResult:
    """Immutable snapshot of a streaming recognition result."""

    text: str
    is_final: bool


StreamingCallback = Callable[[StreamingResult], None]

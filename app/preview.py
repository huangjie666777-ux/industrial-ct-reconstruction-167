"""PNG preview rendering.

The preview uses a simple min/max stretch and only affects the rendered
PNG -- the floating point NPY payload is never modified.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def render_png(image: np.ndarray) -> bytes:
    """Return 8-bit grayscale PNG bytes for a 2-D float image.

    Values at ``min`` map to black and ``max`` to white; clipping is applied
    to the preview copy only.  Constant images render as medium grey.
    """
    image = np.asarray(image, dtype=np.float64)
    lo = float(np.min(image))
    hi = float(np.max(image))
    if hi > lo:
        scaled = (image - lo) * (255.0 / (hi - lo))
    else:
        scaled = np.full_like(image, 128.0)
    scaled = np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)

    buffer = io.BytesIO()
    # Image rows already follow the (row, col) = (+y, +x) convention.
    Image.fromarray(scaled, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()

"""End-to-end reconstruction pipeline shared by the API and the examples.

This module wires together the cross-file responsibilities:
:mod:`app.validation` for input rules, :mod:`app.calibration` for the dark
and flat correction and :mod:`app.fbp` for the filtered back-projection.
"""

from __future__ import annotations

import io

import numpy as np

from .calibration import line_integrals
from .fbp import fbp_reconstruct
from .validation import ReconstructionInput


def reconstruct(data: ReconstructionInput) -> np.ndarray:
    """Calibrate raw intensities and run FBP; returns attenuation in 1/mm."""
    sinogram = line_integrals(data.intensities, data.dark, data.flat)
    image = fbp_reconstruct(
        sinogram,
        det_spacing=data.det_spacing,
        center=data.center,
        output_size=data.output_size,
        pixel_spacing=data.pixel_spacing,
        filter_name=data.filter_name,
    )
    return image


def encode_npy(image: np.ndarray) -> bytes:
    """Serialise the reconstruction as a raw ``.npy`` byte string."""
    buffer = io.BytesIO()
    np.save(buffer, image, allow_pickle=False)
    return buffer.getvalue()


def numeric_ranges(
    image: np.ndarray,
    sinogram: np.ndarray,
) -> dict[str, float]:
    """Honest value-range statistics (negative values included)."""
    return {
        "recon_min": float(np.min(image)),
        "recon_max": float(np.max(image)),
        "recon_mean": float(np.mean(image)),
        "recon_std": float(np.std(image)),
        "sinogram_min": float(np.min(sinogram)),
        "sinogram_max": float(np.max(sinogram)),
    }

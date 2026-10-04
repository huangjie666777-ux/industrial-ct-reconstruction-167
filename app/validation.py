"""Validation of HTTP-supplied NPZ data and reconstruction parameters.

The same helpers are reused by the API layer and by the test-suite so that
limits stay in one place.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

MAX_ANGLES = 360
MAX_DETECTORS = 512
MAX_OUTPUT_SIZE = 256

# Reject oversized uploads / decompression-bombs early.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024

REQUIRED_KEYS = ("intensities", "dark", "flat")
SUPPORTED_FILTERS = ("ram-lak", "hann")


class InputError(ValueError):
    """Raised for any client-side input problem; maps to HTTP 400."""


@dataclass(frozen=True)
class ReconstructionInput:
    intensities: np.ndarray  # shape (n_angles, n_detectors), float64
    dark: np.ndarray         # shape (n_detectors,), float64
    flat: np.ndarray         # shape (n_detectors,), float64
    det_spacing: float       # millimetres per detector element
    center: float            # fractional rotation-centre detector index
    output_size: int         # edge length of the square output image
    pixel_spacing: float     # millimetres per output pixel
    filter_name: str         # "ram-lak" or "hann"


def _finite_float_array(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise InputError(f"{name} must be a NumPy array")
    # allow_pickle=False already refuses most object arrays; defend explicitly
    if value.dtype.kind == "O":
        raise InputError(f"{name} must not be an object array")
    if value.dtype.kind not in "fiu":
        raise InputError(
            f"{name} must contain real (non-complex) numeric data, "
            f"got dtype {value.dtype!r}"
        )
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise InputError(f"{name} contains NaN or infinite values")
    return array


def load_npz(blob: bytes) -> dict[str, np.ndarray]:
    """Load, size-check and validate the arrays stored in an uploaded NPZ.

    ``allow_pickle=False`` guarantees that object arrays cannot be loaded.
    """
    if not isinstance(blob, (bytes, bytearray)) or len(blob) == 0:
        raise InputError("empty upload")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise InputError(
            f"upload too large: {len(blob)} bytes > {MAX_UPLOAD_BYTES} bytes"
        )

    npz = None
    try:
        npz = np.load(io.BytesIO(bytes(blob)), allow_pickle=False)
        names = set(npz.files)
        missing = [key for key in REQUIRED_KEYS if key not in names]
        if missing:
            raise InputError(f"NPZ is missing required array(s): {missing}")

        total_bytes = 0
        arrays: dict[str, np.ndarray] = {}
        for key in npz.files:
            # Object arrays raise on access because allow_pickle=False.
            raw = npz[key]
            total_bytes += int(raw.size) * max(int(raw.dtype.itemsize), 1)
            if total_bytes > max(MAX_DECOMPRESSED_BYTES, 0):
                raise InputError(
                    "decompressed payload exceeds "
                    f"{MAX_DECOMPRESSED_BYTES} bytes"
                )
            arrays[key] = _finite_float_array(key, raw)
    except InputError:
        raise
    except Exception as exc:  # corrupt zip / pickle / object-array failures
        raise InputError(f"cannot read NPZ file: {exc}") from exc
    finally:
        if npz is not None:
            npz.close()

    return arrays


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise InputError(f"{name} must be a finite number")
    return result


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise InputError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be an integer") from exc
    return result


def validate_input(
    arrays: dict[str, np.ndarray],
    *,
    det_spacing: object,
    center: object,
    output_size: object,
    pixel_spacing: object,
    filter_name: object,
) -> ReconstructionInput:
    """Validate shapes, value ranges and scalar parameters together."""
    intensities = arrays["intensities"]
    dark = arrays["dark"]
    flat = arrays["flat"]

    if intensities.ndim != 2:
        raise InputError(
            f"intensities must be 2-D (angles, detectors), got shape "
            f"{intensities.shape}"
        )
    n_angles, n_detectors = intensities.shape
    if not (1 <= n_angles <= MAX_ANGLES):
        raise InputError(
            f"angle count must be in [1, {MAX_ANGLES}], got {n_angles}"
        )
    if not (2 <= n_detectors <= MAX_DETECTORS):
        raise InputError(
            f"detector count must be in [2, {MAX_DETECTORS}], got "
            f"{n_detectors}"
        )
    if dark.shape != (n_detectors,):
        raise InputError(
            f"dark must have shape ({n_detectors},), got {dark.shape}"
        )
    if flat.shape != (n_detectors,):
        raise InputError(
            f"flat must have shape ({n_detectors},), got {flat.shape}"
        )

    det_spacing_f = _strict_float("det_spacing", det_spacing)
    if det_spacing_f <= 0.0:
        raise InputError("det_spacing must be > 0 millimetres")

    pixel_spacing_f = _strict_float("pixel_spacing", pixel_spacing)
    if pixel_spacing_f <= 0.0:
        raise InputError("pixel_spacing must be > 0 millimetres")

    center_f = _strict_float("center", center)

    output_size_i = _strict_int("output_size", output_size)
    if not (1 <= output_size_i <= MAX_OUTPUT_SIZE):
        raise InputError(
            f"output_size must be in [1, {MAX_OUTPUT_SIZE}], got "
            f"{output_size_i}"
        )

    if filter_name not in SUPPORTED_FILTERS:
        raise InputError(
            f"filter must be one of {SUPPORTED_FILTERS}, got {filter_name!r}"
        )

    return ReconstructionInput(
        intensities=np.array(intensities, dtype=np.float64, copy=True),
        dark=np.array(dark, dtype=np.float64, copy=True),
        flat=np.array(flat, dtype=np.float64, copy=True),
        det_spacing=det_spacing_f,
        center=center_f,
        output_size=output_size_i,
        pixel_spacing=pixel_spacing_f,
        filter_name=str(filter_name),
    )

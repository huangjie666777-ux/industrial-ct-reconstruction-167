"""Parallel-beam filtered back-projection implemented from scratch with NumPy.

Geometry convention (see README for details):
* views are equally spaced over 180 degrees: ``theta_a = a * pi / n_angles``,
  starting at 0 with the end-point excluded;
* detector coordinate ``s_j = (j - center) * det_spacing`` in millimetres;
* the reconstructed image has its geometric centre as origin, columns point
  to ``+x`` and rows point to ``+y``; at ``theta = 0`` the ray normal is
  ``+x``, so a ray through image point ``(x, y)`` hits detector position
  ``t = x * cos(theta) + y * sin(theta)``.

Only ``numpy`` and ``numpy.fft`` primitives are used -- no tomography /
reconstruction library and no per-image normalisation.
"""

from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    size = 1
    while size < n:
        size <<= 1
    return size


def ramp_filter(
    padded_length: int,
    det_spacing: float,
    filter_name: str,
) -> np.ndarray:
    """Frequency-domain filter sampled on the padded DFT grid.

    Frequencies use *physical* units (cycles per millimetre) built from the
    real detector spacing, and the FFT length covers at least ``2M - 1`` so
    that linear convolution never wraps around.

    * ``ram-lak``: ``2 * |f|`` up to the detector Nyquist ``1 / (2 d)``;
    * ``hann``: the ramp multiplied by a Hann window
      ``0.5 * (1 + cos(pi * f / f_nyquist))`` over the pass band.

    The factor of 2 belongs to the continuous FBP convention
    ``p_filt(t) = integral 2 |f| P(f) exp(i 2 pi f t) df``; without it the
    reconstruction comes out at exactly half the true attenuation.
    """
    n = padded_length
    # DFT bin index with negative-frequency wrap-around.
    bins = np.arange(n)
    bins = np.where(bins <= n // 2, bins, bins - n)
    frequencies = bins / (n * det_spacing)  # cycles / mm

    f_nyquist = 0.5 / det_spacing
    abs_f = np.abs(frequencies)
    response = np.where(abs_f <= f_nyquist, 2.0 * abs_f, 0.0)

    if filter_name == "hann":
        inside = abs_f <= f_nyquist
        window = 0.5 * (1.0 + np.cos(np.pi * frequencies / f_nyquist))
        response = np.where(inside, response * window, 0.0)
    elif filter_name != "ram-lak":
        raise ValueError(f"unsupported filter: {filter_name!r}")

    return response


def filter_sinogram(
    sinogram: np.ndarray,
    det_spacing: float,
    filter_name: str,
) -> tuple[np.ndarray, int]:
    """Apply the ramp filter with zero-padded FFT convolution.

    Returns the filtered views (cropped back to the detector width) and the
    FFT length used.  Each view is padded to a power of two >= ``2M - 1``.
    """
    n_views, n_detectors = sinogram.shape
    pad_length = _next_pow2(max(2 * n_detectors - 1, 1))

    response = ramp_filter(pad_length, det_spacing, filter_name)

    padded = np.zeros((n_views, pad_length), dtype=np.float64)
    padded[:, :n_detectors] = sinogram

    spectrum = np.fft.fft(padded, axis=1)
    filtered = np.fft.ifft(spectrum * response[np.newaxis, :], axis=1).real
    # Discrete-to-continuous scaling: IFFT gives a bin-sum, multiplying by
    # the sample spacing turns it into an integral over frequency.
    filtered *= det_spacing
    return filtered[:, :n_detectors], pad_length


def _sample_views_zero(
    filtered_views: np.ndarray,
    sample_index: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate every view at fractional detector indices.

    Values outside the detector range are exactly zero (no edge clamping).
    """
    n_detectors = filtered_views.shape[1]
    lower = np.floor(sample_index)
    frac = sample_index - lower
    j0 = lower.astype(np.int64)
    j1 = j0 + 1

    valid0 = (j0 >= 0) & (j0 < n_detectors)
    valid1 = (j1 >= 0) & (j1 < n_detectors)

    j0c = np.clip(j0, 0, n_detectors - 1)
    j1c = np.clip(j1, 0, n_detectors - 1)

    values = (
        np.where(valid0, filtered_views[:, j0c] * (1.0 - frac), 0.0)
        + np.where(valid1, filtered_views[:, j1c] * frac, 0.0)
    )
    return values


def backproject(
    filtered_views: np.ndarray,
    *,
    angles: np.ndarray,
    center: float,
    det_spacing: float,
    output_size: int,
    pixel_spacing: float,
) -> np.ndarray:
    """Back-project filtered views onto a square pixel grid.

    Output indexing is image convention: ``output[row, col]``, with columns
    toward ``+x`` and rows toward ``+y``.  Angular integration uses the
    constant step between the equally spaced views.
    """
    half = (output_size - 1) / 2.0
    cols = np.arange(output_size, dtype=np.float64)
    x = (cols - half) * pixel_spacing          # shape (N,)
    y = (half - cols) * pixel_spacing          # rows: top -> +y

    image = np.zeros((output_size, output_size), dtype=np.float64)
    angle_step = float(np.abs(angles[1] - angles[0])) if angles.size > 1 else 0.0

    for a, theta in enumerate(angles):
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        # detector hit coordinate t for every pixel, then fractional index
        t = y[:, np.newaxis] * sin_t + x[np.newaxis, :] * cos_t
        sample_index = t / det_spacing + center
        sampled = _sample_views_zero(
            filtered_views[a : a + 1, :], sample_index
        )[0]
        image += sampled

    if angles.size > 1:
        image *= angle_step
    return image


def fbp_reconstruct(
    sinogram: np.ndarray,
    *,
    det_spacing: float,
    center: float,
    output_size: int,
    pixel_spacing: float,
    filter_name: str = "ram-lak",
) -> np.ndarray:
    """Full parallel-beam FBP pipeline; attenuation in 1/mm.

    Parameters
    ----------
    sinogram:
        Line integrals with shape ``(n_angles, n_detectors)`` for views
        ``theta = 0, dtheta, ..., pi - dtheta``.
    det_spacing:
        Millimetres between adjacent detector elements.
    center:
        Fractional detector index of the rotation centre.
    output_size:
        Edge length ``N`` of the square reconstruction.
    pixel_spacing:
        Millimetres per output pixel.
    filter_name:
        ``"ram-lak"`` or ``"hann"``.
    """
    sinogram = np.ascontiguousarray(sinogram, dtype=np.float64)
    if sinogram.ndim != 2:
        raise ValueError("sinogram must be 2-D")
    n_angles = sinogram.shape[0]
    if n_angles < 1:
        raise ValueError("at least one view is required")

    angle_step = np.pi / n_angles
    angles = angle_step * np.arange(n_angles, dtype=np.float64)

    filtered, _pad = filter_sinogram(sinogram, det_spacing, filter_name)
    return backproject(
        filtered,
        angles=angles,
        center=center,
        det_spacing=det_spacing,
        output_size=output_size,
        pixel_spacing=pixel_spacing,
    )

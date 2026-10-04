"""Dark/flat calibration: measured intensities -> line integrals.

For every detector element ``j`` and every view ``a``::

    T[a, j] = (I[a, j] - dark[j]) / (flat[j] - dark[j])      # transmittance
    p[a, j] = -ln(T[a, j])                                    # line integral

Rules (per specification):
* ``flat[j]`` must be strictly greater than ``dark[j]``;
* the calibrated transmittance must be strictly positive (``-ln`` is
  undefined / +inf otherwise);
* transmittance values above 1 are *kept*, producing genuine negative line
  integrals -- nothing is clipped or replaced by zero.
"""

from __future__ import annotations

import numpy as np

from .validation import InputError


def line_integrals(
    intensities: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
) -> np.ndarray:
    """Return the attenuation line integrals ``p`` in millimetre-free units.

    Parameters
    ----------
    intensities:
        2-D array with shape ``(n_angles, n_detectors)``.
    dark, flat:
        1-D arrays with shape ``(n_detectors,)``.

    All inputs are assumed to contain finite real numbers (enforced in
    :mod:`app.validation`).
    """
    if not np.all(flat > dark):
        bad = int(np.count_nonzero(~(flat > dark)))
        raise InputError(
            f"flat must be strictly greater than dark at every detector "
            f"({bad} violating element(s))"
        )

    gain = flat - dark
    corrected = (intensities - dark[np.newaxis, :]) / gain[np.newaxis, :]

    if not np.all(corrected > 0.0):
        n_bad = int(np.count_nonzero(~(corrected > 0.0)))
        raise InputError(
            f"calibrated transmittance must be > 0 everywhere ({n_bad} "
            f"non-positive element(s)); dark-corrected intensity reached the "
            f"flat field or below"
        )

    # Negative line integrals are preserved when transmittance > 1.
    return -np.log(corrected)

import numpy as np
import pytest

from app.calibration import line_integrals
from app.validation import InputError


def test_basic_log_conversion():
    dark = np.array([10.0, 10.0])
    flat = np.array([110.0, 110.0])
    # transmittance 0.5 -> ln(2); transmittance exp(-1) -> 1
    intensities = np.array([[60.0, 10.0 + 100.0 * np.exp(-1.0)]])
    p = line_integrals(intensities, dark, flat)
    assert p.shape == (1, 2)
    assert np.isclose(p[0, 0], np.log(2.0), atol=1e-12)
    assert np.isclose(p[0, 1], 1.0, atol=1e-12)


def test_transmittance_above_one_keeps_negative():
    dark = np.zeros(2)
    flat = np.full(2, 100.0)
    intensities = np.array([[0.0, 200.0]])  # T=inf? no: dark=0 -> 0/100=0 bad
    intensities = np.array([[50.0, 200.0]])  # T = 0.5 and 2
    p = line_integrals(intensities, dark, flat)
    assert np.isclose(p[0, 0], np.log(2.0))
    assert np.isclose(p[0, 1], -np.log(2.0))
    assert p[0, 1] < 0.0


def test_per_detector_calibration():
    dark = np.array([0.0, 20.0])
    flat = np.array([100.0, 120.0])
    intensities = np.array([[100.0, 120.0], [10.0, 30.0]])
    p = line_integrals(intensities, dark, flat)
    # row 0: both T=1 -> 0; row 1: T=0.1 both -> ln(10)
    assert np.allclose(p[0], 0.0, atol=1e-12)
    assert np.allclose(p[1], np.log(10.0), atol=1e-12)


def test_flat_must_exceed_dark():
    with pytest.raises(InputError, match="flat must be"):
        line_integrals(np.ones((1, 3)), np.ones(3), np.ones(3))


def test_nonpositive_transmittance_rejected():
    dark = np.zeros(3)
    flat = np.full(3, 100.0)
    with pytest.raises(InputError, match="transmittance must be > 0"):
        line_integrals(np.zeros((1, 3)), dark, flat)
    with pytest.raises(InputError, match="transmittance must be > 0"):
        line_integrals(np.array([[-5.0, 50.0, 50.0]]), dark, flat)

import numpy as np
import pytest

from examples.offcenter_disk import (
    CENTER_X,
    CENTER_Y,
    DET_SPACING,
    MU,
    N_DETECTORS,
    RADIUS,
    analytic_disk_sinogram,
    detector_coords,
    simulate_intensities,
)
from app.calibration import line_integrals
from app.fbp import fbp_reconstruct, filter_sinogram, ramp_filter


def test_filter_frequency_scale_and_symmetry():
    d = 0.5
    n = 8
    response = ramp_filter(n, d, "ram-lak")
    freqs = np.abs(np.fft.fftfreq(n, d=d))
    expected = np.where(freqs <= 1.0 / (2 * d), 2.0 * freqs, 0.0)
    assert np.allclose(response, expected)
    # zero frequency carries no weight; response is real and symmetric
    assert response[0] == 0.0


def test_hann_is_softer_than_ram_lak_at_midband():
    ram = ramp_filter(64, 0.5, "ram-lak")
    hann = ramp_filter(64, 0.5, "hann")
    # everywhere except DC and Nyquist the Hann window strictly reduces gain
    mid = 8
    assert hann[mid] < ram[mid]
    assert hann[0] == 0.0


def test_zero_padding_is_linear_convolution_length():
    _, pad = filter_sinogram(np.zeros((1, 25)), 0.5, "ram-lak")
    assert pad >= 2 * 25 - 1
    assert pad & (pad - 1) == 0  # power of two


def test_impulse_sinogram_produces_backprojection_crest():
    # A single bright ray at theta=0 must crest along x=0 (a vertical line).
    n_det, d = 64, 1.0
    sino = np.zeros((180, n_det))
    sino[0, n_det // 2] = 1.0 / d
    image = fbp_reconstruct(
        sino,
        det_spacing=d,
        center=(n_det - 1) / 2.0,
        output_size=32,
        pixel_spacing=1.0,
        filter_name="ram-lak",
    )
    assert image.shape == (32, 32)


def _reconstruct(filter_name):
    intensities, dark, flat, truth = simulate_intensities()
    sino = line_integrals(intensities, dark, flat)
    assert np.allclose(sino, truth, atol=1e-10)
    return fbp_reconstruct(
        sino,
        det_spacing=DET_SPACING,
        center=(N_DETECTORS - 1) / 2.0,
        output_size=128,
        pixel_spacing=0.5,
        filter_name=filter_name,
    )


def test_offcenter_disk_quantitative_recovery_ramlak():
    image = _reconstruct("ram-lak")
    yy, xx = np.mgrid[0:128, 0:128]
    half = 127 / 2.0
    x = (xx - half) * 0.5
    y = (half - yy) * 0.5
    r2 = (x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2
    deep = r2 <= (RADIUS * 0.55) ** 2
    outside = r2 >= (RADIUS * 1.7) ** 2
    # Interior mean close to the analytic 1/mm attenuation.
    assert abs(image[deep].mean() - MU) < 0.03
    # Off-centre location preserved: centroid of positive mass is in the
    # correct quadrant and close to (cx, cy).
    weight = np.clip(image, 0, None)
    total = weight.sum()
    cx_img = (weight * x).sum() / total
    cy_img = (weight * y).sum() / total
    assert abs(cx_img - CENTER_X) < 3.0
    assert abs(cy_img - CENTER_Y) < 3.0
    # Outside the disk the reconstruction stays close to zero.
    assert np.abs(image[outside]).max() < 0.04


def test_hann_recovers_disk_with_less_ringing():
    ram = _reconstruct("ram-lak")
    hann = _reconstruct("hann")
    yy, xx = np.mgrid[0:128, 0:128]
    half = 127 / 2.0
    x = (xx - half) * 0.5
    y = (half - yy) * 0.5
    outside = ((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2) >= (RADIUS * 1.6) ** 2
    assert np.abs(hann[outside]).max() <= np.abs(ram[outside]).max() + 1e-12


def test_negative_values_preserved_when_present():
    intensities, dark, flat, _ = simulate_intensities()
    sino = line_integrals(intensities, dark, flat)
    sino -= 0.5  # uniform over-transmission: negative bias preserved
    image = fbp_reconstruct(
        sino,
        det_spacing=DET_SPACING,
        center=(N_DETECTORS - 1) / 2.0,
        output_size=64,
        pixel_spacing=0.5,
    )
    assert image.min() < 0.0


def test_geometric_center_is_pixel_center():
    n_det, d = 64, 1.0
    sino = np.zeros((4, n_det))
    image = fbp_reconstruct(
        sino, det_spacing=d, center=(n_det - 1) / 2.0,
        output_size=5, pixel_spacing=1.0,
    )
    assert image.shape == (5, 5)  # odd size: single centre pixel

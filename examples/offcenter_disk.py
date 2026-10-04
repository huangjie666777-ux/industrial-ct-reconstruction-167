"""Analytic projection example: one homogeneous, off-centre disk.

A disk of radius ``R``, attenuation coefficient ``mu`` and centre
``(cx, cy)`` has a known parallel-beam X-ray transform.  A ray with normal
angle ``theta`` and detector coordinate ``s`` travels through the disk along

    L(s, theta) = 2 * sqrt(R^2 - (s - s0(theta))^2)

when ``|s - s0(theta)| < R`` and zero otherwise, with
``s0(theta) = cx cos(theta) + cy sin(theta)`` the projected centre.

The ideal transmitted intensity is ``I = I0 exp(-mu L)``; we also add a
constant dark signal so the calibration step has work to do.  The script
writes such a synthetic NPZ and optionally rebuilds it, letting the FBP
result be compared directly with the analytic ground truth.

Run::

    .venv/bin/python examples/offcenter_disk.py --npz /tmp/disk.npz
    .venv/bin/python examples/offcenter_disk.py --npz /tmp/disk.npz \\
        --reconstruct /tmp/disk_recon.npy
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fbp import fbp_reconstruct
from app.calibration import line_integrals

# Geometry / phantom defaults (millimetres).
N_ANGLES = 360
N_DETECTORS = 256
DET_SPACING = 0.5
RADIUS = 12.0
MU = 0.2                 # attenuation coefficient, 1/mm
CENTER_X = 9.0           # deliberately off-centre disk
CENTER_Y = -6.0
DARK = 50.0
FLAT = 1000.0            # I0 + DARK


def detector_coords(n_detectors: int, det_spacing: float) -> np.ndarray:
    """Detector coordinates with the rotation centre on the central element."""
    center_index = (n_detectors - 1) / 2.0
    return (np.arange(n_detectors, dtype=np.float64) - center_index) * det_spacing


def analytic_disk_sinogram(
    angles: np.ndarray,
    s: np.ndarray,
    *,
    radius: float = RADIUS,
    mu: float = MU,
    cx: float = CENTER_X,
    cy: float = CENTER_Y,
) -> np.ndarray:
    """Exact line-integral sinogram ``mu * chord length`` of an off-centre disk."""
    projected_centre = cx * np.cos(angles)[:, None] + cy * np.sin(angles)[:, None]
    offset = s[None, :] - projected_centre
    inside = np.abs(offset) < radius
    chord = np.where(inside, 2.0 * np.sqrt(np.maximum(radius**2 - offset**2, 0.0)), 0.0)
    return mu * chord


def simulate_intensities(
    n_angles: int = N_ANGLES,
    n_detectors: int = N_DETECTORS,
    det_spacing: float = DET_SPACING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return intensities, dark, flat and the ground-truth sinogram."""
    angles = np.pi * np.arange(n_angles, dtype=np.float64) / n_angles
    s = detector_coords(n_detectors, det_spacing)
    truth = analytic_disk_sinogram(angles, s)

    # Detector samples I = DARK + (FLAT - DARK) * exp(-line integral).
    i0 = FLAT - DARK
    intensities = DARK + i0 * np.exp(-truth)
    dark = np.full(n_detectors, DARK, dtype=np.float64)
    flat = np.full(n_detectors, FLAT, dtype=np.float64)
    return intensities, dark, flat, truth


def save_npz(path: str) -> dict[str, float]:
    intensities, dark, flat, truth = simulate_intensities()
    np.savez(path, intensities=intensities, dark=dark, flat=flat)
    return {
        "path": path,
        "sinogram_min": float(truth.min()),
        "sinogram_max": float(truth.max()),
    }


def rebuild(npz_path: str, out_path: str, filter_name: str = "ram-lak") -> dict[str, float]:
    data = np.load(npz_path)
    sinogram = line_integrals(data["intensities"], data["dark"], data["flat"])
    image = fbp_reconstruct(
        sinogram,
        det_spacing=DET_SPACING,
        center=(N_DETECTORS - 1) / 2.0,
        output_size=128,
        pixel_spacing=0.5,
        filter_name=filter_name,
    )
    np.save(out_path, image, allow_pickle=False)

    yy, xx = np.mgrid[0:128, 0:128]
    half = 127 / 2.0
    x = (xx - half) * 0.5
    y = (half - yy) * 0.5
    inside = (x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2 <= RADIUS**2
    return {
        "recon_path": out_path,
        "recon_inside_mean": float(image[inside].mean()),
        "recon_inside_min": float(image[inside].min()),
        "recon_outside_abs_max": float(np.abs(image[~inside]).max()),
        "truth_mu": MU,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="output NPZ path")
    parser.add_argument("--reconstruct", help="optional reconstruction NPY path")
    parser.add_argument("--filter", default="ram-lak", choices=("ram-lak", "hann"))
    args = parser.parse_args()

    summary = save_npz(args.npz)
    if args.reconstruct:
        summary.update(rebuild(args.npz, args.reconstruct, args.filter))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

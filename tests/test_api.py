import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from examples.offcenter_disk import simulate_intensities, DET_SPACING, N_DETECTORS

client = TestClient(app)


def _npz_upload():
    intensities, dark, flat, _ = simulate_intensities(n_angles=60, n_detectors=128)
    buf = io.BytesIO()
    np.savez(buf, intensities=intensities, dark=dark, flat=flat)
    return buf.getvalue()


FORM = {
    "det_spacing": str(DET_SPACING),
    "center": str((N_DETECTORS - 1) / 2.0),
    "output_size": "64",
    "pixel_spacing": "0.5",
    "filter": "ram-lak",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_reconstruct_returns_zip_with_npy_png_metadata():
    response = client.post(
        "/reconstruct",
        files={"file": ("scan.npz", _npz_upload(), "application/octet-stream")},
        data=FORM,
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(zf.namelist()) == {"reconstruction.npy", "preview.png", "metadata.json"}

    image = np.load(io.BytesIO(zf.read("reconstruction.npy")))
    assert image.shape == (64, 64)
    assert image.dtype == np.float64

    png = Image.open(io.BytesIO(zf.read("preview.png")))
    assert png.format == "PNG"
    assert png.size == (64, 64)

    meta = json.loads(zf.read("metadata.json"))
    assert meta["parameters"]["n_angles"] == 60
    assert meta["parameters"]["filter"] == "ram-lak"
    rng = meta["numeric_ranges"]
    assert set(rng) >= {"recon_min", "recon_max", "sinogram_min", "sinogram_max"}
    assert rng["recon_min"] <= rng["recon_max"]


def test_hann_filter_accepted():
    data = dict(FORM)
    data["filter"] = "hann"
    response = client.post(
        "/reconstruct",
        files={"file": ("scan.npz", _npz_upload(), "application/octet-stream")},
        data=data,
    )
    assert response.status_code == 200, response.text


def test_bad_filter_rejected():
    data = dict(FORM)
    data["filter"] = "cosine"
    response = client.post(
        "/reconstruct",
        files={"file": ("scan.npz", _npz_upload())},
        data=data,
    )
    assert response.status_code == 400
    assert "filter" in response.json()["error"]


def test_calibration_failure_is_http_400():
    intensities, dark, flat, _ = simulate_intensities(n_angles=40, n_detectors=128)
    flat = dark.copy() + 0.0  # flat must exceed dark
    buf = io.BytesIO()
    np.savez(buf, intensities=intensities, dark=dark, flat=flat + 1.0)
    # make some transmittances non-positive: inflate dark above intensity
    dark2 = np.full_like(dark, 2000.0)
    buf = io.BytesIO()
    np.savez(buf, intensities=intensities, dark=dark2, flat=np.full_like(flat, 3000.0))
    response = client.post(
        "/reconstruct",
        files={"file": ("scan.npz", buf.getvalue())},
        data=FORM,
    )
    assert response.status_code == 400
    assert "transmittance" in response.json()["error"]


def test_oversized_output_rejected():
    data = dict(FORM)
    data["output_size"] = "257"
    response = client.post(
        "/reconstruct",
        files={"file": ("scan.npz", _npz_upload())},
        data=data,
    )
    assert response.status_code == 400


def test_garbage_upload_rejected():
    response = client.post(
        "/reconstruct",
        files={"file": ("scan.npz", b"hello")},
        data=FORM,
    )
    assert response.status_code == 400

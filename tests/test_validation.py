import io

import numpy as np
import pytest

from app.validation import (
    MAX_ANGLES,
    MAX_DECOMPRESSED_BYTES,
    MAX_DETECTORS,
    MAX_OUTPUT_SIZE,
    InputError,
    load_npz,
    validate_input,
)


def make_arrays(n_angles=8, n_detectors=16):
    return {
        "intensities": np.full((n_angles, n_detectors), 500.0),
        "dark": np.zeros(n_detectors),
        "flat": np.full(n_detectors, 1000.0),
    }


def npz_bytes(**arrays):
    return io.BytesIO()  # placeholder, replaced below


def _npz(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


VALID = dict(
    det_spacing=0.5,
    center=7.5,
    output_size=32,
    pixel_spacing=0.5,
    filter_name="ram-lak",
)


def test_valid_roundtrip():
    arrays = load_npz(_npz(**make_arrays()))
    data = validate_input(arrays, **VALID)
    assert data.intensities.shape == (8, 16)
    assert data.intensities.dtype == np.float64


def test_missing_key_rejected():
    arrays = make_arrays()
    del arrays["flat"]
    with pytest.raises(InputError, match="missing required"):
        load_npz(_npz(**arrays))


def test_shape_mismatch_rejected():
    arrays = make_arrays()
    arrays["dark"] = np.zeros(15)
    with pytest.raises(InputError, match="dark must have shape"):
        validate_input(load_npz(_npz(**arrays)), **VALID)


def test_nonfinite_rejected():
    arrays = make_arrays()
    arrays["flat"][3] = np.nan
    with pytest.raises(InputError, match="finite"):
        load_npz(_npz(**arrays))
    arrays = make_arrays()
    arrays["intensities"][0, 0] = np.inf
    with pytest.raises(InputError, match="finite"):
        load_npz(_npz(**arrays))


def test_object_array_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    obj = np.empty(2, dtype=object)
    obj[0] = make_arrays()["intensities"]
    obj[1] = "x"
    np.savez(path, intensities=obj, dark=np.zeros(16), flat=np.ones(16))
    with pytest.raises(InputError, match="(?i)object"):
        load_npz(path.read_bytes())


def test_dimension_limits():
    arrays = make_arrays(n_angles=MAX_ANGLES + 1, n_detectors=16)
    with pytest.raises(InputError, match="angle count"):
        validate_input(load_npz(_npz(**arrays)), **VALID)

    arrays = make_arrays(n_angles=8, n_detectors=MAX_DETECTORS + 1)
    with pytest.raises(InputError, match="detector count"):
        validate_input(load_npz(_npz(**arrays)), **VALID)

    params = dict(VALID)
    params["output_size"] = MAX_OUTPUT_SIZE + 1
    with pytest.raises(InputError, match="output_size"):
        validate_input(load_npz(_npz(**make_arrays())), **params)


def test_bad_scalar_parameters():
    for key, value in {
        "det_spacing": 0.0,
        "pixel_spacing": -1.0,
        "output_size": 1.5,
        "filter_name": "shepp",
    }.items():
        params = dict(VALID)
        params[key] = value
        with pytest.raises(InputError):
            validate_input(load_npz(_npz(**make_arrays())), **params)


def test_compressed_size_guard(monkeypatch):
    from app import validation

    monkeypatch.setattr(validation, "MAX_DECOMPRESSED_BYTES", 10)
    with pytest.raises(InputError, match="decompressed"):
        load_npz(_npz(**make_arrays()))


def test_truncated_upload_rejected():
    with pytest.raises(InputError, match="cannot read NPZ"):
        load_npz(b"not an npz file at all")
    with pytest.raises(InputError, match="empty upload"):
        load_npz(b"")


def test_complex_dtype_rejected():
    arrays = make_arrays()
    arrays["flat"] = np.ones(16, dtype=np.complex128)
    with pytest.raises(InputError, match="non-complex"):
        load_npz(_npz(**arrays))


def test_decompressed_constant_exists():
    assert MAX_DECOMPRESSED_BYTES > 0

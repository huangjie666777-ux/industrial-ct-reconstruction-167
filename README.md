# Parallel-beam CT reconstruction backend

A from-scratch, pure-backend parallel-beam CT reconstructor using
**Python 3.10**, **FastAPI 0.115.12** and **NumPy 2.2.6** (Pillow is used
only to encode the PNG preview). Dark/flat calibration, filtered
back-projection (FBP) and HTTP delivery live in separate modules and are
wired together by a shared pipeline.

```
app/
  validation.py   # NPZ loading, size guards, shape/value/parameter checks
  calibration.py  # per-detector dark/flat correction -> line integrals
  fbp.py          # NumPy FFT ramp filtering + linear back-projection
  pipeline.py     # calibration + FBP composition, NPY encoding, statistics
  preview.py      # min/max stretched PNG preview (NPY is never modified)
  main.py         # FastAPI app: POST /reconstruct -> ZIP download
examples/
  offcenter_disk.py  # analytic off-centre disk projection example
tests/            # pytest suite (validation, calibration, FBP, HTTP)
```

## Input format

`POST /reconstruct` is a `multipart/form-data` request:

| field | type | meaning |
| --- | --- | --- |
| `file` | NPZ upload | scan data (below) |
| `det_spacing` | float > 0 | detector element spacing in millimetres |
| `center` | finite float | fractional detector index of the rotation centre |
| `output_size` | int in [1, 256] | edge length of the square image |
| `pixel_spacing` | float > 0 | output pixel spacing in millimetres |
| `filter` | `ram-lak` or `hann` | ramp filter (default `ram-lak`) |

The NPZ (`numpy.savez` / `savez_compressed`, opened with
`allow_pickle=False`) must contain:

* `intensities`: 2-D array, shape `(n_angles, n_detectors)`, real numeric;
  up to **360 angles** and **512 detectors**;
* `dark`, `flat`: 1-D arrays, both exactly `n_detectors` wide;
* all entries must be finite (no `NaN`/`inf`); object arrays are rejected.

Uploads are capped at 16 MiB compressed and 64 MiB decompressed, and every
scalar parameter is validated; any failure returns HTTP 400 with a JSON
`error` message.

## Geometry and angular range

Views are **equally spaced over 180 degrees**, starting at 0 with the
end-point excluded:

```
theta_a = a * pi / n_angles,   a = 0, 1, ..., n_angles - 1
```

The image is indexed as `image[row, col]` with the **geometric centre of
the image as origin**, columns toward **+x**, rows toward **+y**. At
`theta = 0` the ray normal points along **+x**. Detector coordinates are

```
s_j = (j - center_index) * det_spacing     # millimetres
```

and a pixel at `(x, y)` is sampled at
`t = x cos(theta) + y sin(theta)`. Linear interpolation is used on
sampling and rays that fall outside the detector contribute exactly zero.
Back-projection integrates with the constant angular step `pi/n_angles`,
so the output is an **attenuation coefficient in 1/mm** (not normalised
per image).

## Calibration

For each detector element independently:

```
T = (I - dark) / (flat - dark)
line integral p = -ln(T)
```

* `flat` must be strictly greater than `dark` everywhere;
* calibrated transmittance must be strictly positive (otherwise `-ln` is
  undefined) and the request is rejected with HTTP 400;
* transmittance above 1 is **kept**, yielding genuine **negative line
  integrals** — no clipping and bad values are never replaced by zero.

## FBP implementation

* Each view is zero-padded to a power of two `>= 2M - 1`, so the FFT
  performs a linear (non-circular) convolution.
* Frequencies are physical: bin `k` maps to `k / (N_pad * det_spacing)`
  cycles/mm, with the detector Nyquist at `1/(2 det_spacing)`.
* Ram-Lak response is `2|f|` inside the pass band (the factor 2 belongs to
  the continuous FBP convention); Hann multiplies the ramp by
  `0.5(1 + cos(pi f/f_Nyquist))`.
* Back-projection interpolates filtered views linearly; out-of-range
  samples are zero. Negative reconstruction values are preserved; there is
  no clipping or per-image normalisation.

No tomography/reconstruction library is used — only `numpy.fft`
primitives — and no existing reconstruction function is called.

## Response format

`200 OK` returns `application/zip` (`reconstruction.zip`) containing:

* `reconstruction.npy` — float64 2-D array (`np.save`, shape
  `(output_size, output_size)`), attenuation in 1/mm, negatives preserved;
* `preview.png` — 8-bit grayscale preview with a min/max stretch that
  affects **only the PNG**;
* `metadata.json` — echoed parameters, units and honest numeric ranges
  (`recon_min/max/mean/std`, `sinogram_min/max`).

## Run

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Analytic example (off-centre disk)

A homogeneous disk has the exact X-ray transform
`p(s) = 2 mu sqrt(R^2 - (s-s0)^2)` inside the disk and zero outside. The
example simulates `I = dark + (flat-dark) exp(-p)` for a deliberately
off-centre disk, writes the NPZ and can rebuild it:

```bash
.venv/bin/python examples/offcenter_disk.py \
    --npz /tmp/disk.npz --reconstruct /tmp/disk_recon.npy
```

The interior mean is close to the analytic `mu = 0.2 /mm` even though the
disk is off-centre.

### curl demo

```bash
curl -X POST http://127.0.0.1:8000/reconstruct \
    -F 'file=@/tmp/disk.npz' \
    -F 'det_spacing=0.5' -F 'center=127.5' \
    -F 'output_size=128' -F 'pixel_spacing=0.5' \
    -F 'filter=ram-lak' -o reconstruction.zip
```

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app examples tests
```

The suite covers input rejection (object arrays, bad shapes, non-finite
values, limits, illegal scalars, decompression guard), calibration
(including preserved negative integrals), ramp/Hann filter behaviour,
quantitative recovery of the analytic disk, geometry conventions, and the
full HTTP ZIP/NPY/PNG path via FastAPI's test client.


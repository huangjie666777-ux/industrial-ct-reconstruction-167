"""FastAPI application: upload an NPZ scan, download an FBP reconstruction.

Endpoints
---------
``GET  /health`` -- liveness probe.
``POST /reconstruct`` -- multipart form:
    * ``file``: NPZ with ``intensities`` (n_angles x n_detectors), ``dark``
      and ``flat`` (n_detectors,);
    * ``det_spacing`` (float, mm), ``center`` (float detector index),
      ``output_size`` (int <= 256), ``pixel_spacing`` (float, mm),
      ``filter`` (``ram-lak`` or ``hann``).

Responds with ``application/zip`` containing ``reconstruction.npy``,
``preview.png`` and ``metadata.json`` (echoed parameters plus value ranges).
"""

from __future__ import annotations

import json
import io
import zipfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .calibration import line_integrals
from .fbp import fbp_reconstruct
from .pipeline import encode_npy, numeric_ranges
from .preview import render_png
from .validation import InputError, load_npz, validate_input

app = FastAPI(
    title="Parallel-beam CT reconstruction",
    version="1.0.0",
    description="Dark/flat calibration plus from-scratch NumPy FBP.",
)


@app.exception_handler(InputError)
async def input_error_handler(_request, exc: InputError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(..., description="NPZ scan archive"),
    det_spacing: float = Form(..., description="mm per detector element"),
    center: float = Form(..., description="fractional rotation-centre index"),
    output_size: int = Form(..., description="square output edge length"),
    pixel_spacing: float = Form(..., description="mm per output pixel"),
    filter: str = Form("ram-lak", description="ram-lak or hann"),
) -> Response:
    blob = await file.read()
    arrays = load_npz(blob)
    data = validate_input(
        arrays,
        det_spacing=det_spacing,
        center=center,
        output_size=output_size,
        pixel_spacing=pixel_spacing,
        filter_name=filter,
    )

    sinogram = line_integrals(data.intensities, data.dark, data.flat)
    image = fbp_reconstruct(
        sinogram,
        det_spacing=data.det_spacing,
        center=data.center,
        output_size=data.output_size,
        pixel_spacing=data.pixel_spacing,
        filter_name=data.filter_name,
    )

    npy_bytes = encode_npy(image)
    png_bytes = render_png(image)

    metadata = {
        "parameters": {
            "n_angles": int(data.intensities.shape[0]),
            "n_detectors": int(data.intensities.shape[1]),
            "det_spacing_mm": data.det_spacing,
            "center_index": data.center,
            "output_size": data.output_size,
            "pixel_spacing_mm": data.pixel_spacing,
            "filter": data.filter_name,
            "angles_deg_start": 0.0,
            "angles_deg_stop_exclusive": 180.0,
        },
        "units": {
            "reconstruction": "attenuation coefficient, 1/mm",
            "geometry": "millimetres",
        },
        "numeric_ranges": numeric_ranges(image, sinogram),
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reconstruction.npy", npy_bytes)
        zf.writestr("preview.png", png_bytes)
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="reconstruction.zip"'
        },
    )

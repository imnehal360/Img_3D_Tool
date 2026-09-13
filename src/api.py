from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import run_single_image_pipeline


_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def create_app(
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    generator: Any | None = None,
    output_root: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Image-to-3D GLB API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.environ.get("IMG3D_CORS_ORIGINS", "*").split(",")],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    root = Path(output_root or os.environ.get("IMG3D_OUTPUT_DIR", "outputs/api"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_class=FileResponse)
    async def generate_model(
        image: UploadFile | None = File(None),
        file: UploadFile | None = File(None),
    ) -> FileResponse:
        image = image or file
        if image is None:
            raise HTTPException(
                status_code=422,
                detail="Provide an image upload using form field 'image' or 'file'.",
            )
        if image.content_type not in _ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WEBP image.")

        job_dir = root / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"source{_EXTENSIONS[image.content_type]}"

        try:
            with input_path.open("wb") as destination:
                shutil.copyfileobj(image.file, destination)
            result = run_single_image_pipeline(input_path, job_dir, config=config, generator=generator)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            await image.close()

        return FileResponse(path=result.model_path, media_type="model/gltf-binary", filename="model.glb")

    return app


app = create_app()

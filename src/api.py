from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

    @app.get("/examples")
    def list_examples() -> dict[str, list[dict[str, Any]]]:
        examples = []
        if root.exists():
            for job_dir in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                model_path = job_dir / "model.glb"
                if not job_dir.is_dir() or not model_path.exists():
                    continue
                input_path = next(
                    (candidate for candidate in job_dir.glob("source.*") if candidate.is_file()),
                    job_dir / "input.png",
                )
                if not input_path.exists():
                    input_path = job_dir / "processed.png"
                if not input_path.exists():
                    continue
                example_id = job_dir.name
                meshy_path = next((candidate for candidate in (
                    job_dir / "meshy.glb", job_dir / "meshy_model.glb", job_dir / "meshy" / "model.glb"
                ) if candidate.is_file()), None)
                meshy_image_path = next((candidate for candidate in (
                    job_dir / "meshy.png", job_dir / "meshy.jpg", job_dir / "meshy.jpeg", job_dir / "meshy.webp"
                ) if candidate.is_file()), None)
                item: dict[str, Any] = {
                    "id": example_id,
                    "title": f"Generated example {len(examples) + 1}",
                    "input_url": f"/examples/{example_id}/input",
                    "system_model_url": f"/examples/{example_id}/model.glb",
                    "model_url": f"/examples/{example_id}/model.glb",
                    "meshy_model_url": f"/examples/{example_id}/meshy.glb" if meshy_path else None,
                    "meshy_image_url": f"/examples/{example_id}/meshy-preview" if meshy_image_path else None,
                }
                metrics_path = job_dir / "metrics.json"
                if metrics_path.is_file():
                    try:
                        item["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
                examples.append(item)
        return {"examples": examples}

    def _example_file(example_id: str, filename: str) -> Path:
        if Path(example_id).name != example_id or not example_id:
            raise HTTPException(status_code=400, detail="Invalid example id.")
        path = root / example_id / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Example file not found.")
        return path

    @app.get("/examples/{example_id}/model.glb", response_class=FileResponse)
    def example_model(example_id: str) -> FileResponse:
        return FileResponse(_example_file(example_id, "model.glb"), media_type="model/gltf-binary", filename="model.glb")

    @app.get("/examples/{example_id}/meshy.glb", response_class=FileResponse)
    def example_meshy_model(example_id: str) -> FileResponse:
        job_dir = root / example_id
        if Path(example_id).name != example_id or not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Example not found.")
        path = next((candidate for candidate in (
            job_dir / "meshy.glb", job_dir / "meshy_model.glb", job_dir / "meshy" / "model.glb"
        ) if candidate.is_file()), None)
        if path is None:
            raise HTTPException(status_code=404, detail="Meshy model not found for this example.")
        return FileResponse(path, media_type="model/gltf-binary", filename="meshy.glb")

    @app.get("/examples/{example_id}/meshy-preview", response_class=FileResponse)
    def example_meshy_preview(example_id: str) -> FileResponse:
        job_dir = root / example_id
        if Path(example_id).name != example_id or not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Example not found.")
        path = next((candidate for candidate in (
            job_dir / "meshy.png", job_dir / "meshy.jpg", job_dir / "meshy.jpeg", job_dir / "meshy.webp"
        ) if candidate.is_file()), None)
        if path is None:
            raise HTTPException(status_code=404, detail="Meshy preview not found for this example.")
        return FileResponse(path)

    @app.post("/examples/{example_id}/meshy-preview")
    async def upload_meshy_preview(
        example_id: str,
        image: UploadFile = File(...),
    ) -> dict[str, str]:
        job_dir = root / example_id
        if Path(example_id).name != example_id or not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Example not found.")
        if image.content_type not in _ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WEBP image.")

        destination = job_dir / "meshy.png"
        try:
            with destination.open("wb") as output:
                shutil.copyfileobj(image.file, output)
        finally:
            await image.close()

        return {
            "message": "Meshy screenshot saved.",
            "meshy_image_url": f"/examples/{example_id}/meshy-preview",
        }

    @app.get("/examples/{example_id}/input", response_class=FileResponse)
    def example_input(example_id: str) -> FileResponse:
        job_dir = root / example_id
        if Path(example_id).name != example_id or not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Example not found.")
        input_path = next((candidate for candidate in job_dir.glob("source.*") if candidate.is_file()), None)
        input_path = input_path or (job_dir / "input.png")
        input_path = input_path if input_path.exists() else (job_dir / "processed.png")
        if not input_path.is_file():
            raise HTTPException(status_code=404, detail="Example input not found.")
        return FileResponse(input_path)

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

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()

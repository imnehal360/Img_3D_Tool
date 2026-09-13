from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

from typing import Any

import numpy as np
import trimesh
from PIL import Image


@dataclass
class MeshStats:
    valid: bool
    vertices: int
    faces: int
    components: int | None
    bounding_box: list[float] | None
    file_size_mb: float
    texture_count: int
    texture_resolutions: list[list[int]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_glb(path: str | Path) -> trimesh.Scene | trimesh.Trimesh:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"GLB not found: {model_path}")
    if model_path.suffix.lower() != ".glb":
        raise ValueError(f"Expected .glb output, got {model_path.suffix}")
    return trimesh.load(model_path, force="scene")


def apply_neutral_grayscale(path: str | Path) -> Path:
    """Apply a light matte clay material while preserving geometry and alpha."""
    model_path = Path(path)
    scene_or_mesh = load_glb(model_path)
    meshes = _extract_meshes(scene_or_mesh)

    for mesh in meshes:
        material = getattr(mesh.visual, "material", None)
        image = (getattr(material, "baseColorTexture", None) or getattr(material, "image", None)) if material is not None else None
        if image is not None:
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            clay = Image.new("RGBA" if alpha is not None else "RGB", image.size, (190, 190, 190, 255) if alpha is not None else (190, 190, 190))
            if alpha is not None:
                clay.putalpha(alpha)
            if hasattr(material, "baseColorTexture"):
                material.baseColorTexture = clay
            else:
                material.image = clay

        colors = np.asarray(getattr(mesh.visual, "vertex_colors", []))
        if colors.ndim == 2 and colors.shape[1] >= 3:
            luminance = np.rint(
                0.2126 * colors[:, 0]
                + 0.7152 * colors[:, 1]
                + 0.0722 * colors[:, 2]
            ).astype(colors.dtype)
            colors = colors.copy()
            colors[:, :3] = luminance[:, None]
            mesh.visual.vertex_colors = colors

        if material is not None:
            if hasattr(material, "roughnessFactor"):
                material.roughnessFactor = 1.0
            if hasattr(material, "metallicFactor"):
                material.metallicFactor = 0.0

        if material is not None and image is None:
            factor = getattr(material, "baseColorFactor", None)
            if factor is not None and len(factor) >= 3:
                factor = list(factor)
                luminance = 0.2126 * factor[0] + 0.7152 * factor[1] + 0.0722 * factor[2]
                factor[:3] = [luminance] * 3
                material.baseColorFactor = factor

    scene_or_mesh.export(model_path)
    return model_path

def get_mesh_stats(path: str | Path) -> MeshStats:
    model_path = Path(path)
    try:
        scene_or_mesh = load_glb(model_path)
        meshes = _extract_meshes(scene_or_mesh)
        if not meshes:
            return _invalid(model_path, "No mesh geometry found.")

        vertices = 0
        faces = 0
        bounds = []
        texture_resolutions: list[list[int]] = []
        components = 0

        for mesh in meshes:
            if mesh.vertices.size == 0 or mesh.faces.size == 0:
                continue
            if not np.isfinite(mesh.vertices).all():
                return _invalid(model_path, "Mesh contains NaN or infinite vertex coordinates.")
            vertices += int(len(mesh.vertices))
            faces += int(len(mesh.faces))
            bounds.append(mesh.bounds)
            components += _component_count(mesh)
            texture_resolutions.extend(_texture_resolutions(mesh))

        if vertices == 0 or faces == 0:
            return _invalid(model_path, "Mesh has no vertices or faces.")

        merged_bounds = np.array(bounds)
        minimum = merged_bounds[:, 0, :].min(axis=0)
        maximum = merged_bounds[:, 1, :].max(axis=0)
        bbox = [round(float(value), 6) for value in (maximum - minimum).tolist()]

        return MeshStats(
            valid=True,
            vertices=vertices,
            faces=faces,
            components=components,
            bounding_box=bbox,
            file_size_mb=round(model_path.stat().st_size / (1024 * 1024), 4),
            texture_count=len(texture_resolutions),
            texture_resolutions=texture_resolutions,
        )
    except Exception as exc:
        return _invalid(model_path, str(exc))


def _extract_meshes(scene_or_mesh: trimesh.Scene | trimesh.Trimesh) -> list[trimesh.Trimesh]:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return [scene_or_mesh]
    return [geometry for geometry in scene_or_mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]


def _component_count(mesh: trimesh.Trimesh) -> int:
    try:
        return int(len(mesh.split(only_watertight=False)))
    except Exception:
        return 1


def _texture_resolutions(mesh: trimesh.Trimesh) -> list[list[int]]:
    material = getattr(mesh.visual, "material", None)
    images = []
    for attr in ("image", "baseColorTexture", "metallicRoughnessTexture", "normalTexture"):
        image = getattr(material, attr, None)
        if image is not None and hasattr(image, "size"):
            images.append([int(image.size[0]), int(image.size[1])])
    return images


def _invalid(model_path: Path, error: str) -> MeshStats:
    size = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else math.nan
    return MeshStats(
        valid=False,
        vertices=0,
        faces=0,
        components=None,
        bounding_box=None,
        file_size_mb=round(size, 4) if math.isfinite(size) else 0.0,
        texture_count=0,
        texture_resolutions=[],
        error=error,
    )

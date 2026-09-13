# Image-to-3D Generation

This repository implements the first P0 slice from the SRS: image input, preprocessing, open-source image-to-3D generation, GLB export, a basic preview, and measured metrics.

## Selected Model

Current selected model: **pretrained original TRELLIS**.

Why: the project now targets Microsoft's pretrained `microsoft/TRELLIS-image-large` model and exports GLB through the official TRELLIS postprocessing utilities.

Important caveat: original TRELLIS requires a CUDA GPU with roughly 16 GB VRAM or more. Colab A100 is safer than T4. If the selected Colab runtime cannot realistically run TRELLIS, stop and document the blocker rather than fabricating outputs.

## Current Milestone

Implemented:

- image validation and preprocessing
- optional background removal hook
- original TRELLIS generator wrapper
- GPU/environment capture
- GLB validation and mesh statistics
- HTML preview for Colab
- metrics JSON output
- reproducible Colab notebook skeleton

Those should come after the single-image pipeline has been run successfully on a compatible Colab GPU.

## Colab Usage

1. Open `notebooks/Image_to_3D.ipynb` in Google Colab.
2. Edit `PROJECT_REPO_URL` to point to your GitHub repository.
3. Enable a CUDA GPU, ideally A100 or another runtime with at least 16 GB VRAM.
4. Run cells from top to bottom.
5. Upload a PNG/JPG/WEBP image with one clear object.

The output directory will contain:

```text
input.png
processed.png
model.glb
preview.html
metrics.json
```

## Local Smoke Test

Local tests verify preprocessing, GLB stats, metrics serialization, and the pipeline wrapper with a synthetic test generator:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
```



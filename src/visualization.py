from __future__ import annotations

from pathlib import Path
import base64


def write_basic_glb_preview(model_path: str | Path, output_path: str | Path) -> Path:
    model = Path(model_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_data = base64.b64encode(model.read_bytes()).decode("ascii")
    model_uri = f"data:model/gltf-binary;base64,{model_data}"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js" onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/@google/model-viewer/dist/model-viewer.min.js';"></script>
  <style>
    html, body {{ margin: 0; height: 100%; }}
    model-viewer {{
      width: 100%;
      height: 100vh;
      background: #151515;
      filter: grayscale(1) contrast(1.08);
    }}
  </style>
</head>
<body>
  <model-viewer
    src="{model_uri}"
    camera-controls
    auto-rotate
    shadow-intensity="1.35"
    shadow-softness="0.75"
    exposure="0.9"
    tone-mapping="neutral"
  ></model-viewer>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out

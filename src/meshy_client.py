from __future__ import annotations

import os


def require_meshy_api_key() -> str:
    key = os.getenv("MESHY_API_KEY")
    if not key:
        raise RuntimeError("MESHY_API_KEY is not set. Use an environment variable or Colab secret.")
    return key


class MeshyClient:
    """Placeholder for the later benchmark milestone; no web scraping or fake results."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or require_meshy_api_key()

    def generate_from_image(self, image_path: str):
        raise NotImplementedError("Meshy API integration is planned after the P0 single-image pipeline works.")

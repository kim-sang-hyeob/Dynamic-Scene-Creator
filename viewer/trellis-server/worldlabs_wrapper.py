"""
World Labs Marble API Wrapper
Handles text/image → 3D Map generation via World Labs API.
Downloads SPZ output and converts to .splat format using existing converter.

API flow:
  1. POST /marble/v1/worlds:generate → generation_id
  2. Poll GET /marble/v1/worlds/{id} until state=complete
  3. Download SPZ from output URL
  4. Convert SPZ → .splat via convert_spz_to_splat.py
"""

import base64
import logging
import os
import sys
import tempfile
import time

import httpx

# Import existing SPZ→.splat converter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from convert_spz_to_splat import process_spz_to_splat_vectorized

logger = logging.getLogger(__name__)

POLL_INTERVAL = 2.0   # seconds between status polls
POLL_TIMEOUT = 120.0   # max seconds to wait for generation


class WorldLabsGenerator:
    API_BASE = "https://api.worldlabs.ai"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client(
            timeout=60,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        )
        logger.info("WorldLabsGenerator initialized")

    def generate_map(self, prompt_type: str, prompt_data: str, seed: int = -1) -> dict:
        """
        Generate a 3D map from text or image prompt.

        Args:
            prompt_type: "text" or "image"
            prompt_data: text string or base64-encoded image
            seed: random seed (-1 for random)

        Returns:
            dict with keys: splat_bytes, gaussian_count, generation_time
        """
        start_time = time.time()

        # 1. Submit generation request
        generation_id = self._submit(prompt_type, prompt_data, seed)
        logger.info(f"World Labs generation submitted: {generation_id}")

        # 2. Poll until complete
        result = self._poll(generation_id)
        logger.info(f"World Labs generation complete: {generation_id}")

        # 3. Download SPZ
        spz_url = result["output"]["spz_url"]
        spz_bytes = self._download(spz_url)
        logger.info(f"Downloaded SPZ: {len(spz_bytes)} bytes")

        # 4. Convert SPZ → .splat
        splat_bytes = self._spz_bytes_to_splat(spz_bytes)
        gaussian_count = len(splat_bytes) // 32

        generation_time = time.time() - start_time
        logger.info(f"Converted to .splat: {gaussian_count} gaussians in {generation_time:.1f}s")

        return {
            "splat_bytes": splat_bytes,
            "gaussian_count": gaussian_count,
            "generation_time": generation_time,
        }

    def _submit(self, prompt_type: str, prompt_data: str, seed: int) -> str:
        """Submit generation request, return generation_id."""
        prompt = {}
        if prompt_type == "text":
            prompt["text"] = prompt_data
        else:
            # Image: base64 → data URI
            prompt["image_uri"] = f"data:image/png;base64,{prompt_data}"

        body = {"prompt": prompt}
        if seed >= 0:
            body["seed"] = seed

        resp = self.client.post(
            f"{self.API_BASE}/marble/v1/worlds:generate",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["generation_id"]

    def _poll(self, generation_id: str) -> dict:
        """Poll until generation is complete or failed."""
        deadline = time.time() + POLL_TIMEOUT

        while time.time() < deadline:
            resp = self.client.get(
                f"{self.API_BASE}/marble/v1/worlds/{generation_id}",
            )
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state", "unknown")

            if state == "complete":
                return data
            elif state == "failed":
                error_msg = data.get("error", "Generation failed")
                raise RuntimeError(f"World Labs generation failed: {error_msg}")

            logger.debug(f"Polling {generation_id}: state={state}")
            time.sleep(POLL_INTERVAL)

        raise TimeoutError(f"World Labs generation timed out after {POLL_TIMEOUT}s")

    def _download(self, url: str) -> bytes:
        """Download SPZ file from URL."""
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.content

    def _spz_bytes_to_splat(self, spz_bytes: bytes) -> bytes:
        """Convert SPZ bytes to .splat format using existing converter."""
        with tempfile.NamedTemporaryFile(suffix='.spz', delete=True) as f:
            f.write(spz_bytes)
            f.flush()
            return process_spz_to_splat_vectorized(f.name)

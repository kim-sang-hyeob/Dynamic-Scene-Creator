#!/usr/bin/env python3
"""
Scene Composer — Static file server + TRELLIS API proxy + World Labs API.
Serves files with correct MIME types for ES modules.
Proxies /api/* requests to TRELLIS server (localhost:8000).
Handles /api/generate-map directly (no TRELLIS dependency).

Usage:
    WORLDLABS_API_KEY=xxx python3 server.py [--port 8080]
"""

import os
import sys
import json
import base64
import argparse
import tempfile
import threading
import time
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

PORT = 8080
TRELLIS_URL = "http://localhost:8000"

# World Labs API
WORLDLABS_API_BASE = "https://api.worldlabs.ai"
WORLDLABS_API_KEY = os.environ.get("WORLDLABS_API_KEY", "")
WORLDLABS_POLL_INTERVAL = 2.0
WORLDLABS_POLL_TIMEOUT = 900.0  # 15min — generation typically 5-7min, sometimes longer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("scene-composer")

# Import SPZ→.splat converter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
try:
    from convert_spz_to_splat import process_spz_to_splat_vectorized
    HAS_SPZ = True
except ImportError:
    HAS_SPZ = False
    logger.warning("spz library not available — World Labs map generation disabled")


class ComposerHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.json': 'application/json',
        '.wasm': 'application/wasm',
        '.splat': 'application/octet-stream',
        '.splatv': 'application/octet-stream',
        '.ply': 'application/octet-stream',
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy("GET")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/generate-map":
            self._handle_generate_map()
        elif self.path.startswith("/api/"):
            self._proxy("POST")
        else:
            self.send_error(405, "POST not allowed for static files")

    # ── World Labs: /api/generate-map ────────────────────────────────

    def _handle_generate_map(self):
        """Handle map generation via World Labs API directly."""
        if not WORLDLABS_API_KEY:
            self._json_error(500, "WORLDLABS_API_KEY not set. Start server with: WORLDLABS_API_KEY=xxx python3 server.py")
            return
        if not HAS_SPZ:
            self._json_error(500, "spz library not installed. Run: pip install spz")
            return

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

        prompt_text = body.get("prompt_text", "")
        prompt_image = body.get("prompt_image", "")  # base64
        prompt_type = body.get("prompt_type", "text")
        seed = body.get("seed", -1)

        if not prompt_text and not prompt_image:
            self._json_error(400, "Text prompt or image is required")
            return

        try:
            start_time = time.time()

            # Upload image as media asset if provided
            media_asset_id = None
            if prompt_image and prompt_type == "image":
                media_asset_id = self._wl_upload_image(prompt_image)
                logger.info(f"Uploaded image as media asset: {media_asset_id}")

            # 1. Submit generation request
            generation_id = self._wl_submit(
                "image" if media_asset_id else "text",
                prompt_text,
                seed,
                media_asset_id=media_asset_id,
            )
            logger.info(f"World Labs submitted: {generation_id}")

            # 2. Poll until complete
            result = self._wl_poll(generation_id)
            logger.info(f"World Labs complete: {generation_id}")

            # 3. Download SPZ (use full_res, fallback to 500k, 100k)
            spz_urls = result["assets"]["splats"]["spz_urls"]
            spz_url = spz_urls.get("full_res") or spz_urls.get("500k") or spz_urls.get("100k")
            if not spz_url:
                raise RuntimeError("No SPZ URL in response")
            spz_bytes = self._wl_download(spz_url)
            logger.info(f"Downloaded SPZ: {len(spz_bytes)} bytes")

            # 4. Convert SPZ → .splat
            with tempfile.NamedTemporaryFile(suffix='.spz', delete=True) as f:
                f.write(spz_bytes)
                f.flush()
                splat_bytes = process_spz_to_splat_vectorized(f.name)

            gaussian_count = len(splat_bytes) // 32
            generation_time = time.time() - start_time
            logger.info(f"Converted to .splat: {gaussian_count} gaussians in {generation_time:.1f}s")

            # 5. Respond
            response = json.dumps({
                "status": "success",
                "splat_data": base64.b64encode(splat_bytes).decode(),
                "gaussian_count": gaussian_count,
                "generation_time": generation_time,
            })
            self._json_response(200, response)

        except HTTPError as e:
            body = e.read().decode(errors='replace')
            logger.error(f"World Labs HTTP {e.code}: {body}")
            self._json_error(e.code, f"World Labs API error ({e.code}): {body}")
        except Exception as e:
            logger.error(f"World Labs error: {e}")
            self._json_error(500, str(e))

    def _wl_upload_image(self, image_b64):
        """Upload base64 image as media asset. Returns media_asset_id."""
        image_bytes = base64.b64decode(image_b64)

        # 1. Prepare upload
        prep_body = json.dumps({
            "file_name": "prompt.png",
            "kind": "image",
            "extension": "png",
        }).encode()
        req = Request(
            f"{WORLDLABS_API_BASE}/marble/v1/media-assets:prepare_upload",
            data=prep_body, method="POST",
        )
        req.add_header("WLT-Api-Key", WORLDLABS_API_KEY)
        req.add_header("Content-Type", "application/json")

        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        asset = data["media_asset"]
        media_asset_id = asset.get("media_asset_id") or asset.get("id")
        upload_url = data["upload_info"]["upload_url"]
        logger.info(f"Prepared media asset: {media_asset_id}")

        if not media_asset_id or not upload_url:
            raise RuntimeError(f"Unexpected prepare_upload response: {json.dumps(data)[:300]}")

        # 2. PUT raw bytes to signed URL
        req = Request(upload_url, data=image_bytes, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        for hdr, val in data["upload_info"].get("required_headers", {}).items():
            req.add_header(hdr, val)

        with urlopen(req, timeout=60) as resp:
            pass  # 200 OK = success

        return media_asset_id

    def _wl_submit(self, prompt_type, prompt_text, seed, media_asset_id=None):
        """Submit generation request to World Labs API.
        Returns operation_id for polling."""
        world_prompt = {"type": prompt_type}

        if prompt_type == "image" and media_asset_id:
            world_prompt["image_prompt"] = {
                "source": "media_asset",
                "media_asset_id": media_asset_id,
            }
            if prompt_text:
                world_prompt["text_prompt"] = prompt_text
        else:
            world_prompt["type"] = "text"
            world_prompt["text_prompt"] = prompt_text

        body = {
            "display_name": "Scene Composer Map",
            "world_prompt": world_prompt,
        }

        data = json.dumps(body).encode()
        req = Request(
            f"{WORLDLABS_API_BASE}/marble/v1/worlds:generate",
            data=data,
            method="POST",
        )
        req.add_header("WLT-Api-Key", WORLDLABS_API_KEY)
        req.add_header("Content-Type", "application/json")

        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["operation_id"]

    def _wl_poll(self, operation_id):
        """Poll World Labs operation until done."""
        deadline = time.time() + WORLDLABS_POLL_TIMEOUT

        while time.time() < deadline:
            req = Request(
                f"{WORLDLABS_API_BASE}/marble/v1/operations/{operation_id}",
                method="GET",
            )
            req.add_header("WLT-Api-Key", WORLDLABS_API_KEY)

            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            if data.get("done"):
                return data.get("response", data)
            if data.get("error"):
                raise RuntimeError(f"Generation failed: {data['error']}")

            elapsed = int(time.time() - (deadline - WORLDLABS_POLL_TIMEOUT))
            status = data.get("metadata", {}).get("progress", {}).get("status", "?")
            logger.info(f"  Polling [{elapsed}s] status={status}")
            time.sleep(WORLDLABS_POLL_INTERVAL)

        raise TimeoutError(f"Generation timed out after {WORLDLABS_POLL_TIMEOUT}s")

    def _wl_download(self, url):
        """Download file from URL."""
        req = Request(url, method="GET")
        with urlopen(req, timeout=60) as resp:
            return resp.read()

    # ── JSON response helpers ────────────────────────────────────────

    def _json_response(self, status, body_str):
        body_bytes = body_str.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body_bytes)

    def _json_error(self, status, message):
        self._json_response(status, json.dumps({"status": "error", "error": message}))

    # ── TRELLIS Proxy (other /api/* routes) ──────────────────────────

    def _proxy(self, method):
        """Forward /api/* to TRELLIS server (localhost:8000)."""
        target_path = self.path[4:]  # strip "/api" prefix
        target_url = TRELLIS_URL + target_path

        try:
            body = None
            if method == "POST":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else None

            req = Request(target_url, data=body, method=method)
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))

            with urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        except URLError as e:
            self._json_error(502, f"TRELLIS server unavailable: {e.reason}")
        except Exception as e:
            self._json_error(500, f"Proxy error: {e}")

    def log_message(self, format, *args):
        if args and '200' not in str(args[1]):
            super().log_message(format, *args)


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a separate thread (prevents proxy from blocking static files)."""
    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._handle, args=(request, client_address))
        thread.daemon = True
        thread.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    parser = argparse.ArgumentParser(description="Scene Composer dev server")
    parser.add_argument('--port', type=int, default=PORT, help=f'Port (default: {PORT})')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    wl_status = "READY" if (WORLDLABS_API_KEY and HAS_SPZ) else "DISABLED"
    if not WORLDLABS_API_KEY:
        wl_status += " (no WORLDLABS_API_KEY)"
    elif not HAS_SPZ:
        wl_status += " (spz library missing)"

    print("=" * 50)
    print("  4DGS Scene Composer")
    print("=" * 50)
    print(f"  URL:        http://localhost:{args.port}")
    print(f"  Dir:        {os.getcwd()}")
    print(f"  TRELLIS:    {TRELLIS_URL} (proxied via /api/*)")
    print(f"  World Labs: {wl_status}")
    print("=" * 50)
    print("  Ctrl+C to stop")
    print()

    httpd = ThreadedHTTPServer(("", args.port), ComposerHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()

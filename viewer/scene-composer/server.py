#!/usr/bin/env python3
"""
Scene Composer — Static file server + TRELLIS API proxy.
Serves files with correct MIME types for ES modules.
Proxies /api/* requests to TRELLIS server (localhost:8000) so
the browser only needs access to port 8080.

Usage: python3 server.py [--port 8080]
"""

import os
import argparse
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

PORT = 8080
TRELLIS_URL = "http://localhost:8000"


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
        if self.path.startswith("/api/"):
            self._proxy("POST")
        else:
            self.send_error(405, "POST not allowed for static files")

    def _proxy(self, method):
        """Forward /api/* to TRELLIS server (localhost:8000)."""
        target_path = self.path[4:]  # strip "/api" prefix
        target_url = TRELLIS_URL + target_path

        try:
            # Read request body for POST
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
            error_msg = f'{{"status":"error","error":"TRELLIS server unavailable: {e.reason}"}}'
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(error_msg.encode())
        except Exception as e:
            error_msg = f'{{"status":"error","error":"Proxy error: {e}"}}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(error_msg.encode())

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

    print("=" * 50)
    print("  4DGS Scene Composer")
    print("=" * 50)
    print(f"  URL:     http://localhost:{args.port}")
    print(f"  Dir:     {os.getcwd()}")
    print(f"  TRELLIS: {TRELLIS_URL} (proxied via /api/*)")
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

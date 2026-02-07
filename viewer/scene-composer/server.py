#!/usr/bin/env python3
"""
Scene Composer — Static file server.
Serves files with correct MIME types for ES modules.
Browser connects to TRELLIS server directly (CORS enabled on TRELLIS side).

Usage: python server.py [--port 8080]
"""

import os
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8080


class CORSHandler(SimpleHTTPRequestHandler):
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

    def log_message(self, format, *args):
        if args and '200' not in str(args[1]):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Scene Composer dev server")
    parser.add_argument('--port', type=int, default=PORT, help=f'Port (default: {PORT})')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 50)
    print("  4DGS Scene Composer")
    print("=" * 50)
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  Dir:  {os.getcwd()}")
    print("=" * 50)
    print("  Ctrl+C to stop")
    print()

    httpd = HTTPServer(("", args.port), CORSHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()

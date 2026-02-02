#!/usr/bin/env python3
"""
3DGS Box Recorder Server
Saves captured frames and images in Unity-compatible format

Usage: python server.py
"""

import os
import json
import base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from datetime import datetime

PORT = 8075
OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")


class RecorderHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        path = urlparse(self.path).path
        
        if path == "/save_json":
            self.save_json()
        elif path == "/save_image":
            self.save_image()
        elif path == "/clear_output":
            self.clear_output()
        else:
            self.send_error(404, "Not Found")
    
    def save_json(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Save in Unity-compatible format
            filepath = os.path.join(OUTPUT_DIR, "full_data.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "path": filepath}).encode())
            print(f"✓ Saved JSON: {filepath} ({len(data.get('frames', []))} frames)")
            
        except Exception as e:
            print(f"✗ JSON Error: {e}")
            self.send_error(500, str(e))
    
    def save_image(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            frame_index = data.get("frameIndex", 0)
            image_data = data.get("imageData", "")
            
            os.makedirs(IMAGES_DIR, exist_ok=True)
            
            # Remove data URL prefix
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            filename = f"frame_{frame_index:04d}.png"
            filepath = os.path.join(IMAGES_DIR, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode())
            
            # Only print every 10th frame
            if frame_index % 10 == 0:
                print(f"✓ Saved image: {filepath}")
            
        except Exception as e:
            print(f"✗ Image Error: {e}")
            self.send_error(500, str(e))
    
    def clear_output(self):
        try:
            import shutil
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
            os.makedirs(IMAGES_DIR, exist_ok=True)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode())
            print("✓ Cleared output directory")
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    print("=" * 50)
    print("🎮 3DGS Box Recorder Server")
    print("=" * 50)
    print(f"📍 URL:     http://localhost:{PORT}")
    print(f"📁 Output:  {os.path.abspath(OUTPUT_DIR)}")
    print(f"🖼️  Images:  {os.path.abspath(IMAGES_DIR)}")
    print("=" * 50)
    print()
    print("Controls:")
    print("  WASD    - Move box")
    print("  Q/E     - Rotate box")
    print("  Space   - Jump")
    print("  R       - Start/Stop recording")
    print()
    print("Press Ctrl+C to stop server")
    print("=" * 50)
    
    httpd = HTTPServer(("", PORT), RecorderHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

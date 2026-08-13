"""Static dev server with correct MIME types for .mjs/.wasm (Windows-safe).

Usage: python serve.py [port] [--dir <root>]
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self):
        # Dev server: never let the browser cache stale MIME types / content.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        # Ignore conditional requests so cached entries with wrong headers refresh
        # with a full 200 instead of a 304 that preserves the stale Content-Type.
        del self.headers["If-Modified-Since"]
        del self.headers["If-None-Match"]
        super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8123
    root = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "."
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=root)).serve_forever()

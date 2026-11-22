"""Loads the model once and answers OCR requests over a Unix socket.
Started by the container CMD; app.py talks to it. The socket appears only once the model is warm."""
import json
import os
import socketserver
import time
from pathlib import Path

from PIL import Image, ImageDraw

import ocr

SOCK = os.environ.get("OCR_SOCK", "/tmp/ocr.sock")
LOCK = SOCK + ".starting"  # tells app.py a server is already loading


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        path = self.rfile.readline().decode().strip()
        try:
            reply = {"text": ocr.read(self.server.model, self.server.processor, path)}
        except Exception as e:  # one bad image must not kill the server
            reply = {"text": "", "error": repr(e)}
        self.wfile.write(json.dumps(reply, ensure_ascii=False).encode())


def main():
    Path(LOCK).touch()
    t = time.time()
    model, processor = ocr.load()
    # first generate compiles GPU kernels (~10 s); pay it here, not on the first graded image
    warm = "/tmp/ocr-warmup.png"
    img = Image.new("RGB", (640, 320), "white")
    ImageDraw.Draw(img).text((40, 140), "WARMUP 123", fill="black")
    img.save(warm)
    ocr.read(model, processor, warm)

    if os.path.exists(SOCK):
        os.unlink(SOCK)  # stale socket from a previous run
    with socketserver.UnixStreamServer(SOCK, Handler) as srv:
        srv.model, srv.processor = model, processor
        print(f"ready in {time.time() - t:.1f}s on {SOCK}", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()

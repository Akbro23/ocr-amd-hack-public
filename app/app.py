"""Harness entry point: python3 /app/app.py --input-image <path>
Stdlib only so it starts instantly; the model lives in server.py (started by the container CMD)."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SOCK = os.environ.get("OCR_SOCK", "/tmp/ocr.sock")
OUT_DIR = Path(os.environ.get("OCR_OUT", "/app/output"))
# the per-image limit is 30 s and a timeout zeroes the whole run, so give up early with an empty answer.
# Harness: container start -> one ungraded warm-up image -> graded images, so the server must be
# ready within ~2 x WAIT_S of container start or graded image 1 comes back empty.
WAIT_S = 27


def start_server():
    """Fallback if the container CMD didn't start the server (e.g. harness overrode it)."""
    if os.path.exists(SOCK + ".starting"):
        return  # already loading
    log = open("/tmp/ocr-server.log", "a")
    subprocess.Popen([sys.executable, str(Path(__file__).with_name("server.py"))],
                     stdout=log, stderr=log, start_new_session=True)


def ask(image):
    deadline = time.time() + WAIT_S
    started = False
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.settimeout(max(0.1, deadline - time.time()))
                s.connect(SOCK)
                s.sendall(f"{image}\n".encode())
                s.shutdown(socket.SHUT_WR)
                return json.loads(b"".join(iter(lambda: s.recv(65536), b"")).decode())
        except (FileNotFoundError, ConnectionRefusedError):
            if not started:
                start_server()
                started = True
            time.sleep(0.5)
        except Exception as e:  # timeout or broken reply
            return {"text": "", "error": repr(e)}
    return {"text": "", "error": "server not ready"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    image = Path(parser.parse_args().input_image).resolve()
    reply = ask(image)
    if "error" in reply:
        print(f"ocr error: {reply['error']}", file=sys.stderr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{image.stem}_output.json"
    out.write_text(json.dumps({"text": reply["text"]}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

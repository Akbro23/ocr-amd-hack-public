"""Reads an image config JSON on stdin, sets our runtime settings, writes it to stdout.
Used by build_image.sh: crane config IMG | python tools/image_config.py | crane edit config IMG"""
import json
import sys

DEPS = "/opt/ocr/deps"

cfg = json.load(sys.stdin)
c = cfg.setdefault("config", {})
env = dict(e.split("=", 1) for e in c.get("Env") or [])
env["OCR_MODEL"] = "/models/Qwen3.5-9B"
env["PYTHONPATH"] = ":".join([DEPS] + [p for p in env.get("PYTHONPATH", "").split(":") if p and p != DEPS])
c["Env"] = [f"{k}={v}" for k, v in env.items()]
c["Entrypoint"] = None
c["Cmd"] = ["python3", "/app/server.py"]
# the socket exists only once the model is loaded and warm (durations in ns)
c["Healthcheck"] = {"Test": ["CMD-SHELL", "test -S /tmp/ocr.sock"],
                    "Interval": 5 * 10**9, "StartPeriod": 600 * 10**9}

# `docker history` pairs history entries with layers: make sure every appended layer has one
hist = cfg.setdefault("history", [])
missing = len(cfg["rootfs"]["diff_ids"]) - sum(not h.get("empty_layer") for h in hist)
hist += [{"created_by": "ocr layer (crane append)"} for _ in range(missing)]

json.dump(cfg, sys.stdout)

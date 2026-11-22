"""Score the reader on a folder containing images + labels.json. Run on the notebook:

    python tools/eval.py data/samples
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch  # noqa: E402
import ocr  # noqa: E402


def normalize(s):
    """The graders' rules: uppercase, drop whitespace and - . · _"""
    return re.sub(r"[\s\-._·]", "", s.upper())


def main(folder):
    folder = Path(folder)
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    t = time.time()
    model, processor = ocr.load()
    print(f"model load {time.time() - t:.1f}s")
    torch.cuda.reset_peak_memory_stats()
    passed, slowest = 0, 0.0
    by_tag = {}  # tag -> [passed, total]; tags come from make_testset names: <id>_<category>_<degradations>
    for name, expected in labels.items():
        t = time.time()
        got = ocr.read(model, processor, folder / name)
        dt = time.time() - t
        slowest = max(slowest, dt)
        ok = normalize(got) == normalize(expected)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {dt:5.1f}s {name:40} got={got!r} want={expected!r}")
        parts = Path(name).stem.split("_")
        for tag in parts[1:2] + (parts[2].split("+") if len(parts) > 2 else []):
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][0] += ok
            by_tag[tag][1] += 1
    peak = torch.cuda.max_memory_allocated() / 2**30
    if len(by_tag) > 1:
        print("\n" + "  ".join(f"{k} {p}/{n}" for k, (p, n) in sorted(by_tag.items())))
    print(f"\n{passed}/{len(labels)} passed, slowest {slowest:.1f}s, peak VRAM (torch) {peak:.1f} GiB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/samples")

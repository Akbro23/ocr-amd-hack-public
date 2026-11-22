"""Extract the 10 sample images from overview.pdf into data/samples with labels.json.
Run on the notebook (needs the dev extra):

    uv pip install --system -r pyproject.toml --extra dev
    python tools/extract_samples.py
"""
import io
import json
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# (filename, expected answer) in document order; extensions follow the PDF captions
SAMPLES = [
    ("01_ca_clean.png", "7ABC123"),
    ("02_beijing.png", "京A·12345"),
    ("03_ny_angled.jpg", "JHT 2951"),
    ("04_motion_blur.png", "5XYZ891"),
    ("05_shanghai_lowlight.jpg", "沪B·88888"),
    ("06_stop_clean.png", "STOP"),
    ("07_stop_noise.tiff", "STOP"),
    ("08_speed_limit.jpg", "SPEED LIMIT 65"),
    ("09_road_work.png", "ROAD WORK AHEAD"),
    ("10_advisory.tiff", "35"),
]


def main():
    doc = pymupdf.open(ROOT / "overview.pdf")
    found = sorted(
        (page.number, page.get_image_rects(img[0])[0].y0, img[0])
        for page in doc for img in page.get_images(full=True)
    )
    assert len(found) == len(SAMPLES), f"expected {len(SAMPLES)} images in the PDF, found {len(found)}"
    out = ROOT / "data" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for (_, _, xref), (name, _) in zip(found, SAMPLES):
        Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB").save(out / name)
    (out / "labels.json").write_text(json.dumps(dict(SAMPLES), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(SAMPLES)} images to {out}")


if __name__ == "__main__":
    main()

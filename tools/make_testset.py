"""Generate a harder synthetic test set (plates + road signs, degraded) with labels.json. Run on the notebook:

    python tools/make_testset.py data/hard 150         # out dir, count (seeded: same set every time)
    python tools/make_testset.py data/hard3 150 3      # severity 3: every degradation ~3x stronger, more stacked

File names are <id>_<category>_<degradations>.<ext> so eval failures can be grouped.
"""
import io
import json
import math
import random
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "data" / "fonts"
DEJAVU = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.tar.bz2"
# release assets, not /raw/ links: the notebook proxy allows the former, not raw.githubusercontent.com
NOTO_SC = "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/18_NotoSansSC.zip"

L = "ABCDEFGHJKLMNPRSTUVWXYZ"  # plates skip I, O, Q
D = "0123456789"
SEV = 1.0  # degradation strength multiplier, set from argv
US_PATTERNS = ["DLLLDDD", "LLL DDDD", "LLL-DDDD", "LL DDDDD", "DDD LLL", "LLLDDD", "DLLD DDD"]
# (state banner, slogan, banner colour, number colour, plate colour): real names, generic designs
STATES = [
    ("CALIFORNIA", "dmv.ca.gov", (190, 30, 45), (25, 35, 110), (235, 240, 250)),
    ("NEW YORK", "EXCELSIOR", (20, 50, 130), (20, 50, 130), (250, 240, 220)),
    ("TEXAS", "THE LONE STAR STATE", (0, 0, 0), (0, 0, 0), (245, 245, 245)),
    ("FLORIDA", "SUNSHINE STATE", (0, 110, 60), (10, 90, 50), (250, 250, 250)),
    ("ILLINOIS", "LAND OF LINCOLN", (160, 20, 30), (150, 20, 30), (245, 245, 250)),
    ("ARIZONA", "GRAND CANYON STATE", (120, 40, 20), (20, 20, 90), (240, 220, 180)),
    ("NEVADA", "THE SILVER STATE", (30, 50, 120), (30, 50, 120), (210, 220, 235)),
    ("NEW JERSEY", "GARDEN STATE", (0, 0, 0), (0, 0, 0), (245, 230, 150)),
    ("OHIO", "BIRTHPLACE OF AVIATION", (150, 20, 30), (20, 20, 20), (240, 240, 240)),
    ("WASHINGTON", "EVERGREEN STATE", (20, 60, 120), (20, 60, 120), (240, 245, 250)),
    ("GEORGIA", "PEACH STATE", (0, 0, 0), (0, 0, 0), (250, 240, 225)),
    ("MICHIGAN", "PURE MICHIGAN", (20, 40, 100), (20, 40, 100), (245, 245, 245)),
]
PROVINCES = "京沪粤苏浙川鲁豫鄂湘闽皖津渝冀晋辽吉黑赣桂琼贵云陕甘青宁新蒙"
WARNINGS = ["ROAD WORK AHEAD", "DETOUR AHEAD", "BRIDGE MAY BE ICY", "RIGHT LANE CLOSED AHEAD",
            "LEFT LANE ENDS", "ONE LANE ROAD AHEAD", "FLAGGER AHEAD", "BE PREPARED TO STOP",
            "ROAD CLOSED AHEAD", "SHOULDER WORK", "LOOSE GRAVEL", "WORKERS AHEAD", "UNEVEN LANES",
            "ICY ROAD", "SLOW TRAFFIC AHEAD", "LANE ENDS MERGE LEFT"]
WORK_RECTS = ["END ROAD WORK", "ROAD CLOSED", "DETOUR", "LANE CLOSED", "SLOW", "ROAD CLOSED TO THRU TRAFFIC"]


def fetch(url):  # curl with retries: the notebook egress proxy intermittently answers 503
    return subprocess.run(["curl", "-fsSL", "--retry", "8", "--retry-delay", "5", "--retry-all-errors", url], check=True, capture_output=True).stdout


def fonts():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    bold, cond, cjk = (FONT_DIR / n for n in ("DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf", "NotoSansSC-Bold.otf"))
    if not (bold.exists() and cond.exists()):
        with tarfile.open(fileobj=io.BytesIO(fetch(DEJAVU)), mode="r:bz2") as t:
            for m in t.getmembers():
                if Path(m.name).name in (bold.name, cond.name):
                    (FONT_DIR / Path(m.name).name).write_bytes(t.extractfile(m).read())
    if not cjk.exists():
        with zipfile.ZipFile(io.BytesIO(fetch(NOTO_SC))) as z:
            cjk.write_bytes(z.read(next(n for n in z.namelist() if n.endswith("/" + cjk.name) or n == cjk.name)))
    return SimpleNamespace(bold=str(bold), cond=str(cond), cjk=str(cjk))


def text(d, center, s, font, size, fill, max_w):
    """Draw s centred at center, shrinking the font until it fits max_w."""
    f = ImageFont.truetype(font, size)
    while d.textlength(s, font=f) > max_w and size > 10:
        size -= 2
        f = ImageFont.truetype(font, size)
    d.text(center, s, font=f, fill=fill, anchor="mm")


def full(im):
    return Image.new("L", im.size, 255)


# ---- objects: each returns (RGB image, paste mask, expected answer) ----

def plate_us(r, F):
    state, slogan, banner_c, num_c, bg = r.choice(STATES)
    num = "".join(r.choice(L) if c == "L" else r.choice(D) if c == "D" else c for c in r.choice(US_PATTERNS))
    im = Image.new("RGB", (600, 300), bg)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((4, 4, 595, 295), 24, outline=(40, 40, 40), width=6)
    top, bottom = (state, slogan) if r.random() < 0.8 else (slogan, state)
    text(d, (300, 55), top, F.bold, 52, banner_c, 430)
    text(d, (300, 165), num, F.cond, 150, num_c, 540)
    text(d, (300, 262), bottom, F.bold, 28, banner_c, 480)
    for x in (55, 545):
        d.ellipse((x - 9, 46, x + 9, 64), fill=(150, 150, 150))
    if r.random() < 0.3:  # registration sticker: printed around the number, must be dropped
        d.rectangle((470, 20, 580, 90), fill=r.choice([(230, 60, 60), (60, 160, 60), (240, 200, 40)]))
        text(d, (525, 55), r.choice(["JAN", "MAR", "JUN", "SEP", "27", "28"]), F.bold, 34, (0, 0, 0), 100)
    return im, full(im), num


def plate_cn(r, F):
    kind = r.choice(["blue", "blue", "yellow", "green"])
    tail = (r.choice("DF") + "".join(r.choice(D) for _ in range(5)) if kind == "green"
            else "".join(r.choice(L + D) for _ in range(5)))
    label = f"{r.choice(PROVINCES)}{r.choice(L)}·{tail}"
    bg, fg = {"blue": ((17, 72, 166), (255, 255, 255)), "yellow": ((242, 190, 30), (0, 0, 0)),
              "green": ((150, 215, 150), (0, 0, 0))}[kind]
    w = 520 if kind == "green" else 480
    im = Image.new("RGB", (w, 150), bg)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((5, 5, w - 6, 144), 12, outline=fg, width=4)
    text(d, (w // 2, 72), label, F.cjk, 110, fg, w - 40)
    return im, full(im), label


def stop(r, F):
    pts = lambda rad: [(250 + rad * math.cos(math.radians(22.5 + 45 * k)),
                        250 + rad * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
    im, mask = Image.new("RGB", (500, 500)), Image.new("L", (500, 500))
    ImageDraw.Draw(mask).polygon(pts(245), fill=255)
    d = ImageDraw.Draw(im)
    d.polygon(pts(245), fill=(255, 255, 255))
    d.polygon(pts(228), fill=(185, 25, 35))
    text(d, (250, 250), "STOP", F.bold, 150, (255, 255, 255), 400)
    return im, mask, "STOP"


def speed(r, F):
    w1, w2 = r.choice([("SPEED", "LIMIT")] * 4 + [("MINIMUM", "SPEED")])
    n = r.randrange(15, 90, 5)
    im = Image.new("RGB", (360, 460), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle((14, 14, 345, 445), outline=(10, 10, 10), width=7)
    text(d, (180, 85), w1, F.bold, 72, (10, 10, 10), 300)
    text(d, (180, 165), w2, F.bold, 72, (10, 10, 10), 300)
    text(d, (180, 330), str(n), F.bold, 180, (10, 10, 10), 300)
    return im, full(im), f"{w1} {w2} {n}"


def advisory(r, F):
    n = r.randrange(10, 65, 5)
    mph = r.random() < 0.3  # unit printed -> part of the answer; absent -> must not be added
    im = Image.new("RGB", (360, 360), (245, 195, 25))
    d = ImageDraw.Draw(im)
    d.rectangle((12, 12, 347, 347), outline=(10, 10, 10), width=7)
    if mph:
        text(d, (180, 145), str(n), F.bold, 170, (10, 10, 10), 300)
        text(d, (180, 285), "MPH", F.bold, 80, (10, 10, 10), 300)
    else:
        text(d, (180, 180), str(n), F.bold, 210, (10, 10, 10), 300)
    return im, full(im), f"{n} MPH" if mph else str(n)


def warning(r, F):
    msg = r.choice(WARNINGS)
    lines = msg.split()
    size = {1: 110, 2: 95, 3: 78, 4: 62}[len(lines)]
    diamond = lambda m: [(260, m), (520 - m, 260), (260, 520 - m), (m, 260)]
    im, mask = Image.new("RGB", (520, 520)), Image.new("L", (520, 520))
    ImageDraw.Draw(mask).polygon(diamond(2), fill=255)
    d = ImageDraw.Draw(im)
    d.polygon(diamond(2), fill=r.choice([(240, 130, 20), (245, 200, 30)]))
    d.polygon(diamond(22), outline=(10, 10, 10), width=8)
    y0 = 260 - (len(lines) - 1) * size * 0.55
    for i, line in enumerate(lines):
        text(d, (260, y0 + i * size * 1.1), line, F.bold, size, (10, 10, 10), 270)
    return im, mask, msg


def worksign(r, F):
    msg = r.choice(WORK_RECTS)
    words = msg.split()
    lines = [msg] if len(words) == 1 else [" ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2:])]
    im = Image.new("RGB", (640, 320), (240, 130, 20))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((12, 12, 627, 307), 18, outline=(10, 10, 10), width=7)
    for i, line in enumerate(lines):
        y = 160 if len(lines) == 1 else 110 + i * 105
        text(d, (320, y), line, F.bold, 90, (10, 10, 10), 560)
    return im, full(im), msg


OBJECTS = {"plateUS": (plate_us, 3), "plateCN": (plate_cn, 2), "stop": (stop, 1), "speed": (speed, 2),
           "advisory": (advisory, 1), "warning": (warning, 2), "worksign": (worksign, 1)}


# ---- scene + degradations ----

def background(r, w, h):
    rng = np.random.default_rng(r.randrange(2**32))
    small = rng.normal(r.randint(50, 190), r.uniform(8, 35), (h // 16 + 2, w // 16 + 2, 3))
    small += rng.normal(0, 25, 3)  # colour cast
    return Image.fromarray(small.clip(0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)


def compose(r, obj, mask):
    if r.random() < 0.2:  # small object in a big photo-sized frame
        W = r.randint(1800, 3200)
        H = int(W * r.uniform(0.55, 0.8))
        s = W * r.uniform(0.12, 0.3) / SEV / obj.width
        obj = obj.resize((int(obj.width * s), int(obj.height * s)), Image.LANCZOS)
        mask = mask.resize(obj.size, Image.LANCZOS)
        tags = ["scene"]
    else:
        W, H = int(obj.width * r.uniform(1.15, 1.6)), int(obj.height * r.uniform(1.15, 1.6))
        tags = []
    im = background(r, W, H)
    im.paste(obj, (r.randint(0, W - obj.width), r.randint(0, H - obj.height)), mask)
    return im, tags


def _coeffs(src, dst):  # perspective coefficients mapping output points dst -> input points src
    a, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.linalg.solve(np.array(a, float), np.array(b, float)).tolist()


def perspective(im, r):
    w, h = im.size
    j = lambda: r.uniform(0, min(0.22 * SEV, 0.4))
    quad = [(j() * w, j() * h), (w - j() * w, j() * h), (w - j() * w, h - j() * h), (j() * w, h - j() * h)]
    return im.transform(im.size, Image.PERSPECTIVE, _coeffs([(0, 0), (w, 0), (w, h), (0, h)], quad),
                        Image.BICUBIC, fillcolor=(90, 90, 90))


def rotate(im, r):
    return im.rotate(r.uniform(-15, 15) * min(SEV, 2.5), Image.BICUBIC, expand=True, fillcolor=(90, 90, 90))


def arr_op(fn):
    return lambda im, r: Image.fromarray(fn(np.asarray(im, float), r).clip(0, 255).astype(np.uint8))


def _dark(a, r):
    return a * r.uniform(0.12, 0.35) / SEV


def _glare(a, r):
    h, w = a.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    cx, cy, rad = r.uniform(0, w), r.uniform(0, h), r.uniform(0.15, 0.5) * max(w, h)
    return a + (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2)) * r.uniform(120, 230) * SEV)[..., None]


def _motion(a, r):
    k, axis = int(r.randint(3, 9) * SEV), r.choice([0, 1])
    return sum(np.roll(a, s, axis=axis) for s in range(-k, k + 1)) / (2 * k + 1)


def _noise(a, r):
    return a + np.random.default_rng(r.randrange(2**32)).normal(0, r.uniform(15, 45) * SEV, a.shape)


def blur(im, r):
    return im.filter(ImageFilter.GaussianBlur(r.uniform(1.5, 4) * SEV))


def lowres(im, r):
    f = r.uniform(0.15, 0.35) / SEV
    return im.resize((max(8, int(im.width * f)), max(8, int(im.height * f))), Image.BILINEAR).resize(im.size, Image.BILINEAR)


# applied in this order (geometry, light, optics, sensor)
DEGRADE = {"persp": perspective, "rot": rotate, "dark": arr_op(_dark), "glare": arr_op(_glare),
           "blur": blur, "motion": arr_op(_motion), "lowres": lowres, "noise": arr_op(_noise)}


def save(im, path, r):
    if path.suffix == ".jpg":
        im.save(path, quality=max(5, int(r.randint(15, 70) / SEV)))
    elif path.suffix == ".tif" and r.random() < 0.25:  # 16-bit greyscale TIFF
        g = np.asarray(im.convert("L"), np.uint16) * 257
        Image.fromarray(g, "I;16").save(path)
    else:
        im.save(path)


def main(out, n, sev):
    global SEV
    SEV = sev
    r = random.Random(1234)
    F = fonts()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names, weights = list(OBJECTS), [w for _, w in OBJECTS.values()]
    labels = {}
    for i in range(n):
        cat = r.choices(names, weights)[0]
        obj, mask, label = OBJECTS[cat][0](r, F)
        im, tags = compose(r, obj, mask)
        picked = r.sample(list(DEGRADE), r.choices([0, 1, 2, 3, 4], [1, 3, 4, 2, 0] if SEV <= 1 else [0, 1, 3, 4, 2])[0])
        for t in DEGRADE:
            if t in picked:
                im = DEGRADE[t](im, r)
                tags.append(t)
        name = f"{i:03d}_{cat}_{'+'.join(tags) or 'clean'}.{r.choice(['png', 'jpg', 'tif'])}"
        save(im, out / name, r)
        labels[name] = label
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {n} images to {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "data/hard", int(a[1]) if len(a) > 1 else 150, float(a[2]) if len(a) > 2 else 1.0)

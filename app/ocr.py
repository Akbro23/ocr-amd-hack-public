"""Qwen3.5-9B text reader: load() once, then read(model, processor, path) -> text."""
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = os.environ.get("OCR_MODEL", "/models/Qwen3.5-9B")
# shortcut: fixed resize bounds, tune them in Phase 2 against the eval set
MAX_SIDE, MIN_SIDE = 1280, 448

PROMPT = """Read the text in this photo of a license plate or road sign.
Rules:
- License plate: return only the registration number. Leave out state names, slogans and anything else printed around it (e.g. CALIFORNIA, EXCELSIOR).
- Chinese plate: keep the leading province character and letter, e.g. 京A·12345.
- Road sign: return every word and number printed on the sign, top to bottom, joined with single spaces (e.g. SPEED LIMIT 65, ROAD WORK AHEAD).
- Do not add units, labels, quotes or explanation.
Answer with the text only, on one line."""


def load():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda"  # "cuda" is the HIP device on ROCm
    ).eval()
    return model, processor


def prepare(path):
    img = Image.open(path)
    if img.mode.startswith("I"):  # 16/32-bit TIFF: scale to 8-bit instead of clipping to white
        img = img.convert("I").point(lambda v: v / 256).convert("L")
    img = img.convert("RGB")
    w, h = img.size
    # downscale huge images, upscale tiny ones, never past MAX_SIDE
    scale = min(MAX_SIDE / max(w, h), max(1.0, MIN_SIDE / min(w, h)))
    if scale != 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


@torch.inference_mode()
def read(model, processor, path):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": prepare(path)},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return clean(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def clean(raw):
    text = raw.split("</think>")[-1].strip()  # drop an empty/leaked thinking block
    for prefix in ("text:", "answer:", "plate:", "output:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    text = text.strip().strip("`\"'“”")
    for dot in "•・‧∙":  # odd interpuncts the graders don't strip
        text = text.replace(dot, "·")
    return " ".join(text.split())

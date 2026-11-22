"""Checks for output cleanup and grader normalization. Run on the notebook: python tests/test_text.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tools")]
from ocr import clean  # noqa: E402
from eval import normalize  # noqa: E402

assert clean("<think>\n\n</think>\n\n7ABC123") == "7ABC123"
assert clean('Plate: "JHT 2951"') == "JHT 2951"
assert clean("ROAD\nWORK\nAHEAD") == "ROAD WORK AHEAD"
assert clean("京A•12345") == "京A·12345"
assert normalize("7-abc 123") == normalize("7ABC123") == "7ABC123"
assert normalize("京A·12345") == "京A12345"
assert normalize("CALIFORNIA 7ABC123") != normalize("7ABC123")
print("ok")

"""Shared result table and placeholder figure helpers."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NA = "n/a"


def write_csv(name: str, headers: list[str], rows: list[list[object]]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def placeholder_png(name: str, title: str, reason: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    image = np.full((500, 1000, 3), 255, dtype=np.uint8)
    cv2.putText(image, _ascii_vietnamese(title), (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "Chua co du lieu de tao bieu do", (40, 230), cv2.FONT_HERSHEY_SIMPLEX,
                .9, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(image, _ascii_vietnamese(reason[:100]), (40, 290), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (80, 80, 80), 1, cv2.LINE_AA)
    path = RESULTS / name
    encoded, data = cv2.imencode(".png", image)
    if encoded:
        path.write_bytes(data.tobytes())
    return path


def _ascii_vietnamese(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char) and ord(char) < 128)

"""TuSimple annotation loading and validation utilities."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Iterable

H_SAMPLES = list(range(160, 711, 10))


def read_annotations(label_files: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read JSON-lines annotations, rejecting malformed records."""
    records: list[dict[str, Any]] = []
    for label_file in label_files:
        path = Path(label_file)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                _validate_record(record, path, line_number)
                records.append(record)
    return records


def validate_dataset(root: str | Path, label_files: Iterable[str | Path]) -> list[str]:
    """Return validation errors for labels and referenced images."""
    root_path = Path(root)
    errors: list[str] = []
    for label_file in label_files:
        path = Path(label_file)
        try:
            records = read_annotations([path])
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        for record in records:
            image_path = root_path / record["raw_file"]
            if not image_path.is_file():
                errors.append(f"{path}: missing image {image_path}")
    return errors


def create_preview(root: str | Path, label_file: str | Path, output_dir: str | Path,
                   limit: int | None = None) -> int:
    """Write annotated final-frame previews; return the number written."""
    import cv2

    root_path, out_path = Path(root), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records = read_annotations([label_file])
    count = 0
    for record in records[:limit]:
        image_path = root_path / record["raw_file"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for lane in record["lanes"]:
            points = [(int(x), int(y)) for x, y in zip(lane, record["h_samples"]) if x >= 0]
            for point in points:
                cv2.circle(image, point, 3, (0, 255, 0), -1)
        target = out_path / f"{count:06d}.png"
        if cv2.imwrite(str(target), image):
            count += 1
    return count


def _validate_record(record: Any, path: Path, line_number: int) -> None:
    where = f"{path}:{line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{where}: annotation must be an object")
    for key in ("raw_file", "h_samples", "lanes"):
        if key not in record:
            raise ValueError(f"{where}: missing {key}")
    if not isinstance(record["raw_file"], str):
        raise ValueError(f"{where}: raw_file must be a string")
    samples, lanes = record["h_samples"], record["lanes"]
    if not isinstance(samples, list) or not all(isinstance(y, int) for y in samples):
        raise ValueError(f"{where}: h_samples must be an integer list")
    if not isinstance(lanes, list):
        raise ValueError(f"{where}: lanes must be a list")
    for lane in lanes:
        if not isinstance(lane, list) or len(lane) != len(samples):
            raise ValueError(f"{where}: each lane must match h_samples length")
        if not all(isinstance(x, int) for x in lane):
            raise ValueError(f"{where}: lane coordinates must be integers")


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate TuSimple labels and write previews")
    subparsers = parser.add_subparsers(dest="command", required=True)
    tusimple = subparsers.add_parser("tusimple")
    tusimple.add_argument("--root", required=True)
    tusimple.add_argument("--labels", nargs="+", required=True)
    tusimple.add_argument("--preview", required=True)
    tusimple.add_argument("--limit", type=int)
    args = parser.parse_args()
    errors = validate_dataset(args.root, args.labels)
    if errors:
        for error in errors:
            print(f"Lỗi xác thực: {error}")
        raise SystemExit(1)
    total = sum(create_preview(args.root, label, args.preview, args.limit) for label in args.labels)
    print(f"Xác thực thành công; đã ghi {total} ảnh xem trước")


if __name__ == "__main__":
    main()

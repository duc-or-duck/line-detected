import json

import pytest

from src.dataset import H_SAMPLES, read_annotations, validate_dataset


def test_read_and_validate_annotation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "frame.jpg").write_bytes(b"fixture")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"raw_file": "frame.jpg", "h_samples": [160, 170],
                                 "lanes": [[10, -2]]}) + "\n", encoding="utf-8")
    assert len(read_annotations([labels])) == 1
    assert validate_dataset(root, [labels]) == []


def test_rejects_mismatched_lane_length(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"raw_file": "x.jpg", "h_samples": [160, 170],
                                 "lanes": [[10]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="match h_samples"):
        read_annotations([labels])


def test_tusimple_sample_rows():
    assert H_SAMPLES == list(range(160, 711, 10))

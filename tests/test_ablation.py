from eval.ablation import _dashcam_rows, _warning_rows


def test_dashcam_manual_labels_produce_frame_rates(tmp_path):
    labels = tmp_path / "dashcam.csv"
    labels.write_text("condition,video,frame_index,baseline_correct,proposed_correct\n"
                      "day,missing.mp4,1,true,false\n", encoding="utf-8")
    rows = _dashcam_rows(str(labels))
    assert rows[0][1:4] == [1, 100.0, 0.0]
    assert rows[0][4] == "n/a"
    assert rows[1][1:4] == [0, "n/a", "n/a"]


def test_warning_annotations_compare_offset_only_with_tlc_hysteresis(tmp_path):
    labels = tmp_path / "warning.csv"
    labels.write_text("offset_m,offset_velocity_mps,true_departure\n0.3,0,false\n", encoding="utf-8")
    config = {"departure": {"lane_width_m": 3.7, "on_offset_m": .5, "off_offset_m": .2,
                            "tlc_threshold_s": .8, "consecutive_frames": 1,
                            "offset_only_threshold_m": .2},
              "dataset_metadata": {"dashcam_fps": 60}}
    rows = _warning_rows(str(labels), config)
    assert rows[0][1] == 3600
    assert rows[1][1] == 0

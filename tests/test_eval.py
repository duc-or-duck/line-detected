from eval.tusimple_eval import score_lanes


def test_tusimple_lane_metrics_and_pixel_error():
    score = score_lanes([[10, 13, -2]], [[10, 12, -2]], 5, .85)
    assert score["accuracy"] == 1
    assert score["fp"] == 0
    assert score["fn"] == 0
    assert score["pixel_error"] == 0.5


def test_unknown_tusimple_thresholds_yield_unavailable_metrics():
    score = score_lanes([[10]], [[10]], None, None)
    assert score["accuracy"] is None
    assert score["fp"] is None


def test_tusimple_matching_is_independent_of_lane_list_order():
    score = score_lanes([[80, 80], [20, 20]], [[20, 20], [80, 80]], 5, .85)
    assert score["accuracy"] == 1
    assert score["pixel_error"] == 0

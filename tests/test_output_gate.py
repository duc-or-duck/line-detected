import numpy as np
from src.lane_fit import output_geometry_is_valid


def test_output_gate_rejects_hook_and_keeps_parallel_lanes():
    bounds = {"left": (.2, .9), "right": (.2, .9)}
    valid = np.array([0., -.25, .35])
    assert output_geometry_is_valid(valid, valid + [0, 0, .3], bounds, 1280, {})
    hook = np.array([2.2, -2.4, .8])
    assert not output_geometry_is_valid(hook, hook + [0, 0, .3], bounds, 1280, {})

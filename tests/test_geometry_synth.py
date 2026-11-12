import numpy as np
from tools.synth_scene import run


def test_synthetic_geometry_recovers_monotonic_lanes():
    for kind, brightness, dashed in (("straight", 1, False), ("curve", 1, False),
                                     ("straight", .15, True)):
        result, _ = run(kind, brightness, dashed)
        assert result["lane_measurement_available"]
        rows = np.linspace(.25, .75, 8)
        left = np.polyval(result["left"], rows)
        right = np.polyval(result["right"], rows)
        assert np.all(left < right)
        assert np.all((left >= 0) & (right <= 1))
        assert result["offset_px"] is not None

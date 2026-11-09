import numpy as np
import pytest

from src.kalman import GATING_THRESHOLD, LaneKalmanFilter


def make_filter():
    return LaneKalmanFilter(1.0, np.eye(6) * 1e-5, np.eye(3) * 1e-3,
                            np.eye(6) * 0.01, initial_state=np.array([.2, .3, .4, 0, 0, 0]))


def test_model_and_predict_update_match_opencv():
    cv2 = pytest.importorskip("cv2")
    ours = make_filter()
    reference = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)
    reference.transitionMatrix = ours.F.copy()
    reference.measurementMatrix = ours.H.copy()
    reference.processNoiseCov = ours.Q.copy()
    reference.measurementNoiseCov = ours.R.copy()
    reference.errorCovPost = ours.P.copy()
    reference.statePost = ours.x.copy()
    assert np.array_equal(ours.F[:3, 3:], np.eye(3))
    assert np.array_equal(ours.H[:, :3], np.eye(3))
    for measurement in ([.21, .29, .41], [.22, .31, .4], [.23, .32, .39]):
        ours.predict()
        ref_prediction = reference.predict().copy()
        np.testing.assert_allclose(ours.x, ref_prediction, atol=1e-4)
        ours.update(np.array(measurement))
        ref_state = reference.correct(np.asarray(measurement, dtype=np.float64).reshape(3, 1))
        np.testing.assert_allclose(ours.x, ref_state, atol=1e-4)


def test_gating_rejects_outlier_and_keeps_prediction():
    kalman = make_filter()
    kalman.step(np.array([.2, .3, .4]))
    before = kalman.x.copy()
    estimate = kalman.step(np.array([.99, .99, .99]))
    assert kalman.last_mahalanobis > GATING_THRESHOLD
    assert kalman.last_accepted is False
    np.testing.assert_allclose(estimate, (kalman.F @ before)[:3, 0])


@pytest.mark.parametrize("noise,dropout", [(0.005, 0.0), (0.04, 0.25)])
def test_filter_reduces_curve_rmse(noise, dropout):
    rng = np.random.default_rng(42)
    truth = np.column_stack([.2 + .002 * np.arange(120),
                             .3 - .001 * np.arange(120),
                             .4 + .0005 * np.arange(120)])
    observations = truth + rng.normal(0, noise, truth.shape)
    missing = rng.random(120) < dropout
    kalman = LaneKalmanFilter(1.0, np.eye(6) * 1e-6, np.eye(3) * noise**2,
                              np.eye(6) * 0.1, initial_state=np.r_[truth[0], 0, 0, 0])
    estimates = []
    for i, obs in enumerate(observations):
        estimates.append(kalman.step(None if missing[i] else obs))
    mask = ~missing
    raw_rmse = np.sqrt(np.mean((observations[mask] - truth[mask]) ** 2))
    filtered_rmse = np.sqrt(np.mean((np.asarray(estimates)[mask] - truth[mask]) ** 2))
    assert filtered_rmse < raw_rmse

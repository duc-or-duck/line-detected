"""NumPy-only constant-velocity Kalman filter for normalized lane coefficients."""

from __future__ import annotations

import numpy as np


GATING_THRESHOLD = 11.34


class LaneKalmanFilter:
    """Track [A, B, C, A', B', C'] from measurements [A, B, C].

    Coefficients and their velocities are expressed in normalized image
    coordinates. Covariance matrices are parameters and must come from config.
    """

    def __init__(self, dt: float, process_noise: np.ndarray,
                 measurement_noise: np.ndarray, initial_covariance: np.ndarray,
                 initial_state: np.ndarray | None = None,
                 gating_threshold: float = GATING_THRESHOLD) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.F = np.eye(6, dtype=np.float64)
        self.F[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3, dtype=np.float64)
        self.Q = _matrix(process_noise, (6, 6), "process_noise")
        self.R = _matrix(measurement_noise, (3, 3), "measurement_noise")
        self.P = _matrix(initial_covariance, (6, 6), "initial_covariance")
        self.x = np.zeros((6, 1), dtype=np.float64) if initial_state is None else _vector(initial_state, 6)
        self.gating_threshold = float(gating_threshold)
        self.last_mahalanobis: float | None = None
        self.last_accepted: bool | None = None

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = (self.P + self.P.T) * 0.5
        return self.x[:3, 0].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = _vector(measurement, 3)
        innovation = z - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        try:
            solved = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance must be invertible") from exc
        distance = float((innovation.T @ solved).item())
        self.last_mahalanobis = distance
        self.last_accepted = distance <= self.gating_threshold
        if self.last_accepted:
            gain = np.linalg.solve(innovation_covariance.T, (self.P @ self.H.T).T).T
            self.x = self.x + gain @ innovation
            # Joseph form maintains symmetry and positive semidefiniteness.
            residual = np.eye(6) - gain @ self.H
            self.P = residual @ self.P @ residual.T + gain @ self.R @ gain.T
            self.P = (self.P + self.P.T) * 0.5
        return self.x[:3, 0].copy()

    def step(self, measurement: np.ndarray | None) -> np.ndarray:
        """Predict once, then update if a measurement is available and accepted."""
        self.predict()
        if measurement is not None:
            self.update(measurement)
        else:
            self.last_accepted = False
            self.last_mahalanobis = None
        return self.x[:3, 0].copy()


def _matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return array.copy()


def _vector(value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"vector must contain {size} finite values")
    return array.reshape(size, 1)

"""
KalmanBoxTracker for constant-velocity bounding box tracking in AnimalLens.
"""
from __future__ import annotations

import numpy as np
from typing import List, Sequence


class KalmanBoxTracker:
    """
    Tracks bounding boxes [x_min, y_min, x_max, y_max] using a linear Kalman filter.
    State representation: [x1, y1, x2, y2, vx1, vy1, vx2, vy2].
    """
    count = 0

    def __init__(self, bbox: Sequence[float]) -> None:
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        # 8-dim state: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.x = np.zeros((8, 1), dtype=np.float64)
        for i in range(4):
            self.x[i, 0] = bbox[i]

        # State transition matrix F (dt = 1)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrix P
        self.P = np.eye(8, dtype=np.float64) * 10.0
        for i in range(4, 8):
            self.P[i, i] *= 10.0

        # Process noise Q
        self.Q = np.eye(8, dtype=np.float64) * 0.01
        for i in range(4, 8):
            self.Q[i, i] *= 0.1

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float64) * 0.1

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history: List[List[float]] = []

    def predict(self) -> List[float]:
        """Advance the state vector and return the predicted bounding box."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.get_state()

    def update(self, bbox: Sequence[float]) -> None:
        """Update the state vector with an observed bounding box measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float64)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(8, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> List[float]:
        """Return current estimated bounding box [x1, y1, x2, y2]."""
        return [float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])]

    def get_velocity(self) -> float:
        """Return scalar magnitude of the velocity vector."""
        vx = (self.x[4, 0] + self.x[6, 0]) / 2.0
        vy = (self.x[5, 0] + self.x[7, 0]) / 2.0
        return float(np.sqrt(vx * vx + vy * vy))

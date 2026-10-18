"""Trajectory conditioning and velocity estimation via Savitzky-Golay filtering.

Eliminates high-frequency optical tracking jitter and derives biomechanically realistic
velocities and accelerations.
"""

from typing import Tuple
import numpy as np

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except (ImportError, MemoryError):
    HAS_SCIPY = False

    def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0):
        """Pure NumPy fallback for Savitzky-Golay when SciPy DLL cannot allocate memory."""
        if deriv == 0:
            # Moving average fallback
            kernel = np.ones(window_length) / window_length
            pad_width = window_length // 2
            x_pad = np.pad(x, pad_width, mode='edge')
            return np.convolve(x_pad, kernel, mode='valid')
        elif deriv == 1:
            return np.gradient(x, delta)
        elif deriv == 2:
            return np.gradient(np.gradient(x, delta), delta)
        return x


class TrajectoryFilter:
    """Savitzky-Golay low-pass filter and finite difference differentiator."""

    def __init__(
        self,
        fps: float = 25.0,
        window_length: int = 7,
        polyorder: int = 2,
        max_speed_mps: float = 11.5,
        max_accel_mps2: float = 7.0,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.window_length = window_length if window_length % 2 != 0 else window_length + 1
        self.polyorder = polyorder
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2

    def smooth_trajectory(self, positions: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to a sequence of (x, y) player coordinates.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (T, 2) where T is the number of frames.

        Returns
        -------
        np.ndarray
            Smoothed positions of shape (T, 2).
        """
        if len(positions) < self.window_length:
            return positions.copy()

        smoothed_x = savgol_filter(positions[:, 0], window_length=self.window_length, polyorder=self.polyorder)
        smoothed_y = savgol_filter(positions[:, 1], window_length=self.window_length, polyorder=self.polyorder)
        return np.stack([smoothed_x, smoothed_y], axis=-1)

    def estimate_velocity(
        self,
        positions: np.ndarray,
        smooth_first: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity (m/s) and acceleration (m/s^2) from position trajectory.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (T, 2) in meters.
        smooth_first : bool
            Whether to apply Savitzky-Golay smoothing before numerical differentiation.

        Returns
        -------
        velocities : np.ndarray
            Velocity vectors (vx, vy) of shape (T, 2) in m/s.
        accelerations : np.ndarray
            Acceleration vectors (ax, ay) of shape (T, 2) in m/s^2.
        """
        T = len(positions)
        if T < 2:
            return np.zeros_like(positions), np.zeros_like(positions)

        pos = self.smooth_trajectory(positions) if smooth_first else positions

        # Savitzky-Golay direct 1st and 2nd derivatives if sequence is long enough
        if T >= self.window_length:
            vx = savgol_filter(pos[:, 0], window_length=self.window_length, polyorder=self.polyorder, deriv=1, delta=self.dt)
            vy = savgol_filter(pos[:, 1], window_length=self.window_length, polyorder=self.polyorder, deriv=1, delta=self.dt)
            ax = savgol_filter(pos[:, 0], window_length=self.window_length, polyorder=self.polyorder, deriv=2, delta=self.dt)
            ay = savgol_filter(pos[:, 1], window_length=self.window_length, polyorder=self.polyorder, deriv=2, delta=self.dt)
        else:
            # Central difference fallback for short sequences
            vx = np.gradient(pos[:, 0], self.dt)
            vy = np.gradient(pos[:, 1], self.dt)
            ax = np.gradient(vx, self.dt)
            ay = np.gradient(vy, self.dt)

        velocities = np.stack([vx, vy], axis=-1)
        accelerations = np.stack([ax, ay], axis=-1)

        # Biomechanical clamping
        speeds = np.linalg.norm(velocities, axis=-1, keepdims=True)
        speed_clamping = np.maximum(1.0, speeds / self.max_speed_mps)
        velocities = velocities / speed_clamping

        accel_mags = np.linalg.norm(accelerations, axis=-1, keepdims=True)
        accel_clamping = np.maximum(1.0, accel_mags / self.max_accel_mps2)
        accelerations = accelerations / accel_clamping

        return velocities, accelerations

"""Unit tests for coordinate transformers, Savitzky-Golay smoothing, and scenario loaders."""

import numpy as np
import pytest

from app.config import DEFAULT_PITCH
from app.data.coordinate_transformer import CanonicalCoordinateTransformer
from app.data.filter import TrajectoryFilter
from app.data.metrica_loader import generate_synthetic_turnover_scenario


class TestDataPipeline:
    """Verifies coordinate standardization and signal processing."""

    @pytest.fixture
    def transformer(self):
        return CanonicalCoordinateTransformer(pitch_cfg=DEFAULT_PITCH)

    def test_coordinate_transformation_reversibility(self, transformer):
        """Converting normalized to meters and back yields original coordinates."""
        orig_norm = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.75],
        ])

        metric_coords = transformer.metrica_to_metric(orig_norm)
        reconstructed_norm = transformer.metric_to_metrica(metric_coords)

        assert np.allclose(orig_norm, reconstructed_norm)

    def test_metrica_origin_inversion(self, transformer):
        """Metrica (0,0) top-left maps to FIFA (0, 68m) top-left in Cartesian."""
        top_left_norm = (0.0, 0.0)
        top_left_m = transformer.metrica_to_metric(top_left_norm)
        assert np.isclose(top_left_m[0], 0.0)
        assert np.isclose(top_left_m[1], 68.0)

        bottom_right_norm = (1.0, 1.0)
        bottom_right_m = transformer.metrica_to_metric(bottom_right_norm)
        assert np.isclose(bottom_right_m[0], 105.0)
        assert np.isclose(bottom_right_m[1], 0.0)

    def test_pitch_inversion_reversibility(self, transformer):
        """Flipping pitch twice returns to original coordinates."""
        coords = np.array([[30.0, 20.0], [80.0, 50.0]])
        flipped = transformer.invert_attack_direction(coords)
        double_flipped = transformer.invert_attack_direction(flipped)
        assert np.allclose(coords, double_flipped)

    def test_savitzky_golay_noise_reduction_and_velocity_bounds(self):
        """Savitzky-Golay filtering dampens artificial jitter and clamps max sprinting speed."""
        np.random.seed(42)
        fps = 25.0
        T = 50
        t_arr = np.linspace(0, 2.0, T)
        # Smooth line + high frequency optical noise
        true_x = 5.0 * t_arr  # 5 m/s speed
        noisy_x = true_x + np.random.normal(0, 0.3, T)
        noisy_y = np.full(T, 34.0) + np.random.normal(0, 0.3, T)

        positions = np.stack([noisy_x, noisy_y], axis=-1)

        filter_engine = TrajectoryFilter(fps=fps, max_speed_mps=11.5, max_accel_mps2=7.0)
        smoothed = filter_engine.smooth_trajectory(positions)
        velocities, accelerations = filter_engine.estimate_velocity(positions)

        # Smoothed positions should have lower total variation / jitter than raw noisy positions
        raw_jitter = np.sum(np.abs(np.diff(positions[:, 0], n=2)))
        smoothed_jitter = np.sum(np.abs(np.diff(smoothed[:, 0], n=2)))
        assert smoothed_jitter < raw_jitter

        # Speeds and accelerations must strictly adhere to biomechanical bounds
        speeds = np.linalg.norm(velocities, axis=-1)
        assert np.all(speeds <= 11.5 + 1e-4)

    def test_synthetic_turnover_generator_contract(self):
        """Synthetic scenario generator creates 75 continuous frames with 22 players."""
        frames = generate_synthetic_turnover_scenario(n_frames=75, fps=25.0)
        assert len(frames) == 75

        for f in [frames[0], frames[37], frames[-1]]:
            assert len(f.players) == 22
            att_count = sum(1 for p in f.players if p.team == "att")
            def_count = sum(1 for p in f.players if p.team == "def")
            assert att_count == 11
            assert def_count == 11
            assert 0.0 <= f.ball_position[0] <= 105.0
            assert 0.0 <= f.ball_position[1] <= 68.0

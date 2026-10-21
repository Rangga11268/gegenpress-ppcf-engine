"""Unit tests for kinematics, neuromuscular reaction latency, and arrival time formulas."""

import numpy as np
import pytest

from app.config import DEFAULT_PHYSICS
from app.core.kinematics import (
    compute_ball_travel_time,
    compute_player_arrival_time,
    compute_reaction_position,
)


class TestKinematics:
    """Verifies physical movement equations and boundary conditions."""

    def test_reaction_position_outfield_vs_gk(self):
        """Outfielders (tau=0.7s) displace further during reaction than GKs (tau=0.5s) at equal speed."""
        pos = np.array([[50.0, 34.0], [50.0, 34.0]])
        vel = np.array([[4.0, 0.0], [4.0, 0.0]])
        taus = np.array([DEFAULT_PHYSICS.tau_outfield, DEFAULT_PHYSICS.tau_goalkeeper])

        r_react = compute_reaction_position(pos, vel, taus)

        # Outfielder r_react_x = 50 + 4 * 0.7 = 52.8
        # GK r_react_x = 50 + 4 * 0.5 = 52.0
        assert np.isclose(r_react[0, 0], 52.8)
        assert np.isclose(r_react[1, 0], 52.0)
        assert r_react[0, 0] > r_react[1, 0]

    def test_stationary_player_reaction_position(self):
        """Stationary player (v=0) has reaction position equal to original position."""
        pos = np.array([[25.0, 15.0]])
        vel = np.array([[0.0, 0.0]])
        r_react = compute_reaction_position(pos, vel, 0.7)
        assert np.allclose(r_react, pos)

    def test_arrival_time_acceleration_phase(self):
        """For short distances d <= d_accel (8.0m), t_run = sqrt(2*d / a_max)."""
        d = np.array([2.0, 8.0])  # d_accel = 8^2 / (2*4) = 8.0m
        tau = 0.7
        # For d=2.0: t_run = sqrt(2*2 / 4) = 1.0s -> t_arrive = 1.7s
        # For d=8.0: t_run = sqrt(2*8 / 4) = 2.0s -> t_arrive = 2.7s
        t_arr = compute_player_arrival_time(d, tau, DEFAULT_PHYSICS)
        assert np.isclose(t_arr[0], 1.7)
        assert np.isclose(t_arr[1], 2.7)

    def test_arrival_time_cruising_phase(self):
        """For long distances d > d_accel, t_run = t_accel + (d - d_accel) / v_max."""
        # d = 24.0m: t_accel = 2.0s (covers 8m). Remaining 16m at 8m/s takes 2.0s. Total t_run = 4.0s.
        d = np.array([24.0])
        tau = 0.5
        t_arr = compute_player_arrival_time(d, tau, DEFAULT_PHYSICS)
        assert np.isclose(t_arr[0], 4.5)  # 0.5 + 4.0 = 4.5s

    def test_arrival_time_monotonicity(self):
        """Arrival time must strictly increase as distance increases."""
        distances = np.linspace(0.0, 80.0, 50)
        t_arr = compute_player_arrival_time(distances, 0.7, DEFAULT_PHYSICS)
        assert np.all(np.diff(t_arr) > 0.0)

    def test_zero_distance_equals_reaction_time(self):
        """At destination (d=0), arrival time is purely the reaction latency tau."""
        t_arr = compute_player_arrival_time(np.array([0.0]), 0.7, DEFAULT_PHYSICS)
        assert np.isclose(t_arr[0], 0.7)

    def test_ball_travel_time_linear_scaling(self):
        """Ball travel time scales linearly with distance: t = d / v_ball."""
        targets = np.array([[[15.0, 34.0], [30.0, 34.0]]])  # shape (1, 2, 2)
        ball_pos = np.array([0.0, 34.0])
        # v_ball = 15.0 m/s -> dist 15m takes 1.0s, dist 30m takes 2.0s
        t_ball = compute_ball_travel_time(targets, ball_pos, DEFAULT_PHYSICS)
        assert np.isclose(t_ball[0, 0], 1.0)
        assert np.isclose(t_ball[0, 1], 2.0)

    def test_kinematic_time_frame_consistency(self):
        """Verifies Spearman (2018) physical time-frame consistency.
        
        Reaction position r_react = r + v*tau represents player position at t = tau.
        Sprinting begins at t = tau towards target r_target covering distance d = ||r_target - r_react||.
        Hence t_arrival = tau + t_run(d) is the single coherent physical clock time.
        """
        r0 = np.array([40.0, 30.0])
        v0 = np.array([2.0, 0.0])
        tau = 0.7
        r_target = np.array([55.0, 30.0])

        r_react = compute_reaction_position(r0, v0, tau)
        # In tau=0.7s at 2m/s, player displaces 1.4m to x=41.4m
        assert np.allclose(r_react, [41.4, 30.0])

        dist_to_target = float(np.linalg.norm(r_target - r_react))
        assert np.isclose(dist_to_target, 13.6)

        t_arr = compute_player_arrival_time(np.array([dist_to_target]), tau, DEFAULT_PHYSICS)
        # d_accel = 8^2 / (2*4) = 8.0m (taking 2.0s). Remaining 5.6m at 8m/s takes 0.7s.
        # t_run = 2.0 + 0.7 = 2.7s. Total t_arr = 0.7 (reaction delay) + 2.7 (sprint) = 3.4s.
        assert np.isclose(t_arr[0], 3.4)

    def test_kinematic_negative_distance_clamped(self):
        """Negative distances must be safely clamped to zero."""
        d_neg = np.array([-5.0, 0.0])
        t_arr = compute_player_arrival_time(d_neg, 0.7, DEFAULT_PHYSICS)
        assert np.isclose(t_arr[0], 0.7)
        assert np.isclose(t_arr[1], 0.7)

    def test_kinematic_sensitivity_reaction_latency(self):
        """Sensitivity of arrival time to reaction latency tau: d(t_arr)/d(tau) == 1.0."""
        d = np.array([12.0])
        taus = np.linspace(0.2, 1.2, 6)
        times = [float(compute_player_arrival_time(d, tau, DEFAULT_PHYSICS)[0]) for tau in taus]
        slopes = np.diff(times) / np.diff(taus)
        assert np.allclose(slopes, 1.0, atol=1e-6)

    def test_kinematic_sensitivity_acceleration_and_velocity(self):
        """Higher acceleration and maximum speed monotonically decrease arrival times."""
        from app.config import PhysicsConfig

        d_long = np.array([30.0])
        tau = 0.7

        # Vary acceleration a_max
        t_low_a = float(compute_player_arrival_time(d_long, tau, PhysicsConfig(a_max=2.0, v_max=8.0))[0])
        t_high_a = float(compute_player_arrival_time(d_long, tau, PhysicsConfig(a_max=5.0, v_max=8.0))[0])
        assert t_high_a < t_low_a

        # Vary maximum speed v_max
        t_low_v = float(compute_player_arrival_time(d_long, tau, PhysicsConfig(a_max=4.0, v_max=6.0))[0])
        t_high_v = float(compute_player_arrival_time(d_long, tau, PhysicsConfig(a_max=4.0, v_max=10.0))[0])
        assert t_high_v < t_low_v

    def test_kinematic_rejects_non_finite_inputs(self):
        """Verify compute_reaction_position and compute_player_arrival_time reject NaN and Inf."""
        pos = np.array([[50.0, 34.0]])
        vel = np.array([[4.0, 0.0]])
        d = np.array([15.0])

        for bad in [np.nan, np.inf, -np.inf]:
            with pytest.raises(ValueError, match="positions and velocities must contain only finite numbers"):
                compute_reaction_position(np.array([[bad, 34.0]]), vel, 0.7)

            with pytest.raises(ValueError, match="distances must contain only finite numbers"):
                compute_player_arrival_time(np.array([bad]), 0.7, DEFAULT_PHYSICS)

            with pytest.raises(ValueError, match="tau must (contain only finite numbers|be a finite number)"):
                compute_player_arrival_time(d, bad, DEFAULT_PHYSICS)



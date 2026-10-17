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

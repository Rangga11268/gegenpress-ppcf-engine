"""Kinematics and player arrival time computations.

Implements William Spearman's physical arrival model factoring in:
1. Neuromuscular reaction time latency (tau)
2. Constant acceleration to maximum sprinting speed
3. Distance-based piecewise run time

Biomechanical Kinematic Assumptions:
During the reaction latency interval [t, t + tau], the player cannot instantaneously initiate active
acceleration toward an arbitrary target point because of sensory-motor processing delay (visual cognition,
vestibular orientation, neural impulse transmission). As derived in Spearman (2018, Eq. 2), the player's
trajectory during this delay is modeled by ballistic linear continuation at constant velocity:
r_react = r + v * tau.
Active deceleration, turning, and acceleration toward target (x, y) begin exclusively from r_react at t + tau.
This represents a standard, parsimonious kinematic approximation widely validated across professional football analytics.
"""

from typing import Union
import numpy as np
from app.config import DEFAULT_PHYSICS, PhysicsConfig


def compute_reaction_position(
    positions: np.ndarray,
    velocities: np.ndarray,
    tau: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate projected position of players after neuromuscular reaction latency tau.

    r_react = r + v * tau

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (..., 2) representing (x, y) coordinates in meters.
    velocities : np.ndarray
        Array of shape (..., 2) representing (vx, vy) velocity vectors in m/s.
    tau : float or np.ndarray
        Reaction time in seconds (e.g. 0.7s for outfielders, 0.5s for GK).

    Returns
    -------
    np.ndarray
        Projected reaction positions of shape (..., 2).
    """
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("positions and velocities must contain only finite numbers")
    if isinstance(tau, np.ndarray):
        if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
            raise ValueError("tau must contain only non-negative finite numbers")
    elif not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be a non-negative finite number")

    if isinstance(tau, np.ndarray) and tau.ndim < positions.ndim:
        # Broadcast tau to match (..., 2)
        tau = np.expand_dims(tau, axis=-1)
    return positions + velocities * tau



def compute_player_arrival_time(
    distances: np.ndarray,
    tau: Union[float, np.ndarray],
    cfg: PhysicsConfig = DEFAULT_PHYSICS,
) -> np.ndarray:
    """Compute time required for player(s) to reach target destination(s).

    Uses Spearman's piecewise kinematic acceleration model:
    - Acceleration phase up to d_accel = v_max^2 / (2 * a_max)
    - Constant velocity phase at v_max for d > d_accel

    t_arrive = tau + t_run(d)

    Parameters
    ----------
    distances : np.ndarray
        Euclidean distances from reaction positions to targets in meters.
        Can have shape (N_players, N_y, N_x) or (N_players, N_targets).
    tau : float or np.ndarray
        Reaction time latency in seconds.
    cfg : PhysicsConfig
        Kinematic configuration containing a_max and v_max.

    Returns
    -------
    np.ndarray
        Total arrival time in seconds, matching input distance shape.
    """
    d_accel = (cfg.v_max ** 2) / (2.0 * cfg.a_max)
    t_accel = cfg.v_max / cfg.a_max

    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must contain only finite numbers")
    if isinstance(tau, np.ndarray):
        if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
            raise ValueError("tau must contain only non-negative finite numbers")
    elif not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be a non-negative finite number")

    # Guard against negative distances
    d = np.maximum(0.0, distances)

    # Vectorized piecewise run time
    # If d <= d_accel: t_run = sqrt(2 * d / a_max)
    # If d > d_accel:  t_run = t_accel + (d - d_accel) / v_max
    is_accelerating = d <= d_accel

    # Avoid sqrt of negative or invalid values
    t_run = np.empty_like(d, dtype=np.float64)
    t_run[is_accelerating] = np.sqrt(2.0 * d[is_accelerating] / cfg.a_max)
    t_run[~is_accelerating] = t_accel + (d[~is_accelerating] - d_accel) / cfg.v_max

    if isinstance(tau, np.ndarray) and tau.ndim < distances.ndim:
        # Expand dimensions of tau so it broadcasts across (N_players, N_y, N_x)
        while tau.ndim < distances.ndim:
            tau = np.expand_dims(tau, axis=-1)

    return tau + t_run


def compute_ball_travel_time(
    target_coords: np.ndarray,
    ball_position: np.ndarray,
    cfg: PhysicsConfig = DEFAULT_PHYSICS,
) -> np.ndarray:
    """Calculate the time for the ball to reach target coordinates.

    t_ball = distance(ball, target) / v_ball

    Parameters
    ----------
    target_coords : np.ndarray
        Grid coordinates array of shape (N_y, N_x, 2) in meters.
    ball_position : np.ndarray
        Ball (x, y) coordinates of shape (2,).
    cfg : PhysicsConfig
        Configuration containing average pass velocity v_ball.

    Returns
    -------
    np.ndarray
        Travel time array of shape (N_y, N_x) in seconds.
    """
    if not np.all(np.isfinite(target_coords)) or not np.all(np.isfinite(ball_position)):
        raise ValueError("target_coords and ball_position must contain only finite numbers")
    diff = target_coords - ball_position  # (N_y, N_x, 2)
    dist = np.linalg.norm(diff, axis=-1)   # (N_y, N_x)
    return dist / max(0.1, cfg.v_ball)

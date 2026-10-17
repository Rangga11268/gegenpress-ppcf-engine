"""Tactical metrics and Gegenpressing analytical models.

Implements:
1. Spatial Dominance Contraction (DCR) - 3.0s post-turnover Gegenpress collapse
2. Cover Shadow Passing Occlusion Cone - Angular passing vector blockade
3. Counter-Attack Risk Surface (CARS) - High-line space threat & Sweeper-Keeper mitigation
4. Threat-Weighted Pitch Control (TWPC) - Expected Threat (xT) integration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH, PhysicsConfig, PitchConfig
from app.core.kinematics import compute_player_arrival_time, compute_reaction_position
from app.core.ppcf_engine import (
    FrameData,
    PlayerFrame,
    PPCFEngine,
    PPCFResult,
    generate_pitch_grid,
)


@dataclass
class DCRResult:
    """Result of Dominance Contraction Ratio calculation over a turnover window."""
    t_start: float
    t_end: float
    initial_opp_area_m2: float
    final_opp_area_m2: float
    contraction_ratio: float  # (initial - final) / initial in [0.0, 1.0]
    is_elite_gegenpress: bool  # True if contraction >= 65% in <= 3.0 seconds


@dataclass
class PassingLaneOcclusion:
    """Occlusion state for an intended passing target."""
    target_id: str
    target_pos: Tuple[float, float]
    defender_id: str
    defender_pos: Tuple[float, float]
    shadow_angle_deg: float
    target_angle_deg: float
    is_occluded: bool
    occlusion_prob: float  # 0.0 (open) to 1.0 (blocked)


@dataclass
class CARSResult:
    """Result of Counter-Attack Risk Surface and Sweeper-Keeper mitigation."""
    gross_risk_area_m2: float
    net_risk_area_m2: float
    sweeper_mitigated_area_m2: float
    sweeper_mitigation_pct: float
    risk_surface: np.ndarray  # (n_y, n_x) 2D array of net risk intensity in [0.0, 1.0]


# 12x8 Standard Karun Singh Expected Threat (xT) grid (attacking left to right: x=0 to x=105)
# Values represent probability of a goal occurring within next 5 actions from each zone
BASE_XT_12x8 = np.array([
    [0.006, 0.007, 0.008, 0.010, 0.012, 0.015, 0.019, 0.024, 0.032, 0.046, 0.075, 0.108],
    [0.007, 0.008, 0.009, 0.011, 0.013, 0.017, 0.022, 0.029, 0.041, 0.064, 0.114, 0.180],
    [0.008, 0.009, 0.010, 0.012, 0.015, 0.019, 0.026, 0.038, 0.059, 0.108, 0.257, 0.380],
    [0.009, 0.010, 0.011, 0.013, 0.016, 0.021, 0.030, 0.045, 0.078, 0.160, 0.350, 0.540],
    [0.009, 0.010, 0.011, 0.013, 0.016, 0.021, 0.030, 0.045, 0.078, 0.160, 0.350, 0.540],
    [0.008, 0.009, 0.010, 0.012, 0.015, 0.019, 0.026, 0.038, 0.059, 0.108, 0.257, 0.380],
    [0.007, 0.008, 0.009, 0.011, 0.013, 0.017, 0.022, 0.029, 0.041, 0.064, 0.114, 0.180],
    [0.006, 0.007, 0.008, 0.010, 0.012, 0.015, 0.019, 0.024, 0.032, 0.046, 0.075, 0.108],
], dtype=np.float64)


def get_interpolated_xt_grid(pitch_cfg: PitchConfig = DEFAULT_PITCH) -> np.ndarray:
    """Interpolate standard 12x8 Expected Threat (xT) matrix to full pitch grid resolution (n_y, n_x).

    Returns
    -------
    np.ndarray
        Array of shape (n_y, n_x) containing xT values in range [0.0, 1.0].
    """
    from scipy.ndimage import zoom

    zoom_y = pitch_cfg.n_y / BASE_XT_12x8.shape[0]
    zoom_x = pitch_cfg.n_x / BASE_XT_12x8.shape[1]
    xt_grid = zoom(BASE_XT_12x8, (zoom_y, zoom_x), order=1)
    # Ensure exact match in shape
    return np.clip(xt_grid[:pitch_cfg.n_y, :pitch_cfg.n_x], 0.0, 1.0)


def compute_spatial_dominance_collapse(
    frame_start: PPCFResult,
    frame_end: PPCFResult,
    cell_area_m2: float = 1.0,
) -> DCRResult:
    """Compute Dominance Contraction Ratio (DCR) over a post-turnover time window.

    Quantifies the rate at which Gegenpressing suffocates opposition controllable space:
    DCR = 1.0 - (Area_final / Area_initial)

    Parameters
    ----------
    frame_start : PPCFResult
        PPCF at the instant of turnover (t=0).
    frame_end : PPCFResult
        PPCF after pressing sequence (e.g. t=3.0s).
    cell_area_m2 : float
        Area of each grid cell in square meters (default 1.0m^2).

    Returns
    -------
    DCRResult
        Detailed metrics including contraction ratio and elite pressing indicator.
    """
    # Opponent (defending team in LFC's perspective) area is cells where ppcf_def > 0.5
    opp_cells_start = np.count_nonzero(frame_start.ppcf_def > 0.5)
    opp_cells_end = np.count_nonzero(frame_end.ppcf_def > 0.5)

    area_start = float(opp_cells_start * cell_area_m2)
    area_end = float(opp_cells_end * cell_area_m2)

    if area_start <= 1e-3:
        contraction = 0.0
    else:
        contraction = (area_start - area_end) / area_start
        contraction = float(np.clip(contraction, 0.0, 1.0))

    dt = frame_end.timestamp - frame_start.timestamp
    # Elite Gegenpress benchmark: >= 65% contraction within <= 3.5 seconds
    is_elite = (contraction >= 0.65) and (dt <= 3.5)

    return DCRResult(
        t_start=frame_start.timestamp,
        t_end=frame_end.timestamp,
        initial_opp_area_m2=area_start,
        final_opp_area_m2=area_end,
        contraction_ratio=contraction,
        is_elite_gegenpress=is_elite,
    )


def compute_cover_shadow_occlusion(
    ball_position: Tuple[float, float],
    pressing_defenders: List[PlayerFrame],
    passing_targets: List[PlayerFrame],
    tackle_radius: float = DEFAULT_PHYSICS.tackle_radius,
) -> List[PassingLaneOcclusion]:
    """Calculate geometric cover shadow occlusion cone for passing options.

    For each pressing defender closing on the ball carrier, checks if potential passing
    targets fall inside the occlusion angle:
    theta_shadow = 2 * arcsin(min(1.0, R_eff / distance(ball, defender)))

    Parameters
    ----------
    ball_position : Tuple[float, float]
        Current (x, y) coordinates of the ball carrier.
    pressing_defenders : List[PlayerFrame]
        Defending players pressing the ball.
    passing_targets : List[PlayerFrame]
        Available teammate receiving options.
    tackle_radius : float
        Effective intercept/tackle reach in meters (default 1.8m).

    Returns
    -------
    List[PassingLaneOcclusion]
        List of occlusion results for each passing lane.
    """
    results = []
    r_ball = np.array(ball_position, dtype=np.float64)

    for defender in pressing_defenders:
        r_def = np.array(defender.position, dtype=np.float64)
        v_def = r_def - r_ball
        d_def = float(np.linalg.norm(v_def))

        if d_def < 0.1:
            # Defender right on top of ball: full 360 occlusion
            half_cone = np.pi
        else:
            ratio = min(1.0, tackle_radius / d_def)
            half_cone = np.arcsin(ratio)

        u_def = v_def / max(1e-4, d_def)

        for target in passing_targets:
            r_tgt = np.array(target.position, dtype=np.float64)
            v_tgt = r_tgt - r_ball
            d_tgt = float(np.linalg.norm(v_tgt))

            dist_def_to_tgt = float(np.linalg.norm(r_def - r_tgt))
            if dist_def_to_tgt <= tackle_radius:
                # Defender is tightly marking the intended receiver: passing lane denied
                results.append(
                    PassingLaneOcclusion(
                        target_id=target.player_id,
                        target_pos=target.position,
                        defender_id=defender.player_id,
                        defender_pos=defender.position,
                        shadow_angle_deg=float(np.degrees(2.0 * half_cone)),
                        target_angle_deg=0.0,
                        is_occluded=True,
                        occlusion_prob=1.0,
                    )
                )
                continue

            if d_tgt <= d_def:
                # Target is closer to ball than defender: cannot be in defender's shadow
                results.append(
                    PassingLaneOcclusion(
                        target_id=target.player_id,
                        target_pos=target.position,
                        defender_id=defender.player_id,
                        defender_pos=defender.position,
                        shadow_angle_deg=float(np.degrees(2.0 * half_cone)),
                        target_angle_deg=0.0,
                        is_occluded=False,
                        occlusion_prob=0.0,
                    )
                )
                continue

            u_tgt = v_tgt / max(1e-4, d_tgt)
            # Dot product between unit vectors
            cos_theta = float(np.clip(np.dot(u_def, u_tgt), -1.0, 1.0))
            angle_rad = np.arccos(cos_theta)
            is_occluded = angle_rad <= half_cone

            # Sigmoid probability of occlusion based on angular proximity to cone edge
            if is_occluded:
                # Inside shadow cone: high probability
                prob = 1.0 - 0.5 * (angle_rad / max(1e-4, half_cone))
            else:
                margin = angle_rad - half_cone
                prob = max(0.0, 0.5 * np.exp(-margin / 0.2))

            results.append(
                PassingLaneOcclusion(
                    target_id=target.player_id,
                    target_pos=target.position,
                    defender_id=defender.player_id,
                    defender_pos=defender.position,
                    shadow_angle_deg=float(np.degrees(2.0 * half_cone)),
                    target_angle_deg=float(np.degrees(angle_rad)),
                    is_occluded=bool(is_occluded),
                    occlusion_prob=float(np.clip(prob, 0.0, 1.0)),
                )
            )

    return results


def compute_counter_attack_risk_surface(
    ppcf: PPCFResult,
    goalkeeper: Optional[PlayerFrame],
    opp_strikers: List[PlayerFrame],
    defensive_line_x: float,
    defending_goal_x: float = 0.0,
    pitch_cfg: PitchConfig = DEFAULT_PITCH,
    physics_cfg: PhysicsConfig = DEFAULT_PHYSICS,
) -> CARSResult:
    """Compute Counter-Attack Risk Surface (CARS) in the space behind a high defensive line.

    Quantifies the threat of direct balls over the top of center-backs, and models the
    mitigating effect of a high-line sweeper-keeper (e.g. Alisson Becker).

    CARS_gross(r) = (1 - PPCF_att(r)) * exp(-dist(r, goal)^2 / (2 * sigma_threat^2))
    CARS_net(r)   = CARS_gross(r) * I(t_arrive_GK(r) > min(t_arrive_Striker(r)))

    Parameters
    ----------
    ppcf : PPCFResult
        Computed pitch control surface for the frame.
    goalkeeper : Optional[PlayerFrame]
        Sweeper-keeper tracking state (e.g. Alisson).
    opp_strikers : List[PlayerFrame]
        Opponent fast runners/strikers chasing balls over the top.
    defensive_line_x : float
        Mean x-coordinate of the center-backs forming the high line.
    defending_goal_x : float
        X-coordinate of the defended goal (0.0 or 105.0).

    Returns
    -------
    CARSResult
        Gross risk area, net risk area after GK sweeping, and 2D net risk heatmap.
    """
    grid_coords, _, _ = generate_pitch_grid(pitch_cfg)
    xx = grid_coords[:, :, 0]
    yy = grid_coords[:, :, 1]
    n_y, n_x = xx.shape

    # 1. Identify space behind defensive line:
    if defending_goal_x == 0.0:
        # Defending left goal, attacking right: behind line is x < defensive_line_x
        behind_line_mask = xx < defensive_line_x
    else:
        # Defending right goal: behind line is x > defensive_line_x
        behind_line_mask = xx > defensive_line_x

    # 2. Distance to defended goal center (defending_goal_x, pitch_cfg.width / 2)
    goal_center = np.array([defending_goal_x, pitch_cfg.width / 2.0], dtype=np.float64)
    dist_to_goal = np.linalg.norm(grid_coords - goal_center, axis=-1)

    # 3. Spatial threat weighting: closer to goal has exponentially higher lethality
    sigma_threat = 25.0  # meters
    threat_weight = np.exp(-(dist_to_goal ** 2) / (2.0 * sigma_threat ** 2))

    # 4. Compute kinematic arrival times for GK and opponent strikers
    if goalkeeper is not None and len(opp_strikers) > 0:
        # Compute GK arrival time across grid
        gk_pos = np.array(goalkeeper.position, dtype=np.float64)
        gk_vel = np.array(goalkeeper.velocity, dtype=np.float64)
        gk_react = compute_reaction_position(gk_pos, gk_vel, physics_cfg.tau_goalkeeper)
        gk_dist = np.linalg.norm(grid_coords - gk_react, axis=-1)
        t_arrive_gk = compute_player_arrival_time(gk_dist, physics_cfg.tau_goalkeeper, physics_cfg)

        # Compute minimum arrival time across opponent strikers
        striker_arrive_times = []
        for s in opp_strikers:
            s_pos = np.array(s.position, dtype=np.float64)
            s_vel = np.array(s.velocity, dtype=np.float64)
            s_react = compute_reaction_position(s_pos, s_vel, physics_cfg.tau_outfield)
            s_dist = np.linalg.norm(grid_coords - s_react, axis=-1)
            t_arr = compute_player_arrival_time(s_dist, physics_cfg.tau_outfield, physics_cfg)
            striker_arrive_times.append(t_arr)

        min_t_arrive_strikers = np.min(np.stack(striker_arrive_times, axis=0), axis=0)

        # Space behind the high line where strikers can arrive in under 5.0s
        striker_accessible = (min_t_arrive_strikers < 5.0) & behind_line_mask
        gross_risk = np.where(striker_accessible, threat_weight, 0.0)

        # Sweeper-keeper neutralizes danger if GK arrives strictly before the striker
        gk_swept_mask = (t_arrive_gk <= min_t_arrive_strikers) & behind_line_mask
        net_risk = np.where(gk_swept_mask, 0.0, gross_risk)
    else:
        gross_risk = np.where(behind_line_mask, threat_weight, 0.0)
        net_risk = gross_risk

    gross_area = float(np.count_nonzero(gross_risk > 0.2) * (pitch_cfg.cell_size ** 2))
    net_area = float(np.count_nonzero(net_risk > 0.2) * (pitch_cfg.cell_size ** 2))
    mitigated_area = max(0.0, gross_area - net_area)
    mitigated_pct = (mitigated_area / gross_area * 100.0) if gross_area > 0.0 else 0.0

    return CARSResult(
        gross_risk_area_m2=gross_area,
        net_risk_area_m2=net_area,
        sweeper_mitigated_area_m2=mitigated_area,
        sweeper_mitigation_pct=float(mitigated_pct),
        risk_surface=net_risk,
    )


def compute_threat_weighted_pitch_control(
    ppcf: PPCFResult,
    pitch_cfg: PitchConfig = DEFAULT_PITCH,
) -> Tuple[np.ndarray, float]:
    """Compute Threat-Weighted Pitch Control (TWPC) combining PPCF with Expected Threat (xT).

    TWPC(r) = PPCF_att(r) * xT(r)

    Parameters
    ----------
    ppcf : PPCFResult
        Computed pitch control surface.
    pitch_cfg : PitchConfig
        Pitch configuration.

    Returns
    -------
    twpc_surface : np.ndarray
        2D grid of threat-weighted control of shape (n_y, n_x).
    total_twpc : float
        Sum of threat-weighted control across all pitch cells.
    """
    xt_grid = get_interpolated_xt_grid(pitch_cfg)
    twpc_surface = ppcf.ppcf_att * xt_grid
    total_twpc = float(np.sum(twpc_surface))
    return twpc_surface, total_twpc

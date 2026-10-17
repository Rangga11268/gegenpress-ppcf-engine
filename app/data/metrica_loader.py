"""Metrica Sports tracking data parser and tactical scenario generator.

Supports both loading raw Metrica format CSVs and generating deterministic,
reproducible Gegenpress turnover match sequences (25 FPS).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import DEFAULT_PITCH, PitchConfig
from app.core.ppcf_engine import FrameData, PlayerFrame
from app.data.coordinate_transformer import CanonicalCoordinateTransformer


def generate_synthetic_turnover_scenario(
    n_frames: int = 75,  # 3.0 seconds at 25 FPS
    fps: float = 25.0,
    pitch_cfg: PitchConfig = DEFAULT_PITCH,
) -> List[FrameData]:
    """Generate a realistic 3-second Gegenpress turnover sequence.

    Scenario Description (Liverpool 4-3-3 High Press vs Opposition Deep Build-up):
    - t=0.0s (Frame 0): Liverpool loses possession in the opposition half (x=70m, y=34m).
      Opposition center-back has the ball, with spread-out teammates.
    - t=0.0s -> 3.0s: Liverpool front-3 (Salah, Darwin, Diaz) and midfielders (Szoboszlai, Mac Allister)
      execute coordinated high-line pressing, collapsing the opposition's controllable space.
    - Defensive line (Van Dijk, Konate) holds a high line at x=52m.
    - Goalkeeper (Alisson Becker) sweeps high at x=28m.

    Returns
    -------
    List[FrameData]
        Sequence of 75 frames of synchronized tracking data.
    """
    frames = []
    dt = 1.0 / fps

    # Ball motion: trapped carrier forced back from x=68 to x=89
    ball_start = np.array([68.0, 34.0])
    ball_end = np.array([89.0, 34.0])

    # Liverpool players (4-3-3 high pressing surge forward into opposition third)
    # (id, is_gk, start_pos, end_pos)
    lfc_roster = [
        ("lfc_gk_alisson", True, [16.0, 34.0], [30.0, 34.0]),
        ("lfc_cb_vandijk", False, [35.0, 26.0], [62.0, 24.0]),
        ("lfc_cb_konate", False, [35.0, 42.0], [62.0, 44.0]),
        ("lfc_rb_trent", False, [38.0, 10.0], [65.0, 10.0]),
        ("lfc_lb_robertson", False, [38.0, 58.0], [65.0, 58.0]),
        ("lfc_dm_endo", False, [42.0, 34.0], [72.0, 34.0]),
        ("lfc_cm_macallister", False, [44.0, 20.0], [80.0, 22.0]),
        ("lfc_cm_szoboszlai", False, [44.0, 48.0], [80.0, 46.0]),
        ("lfc_rw_salah", False, [50.0, 15.0], [86.0, 20.0]),
        ("lfc_cf_darwin", False, [52.0, 34.0], [86.0, 34.0]),
        ("lfc_lw_diaz", False, [50.0, 53.0], [86.0, 48.0]),
    ]

    # Opposition players (initially open in midfield, then squeezed into deep defensive box)
    opp_roster = [
        ("opp_gk", True, [100.0, 34.0], [102.0, 34.0]),
        ("opp_cb_left", False, [82.0, 20.0], [97.0, 28.0]),
        ("opp_cb_right", False, [82.0, 48.0], [97.0, 40.0]),
        ("opp_cb_ballcarrier", False, [68.0, 34.0], [89.0, 34.0]),  # Pressed back into box
        ("opp_dm_pivot", False, [60.0, 18.0], [96.0, 22.0]),
        ("opp_cm_left", False, [60.0, 50.0], [96.0, 46.0]),
        ("opp_cm_right", False, [52.0, 12.0], [95.0, 16.0]),
        ("opp_rb", False, [74.0, 8.0], [92.0, 12.0]),
        ("opp_lb", False, [74.0, 60.0], [92.0, 56.0]),
        ("opp_cf_target", False, [52.0, 34.0], [88.0, 34.0]),
        ("opp_w_right", False, [52.0, 56.0], [95.0, 52.0]),
    ]

    duration = (n_frames - 1) * dt

    for f_idx in range(n_frames):
        alpha = f_idx / max(1, n_frames - 1)
        t = f_idx * dt

        # Smooth cubic ease-in-out motion: 3*alpha^2 - 2*alpha^3
        s_alpha = 3.0 * (alpha ** 2) - 2.0 * (alpha ** 3)

        # Ball motion
        ball_pos = ball_start + (ball_end - ball_start) * s_alpha

        players = []

        # Update Liverpool players
        for p_id, is_gk, start, end in lfc_roster:
            p_start = np.array(start, dtype=np.float64)
            p_end = np.array(end, dtype=np.float64)
            pos = p_start + (p_end - p_start) * s_alpha
            vel = (p_end - p_start) / max(0.1, duration)

            players.append(
                PlayerFrame(
                    player_id=p_id,
                    team="att",
                    position=(float(pos[0]), float(pos[1])),
                    velocity=(float(vel[0]), float(vel[1])),
                    is_goalkeeper=is_gk,
                )
            )

        # Update Opposition players
        for p_id, is_gk, start, end in opp_roster:
            p_start = np.array(start, dtype=np.float64)
            p_end = np.array(end, dtype=np.float64)
            pos = p_start + (p_end - p_start) * s_alpha
            vel = (p_end - p_start) / max(0.1, duration)

            players.append(
                PlayerFrame(
                    player_id=p_id,
                    team="def",
                    position=(float(pos[0]), float(pos[1])),
                    velocity=(float(vel[0]), float(vel[1])),
                    is_goalkeeper=is_gk,
                )
            )

        frames.append(
            FrameData(
                frame_idx=f_idx,
                timestamp=round(t, 4),
                ball_position=(float(ball_pos[0]), float(ball_pos[1])),
                ball_velocity=(float((ball_end[0] - ball_start[0]) / duration), 0.0),
                players=players,
            )
        )

    return frames

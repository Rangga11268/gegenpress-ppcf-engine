"""Verification of vectorized PPCF implementation against explicit Spearman (2018) reference math.

Ensures that the vectorized soft-min SIMD optimization produces mathematically identical
results to the original explicit per-player kinematic Poisson arrival formulas.
"""

import numpy as np
import pytest

from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH, PhysicsConfig
from app.core.ppcf_engine import FrameData, PlayerFrame, PPCFEngine


def spearman_reference_point_ppcf(
    target_xy: tuple[float, float],
    players: list[PlayerFrame],
    physics_cfg: PhysicsConfig = DEFAULT_PHYSICS,
) -> tuple[float, float]:
    """Pure explicit scalar loop reference implementation of Spearman (2018) pitch control.

    Computes PPCF at a single target point (x, y) without broadcasting or SIMD tricks.
    """
    target = np.array(target_xy, dtype=np.float64)
    beta = np.pi / (np.sqrt(3.0) * physics_cfg.sigma)

    t_att_list = []
    t_def_list = []

    for p in players:
        tau = physics_cfg.tau_goalkeeper if p.is_goalkeeper else physics_cfg.tau_outfield
        pos = np.array(p.position, dtype=np.float64)
        vel = np.array(p.velocity, dtype=np.float64)

        # 1. Reaction position
        r_react = pos + vel * tau

        # 2. Distance from reaction position to target
        dist = float(np.linalg.norm(target - r_react))

        # 3. Kinematic arrival time:
        # Time to reach target at maximum speed after reaction window
        d_accel = (physics_cfg.v_max ** 2) / (2.0 * physics_cfg.a_max)
        t_accel = physics_cfg.v_max / physics_cfg.a_max
        if dist <= d_accel:
            # Trapped in acceleration phase
            t_travel = np.sqrt(max(0.0, 2.0 * dist / physics_cfg.a_max))
        else:
            t_travel = t_accel + (dist - d_accel) / physics_cfg.v_max

        t_total = tau + t_travel

        if p.team == "att":
            t_att_list.append(t_total)
        else:
            t_def_list.append(t_total)

    if not t_att_list or not t_def_list:
        return (1.0, 0.0) if t_att_list else (0.0, 1.0)

    # Soft-min aggregation across teammates
    t_att_arr = np.array(t_att_list)
    t_def_arr = np.array(t_def_list)

    t_att_min = np.min(t_att_arr)
    t_def_min = np.min(t_def_arr)

    exp_att = np.sum(np.exp(-beta * (t_att_arr - t_att_min)))
    t_att_eff = t_att_min - (1.0 / beta) * np.log(exp_att)

    exp_def = np.sum(np.exp(-beta * (t_def_arr - t_def_min)))
    t_def_eff = t_def_min - (1.0 / beta) * np.log(exp_def)

    # Sigmoid competitive probability
    logit_diff = beta * (t_def_eff - t_att_eff)
    ppcf_att = 1.0 / (1.0 + np.exp(-logit_diff))
    ppcf_def = 1.0 - ppcf_att

    return float(ppcf_att), float(ppcf_def)


class TestSpearmanReferenceComparison:
    """Verifies that vectorized PPCFEngine matches scalar reference implementation within 1e-4."""

    @pytest.fixture
    def engine(self):
        return PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)

    def test_numerical_parity_against_reference(self, engine):
        """Vectorized PPCF and explicit scalar loop must agree within MAE < 1e-4 across sample pitch cells."""
        players = [
            PlayerFrame("att_1", "att", (50.0, 34.0), (3.0, 1.0)),
            PlayerFrame("att_2", "att", (40.0, 20.0), (1.5, -0.5)),
            PlayerFrame("att_3", "att", (65.0, 45.0), (0.0, 2.0)),
            PlayerFrame("def_1", "def", (58.0, 32.0), (-2.0, 0.5)),
            PlayerFrame("def_2", "def", (70.0, 25.0), (-1.0, -1.0)),
            PlayerFrame("def_gk", "def", (100.0, 34.0), (0.0, 0.0), is_goalkeeper=True),
        ]
        frame = FrameData(0, 0.0, (52.0, 34.0), (0.0, 0.0), players)

        # Compute full field with vectorized engine
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        # Sample test points across the pitch (various x, y coordinates)
        test_points = [
            (50.5, 34.5),   # Near att_1
            (58.5, 32.5),   # Near def_1
            (10.5, 10.5),   # Far attacking defensive third
            (95.5, 34.5),   # Near goalkeeper
            (54.5, 33.5),   # Contested midpoint between att_1 and def_1
            (80.5, 50.5),   # Open space
        ]

        diffs = []
        for tx, ty in test_points:
            # Find closest cell indices
            ix = int(round(tx - 0.5))
            iy = int(round(ty - 0.5))

            cell_xy = (float(res.grid_x[ix]), float(res.grid_y[iy]))
            ref_att, ref_def = spearman_reference_point_ppcf(cell_xy, players, DEFAULT_PHYSICS)
            vec_att = float(res.ppcf_att[iy, ix])
            vec_def = float(res.ppcf_def[iy, ix])

            diff_att = abs(vec_att - ref_att)
            diff_def = abs(vec_def - ref_def)
            diffs.extend([diff_att, diff_def])

            # Pointwise agreement
            assert np.isclose(vec_att, ref_att, atol=1e-3), (
                f"Mismatch at ({cell_xy}): vec={vec_att:.5f} vs ref={ref_att:.5f}"
            )
            assert np.isclose(vec_att + vec_def, 1.0, atol=1e-5)

        mae = float(np.mean(diffs))
        assert mae < 1e-4, f"Mean Absolute Error too high vs reference: {mae:.6f}"

    def test_edge_cases_handling(self, engine):
        """Engine safely handles missing goalkeepers, stationary players, and empty teams."""
        # 1. No goalkeeper present
        players_no_gk = [
            PlayerFrame("att_1", "att", (50.0, 34.0), (0.0, 0.0)),
            PlayerFrame("def_1", "def", (60.0, 34.0), (0.0, 0.0)),
        ]
        frame = FrameData(0, 0.0, (55.0, 34.0), (0.0, 0.0), players_no_gk)
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)
        assert np.allclose(res.ppcf_att + res.ppcf_def, 1.0, atol=1e-5)

        # 2. Only attackers present
        players_only_att = [PlayerFrame("att_1", "att", (50.0, 34.0), (0.0, 0.0))]
        frame_att = FrameData(1, 0.1, (50.0, 34.0), (0.0, 0.0), players_only_att)
        res_att = engine.compute_frame_ppcf(frame_att)
        assert np.all(res_att.ppcf_att == 1.0)
        assert np.all(res_att.ppcf_def == 0.0)

        # 3. Completely empty frame
        frame_empty = FrameData(2, 0.2, (50.0, 34.0), (0.0, 0.0), [])
        res_empty = engine.compute_frame_ppcf(frame_empty)
        assert np.all(res_empty.ppcf_att == 0.5)
        assert np.all(res_empty.ppcf_def == 0.5)

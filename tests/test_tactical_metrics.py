"""Unit tests for tactical Gegenpressing metrics: DCR, Cover Shadow, CARS, and TWPC."""

import numpy as np
import pytest

from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH
from app.core.ppcf_engine import FrameData, PlayerFrame, PPCFEngine, PPCFResult
from app.core.tactical_metrics import (
    compute_counter_attack_risk_surface,
    compute_cover_shadow_occlusion,
    compute_spatial_dominance_collapse,
    compute_threat_weighted_pitch_control,
    get_interpolated_xt_grid,
)


class TestTacticalMetrics:
    """Verifies tactical algorithms and mathematical edge cases."""

    def test_dcr_exact_contraction_calculation(self):
        """Dominance Contraction Ratio = 1.0 - (Area_final / Area_initial)."""
        grid_x = np.linspace(0.5, 104.5, 105)
        grid_y = np.linspace(0.5, 67.5, 68)

        # Initial: opponent controls 3,000 cells
        ppcf_def_0 = np.zeros((68, 105))
        ppcf_def_0.flat[:3000] = 0.8
        f0_res = PPCFResult(0, 0.0, grid_x, grid_y, 1.0 - ppcf_def_0, ppcf_def_0)

        # Final: opponent controls 900 cells
        ppcf_def_1 = np.zeros((68, 105))
        ppcf_def_1.flat[:900] = 0.8
        f1_res = PPCFResult(74, 3.0, grid_x, grid_y, 1.0 - ppcf_def_1, ppcf_def_1)

        dcr = compute_spatial_dominance_collapse(f0_res, f1_res)

        # (3000 - 900) / 3000 = 0.70 (70% contraction)
        assert np.isclose(dcr.contraction_ratio, 0.70)
        assert dcr.is_elite_gegenpress is True

    def test_dcr_zero_initial_area_guard(self):
        """Zero initial area guards against division by zero."""
        grid_x = np.linspace(0.5, 104.5, 105)
        grid_y = np.linspace(0.5, 67.5, 68)
        empty = np.zeros((68, 105))
        res_empty = PPCFResult(0, 0.0, grid_x, grid_y, np.ones((68, 105)), empty)

        dcr = compute_spatial_dominance_collapse(res_empty, res_empty)
        assert dcr.contraction_ratio == 0.0
        assert dcr.is_elite_gegenpress is False

    def test_cover_shadow_in_line_is_occluded(self):
        """Target directly behind defender relative to the ball is inside the shadow cone."""
        ball = (50.0, 34.0)
        # Defender is between ball and target along y=34
        defender = PlayerFrame("def_1", "att", (55.0, 34.0), (0.0, 0.0))  # 5m from ball
        target_in_line = PlayerFrame("tgt_1", "def", (65.0, 34.0), (0.0, 0.0))  # 15m from ball

        occlusions = compute_cover_shadow_occlusion(ball, [defender], [target_in_line])

        assert len(occlusions) == 1
        assert occlusions[0].is_occluded is True
        assert occlusions[0].occlusion_prob > 0.8

    def test_cover_shadow_orthogonal_is_open(self):
        """Target perpendicular to defender line of sight is not occluded."""
        ball = (50.0, 34.0)
        defender = PlayerFrame("def_1", "att", (55.0, 34.0), (0.0, 0.0))  # Along x
        target_orthogonal = PlayerFrame("tgt_ortho", "def", (50.0, 55.0), (0.0, 0.0))  # Along y

        occlusions = compute_cover_shadow_occlusion(ball, [defender], [target_orthogonal])

        assert len(occlusions) == 1
        assert occlusions[0].is_occluded is False
        assert occlusions[0].occlusion_prob < 0.1

    def test_cars_sweeper_keeper_mitigation(self):
        """Sweeper-keeper reduces gross vulnerability area in high-line counter-attack space."""
        engine = PPCFEngine()
        gk = PlayerFrame("alisson", "att", (18.0, 34.0), (0.0, 0.0), is_goalkeeper=True)
        striker = PlayerFrame("striker", "def", (42.0, 34.0), (-2.0, 0.0))

        frame = FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), [gk, striker])
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        cars = compute_counter_attack_risk_surface(
            ppcf=res,
            goalkeeper=gk,
            opp_strikers=[striker],
            defensive_line_x=45.0,
        )

        assert cars.gross_risk_area_m2 > 0.0
        assert cars.sweeper_mitigated_area_m2 > 0.0
        assert cars.net_risk_area_m2 <= cars.gross_risk_area_m2
        assert cars.sweeper_mitigation_pct > 0.0

    def test_xt_grid_interpolation_and_twpc(self):
        """Interpolated xT grid matches pitch resolution and yields positive TWPC."""
        xt_grid = get_interpolated_xt_grid(DEFAULT_PITCH)
        assert xt_grid.shape == (68, 105)
        # xT is highest in opposition penalty box (right side x -> 105)
        assert np.mean(xt_grid[:, 90:]) > np.mean(xt_grid[:, :20])

        # Test TWPC calculation
        engine = PPCFEngine()
        frame = FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), [
            PlayerFrame("p1", "att", (95.0, 34.0), (0.0, 0.0)),
            PlayerFrame("p2", "def", (10.0, 34.0), (0.0, 0.0)),
        ])
        res = engine.compute_frame_ppcf(frame)
        twpc_surface, total_twpc = compute_threat_weighted_pitch_control(res, DEFAULT_PITCH)

        assert twpc_surface.shape == (68, 105)
        assert total_twpc > 0.0

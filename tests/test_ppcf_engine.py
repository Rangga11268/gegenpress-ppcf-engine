"""Unit tests for PPCFEngine, probability conservation, and offside line masking."""

import numpy as np
import pytest

from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH
from app.core.ppcf_engine import (
    FrameData,
    PlayerFrame,
    PPCFEngine,
    generate_pitch_grid,
)


class TestPPCFEngine:
    """Verifies numerical properties and boundary invariants of the PPCF field."""

    @pytest.fixture
    def engine(self):
        return PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)

    def test_grid_generation_dimensions(self, engine):
        """Standard FIFA grid has 105 length, 68 width, resulting in 68 rows and 105 cols."""
        grid, x_coords, y_coords = generate_pitch_grid(DEFAULT_PITCH)
        assert grid.shape == (68, 105, 2)
        assert len(x_coords) == 105
        assert len(y_coords) == 68
        assert np.isclose(x_coords[0], 0.5)
        assert np.isclose(x_coords[-1], 104.5)

    def test_conservation_of_probability(self, engine):
        """Invariant: ppcf_att + ppcf_def == 1.0 across all 7,140 grid cells."""
        players = [
            PlayerFrame("p_att_1", "att", (50.0, 34.0), (2.0, 0.0)),
            PlayerFrame("p_def_1", "def", (60.0, 34.0), (-1.0, 0.0)),
            PlayerFrame("p_att_gk", "att", (16.0, 34.0), (0.0, 0.0), is_goalkeeper=True),
            PlayerFrame("p_def_gk", "def", (100.0, 34.0), (0.0, 0.0), is_goalkeeper=True),
        ]
        frame = FrameData(0, 0.0, (55.0, 34.0), (0.0, 0.0), players)

        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        total_prob = res.ppcf_att + res.ppcf_def
        assert np.allclose(total_prob, 1.0, atol=1e-5)
        assert np.all(res.ppcf_att >= 0.0)
        assert np.all(res.ppcf_att <= 1.0)
        assert np.all(res.ppcf_def >= 0.0)
        assert np.all(res.ppcf_def <= 1.0)

    def test_offside_line_calculation(self, engine):
        """Second-to-last defender sets the offside line (Law 11)."""
        def_pos = np.array([
            [102.0, 34.0],  # GK (deepest defender)
            [85.0, 24.0],   # Second-to-last defender -> offside line = 85.0
            [75.0, 40.0],   # CB
            [65.0, 15.0],   # Fullback
        ])
        def_gk = np.array([True, False, False, False])

        # Attacking left-to-right (direction=1)
        line_x = engine.calculate_offside_line(def_pos, def_gk, attack_direction=1)
        assert np.isclose(line_x, 85.0)

        # Attacking right-to-left (direction=-1)
        def_pos_rev = np.array([
            [2.0, 34.0],   # GK
            [20.0, 24.0],  # Second-to-last defender -> offside line = 20.0
            [35.0, 40.0],
        ])
        line_rev = engine.calculate_offside_line(def_pos_rev, def_gk[:3], attack_direction=-1)
        assert np.isclose(line_rev, 20.0)

    def test_offside_mask_clamps_attacking_control(self, engine):
        """When offside mask is active, attacking PPCF is clamped to 0.0 beyond offside line."""
        players = [
            PlayerFrame("att_striker", "att", (90.0, 34.0), (0.0, 0.0)),  # Offside striker
            PlayerFrame("def_cb", "def", (80.0, 34.0), (0.0, 0.0)),        # 2nd last defender (x=80)
            PlayerFrame("def_gk", "def", (102.0, 34.0), (0.0, 0.0), is_goalkeeper=True),
        ]
        # Ball is behind the offside line at x=50
        frame = FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), players)

        res_masked = engine.compute_frame_ppcf(frame, attack_direction=1, apply_offside_mask=True)

        # In zone x > 80m and x > ball_x (50m), attacking PPCF must be 0.0
        offside_zone = res_masked.grid_x > 80.0
        assert np.all(res_masked.ppcf_att[:, offside_zone] == 0.0)
        assert np.all(res_masked.ppcf_def[:, offside_zone] == 1.0)

    def test_spatial_proximity_dominance(self, engine):
        """A player located at (x, y) should have near 100% control over their immediate cell."""
        players = [
            PlayerFrame("att_p", "att", (30.0, 34.0), (0.0, 0.0)),
            PlayerFrame("def_p", "def", (80.0, 34.0), (0.0, 0.0)),
        ]
        frame = FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), players)

        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        # Attacker at cell (y=34, x=30)
        cell_y = int(34)
        cell_x = int(30)
        assert res.ppcf_att[cell_y, cell_x] > 0.95
        assert res.ppcf_def[cell_y, cell_x] < 0.05

        # Defender at cell (y=34, x=80)
        assert res.ppcf_def[cell_y, int(80)] > 0.95
        assert res.ppcf_att[cell_y, int(80)] < 0.05

    def test_offside_halfway_line_clamp(self, engine):
        """FIFA Law 11: Offside line cannot cross halfway line (52.5m)."""
        # Defenders deep in opponent's half (x < 52.5m)
        def_pos = np.array([
            [40.0, 34.0],  # GK
            [30.0, 24.0],  # 2nd last defender at 30m
        ])
        line_x = engine.calculate_offside_line(def_pos, attack_direction=1)
        assert np.isclose(line_x, 52.5), f"Expected halfway clamp 52.5, got {line_x}"

        # Attacking right-to-left: defenders deep in opponent half (x > 52.5m)
        def_pos_rev = np.array([
            [70.0, 34.0],
            [80.0, 24.0],
        ])
        line_rev = engine.calculate_offside_line(def_pos_rev, attack_direction=-1)
        assert np.isclose(line_rev, 52.5), f"Expected halfway clamp 52.5, got {line_rev}"

    def test_offside_behind_ball_restriction(self, engine):
        """FIFA Law 11: If ball is ahead of 2nd last defender, offside line is at the ball."""
        def_pos = np.array([
            [100.0, 34.0],
            [75.0, 24.0],  # 2nd last defender at 75m
        ])
        # Ball is at 82m (ahead of defender)
        line_x = engine.calculate_offside_line(def_pos, attack_direction=1, ball_position=(82.0, 34.0))
        assert np.isclose(line_x, 82.0)

        # Attacking right-to-left: ball at 20m, 2nd defender at 30m
        def_pos_rev = np.array([
            [5.0, 34.0],
            [30.0, 24.0],
        ])
        line_rev = engine.calculate_offside_line(def_pos_rev, attack_direction=-1, ball_position=(20.0, 34.0))
        assert np.isclose(line_rev, 20.0)

    def test_compute_frame_input_validation(self, engine):
        """Rejects unknown player teams and non-finite position/velocity coordinates."""
        invalid_team = [
            PlayerFrame("p1", "neutral", (50.0, 34.0), (0.0, 0.0)),
        ]
        with pytest.raises(ValueError, match="team must be 'att' or 'def'"):
            engine.compute_frame_ppcf(FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), invalid_team))

        nan_pos = [
            PlayerFrame("p1", "att", (np.nan, 34.0), (0.0, 0.0)),
        ]
        with pytest.raises(ValueError, match="position must contain finite numbers"):
            engine.compute_frame_ppcf(FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), nan_pos))

        with pytest.raises(ValueError, match="Ball position must contain finite numbers"):
            engine.compute_frame_ppcf(FrameData(0, 0.0, (np.inf, 34.0), (0.0, 0.0), []))

    def test_incomplete_players_asymmetric_squads(self, engine):
        """PPCF handles asymmetric squads (e.g. red card 10 vs 11) while conserving probability."""
        att_players = [
            PlayerFrame(f"att_{i}", "att", (40.0 + i * 2, 20.0 + i * 3), (1.0, 0.0))
            for i in range(11)
        ]
        # Defending team has only 9 outfielders + 1 GK (10 players total)
        def_players = [
            PlayerFrame(f"def_{i}", "def", (60.0 + i * 3, 20.0 + i * 3), (-1.0, 0.0))
            for i in range(9)
        ] + [PlayerFrame("def_gk", "def", (100.0, 34.0), (0.0, 0.0), is_goalkeeper=True)]

        frame = FrameData(1, 0.04, (55.0, 34.0), (0.0, 0.0), att_players + def_players)
        res = engine.compute_frame_ppcf(frame, attack_direction=1, apply_offside_mask=True)

        assert res.ppcf_att.shape == (68, 105)
        assert np.allclose(res.ppcf_att + res.ppcf_def, 1.0, atol=1e-5)
        assert 0.0 <= res.att_area_pct <= 1.0
        assert 0.0 <= res.def_area_pct <= 1.0

    def test_out_of_bounds_player_coordinates_robustness(self, engine):
        """Players outside pitch boundaries (e.g. momentum carry or celebrating) do not crash engine."""
        players = [
            PlayerFrame("p_out_left", "att", (-5.0, 34.0), (1.0, 0.0)),
            PlayerFrame("p_out_top", "def", (50.0, 75.0), (0.0, -1.0)),
            PlayerFrame("p_gk", "def", (102.0, 34.0), (0.0, 0.0), is_goalkeeper=True),
        ]
        frame = FrameData(2, 0.08, (50.0, 34.0), (0.0, 0.0), players)
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        assert np.all(np.isfinite(res.ppcf_att))
        assert np.all(np.isfinite(res.ppcf_def))
        assert np.allclose(res.ppcf_att + res.ppcf_def, 1.0, atol=1e-5)

    def test_ball_past_last_defender_breakaway(self, engine):
        """Breakaway scenario: when ball is carried past the deepest defender,
        offside line is set to the ball position (Law 11)."""
        def_pos = np.array([
            [95.0, 34.0],  # GK at 95m
            [80.0, 25.0],  # 2nd last defender at 80m
        ])
        # Striker dribbles ball to x=97m (past GK and all defenders)
        line_x = engine.calculate_offside_line(
            def_pos,
            attack_direction=1,
            ball_position=(97.0, 34.0),
        )
        assert np.isclose(line_x, 97.0), f"Offside line must be at ball position 97.0, got {line_x}"

    def test_attack_direction_validation(self, engine):
        """attack_direction must be strictly 1 or -1."""
        def_pos = np.array([[95.0, 34.0], [80.0, 25.0]])
        with pytest.raises(ValueError, match="attack_direction must be strictly 1 or -1"):
            engine.calculate_offside_line(def_pos, attack_direction=0)

        with pytest.raises(ValueError, match="attack_direction must be strictly 1 or -1"):
            engine.calculate_offside_line(def_pos, attack_direction=2)

        frame = FrameData(0, 0.0, (50.0, 34.0), (0.0, 0.0), [])
        with pytest.raises(ValueError, match="attack_direction must be strictly 1 or -1"):
            engine.compute_frame_ppcf(frame, attack_direction=0)

    def test_single_player_vs_n_players(self, engine):
        """1 attacking player vs 10 defenders maintains probability conservation and local control."""
        att = [PlayerFrame("att_lone", "att", (25.0, 34.0), (0.0, 0.0))]
        defs = [
            PlayerFrame(f"def_{i}", "def", (60.0 + (i % 3) * 5, 20.0 + (i // 3) * 10), (0.0, 0.0))
            for i in range(10)
        ]
        frame = FrameData(1, 0.0, (25.0, 34.0), (0.0, 0.0), att + defs)
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        # Probability conservation
        assert np.allclose(res.ppcf_att + res.ppcf_def, 1.0, atol=1e-5)
        # Lone attacker dominates their immediate cell
        assert res.ppcf_att[34, 25] > 0.85
        # Defenders dominate their territory
        assert res.ppcf_def[34, 70] > 0.90

    def test_overlapping_player_coordinates(self, engine):
        """Attacker and defender at exact identical coordinates (duel/collision) resolves without NaN."""
        players = [
            PlayerFrame("att_duel", "att", (50.0, 34.0), (0.0, 0.0)),
            PlayerFrame("def_duel", "def", (50.0, 34.0), (0.0, 0.0)),
        ]
        frame = FrameData(2, 0.0, (50.0, 34.0), (0.0, 0.0), players)
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        assert np.all(np.isfinite(res.ppcf_att))
        assert np.all(np.isfinite(res.ppcf_def))
        # Identical location and kinematics -> exactly 50/50 control at that cell
        assert np.isclose(res.ppcf_att[34, 50], 0.50, atol=1e-3)
        assert np.isclose(res.ppcf_def[34, 50], 0.50, atol=1e-3)

    def test_pitch_corner_boundary_coordinates(self, engine):
        """Players located at pitch corners (0,0), (105,0), (0,68), (105,68) produce valid fields."""
        corner_players = [
            PlayerFrame("att_corner_bl", "att", (0.0, 0.0), (0.0, 0.0)),
            PlayerFrame("att_corner_tl", "att", (0.0, 68.0), (0.0, 0.0)),
            PlayerFrame("def_corner_br", "def", (105.0, 0.0), (0.0, 0.0)),
            PlayerFrame("def_corner_tr", "def", (105.0, 68.0), (0.0, 0.0)),
        ]
        frame = FrameData(3, 0.0, (52.5, 34.0), (0.0, 0.0), corner_players)
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        assert np.all(np.isfinite(res.ppcf_att))
        assert np.all(np.isfinite(res.ppcf_def))
        assert np.allclose(res.ppcf_att + res.ppcf_def, 1.0, atol=1e-5)

    def test_symmetric_parity_across_halfway_line(self, engine):
        """Symmetric player layout produces symmetric control across halfway line (x=52.5m)."""
        players = [
            PlayerFrame("att_sym", "att", (32.5, 34.0), (0.0, 0.0)),
            PlayerFrame("def_sym", "def", (72.5, 34.0), (0.0, 0.0)),
        ]
        frame = FrameData(4, 0.0, (52.5, 34.0), (0.0, 0.0), players)
        res = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

        # At the exact midpoint x=52.5m, y=34m, control should be exactly 50%
        mid_x_idx = int(52.5)
        mid_y_idx = int(34.0)
        assert np.isclose(res.ppcf_att[mid_y_idx, mid_x_idx], 0.50, atol=0.01)
        assert np.isclose(res.ppcf_def[mid_y_idx, mid_x_idx], 0.50, atol=0.01)





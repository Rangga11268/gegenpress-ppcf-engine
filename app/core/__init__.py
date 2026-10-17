"""Core modules for GegenPress-PPCF Engine."""

from app.core.kinematics import (
    compute_ball_travel_time,
    compute_player_arrival_time,
    compute_reaction_position,
)
from app.core.ppcf_engine import (
    FrameData,
    PlayerFrame,
    PPCFEngine,
    PPCFResult,
    generate_pitch_grid,
)
from app.core.tactical_metrics import (
    CARSResult,
    DCRResult,
    PassingLaneOcclusion,
    compute_counter_attack_risk_surface,
    compute_cover_shadow_occlusion,
    compute_spatial_dominance_collapse,
    compute_threat_weighted_pitch_control,
    get_interpolated_xt_grid,
)

__all__ = [
    "compute_reaction_position",
    "compute_player_arrival_time",
    "compute_ball_travel_time",
    "PlayerFrame",
    "FrameData",
    "PPCFResult",
    "generate_pitch_grid",
    "PPCFEngine",
    "DCRResult",
    "PassingLaneOcclusion",
    "CARSResult",
    "compute_spatial_dominance_collapse",
    "compute_cover_shadow_occlusion",
    "compute_counter_attack_risk_surface",
    "compute_threat_weighted_pitch_control",
    "get_interpolated_xt_grid",
]

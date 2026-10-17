"""Pydantic schemas for the GegenPress-PPCF REST Telemetry API."""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class PlayerInput(BaseModel):
    player_id: str = Field(..., description="Unique player identifier")
    team: str = Field(..., description="'att' for attacking or 'def' for defending")
    x: float = Field(..., description="X coordinate in meters [0.0, 105.0]")
    y: float = Field(..., description="Y coordinate in meters [0.0, 68.0]")
    vx: float = Field(0.0, description="Velocity X in m/s")
    vy: float = Field(0.0, description="Velocity Y in m/s")
    is_goalkeeper: bool = Field(False, description="True if player is goalkeeper")


class FrameInput(BaseModel):
    frame_idx: int = Field(0, description="Frame index")
    timestamp: float = Field(0.0, description="Timestamp in seconds")
    ball_x: float = Field(..., description="Ball X coordinate in meters")
    ball_y: float = Field(..., description="Ball Y coordinate in meters")
    players: List[PlayerInput] = Field(default_factory=list, description="List of active players on pitch")


class PPCFResponse(BaseModel):
    frame_idx: int
    timestamp: float
    offside_line: Optional[float]
    att_area_pct: float
    def_area_pct: float
    total_twpc: float
    latency_ms: float


class DCRRequest(BaseModel):
    frame_start: FrameInput
    frame_end: FrameInput


class DCRResponse(BaseModel):
    t_start: float
    t_end: float
    initial_opp_area_m2: float
    final_opp_area_m2: float
    contraction_ratio: float
    is_elite_gegenpress: bool
    latency_ms: float


class CARSRequest(BaseModel):
    frame: FrameInput
    defensive_line_x: float = Field(52.0, description="Mean x-coordinate of the high defensive line")


class CARSResponse(BaseModel):
    gross_risk_area_m2: float
    net_risk_area_m2: float
    sweeper_mitigated_area_m2: float
    sweeper_mitigation_pct: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    pitch_dimensions: Tuple[float, float]

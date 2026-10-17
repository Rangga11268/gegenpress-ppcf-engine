"""FastAPI route handlers for GegenPress-PPCF streaming & telemetry endpoints."""

import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    CARSRequest,
    CARSResponse,
    DCRRequest,
    DCRResponse,
    FrameInput,
    HealthResponse,
    PPCFResponse,
)
from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH
from app.core.ppcf_engine import FrameData, PPCFEngine, PlayerFrame
from app.core.tactical_metrics import (
    compute_counter_attack_risk_surface,
    compute_spatial_dominance_collapse,
    compute_threat_weighted_pitch_control,
)

app = FastAPI(
    title="GegenPress-PPCF Analytics Engine",
    description="Vectorized Spatio-Temporal Potential Pitch Control & Tactical Pressing API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)


def _convert_frame_input(inp: FrameInput) -> FrameData:
    players = [
        PlayerFrame(
            player_id=p.player_id,
            team=p.team,
            position=(p.x, p.y),
            velocity=(p.vx, p.vy),
            is_goalkeeper=p.is_goalkeeper,
        )
        for p in inp.players
    ]
    return FrameData(
        frame_idx=inp.frame_idx,
        timestamp=inp.timestamp,
        ball_position=(inp.ball_x, inp.ball_y),
        players=players,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health status and engine capabilities endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        engine="Vectorized 4D NumPy Spearman PPCF",
        pitch_dimensions=(DEFAULT_PITCH.length, DEFAULT_PITCH.width),
    )


@app.post("/v1/pitch-control", response_model=PPCFResponse)
async def compute_pitch_control(request: FrameInput):
    """Compute potential pitch control field and threat metrics for a single frame."""
    t0 = time.perf_counter()
    frame = _convert_frame_input(request)
    result = engine.compute_frame_ppcf(frame, attack_direction=1, apply_offside_mask=True)
    _, total_twpc = compute_threat_weighted_pitch_control(result, DEFAULT_PITCH)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PPCFResponse(
        frame_idx=result.frame_idx,
        timestamp=result.timestamp,
        offside_line=result.offside_line,
        att_area_pct=result.att_area_pct,
        def_area_pct=result.def_area_pct,
        total_twpc=round(total_twpc, 4),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/v1/metrics/dcr", response_model=DCRResponse)
async def evaluate_dcr(request: DCRRequest):
    """Evaluate Dominance Contraction Ratio (DCR) over a post-turnover window."""
    t0 = time.perf_counter()
    f_start = _convert_frame_input(request.frame_start)
    f_end = _convert_frame_input(request.frame_end)

    res_start = engine.compute_frame_ppcf(f_start)
    res_end = engine.compute_frame_ppcf(f_end)

    dcr = compute_spatial_dominance_collapse(res_start, res_end)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return DCRResponse(
        t_start=dcr.t_start,
        t_end=dcr.t_end,
        initial_opp_area_m2=round(dcr.initial_opp_area_m2, 2),
        final_opp_area_m2=round(dcr.final_opp_area_m2, 2),
        contraction_ratio=round(dcr.contraction_ratio, 4),
        is_elite_gegenpress=dcr.is_elite_gegenpress,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/v1/metrics/cars", response_model=CARSResponse)
async def evaluate_cars(request: CARSRequest):
    """Evaluate Counter-Attack Risk Surface (CARS) and Sweeper-Keeper mitigation."""
    t0 = time.perf_counter()
    frame = _convert_frame_input(request.frame)
    res = engine.compute_frame_ppcf(frame)

    gk = next((p for p in frame.players if p.team == "att" and p.is_goalkeeper), None)
    strikers = [p for p in frame.players if p.team == "def" and not p.is_goalkeeper]

    cars = compute_counter_attack_risk_surface(
        ppcf=res,
        goalkeeper=gk,
        opp_strikers=strikers,
        defensive_line_x=request.defensive_line_x,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return CARSResponse(
        gross_risk_area_m2=round(cars.gross_risk_area_m2, 2),
        net_risk_area_m2=round(cars.net_risk_area_m2, 2),
        sweeper_mitigated_area_m2=round(cars.sweeper_mitigated_area_m2, 2),
        sweeper_mitigation_pct=round(cars.sweeper_mitigation_pct, 2),
        latency_ms=round(latency_ms, 2),
    )

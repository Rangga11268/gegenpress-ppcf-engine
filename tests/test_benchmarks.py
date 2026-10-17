import time
import numpy as np
import pytest
from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH
from app.core.ppcf_engine import PPCFEngine
from app.data.metrica_loader import generate_synthetic_turnover_scenario


def test_per_frame_latency_under_50ms():
    """Verify that a full 105x68 pitch control frame executes well below 50ms real-time latency on CPU."""
    engine = PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)
    frames = generate_synthetic_turnover_scenario(n_frames=5)

    # Warmup
    _ = engine.compute_frame_ppcf(frames[0], apply_offside_mask=True)

    latencies = []
    for f in frames:
        t0 = time.perf_counter()
        _ = engine.compute_frame_ppcf(f, apply_offside_mask=True)
        t_el = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_el)

    median_lat = float(np.median(latencies))
    # Standard broadcast streaming requires < 50ms (25-30 FPS)
    assert median_lat < 50.0, f"Expected median latency < 50ms, got {median_lat:.2f}ms"


def test_throughput_fps_exceeds_broadcast_standard():
    """Verify that execution throughput exceeds broadcast 25 FPS requirement."""
    engine = PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)
    frames = generate_synthetic_turnover_scenario(n_frames=10)

    t0 = time.perf_counter()
    for f in frames:
        _ = engine.compute_frame_ppcf(f, apply_offside_mask=True)
    total_time = time.perf_counter() - t0
    fps = len(frames) / total_time
    assert fps >= 25.0, f"Expected >= 25 FPS, achieved {fps:.1f} FPS"

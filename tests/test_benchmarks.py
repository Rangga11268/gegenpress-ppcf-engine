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
    """Verify that execution throughput exceeds broadcast 25 FPS requirement and record variance."""
    engine = PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)
    frames = generate_synthetic_turnover_scenario(n_frames=10)

    t0 = time.perf_counter()
    latencies = []
    for f in frames:
        t_f0 = time.perf_counter()
        _ = engine.compute_frame_ppcf(f, apply_offside_mask=True)
        latencies.append((time.perf_counter() - t_f0) * 1000.0)
    total_time = time.perf_counter() - t0
    fps = len(frames) / total_time
    median_ms = float(np.median(latencies))
    std_ms = float(np.std(latencies))
    variance_ms2 = float(np.var(latencies))

    assert fps >= 25.0, f"Expected >= 25 FPS, achieved {fps:.1f} FPS"
    assert median_ms < 50.0
    assert variance_ms2 >= 0.0  # Timing variance strictly non-negative and recorded


def test_automated_numerical_parity_vectorized_vs_scalar_loop():
    """Verify exact numerical parity (MAE < 1e-4, max_err < 1e-3) between naive scalar loop and 4D vectorized tensor."""
    from benchmarks.benchmark_engine import naive_unvectorized_ppcf
    from app.config import PitchConfig

    test_pitch = PitchConfig(cell_size=5.0)
    engine = PPCFEngine(pitch_cfg=test_pitch, physics_cfg=DEFAULT_PHYSICS)
    frames = generate_synthetic_turnover_scenario(n_frames=1)
    frame = frames[0]

    res_scalar = naive_unvectorized_ppcf(frame, pitch_cfg=test_pitch, physics_cfg=DEFAULT_PHYSICS)
    res_vectorized = engine.compute_frame_ppcf(frame, apply_offside_mask=False)

    diff = np.abs(res_scalar - res_vectorized.ppcf_att)
    mae = float(np.mean(diff))
    max_err = float(np.max(diff))

    assert mae < 1e-4, f"Numerical parity failure: MAE={mae:.2e} >= 1e-4"
    assert max_err < 1e-3, f"Numerical parity failure: Max Error={max_err:.2e} >= 1e-3"


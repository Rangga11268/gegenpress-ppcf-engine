import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH
from app.core.kinematics import compute_player_arrival_time, compute_reaction_position
from app.core.ppcf_engine import FrameData, PPCFEngine, generate_pitch_grid
from app.data.metrica_loader import generate_synthetic_turnover_scenario


def naive_unvectorized_ppcf(frame: FrameData, grid_nx: int = 35, grid_ny: int = 23) -> np.ndarray:
    """Simulates naive cell-by-cell player-by-player nested loop evaluation."""
    x_coords = np.linspace(0.5, 104.5, grid_nx)
    y_coords = np.linspace(0.5, 67.5, grid_ny)
    ppcf_att = np.zeros((grid_ny, grid_nx), dtype=np.float64)

    att_players = [p for p in frame.players if p.team == "att"]
    def_players = [p for p in frame.players if p.team == "def"]

    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            target = np.array([x, y])
            min_t_att = 999.0
            min_t_def = 999.0

            for p in att_players:
                tau = DEFAULT_PHYSICS.tau_goalkeeper if p.is_goalkeeper else DEFAULT_PHYSICS.tau_outfield
                r_pos = compute_reaction_position(np.array(p.position), np.array(p.velocity), tau)
                dist = np.linalg.norm(r_pos - target)
                t_tot = float(compute_player_arrival_time(np.array([dist]), tau, DEFAULT_PHYSICS)[0])
                if t_tot < min_t_att:
                    min_t_att = t_tot

            for p in def_players:
                tau = DEFAULT_PHYSICS.tau_goalkeeper if p.is_goalkeeper else DEFAULT_PHYSICS.tau_outfield
                r_pos = compute_reaction_position(np.array(p.position), np.array(p.velocity), tau)
                dist = np.linalg.norm(r_pos - target)
                t_tot = float(compute_player_arrival_time(np.array([dist]), tau, DEFAULT_PHYSICS)[0])
                if t_tot < min_t_def:
                    min_t_def = t_tot

            # Logistic transition
            ppcf_att[j, i] = 1.0 / (1.0 + np.exp(-3.0 * (min_t_def - min_t_att)))

    return ppcf_att


def run_benchmark():
    print("=" * 76)
    print("   GEGENPRESS-PPCF: COMPUTATIONAL ACCELERATION PROFILER")
    print("      Vectorized 4D NumPy Tensor Broadcasting vs Naive Looping")
    print("=" * 76)

    engine = PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)
    frames = generate_synthetic_turnover_scenario(n_frames=10)
    test_frame = frames[0]

    # Benchmark 1: Naive Loop on coarse grid (35x23 = 805 cells)
    print("\n[1/3] Benchmarking Naive Nested Loops (Coarse 35x23 grid = 805 cells)...")
    t0 = time.perf_counter()
    _ = naive_unvectorized_ppcf(test_frame, grid_nx=35, grid_ny=23)
    t_naive_coarse = (time.perf_counter() - t0) * 1000.0
    # Extrapolate to full 105x68 grid (7,140 cells)
    extrapolated_naive_full = t_naive_coarse * (7140.0 / 805.0)
    print(f"      Naive Coarse Grid (805 cells)   : {t_naive_coarse:.2f} ms")
    print(f"      Extrapolated Full (7,140 cells) : {extrapolated_naive_full:.2f} ms (~{extrapolated_naive_full/1000.0:.2f} s/frame)")

    # Benchmark 2: Vectorized 4D NumPy on Full Grid (105x68 = 7,140 cells)
    print("\n[2/3] Benchmarking Vectorized 4D Engine (Full FIFA 105x68 grid = 7,140 cells)...")
    # Warmup
    _ = engine.compute_frame_ppcf(test_frame)

    latencies = []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = engine.compute_frame_ppcf(test_frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_vec = float(np.median(latencies))
    p95_vec = float(np.percentile(latencies, 95))
    fps = 1000.0 / median_vec
    speedup = extrapolated_naive_full / median_vec

    print(f"      Vectorized Median Latency : {median_vec:.2f} ms")
    print(f"      Vectorized P95 Latency    : {p95_vec:.2f} ms")
    print(f"      Real-Time Throughput      : {fps:.1f} FPS")
    print(f"      Effective Speedup Ratio   : {speedup:.1f}x")

    # Benchmark 3: Verification against Broadcast Real-Time Constraint
    print("\n[3/3] Broadcast Real-Time Feasibility Audit:")
    is_realtime = fps >= 25.0
    print(f"      Broadcast Standard (>= 25 FPS)  : {'PASSED' if is_realtime else 'FAILED'}")
    print(f"      Per-Frame Latency Budget (<50ms): {'PASSED' if median_vec < 50.0 else 'FAILED'}")
    print("=" * 76)


if __name__ == "__main__":
    run_benchmark()

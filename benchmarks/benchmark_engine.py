import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import DEFAULT_PHYSICS, DEFAULT_PITCH, PhysicsConfig, PitchConfig
from app.core.kinematics import compute_player_arrival_time, compute_reaction_position
from app.core.ppcf_engine import FrameData, PPCFEngine, generate_pitch_grid
from app.data.metrica_loader import generate_synthetic_turnover_scenario


def naive_unvectorized_ppcf(
    frame: FrameData,
    pitch_cfg: PitchConfig = DEFAULT_PITCH,
    physics_cfg: PhysicsConfig = DEFAULT_PHYSICS,
) -> np.ndarray:
    """Exact scalar loop reference implementation of Spearman (2018) soft-min PPCF.

    Uses identical mathematical formulas and parameters to the vectorized engine:
    1. Projected reaction positions r_react = r + v * tau
    2. Piecewise kinematic arrival time: tau + t_run(d)
    3. Exponential soft-min team arrival time: t_eff = t_min - (1/beta)*ln(sum exp(-beta*(t - t_min)))
    4. Competitive sigmoid probability: 1 / (1 + exp(-beta * (t_def_eff - t_att_eff)))
    """
    _, x_coords, y_coords = generate_pitch_grid(pitch_cfg)
    grid_ny = len(y_coords)
    grid_nx = len(x_coords)
    ppcf_att = np.zeros((grid_ny, grid_nx), dtype=np.float64)

    beta = np.pi / (np.sqrt(3.0) * physics_cfg.sigma)
    d_accel = (physics_cfg.v_max ** 2) / (2.0 * physics_cfg.a_max)
    t_accel = physics_cfg.v_max / physics_cfg.a_max

    att_players = [p for p in frame.players if p.team == "att"]
    def_players = [p for p in frame.players if p.team == "def"]

    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            target = np.array([x, y], dtype=np.float64)

            # 1. Attacking players arrival times
            t_att_list = []
            for p in att_players:
                tau = physics_cfg.tau_goalkeeper if p.is_goalkeeper else physics_cfg.tau_outfield
                pos = np.array(p.position, dtype=np.float64)
                vel = np.array(p.velocity, dtype=np.float64)
                r_react = pos + vel * tau
                dist = float(np.linalg.norm(r_react - target))
                if dist <= d_accel:
                    t_travel = np.sqrt(max(0.0, 2.0 * dist / physics_cfg.a_max))
                else:
                    t_travel = t_accel + (dist - d_accel) / physics_cfg.v_max
                t_att_list.append(tau + t_travel)

            # 2. Defending players arrival times
            t_def_list = []
            for p in def_players:
                tau = physics_cfg.tau_goalkeeper if p.is_goalkeeper else physics_cfg.tau_outfield
                pos = np.array(p.position, dtype=np.float64)
                vel = np.array(p.velocity, dtype=np.float64)
                r_react = pos + vel * tau
                dist = float(np.linalg.norm(r_react - target))
                if dist <= d_accel:
                    t_travel = np.sqrt(max(0.0, 2.0 * dist / physics_cfg.a_max))
                else:
                    t_travel = t_accel + (dist - d_accel) / physics_cfg.v_max
                t_def_list.append(tau + t_travel)

            if not t_att_list:
                ppcf_att[j, i] = 0.0
                continue
            if not t_def_list:
                ppcf_att[j, i] = 1.0
                continue

            # 3. Soft-min team effective arrival times
            t_att_arr = np.array(t_att_list, dtype=np.float64)
            t_def_arr = np.array(t_def_list, dtype=np.float64)

            t_att_min = np.min(t_att_arr)
            t_def_min = np.min(t_def_arr)

            exp_att = np.sum(np.exp(-beta * (t_att_arr - t_att_min)))
            t_att_eff = t_att_min - (1.0 / beta) * np.log(exp_att)

            exp_def = np.sum(np.exp(-beta * (t_def_arr - t_def_min)))
            t_def_eff = t_def_min - (1.0 / beta) * np.log(exp_def)

            # 4. Competitive probability
            logit_diff = beta * (t_def_eff - t_att_eff)
            ppcf_att[j, i] = 1.0 / (1.0 + np.exp(-logit_diff))

    return ppcf_att


def run_benchmark():
    print("=" * 76)
    print("   GEGENPRESS-PPCF: COMPUTATIONAL ACCELERATION PROFILER")
    print("      Vectorized 4D NumPy Tensor Broadcasting vs Naive Looping")
    print("=" * 76)

    engine = PPCFEngine(pitch_cfg=DEFAULT_PITCH, physics_cfg=DEFAULT_PHYSICS)
    frames = generate_synthetic_turnover_scenario(n_frames=10)
    test_frame = frames[0]

    # Benchmark 1: Exact Numerical Parity Audit on Identical Math
    print("\n[1/3] Benchmarking Numerical Parity (Identical Spearman Math on Coarse Grid)...")
    coarse_pitch = PitchConfig(cell_size=5.0)
    coarse_engine = PPCFEngine(pitch_cfg=coarse_pitch, physics_cfg=DEFAULT_PHYSICS)
    res_naive_sample = naive_unvectorized_ppcf(
        test_frame, pitch_cfg=coarse_pitch, physics_cfg=DEFAULT_PHYSICS
    )
    res_vec_sample = coarse_engine.compute_frame_ppcf(test_frame, apply_offside_mask=False)
    mae = float(np.mean(np.abs(res_naive_sample - res_vec_sample.ppcf_att)))
    max_err = float(np.max(np.abs(res_naive_sample - res_vec_sample.ppcf_att)))
    print(f"      Grid Resolution (cell_size=5m)  : {coarse_pitch.n_x}x{coarse_pitch.n_y} ({coarse_pitch.total_cells} cells)")
    print(f"      Numerical Parity Check          : {'PASSED' if max_err < 1e-4 else 'FAILED'}")
    print(f"      Mean Absolute Error (MAE)       : {mae:.2e}")
    print(f"      Max Coordinate Discrepancy      : {max_err:.2e}")

    # Benchmark 2: Scaled Timing on Identical Algorithmic Formulation
    print("\n[2/3] Benchmarking Runtime Scaling (Vectorized 4D Tensor vs Naive Loop)...")
    coarse_timing_pitch = PitchConfig(cell_size=3.0)
    t0 = time.perf_counter()
    _ = naive_unvectorized_ppcf(test_frame, pitch_cfg=coarse_timing_pitch, physics_cfg=DEFAULT_PHYSICS)
    t_naive_coarse = (time.perf_counter() - t0) * 1000.0
    extrapolated_naive_full = t_naive_coarse * (7140.0 / coarse_timing_pitch.total_cells)
    print(f"      Naive Loop Coarse ({coarse_timing_pitch.total_cells} cells)  : {t_naive_coarse:.2f} ms")
    print(f"      Extrapolated Full (7,140 cells) : {extrapolated_naive_full:.2f} ms (~{extrapolated_naive_full/1000.0:.2f} s/frame)")

    # Multi-iteration Benchmark with Warmup
    print("      Executing 10 warmup iterations...")
    for _ in range(10):
        _ = engine.compute_frame_ppcf(test_frame)

    print("      Measuring 50 production iterations...")
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = engine.compute_frame_ppcf(test_frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    min_vec = float(np.min(latencies))
    median_vec = float(np.median(latencies))
    mean_vec = float(np.mean(latencies))
    max_vec = float(np.max(latencies))
    std_vec = float(np.std(latencies))
    var_vec = float(np.var(latencies))
    p95_vec = float(np.percentile(latencies, 95))
    p99_vec = float(np.percentile(latencies, 99))
    fps = 1000.0 / median_vec
    speedup = extrapolated_naive_full / median_vec

    print(f"      Latency Min / Median / Mean     : {min_vec:.2f} / {median_vec:.2f} / {mean_vec:.2f} ms")
    print(f"      Latency Max / P95 / P99         : {max_vec:.2f} / {p95_vec:.2f} / {p99_vec:.2f} ms")
    print(f"      Variance / StdDev               : {var_vec:.4f} ms^2 (+/- {std_vec:.2f} ms)")
    print(f"      Real-Time Throughput            : {fps:.1f} FPS (Median)")
    print(f"      Effective Speedup Ratio         : {speedup:.1f}x")

    # Benchmark 3: Verification against Broadcast Real-Time Constraint
    print("\n[3/3] Broadcast Real-Time Feasibility Audit:")
    is_realtime = fps >= 25.0
    print(f"      Broadcast Standard (>= 25 FPS)  : {'PASSED' if is_realtime else 'FAILED'}")
    print(f"      Per-Frame Latency Budget (<50ms): {'PASSED' if median_vec < 50.0 else 'FAILED'}")
    print("=" * 76)


if __name__ == "__main__":
    run_benchmark()

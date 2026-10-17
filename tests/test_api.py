import pytest
from fastapi.testclient import TestClient
from app.api.routes import app
from app.api.schemas import PlayerInput, FrameInput

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["engine"] == "Vectorized 4D NumPy Spearman PPCF"


def test_pitch_control_endpoint():
    payload = {
        "frame_idx": 1,
        "timestamp": 0.04,
        "ball_x": 52.5,
        "ball_y": 34.0,
        "players": [
            {"player_id": "A1", "team": "att", "x": 60.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "is_goalkeeper": False},
            {"player_id": "A2", "team": "att", "x": 75.0, "y": 20.0, "vx": 1.0, "vy": 0.0, "is_goalkeeper": False},
            {"player_id": "D1", "team": "def", "x": 58.0, "y": 34.0, "vx": -1.0, "vy": 0.0, "is_goalkeeper": False},
            {"player_id": "GK", "team": "def", "x": 102.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "is_goalkeeper": True},
        ],
    }
    response = client.post("/v1/pitch-control", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["frame_idx"] == 1
    assert "offside_line" in data
    assert "att_area_pct" in data
    assert "def_area_pct" in data
    assert "total_twpc" in data
    assert "latency_ms" in data
    assert data["latency_ms"] < 100.0


def test_metrics_dcr_endpoint():
    from app.data.metrica_loader import generate_synthetic_turnover_scenario
    frames = generate_synthetic_turnover_scenario(n_frames=75)
    f0, f1 = frames[0], frames[-1]

    def _to_dict(f):
        return {
            "frame_idx": f.frame_idx,
            "timestamp": f.timestamp,
            "ball_x": f.ball_position[0],
            "ball_y": f.ball_position[1],
            "players": [
                {
                    "player_id": p.player_id,
                    "team": p.team,
                    "x": p.position[0],
                    "y": p.position[1],
                    "vx": p.velocity[0],
                    "vy": p.velocity[1],
                    "is_goalkeeper": p.is_goalkeeper,
                }
                for p in f.players
            ],
        }

    payload = {"frame_start": _to_dict(f0), "frame_end": _to_dict(f1)}
    response = client.post("/v1/metrics/dcr", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "contraction_ratio" in data
    assert "initial_opp_area_m2" in data
    assert "final_opp_area_m2" in data
    assert "is_elite_gegenpress" in data
    assert data["contraction_ratio"] >= 0.65
    assert data["is_elite_gegenpress"] is True


def test_metrics_cars_endpoint():
    frame = {
        "frame_idx": 10,
        "timestamp": 0.4,
        "ball_x": 50.0,
        "ball_y": 34.0,
        "players": [
            {"player_id": "A_GK", "team": "att", "x": 28.0, "y": 34.0, "vx": 2.0, "vy": 0.0, "is_goalkeeper": True},
            {"player_id": "A_CB", "team": "att", "x": 52.0, "y": 30.0, "vx": 0.0, "vy": 0.0, "is_goalkeeper": False},
            {"player_id": "D_ST", "team": "def", "x": 45.0, "y": 34.0, "vx": 3.0, "vy": 0.0, "is_goalkeeper": False},
        ],
    }
    payload = {"frame": frame, "defensive_line_x": 52.0}
    response = client.post("/v1/metrics/cars", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "gross_risk_area_m2" in data
    assert "net_risk_area_m2" in data
    assert "sweeper_mitigated_area_m2" in data
    assert "sweeper_mitigation_pct" in data

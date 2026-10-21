"""Configuration and physical constants for GegenPress-PPCF Engine.

Based on Dr. William Spearman (2018) 'Beyond Expected Goals: Resolving the Potential
Pitch Control Field in Soccer' (MIT Sloan Sports Analytics Conference) and Liverpool FC
tactical spatial pressing frameworks.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchConfig:
    """Standard FIFA Pitch dimensions in meters."""
    length: float = 105.0  # x-axis (0.0 to 105.0)
    width: float = 68.0    # y-axis (0.0 to 68.0)
    cell_size: float = 1.0 # 1 meter spatial grid resolution
    x_min: float = 0.0
    x_max: float = 105.0
    y_min: float = 0.0
    y_max: float = 68.0

    @property
    def n_x(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell_size))

    @property
    def n_y(self) -> int:
        return int(round((self.y_max - self.y_min) / self.cell_size))

    @property
    def total_cells(self) -> int:
        return self.n_x * self.n_y


@dataclass(frozen=True)
class PhysicsConfig:
    """Kinematics and stochastic parameters for pitch control calculation."""
    # Reaction latencies (seconds)
    tau_outfield: float = 0.7
    tau_goalkeeper: float = 0.5

    # Player movement capabilities
    a_max: float = 4.0  # Maximum acceleration (m/s^2)
    v_max: float = 8.0  # Maximum sprinting velocity (m/s)

    # Ball dynamics
    v_ball: float = 15.0  # Mean pass ball speed (m/s)

    # Spearman stochastic parameters
    sigma: float = 0.45       # Arrival distribution dispersion (seconds)
    lambda_peak: float = 4.3  # Poisson hazard rate intensity (s^-1)

    # Time integration parameters (truncated horizon)
    horizon_seconds: float = 2.0  # Integration window past ball arrival
    dt: float = 0.1               # Time discretization step (seconds)

    # Tactical geometry
    tackle_radius: float = 1.8   # Effective tackle/intercept radius (meters)
    post_turnover_window: float = 3.0  # Spatial dominance collapse window (seconds)

    def __post_init__(self) -> None:
        import math
        for field_name in ["a_max", "v_max", "v_ball", "tau_outfield", "tau_goalkeeper", "sigma", "lambda_peak", "horizon_seconds", "dt"]:
            val = getattr(self, field_name)
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"PhysicsConfig parameter '{field_name}' must be a strictly positive finite number, got {val}")



DEFAULT_PITCH = PitchConfig()
DEFAULT_PHYSICS = PhysicsConfig()

FIFA_PITCH_LENGTH: float = DEFAULT_PITCH.length
FIFA_PITCH_WIDTH: float = DEFAULT_PITCH.width

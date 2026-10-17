"""Coordinate transformations between tracking systems and standard FIFA metric pitch."""

from typing import Tuple, Union
import numpy as np

from app.config import DEFAULT_PITCH, PitchConfig


class CanonicalCoordinateTransformer:
    """Transforms normalized tracking coordinates [0, 1] to metric FIFA pitch [0, 105]x[0, 68].

    Metrica Sports uses:
    - x in [0.0, 1.0] from left goal line to right goal line
    - y in [0.0, 1.0] from top touchline to bottom touchline (inverted y-axis)

    FIFA Standard Cartesian uses:
    - x in [0.0, 105.0] meters
    - y in [0.0, 68.0] meters (bottom touchline = 0, top touchline = 68)
    """

    def __init__(self, pitch_cfg: PitchConfig = DEFAULT_PITCH):
        self.pitch_cfg = pitch_cfg

    def metrica_to_metric(
        self,
        coords: Union[Tuple[float, float], np.ndarray],
    ) -> np.ndarray:
        """Convert Metrica [0, 1] normalized coordinates to meters.

        Parameters
        ----------
        coords : Tuple[float, float] or np.ndarray
            Array of shape (..., 2) with values in range [0.0, 1.0].

        Returns
        -------
        np.ndarray
            Array of shape (..., 2) with coordinates in meters.
        """
        arr = np.asarray(coords, dtype=np.float64)
        x_norm = arr[..., 0]
        y_norm = arr[..., 1]

        x_m = x_norm * self.pitch_cfg.length
        # Invert y because Metrica y=0 is top touchline, whereas standard y=0 is bottom touchline
        y_m = (1.0 - y_norm) * self.pitch_cfg.width

        return np.stack([x_m, y_m], axis=-1)

    def metric_to_metrica(
        self,
        coords: Union[Tuple[float, float], np.ndarray],
    ) -> np.ndarray:
        """Convert metric coordinates (meters) back to Metrica [0, 1] normalized range.

        Parameters
        ----------
        coords : Tuple[float, float] or np.ndarray
            Coordinates in meters of shape (..., 2).

        Returns
        -------
        np.ndarray
            Normalized coordinates in [0.0, 1.0].
        """
        arr = np.asarray(coords, dtype=np.float64)
        x_m = arr[..., 0]
        y_m = arr[..., 1]

        x_norm = x_m / self.pitch_cfg.length
        y_norm = 1.0 - (y_m / self.pitch_cfg.width)

        return np.stack([x_norm, y_norm], axis=-1)

    def invert_attack_direction(
        self,
        coords: Union[Tuple[float, float], np.ndarray],
    ) -> np.ndarray:
        """Flip pitch 180 degrees so that the reference team always attacks left-to-right (x -> 105).

        x' = length - x
        y' = width - y
        """
        arr = np.asarray(coords, dtype=np.float64)
        x_flip = self.pitch_cfg.length - arr[..., 0]
        y_flip = self.pitch_cfg.width - arr[..., 1]
        return np.stack([x_flip, y_flip], axis=-1)

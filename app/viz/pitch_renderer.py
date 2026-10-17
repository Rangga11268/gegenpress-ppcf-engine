"""Native FIFA standard pitch renderer and tactical heatmap visualizer using Matplotlib.

Zero-external-dependency rendering with full pitch geometry (penalty areas, center circle,
penalty arcs, penalty spots) and overlays for PPCF surfaces, velocity quivers, and cover shadows.
"""

from typing import List, Optional, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from app.config import DEFAULT_PITCH, PitchConfig
from app.core.ppcf_engine import FrameData, PPCFResult


def draw_pitch(
    ax: plt.Axes,
    pitch_cfg: PitchConfig = DEFAULT_PITCH,
    pitch_color: str = "#1a472a",  # Anfield deep pitch green
    line_color: str = "white",
    line_width: float = 1.5,
) -> plt.Axes:
    """Draw a standard FIFA football pitch on a matplotlib Axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to render pitch onto.
    pitch_cfg : PitchConfig
        Dimensions of the pitch (default 105m x 68m).
    pitch_color : str
        Background grass color.
    line_color : str
        Color of pitch markings.
    line_width : float
        Line thickness.

    Returns
    -------
    plt.Axes
        Configured matplotlib axes.
    """
    length = pitch_cfg.length
    width = pitch_cfg.width

    ax.set_facecolor(pitch_color)
    ax.set_xlim(-2.0, length + 2.0)
    ax.set_ylim(-2.0, width + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer boundary
    outer_box = patches.Rectangle(
        (0.0, 0.0), length, width,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(outer_box)

    # Halfway line
    ax.plot([length / 2.0, length / 2.0], [0.0, width], color=line_color, linewidth=line_width, zorder=2)

    # Center circle (radius = 9.15m)
    center_circle = patches.Circle(
        (length / 2.0, width / 2.0), 9.15,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(center_circle)
    # Center spot
    ax.scatter(length / 2.0, width / 2.0, color=line_color, s=20, zorder=3)

    # Left Penalty Area: 16.5m x 40.32m
    pen_y_min = (width - 40.32) / 2.0
    left_pen = patches.Rectangle(
        (0.0, pen_y_min), 16.5, 40.32,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(left_pen)

    # Left 6-yard Box: 5.5m x 18.32m
    box_y_min = (width - 18.32) / 2.0
    left_six = patches.Rectangle(
        (0.0, box_y_min), 5.5, 18.32,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(left_six)

    # Left Penalty Spot (11m) and Arc
    ax.scatter(11.0, width / 2.0, color=line_color, s=15, zorder=3)
    left_arc = patches.Arc(
        (11.0, width / 2.0), 18.3, 18.3,
        angle=0, theta1=308, theta2=52, color=line_color, linewidth=line_width, zorder=2,
    )
    ax.add_patch(left_arc)

    # Right Penalty Area
    right_pen = patches.Rectangle(
        (length - 16.5, pen_y_min), 16.5, 40.32,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(right_pen)

    # Right 6-yard Box
    right_six = patches.Rectangle(
        (length - 5.5, box_y_min), 5.5, 18.32,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(right_six)

    # Right Penalty Spot (11m from right goal) and Arc
    ax.scatter(length - 11.0, width / 2.0, color=line_color, s=15, zorder=3)
    right_arc = patches.Arc(
        (length - 11.0, width / 2.0), 18.3, 18.3,
        angle=0, theta1=128, theta2=232, color=line_color, linewidth=line_width, zorder=2,
    )
    ax.add_patch(right_arc)

    # Goals (depth = 2m, width = 7.32m)
    goal_y_min = (width - 7.32) / 2.0
    left_goal = patches.Rectangle(
        (-2.0, goal_y_min), 2.0, 7.32,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    right_goal = patches.Rectangle(
        (length, goal_y_min), 2.0, 7.32,
        linewidth=line_width, edgecolor=line_color, facecolor="none", zorder=2,
    )
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)

    return ax


def render_ppcf_frame(
    frame: FrameData,
    ppcf_res: PPCFResult,
    pitch_cfg: PitchConfig = DEFAULT_PITCH,
    att_color: str = "#c8102e",     # Liverpool FC Red
    def_color: str = "#004890",     # Deep Royal Blue
    show_quivers: bool = True,
    show_offside: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a comprehensive tactical PPCF heatmap and player overlay.

    Parameters
    ----------
    frame : FrameData
        Tracking frame data.
    ppcf_res : PPCFResult
        Computed pitch control result.
    pitch_cfg : PitchConfig
        Pitch dimensions.
    att_color : str
        Jersey color for attacking team.
    def_color : str
        Jersey color for defending team.
    show_quivers : bool
        Whether to draw velocity direction arrows.
    show_offside : bool
        Whether to render the offside threshold line.
    figsize : Tuple[int, int]
        Figure width and height in inches.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1713")
    draw_pitch(ax, pitch_cfg=pitch_cfg)

    # 1. Overlay Pitch Control 2D Contour
    # ppcf_res.ppcf_att has shape (n_y, n_x)
    xx, yy = np.meshgrid(ppcf_res.grid_x, ppcf_res.grid_y)
    contour = ax.contourf(
        xx, yy, ppcf_res.ppcf_att,
        levels=np.linspace(0.0, 1.0, 21),
        cmap="coolwarm",
        alpha=0.60,
        zorder=1,
    )

    # 2. Render Offside Line
    if show_offside and ppcf_res.offside_line is not None:
        ax.axvline(
            ppcf_res.offside_line,
            color="#ffd700",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            zorder=3,
            label=f"Offside Line (x={ppcf_res.offside_line:.1f}m)",
        )

    # 3. Render Players
    for p in frame.players:
        x, y = p.position
        vx, vy = p.velocity

        if p.team == "att":
            color = "#ffb81c" if p.is_goalkeeper else att_color  # Gold for GK, red for outfield
            edge_color = "white"
        else:
            color = "#00ffcc" if p.is_goalkeeper else def_color  # Cyan for GK, blue for outfield
            edge_color = "black"

        ax.scatter(x, y, color=color, s=120, edgecolors=edge_color, linewidth=1.5, zorder=5)

        # Draw velocity vectors
        if show_quivers and (abs(vx) > 0.1 or abs(vy) > 0.1):
            ax.quiver(
                x, y, vx, vy,
                angles="xy", scale_units="xy", scale=1.5,
                color="white", width=0.004, zorder=6,
            )

    # 4. Render Ball
    bx, by = frame.ball_position
    ax.scatter(bx, by, color="white", s=80, edgecolors="black", linewidth=2.0, zorder=7)

    # Title & Metadata Annotation
    title_text = (
        f"GegenPress-PPCF Field | Frame {frame.frame_idx} (t={frame.timestamp:.2f}s)\n"
        f"Liverpool Control: {ppcf_res.att_area_pct * 100.0:.1f}% | "
        f"Opposition Control: {ppcf_res.def_area_pct * 100.0:.1f}%"
    )
    ax.set_title(title_text, color="white", fontsize=12, pad=12, fontweight="bold")

    plt.tight_layout()
    return fig, ax

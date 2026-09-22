"""
LiuZhen SmallCage 四类标准图（只读移植布局/配色）：

1. com_trajectory_heatmap  — MotionTrajectoryandHeatmap.py
2. body_center             — MotionStatu.up_down_analyze
3. position_analysis       — MotionPositionAnalysis 六宫格 All Days
4. behavior_comparison     — BehaviorStateVisualization.plot_proportion_summary
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import save_figure
from ..io.schema import parse_session_name

# MotionTrajectoryandHeatmap 轨迹配色（含 Turning）
TRAJ_COLORS = {
    0: "black",
    2: "green",
    3: "blue",
    4: "yellow",
}
TRAJ_LABELS = {0: "Static", 2: "Walking", 3: "Climbing", 4: "Turning"}

# BehaviorStateVisualization
BEHAVIOR_COLORS = {
    "Static": "#9E9E9E",
    "Walking": "#64B5F6",
    "Climbing": "#81C784",
}
BEHAVIOR_ORDER = ["Static", "Walking", "Climbing"]
LOCOMOTION_ORDER = ["Walking", "Climbing"]
LOCOMOTION_PATTERNS = (2, 3)

HEATMAP_EXTENT = [150, 650, 350, 50]


def simple_com(df: pd.DataFrame) -> pd.DataFrame:
    """SmallCage 固定三点均值重心（非加权）。"""
    out = df.copy()
    out["center_x"] = (out["Head_x"] + out["Back_x"] + out["Tail(Root)_x"]) / 3
    out["center_y"] = (out["Head_y"] + out["Back_y"] + out["Tail(Root)_y"]) / 3
    return out


def _clip_frames(n: int, max_frames: int) -> int:
    if max_frames and max_frames > 0:
        return min(n, int(max_frames))
    return n


def _segment_frame_bounds(segments: pd.DataFrame, window_size: int, max_frames: int) -> pd.DataFrame:
    segs = segments.copy()
    if "start_frame" not in segs.columns:
        segs["start_frame"] = segs["start"].astype(int) * window_size
        segs["end_frame"] = segs["end"].astype(int) * window_size
    segs = segs[segs["start_frame"] < max_frames].copy()
    if segs.empty:
        return segs
    segs.loc[segs["end_frame"] > max_frames, "end_frame"] = max_frames
    return segs


def _resolve_background(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    return None


def plot_com_trajectory_heatmap(
    keypoints: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
    fps: float = 25.0,
    window_size: int = 13,
    max_sec: float = 1800.0,
    pixel_to_cm: float = 0.3202,
    background_image: Optional[str] = None,
) -> list:
    """Center of Mass Trajectory + Residence Time Heatmap（并排）。"""
    max_frames = int(max_sec * fps) if max_sec and max_sec > 0 else len(keypoints)
    df = simple_com(keypoints.iloc[:_clip_frames(len(keypoints), max_frames)])
    segs = _segment_frame_bounds(segments, window_size, len(df))
    n = len(df)

    total_distance_px = 0.0
    for _, seg in segs.iterrows():
        pattern = int(seg["pattern"]) if "pattern" in segs.columns else 0
        if pattern not in LOCOMOTION_PATTERNS:
            continue
        a = max(0, min(int(seg["start_frame"]), n - 1))
        b = max(a, min(int(seg["end_frame"]), n - 1))
        traj = df.iloc[a : b + 1]
        dx = traj["center_x"].diff()
        dy = traj["center_y"].diff()
        total_distance_px += float(np.sqrt(dx**2 + dy**2).sum())
    total_distance_cm = total_distance_px * pixel_to_cm

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    for _, seg in segs.iterrows():
        pattern = int(seg["pattern"]) if "pattern" in segs.columns else 0
        a = max(0, min(int(seg["start_frame"]), n - 1))
        b = max(a, min(int(seg["end_frame"]), n - 1))
        traj = df.iloc[a : b + 1]
        ax1.plot(
            traj["center_x"],
            traj["center_y"],
            color=TRAJ_COLORS.get(pattern, "gray"),
            alpha=0.7,
            linewidth=1,
        )
    ax1.text(
        0.02,
        0.98,
        f"Total Distance: {total_distance_cm:.2f} cm ({total_distance_px:.2f} pixels)",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax1.set_title("Center of Mass Trajectory")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.invert_yaxis()
    legend_elements = [
        plt.Line2D([0], [0], color=TRAJ_COLORS[i], label=TRAJ_LABELS[i])
        for i in (0, 2, 3, 4)
    ]
    ax1.legend(handles=legend_elements)

    ax2 = fig.add_subplot(122)
    H, _, _ = np.histogram2d(df["center_x"].dropna(), df["center_y"].dropna(), bins=[50, 50])
    bg = _resolve_background(background_image)
    if bg is not None:
        background = plt.imread(str(bg))
        ax2.imshow(background, extent=HEATMAP_EXTENT, alpha=0.6)
    vmax = np.percentile(H, 95) if H.max() > 0 else 1
    im = ax2.imshow(
        H.T,
        alpha=0.4,
        cmap="YlOrRd",
        extent=HEATMAP_EXTENT,
        norm=plt.Normalize(vmin=0, vmax=vmax),
    )
    fig.colorbar(im, ax=ax2, label="Counts")
    ax2.set_title("Residence Time Heatmap")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")

    fig.tight_layout()
    return save_figure(fig, Path(output_dir) / f"{stem}_com_trajectory_heatmap", formats)


def plot_body_center(
    keypoints: pd.DataFrame,
    *,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
    fps: float = 25.0,
    max_sec: float = 1800.0,
    climbing_y: float = 202.0,
    width: float = 720.0,
    height: float = 406.0,
) -> list:
    """Body Center Position Tracking + Over Time。"""
    max_frames = int(max_sec * fps) if max_sec and max_sec > 0 else len(keypoints)
    df = simple_com(keypoints.iloc[:_clip_frames(len(keypoints), max_frames)])
    center_x = df["center_x"]
    center_y = df["center_y"]
    up_mask = center_y <= climbing_y
    down_mask = center_y > climbing_y

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].scatter(center_x[up_mask], center_y[up_mask], color="green", s=10, label="Center (Up)")
    axs[0].scatter(center_x[down_mask], center_y[down_mask], color="red", s=10, label="Center (Down)")
    axs[0].set_title("Body Center Position Tracking")
    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")
    axs[0].axhline(y=climbing_y, color="gray", linestyle="--", label="Boundary")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlim(0, width)
    axs[0].set_ylim(0, height if height >= 404 else 404)
    axs[0].invert_yaxis()

    frames = np.arange(len(df)) / float(fps)
    axs[1].plot(frames, center_y, color="blue", label="Center Position")
    axs[1].scatter(frames[up_mask], center_y[up_mask], color="green", s=10, label="Up")
    axs[1].scatter(frames[down_mask], center_y[down_mask], color="red", s=10, label="Down")
    axs[1].set_title("Body Center Position Over Time")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Y Position")
    axs[1].axhline(y=climbing_y, color="gray", linestyle="--", label="Boundary")
    axs[1].legend()
    axs[1].grid()
    axs[1].invert_yaxis()

    fig.tight_layout()
    return save_figure(fig, Path(output_dir) / f"{stem}_body_center", formats)


def _locomotion_frame_mask(n_frames: int, segments: pd.DataFrame, window_size: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    if segments is None or segments.empty:
        return mask
    segs = _segment_frame_bounds(segments, window_size, n_frames)
    for _, row in segs.iterrows():
        if int(row["pattern"]) not in LOCOMOTION_PATTERNS:
            continue
        a = max(0, int(row["start_frame"]))
        b = min(n_frames, int(row["end_frame"]) + 1)
        if b > a:
            mask[a:b] = True
    return mask


def _frame_distance_cm(
    df: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    window_size: int,
    pixel_to_cm: float,
) -> pd.Series:
    # 与 MotionPositionAnalysis.compute_frame_distance 一致（含字面阈值混用）。
    dx = df["center_x"].diff()
    dy = df["center_y"].diff()
    dist = (np.sqrt(dx**2 + dy**2) * pixel_to_cm).fillna(0.0)
    thresh = 40.0 / pixel_to_cm if pixel_to_cm else 1e9
    dist = dist.where(dist <= thresh, 0.0)
    loco = _locomotion_frame_mask(len(df), segments, window_size)
    return dist.where(loco, 0.0)


def _session_meta(stem: str) -> dict:
    info = parse_session_name(stem) or {}
    return {
        "animal_id": info.get("animal_id") or stem,
        "day_label": info.get("day_label") or stem,
        "day_num": info.get("day_num"),
        "sort_key": info.get("day_num") if info.get("day_num") is not None else stem,
        "stem": info.get("stem") or stem,
    }


def plot_position_analysis(
    sessions: Sequence[Mapping],
    *,
    output_dir: Path,
    formats: Sequence[str],
    fps: float = 25.0,
    window_size: int = 13,
    max_sec: float = 1800.0,
    pixel_to_cm: float = 0.3202,
    width: float = 720.0,
    height: float = 406.0,
    stem: str = "all_sessions",
) -> list:
    """
    Monkey Position Analysis — All Class/Days 六宫格。
    sessions: [{stem, keypoints, segments}, ...]
    """
    if not sessions:
        return []

    max_frames = int(max_sec * fps) if max_sec and max_sec > 0 else 0
    x_th, y_th = width / 2, height / 2

    # time / distance / velocity by animal × day × region
    monkey_time: dict = defaultdict(lambda: defaultdict(lambda: {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}))
    monkey_dist: dict = defaultdict(lambda: defaultdict(lambda: {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}))
    monkey_vel: dict = defaultdict(lambda: defaultdict(lambda: {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}))
    animal_ids = []
    days = []

    for item in sessions:
        meta = _session_meta(item.get("stem", "session"))
        aid, day = meta["animal_id"], meta["day_label"]
        if aid not in animal_ids:
            animal_ids.append(aid)
        if day not in days:
            days.append(day)

        kp = item.get("keypoints")
        segs = item.get("segments")
        if kp is None or segs is None:
            continue
        n = _clip_frames(len(kp), max_frames) if max_frames else len(kp)
        df = simple_com(kp.iloc[:n])
        dist = _frame_distance_cm(df, segs, window_size=window_size, pixel_to_cm=pixel_to_cm)
        vel = dist * fps
        pos_x = np.where(df["center_x"] < x_th, "left", "right")
        pos_y = np.where(df["center_y"] < y_th, "top", "bottom")

        for region, labels in (("top", pos_y), ("bottom", pos_y), ("left", pos_x), ("right", pos_x)):
            mask = labels == region
            monkey_time[aid][day][region] = float(mask.sum()) / fps
            monkey_dist[aid][day][region] = float(dist[mask].sum())
            valid = vel[mask & (vel > 0)]
            monkey_vel[aid][day][region] = float(valid.mean()) if len(valid) else 0.0

    days = sorted(
        days,
        key=lambda s: int(s.split()[-1]) if isinstance(s, str) and s.startswith("Day ") else str(s),
    )
    animal_ids = sorted(animal_ids)

    x = []
    current_pos = 0
    n_days = len(days)
    for _ in animal_ids:
        for j in range(n_days):
            x.append(current_pos + j)
        current_pos += n_days + 1
    x = np.array(x, dtype=float)
    width_bar = 0.8

    def _stack(source, a, b):
        first, second = [], []
        for mid in animal_ids:
            for day in days:
                first.append(source[mid][day][a])
                second.append(source[mid][day][b])
        return first, second

    bottom_t, top_t = _stack(monkey_time, "bottom", "top")
    right_t, left_t = _stack(monkey_time, "right", "left")
    bottom_d, top_d = _stack(monkey_dist, "bottom", "top")
    right_d, left_d = _stack(monkey_dist, "right", "left")
    bottom_v, top_v = _stack(monkey_vel, "bottom", "top")
    right_v, left_v = _stack(monkey_vel, "right", "left")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(30, 24))
    ax1.bar(x, bottom_t, width_bar, label="Bottom", color="steelblue", alpha=0.6)
    ax1.bar(x, top_t, width_bar, bottom=bottom_t, label="Top", color="lightblue", alpha=0.6)
    ax2.bar(x, right_t, width_bar, label="Right", color="forestgreen", alpha=0.6)
    ax2.bar(x, left_t, width_bar, bottom=right_t, label="Left", color="lightgreen", alpha=0.6)
    ax3.bar(x, bottom_d, width_bar, label="Bottom", color="steelblue", alpha=0.6)
    ax3.bar(x, top_d, width_bar, bottom=bottom_d, label="Top", color="lightblue", alpha=0.6)
    ax4.bar(x, right_d, width_bar, label="Right", color="forestgreen", alpha=0.6)
    ax4.bar(x, left_d, width_bar, bottom=right_d, label="Left", color="lightgreen", alpha=0.6)
    ax5.bar(x, bottom_v, width_bar, label="Bottom", color="steelblue", alpha=0.6)
    ax5.bar(x, top_v, width_bar, bottom=bottom_v, label="Top", color="lightblue", alpha=0.6)
    ax6.bar(x, right_v, width_bar, label="Right", color="forestgreen", alpha=0.6)
    ax6.bar(x, left_v, width_bar, bottom=right_v, label="Left", color="lightgreen", alpha=0.6)

    labels = [f"Monkey {mid}\n{day}" for mid in animal_ids for day in days]
    bg_colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "lightpink"]
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.legend()
        for i in range(len(animal_ids) - 1):
            if n_days > 0 and (i * n_days + n_days - 1) < len(x):
                separator_pos = x[i * n_days + n_days - 1] + 0.5
                ax.axvline(x=separator_pos, color="red", linestyle="-", alpha=0.5, linewidth=2)
        for i in range(len(animal_ids)):
            if n_days == 0:
                continue
            start = x[i * n_days] - 0.5
            end = x[i * n_days + n_days - 1] + 0.5
            ax.axvspan(start, end, alpha=0.1, color=bg_colors[i % len(bg_colors)])

    ax1.set_title("Time Distribution (Top vs Bottom)")
    ax2.set_title("Time Distribution (Left vs Right)")
    ax3.set_title("Distance (Walking+Climbing only, Top vs Bottom)")
    ax4.set_title("Distance (Walking+Climbing only, Left vs Right)")
    ax5.set_title("Velocity (Walking+Climbing only, Top vs Bottom)")
    ax6.set_title("Velocity (Walking+Climbing only, Left vs Right)")
    ax1.set_ylabel("Total Duration (s)")
    ax2.set_ylabel("Total Duration (s)")
    ax3.set_ylabel("Locomotion Distance (cm)")
    ax4.set_ylabel("Locomotion Distance (cm)")
    ax5.set_ylabel("Average Velocity (cm/s)")
    ax6.set_ylabel("Average Velocity (cm/s)")
    fig.suptitle("Monkey Position Analysis - All Class", fontsize=16)
    fig.tight_layout()
    return save_figure(fig, Path(output_dir) / f"{stem}_position_analysis", formats)


def compute_behavior_summary_row(
    stem: str,
    segments: pd.DataFrame,
    keypoints: Optional[pd.DataFrame],
    *,
    fps: float,
    window_size: int,
    max_sec: float,
    pixel_to_cm: float,
) -> dict:
    """单会话行为时间/距离/速度汇总（对齐 BehaviorStateVisualization）。"""
    from .report import expand_segments, merge_adjacent, _budget

    meta = _session_meta(stem)
    total_duration = max_sec if max_sec and max_sec > 0 else 1800.0
    segs = merge_adjacent(expand_segments(segments, window_size, fps, max_sec))
    times, pcts, _ = _budget(segs, total_duration)

    row = {
        **meta,
        "static_time": times["Static"],
        "walking_time": times["Walking"],
        "climbing_time": times["Climbing"],
        "static_pct": pcts["Static"],
        "walking_pct": pcts["Walking"],
        "climbing_pct": pcts["Climbing"],
        "walking_distance": 0.0,
        "climbing_distance": 0.0,
        "walking_speed": 0.0,
        "climbing_speed": 0.0,
    }
    if keypoints is None or keypoints.empty or segments is None or segments.empty:
        return row

    max_frames = int(total_duration * fps)
    df = simple_com(keypoints.iloc[:_clip_frames(len(keypoints), max_frames)])
    segs_f = _segment_frame_bounds(segments, window_size, len(df))
    walk_d = climb_d = walk_t = climb_t = 0.0
    for _, seg in segs_f.iterrows():
        pattern = int(seg["pattern"])
        if pattern not in LOCOMOTION_PATTERNS:
            continue
        a = max(0, min(int(seg["start_frame"]), len(df) - 1))
        b = max(a, min(int(seg["end_frame"]), len(df) - 1))
        duration = (b - a) / fps
        traj = df.iloc[a : b + 1]
        distance = float(np.sqrt(traj["center_x"].diff() ** 2 + traj["center_y"].diff() ** 2).sum()) * pixel_to_cm
        if pattern == 2:
            walk_d += distance
            walk_t += duration
        else:
            climb_d += distance
            climb_t += duration
    row["walking_distance"] = walk_d
    row["climbing_distance"] = climb_d
    row["walking_speed"] = walk_d / walk_t if walk_t > 0 else 0.0
    row["climbing_speed"] = climb_d / climb_t if climb_t > 0 else 0.0
    return row


def plot_behavior_comparison(
    sessions: Sequence[Mapping],
    *,
    output_dir: Path,
    formats: Sequence[str],
    fps: float = 25.0,
    window_size: int = 13,
    max_sec: float = 1800.0,
    pixel_to_cm: float = 0.3202,
    stem: str = "all_sessions",
) -> list:
    """Behavior State Comparison — Time / Distance / Speed（2×2）。"""
    rows = []
    for item in sessions:
        s = item.get("stem", "session")
        rows.append(
            compute_behavior_summary_row(
                s,
                item["segments"],
                item.get("keypoints"),
                fps=fps,
                window_size=window_size,
                max_sec=max_sec,
                pixel_to_cm=pixel_to_cm,
            )
        )
    if not rows:
        return []
    summary_df = pd.DataFrame(rows).sort_values(["animal_id", "sort_key"]).reset_index(drop=True)
    csv_path = Path(output_dir) / f"{stem}_behavior_state_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    labels = [f"{r['animal_id']}\n{r['day_label']}" for _, r in summary_df.iterrows()]
    x = np.arange(len(summary_df))
    fig_w = max(12, 1.4 * len(summary_df) + 3)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10))
    fig.suptitle("Behavior State Comparison — Time / Distance / Speed", fontsize=15, fontweight="bold")

    def _add_separators(ax):
        aids = summary_df["animal_id"].tolist()
        for i in range(1, len(aids)):
            if aids[i] != aids[i - 1]:
                ax.axvline(i - 0.5, color="#E57373", linestyle="--", alpha=0.7, linewidth=1.5)

    def _style_xaxis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _add_separators(ax)

    ax = axes[0, 0]
    bottom = np.zeros(len(summary_df))
    for label in BEHAVIOR_ORDER:
        vals = summary_df[f"{label.lower()}_pct"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=BEHAVIOR_COLORS[label], label=label, edgecolor="white", linewidth=0.4, width=0.75)
        bottom += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Time Share (%)", fontsize=11)
    ax.set_title("Time Proportion", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    _style_xaxis(ax)

    ax = axes[0, 1]
    bottom = np.zeros(len(summary_df))
    for label in BEHAVIOR_ORDER:
        vals = summary_df[f"{label.lower()}_time"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=BEHAVIOR_COLORS[label], label=label, edgecolor="white", linewidth=0.4, width=0.75)
        bottom += vals
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title("Time by Behavior", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    _style_xaxis(ax)

    ax = axes[1, 0]
    bottom = np.zeros(len(summary_df))
    for label in LOCOMOTION_ORDER:
        vals = summary_df[f"{label.lower()}_distance"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=BEHAVIOR_COLORS[label], label=label, edgecolor="white", linewidth=0.4, width=0.75)
        bottom += vals
    ax.set_ylabel("Distance (cm)", fontsize=11)
    ax.set_title("Distance (Walking + Climbing)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    _style_xaxis(ax)

    ax = axes[1, 1]
    w = 0.35
    ax.bar(x - w / 2, summary_df["walking_speed"].to_numpy(dtype=float), w, color=BEHAVIOR_COLORS["Walking"], label="Walking", edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, summary_df["climbing_speed"].to_numpy(dtype=float), w, color=BEHAVIOR_COLORS["Climbing"], label="Climbing", edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Speed (cm/s)", fontsize=11)
    ax.set_title("Average Speed (Walking vs Climbing)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    _style_xaxis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    saved = save_figure(fig, Path(output_dir) / f"{stem}_behavior_comparison", formats)
    return saved

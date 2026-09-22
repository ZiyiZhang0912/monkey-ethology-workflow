from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _angle_change(p1, center, p2) -> float:
    v1 = np.array([p1[0] - center[0], p1[1] - center[1]], dtype=float)
    v2 = np.array([p2[0] - center[0], p2[1] - center[1]], dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    diff = a2 - a1
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return float(np.degrees(diff))


def _in_roi(points: np.ndarray, x_min, x_max, y_min, y_max, threshold=0.7) -> bool:
    valid = points[np.isfinite(points).all(axis=1)]
    if len(valid) == 0:
        return False
    inside = (
        (valid[:, 0] >= x_min)
        & (valid[:, 0] <= x_max)
        & (valid[:, 1] >= y_min)
        & (valid[:, 1] <= y_max)
    )
    return float(inside.mean()) >= threshold


def detect_circling(
    keypoints: pd.DataFrame,
    *,
    body_part: str = "Head",
    min_likelihood: float = 0.5,
    window_size: int = 40,
    min_angle: float = 180,
    min_radius: float = 15,
    max_radius: float = 250,
    x_min: float = 200,
    x_max: float = 550,
    y_min: float = 150,
    y_max: float = 300,
    max_radius_cv: float = 0.8,
    fps: float = 25,
) -> pd.DataFrame:
    """Circling detection from cumulative turning angle along a body-part trajectory."""
    x_col, y_col, p_col = f"{body_part}_x", f"{body_part}_y", f"{body_part}_p"
    if x_col not in keypoints.columns:
        return pd.DataFrame()
    x = keypoints[x_col].to_numpy(dtype=float)
    y = keypoints[y_col].to_numpy(dtype=float)
    p = keypoints[p_col].to_numpy(dtype=float) if p_col in keypoints.columns else np.ones_like(x)
    traj = np.column_stack([x, y])
    traj[p < min_likelihood] = np.nan
    n = len(traj)
    if n < window_size:
        return pd.DataFrame()

    events: List[dict] = []
    for i in range(0, n - window_size + 1):
        window = traj[i : i + window_size]
        valid = window[np.isfinite(window).all(axis=1)]
        if len(valid) < window_size * 0.6:
            continue
        if not _in_roi(valid, x_min, x_max, y_min, y_max):
            continue
        center = np.median(valid, axis=0)
        radii = np.hypot(valid[:, 0] - center[0], valid[:, 1] - center[1])
        avg_r = float(np.mean(radii))
        if avg_r < min_radius or avg_r > max_radius:
            continue
        cv = float(np.std(radii) / avg_r) if avg_r > 0 else np.inf
        if cv > max_radius_cv:
            continue
        total = 0.0
        for j in range(1, len(valid)):
            total += _angle_change(valid[j - 1], center, valid[j])
        if abs(total) >= min_angle:
            events.append(
                {
                    "start_frame": i,
                    "end_frame": i + window_size - 1,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "radius": avg_r,
                    "direction": "clockwise" if total < 0 else "counterclockwise",
                    "total_angle": float(total),
                    "radius_cv": cv,
                    "start_sec": i / fps,
                    "end_sec": (i + window_size - 1) / fps,
                }
            )
    return pd.DataFrame(_merge_events(events))


def _merge_events(events: List[dict], gap: int = 15) -> List[dict]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e["start_frame"])
    merged = [events[0].copy()]
    for ev in events[1:]:
        last = merged[-1]
        if ev["start_frame"] <= last["end_frame"] + gap:
            last["end_frame"] = max(last["end_frame"], ev["end_frame"])
            last["end_sec"] = ev.get("end_sec", last.get("end_sec"))
            last["total_angle"] = last["total_angle"] + ev["total_angle"]
            last["radius"] = 0.5 * (last["radius"] + ev["radius"])
        else:
            merged.append(ev.copy())
    return merged

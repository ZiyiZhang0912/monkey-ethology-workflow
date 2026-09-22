from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..io.schema import joint_columns


def median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(values) == 0:
        return np.asarray(values, dtype=float).copy()
    pad = window // 2
    arr = np.asarray(values, dtype=float)
    padded = np.pad(arr, pad, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(arr))])


def smooth_keypoints(
    df: pd.DataFrame,
    joints: Sequence[str],
    window: int = 5,
    p_threshold: float = 0.5,
) -> pd.DataFrame:
    """Keep last valid coordinate for low-confidence frames, then median-filter."""
    if window is None or int(window) <= 0:
        return df.copy()
    out = df.copy()
    for joint in joints:
        x_col, y_col, p_col = joint_columns(joint)
        if x_col not in out.columns or y_col not in out.columns:
            continue
        xs = out[x_col].to_numpy(dtype=float, copy=True)
        ys = out[y_col].to_numpy(dtype=float, copy=True)
        if p_col in out.columns:
            ps = out[p_col].to_numpy(dtype=float)
            last_x, last_y = xs[0], ys[0]
            for i in range(len(xs)):
                if np.isfinite(ps[i]) and ps[i] >= p_threshold:
                    last_x, last_y = xs[i], ys[i]
                else:
                    xs[i], ys[i] = last_x, last_y
        out[x_col] = median_filter_1d(xs, window)
        out[y_col] = median_filter_1d(ys, window)
    return out

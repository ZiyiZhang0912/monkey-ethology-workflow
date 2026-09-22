from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..io.schema import compute_com, joint_columns


def _periodicity_score(signal: np.ndarray, fps: float) -> float:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 0.0
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    spec = np.abs(np.fft.rfft(x))
    if len(spec) < 2:
        return 0.0
    spec[0] = 0
    mean_mag = np.mean(spec) or 1.0
    return float(np.max(spec) / mean_mag)


def _fit_circle(x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        return None, None
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = params
        r2 = cx ** 2 + cy ** 2 - c
        if r2 <= 0 or r2 > 1e8:
            return None, None
        radius = np.sqrt(r2)
        err = np.mean(np.abs(np.hypot(x - cx, y - cy) - radius))
        return (cx, cy, radius), float(err)
    except np.linalg.LinAlgError:
        return None, None


def detect_stereotypy(
    keypoints: pd.DataFrame,
    *,
    joints: Sequence[str],
    fps: float = 25,
    window_sec: float = 3.0,
    step_sec: float = 1.0,
    p_threshold: float = 0.5,
    periodicity_threshold: float = 8.0,
    angular_std_threshold: float = 2.0,
    circle_error_threshold: float = 700.0,
    cumulative_displacement_threshold: float = 2 * np.pi,
    head_joint: str = "Head",
    orientation_from: str = "Back",
) -> pd.DataFrame:
    """
    四特征刻板检测：
    周期性 × 角速度标准差 × (圆拟合误差 或 累积角位移)。
    """
    com = compute_com(keypoints, joints, p_threshold=p_threshold, weighted=True)
    valid = com["com_p"].to_numpy() >= p_threshold
    idx = np.where(valid)[0]
    if len(idx) < 10:
        return pd.DataFrame()

    cx = com.loc[idx, "com_x"].to_numpy()
    cy = com.loc[idx, "com_y"].to_numpy()
    t = idx / float(fps)

    hx, hy = f"{head_joint}_x", f"{head_joint}_y"
    bx, by = f"{orientation_from}_x", f"{orientation_from}_y"
    if hx in keypoints.columns and bx in keypoints.columns:
        head = keypoints.loc[idx, [hx, hy]].to_numpy(dtype=float)
        back = keypoints.loc[idx, [bx, by]].to_numpy(dtype=float)
        angles = np.arctan2(head[:, 1] - back[:, 1], head[:, 0] - back[:, 0])
    else:
        angles = np.arctan2(np.diff(cy, prepend=cy[0]), np.diff(cx, prepend=cx[0]))

    d_ang = np.diff(angles)
    d_ang = np.arctan2(np.sin(d_ang), np.cos(d_ang))
    dt = np.diff(t)
    dt[dt == 0] = 1.0 / fps
    ang_vel = d_ang / dt

    win = int(window_sec * fps)
    step = max(int(step_sec * fps), 1)
    rows = []
    n = len(cx)
    for i in range(0, n - win, step):
        sl = slice(i, i + win)
        per = _periodicity_score(cx[sl], fps)
        av = ang_vel[i : i + win - 1] if i + win - 1 <= len(ang_vel) else ang_vel[i:]
        ang_std = float(np.nanstd(av)) if len(av) else 0.0
        _, cerr = _fit_circle(cx[sl], cy[sl])
        cum = float(np.nansum(av) * (1.0 / fps)) if len(av) else 0.0
        high_per = per > periodicity_threshold
        high_ang = ang_std > angular_std_threshold
        low_err = cerr is not None and cerr < circle_error_threshold
        high_cum = abs(cum) > cumulative_displacement_threshold
        is_st = bool(high_per and high_ang and (low_err or high_cum))
        conf = sum([high_per, high_ang, low_err, high_cum]) / 4.0
        rows.append(
            {
                "start_frame": int(idx[i]),
                "end_frame": int(idx[min(i + win - 1, len(idx) - 1)]),
                "time": float(t[i + win // 2]),
                "is_stereotypy": is_st,
                "periodicity_score": per,
                "angular_std": ang_std,
                "circle_error": np.nan if cerr is None else cerr,
                "cumulative_displacement": cum,
                "confidence": conf,
            }
        )
    return pd.DataFrame(rows)

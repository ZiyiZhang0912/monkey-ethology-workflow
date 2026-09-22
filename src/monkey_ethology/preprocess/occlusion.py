from __future__ import annotations

"""Occlusion detection and interpolation for low-confidence keypoints."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..io.schema import joint_columns

try:
    from scipy import interpolate as scipy_interpolate
except ImportError:  # pragma: no cover
    scipy_interpolate = None


class OcclusionHandler:
    """Low-confidence occlusion interpolation with displacement/velocity constraints."""

    def __init__(
        self,
        joints: Sequence[str],
        p_threshold: float = 0.4,
        min_occlusion_duration: int = 3,
        sitting_min_frames: int = 30,
        sitting_joints: Sequence[str] = ("Back", "Tail(Root)"),
        max_displacement: float = 100,
        max_velocity: float = 50,
        bounds: Optional[Dict[str, float]] = None,
    ):
        self.joints = list(joints)
        self.p_threshold = p_threshold
        self.min_occlusion_duration = min_occlusion_duration
        self.sitting_min_frames = sitting_min_frames
        self.sitting_joints = list(sitting_joints)
        self.max_displacement = max_displacement
        self.max_velocity = max_velocity
        self.bounds = bounds or {"x_min": 0, "x_max": 1920, "y_min": 0, "y_max": 1080}

    def detect_periods(self, values: np.ndarray) -> List[Tuple[int, int]]:
        periods: List[Tuple[int, int]] = []
        start = None
        for i, low in enumerate(values):
            if low:
                if start is None:
                    start = i
            elif start is not None:
                if i - start >= self.min_occlusion_duration:
                    periods.append((start, i))
                start = None
        if start is not None and len(values) - start >= self.min_occlusion_duration:
            periods.append((start, len(values)))
        return periods

    def detect_sitting(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        masks = []
        for joint in self.sitting_joints:
            p_col = f"{joint}_p"
            if p_col in data.columns:
                masks.append(data[p_col].to_numpy() < self.p_threshold)
        if len(masks) < 2:
            return []
        both = np.logical_and.reduce(masks)
        old_min = self.min_occlusion_duration
        self.min_occlusion_duration = self.sitting_min_frames
        periods = self.detect_periods(both)
        self.min_occlusion_duration = old_min
        return periods

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        for joint in self.joints:
            x_col, y_col, p_col = joint_columns(joint)
            if not all(c in out.columns for c in (x_col, y_col, p_col)):
                continue
            low = out[p_col].to_numpy() < self.p_threshold
            for start, end in self.detect_periods(low):
                out = self._safe_interpolation(out, joint, start, end)
        return out

    def _safe_interpolation(
        self, data: pd.DataFrame, joint: str, start_frame: int, end_frame: int
    ) -> pd.DataFrame:
        result = data.copy()
        x_col, y_col, p_col = joint_columns(joint)
        before_data = data.iloc[:start_frame]
        after_data = data.iloc[end_frame:]
        if len(before_data) < 2 or len(after_data) < 2:
            return self._constrained_interpolation(data, joint, start_frame, end_frame)

        n_points = min(5, len(before_data), len(after_data))
        before_x = before_data[x_col].iloc[-n_points:].to_numpy(dtype=float)
        before_y = before_data[y_col].iloc[-n_points:].to_numpy(dtype=float)
        before_p = before_data[p_col].iloc[-n_points:].to_numpy(dtype=float)
        after_x = after_data[x_col].iloc[:n_points].to_numpy(dtype=float)
        after_y = after_data[y_col].iloc[:n_points].to_numpy(dtype=float)
        after_p = after_data[p_col].iloc[:n_points].to_numpy(dtype=float)

        before_frames = list(range(start_frame - n_points, start_frame))
        after_frames = list(range(end_frame, end_frame + n_points))
        all_frames = before_frames + after_frames
        all_x = np.concatenate([before_x, after_x])
        all_y = np.concatenate([before_y, after_y])
        all_p = np.concatenate([before_p, after_p])
        interp_frames = list(range(start_frame, end_frame))
        if not interp_frames:
            return result

        try:
            if scipy_interpolate is not None:
                x_interp = scipy_interpolate.interp1d(
                    all_frames, all_x, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                y_interp = scipy_interpolate.interp1d(
                    all_frames, all_y, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                p_interp = scipy_interpolate.interp1d(
                    all_frames, all_p, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                interpolated_x = np.asarray(x_interp(interp_frames), dtype=float)
                interpolated_y = np.asarray(y_interp(interp_frames), dtype=float)
                interpolated_p = np.asarray(p_interp(interp_frames), dtype=float)
            else:
                interpolated_x = np.interp(interp_frames, all_frames, all_x)
                interpolated_y = np.interp(interp_frames, all_frames, all_y)
                interpolated_p = np.interp(interp_frames, all_frames, all_p)

            interpolated_x, interpolated_y = self._apply_position_constraints(
                interpolated_x,
                interpolated_y,
                float(before_x[-1]),
                float(before_y[-1]),
                float(after_x[0]),
                float(after_y[0]),
            )
            result.loc[start_frame : end_frame - 1, x_col] = interpolated_x
            result.loc[start_frame : end_frame - 1, y_col] = interpolated_y
            result.loc[start_frame : end_frame - 1, p_col] = interpolated_p
        except Exception:
            return self._constrained_interpolation(data, joint, start_frame, end_frame)
        return result

    def _apply_position_constraints(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        before_x: float,
        before_y: float,
        after_x: float,
        after_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        constrained_x = np.asarray(x_values, dtype=float).copy()
        constrained_y = np.asarray(y_values, dtype=float).copy()
        constrained_x = np.clip(constrained_x, self.bounds["x_min"], self.bounds["x_max"])
        constrained_y = np.clip(constrained_y, self.bounds["y_min"], self.bounds["y_max"])

        total_displacement = float(np.hypot(after_x - before_x, after_y - before_y))
        if total_displacement > self.max_displacement:
            n_frames = len(constrained_x)
            for i in range(n_frames):
                alpha = (i + 1) / (n_frames + 1)
                constrained_x[i] = before_x + alpha * (after_x - before_x)
                constrained_y[i] = before_y + alpha * (after_y - before_y)

        for i in range(1, len(constrained_x)):
            dx = constrained_x[i] - constrained_x[i - 1]
            dy = constrained_y[i] - constrained_y[i - 1]
            velocity = float(np.hypot(dx, dy))
            if velocity > self.max_velocity and velocity > 0:
                scale = self.max_velocity / velocity
                constrained_x[i] = constrained_x[i - 1] + dx * scale
                constrained_y[i] = constrained_y[i - 1] + dy * scale

        constrained_x = self._smooth_trajectory(constrained_x)
        constrained_y = self._smooth_trajectory(constrained_y)
        return constrained_x, constrained_y

    @staticmethod
    def _smooth_trajectory(values: np.ndarray, window_size: int = 3) -> np.ndarray:
        if len(values) < window_size:
            return values
        smoothed = values.copy()
        for i in range(1, len(values) - 1):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed[i] = np.mean(values[start_idx:end_idx])
        return smoothed

    def _constrained_interpolation(
        self, data: pd.DataFrame, joint: str, start_frame: int, end_frame: int
    ) -> pd.DataFrame:
        result = data.copy()
        x_col, y_col, p_col = joint_columns(joint)
        before_x = (
            data[x_col].iloc[start_frame - 1]
            if start_frame > 0
            else data[x_col].iloc[start_frame]
        )
        before_y = (
            data[y_col].iloc[start_frame - 1]
            if start_frame > 0
            else data[y_col].iloc[start_frame]
        )
        before_p = (
            data[p_col].iloc[start_frame - 1]
            if start_frame > 0
            else data[p_col].iloc[start_frame]
        )
        after_x = (
            data[x_col].iloc[end_frame]
            if end_frame < len(data)
            else data[x_col].iloc[end_frame - 1]
        )
        after_y = (
            data[y_col].iloc[end_frame]
            if end_frame < len(data)
            else data[y_col].iloc[end_frame - 1]
        )
        after_p = (
            data[p_col].iloc[end_frame]
            if end_frame < len(data)
            else data[p_col].iloc[end_frame - 1]
        )
        duration = end_frame - start_frame
        if duration > 0:
            for i in range(duration):
                alpha = (i + 1) / duration
                result.loc[start_frame + i, x_col] = before_x + alpha * (after_x - before_x)
                result.loc[start_frame + i, y_col] = before_y + alpha * (after_y - before_y)
                result.loc[start_frame + i, p_col] = before_p + alpha * (after_p - before_p)
        return result

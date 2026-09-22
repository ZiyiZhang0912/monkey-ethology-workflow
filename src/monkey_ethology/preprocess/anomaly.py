from __future__ import annotations

"""Position anomaly detection and repair (boundary, jump, velocity, continuity)."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..io.schema import joint_columns


class AnomalyDetector:
    """Boundary / displacement / velocity / acceleration / continuity repair."""

    def __init__(
        self,
        joints: Sequence[str],
        max_displacement: float = 100,
        max_velocity: float = 50,
        max_acceleration: float = 30,
        bounds: Optional[Dict[str, float]] = None,
    ):
        self.joints = list(joints)
        self.max_displacement = max_displacement
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.bounds = bounds or {"x_min": 0, "x_max": 1920, "y_min": 0, "y_max": 1080}

    def detect(self, data: pd.DataFrame) -> Dict[str, List[dict]]:
        anomalies: Dict[str, List[dict]] = {}
        for joint in self.joints:
            x_col, y_col, _ = joint_columns(joint)
            if not all(c in data.columns for c in (x_col, y_col)):
                continue
            found: List[dict] = []

            # Match pandas Series.diff() NaN semantics used by PositionAnomalyDetector.
            x = data[x_col]
            y = data[y_col]

            x_oob = (x < self.bounds["x_min"]) | (x > self.bounds["x_max"])
            y_oob = (y < self.bounds["y_min"]) | (y > self.bounds["y_max"])
            oob_x = data.index[x_oob].tolist()
            oob_y = data.index[y_oob].tolist()
            if oob_x:
                found.append({"type": "boundary_x", "indices": oob_x})
            if oob_y:
                found.append({"type": "boundary_y", "indices": oob_y})

            dx = x.diff()
            dy = y.diff()
            displacement = np.sqrt(dx**2 + dy**2)
            jump = data.index[displacement > self.max_displacement].tolist()
            if jump:
                found.append({"type": "displacement", "indices": jump})

            velocity = displacement
            high_v = data.index[velocity > self.max_velocity].tolist()
            if high_v:
                found.append({"type": "velocity", "indices": high_v})

            acceleration = velocity.diff()
            high_a = data.index[acceleration > self.max_acceleration].tolist()
            if high_a:
                found.append({"type": "acceleration", "indices": high_a})

            d2x = dx.diff()
            d2y = dy.diff()
            curvature = np.sqrt(d2x**2 + d2y**2)
            mean_c = float(curvature.mean())
            std_c = float(curvature.std())
            thr = mean_c + 3 * std_c
            high_c = data.index[curvature > thr].tolist() if thr > 0 else []
            if high_c:
                found.append({"type": "continuity", "indices": high_c})

            if found:
                anomalies[joint] = found
        return anomalies

    def fix(self, data: pd.DataFrame, anomalies: Dict[str, List[dict]]) -> pd.DataFrame:
        out = data.copy()
        for joint, items in anomalies.items():
            x_col, y_col, _ = joint_columns(joint)
            for item in items:
                typ = item.get("type")
                indices = item.get("indices", [])
                if typ == "boundary_x":
                    out = self._fix_boundary(out, x_col, indices, "x")
                elif typ == "boundary_y":
                    out = self._fix_boundary(out, y_col, indices, "y")
                elif typ == "displacement":
                    out = self._fix_displacement(out, joint, indices)
                elif typ == "velocity":
                    out = self._fix_velocity(out, joint, indices)
                elif typ == "acceleration":
                    out = self._fix_acceleration(out, joint, indices)
                elif typ == "continuity":
                    out = self._fix_continuity(out, joint, indices)
        return out

    def _fix_boundary(
        self, data: pd.DataFrame, col: str, indices: List[int], coord: str
    ) -> pd.DataFrame:
        lo = self.bounds["x_min"] if coord == "x" else self.bounds["y_min"]
        hi = self.bounds["x_max"] if coord == "x" else self.bounds["y_max"]
        for idx in indices:
            data.loc[idx, col] = float(np.clip(data.loc[idx, col], lo, hi))
        return data

    def _fix_displacement(self, data: pd.DataFrame, joint: str, indices: List[int]) -> pd.DataFrame:
        x_col, y_col, _ = joint_columns(joint)
        for idx in indices:
            if 0 < idx < len(data) - 1:
                data.loc[idx, x_col] = (data.loc[idx - 1, x_col] + data.loc[idx + 1, x_col]) / 2
                data.loc[idx, y_col] = (data.loc[idx - 1, y_col] + data.loc[idx + 1, y_col]) / 2
        return data

    def _fix_velocity(self, data: pd.DataFrame, joint: str, indices: List[int]) -> pd.DataFrame:
        x_col, y_col, _ = joint_columns(joint)
        for idx in indices:
            if idx > 0:
                prev_x = data.loc[idx - 1, x_col]
                prev_y = data.loc[idx - 1, y_col]
                dx = data.loc[idx, x_col] - prev_x
                dy = data.loc[idx, y_col] - prev_y
                velocity = float(np.sqrt(dx**2 + dy**2))
                if velocity > self.max_velocity:
                    scale = self.max_velocity / velocity
                    data.loc[idx, x_col] = prev_x + dx * scale
                    data.loc[idx, y_col] = prev_y + dy * scale
        return data

    def _fix_acceleration(self, data: pd.DataFrame, joint: str, indices: List[int]) -> pd.DataFrame:
        """Match PositionAnomalyDetector._fix_acceleration_anomalies."""
        x_col, y_col, _ = joint_columns(joint)
        for idx in indices:
            if idx > 1:
                prev_x = data.loc[idx - 1, x_col]
                prev_y = data.loc[idx - 1, y_col]
                next_x = data.loc[idx + 1, x_col] if idx < len(data) - 1 else prev_x
                next_y = data.loc[idx + 1, y_col] if idx < len(data) - 1 else prev_y
                data.loc[idx, x_col] = (prev_x + next_x) / 2
                data.loc[idx, y_col] = (prev_y + next_y) / 2
        return data

    def _fix_continuity(self, data: pd.DataFrame, joint: str, indices: List[int]) -> pd.DataFrame:
        """Continuity repair: replace anomalous frame by local 5-frame mean."""
        x_col, y_col, _ = joint_columns(joint)
        window_size = 5
        for idx in indices:
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(data), idx + window_size // 2 + 1)
            data.loc[idx, x_col] = data.loc[start_idx : end_idx - 1, x_col].mean()
            data.loc[idx, y_col] = data.loc[start_idx : end_idx - 1, y_col].mean()
        return data

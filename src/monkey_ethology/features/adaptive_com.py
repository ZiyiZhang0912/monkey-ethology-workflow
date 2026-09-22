from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..io.schema import joint_columns


class AdaptiveCOMExtractor:
    """
    滑窗自适应重心特征。
    关节点由配置指定，不再写死 Head/Back/Tail。
    """

    def __init__(
        self,
        joints: Sequence[str],
        window_size: int = 13,
        fps: float = 25,
        p_threshold: float = 0.4,
        orientation_pairs: Sequence[Sequence[str]] = None,
        head_joint: str = "Head",
        tail_joint: str = "Tail(Root)",
    ):
        self.joints = list(joints)
        self.window_size = int(window_size)
        self.fps = float(fps)
        self.p_threshold = float(p_threshold)
        self.orientation_pairs = [list(p) for p in (orientation_pairs or [])]
        self.head_joint = head_joint
        self.tail_joint = tail_joint

    def available_joints(self, window: pd.DataFrame) -> list:
        available = []
        for joint in self.joints:
            p_col = f"{joint}_p"
            if p_col in window.columns and window[p_col].mean() >= self.p_threshold:
                available.append(joint)
        return available

    def com(self, window: pd.DataFrame, available: Sequence[str]):
        if not available:
            n = len(window)
            return np.zeros(n), np.zeros(n)
        xs, ys = [], []
        for joint in available:
            x_col, y_col, _ = joint_columns(joint)
            xs.append(window[x_col].to_numpy(dtype=float))
            ys.append(window[y_col].to_numpy(dtype=float))
        return np.mean(xs, axis=0), np.mean(ys, axis=0)

    def orientation(self, window: pd.DataFrame, available: Sequence[str]) -> np.ndarray:
        pairs = self.orientation_pairs or []
        for a, b in pairs:
            if a in available and b in available:
                dx = window[f"{a}_x"].to_numpy(dtype=float) - window[f"{b}_x"].to_numpy(dtype=float)
                dy = window[f"{a}_y"].to_numpy(dtype=float) - window[f"{b}_y"].to_numpy(dtype=float)
                return np.degrees(np.arctan2(dy, dx))
        return np.zeros(len(window))

    def body_area(self, window: pd.DataFrame, available: Sequence[str]) -> np.ndarray:
        n = len(window)
        areas = np.zeros(n)
        if len(available) < 2:
            return areas
        for i in range(n):
            pts = np.array(
                [[window[f"{j}_x"].iloc[i], window[f"{j}_y"].iloc[i]] for j in available],
                dtype=float,
            )
            if len(pts) >= 3:
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                areas[i] = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
            else:
                areas[i] = np.linalg.norm(pts[1] - pts[0])
        return areas

    @staticmethod
    def _periodicity(signal: np.ndarray) -> float:
        if len(signal) < 2:
            return 0.0
        spec = np.abs(np.fft.fft(signal))
        spec = spec[: len(spec) // 2]
        spec[0] = 0
        return float(np.max(spec) / len(signal)) if len(spec) else 0.0

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        w = self.window_size
        if n < w:
            return pd.DataFrame()
        rows = []
        for i in range(0, n - w + 1, w):
            window = data.iloc[i : i + w].reset_index(drop=True)
            available = self.available_joints(window)
            com_x, com_y = self.com(window, available)
            ori = self.orientation(window, available)
            area = self.body_area(window, available)
            disp_x = abs(com_x[-1] - com_x[0]) if len(com_x) else 0.0
            disp_y = abs(com_y[-1] - com_y[0]) if len(com_y) else 0.0
            positions = np.column_stack([com_x, com_y])
            vel = np.zeros(len(window))
            if len(window) > 1:
                vel[1:] = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1)) * self.fps
            acc = np.zeros(len(window))
            if len(window) > 2:
                acc[2:] = np.abs(np.diff(vel[1:])) * self.fps
            ang = np.zeros(len(window))
            if len(window) > 1:
                dori = np.diff(ori)
                dori = (dori + 180) % 360 - 180
                ang[1:] = dori * self.fps
            ht = 0.0
            if self.head_joint in available and self.tail_joint in available:
                ht = float(
                    np.abs(
                        window[f"{self.head_joint}_y"] - window[f"{self.tail_joint}_y"]
                    ).mean()
                )
            means = {}
            for joint in self.joints:
                # Tail(Root) is exported as mean_tail_{x,y}
                if joint == "Tail(Root)":
                    key = "tail"
                else:
                    key = joint.lower().replace("(root)", "").replace(" ", "")
                if joint in available:
                    means[f"mean_{key}_x"] = float(window[f"{joint}_x"].mean())
                    means[f"mean_{key}_y"] = float(window[f"{joint}_y"].mean())
                else:
                    means[f"mean_{key}_x"] = 0.0
                    means[f"mean_{key}_y"] = 0.0
            rows.append(
                {
                    "mean_com_x": float(pd.Series(com_x).mean()),
                    "std_com_x": float(pd.Series(com_x).std()),
                    "mean_com_y": float(pd.Series(com_y).mean()),
                    "std_com_y": float(pd.Series(com_y).std()),
                    "displacement": float(np.hypot(disp_x, disp_y)),
                    "mean_body_area": float(pd.Series(area).mean()),
                    "std_body_area": float(pd.Series(area).std()),
                    "mean_orientation": float(pd.Series(ori).mean()),
                    "std_orientation": float(pd.Series(ori).std()),
                    "available_joints_count": float(len(available)),
                    "available_joints": ",".join(available) if available else "none",
                    "com_x_difference": float(disp_x),
                    "com_y_difference": float(disp_y),
                    "head_tail_y_difference": ht,
                    "mean_velocity": float(np.mean(vel)),
                    "max_velocity": float(np.max(vel)),
                    "mean_acceleration": float(np.mean(acc)),
                    "max_acceleration": float(np.max(acc)),
                    "mean_angular_velocity": float(np.mean(ang)),
                    "max_angular_velocity": float(np.max(ang)),
                    "mean_periodicity": self._periodicity(np.hypot(com_x, com_y)),
                    "max_motion_change": float(np.std(vel) * np.std(acc)),
                    "window_index": i // w,
                    **means,
                }
            )
        return pd.DataFrame(rows)

from __future__ import annotations

"""
Rule-based locomotion segmentation (Static / Walking / Climbing).

Stage 1: consecutive motion windows → Static (0) vs Motion (1).
Stage 2: within motion segments, sub-windows → Walking (2) / Climbing (3) / potential Static (0).

The original mutates ``motion_segments`` with ``remove`` while iterating; that skip
Segment indexing follows inclusive window bounds used by the ethogram time mapping.
"""

from typing import Dict, List, Optional, Sequence

import pandas as pd


DEFAULT_LABELS = {0: "Static", 2: "Walking", 3: "Climbing"}


def segment_locomotion(
    features: pd.DataFrame,
    *,
    velocity_threshold: float = 10.0,
    displacement_threshold: float = 4.0,
    window_threshold: int = 5,
    head_tail_difference_threshold: float = 75.0,
    com_x_threshold: float = 5.5,
    com_y_threshold: float = 10.5,
    climbing_y: float = 202.0,
    subwindow_size: int = 4,
    labels: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Two-stage threshold classifier matching MotionClassifierWithReasons."""
    labels = labels or DEFAULT_LABELS
    window_threshold = int(window_threshold)
    feats = features.reset_index(drop=True)

    motion_segments: List[dict] = []
    current_segment = {
        "start": None,
        "end": None,
        "state": None,
        "pattern": None,
        "reasons": [],
    }
    consecutive_motion_windows = 0

    for index, row in feats.iterrows():
        is_vertical_motion = False
        vertical_motion_reason = ""
        if index >= window_threshold:
            y_change = abs(
                feats.loc[index, "mean_com_y"]
                - feats.loc[index - window_threshold, "mean_com_y"]
            )
            window_velocity = feats.loc[
                index - window_threshold : index, "mean_velocity"
            ].mean()
            is_vertical_motion = (
                y_change > com_y_threshold and window_velocity > velocity_threshold
            )
            if is_vertical_motion:
                vertical_motion_reason = (
                    f"垂直运动检测: y变化={y_change:.2f} > 阈值{com_y_threshold}, "
                    f"窗口速度={window_velocity:.2f} > 阈值{velocity_threshold}"
                )

        current_reasons: List[str] = []
        velocity_condition = row["mean_velocity"] > velocity_threshold
        if velocity_condition:
            current_reasons.append(
                f"速度条件满足: {row['mean_velocity']:.2f} > 阈值{velocity_threshold}"
            )
        else:
            current_reasons.append(
                f"速度条件不满足: {row['mean_velocity']:.2f} <= 阈值{velocity_threshold}"
            )

        displacement_condition = row["displacement"] > displacement_threshold
        if displacement_condition:
            current_reasons.append(
                f"位移条件满足: {row['displacement']:.2f} > 阈值{displacement_threshold}"
            )
        else:
            current_reasons.append(
                f"位移条件不满足: {row['displacement']:.2f} <= 阈值{displacement_threshold}"
            )

        if (velocity_condition and displacement_condition) or is_vertical_motion:
            consecutive_motion_windows += 1
            if is_vertical_motion:
                current_reasons.append(vertical_motion_reason)
        else:
            consecutive_motion_windows = 0

        if consecutive_motion_windows >= window_threshold:
            if current_segment["state"] != 1:
                if current_segment["start"] is not None:
                    current_segment["end"] = index - window_threshold
                    current_segment["pattern"] = 0
                    current_segment["condition"] = "static"
                    segment_data = feats.loc[
                        current_segment["start"] : index - window_threshold
                    ]
                    avg_velocity = segment_data["mean_velocity"].mean()
                    avg_displacement = segment_data["displacement"].mean()
                    current_segment["reasons"] = [
                        f"该片段平均速度为{avg_velocity:.2f}，低于阈值{velocity_threshold}",
                        f"该片段平均位移为{avg_displacement:.2f}，低于阈值{displacement_threshold}",
                        "没有检测到显著的垂直运动（Y方向变化不足或速度不足）",
                        "以上特征导致该片段被判定为静止状态",
                    ]
                    motion_segments.append(current_segment)
                current_segment = {
                    "start": index - window_threshold + 1,
                    "end": None,
                    "state": 1,
                    "pattern": None,
                    "condition": None,
                    "reasons": [],
                }
        else:
            if current_segment["state"] != 0:
                if current_segment["start"] is not None:
                    current_segment["end"] = index
                    motion_segments.append(current_segment)
                current_frame = feats.loc[index]
                current_segment = {
                    "start": index + 1,
                    "end": None,
                    "state": 0,
                    "pattern": 0,
                    "condition": "static",
                    "reasons": [
                        f"当前速度为{current_frame['mean_velocity']:.2f}，低于阈值{velocity_threshold}",
                        f"当前位移为{current_frame['displacement']:.2f}，低于阈值{displacement_threshold}",
                        "没有检测到显著的垂直运动（Y方向变化不足或速度不足）",
                        "以上特征导致该片段被判定为静止状态",
                    ],
                }

    if current_segment["start"] is not None:
        current_segment["end"] = len(feats) - 1
        if current_segment["state"] == 0:
            current_segment["pattern"] = 0
            current_segment["condition"] = "static"
            segment_data = feats.loc[current_segment["start"] : len(feats) - 1]
            avg_velocity = segment_data["mean_velocity"].mean()
            avg_displacement = segment_data["displacement"].mean()
            current_segment["reasons"] = [
                f"该片段平均速度为{avg_velocity:.2f}，低于阈值{velocity_threshold}",
                f"该片段平均位移为{avg_displacement:.2f}，低于阈值{displacement_threshold}",
                "没有检测到显著的垂直运动（Y方向变化不足或速度不足）",
                "以上特征导致该片段被判定为静止状态",
            ]
        motion_segments.append(current_segment)

    # Stage 2 — iterate a snapshot; remove+append like the original (order side-effects preserved).
    for segment in list(motion_segments):
        start, end = segment["start"], segment["end"]
        if segment["state"] == 0:
            segment["pattern"] = 0
            segment["condition"] = "static"
            if not segment.get("reasons"):
                segment_data = feats.loc[start:end]
                avg_velocity = segment_data["mean_velocity"].mean()
                avg_displacement = segment_data["displacement"].mean()
                segment["reasons"] = [
                    f"该片段平均速度为{avg_velocity:.2f}，低于阈值{velocity_threshold}",
                    f"该片段平均位移为{avg_displacement:.2f}，低于阈值{displacement_threshold}",
                    "没有检测到显著的垂直运动（Y方向变化不足或速度不足）",
                    "以上特征导致该片段被判定为静止状态",
                ]
        elif segment["state"] == 1:
            window_size = int(subwindow_size)
            sub_segments = []
            current_pattern = None
            sub_start = start
            condition = None
            pattern_reasons: List[str] = []

            for i in range(start, end, window_size):
                window_end = min(i + window_size, end)
                window_data = feats.loc[i:window_end]

                com_x_change = abs(window_data["com_x_difference"].mean())
                com_y_change = abs(window_data["com_y_difference"].mean())
                head_tail_difference = abs(window_data["head_tail_y_difference"].mean())
                mean_com_y = window_data["mean_com_y"].mean()

                pattern_reasons = []
                segment_data = feats.loc[i:window_end]
                avg_velocity = segment_data["mean_velocity"].mean()
                avg_displacement = segment_data["displacement"].mean()
                motion_condition = (
                    avg_velocity >= velocity_threshold
                    and avg_displacement >= displacement_threshold
                )

                if not motion_condition:
                    pattern = 0
                    condition = "potential_static"
                    pattern_reasons.append(
                        f"该窗口可能为静止状态: 平均速度={avg_velocity:.2f}，阈值={velocity_threshold}; "
                        f"平均位移={avg_displacement:.2f}，阈值={displacement_threshold}"
                    )
                elif mean_com_y < climbing_y:
                    pattern = 3
                    condition = "high_position"
                    pattern_reasons.append(
                        f"重心位置判定为攀爬: 平均y坐标={mean_com_y:.2f} < 阈值{climbing_y}"
                    )
                elif (
                    com_x_change < com_x_threshold and com_y_change > com_y_threshold
                ) or head_tail_difference > head_tail_difference_threshold:
                    pattern = 3
                    condition = (
                        "vertical_motion"
                        if com_y_change > com_y_threshold
                        else "head_tail_difference"
                    )
                    if com_x_change < com_x_threshold and com_y_change > com_y_threshold:
                        pattern_reasons.append(
                            f"垂直运动判定为攀爬: X变化={com_x_change:.2f} < 阈值{com_x_threshold} "
                            f"且 Y变化={com_y_change:.2f} > 阈值{com_y_threshold}"
                        )
                    if head_tail_difference > head_tail_difference_threshold:
                        pattern_reasons.append(
                            f"头尾差值判定为攀爬: 头尾差值={head_tail_difference:.2f} "
                            f"> 阈值{head_tail_difference_threshold}"
                        )
                else:
                    pattern = 2
                    condition = "horizontal_motion"
                    pattern_reasons.append(
                        f"水平运动判定为行走: X变化={com_x_change:.2f} >= 阈值{com_x_threshold} "
                        f"且 Y变化={com_y_change:.2f} <= 阈值{com_y_threshold} "
                        f"且 头尾差值={head_tail_difference:.2f} <= 阈值{head_tail_difference_threshold}"
                    )

                if current_pattern is not None and current_pattern != pattern:
                    current_reasons = pattern_reasons.copy()
                    sub_segments.append(
                        {
                            "start": sub_start,
                            "end": i - 1,
                            "pattern": current_pattern,
                            "condition": condition,
                            "reasons": current_reasons,
                        }
                    )
                    sub_start = i

                current_pattern = pattern

            if sub_start < end:
                final_reasons = pattern_reasons.copy()
                sub_segments.append(
                    {
                        "start": sub_start,
                        "end": end,
                        "pattern": current_pattern,
                        "condition": condition,
                        "reasons": final_reasons,
                    }
                )

            if sub_segments:
                motion_segments.remove(segment)
                for sub_seg in sub_segments:
                    motion_segments.append(
                        {
                            "start": sub_seg["start"],
                            "end": sub_seg["end"],
                            "state": 1,
                            "pattern": sub_seg["pattern"],
                            "condition": sub_seg["condition"],
                            "reasons": sub_seg["reasons"],
                        }
                    )

    cleaned: List[dict] = []
    for segment in motion_segments:
        if (
            segment["start"] is None
            or segment["end"] is None
            or segment["end"] < segment["start"]
        ):
            continue
        seg = dict(segment)
        seg["duration"] = int(seg["end"] - seg["start"] + 1)
        seg["label"] = labels.get(int(seg.get("pattern", 0) or 0), "Static")
        if isinstance(seg.get("reasons"), list):
            seg["reasons"] = "; ".join(seg["reasons"])
        cleaned.append(seg)
    cleaned.sort(key=lambda s: s["start"])
    return cleaned


def segments_to_frame(segments: Sequence[dict]) -> pd.DataFrame:
    cols = ["start", "end", "duration", "state", "pattern", "condition", "label", "reasons"]
    if not segments:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(segments)[cols]

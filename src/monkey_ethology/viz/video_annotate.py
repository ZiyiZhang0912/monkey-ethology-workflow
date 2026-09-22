"""
用分类结果在原始视频上叠加关键节点 / 骨架 / 行为标签。

对齐 LiuZhenGroup/SmallCage/MotionVideoAnnotator.py：
  - 骨架颜色随行为变化（Walking=绿, Climbing=蓝, Static=黑）
  - 低置信度点标红并写 likelihood
  - 重心处写 ``Behavior: {label}``
  - segment 的 start/end 为窗口索引，乘 window_size 映射到帧
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

VIDEO_EXTENSIONS = (".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV")

DEFAULT_BEHAVIOR_COLORS_BGR = {
    "Static": (0, 0, 0),
    "Walking": (0, 255, 0),
    "Climbing": (255, 0, 0),
    "Turning": (0, 255, 255),
    "Transform": (255, 255, 255),
}

DEFAULT_ACTION_MAP = {
    0: "Static",
    2: "Walking",
    3: "Climbing",
    4: "Turning",
}


def find_video_for_stem(video_dir: PathLike, stem: str) -> Optional[Path]:
    """在目录中查找与 stem 同名的视频。"""
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        return None
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    wanted = {f"{stem}{ext}".lower() for ext in VIDEO_EXTENSIONS}
    for path in video_dir.iterdir():
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def resolve_video_path(
    stem: str,
    *,
    video_path: Optional[PathLike] = None,
    video_dir: Optional[PathLike] = None,
    video_map: Optional[Dict[str, PathLike]] = None,
) -> Optional[Path]:
    """按优先级解析会话对应视频：显式路径 → map → 目录同名。"""
    if video_path:
        p = Path(video_path)
        return p if p.is_file() else None
    if video_map and stem in video_map:
        p = Path(video_map[stem])
        return p if p.is_file() else None
    if video_dir:
        return find_video_for_stem(video_dir, stem)
    return None


def discover_annotation_jobs(
    video_dir: PathLike,
    data_dir: PathLike,
    *,
    stem: Optional[str] = None,
) -> List[Tuple[str, Path, Path, Path]]:
    """
    发现可标注任务：同时存在视频、*_processed.csv、*_features-segment.csv。
    返回 [(stem, video, keypoints, segments), ...]
    """
    data_dir = Path(data_dir)
    jobs: List[Tuple[str, Path, Path, Path]] = []
    for processed in sorted(data_dir.glob("*_processed.csv")):
        s = processed.name.replace("_processed.csv", "")
        if stem and s != stem:
            continue
        segment = data_dir / f"{s}_features-segment.csv"
        if not segment.is_file():
            continue
        video = find_video_for_stem(video_dir, s)
        if not video:
            continue
        jobs.append((s, video, processed, segment))
    return jobs


def build_frame_labels(
    segments: pd.DataFrame,
    n_frames: int,
    *,
    window_size: int = 13,
    action_map: Optional[Dict[int, str]] = None,
    label_col: str = "label",
    pattern_col: str = "pattern",
) -> List[str]:
    """将窗口级 segment 展开为逐帧行为标签。"""
    action_map = action_map or DEFAULT_ACTION_MAP
    labels = ["Transform"] * int(n_frames)
    if segments is None or segments.empty:
        return labels
    for _, row in segments.iterrows():
        start_frame = int(row["start"]) * window_size
        end_frame = int(row["end"]) * window_size
        if label_col in segments.columns and pd.notna(row.get(label_col)):
            action = str(row[label_col])
        else:
            pattern = int(row[pattern_col]) if pattern_col in segments.columns else 0
            action = action_map.get(pattern, action_map.get(0, "Static"))
        start_frame = max(0, start_frame)
        end_frame = min(n_frames - 1, end_frame)
        if end_frame < start_frame:
            continue
        for i in range(start_frame, end_frame + 1):
            labels[i] = action
    return labels


class BehaviorVideoAnnotator:
    """在视频帧上绘制 Head–Back–Tail 骨架与 Behavior 标签。"""

    def __init__(
        self,
        *,
        window_size: int = 13,
        joints: Optional[Sequence[str]] = None,
        behavior_colors_bgr: Optional[Dict[str, Tuple[int, int, int]]] = None,
        action_map: Optional[Dict[int, str]] = None,
        p_threshold: float = 0.5,
        draw_labels: bool = True,
        animal_id: Optional[str] = None,
        show_animal_on_head: bool = False,
    ):
        self.window_size = int(window_size)
        self.joints = list(joints or ["Head", "Back", "Tail(Root)"])
        if len(self.joints) < 3:
            raise ValueError("video annotation 至少需要 3 个关节点（Head/Back/Tail）")
        self.behavior_colors = dict(behavior_colors_bgr or DEFAULT_BEHAVIOR_COLORS_BGR)
        self.action_map = dict(action_map or DEFAULT_ACTION_MAP)
        self.p_threshold = float(p_threshold)
        self.draw_labels = bool(draw_labels)
        self.animal_id = animal_id
        self.show_animal_on_head = bool(show_animal_on_head)
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "视频标注需要 opencv-python（或 opencv-python-headless）。"
                "请执行: pip install opencv-python-headless"
            ) from exc

    def _joint_xy_p(self, row: pd.Series, joint: str) -> Tuple[float, float, float]:
        aliases = {
            joint: joint,
            "Tail": "Tail(Root)",
            "TailRoot": "Tail(Root)",
            "Tail (Root)": "Tail(Root)",
        }
        name = aliases.get(joint, joint)
        for candidate in (name, joint, joint.replace("(Root)", ""), "Tail"):
            x_col, y_col, p_col = f"{candidate}_x", f"{candidate}_y", f"{candidate}_p"
            if x_col in row.index and y_col in row.index:
                x = float(row[x_col]) if pd.notna(row[x_col]) else np.nan
                y = float(row[y_col]) if pd.notna(row[y_col]) else np.nan
                p = float(row[p_col]) if p_col in row.index and pd.notna(row[p_col]) else 0.0
                return x, y, p
        return np.nan, np.nan, 0.0

    def draw_frame(
        self,
        frame,
        keypoints_row: pd.Series,
        action: str,
    ):
        import cv2

        color = self.behavior_colors.get(action, self.behavior_colors.get("Transform", (255, 255, 255)))
        points = []
        likelihoods = []
        short_labels = []
        for i, joint in enumerate(self.joints[:3]):
            x, y, p = self._joint_xy_p(keypoints_row, joint)
            points.append((x, y))
            likelihoods.append(p)
            if i == 0 and self.show_animal_on_head and self.animal_id:
                short_labels.append(str(self.animal_id))
            elif "Tail" in joint:
                short_labels.append("Tail")
            elif "Back" in joint:
                short_labels.append("Back")
            else:
                short_labels.append("Head" if i == 0 else joint)

        valid = [(int(x), int(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
        if len(valid) >= 2:
            pts = np.array(valid, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=len(valid) >= 3, color=color, thickness=2)

        for (x, y), label, likelihood in zip(points, short_labels, likelihoods):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xi, yi = int(x), int(y)
            point_color = (0, 0, 255) if likelihood < self.p_threshold else color
            cv2.circle(frame, (xi, yi), 5, point_color, -1)
            if self.draw_labels:
                cv2.putText(
                    frame,
                    f"{label} ({likelihood:.2f})",
                    (xi + 10, yi),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    point_color,
                    1,
                    cv2.LINE_AA,
                )

        finite = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
        if finite:
            center_x = int(np.mean([p[0] for p in finite]))
            center_y = int(np.mean([p[1] for p in finite]))
            cv2.putText(
                frame,
                f"Behavior: {action}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def annotate(
        self,
        video_path: PathLike,
        keypoints: pd.DataFrame,
        segments: pd.DataFrame,
        output_path: PathLike,
        *,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        progress_every: int = 500,
    ) -> Path:
        import cv2

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = int(round(fps)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(keypoints)
        n_lookup = max(total_frames, len(keypoints))
        action_lookup = build_frame_labels(
            segments,
            n_lookup,
            window_size=self.window_size,
            action_map=self.action_map,
        )

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = None
        for codec in ("mp4v", "avc1", "H264", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            cap.release()
            raise RuntimeError("无法创建视频写入器，请检查 OpenCV 编码器")

        frame_idx = start_frame
        written = 0
        try:
            while True:
                if max_frames is not None and written >= int(max_frames):
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx < len(keypoints):
                    action = (
                        action_lookup[frame_idx]
                        if frame_idx < len(action_lookup)
                        else "Transform"
                    )
                    self.draw_frame(frame, keypoints.iloc[frame_idx], action)
                writer.write(frame)
                frame_idx += 1
                written += 1
                if progress_every and written % progress_every == 0:
                    print(f"  已标注 {written} 帧 ({video_path.name})")
        finally:
            cap.release()
            writer.release()

        return output_path


def annotate_session_video(
    stem: str,
    keypoints: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    video_path: Optional[PathLike] = None,
    video_dir: Optional[PathLike] = None,
    video_map: Optional[Dict[str, PathLike]] = None,
    output_dir: PathLike,
    window_size: int = 13,
    joints: Optional[Sequence[str]] = None,
    animal_id: Optional[str] = None,
    show_animal_on_head: bool = False,
    p_threshold: float = 0.5,
    draw_labels: bool = True,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    behavior_colors_bgr: Optional[Dict[str, Tuple[int, int, int]]] = None,
    action_map: Optional[Dict[int, str]] = None,
) -> Optional[Path]:
    """解析视频并写出 ``{stem}-annotated.mp4``；找不到视频则返回 None。"""
    resolved = resolve_video_path(
        stem, video_path=video_path, video_dir=video_dir, video_map=video_map
    )
    if resolved is None:
        return None
    annotator = BehaviorVideoAnnotator(
        window_size=window_size,
        joints=joints,
        behavior_colors_bgr=behavior_colors_bgr,
        action_map=action_map,
        p_threshold=p_threshold,
        draw_labels=draw_labels,
        animal_id=animal_id,
        show_animal_on_head=show_animal_on_head,
    )
    out = Path(output_dir) / f"{stem}-annotated.mp4"
    return annotator.annotate(
        resolved,
        keypoints,
        segments,
        out,
        max_frames=max_frames,
        start_frame=start_frame,
    )


def hex_to_bgr(color: str) -> Tuple[int, int, int]:
    """#RRGGBB → OpenCV BGR。"""
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"invalid hex color: {color}")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return b, g, r

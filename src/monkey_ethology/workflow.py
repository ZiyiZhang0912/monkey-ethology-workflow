from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

from .classify.circling import detect_circling
from .classify.locomotion import segment_locomotion, segments_to_frame
from .classify.stereotypy import detect_stereotypy
from .config import WorkflowConfig, dump_resolved_config, load_config
from .dlc.project import DLCProjectManager
from .dlc.workflow import convert_dlc_outputs
from .features.adaptive_com import AdaptiveCOMExtractor
from .io.schema import infer_joints, load_keypoints, parse_session_name, save_table
from .preprocess.anomaly import AnomalyDetector
from .preprocess.occlusion import OcclusionHandler
from .preprocess.smooth import smooth_keypoints
from .study import session_allowed, study_config
from .viz.report import AcademicVisualizer
from .viz.video_annotate import annotate_session_video

PathLike = Union[str, Path]


class EthologyWorkflow:
    """
    笼内猕猴行为分析 Workflow。

    阶段：
      1. track()     DeepLabCut 项目创建 / 标注 / 训练 / 推理
      2. ingest()    读取 DLC h5/csv，转为扁平关键点
      3. preprocess() 遮挡插值 → 异常修复 → 平滑
      4. extract_features()
      5. classify()  三态 locomotion + 刻板 + 转圈
      6. visualize() 学术化多图输出
      7. annotate_video() 分类结果叠加到原始视频
      8. run()       从关键点或 DLC 产物一键跑完分类+可视化(+可选视频标注)
    """

    def __init__(self, config: Optional[Union[WorkflowConfig, PathLike]] = None, **load_kwargs):
        if isinstance(config, WorkflowConfig):
            self.config = config
        else:
            self.config = load_config(config, **load_kwargs)
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_resolved_config(self.config, self.output_dir)

    # ------------------------------------------------------------------ DLC
    def track(
        self,
        videos: Sequence[PathLike],
        bodyparts: Optional[Sequence[str]] = None,
        steps: Optional[Sequence[str]] = None,
        config_path: Optional[PathLike] = None,
    ) -> dict:
        if bodyparts:
            self.config.set_bodyparts(bodyparts)
        manager = DLCProjectManager(self.config)
        if config_path:
            manager.config_path = Path(config_path)
        cfg_path = manager.run_steps(
            [str(v) for v in videos],
            steps=steps,
            bodyparts=self.config.bodyparts,
            config_path=str(manager.config_path) if manager.config_path else None,
        )
        return {
            "config_path": None if cfg_path is None else str(cfg_path),
            "available": manager.available,
            "log": manager.log,
            "bodyparts": self.config.bodyparts,
        }

    def convert_dlc(self, paths: Sequence[PathLike]) -> list:
        dest = self.output_dir / "keypoints"
        dest.mkdir(parents=True, exist_ok=True)
        return convert_dlc_outputs(
            [str(p) for p in paths],
            str(dest),
            aliases=self.config.get("bodyparts.aliases"),
        )

    # --------------------------------------------------------------- ingest
    def ingest(self, path: PathLike) -> pd.DataFrame:
        aliases = self.config.get("bodyparts.aliases")
        df = load_keypoints(path, aliases=aliases)
        joints = infer_joints(df.columns)
        if joints:
            current = list(self.config.bodyparts or [])
            if not current:
                self.config.set_bodyparts(joints)
        return df

    # ----------------------------------------------------------- preprocess
    def preprocess(self, keypoints: pd.DataFrame) -> pd.DataFrame:
        joints = self.config.bodyparts or infer_joints(keypoints.columns)
        pre = self.config["preprocess"]
        bounds = {
            "x_min": self.config.get("cage.x_min", 0),
            "x_max": self.config.get("cage.x_max", 1920),
            "y_min": self.config.get("cage.y_min", 0),
            "y_max": self.config.get("cage.y_max", 1080),
        }
        out = keypoints.copy()
        if pre.get("enable_occlusion", True):
            handler = OcclusionHandler(
                joints=joints,
                p_threshold=pre.get("p_threshold", 0.4),
                min_occlusion_duration=pre.get("min_occlusion_duration", 3),
                sitting_min_frames=pre.get("sitting_min_frames", 30),
                sitting_joints=self.config.get("bodyparts.sitting_joints") or joints[-2:],
                max_displacement=pre.get("max_displacement", 100),
                max_velocity=pre.get("max_velocity", 50),
                bounds=bounds,
            )
            out = handler.process(out)
        if pre.get("enable_anomaly", True):
            detector = AnomalyDetector(
                joints=joints,
                max_displacement=pre.get("max_displacement", 100),
                max_velocity=pre.get("max_velocity", 50),
                max_acceleration=pre.get("max_acceleration", 30),
                bounds=bounds,
            )
            anomalies = detector.detect(out)
            if anomalies:
                out = detector.fix(out, anomalies)
        if pre.get("enable_smooth", True):
            out = smooth_keypoints(
                out,
                joints,
                window=pre.get("smooth_window", 5),
                p_threshold=pre.get("smooth_p_threshold", 0.5),
            )
        return out

    def extract_features(self, keypoints: pd.DataFrame) -> pd.DataFrame:
        joints = self.config.get("bodyparts.com_joints") or self.config.bodyparts or infer_joints(keypoints.columns)
        extractor = AdaptiveCOMExtractor(
            joints=joints,
            window_size=self.config.get("features.window_size", 13),
            fps=self.config.get("cage.fps", 25),
            p_threshold=self.config.get("features.p_threshold", 0.4),
            orientation_pairs=self.config.get("bodyparts.orientation_pairs"),
            head_joint=self.config.get("bodyparts.head_joint", "Head"),
            tail_joint=self.config.get("bodyparts.tail_joint", "Tail(Root)"),
        )
        return extractor.transform(keypoints)

    # ------------------------------------------------------------ classify
    def classify_locomotion(self, features: pd.DataFrame, animal_id: Optional[str] = None) -> pd.DataFrame:
        if animal_id:
            self.config.resolve_animal(animal_id)
        loc = self.config["locomotion"]
        segs = segment_locomotion(
            features,
            velocity_threshold=loc.get("velocity_threshold", 10),
            displacement_threshold=loc.get("displacement_threshold", 4),
            window_threshold=loc.get("window_threshold", 5),
            head_tail_difference_threshold=loc.get("head_tail_difference_threshold", 75),
            com_x_threshold=loc.get("com_x_threshold", 5.5),
            com_y_threshold=loc.get("com_y_threshold", 10.5),
            climbing_y=self.config.get("cage.climbing_y", 202),
            subwindow_size=loc.get("subwindow_size", 4),
            labels=loc.get("labels"),
        )
        return segments_to_frame(segs)

    def classify_stereotypy(self, keypoints: pd.DataFrame) -> pd.DataFrame:
        st = self.config["stereotypy"]
        joints = self.config.get("bodyparts.com_joints") or self.config.bodyparts
        return detect_stereotypy(
            keypoints,
            joints=joints,
            fps=self.config.get("cage.fps", 25),
            window_sec=st.get("window_sec", 3),
            step_sec=st.get("step_sec", 1),
            p_threshold=st.get("p_threshold", 0.5),
            periodicity_threshold=st.get("periodicity_threshold", 8),
            angular_std_threshold=st.get("angular_std_threshold", 2),
            circle_error_threshold=st.get("circle_error_threshold", 700),
            cumulative_displacement_threshold=st.get("cumulative_displacement_threshold", 6.28),
            head_joint=self.config.get("bodyparts.head_joint", "Head"),
        )

    def classify_circling(self, keypoints: pd.DataFrame) -> pd.DataFrame:
        cr = self.config["circling"]
        return detect_circling(
            keypoints,
            body_part=cr.get("body_part", self.config.get("bodyparts.head_joint", "Head")),
            min_likelihood=cr.get("min_likelihood", 0.5),
            window_size=cr.get("window_size", 40),
            min_angle=cr.get("min_angle", 180),
            min_radius=cr.get("min_radius", 15),
            max_radius=cr.get("max_radius", 250),
            x_min=cr.get("x_min", 200),
            x_max=cr.get("x_max", 550),
            y_min=cr.get("y_min", 150),
            y_max=cr.get("y_max", 300),
            max_radius_cv=cr.get("max_radius_cv", 0.8),
            fps=self.config.get("cage.fps", 25),
        )

    # --------------------------------------------------------------- viz
    def visualize(
        self,
        stem: str,
        segments: pd.DataFrame,
        keypoints: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
        stereotypy: Optional[pd.DataFrame] = None,
        circling: Optional[pd.DataFrame] = None,
        extra_sessions: Optional[list] = None,
        plots: Optional[Sequence[str]] = None,
    ) -> dict:
        viz_cfg = dict(self.config["visualization"])
        if plots:
            viz_cfg["plots"] = list(plots)
        viz = AcademicVisualizer(
            viz_cfg,
            self.config["cage"],
            self.output_dir / "figures",
            window_size=self.config.get("features.window_size", 13),
        )
        return viz.render_session(
            stem=stem,
            segments=segments,
            keypoints=keypoints,
            features=features,
            stereotypy=stereotypy,
            circling=circling,
            extra_sessions=extra_sessions,
        )

    # ------------------------------------------------------ video annotate
    def annotate_video(
        self,
        stem: str,
        keypoints: pd.DataFrame,
        segments: pd.DataFrame,
        *,
        video_path: Optional[PathLike] = None,
        video_dir: Optional[PathLike] = None,
        animal_id: Optional[str] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> Optional[str]:
        """
        将分类结果叠加到原始视频，写出 ``annotated_videos/{stem}-annotated.mp4``。

        视频查找顺序：``video_path`` → ``video_dir``/同名 → ``video_annotation.video_dir``。
        找不到视频时返回 None（不报错，便于批量跳过）。
        """
        va = self.config.get("video_annotation") or {}
        if not va.get("enabled", True) and video_path is None and video_dir is None:
            return None
        out_subdir = va.get("output_subdir") or "annotated_videos"
        out_dir = self.output_dir / out_subdir
        resolved_dir = video_dir or va.get("video_dir")
        mf = max_frames if max_frames is not None else va.get("max_frames")
        sf = start_frame if start_frame is not None else va.get("start_frame", 0)
        labels_cfg = self.config.get("locomotion.labels") or {}
        action_map = {int(k): str(v) for k, v in labels_cfg.items()} if labels_cfg else None
        path = annotate_session_video(
            stem,
            keypoints,
            segments,
            video_path=video_path,
            video_dir=resolved_dir,
            output_dir=out_dir,
            window_size=int(self.config.get("features.window_size", 13)),
            joints=self.config.get("bodyparts.com_joints") or self.config.bodyparts,
            animal_id=animal_id,
            show_animal_on_head=bool(va.get("show_animal_on_head", False)),
            p_threshold=float(va.get("p_threshold", 0.5)),
            draw_labels=bool(va.get("draw_labels", True)),
            max_frames=None if mf in (None, "", "null") else int(mf),
            start_frame=int(sf or 0),
            action_map=action_map,
        )
        return None if path is None else str(path)

    # ---------------------------------------------------------------- run
    def run(
        self,
        input_path: PathLike,
        *,
        animal_id: Optional[str] = None,
        skip_preprocess: bool = False,
        plots: Optional[Sequence[str]] = None,
        video_path: Optional[PathLike] = None,
        video_dir: Optional[PathLike] = None,
        annotate: Optional[bool] = None,
    ) -> dict:
        """从扁平或原始 DLC 表跑完预处理→特征→分类→可视化→（可选）视频标注。"""
        path = Path(input_path)
        meta = parse_session_name(path.name)
        stem = meta["stem"]
        animal_id = animal_id or meta.get("animal_id")
        if animal_id:
            self.config.resolve_animal(animal_id)

        session_dir = self.output_dir / "sessions" / stem
        session_dir.mkdir(parents=True, exist_ok=True)

        keypoints = self.ingest(path)
        if not skip_preprocess:
            keypoints = self.preprocess(keypoints)
        save_table(keypoints, session_dir / f"{stem}_processed.csv")

        features = self.extract_features(keypoints)
        save_table(features, session_dir / f"{stem}_features.csv")

        segments = pd.DataFrame()
        stereo = pd.DataFrame()
        circling = pd.DataFrame()
        if self.config.get("locomotion.enabled", True):
            segments = self.classify_locomotion(features, animal_id=animal_id)
            save_table(segments, session_dir / f"{stem}_features-segment.csv")
        if self.config.get("stereotypy.enabled", True):
            stereo = self.classify_stereotypy(keypoints)
            if not stereo.empty:
                save_table(stereo, session_dir / f"{stem}_stereotypy.csv")
        if self.config.get("circling.enabled", True):
            circling = self.classify_circling(keypoints)
            if not circling.empty:
                save_table(circling, session_dir / f"{stem}_circling.csv")

        artifacts = self.visualize(
            stem=stem,
            segments=segments if not segments.empty else pd.DataFrame(
                columns=["start", "end", "duration", "state", "pattern", "condition", "label", "reasons"]
            ),
            keypoints=keypoints,
            features=features,
            stereotypy=stereo if not stereo.empty else None,
            circling=circling if not circling.empty else None,
            plots=plots,
        )

        annotated = None
        do_annotate = self.config.get("video_annotation.enabled", True) if annotate is None else annotate
        if do_annotate and not segments.empty:
            annotated = self.annotate_video(
                stem,
                keypoints,
                segments,
                video_path=video_path,
                video_dir=video_dir,
                animal_id=animal_id,
            )
            if annotated:
                artifacts.setdefault("videos", []).append(annotated)

        return {
            "stem": stem,
            "animal_id": animal_id,
            "session_dir": str(session_dir),
            "n_frames": len(keypoints),
            "n_windows": len(features),
            "n_segments": len(segments),
            "n_stereotypy": int(stereo["is_stereotypy"].sum()) if not stereo.empty and "is_stereotypy" in stereo else 0,
            "n_circling": len(circling),
            "annotated_video": annotated,
            "artifacts": artifacts,
        }

    def run_directory(
        self,
        input_dir: PathLike,
        pattern: Optional[str] = None,
        *,
        animals: Optional[Sequence[str]] = None,
        days: Optional[Sequence[int]] = None,
        video_dir: Optional[PathLike] = None,
        annotate: Optional[bool] = None,
        **kwargs,
    ) -> list:
        """
        批量处理目录中的会话。

        个体 / 天数由 configs.study（或本函数 animals/days 参数）决定：
        - 列表非空：只处理匹配项（数量可变，完全由配置决定）
        - 列表为空：自动发现文件名中的全部个体与天数
        """
        folder = Path(input_dir)
        study = study_config(self.config.as_dict())
        pattern = pattern or study.get("input_pattern") or "*.csv"
        results = []
        cohort = []
        files = sorted(p for p in folder.glob(pattern) if "segment" not in p.name and "features" not in p.name)
        if not files:
            files = sorted(folder.glob(pattern))
        if video_dir is not None:
            kwargs = {**kwargs, "video_dir": video_dir}
        if annotate is not None:
            kwargs = {**kwargs, "annotate": annotate}
        for path in files:
            meta = parse_session_name(path.name) or {}
            stem = meta.get("stem") or path.stem
            if not session_allowed(stem, study, animals_override=animals, days_override=days):
                continue
            result = self.run(path, **kwargs)
            results.append(result)
            session_dir = Path(result["session_dir"])
            stem = result["stem"]
            seg_path = session_dir / f"{stem}_features-segment.csv"
            kp_path = session_dir / f"{stem}_processed.csv"
            if seg_path.is_file():
                entry = {
                    "stem": stem,
                    "label": (parse_session_name(stem) or {}).get("day_label", stem),
                    "segments": pd.read_csv(seg_path),
                    "keypoints": pd.read_csv(kp_path) if kp_path.is_file() else None,
                }
                cohort.append(entry)
        if cohort and self.config.get("visualization.plots"):
            viz = AcademicVisualizer(
                self.config["visualization"],
                self.config["cage"],
                self.output_dir / "figures",
                window_size=self.config.get("features.window_size", 13),
            )
            viz.render_cohort(cohort, "all_sessions")
        return results

    def visualize_from_sessions(
        self,
        session_dirs: Sequence[PathLike],
        *,
        plots: Optional[Sequence[str]] = None,
        stem: str = "all_sessions",
        animals: Optional[Sequence[str]] = None,
        days: Optional[Sequence[int]] = None,
    ) -> dict:
        """
        从已有 sessions/{stem}/ 目录批量出图（含四类标准图与多日对比）。
        每个目录需含 {stem}_processed.csv 与 {stem}_features-segment.csv。
        可用 study.animals / study.days（或本函数参数）筛选任意数量个体与天数。
        """
        study = study_config(self.config.as_dict())
        viz_cfg = dict(self.config["visualization"])
        if plots:
            viz_cfg["plots"] = list(plots)
        viz = AcademicVisualizer(
            viz_cfg,
            self.config["cage"],
            self.output_dir / "figures",
            window_size=self.config.get("features.window_size", 13),
        )
        cohort = []
        artifacts: dict = {"figures": [], "tables": [], "sessions": []}
        for d in session_dirs:
            d = Path(d)
            if d.is_file():
                d = d.parent
            processed = sorted(d.glob("*_processed.csv"))
            if not processed:
                continue
            p = processed[0]
            s = p.name.replace("_processed.csv", "")
            if not session_allowed(s, study, animals_override=animals, days_override=days):
                continue
            seg_path = d / f"{s}_features-segment.csv"
            if not seg_path.is_file():
                continue
            kp = pd.read_csv(p)
            segs = pd.read_csv(seg_path)
            meta = parse_session_name(s) or {}
            cohort.append(
                {
                    "stem": s,
                    "label": meta.get("day_label", s),
                    "segments": segs,
                    "keypoints": kp,
                }
            )
            session_art = viz.render_session(stem=s, segments=segs, keypoints=kp)
            artifacts["sessions"].append({"stem": s, **session_art})
            artifacts["figures"] += session_art.get("figures", [])
            artifacts["tables"] += session_art.get("tables", [])

        if cohort:
            cohort_art = viz.render_cohort(cohort, stem)
            artifacts["figures"] += cohort_art.get("figures", [])
            artifacts["tables"] += cohort_art.get("tables", [])
        return artifacts

"""DeepLabCut 官方接口封装。未安装 DLC 时以 dry-run 记录将调用的 API。"""

from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from ..config import AgentConfig

DLC_STEPS = [
    "create_project",
    "set_bodyparts",
    "extract_frames",
    "label_frames",
    "check_labels",
    "create_training_dataset",
    "train_network",
    "evaluate_network",
    "analyze_videos",
    "filterpredictions",
    "create_labeled_video",
]


def _try_import_dlc():
    try:
        return importlib.import_module("deeplabcut"), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


class DLCUnavailable(RuntimeError):
    pass


class DLCProjectManager:
    """
    对应 DeepLabCut 2.x/3.x 常用 API：

    - deeplabcut.create_new_project
    - deeplabcut.extract_frames
    - deeplabcut.label_frames
    - deeplabcut.check_labels
    - deeplabcut.create_training_dataset
    - deeplabcut.train_network
    - deeplabcut.evaluate_network
    - deeplabcut.analyze_videos
    - deeplabcut.filterpredictions
    - deeplabcut.create_labeled_video
    - deeplabcut.extract_outlier_frames / refine_labels / merge_datasets
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.dlc, self._import_error = _try_import_dlc()
        self.config_path: Optional[Path] = None
        self.log: List[Dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self.dlc is not None

    def _record(self, api: str, kwargs: dict, result: Any = None, skipped: bool = False) -> None:
        self.log.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "api": api,
                "kwargs": {k: _jsonable(v) for k, v in kwargs.items()},
                "skipped": skipped,
                "result": _jsonable(result),
            }
        )

    def _require(self) -> Any:
        if self.config.get("dlc.dry_run") or self.dlc is None:
            return None
        return self.dlc

    def create_project(
        self,
        videos: Sequence[str],
        *,
        project: Optional[str] = None,
        experimenter: Optional[str] = None,
        working_directory: Optional[str] = None,
        copy_videos: Optional[bool] = None,
        videotype: Optional[str] = None,
        bodyparts: Optional[Sequence[str]] = None,
    ) -> Path:
        videos = [str(Path(v).expanduser().resolve()) for v in videos]
        project = project or self.config.get("project.name", "monkey_cage")
        experimenter = experimenter or self.config.get("project.experimenter", "experimenter")
        working_directory = str(
            Path(
                working_directory
                or self.config.get("project.working_directory")
                or self.config.get("io.output_dir", "./output")
            ).expanduser().resolve()
        )
        copy_videos = self.config.get("project.copy_videos", True) if copy_videos is None else copy_videos
        videotype = videotype or self.config.get("project.videotype", ".mp4")
        kwargs = dict(
            project=project,
            experimenter=experimenter,
            videos=videos,
            working_directory=working_directory,
            copy_videos=copy_videos,
            videotype=videotype,
        )
        dlc = self._require()
        if dlc is None:
            fake = Path(working_directory) / f"{project}-{experimenter}-placeholder"
            fake.mkdir(parents=True, exist_ok=True)
            cfg_path = fake / "config.yaml"
            self._write_placeholder_config(cfg_path, videos, bodyparts)
            self.config_path = cfg_path
            self._record("deeplabcut.create_new_project", kwargs, str(cfg_path), skipped=True)
            return cfg_path

        config_path = dlc.create_new_project(**kwargs)
        self.config_path = Path(config_path)
        self._record("deeplabcut.create_new_project", kwargs, str(self.config_path))
        self.set_bodyparts(bodyparts)
        return self.config_path

    def set_bodyparts(
        self,
        bodyparts: Optional[Sequence[str]] = None,
        skeleton: Optional[Sequence[Sequence[str]]] = None,
        config_path: Optional[Path] = None,
    ) -> Path:
        cfg_path = Path(config_path or self.config_path or "")
        if not cfg_path:
            raise ValueError("尚未创建 DLC 项目，缺少 config.yaml")
        names = list(bodyparts or self.config.bodyparts)
        skeleton = [list(p) for p in (skeleton or self.config.get("bodyparts.skeleton") or [])]
        if not skeleton and len(names) >= 2:
            skeleton = [[names[i], names[i + 1]] for i in range(len(names) - 1)]
        data = {}
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        data["bodyparts"] = names
        data["skeleton"] = skeleton
        data["pcutoff"] = self.config.get("dlc.pcutoff", 0.6)
        data["numframes2pick"] = self.config.get("dlc.numframes2pick", 40)
        data["default_net_type"] = self.config.get("dlc.net_type", "resnet_50")
        data["dotsize"] = data.get("dotsize", 8)
        data["alphavalue"] = data.get("alphavalue", 0.7)
        data["colormap"] = data.get("colormap", "rainbow")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        self.config.set_bodyparts(names, skeleton)
        self._record(
            "edit_config.yaml[bodyparts]",
            {"bodyparts": names, "skeleton": skeleton},
            str(cfg_path),
        )
        return cfg_path

    def extract_frames(self, config_path: Optional[Path] = None) -> None:
        cfg = str(config_path or self.config_path)
        kwargs = dict(
            config=cfg,
            mode=self.config.get("dlc.extract_mode", "automatic"),
            algo=self.config.get("dlc.extract_algo", "kmeans"),
            userfeedback=self.config.get("dlc.userfeedback", False),
        )
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.extract_frames", kwargs, skipped=True)
            return
        dlc.extract_frames(**kwargs)
        self._record("deeplabcut.extract_frames", kwargs)

    def label_frames(self, config_path: Optional[Path] = None) -> None:
        """打开 DLC 标注 GUI。无显示器环境请跳过此步。"""
        kwargs = dict(config=str(config_path or self.config_path))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.label_frames", kwargs, skipped=True)
            return
        dlc.label_frames(**kwargs)
        self._record("deeplabcut.label_frames", kwargs)

    def check_labels(self, config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.check_labels", kwargs, skipped=True)
            return
        dlc.check_labels(**kwargs)
        self._record("deeplabcut.check_labels", kwargs)

    def create_training_dataset(self, config_path: Optional[Path] = None) -> None:
        kwargs = dict(
            config=str(config_path or self.config_path),
            augmenter_type=self.config.get("dlc.augmenter_type", "imgaug"),
            net_type=self.config.get("dlc.net_type", "resnet_50"),
        )
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.create_training_dataset", kwargs, skipped=True)
            return
        dlc.create_training_dataset(**kwargs)
        self._record("deeplabcut.create_training_dataset", kwargs)

    def train_network(self, config_path: Optional[Path] = None) -> None:
        kwargs = dict(
            config=str(config_path or self.config_path),
            shuffle=self.config.get("dlc.shuffle", 1),
            displayiters=self.config.get("dlc.displayiters", 1000),
            saveiters=self.config.get("dlc.saveiters", 10000),
            maxiters=self.config.get("dlc.maxiters", 1030000),
        )
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.train_network", kwargs, skipped=True)
            return
        dlc.train_network(**kwargs)
        self._record("deeplabcut.train_network", kwargs)

    def evaluate_network(self, config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path), plotting=True)
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.evaluate_network", kwargs, skipped=True)
            return
        dlc.evaluate_network(**kwargs)
        self._record("deeplabcut.evaluate_network", kwargs)

    def analyze_videos(
        self,
        videos: Sequence[str],
        config_path: Optional[Path] = None,
        destfolder: Optional[str] = None,
    ) -> List[str]:
        videos = [str(Path(v).expanduser().resolve()) for v in videos]
        destfolder = destfolder or str(self.config.output_dir / "dlc_results")
        Path(destfolder).mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            config=str(config_path or self.config_path),
            videos=videos,
            videotype=self.config.get("project.videotype", ".mp4"),
            save_as_csv=self.config.get("dlc.save_as_csv", True),
            destfolder=destfolder,
        )
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.analyze_videos", kwargs, skipped=True)
            return []
        dlc.analyze_videos(**kwargs)
        self._record("deeplabcut.analyze_videos", kwargs, destfolder)
        return videos

    def filterpredictions(self, videos: Sequence[str], config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path), video=list(videos))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.filterpredictions", kwargs, skipped=True)
            return
        dlc.filterpredictions(kwargs["config"], videos)
        self._record("deeplabcut.filterpredictions", kwargs)

    def create_labeled_video(self, videos: Sequence[str], config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path), videos=list(videos))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.create_labeled_video", kwargs, skipped=True)
            return
        dlc.create_labeled_video(kwargs["config"], videos)
        self._record("deeplabcut.create_labeled_video", kwargs)

    def extract_outlier_frames(self, videos: Sequence[str], config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path), videos=list(videos))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.extract_outlier_frames", kwargs, skipped=True)
            return
        dlc.extract_outlier_frames(kwargs["config"], videos)
        self._record("deeplabcut.extract_outlier_frames", kwargs)

    def refine_labels(self, config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.refine_labels", kwargs, skipped=True)
            return
        dlc.refine_labels(**kwargs)
        self._record("deeplabcut.refine_labels", kwargs)

    def merge_datasets(self, config_path: Optional[Path] = None) -> None:
        kwargs = dict(config=str(config_path or self.config_path))
        dlc = self._require()
        if dlc is None:
            self._record("deeplabcut.merge_datasets", kwargs, skipped=True)
            return
        dlc.merge_datasets(**kwargs)
        self._record("deeplabcut.merge_datasets", kwargs)

    def run_steps(
        self,
        videos: Sequence[str],
        steps: Optional[Sequence[str]] = None,
        bodyparts: Optional[Sequence[str]] = None,
        config_path: Optional[str] = None,
    ) -> Path:
        """按用户指定步骤执行 DLC 流程。"""
        selected = list(steps or DLC_STEPS)
        if config_path:
            self.config_path = Path(config_path)
        if "create_project" in selected or self.config_path is None:
            self.create_project(videos, bodyparts=bodyparts)
            if "create_project" in selected:
                selected = [s for s in selected if s != "create_project"]
        dispatch = {
            "set_bodyparts": lambda: self.set_bodyparts(bodyparts),
            "extract_frames": self.extract_frames,
            "label_frames": self.label_frames,
            "check_labels": self.check_labels,
            "create_training_dataset": self.create_training_dataset,
            "train_network": self.train_network,
            "evaluate_network": self.evaluate_network,
            "analyze_videos": lambda: self.analyze_videos(videos),
            "filterpredictions": lambda: self.filterpredictions(videos),
            "create_labeled_video": lambda: self.create_labeled_video(videos),
            "extract_outlier_frames": lambda: self.extract_outlier_frames(videos),
            "refine_labels": self.refine_labels,
            "merge_datasets": self.merge_datasets,
        }
        for step in selected:
            if step not in dispatch:
                raise ValueError(f"未知 DLC 步骤: {step}，可选: {list(dispatch)}")
            dispatch[step]()
        self._write_call_log()
        return self.config_path

    def _write_placeholder_config(self, path: Path, videos: Sequence[str], bodyparts: Optional[Sequence[str]]) -> None:
        names = list(bodyparts or self.config.bodyparts)
        skeleton = self.config.get("bodyparts.skeleton") or (
            [[names[i], names[i + 1]] for i in range(len(names) - 1)] if len(names) > 1 else []
        )
        video_sets = {v: {"crop": "0, 0, 0, 0"} for v in videos}
        data = {
            "Task": self.config.get("project.name", "monkey_cage"),
            "scorer": self.config.get("project.experimenter", "experimenter"),
            "date": datetime.now().strftime("%b%d"),
            "multianimalproject": False,
            "project_path": str(path.parent),
            "video_sets": video_sets,
            "bodyparts": names,
            "start": 0,
            "stop": 1,
            "numframes2pick": self.config.get("dlc.numframes2pick", 40),
            "skeleton": skeleton,
            "skeleton_color": "black",
            "pcutoff": self.config.get("dlc.pcutoff", 0.6),
            "dotsize": 8,
            "alphavalue": 0.7,
            "colormap": "rainbow",
            "TrainingFraction": [0.95],
            "iteration": 0,
            "default_net_type": self.config.get("dlc.net_type", "resnet_50"),
            "default_augmenter": self.config.get("dlc.augmenter_type", "imgaug"),
            "snapshotindex": -1,
            "batch_size": 8,
            "NOTE": "Placeholder config generated because DeepLabCut is not installed or dry_run=true.",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    def _write_call_log(self) -> None:
        out = self.config.output_dir / "dlc_api_log.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "deeplabcut_available": self.available,
                    "import_error": None if self._import_error is None else str(self._import_error),
                    "config_path": None if self.config_path is None else str(self.config_path),
                    "calls": self.log,
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value

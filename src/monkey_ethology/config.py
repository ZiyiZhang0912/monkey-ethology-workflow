from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

PathLike = Union[str, Path]


def _read_yaml(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点必须是 mapping: {path}")
    return data


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 覆盖 base。"""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _maybe_load(path: Optional[PathLike]) -> Dict[str, Any]:
    if path is None:
        return {}
    return _read_yaml(path)


class WorkflowConfig:
    """可变配置对象：YAML + 运行时覆盖。"""

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        raw = _read_yaml(DEFAULT_CONFIG_PATH)
        if data:
            raw = deep_update(raw, data)
        self._data: Dict[str, Any] = raw

    def copy(self) -> "WorkflowConfig":
        return WorkflowConfig(deepcopy(self._data))

    def as_dict(self) -> Dict[str, Any]:
        return deepcopy(self._data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def update(self, mapping: Mapping[str, Any]) -> None:
        self._data = deep_update(self._data, mapping)

    def merge_file(self, path: PathLike) -> None:
        self.update(_read_yaml(path))

    def apply_bodyparts_file(self, path: PathLike) -> None:
        self._data["bodyparts"] = deep_update(self._data.get("bodyparts", {}), _read_yaml(path))

    def apply_cage_file(self, path: PathLike) -> None:
        cage = _read_yaml(path)
        extra_features = cage.pop("features", None)
        extra_circling = cage.pop("circling", None)
        self._data["cage"] = deep_update(self._data.get("cage", {}), cage)
        if extra_features:
            self._data["features"] = deep_update(self._data.get("features", {}), extra_features)
        if extra_circling:
            self._data["circling"] = deep_update(self._data.get("circling", {}), extra_circling)

    def apply_animal_file(self, path: PathLike) -> None:
        animal = _read_yaml(path)
        loc = {k: v for k, v in animal.items() if k not in {"animal_id", "source_note"}}
        self._data["locomotion"] = deep_update(self._data.get("locomotion", {}), loc)
        if "animal_id" in animal:
            self.set("animal_id", str(animal["animal_id"]))

    def apply_visualization_file(self, path: PathLike) -> None:
        self._data["visualization"] = deep_update(
            self._data.get("visualization", {}), _read_yaml(path)
        )

    def resolve_animal(self, animal_id: Optional[str] = None) -> None:
        """绑定个体 ID。仅当存在 configs/animals/{id}.yaml 时才覆盖阈值，否则保留通用默认。"""
        animal_id = str(animal_id or self.get("animal_id") or "")
        if not animal_id:
            return
        path = PACKAGE_ROOT / "configs" / "animals" / f"{animal_id}.yaml"
        if path.is_file():
            self.apply_animal_file(path)
        self.set("animal_id", animal_id)

    def set_bodyparts(self, names: Sequence[str], skeleton: Optional[Sequence[Sequence[str]]] = None) -> None:
        self.set("bodyparts.names", list(names))
        if skeleton is not None:
            self.set("bodyparts.skeleton", [list(pair) for pair in skeleton])
        else:
            pairs = [[names[i], names[i + 1]] for i in range(len(names) - 1)]
            self.set("bodyparts.skeleton", pairs)
        current_com = list(self.get("bodyparts.com_joints") or [])
        valid = [j for j in current_com if j in names] or list(names)
        self.set("bodyparts.com_joints", valid)

    @property
    def bodyparts(self) -> list:
        return list(self.get("bodyparts.names") or [])

    @property
    def joints(self) -> list:
        return self.bodyparts

    @property
    def output_dir(self) -> Path:
        return Path(self.get("io.output_dir", "./output")).expanduser().resolve()

    def to_yaml(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, allow_unicode=True, sort_keys=False)
        return path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"WorkflowConfig(bodyparts={self.bodyparts}, cage={self.get('cage.name')})"


def _resolve_bundled(kind: str, value: Optional[PathLike]) -> Optional[Path]:
    """允许传入 configs/{kind}/{name}.yaml 的短名，或直接路径。"""
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return path
    bundled = PACKAGE_ROOT / "configs" / kind / f"{value}.yaml"
    if bundled.is_file():
        return bundled
    return path


def load_config(
    config: Optional[PathLike] = None,
    *,
    bodyparts: Optional[PathLike] = None,
    cage: Optional[PathLike] = None,
    animal: Optional[Union[str, PathLike]] = None,
    visualization: Optional[PathLike] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> WorkflowConfig:
    """加载并叠层合并配置。"""
    cfg = WorkflowConfig() if config is None else WorkflowConfig(_read_yaml(config) if Path(config).is_file() else {})
    if config is not None and Path(config).is_file() and Path(config).resolve() != DEFAULT_CONFIG_PATH.resolve():
        cfg = WorkflowConfig()
        cfg.merge_file(config)
    if bodyparts:
        cfg.apply_bodyparts_file(_resolve_bundled("bodyparts", bodyparts) or bodyparts)
    if cage:
        cfg.apply_cage_file(_resolve_bundled("cages", cage) or cage)
    if visualization:
        viz = Path(visualization)
        if not viz.is_file():
            bundled = PACKAGE_ROOT / "configs" / f"{visualization}.yaml"
            if bundled.is_file():
                viz = bundled
        cfg.apply_visualization_file(viz)
    if animal:
        animal_path = Path(animal)
        if animal_path.is_file():
            cfg.apply_animal_file(animal_path)
        else:
            cfg.resolve_animal(str(animal))
    if overrides:
        cfg.update(overrides)
    return cfg


def dump_resolved_config(cfg: WorkflowConfig, output_dir: Optional[PathLike] = None) -> Path:
    out = Path(output_dir or cfg.output_dir) / "resolved_config.yaml"
    return cfg.to_yaml(out)

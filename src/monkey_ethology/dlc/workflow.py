from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .project import DLCProjectManager
from ..config import WorkflowConfig
from ..io.schema import load_keypoints, parse_session_name, save_table


def convert_dlc_outputs(
    paths: Sequence[str],
    output_dir: str,
    aliases: Optional[dict] = None,
) -> list:
    """将 DLC h5/csv 转为扁平关键点表。"""
    out_dir = Path(output_dir)
    written = []
    for path in paths:
        df = load_keypoints(path, aliases=aliases)
        meta = parse_session_name(path)
        dest = out_dir / f"{meta['stem']}.csv"
        save_table(df, dest)
        written.append(str(dest))
    return written


def run_tracking(config: WorkflowConfig, videos: Sequence[str], steps: Optional[Sequence[str]] = None):
    manager = DLCProjectManager(config)
    cfg_path = manager.run_steps(videos, steps=steps, bodyparts=config.bodyparts)
    return {"config_path": str(cfg_path), "log": manager.log, "available": manager.available}

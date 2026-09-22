from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .project import DLCProjectManager
from .workflow import convert_dlc_outputs, run_tracking
from ..io.schema import load_keypoints, save_table
from ..config import AgentConfig


def convert_h5_to_flat_csv(data_folder: str, output_dir: str = None, aliases: dict = None) -> list:
    folder = Path(data_folder)
    dest = Path(output_dir or folder)
    files = list(folder.glob("*.h5")) + list(folder.glob("*.hdf5"))
    return convert_dlc_outputs([str(p) for p in files], str(dest), aliases=aliases)

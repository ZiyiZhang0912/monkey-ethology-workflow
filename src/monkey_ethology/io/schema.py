from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

COORD_SUFFIXES = ("x", "y", "p")

TAIL_ALIASES = {
    "Tail": "Tail(Root)",
    "TailRoot": "Tail(Root)",
    "Tail (Root)": "Tail(Root)",
    "tail": "Tail(Root)",
}


def canonicalize_joint(name: str, aliases: Optional[dict] = None) -> str:
    mapping = dict(TAIL_ALIASES)
    if aliases:
        mapping.update(aliases)
    return mapping.get(name, name)


def parse_session_name(filename: str) -> Optional[dict]:
    """解析 {id}-dayN / {id}-MMDD / 含 DLC 后缀的文件名。"""
    name = Path(filename).name
    name = re.sub(r"\.(csv|h5|hdf5|mp4|avi|mov)$", "", name, flags=re.I)
    for token in (
        "_features-segment",
        "_features",
        "_processed",
        "_reformatted",
        "_optimized",
        "-positions",
    ):
        name = name.replace(token, "")
    if "DLC" in name:
        name = name.split("DLC")[0].rstrip("_- ")

    match = re.fullmatch(r"(\d+)-day(\d+)", name, flags=re.I)
    if match:
        animal_id, day = match.groups()
        day_num = int(day)
        return {
            "animal_id": animal_id,
            "day_num": day_num,
            "day_key": f"day{day_num}",
            "day_label": f"Day {day_num:02d}",
            "stem": f"{animal_id}-day{day_num}",
        }
    match = re.fullmatch(r"(\d+)-(\d{4})", name)
    if match:
        animal_id, mmdd = match.groups()
        return {
            "animal_id": animal_id,
            "day_num": int(mmdd[2:]),
            "day_key": mmdd,
            "day_label": f"{mmdd[:2]}-{mmdd[2:]}",
            "stem": f"{animal_id}-{mmdd}",
        }
    stem = name
    return {
        "animal_id": stem,
        "day_num": None,
        "day_key": "session",
        "day_label": stem,
        "stem": stem,
    }


def infer_joints(columns: Iterable[str]) -> list:
    joints = []
    seen = set()
    for col in columns:
        if not isinstance(col, str) or "_" not in col:
            continue
        joint, suffix = col.rsplit("_", 1)
        if suffix in COORD_SUFFIXES and joint not in seen:
            seen.add(joint)
            joints.append(joint)
    return joints


def joint_columns(joint: str) -> Tuple[str, str, str]:
    return f"{joint}_x", f"{joint}_y", f"{joint}_p"


def ensure_likelihood(df: pd.DataFrame, joints: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for joint in joints:
        p_col = f"{joint}_p"
        if p_col not in out.columns:
            out[p_col] = 1.0
    return out


def flatten_dlc_dataframe(df: pd.DataFrame, aliases: Optional[dict] = None) -> pd.DataFrame:
    """
    将 DeepLabCut 多层表头 / MultiIndex 转为 {joint}_{x|y|p} 宽表。
    兼容：
      - h5 MultiIndex (scorer, bodyparts, coords)
      - 3 行表头 CSV（scorer / bodyparts / coords）
      - 已扁平化的 Head_x 格式
      - reformatted tables that include a bodyparts_x column
    """
    if isinstance(df.columns, pd.MultiIndex):
        return _from_multiindex(df, aliases)

    if _looks_like_three_row_header(df):
        return _from_three_row_csv(df, aliases)

    out = df.copy()
    if "bodyparts_x" in out.columns:
        out = out.drop(columns=["bodyparts_x"])
    out.columns = [str(c) for c in out.columns]
    out = _apply_aliases(out, aliases)
    numeric_cols = [c for c in out.columns if c.rsplit("_", 1)[-1] in COORD_SUFFIXES]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def _looks_like_three_row_header(df: pd.DataFrame) -> bool:
    """识别「首行已作列名、随后两行是 bodyparts/coords」的 DLC CSV。"""
    if df.empty or len(df) < 2:
        return False
    cols_join = " ".join(str(c).lower() for c in df.columns)
    row0 = [str(x).lower() for x in df.iloc[0].tolist()]
    row1 = [str(x).lower() for x in df.iloc[1].tolist()]
    row0_join = " ".join(row0)
    row1_join = " ".join(row1)
    # 列名含 scorer/DLC，且第 0 行为 bodyparts、第 1 行为 coords
    if ("scorer" in cols_join or "dlc_" in cols_join) and (
        "bodyparts" in row0_join or row0[0] == "bodyparts"
    ) and (
        "coords" in row1_join
        or row1[0] == "coords"
        or ("x" in row1_join and ("y" in row1_join or "likelihood" in row1_join))
    ):
        return True
    # 列名已是 scorer 重复值，第 0 行直接是 x/y/likelihood
    if ("x" in row0_join and "likelihood" in row0_join) or "coords" in cols_join:
        return True
    return False


def _from_three_row_csv(df: pd.DataFrame, aliases: Optional[dict]) -> pd.DataFrame:
    """scorer 行已作列名；第 0 行 bodyparts，第 1 行 coords。"""
    work = df.copy()
    bodyparts = work.iloc[0].tolist()
    coords = work.iloc[1].tolist()
    data = work.iloc[2:].reset_index(drop=True)
    new_cols = []
    for bp, coord in zip(bodyparts, coords):
        bp_raw = str(bp).strip()
        coord = str(coord).lower().replace("likelihood", "p").strip()
        if coord not in COORD_SUFFIXES:
            coord = {"x": "x", "y": "y", "p": "p"}.get(coord, coord)
        drop_bp = bp_raw.lower() in {"scorer", "bodyparts", "nan", "coords", ""}
        if drop_bp or coord not in COORD_SUFFIXES:
            new_cols.append(f"_drop_{len(new_cols)}")
        else:
            bp = canonicalize_joint(bp_raw, aliases)
            new_cols.append(f"{bp}_{coord}")
    data.columns = new_cols
    keep = [c for c in data.columns if not c.startswith("_drop_")]
    data = data[keep]
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return _apply_aliases(data, aliases).reset_index(drop=True)


def _from_multiindex(df: pd.DataFrame, aliases: Optional[dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        parts = [str(x) for x in (col if isinstance(col, tuple) else (col,))]
        # (scorer, bodypart, coord)
        if len(parts) >= 3:
            joint = canonicalize_joint(parts[-2], aliases)
            coord = parts[-1].lower().replace("likelihood", "p")
        elif len(parts) == 2:
            joint = canonicalize_joint(parts[0], aliases)
            coord = parts[1].lower().replace("likelihood", "p")
        else:
            continue
        if coord not in COORD_SUFFIXES:
            continue
        out[f"{joint}_{coord}"] = pd.to_numeric(df[col], errors="coerce")
    return out.reset_index(drop=True)


def _apply_aliases(df: pd.DataFrame, aliases: Optional[dict]) -> pd.DataFrame:
    mapping = dict(TAIL_ALIASES)
    if aliases:
        mapping.update(aliases)
    rename = {}
    for col in df.columns:
        if "_" not in str(col):
            continue
        joint, suffix = str(col).rsplit("_", 1)
        canon = mapping.get(joint, joint)
        new = f"{canon}_{suffix}"
        if new != col:
            rename[col] = new
    if rename:
        df = df.rename(columns=rename)
    return df


def load_keypoints(path, aliases: Optional[dict] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        df = pd.read_hdf(path)
    else:
        # 普通扁平表；若像未展开的 DLC 三行表头，改用 MultiIndex 读取
        df = pd.read_csv(path)
        if _looks_like_three_row_header(df) or _is_probably_unparsed_dlc(df):
            try:
                df = pd.read_csv(path, header=[0, 1, 2])
            except Exception:
                pass  # 保留首读结果，交由 flatten 的 three-row 路径处理
    return flatten_dlc_dataframe(df, aliases)


def _is_probably_unparsed_dlc(df: pd.DataFrame) -> bool:
    """兼容旧检测：列名像 DLC，且前两行分别像 bodyparts / coords。"""
    return _looks_like_three_row_header(df)


def compute_com(
    df: pd.DataFrame,
    joints: Sequence[str],
    p_threshold: float = 0.5,
    weighted: bool = True,
) -> pd.DataFrame:
    """按帧用可用关节点计算重心。"""
    xs, ys, ws = [], [], []
    for joint in joints:
        x_col, y_col, p_col = joint_columns(joint)
        if x_col not in df.columns or y_col not in df.columns:
            continue
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        p = (
            pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
            if p_col in df.columns
            else np.ones_like(x)
        )
        valid = (p >= p_threshold) & np.isfinite(x) & np.isfinite(y)
        xs.append(np.where(valid, x, np.nan))
        ys.append(np.where(valid, y, np.nan))
        ws.append(np.where(valid, p if weighted else 1.0, np.nan))
    if not xs:
        n = len(df)
        return pd.DataFrame({"com_x": np.full(n, np.nan), "com_y": np.full(n, np.nan), "com_p": np.full(n, np.nan)})
    X = np.vstack(xs)
    Y = np.vstack(ys)
    W = np.vstack(ws)
    wsum = np.nansum(W, axis=0)
    com_x = np.nansum(X * W, axis=0) / np.where(wsum == 0, np.nan, wsum)
    com_y = np.nansum(Y * W, axis=0) / np.where(wsum == 0, np.nan, wsum)
    with np.errstate(all="ignore"):
        com_p = np.nanmean(W, axis=0)
    return pd.DataFrame({"com_x": com_x, "com_y": com_y, "com_p": com_p})


def window_to_frame(index: int, window_size: int) -> int:
    return int(index) * int(window_size)


def save_table(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

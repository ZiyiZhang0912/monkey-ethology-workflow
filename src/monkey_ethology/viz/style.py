from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

LABELS = {
    "zh": {
        "ethogram": "行为时间谱 (Ethogram)",
        "time": "时间 (s)",
        "time_min": "时间 (min)",
        "proportion": "时间占比 (%)",
        "duration": "片段时长 (s)",
        "count": "片段数",
        "behavior": "行为",
        "trajectory": "轨迹",
        "heatmap": "驻留热图",
        "velocity": "速度 (cm/s)",
        "x": "X (px)",
        "y": "Y (px)",
        "budget": "时间预算",
        "bout": "行为片段时长分布",
        "transition": "行为转移概率",
        "features": "运动学特征",
        "position": "笼内分区",
        "quality": "追踪质量",
        "stereotypy": "刻板行为特征",
        "circling": "转圈事件",
        "from": "从",
        "to": "到",
        "session": "会话",
        "animal": "个体",
        "likelihood": "置信度",
        "frame": "帧",
        "publication": "综合分析",
        "upper": "上部",
        "lower": "下部",
        "left": "左侧",
        "right": "右侧",
        "com_trajectory_heatmap": "重心轨迹与驻留热图",
        "body_center": "身体中心位置",
        "position_analysis": "笼内位置分析",
        "behavior_comparison": "行为状态对比",
    },
    "en": {
        "ethogram": "Ethogram",
        "time": "Time (s)",
        "time_min": "Time (min)",
        "proportion": "Time budget (%)",
        "duration": "Bout duration (s)",
        "count": "Bout count",
        "behavior": "Behavior",
        "trajectory": "Trajectory",
        "heatmap": "Occupancy heatmap",
        "velocity": "Speed (cm/s)",
        "x": "X (px)",
        "y": "Y (px)",
        "budget": "Time budget",
        "bout": "Bout duration distribution",
        "transition": "Transition probability",
        "features": "Kinematic features",
        "position": "Cage quadrants",
        "quality": "Tracking quality",
        "stereotypy": "Stereotypy features",
        "circling": "Circling events",
        "from": "From",
        "to": "To",
        "session": "Session",
        "animal": "Animal",
        "likelihood": "Likelihood",
        "frame": "Frame",
        "publication": "Summary figure",
        "upper": "Upper",
        "lower": "Lower",
        "left": "Left",
        "right": "Right",
        "com_trajectory_heatmap": "CoM trajectory & heatmap",
        "body_center": "Body center position",
        "position_analysis": "Position analysis",
        "behavior_comparison": "Behavior state comparison",
    },
}

# LiuZhen SmallCage BehaviorStateVisualization defaults
DEFAULT_COLORS = {
    "Static": "#9E9E9E",
    "Walking": "#64B5F6",
    "Climbing": "#81C784",
    "stereotypy": "#D55E00",
    "circling_cw": "#CC79A7",
    "circling_ccw": "#E69F00",
}


def _font_list(preferred: str) -> list:
    return [
        preferred,
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Hiragino Sans GB",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]


def apply_style(cfg: Mapping) -> None:
    family = cfg.get("font_family", "Arial")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _font_list(family),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": int(cfg.get("dpi", 300)),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    if cfg.get("style") == "presentation":
        mpl.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 13, "font.size": 12})


def t(cfg: Mapping, key: str) -> str:
    lang = cfg.get("language", "zh")
    return LABELS.get(lang, LABELS["zh"]).get(key, key)


def colors(cfg: Mapping) -> dict:
    merged = dict(DEFAULT_COLORS)
    merged.update(cfg.get("colors") or {})
    return merged


def save_figure(fig, path_stem, formats: Sequence[str]) -> list:
    saved = []
    for fmt in formats:
        dest = f"{path_stem}.{fmt.lstrip('.')}"
        fig.savefig(dest, bbox_inches="tight", facecolor="white")
        saved.append(dest)
    plt.close(fig)
    return saved

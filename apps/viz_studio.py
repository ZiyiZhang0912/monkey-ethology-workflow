"""交互式可视化工作室：选择图类型、配色、语言后重新渲染。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from monkey_ethology.workflow import EthologyWorkflow
from monkey_ethology.config import load_config

ALL_PLOTS = [
    "ethogram", "time_budget", "bout_duration", "transition",
    "trajectory", "heatmap", "velocity", "features",
    "position", "quality", "stereotypy", "circling", "publication_figure",
]


def main():
    st.set_page_config(page_title="Ethology Viz Studio", layout="wide")
    st.title("行为分析可视化工作室")
    st.caption("所有参数仅作用于本仓库 output。")

    segs_path = st.text_input("segment CSV", "")
    kp_path = st.text_input("processed keypoints CSV（可选）", "")
    feat_path = st.text_input("features CSV（可选）", "")
    language = st.selectbox("语言", ["zh", "en"])
    plots = st.multiselect("图类型", ALL_PLOTS, default=["ethogram", "time_budget", "trajectory", "heatmap"])
    dpi = st.slider("DPI", 100, 400, 200)
    colormap = st.selectbox("热图 colormap", ["cividis", "viridis", "magma", "plasma", "Greys"])

    c1, c2, c3 = st.columns(3)
    static_c = c1.color_picker("Static", "#7F7F7F")
    walk_c = c2.color_picker("Walking", "#0072B2")
    climb_c = c3.color_picker("Climbing", "#009E73")

    if st.button("生成") and segs_path:
        cfg = load_config()
        cfg.set("io.output_dir", str(ROOT / "output" / "viz_studio"))
        cfg.set("visualization.language", language)
        cfg.set("visualization.plots", plots)
        cfg.set("visualization.dpi", dpi)
        cfg.set("visualization.colormap", colormap)
        cfg.set("visualization.formats", ["png"])
        cfg.set("visualization.colors.Static", static_c)
        cfg.set("visualization.colors.Walking", walk_c)
        cfg.set("visualization.colors.Climbing", climb_c)
        workflow = EthologyWorkflow(cfg)
        segs = pd.read_csv(segs_path)
        kp = pd.read_csv(kp_path) if kp_path else None
        feat = pd.read_csv(feat_path) if feat_path else None
        stem = Path(segs_path).stem.replace("_features-segment", "")
        artifacts = workflow.visualize(stem, segs, kp, feat, plots=plots)
        st.json(artifacts)
        for fig in artifacts.get("figures", []):
            if fig.endswith(".png"):
                st.image(fig)


if __name__ == "__main__":
    main()

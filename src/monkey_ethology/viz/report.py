from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .style import apply_style, colors, save_figure, t
from .standard_plots import (
    plot_behavior_comparison,
    plot_body_center,
    plot_com_trajectory_heatmap,
    plot_position_analysis,
)
from ..io.schema import compute_com, infer_joints, parse_session_name, window_to_frame
from ..config import PACKAGE_ROOT


def expand_segments(segments: pd.DataFrame, window_size: int, fps: float, max_sec: float = 0) -> pd.DataFrame:
    """Match BehaviorStateVisualization: start_frame = start * window_size."""
    out = segments.copy()
    if "start_sec" not in out.columns:
        out["start_frame"] = out["start"].astype(int) * window_size
        out["end_frame"] = out["end"].astype(int) * window_size
        out["start_sec"] = out["start_frame"] / fps
        out["end_sec"] = out["end_frame"] / fps
    if max_sec and max_sec > 0:
        out = out[out["start_frame"] < max_sec * fps].copy() if "start_frame" in out.columns else out[out["start_sec"] < max_sec].copy()
        out.loc[out["end_sec"] > max_sec, "end_sec"] = max_sec
        if "end_frame" in out.columns:
            out.loc[out["end_frame"] > max_sec * fps, "end_frame"] = int(max_sec * fps)
    out["duration_sec"] = out["end_sec"] - out["start_sec"]
    if "label" not in out.columns:
        mapping = {0: "Static", 2: "Walking", 3: "Climbing"}
        out["label"] = out["pattern"].map(lambda p: mapping.get(int(p), "Static"))
    return out


def merge_adjacent(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, row in df.sort_values("start_sec").iterrows():
        if rows and rows[-1]["label"] == row["label"]:
            rows[-1]["end_sec"] = max(rows[-1]["end_sec"], row["end_sec"])
            rows[-1]["duration_sec"] = rows[-1]["end_sec"] - rows[-1]["start_sec"]
        else:
            rows.append(row.to_dict())
    return pd.DataFrame(rows)


def _session_title(stem: str) -> str:
    """e.g. Monkey 562 Day 01 — Behavior Ethogram"""
    info = parse_session_name(stem) or {}
    animal = info.get("animal_id") or stem
    day = info.get("day_label")
    if day and day != stem:
        return f"Monkey {animal} {day} — Behavior Ethogram"
    return f"{stem} — Behavior Ethogram"


def _format_time_axis_minutes(ax, xmax_sec: float) -> None:
    if xmax_sec <= 0:
        return
    step = 5 * 60 if xmax_sec >= 20 * 60 else 60
    ticks = np.arange(0, xmax_sec + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // 60)}" for t in ticks])
    ax.set_xlabel("Time (min)")


class AcademicVisualizer:
    """用户可通过 visualization.plots / colors / language / formats 自由裁剪输出。"""

    def __init__(self, viz_cfg: Mapping, cage_cfg: Mapping, output_dir, window_size: int = 13):
        self.cfg = dict(viz_cfg)
        self.cage = dict(cage_cfg)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.fps = float(self.cage.get("fps", 25))
        self.max_sec = float(self.cage.get("session_max_sec") or 0)
        self.pixel_to_cm = float(self.cage.get("pixel_to_cm", 1.0))
        apply_style(self.cfg)
        self.palette = colors(self.cfg)
        self.formats = list(self.cfg.get("formats") or ["png"])
        self.requested = set(self.cfg.get("plots") or ["all"])
        if "all" in self.requested:
            self.requested = {
                "ethogram", "raster", "time_budget", "bout_duration", "transition",
                "trajectory", "heatmap", "velocity", "features", "position",
                "quality", "stereotypy", "circling", "publication_figure",
                "com_trajectory_heatmap", "body_center",
                "position_analysis", "behavior_comparison",
            }
        self.background_image = self._resolve_background(self.cfg.get("background_image"))

    def _resolve_background(self, path) -> Optional[str]:
        if not path:
            # 默认使用打包的笼背景
            default = PACKAGE_ROOT / "assets" / "background.jpg"
            return str(default) if default.is_file() else None
        p = Path(path)
        if p.is_file():
            return str(p)
        cand = PACKAGE_ROOT / path
        if cand.is_file():
            return str(cand)
        return None

    def want(self, name: str) -> bool:
        return name in self.requested

    def _cohort_sessions(
        self,
        stem: str,
        segments: pd.DataFrame,
        keypoints: Optional[pd.DataFrame],
        extra_sessions: Optional[List[dict]],
    ) -> List[dict]:
        """合并当前会话与 extra_sessions，供多日对比图使用。"""
        by_stem = {}
        if keypoints is not None and segments is not None:
            by_stem[stem] = {"stem": stem, "segments": segments, "keypoints": keypoints}
        for item in extra_sessions or []:
            s = item.get("stem") or item.get("label")
            if not s:
                continue
            entry = {
                "stem": s,
                "segments": item["segments"],
                "keypoints": item.get("keypoints"),
            }
            if entry["keypoints"] is None and s in by_stem:
                entry["keypoints"] = by_stem[s].get("keypoints")
            by_stem[s] = entry
        return [by_stem[k] for k in sorted(by_stem.keys())]

    def plot_com_trajectory_heatmap(self, keypoints: pd.DataFrame, segs: pd.DataFrame, stem: str) -> list:
        return plot_com_trajectory_heatmap(
            keypoints,
            segs,
            stem=stem,
            output_dir=self.output_dir,
            formats=self.formats,
            fps=self.fps,
            window_size=self.window_size,
            max_sec=self.max_sec,
            pixel_to_cm=self.pixel_to_cm,
            background_image=self.background_image,
        )

    def plot_body_center(self, keypoints: pd.DataFrame, stem: str) -> list:
        return plot_body_center(
            keypoints,
            stem=stem,
            output_dir=self.output_dir,
            formats=self.formats,
            fps=self.fps,
            max_sec=self.max_sec,
            climbing_y=float(self.cage.get("climbing_y", 202)),
            width=float(self.cage.get("width", 720)),
            height=float(self.cage.get("height", 406)),
        )

    def plot_position_analysis(self, sessions: List[dict], stem: str = "all_sessions") -> list:
        return plot_position_analysis(
            sessions,
            output_dir=self.output_dir,
            formats=self.formats,
            fps=self.fps,
            window_size=self.window_size,
            max_sec=self.max_sec,
            pixel_to_cm=self.pixel_to_cm,
            width=float(self.cage.get("width", 720)),
            height=float(self.cage.get("height", 406)),
            stem=stem,
        )

    def plot_behavior_comparison(self, sessions: List[dict], stem: str = "all_sessions") -> list:
        return plot_behavior_comparison(
            sessions,
            output_dir=self.output_dir,
            formats=self.formats,
            fps=self.fps,
            window_size=self.window_size,
            max_sec=self.max_sec,
            pixel_to_cm=self.pixel_to_cm,
            stem=stem,
        )

    def render_cohort(self, sessions: List[dict], stem: str = "all_sessions") -> dict:
        """多会话对比图。个体数/天数可变：栅格按个体分图，对比面板纳入全部会话。"""
        from ..study import group_sessions_by_animal

        artifacts = {"figures": [], "tables": []}
        if self.want("raster"):
            groups = group_sessions_by_animal(sessions)
            if len(groups) <= 1:
                artifacts["figures"] += self.plot_raster(sessions, stem)
            else:
                for aid, items in groups.items():
                    artifacts["figures"] += self.plot_raster(items, f"{aid}_raster")
        if self.want("position_analysis"):
            artifacts["figures"] += self.plot_position_analysis(sessions, stem)
        if self.want("behavior_comparison"):
            artifacts["figures"] += self.plot_behavior_comparison(sessions, stem)
        return artifacts

    def render_session(
        self,
        *,
        stem: str,
        segments: pd.DataFrame,
        keypoints: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
        stereotypy: Optional[pd.DataFrame] = None,
        circling: Optional[pd.DataFrame] = None,
        extra_sessions: Optional[List[dict]] = None,
    ) -> dict:
        segs = merge_adjacent(expand_segments(segments, self.window_size, self.fps, self.max_sec))
        artifacts = {"figures": [], "tables": []}
        if self.want("ethogram"):
            artifacts["figures"] += self.plot_ethogram(segs, stem)
        if self.want("time_budget"):
            artifacts["figures"] += self.plot_time_budget(segs, stem)
        if self.want("bout_duration"):
            artifacts["figures"] += self.plot_bout_duration(segs, stem)
        if self.want("transition"):
            artifacts["figures"] += self.plot_transition(segs, stem)
        if keypoints is not None and self.want("trajectory"):
            artifacts["figures"] += self.plot_trajectory(keypoints, segs, stem)
        if keypoints is not None and self.want("heatmap"):
            artifacts["figures"] += self.plot_heatmap(keypoints, stem)
        if keypoints is not None and self.want("com_trajectory_heatmap"):
            artifacts["figures"] += self.plot_com_trajectory_heatmap(keypoints, segments, stem)
        if keypoints is not None and self.want("body_center"):
            artifacts["figures"] += self.plot_body_center(keypoints, stem)
        if keypoints is not None and self.want("velocity"):
            artifacts["figures"] += self.plot_velocity(keypoints, segs, stem)
        if keypoints is not None and self.want("position"):
            artifacts["figures"] += self.plot_position(keypoints, segs, stem)
        if keypoints is not None and self.want("quality"):
            artifacts["figures"] += self.plot_quality(keypoints, stem)
        if features is not None and self.want("features"):
            artifacts["figures"] += self.plot_features(features, stem)
        if stereotypy is not None and not stereotypy.empty and self.want("stereotypy"):
            artifacts["figures"] += self.plot_stereotypy(stereotypy, stem)
        if circling is not None and not circling.empty and self.want("circling"):
            artifacts["figures"] += self.plot_circling(circling, keypoints, stem)
        if extra_sessions and self.want("raster"):
            artifacts["figures"] += self.plot_raster(extra_sessions, stem)
        if extra_sessions:
            cohort = self._cohort_sessions(stem, segments, keypoints, extra_sessions)
            if self.want("position_analysis"):
                artifacts["figures"] += self.plot_position_analysis(cohort, stem)
            if self.want("behavior_comparison"):
                artifacts["figures"] += self.plot_behavior_comparison(cohort, stem)
        if self.want("publication_figure"):
            artifacts["figures"] += self.plot_publication(segs, keypoints, features, stem)
        if self.cfg.get("export_csv", True):
            table = self.output_dir / f"{stem}_behavior_summary.csv"
            summary = self.summarize(segs, keypoints)
            summary.to_csv(table, index=False)
            artifacts["tables"].append(str(table))
            segs.to_csv(self.output_dir / f"{stem}_ethogram_segments.csv", index=False)
        return artifacts

    def summarize(self, segs: pd.DataFrame, keypoints: Optional[pd.DataFrame]) -> pd.DataFrame:
        times, pcts, total = _budget(segs, self.max_sec)
        rows = []
        for label in ["Static", "Walking", "Climbing"]:
            sub = segs[segs["label"] == label] if not segs.empty else segs
            time_s = times[label]
            rows.append(
                {
                    "behavior": label,
                    "time_sec": time_s,
                    "time_pct": pcts[label],
                    "n_bouts": int(len(sub)),
                    "mean_bout_sec": float(sub["duration_sec"].mean()) if len(sub) else 0.0,
                    "median_bout_sec": float(sub["duration_sec"].median()) if len(sub) else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def plot_ethogram(self, segs: pd.DataFrame, stem: str) -> list:
        """Ethogram + pie matching BehaviorStateVisualization.plot_single_session_ethogram."""
        total_duration = self.max_sec if self.max_sec and self.max_sec > 0 else (
            float(segs["end_sec"].max()) if not segs.empty else 1800.0
        )
        times, pcts, total = _budget(segs, total_duration)
        fig, (ax_etho, ax_pie) = plt.subplots(
            1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3.2, 1]}
        )
        for label in ["Static", "Walking", "Climbing"]:
            spans = [
                (float(row["start_sec"]), float(row["duration_sec"]))
                for _, row in segs.iterrows()
                if row["label"] == label and row["duration_sec"] > 0
            ]
            if spans:
                ax_etho.broken_barh(
                    spans,
                    (0.1, 0.8),
                    facecolors=self.palette.get(label, "#999"),
                    edgecolors="none",
                )
        ax_etho.set_ylim(0, 1)
        ax_etho.set_yticks([0.5])
        ax_etho.set_yticklabels(["Behavior"])
        ax_etho.set_xlim(0, total_duration)
        ax_etho.set_title(_session_title(stem), fontsize=13, fontweight="bold")
        _format_time_axis_minutes(ax_etho, total_duration)
        handles = [
            Patch(facecolor=self.palette[k], edgecolor="none", label=k)
            for k in ["Static", "Walking", "Climbing"]
        ]
        ax_etho.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

        pie_values = [times[k] for k in ["Static", "Walking", "Climbing"]]
        pie_colors = [self.palette[k] for k in ["Static", "Walking", "Climbing"]]
        pie_labels = [
            f"{k}\n{pcts[k]:.1f}%" if times[k] > total * 0.01 else ""
            for k in ["Static", "Walking", "Climbing"]
        ]
        if sum(pie_values) <= 0:
            pie_values = [1, 0, 0]
        ax_pie.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        ax_pie.set_title("Time Share", fontsize=12, fontweight="bold")
        summary_lines = [
            f"{k}: {times[k]:.0f}s ({pcts[k]:.1f}%)" for k in ["Static", "Walking", "Climbing"]
        ]
        ax_pie.text(
            0.5,
            -0.12,
            "\n".join(summary_lines),
            transform=ax_pie.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            family="monospace",
        )
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_ethogram", self.formats)

    def plot_time_budget(self, segs: pd.DataFrame, stem: str) -> list:
        times, pcts, _ = _budget(segs, self.max_sec)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        labels = list(times)
        ax.bar(labels, [pcts[k] for k in labels], color=[self.palette[k] for k in labels], width=0.65)
        ax.set_ylabel(t(self.cfg, "proportion"))
        ax.set_title(t(self.cfg, "budget"))
        ax.set_ylim(0, 100)
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_time_budget", self.formats)

    def plot_bout_duration(self, segs: pd.DataFrame, stem: str) -> list:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        data, labs, cols = [], [], []
        for label in ["Static", "Walking", "Climbing"]:
            vals = segs.loc[segs["label"] == label, "duration_sec"].to_numpy()
            if len(vals):
                data.append(vals)
                labs.append(label)
                cols.append(self.palette[label])
        if data:
            parts = ax.violinplot(data, showmedians=True, showextrema=False)
            for body, col in zip(parts["bodies"], cols):
                body.set_facecolor(col)
                body.set_alpha(0.7)
            parts["cmedians"].set_color("black")
            ax.set_xticks(range(1, len(labs) + 1), labs)
        ax.set_ylabel(t(self.cfg, "duration"))
        ax.set_title(t(self.cfg, "bout"))
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_bout_duration", self.formats)

    def plot_transition(self, segs: pd.DataFrame, stem: str) -> list:
        order = ["Static", "Walking", "Climbing"]
        mat = np.zeros((3, 3))
        labels = segs.sort_values("start_sec")["label"].tolist()
        for a, b in zip(labels, labels[1:]):
            if a in order and b in order:
                mat[order.index(a), order.index(b)] += 1
        rs = mat.sum(axis=1, keepdims=True)
        prob = np.divide(mat, rs, out=np.zeros_like(mat), where=rs > 0)
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        im = ax.imshow(prob, cmap=self.cfg.get("colormap", "cividis"), vmin=0, vmax=1)
        ax.set_xticks(range(3), order, rotation=30, ha="right")
        ax.set_yticks(range(3), order)
        ax.set_xlabel(t(self.cfg, "to"))
        ax.set_ylabel(t(self.cfg, "from"))
        ax.set_title(t(self.cfg, "transition"))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{prob[i, j]:.2f}", ha="center", va="center", color="white" if prob[i, j] > 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_transition", self.formats)

    def _com(self, keypoints: pd.DataFrame) -> pd.DataFrame:
        joints = self.cfg.get("com_joints") or infer_joints(keypoints.columns)
        return compute_com(keypoints, joints, p_threshold=0.3, weighted=True)

    def _frame_labels(self, n: int, segs: pd.DataFrame) -> np.ndarray:
        labels = np.array(["Static"] * n, dtype=object)
        for _, row in segs.iterrows():
            a = int(row.get("start_frame", row["start_sec"] * self.fps))
            b = int(row.get("end_frame", row["end_sec"] * self.fps))
            labels[max(0, a) : min(n, b + 1)] = row["label"]
        return labels

    def plot_trajectory(self, keypoints: pd.DataFrame, segs: pd.DataFrame, stem: str) -> list:
        com = self._com(keypoints)
        n = min(len(com), int(self.max_sec * self.fps) if self.max_sec else len(com))
        com = com.iloc[:n]
        labs = self._frame_labels(n, segs)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        bg = self.cfg.get("background_image")
        if bg and Path(bg).is_file():
            img = plt.imread(bg)
            ax.imshow(img, extent=[0, self.cage.get("width", 720), self.cage.get("height", 406), 0], alpha=0.55)
        for label in ["Static", "Walking", "Climbing"]:
            mask = labs == label
            ax.plot(com.loc[mask, "com_x"], com.loc[mask, "com_y"], ".", ms=1.2, color=self.palette[label], label=label, alpha=0.7)
        ax.set_xlim(self.cage.get("x_min", 0), self.cage.get("x_max", self.cage.get("width", 720)))
        ax.set_ylim(self.cage.get("y_max", self.cage.get("height", 406)), self.cage.get("y_min", 0))
        ax.set_xlabel(t(self.cfg, "x"))
        ax.set_ylabel(t(self.cfg, "y"))
        ax.set_title(t(self.cfg, "trajectory"))
        ax.legend(markerscale=6, loc="upper right")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_trajectory", self.formats)

    def plot_heatmap(self, keypoints: pd.DataFrame, stem: str) -> list:
        com = self._com(keypoints)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        hb = ax.hist2d(
            com["com_x"].dropna(),
            com["com_y"].dropna(),
            bins=int(self.cfg.get("heatmap_bins", 40)),
            cmap=self.cfg.get("colormap", "cividis"),
            range=[
                [self.cage.get("x_min", 0), self.cage.get("x_max", self.cage.get("width", 720))],
                [self.cage.get("y_min", 0), self.cage.get("y_max", self.cage.get("height", 406))],
            ],
        )
        ax.set_ylim(self.cage.get("y_max", 406), self.cage.get("y_min", 0))
        ax.set_xlabel(t(self.cfg, "x"))
        ax.set_ylabel(t(self.cfg, "y"))
        ax.set_title(t(self.cfg, "heatmap"))
        fig.colorbar(hb[3], ax=ax, fraction=0.046)
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_heatmap", self.formats)

    def plot_velocity(self, keypoints: pd.DataFrame, segs: pd.DataFrame, stem: str) -> list:
        com = self._com(keypoints)
        dx = com["com_x"].diff()
        dy = com["com_y"].diff()
        speed = np.hypot(dx, dy) * self.pixel_to_cm * self.fps
        time = np.arange(len(speed)) / self.fps
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.plot(time, speed, color="#333", lw=0.6, alpha=0.85)
        ymax = np.nanpercentile(speed, 99) if np.isfinite(speed).any() else 1
        for _, row in segs.iterrows():
            ax.axvspan(row["start_sec"], row["end_sec"], color=self.palette.get(row["label"], "#ddd"), alpha=0.18, lw=0)
        ax.set_xlabel(t(self.cfg, "time"))
        ax.set_ylabel(t(self.cfg, "velocity"))
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
        ax.set_title(t(self.cfg, "velocity"))
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_velocity", self.formats)

    def plot_position(self, keypoints: pd.DataFrame, segs: pd.DataFrame, stem: str) -> list:
        com = self._com(keypoints)
        w = float(self.cage.get("width", 720))
        h = float(self.cage.get("height", 406))
        xth, yth = w / 2, h / 2
        quad = pd.Series("other", index=com.index)
        quad[(com["com_x"] < xth) & (com["com_y"] < yth)] = f"{t(self.cfg, 'left')}/{t(self.cfg, 'upper')}"
        quad[(com["com_x"] >= xth) & (com["com_y"] < yth)] = f"{t(self.cfg, 'right')}/{t(self.cfg, 'upper')}"
        quad[(com["com_x"] < xth) & (com["com_y"] >= yth)] = f"{t(self.cfg, 'left')}/{t(self.cfg, 'lower')}"
        quad[(com["com_x"] >= xth) & (com["com_y"] >= yth)] = f"{t(self.cfg, 'right')}/{t(self.cfg, 'lower')}"
        counts = quad.value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        counts.plot(kind="bar", ax=ax, color="#0072B2", width=0.7)
        ax.set_ylabel(t(self.cfg, "proportion"))
        ax.set_title(t(self.cfg, "position"))
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_position", self.formats)

    def plot_quality(self, keypoints: pd.DataFrame, stem: str) -> list:
        joints = infer_joints(keypoints.columns)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for joint in joints:
            p_col = f"{joint}_p"
            if p_col in keypoints.columns:
                ax.plot(np.arange(len(keypoints)) / self.fps, keypoints[p_col], lw=0.7, label=joint, alpha=0.85)
        ax.axhline(0.6, color="#D55E00", ls="--", lw=0.8, label="p=0.6")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(t(self.cfg, "time"))
        ax.set_ylabel(t(self.cfg, "likelihood"))
        ax.set_title(t(self.cfg, "quality"))
        ax.legend(ncol=min(4, max(len(joints), 1)))
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_quality", self.formats)

    def plot_features(self, features: pd.DataFrame, stem: str) -> list:
        cols = [c for c in ["mean_velocity", "displacement", "head_tail_y_difference", "mean_com_y", "mean_periodicity"] if c in features.columns]
        if not cols:
            return []
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.1 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        x = features.get("window_index", pd.Series(range(len(features)))) * self.window_size / self.fps
        for ax, col in zip(axes, cols):
            ax.plot(x, features[col], color="#333", lw=0.8)
            ax.set_ylabel(col)
        axes[-1].set_xlabel(t(self.cfg, "time"))
        axes[0].set_title(t(self.cfg, "features"))
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_features", self.formats)

    def plot_stereotypy(self, stereo: pd.DataFrame, stem: str) -> list:
        fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)
        tvec = stereo["time"]
        series = [
            ("periodicity_score", "Periodicity"),
            ("angular_std", "Angular velocity SD"),
            ("circle_error", "Circle fit error"),
            ("cumulative_displacement", "Cumulative angle (rad)"),
        ]
        for ax, (col, title) in zip(axes, series):
            ax.plot(tvec, stereo[col], color="#0072B2", lw=1.0)
            if "is_stereotypy" in stereo.columns:
                hits = stereo[stereo["is_stereotypy"]]
                ax.scatter(hits["time"], hits[col], s=8, color=self.palette["stereotypy"], zorder=3)
            ax.set_ylabel(title)
        axes[-1].set_xlabel(t(self.cfg, "time"))
        axes[0].set_title(t(self.cfg, "stereotypy"))
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_stereotypy", self.formats)

    def plot_circling(self, circling: pd.DataFrame, keypoints: Optional[pd.DataFrame], stem: str) -> list:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        if keypoints is not None:
            com = self._com(keypoints)
            ax.plot(com["com_x"], com["com_y"], color="#bbbbbb", lw=0.4, alpha=0.8)
        for _, ev in circling.iterrows():
            col = self.palette["circling_cw"] if ev.get("direction") == "clockwise" else self.palette["circling_ccw"]
            ax.scatter(ev.get("center_x"), ev.get("center_y"), c=col, s=28, zorder=3)
        ax.set_title(t(self.cfg, "circling") + f" (n={len(circling)})")
        ax.set_xlabel(t(self.cfg, "x"))
        ax.set_ylabel(t(self.cfg, "y"))
        ax.set_ylim(self.cage.get("y_max", 406), self.cage.get("y_min", 0))
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_circling", self.formats)

    def plot_raster(self, sessions: List[dict], stem: str) -> list:
        """Multi-day ethogram raster; titles like BehaviorStateVisualization."""
        n = len(sessions)
        total_duration = self.max_sec if self.max_sec and self.max_sec > 0 else 1800.0
        fig_h = max(2.5, 0.7 * n + 1.5)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        animal_ids = set()
        for i, item in enumerate(sessions):
            segs = merge_adjacent(
                expand_segments(item["segments"], self.window_size, self.fps, self.max_sec)
            )
            y = n - 1 - i
            for label in ["Static", "Walking", "Climbing"]:
                spans = [
                    (float(row["start_sec"]), float(row["duration_sec"]))
                    for _, row in segs.iterrows()
                    if row["label"] == label and row["duration_sec"] > 0
                ]
                if spans:
                    ax.broken_barh(
                        spans,
                        (y + 0.1, 0.8),
                        facecolors=self.palette.get(label, "#999"),
                        edgecolors="none",
                    )
            info = parse_session_name(item.get("stem", "")) or {}
            if info.get("animal_id"):
                animal_ids.add(info["animal_id"])
        day_labels = []
        for item in sessions:
            info = parse_session_name(item.get("stem", item.get("label", ""))) or {}
            day_labels.append(item.get("label") or info.get("day_label") or item.get("stem", ""))
        ax.set_ylim(0, n)
        ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
        ax.set_yticklabels(day_labels, fontsize=10)
        ax.set_xlim(0, total_duration)
        animal = next(iter(animal_ids)) if len(animal_ids) == 1 else stem
        ax.set_title(
            f"Monkey {animal} — Behavior Ethogram (by day)",
            fontsize=13,
            fontweight="bold",
        )
        _format_time_axis_minutes(ax, total_duration)
        handles = [
            Patch(facecolor=self.palette[k], edgecolor="none", label=k)
            for k in ["Static", "Walking", "Climbing"]
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        return save_figure(fig, self.output_dir / f"{stem}_raster", self.formats)

    def plot_publication(self, segs, keypoints, features, stem: str) -> list:
        fig = plt.figure(figsize=(12.5, 8.8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.15], hspace=0.32, wspace=0.28)
        ax1 = fig.add_subplot(gs[0, :])
        for _, row in segs.iterrows():
            ax1.barh(0, row["duration_sec"], left=row["start_sec"], height=0.6, color=self.palette.get(row["label"], "#999"), linewidth=0)
        ax1.set_yticks([])
        ax1.set_xlabel(t(self.cfg, "time"))
        ax1.set_title("A  " + t(self.cfg, "ethogram"), loc="left")
        ax2 = fig.add_subplot(gs[1, 0])
        times, pcts, _ = _budget(segs, self.max_sec)
        ax2.bar(list(times), [pcts[k] for k in times], color=[self.palette[k] for k in times])
        ax2.set_ylabel(t(self.cfg, "proportion"))
        ax2.set_title("B  " + t(self.cfg, "budget"), loc="left")
        ax3 = fig.add_subplot(gs[1, 1])
        if keypoints is not None:
            com = self._com(keypoints)
            hb = ax3.hist2d(
                com["com_x"].dropna(),
                com["com_y"].dropna(),
                bins=35,
                cmap=self.cfg.get("colormap", "cividis"),
                range=[
                    [self.cage.get("x_min", 0), self.cage.get("x_max", self.cage.get("width", 720))],
                    [self.cage.get("y_min", 0), self.cage.get("y_max", self.cage.get("height", 406))],
                ],
            )
            ax3.set_xlim(self.cage.get("x_min", 0), self.cage.get("x_max", 720))
            ax3.set_ylim(self.cage.get("y_max", 406), self.cage.get("y_min", 0))
            fig.colorbar(hb[3], ax=ax3, fraction=0.046)
        ax3.set_title("C  " + t(self.cfg, "heatmap"), loc="left")
        ax3.set_xlabel(t(self.cfg, "x"))
        ax3.set_ylabel(t(self.cfg, "y"))
        fig.suptitle(t(self.cfg, "publication"), y=0.98, fontsize=13)
        return save_figure(fig, self.output_dir / f"{stem}_publication", self.formats)


def _budget(segs: pd.DataFrame, total_duration_sec: float = 0.0):
    """
    BehaviorStateVisualization.compute_state_stats:
    uncovered gaps within the analysis window are counted as Static.
    """
    order = ["Static", "Walking", "Climbing"]
    times = {
        k: float(segs.loc[segs["label"] == k, "duration_sec"].sum()) if not segs.empty else 0.0
        for k in order
    }
    covered = sum(times.values())
    if not total_duration_sec or total_duration_sec <= 0:
        total_duration_sec = covered
    if total_duration_sec > covered:
        times["Static"] += total_duration_sec - covered
    total = sum(times.values()) or 1.0
    pcts = {k: 100.0 * times[k] / total for k in order}
    return times, pcts, total

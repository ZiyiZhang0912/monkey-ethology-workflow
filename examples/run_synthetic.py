"""用合成关键点验证分类与可视化（不依赖 DLC / 原项目数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monkey_ethology.agent import EthologyAgent
from monkey_ethology.config import load_config


def make_synthetic(n_static=400, n_walk=300, n_climb=250, fps=25, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def append_block(n, x0, y0, vx, vy, p=0.92):
        x, y = x0, y0
        for _ in range(n):
            x += vx + rng.normal(0, 0.4)
            y += vy + rng.normal(0, 0.4)
            head = (x, y - 40)
            back = (x, y)
            tail = (x, y + 40)
            rows.append(
                {
                    "Head_x": head[0] + rng.normal(0, 0.3),
                    "Head_y": head[1] + rng.normal(0, 0.3),
                    "Head_p": p,
                    "Back_x": back[0] + rng.normal(0, 0.3),
                    "Back_y": back[1] + rng.normal(0, 0.3),
                    "Back_p": p,
                    "Tail(Root)_x": tail[0] + rng.normal(0, 0.3),
                    "Tail(Root)_y": tail[1] + rng.normal(0, 0.3),
                    "Tail(Root)_p": p,
                }
            )

    append_block(n_static, 200, 300, 0.02, 0.0, p=0.95)
    append_block(n_walk, 220, 310, 2.4, 0.05, p=0.9)
    append_block(n_climb, 500, 280, 0.05, -2.2, p=0.88)
    append_block(n_static // 2, 520, 90, 0.01, 0.0, p=0.93)
    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    return df


def main():
    out = ROOT / "output" / "synthetic_demo"
    cfg = load_config()
    cfg.set("io.output_dir", str(out))
    cfg.set("cage.fps", 25)
    cfg.set("cage.width", 720)
    cfg.set("cage.height", 406)
    cfg.set("cage.climbing_y", 202)
    cfg.set("locomotion.velocity_threshold", 8.0)
    cfg.set("locomotion.displacement_threshold", 3.0)
    cfg.set("locomotion.window_threshold", 3)
    cfg.set("features.window_size", 13)
    cfg.set("visualization.formats", ["png"])
    cfg.set("visualization.dpi", 120)
    cfg.set("stereotypy.enabled", False)
    cfg.set("circling.enabled", False)

    csv_path = out / "sessions" / "demo-day1.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic()
    df.to_csv(csv_path, index=False)

    agent = EthologyAgent(cfg)
    result = agent.run(csv_path, animal_id="default")
    print("合成数据演示完成")
    print(result)


if __name__ == "__main__":
    main()

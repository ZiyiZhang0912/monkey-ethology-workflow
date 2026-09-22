from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monkey_ethology.classify.locomotion import segment_locomotion, segments_to_frame
from monkey_ethology.config import load_config
from monkey_ethology.features.adaptive_com import AdaptiveCOMExtractor
from monkey_ethology.io.schema import flatten_dlc_dataframe, parse_session_name
from monkey_ethology.preprocess.smooth import smooth_keypoints


def test_parse_session_name():
    info = parse_session_name("562-day1_features.csv")
    assert info["animal_id"] == "562"
    assert info["stem"] == "562-day1"


def test_flatten_already_flat():
    df = pd.DataFrame({"Head_x": [1.0], "Head_y": [2.0], "Head_p": [0.9]})
    out = flatten_dlc_dataframe(df)
    assert "Head_x" in out.columns


def test_feature_and_classify_synthetic():
    n = 390
    t = np.arange(n)
    df = pd.DataFrame(
        {
            "Head_x": 100 + t * 2.0,
            "Head_y": 300 + np.sin(t / 8.0),
            "Head_p": 0.95,
            "Back_x": 100 + t * 2.0,
            "Back_y": 340 + np.sin(t / 8.0),
            "Back_p": 0.9,
            "Tail(Root)_x": 100 + t * 2.0,
            "Tail(Root)_y": 380,
            "Tail(Root)_p": 0.9,
        }
    )
    ext = AdaptiveCOMExtractor(["Head", "Back", "Tail(Root)"], window_size=13, fps=25, p_threshold=0.4)
    feat = ext.transform(df)
    assert len(feat) == n // 13
    assert "mean_velocity" in feat.columns
    segs = segments_to_frame(
        segment_locomotion(
            feat,
            velocity_threshold=5,
            displacement_threshold=2,
            window_threshold=2,
            climbing_y=202,
        )
    )
    assert not segs.empty
    assert set(segs["label"]).issubset({"Static", "Walking", "Climbing"})


def test_smooth_keeps_columns():
    df = pd.DataFrame(
        {
            "Head_x": [1, 100, 2, 3, 4],
            "Head_y": [1, 100, 2, 3, 4],
            "Head_p": [0.9, 0.1, 0.9, 0.9, 0.9],
        }
    )
    out = smooth_keypoints(df, ["Head"], window=3, p_threshold=0.5)
    assert len(out) == 5
    assert out["Head_x"].isna().sum() == 0


def test_load_default_config():
    cfg = load_config()
    assert "Head" in cfg.bodyparts
    assert cfg.get("video_annotation.enabled") is True


def test_build_frame_labels_window_expand():
    from monkey_ethology.viz.video_annotate import build_frame_labels

    segs = pd.DataFrame(
        {"start": [0, 2], "end": [1, 2], "pattern": [2, 3], "label": ["Walking", "Climbing"]}
    )
    labels = build_frame_labels(segs, 40, window_size=13)
    assert labels[0] == "Walking"
    assert labels[13] == "Walking"
    assert labels[26] == "Climbing"
    assert labels[27] == "Transform"

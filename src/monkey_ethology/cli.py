from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .workflow import EthologyWorkflow
from .config import PACKAGE_ROOT, load_config
from .dlc.project import DLC_STEPS


def _csv_list(text: str):
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default=None, help="主配置 YAML")
    p.add_argument("--bodyparts-file", type=str, default=None, help="标记点 YAML")
    p.add_argument("--cage", type=str, default=None, help="笼子 YAML 名（如 default）或路径；省略则用主配置 cage 段")
    p.add_argument(
        "--animal",
        type=str,
        default=None,
        help="单个个体 ID（标签）或阈值 YAML；批量时优先用 --animals / study.animals",
    )
    p.add_argument(
        "--animals",
        type=str,
        default=None,
        help="逗号分隔个体列表，覆盖 study.animals；空则自动发现",
    )
    p.add_argument(
        "--days",
        type=str,
        default=None,
        help="逗号分隔天数，如 1,2,3，覆盖 study.days；空则自动发现",
    )
    p.add_argument("--viz-config", type=str, default=None, help="可视化 YAML")
    p.add_argument("--output", type=str, default=None, help="输出目录")
    p.add_argument("--bodyparts", type=str, default=None, help="逗号分隔标记点，例如 Head,Back,Tail(Root)")
    p.add_argument("--plots", type=str, default=None, help="逗号分隔图类型，覆盖 visualization.plots")
    p.add_argument("--language", choices=["zh", "en"], default=None)
    p.add_argument("--formats", type=str, default=None, help="png,pdf,svg")
    p.add_argument("--dpi", type=int, default=None)
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    p = argparse.ArgumentParser(
        prog="ethology",
        description="笼内猕猴 DLC tracking / 行为分类 / 学术可视化 Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("track", parents=[common], help="按 DLC 官方接口创建项目并执行指定步骤")
    t.add_argument("--videos", required=True, help="视频路径，逗号分隔")
    t.add_argument("--steps", default="create_project,set_bodyparts", help="DLC 步骤，逗号分隔")
    t.add_argument("--dlc-config", default=None, help="已有 DLC config.yaml")
    t.add_argument("--dry-run", action="store_true")

    c = sub.add_parser("classify", parents=[common], help="从关键点/DLC 表进行分类")
    c.add_argument("--input", required=True, help="csv/h5 文件或目录")
    c.add_argument("--pattern", default="*.csv")
    c.add_argument("--skip-preprocess", action="store_true")
    c.add_argument("--no-viz", action="store_true")
    c.add_argument("--video-dir", default=None, help="同名原始视频目录，分类后叠加标注")
    c.add_argument("--no-annotate", action="store_true", help="跳过视频标注")

    v = sub.add_parser("visualize", parents=[common], help="对已有 segment/processed 做可视化")
    v.add_argument("--segments", required=True)
    v.add_argument("--keypoints", default=None)
    v.add_argument("--features", default=None)
    v.add_argument("--stereotypy", default=None)
    v.add_argument("--circling", default=None)
    v.add_argument("--stem", default=None)

    a = sub.add_parser(
        "annotate-video",
        parents=[common],
        help="用已有 processed + features-segment 在原始视频上叠加行为标注",
    )
    a.add_argument("--video-dir", required=True, help="原始视频目录（如 562-day1.MP4）")
    a.add_argument(
        "--data-dir",
        default=None,
        help="含 *_processed.csv / *_features-segment.csv 的目录；默认 output/sessions 下各会话",
    )
    a.add_argument("--stem", default=None, help="只处理指定会话，如 562-day1")
    a.add_argument("--max-frames", type=int, default=None, help="仅导出前 N 帧（预览）")
    a.add_argument("--start-frame", type=int, default=0)
    a.add_argument("--show-animal-id", action="store_true", help="Head 标签显示动物 ID")

    r = sub.add_parser("run", parents=[common], help="一键：ingest → 预处理 → 特征 → 分类 → 可视化 → 视频标注")
    r.add_argument("--input", required=True)
    r.add_argument("--pattern", default="*.csv")
    r.add_argument("--skip-preprocess", action="store_true")
    r.add_argument("--video-dir", default=None, help="同名原始视频目录，启用行为叠加标注")
    r.add_argument("--no-annotate", action="store_true", help="跳过视频标注")

    i = sub.add_parser("init-config", help="把默认配置复制到目标目录")
    i.add_argument("--dest", default="./my_ethology_config.yaml")

    return p


def _resolve_named_yaml(kind: str, value: str) -> str:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return str(path)
    bundled = PACKAGE_ROOT / "configs" / kind / f"{value}.yaml"
    if bundled.is_file():
        return str(bundled)
    return value


def _make_workflow(args) -> EthologyWorkflow:
    cage = _resolve_named_yaml("cages", args.cage) if getattr(args, "cage", None) else None
    animal = getattr(args, "animal", None)
    cfg = load_config(
        getattr(args, "config", None),
        bodyparts=getattr(args, "bodyparts_file", None),
        cage=cage,
        animal=animal,
        visualization=getattr(args, "viz_config", None),
    )
    if getattr(args, "animals", None):
        cfg.set("study.animals", _csv_list(args.animals))
    if getattr(args, "days", None):
        cfg.set("study.days", [int(x) for x in _csv_list(args.days)])
    if getattr(args, "output", None):
        cfg.set("io.output_dir", args.output)
        cfg.set("project.working_directory", args.output)
    if getattr(args, "bodyparts", None):
        cfg.set_bodyparts(_csv_list(args.bodyparts))
    if getattr(args, "language", None):
        cfg.set("visualization.language", args.language)
    if getattr(args, "formats", None):
        cfg.set("visualization.formats", _csv_list(args.formats))
    if getattr(args, "dpi", None):
        cfg.set("visualization.dpi", args.dpi)
    if getattr(args, "plots", None):
        cfg.set("visualization.plots", _csv_list(args.plots))
    if getattr(args, "video_dir", None):
        cfg.set("video_annotation.video_dir", args.video_dir)
        cfg.set("video_annotation.enabled", True)
    if getattr(args, "no_annotate", False):
        cfg.set("video_annotation.enabled", False)
    if getattr(args, "show_animal_id", False):
        cfg.set("video_annotation.show_animal_on_head", True)
    if getattr(args, "max_frames", None) is not None:
        cfg.set("video_annotation.max_frames", args.max_frames)
    if args.cmd == "annotate-video" and getattr(args, "start_frame", None) is not None:
        cfg.set("video_annotation.start_frame", int(args.start_frame))
    return EthologyWorkflow(cfg)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "init-config":
        src = PACKAGE_ROOT / "configs" / "default.yaml"
        dest = Path(args.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"已写入 {dest}")
        return 0

    workflow = _make_workflow(args)

    if args.cmd == "track":
        if args.dry_run:
            workflow.config.set("dlc.dry_run", True)
        result = workflow.track(
            videos=_csv_list(args.videos),
            bodyparts=_csv_list(args.bodyparts) if args.bodyparts else None,
            steps=_csv_list(args.steps) or DLC_STEPS[:2],
            config_path=args.dlc_config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.cmd in {"classify", "run"}:
        path = Path(args.input)
        if getattr(args, "no_viz", False):
            workflow.config.set("visualization.plots", [])
        kwargs = dict(
            skip_preprocess=args.skip_preprocess,
            animal_id=args.animal,
            video_dir=getattr(args, "video_dir", None),
            annotate=False if getattr(args, "no_annotate", False) else None,
        )
        if path.is_dir():
            animals = _csv_list(getattr(args, "animals", None))
            days_raw = _csv_list(getattr(args, "days", None))
            days = [int(x) for x in days_raw] if days_raw else None
            results = workflow.run_directory(
                path,
                pattern=args.pattern,
                animals=animals,
                days=days,
                **kwargs,
            )
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        else:
            result = workflow.run(path, **kwargs)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.cmd == "visualize":
        import pandas as pd

        segs = pd.read_csv(args.segments)
        kp = pd.read_csv(args.keypoints) if args.keypoints else None
        feat = pd.read_csv(args.features) if args.features else None
        st = pd.read_csv(args.stereotypy) if args.stereotypy else None
        cr = pd.read_csv(args.circling) if args.circling else None
        stem = args.stem or Path(args.segments).stem.replace("_features-segment", "")
        artifacts = workflow.visualize(stem, segs, kp, feat, st, cr, plots=_csv_list(args.plots))
        print(json.dumps(artifacts, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "annotate-video":
        import pandas as pd

        from .viz.video_annotate import discover_annotation_jobs

        video_dir = Path(args.video_dir)
        if args.data_dir:
            data_root = Path(args.data_dir)
            jobs = discover_annotation_jobs(video_dir, data_root, stem=args.stem)
            # 也支持 sessions/{stem}/ 布局：data_dir 指向 sessions
            if not jobs and data_root.is_dir():
                for sub in sorted(data_root.iterdir()):
                    if not sub.is_dir():
                        continue
                    jobs.extend(discover_annotation_jobs(video_dir, sub, stem=args.stem))
        else:
            sessions_root = workflow.output_dir / "sessions"
            jobs = []
            if sessions_root.is_dir():
                for sub in sorted(sessions_root.iterdir()):
                    if sub.is_dir():
                        jobs.extend(discover_annotation_jobs(video_dir, sub, stem=args.stem))
        written = []
        for stem, video_path, kp_path, seg_path in jobs:
            kp = pd.read_csv(kp_path)
            segs = pd.read_csv(seg_path)
            meta_animal = stem.split("-")[0] if "-" in stem else None
            out = workflow.annotate_video(
                stem,
                kp,
                segs,
                video_path=video_path,
                animal_id=meta_animal,
                max_frames=args.max_frames,
                start_frame=args.start_frame,
            )
            if out:
                written.append(out)
                print(f"已写出 {out}")
            else:
                print(f"跳过 {stem}：未找到视频或标注关闭")
        if not jobs:
            print("没有可处理的 视频+CSV 组合。请确认 --video-dir 与 sessions 数据。")
            return 1
        print(json.dumps({"videos": written}, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

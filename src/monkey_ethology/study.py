"""Study / cohort selection: animals and days are config-driven, not fixed."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .io.schema import parse_session_name


def study_config(cfg: Mapping[str, Any]) -> dict:
    raw = dict(cfg.get("study") or {})
    animals = raw.get("animals") or []
    days = raw.get("days") or []
    return {
        "animals": [str(a) for a in animals],
        "days": [int(d) for d in days],
        "filename_template": str(raw.get("filename_template") or "{animal}-day{day}"),
        "input_pattern": str(raw.get("input_pattern") or "*.csv"),
    }


def session_allowed(
    stem: str,
    study: Mapping[str, Any],
    *,
    animals_override: Optional[Sequence[str]] = None,
    days_override: Optional[Sequence[int]] = None,
) -> bool:
    """
    按 study.animals / study.days 过滤会话。
    对应列表为空（或 override 为 None 且配置为空）表示该维不限制。
    """
    meta = parse_session_name(stem) or {}
    animals = [str(a) for a in (animals_override if animals_override is not None else study.get("animals") or [])]
    days_cfg = days_override if days_override is not None else study.get("days") or []
    days = [int(d) for d in days_cfg]

    aid = str(meta.get("animal_id") or "")
    day_num = meta.get("day_num")

    if animals and aid not in animals:
        return False
    if days:
        if day_num is None or int(day_num) not in days:
            return False
    return True


def expected_stems(study: Mapping[str, Any]) -> list:
    """若 animals 与 days 均已配置，展开为期望会话 stem 列表。"""
    animals = [str(a) for a in (study.get("animals") or [])]
    days = [int(d) for d in (study.get("days") or [])]
    if not animals or not days:
        return []
    tmpl = str(study.get("filename_template") or "{animal}-day{day}")
    return [tmpl.format(animal=a, day=d) for a in animals for d in days]


def group_sessions_by_animal(sessions: Sequence[Mapping[str, Any]]) -> dict:
    """{animal_id: [session_dict, ...]}，按 day_num / stem 排序。"""
    groups: dict = {}
    for item in sessions:
        stem = item.get("stem", "")
        meta = parse_session_name(stem) or {}
        aid = str(meta.get("animal_id") or stem)
        groups.setdefault(aid, []).append(item)

    def _sort_key(it):
        meta = parse_session_name(it.get("stem", "")) or {}
        return (meta.get("day_num") is None, meta.get("day_num") or 0, it.get("stem", ""))

    return {aid: sorted(items, key=_sort_key) for aid, items in groups.items()}

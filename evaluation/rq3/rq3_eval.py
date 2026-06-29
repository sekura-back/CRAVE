from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HOLD_STEPS = 100
DEFAULT_SEED = 460
DEFAULT_N_VAL = 5000
DEFAULT_N_TIGHT = 1000
DEFAULT_N_MASK = 1000
RUNTIME_RULE = "duration + 100"
SAMPLING_POLICY = "uniform_by_region"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.replace(",", "|").split("|") if part.strip()}
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _rule_margin_col(probe: Mapping[str, Any], group: str, rule_id: str) -> str | None:
    rules = probe.get("_rules", {})
    if not isinstance(rules, Mapping):
        return None
    spec = rules.get(group, {}).get(rule_id)
    if isinstance(spec, Mapping):
        value = spec.get("margin_col")
        return str(value) if value else None
    return None


def _first_rule_step(
    probe: Mapping[str, Any],
    *,
    rule_id: str,
    ids_field: str,
    rules_group: str,
) -> int | None:
    rows = list(probe.get("rows", []) or [])
    for index, row in enumerate(rows):
        if rule_id in _ids(row.get(ids_field, [])):
            return index

    margin_col = _rule_margin_col(probe, rules_group, rule_id)
    if margin_col:
        for index, row in enumerate(rows):
            try:
                if float(row[margin_col]) <= 0.0:
                    return index
            except (KeyError, TypeError, ValueError):
                pass
    return None


def _score_fallback_hazard_step(probe: Mapping[str, Any]) -> int | None:
    score = probe.get("score", {})
    if not isinstance(score, Mapping):
        return None
    value = score.get("first_hazard_step")
    return int(value) if value is not None else None


def _score_fallback_alarm_step(probe: Mapping[str, Any], alarm_id: str) -> int | None:
    score = probe.get("score", {})
    if not isinstance(score, Mapping):
        return None
    value = score.get("first_alarm_step")
    if value is not None:
        return int(value)
    if alarm_id in _ids(score.get("prehazard_alarm_rule_ids", [])):
        return -1
    return None


def classify_reach_avoid(
    probe: Mapping[str, Any],
    hazard_id: str,
    alarm_id: str,
) -> dict[str, Any]:
    hazard_step = _first_rule_step(
        probe,
        rule_id=str(hazard_id),
        ids_field="hazard_rule_ids",
        rules_group="hazards",
    )
    if hazard_step is None:
        hazard_step = _score_fallback_hazard_step(probe)

    alarm_step = _first_rule_step(
        probe,
        rule_id=str(alarm_id),
        ids_field="alarm_rule_ids",
        rules_group="alarms",
    )
    if alarm_step is None:
        alarm_step = _score_fallback_alarm_step(probe, str(alarm_id))

    hazard = hazard_step is not None
    alarm_before_hazard = hazard and alarm_step is not None and int(alarm_step) < int(hazard_step)
    tie = hazard and alarm_step is not None and int(alarm_step) == int(hazard_step)
    return {
        "hazard": hazard,
        "alarm": alarm_step is not None,
        "reach_avoid": bool(hazard and not alarm_before_hazard),
        "hazard_step": hazard_step,
        "alarm_step": alarm_step,
        "alarm_before_hazard": bool(alarm_before_hazard),
        "tie": bool(tie),
    }


def _wilson_ci_pct(success: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = success / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [
        round(max(0.0, centre - margin) * 100.0, 2),
        round(min(1.0, centre + margin) * 100.0, 2),
    ]


def _discover_region_paths(root: Path) -> list[Path]:
    stage3_root = root / "results" / "stage3"
    if not stage3_root.exists():
        return []
    return sorted(stage3_root.glob("*/*/coordinated_regions.json"))


def _required_region_fields(region: Mapping[str, Any]) -> None:
    required = [
        "region_id",
        "hazard_id",
        "alarm_id",
        "hazard_driver",
        "masking_variable",
        "driver_duration_range",
        "driver_amp_range",
        "mask_slope_range",
        "bomega_mode",
        "slope_search_method",
    ]
    missing = [field for field in required if field not in region]
    if missing:
        raise ValueError(f"Stage 3 region missing fields: {missing}")


def _load_region_artifacts(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    artifacts = []
    for path in paths:
        payload = _read_json(path)
        if payload.get("runtime_rule") != RUNTIME_RULE:
            raise ValueError(f"Stage 3 runtime_rule must be {RUNTIME_RULE}: {path}")
        for region in payload.get("regions", []):
            _required_region_fields(region)
        artifacts.append({"path": Path(path), "payload": payload})
    return artifacts


def _normalize_name(value: str) -> str:
    name = str(value).strip().lower()
    for prefix in ("simulation_", "ctrl_out_", "ctrl_out_cmd_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _manifest_score(manifest_path: Path, platform: str, regions: Sequence[Mapping[str, Any]]) -> int:
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return -1
    names = {
        _normalize_name(str(point.get("name", "")))
        for point in manifest.get("injection_points", [])
        if str(point.get("name", "")).strip()
    }
    wanted = {
        _normalize_name(str(region["hazard_driver"]))
        for region in regions
    } | {
        _normalize_name(str(region["masking_variable"]))
        for region in regions
    }
    score = sum(10 for item in wanted if item in names)
    haystack = " ".join(
        [
            manifest_path.parent.name,
            str(manifest.get("system_id", "")),
            str(manifest.get("display_name", "")),
        ]
    ).lower()
    if str(platform).lower() in haystack:
        score += 1
    return score


def _resolve_manifest_path(root: Path, platform: str, regions: Sequence[Mapping[str, Any]]) -> Path:
    candidates = sorted((root / "simulators").glob("*/system_manifest.json"))
    if not candidates:
        raise FileNotFoundError(root / "simulators")
    scored = [(path, _manifest_score(path, platform, regions)) for path in candidates]
    best_path, best_score = max(scored, key=lambda item: item[1])
    if best_score < 0:
        raise ValueError(f"could not resolve manifest for platform {platform}")
    return best_path


def _default_executor(manifest_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module = importlib.import_module(f"simulators.{manifest_path.parent.name}.runtime_executor")
    executor = module.create_executor(manifest_path)
    executor.manifest_path = Path(manifest_path)
    return executor


def _simulate_worker(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    manifest_path, task = item
    module = importlib.import_module(f"simulators.{Path(manifest_path).parent.name}.runtime_executor")
    executor = module.create_executor(Path(manifest_path))
    return dict(executor(task))


def _runtime_batch_runner(
    tasks: Sequence[dict[str, Any]],
    *,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    if int(workers) <= 1:
        return [dict(executor(task)) for task in tasks]
    manifest_path = getattr(executor, "manifest_path", None)
    if manifest_path is None:
        return [dict(executor(task)) for task in tasks]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(_simulate_worker, (str(manifest_path), task)): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            results[futures[future]] = dict(future.result())
    return [dict(item) for item in results if item is not None]


def _duration_sample(region: Mapping[str, Any], rng: random.Random) -> int:
    lo, hi = [int(value) for value in region["driver_duration_range"]]
    return rng.randint(min(lo, hi), max(lo, hi))


def _uniform(lo: float, hi: float, rng: random.Random) -> float:
    left, right = min(float(lo), float(hi)), max(float(lo), float(hi))
    if left == right:
        return left
    return rng.uniform(left, right)


def _region_direction(region: Mapping[str, Any]) -> str:
    direction = str(region.get("driver_direction", "")).strip().lower()
    if direction in {"pos", "neg"}:
        return direction
    lo, hi = [float(value) for value in region["driver_amp_range"]]
    return "neg" if hi <= 0.0 else "pos"


def _stage2_platform_name(platform: str) -> str:
    return "tennessee_eastman" if str(platform) == "TE" else str(platform)


def _signed_driver_range(direction: str, lo: float, hi: float) -> list[float]:
    if str(direction) == "neg":
        return [-float(hi), -float(lo)]
    return [float(lo), float(hi)]


def _range_union(left: Sequence[float], right: Sequence[float]) -> list[float]:
    values = [float(value) for value in list(left) + list(right)]
    return [min(values), max(values)]


def _same_float(left: float, right: float, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol


def _same_range(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == 2 and len(right) == 2 and _same_float(left[0], right[0]) and _same_float(left[1], right[1])


def _region_matches_stage2(region: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    duration = int(row["duration"])
    duration_range = region.get("driver_duration_range") or []
    if len(duration_range) != 2 or not int(duration_range[0]) <= duration <= int(duration_range[1]):
        return False
    if str(region.get("hazard_driver", "")).lower() != str(row.get("hazard_driver", "")).lower():
        return False
    if str(region.get("alarm_id", "")) != str(row.get("alarm_id", "")):
        return False
    if str(region.get("hazard_id", "")) != str(row.get("hazard_id", "")):
        return False
    values = region.get("source_driver_amp_range") or region.get("driver_amp_range")
    return bool(values and _same_range([float(values[0]), float(values[1])], row["driver_amp_range"]))


def _stage2_lower_edge(row: Mapping[str, Any]) -> float:
    if row.get("conditional_boundary_amp") is not None:
        return float(row["conditional_boundary_amp"])
    lo, hi = sorted(float(value) for value in row["driver_amp_range"])
    if hi <= 0.0:
        return hi
    return lo


def _region_driver_anchor(region: Mapping[str, Any]) -> float:
    values = region.get("driver_amp_range") or []
    if len(values) != 2:
        raise ValueError(f"Stage 3 region missing driver_amp_range: {region.get('region_id')}")
    return sum(float(value) for value in values) / 2.0


def _stage2_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows: list[dict[str, Any]] = []
    if "points" in payload:
        for point in payload.get("points", []):
            direction = str(point.get("driver_direction", "pos")).strip().lower() or "pos"
            if direction not in {"pos", "neg"}:
                raise ValueError(f"unsupported Stage 2 conditional direction: {direction}")
            magnitude_range = [float(point["driver_amp_range"][0]), float(point["driver_amp_range"][1])]
            stable_range = [float(point["stable_slope_range"][0]), float(point["stable_slope_range"][1])]
            extreme_range = point.get("extreme_slope_range") or stable_range
            row = {
                "stage2_path": Path(path),
                "stage2_boundary_kind": "cond",
                "duration": int(point["duration"]),
                "runtime_steps": int(point["runtime_steps"]),
                "hazard_driver": str(point["hazard_driver"]),
                "hazard_id": str(point["hazard_id"]),
                "alarm_id": str(point["alarm_id"]),
                "driver_direction": direction,
                "driver_amp_range": _signed_driver_range(direction, magnitude_range[0], magnitude_range[1]),
                "mask_slope_range": _range_union(stable_range, extreme_range),
                "extreme_slope_range": extreme_range,
            }
            if point.get("coordinated_extreme_amp") is not None:
                row["conditional_boundary_amp"] = float(point["coordinated_extreme_amp"])
            elif point.get("coordinated_lower_amp") is not None:
                row["conditional_boundary_amp"] = float(point["coordinated_lower_amp"])
            rows.append(row)
        return rows

    for boundary in payload.get("boundaries", []):
        for interval in boundary.get("target_intervals", []):
            signed = _signed_driver_range(str(boundary["direction"]), interval["rate_lo"], interval["rate_hi"])
            rows.append(
                {
                    "stage2_path": Path(path),
                    "stage2_boundary_kind": "base",
                    "duration": int(boundary["duration"]),
                    "runtime_steps": int(boundary["runtime_steps"]),
                    "hazard_driver": str(boundary["hazard_driver"]),
                    "hazard_id": str(interval.get("first_hazard_id") or boundary.get("hazard_id", "")),
                    "alarm_id": str(interval.get("alarm_id") or boundary.get("alarm_id", "")),
                    "driver_amp_range": signed,
                }
            )
    return rows


def _discover_stage2_paths(root: Path, artifacts: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for artifact in artifacts:
        payload = artifact["payload"]
        platform = _stage2_platform_name(str(payload["platform"]))
        driver = str(payload["hazard_driver"]).lower()
        base_dir = root / "results" / "stage2" / platform / driver
        cond = base_dir / "conditional_results.json"
        base = base_dir / "boundary_results.json"
        path = cond if cond.exists() else base
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _stage2_lower_sample(
    sample_record: Mapping[str, Any],
    rng: random.Random,
    abs_eps: float,
) -> dict[str, Any]:
    region = sample_record["region"]
    row = sample_record["stage2_row"]
    sample = _inside_sample(region, rng)
    edge = _stage2_lower_edge(row)
    eps = float(abs_eps)
    sample["duration"] = int(row["duration"])
    direction = str(row.get("driver_direction") or _region_direction(region))
    if direction == "neg":
        sample["driver_amp"] = edge + eps
    else:
        sample["driver_amp"] = edge - eps
    mask_range = row.get("mask_slope_range") or region["mask_slope_range"]
    sample["mask_slope"] = _uniform(mask_range[0], mask_range[1], rng)
    sample["boundary_source"] = "stage2_lower"
    sample["stage2_boundary_kind"] = str(row["stage2_boundary_kind"])
    sample["stage2_boundary_path"] = Path(row["stage2_path"]).as_posix()
    if row.get("conditional_boundary_amp") is not None:
        sample["conditional_boundary_amp"] = float(row["conditional_boundary_amp"])
    elif row.get("extreme_slope_range"):
        lower, upper = sorted(float(value) for value in row["extreme_slope_range"])
        sample["driver_amp"] = _uniform(region["driver_amp_range"][0], region["driver_amp_range"][1], rng)
        sample["mask_slope"] = lower - eps if rng.random() < 0.5 else upper + eps
        sample["boundary_source"] = "stage2_extreme_slope"
    return sample


def _stage2_lower_records(
    artifacts: Sequence[Mapping[str, Any]],
    stage2_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    rows = [row for path in stage2_paths for row in _stage2_rows(path)]
    records: list[dict[str, Any]] = []
    for row in rows:
        edge = _stage2_lower_edge(row)
        nearest_conditional: list[dict[str, Any]] = []
        for artifact in artifacts:
            for region in artifact["payload"].get("regions", []):
                if not region.get("mask_slope_range"):
                    continue
                if _region_matches_stage2(region, row):
                    distance = abs(_region_driver_anchor(region) - edge)
                    record = {
                        "source_path": artifact["path"],
                        "region": region,
                        "stage2_row": row,
                        "stage2_lower_distance": distance,
                    }
                    if row.get("conditional_boundary_amp") is not None or row.get("extreme_slope_range"):
                        nearest_conditional.append(record)
                    elif _same_float(distance, 0.0):
                        records.append(record)
        if nearest_conditional:
            best_distance = min(float(record["stage2_lower_distance"]) for record in nearest_conditional)
            records.extend(
                record
                for record in nearest_conditional
                if _same_float(record["stage2_lower_distance"], best_distance)
            )
    return records


def _inside_sample(region: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    return {
        "duration": _duration_sample(region, rng),
        "driver_amp": _uniform(region["driver_amp_range"][0], region["driver_amp_range"][1], rng),
        "mask_slope": _uniform(region["mask_slope_range"][0], region["mask_slope_range"][1], rng),
    }


def _boundary_sample(
    region: Mapping[str, Any],
    rng: random.Random,
    abs_eps: float,
    *,
    boundary_source: str = "region",
) -> dict[str, Any]:
    sample = _inside_sample(region, rng)
    if boundary_source == "region":
        lo, hi = [float(value) for value in region["driver_amp_range"]]
        sample["driver_amp"] = (
            min(lo, hi) - abs_eps if _region_direction(region) == "pos" else max(lo, hi) + abs_eps
        )
    elif boundary_source == "source_driver":
        values = region.get("source_driver_amp_range")
        if not values or len(values) != 2:
            raise ValueError(f"Stage 3 region missing source_driver_amp_range: {region.get('region_id')}")
        lo, hi = [float(value) for value in values]
        sample["driver_amp"] = (
            min(lo, hi) - abs_eps if _region_direction(region) == "pos" else max(lo, hi) + abs_eps
        )
    elif boundary_source == "extreme_slope":
        values = region.get("extreme_slope_range")
        if not values or len(values) != 2:
            raise ValueError(f"Stage 3 region missing extreme_slope_range: {region.get('region_id')}")
        lo, hi = [float(value) for value in values]
        lower, upper = min(lo, hi), max(lo, hi)
        sample["mask_slope"] = lower - abs_eps if rng.random() < 0.5 else upper + abs_eps
    else:
        raise ValueError(f"unsupported boundary_source: {boundary_source}")
    sample["boundary_source"] = boundary_source
    return sample


def _masking_sample(
    region: Mapping[str, Any],
    rng: random.Random,
    offset_fraction: float,
) -> dict[str, Any]:
    sample = _inside_sample(region, rng)
    lo, hi = [float(value) for value in region["mask_slope_range"]]
    lower, upper = min(lo, hi), max(lo, hi)
    width = max(1e-6, upper - lower)
    offset = max(1e-6, width * float(offset_fraction))
    sample["mask_slope"] = lower - offset if rng.random() < 0.5 else upper + offset
    return sample


def _masking_boundary_sample(
    region: Mapping[str, Any],
    rng: random.Random,
    offset_fraction: float,
) -> dict[str, Any]:
    sample = _inside_sample(region, rng)
    lo, hi = [float(value) for value in region["mask_slope_range"]]
    lower, upper = min(lo, hi), max(lo, hi)
    width = max(1e-6, upper - lower)
    offset = rng.uniform(0.0, width * float(offset_fraction))
    edge = "lower" if rng.random() < 0.5 else "upper"
    inside = rng.random() < 0.5
    if edge == "lower":
        sample["mask_slope"] = lower + offset if inside else lower - offset
    else:
        sample["mask_slope"] = upper - offset if inside else upper + offset
    sample["mask_boundary_edge"] = edge
    sample["mask_boundary_side"] = "inside" if inside else "outside"
    return sample


def _dual_task(region: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    duration = int(sample["duration"])
    return {
        "runtime_args": {"steps": duration + HOLD_STEPS},
        "injection": {
            "points": [
                {
                    "point": str(region["hazard_driver"]),
                    "amp": float(sample["driver_amp"]),
                    "duration": duration,
                    "t_start": 0,
                    "role": "manip",
                },
                {
                    "point": str(region["masking_variable"]),
                    "amp": float(sample["mask_slope"]),
                    "duration": duration,
                    "t_start": 0,
                    "role": "mask",
                },
            ]
        },
    }


def _case_row(
    *,
    metric: str,
    source_path: Path,
    region: Mapping[str, Any],
    sample: Mapping[str, Any],
    replay: Mapping[str, Any],
    metric_success: bool,
) -> dict[str, Any]:
    row = {
        "metric": metric,
        "source_stage3_regions": source_path.as_posix(),
        "region_id": str(region["region_id"]),
        "hazard_id": str(region["hazard_id"]),
        "alarm_id": str(region["alarm_id"]),
        "hazard_driver": str(region["hazard_driver"]),
        "masking_variable": str(region["masking_variable"]),
        "bomega_mode": str(region["bomega_mode"]),
        "slope_search_method": str(region["slope_search_method"]),
        "duration": int(sample["duration"]),
        "driver_amp": float(sample["driver_amp"]),
        "mask_slope": float(sample["mask_slope"]),
        "metric_success": bool(metric_success),
    }
    if sample.get("boundary_source"):
        row["boundary_source"] = str(sample["boundary_source"])
    if sample.get("stage2_boundary_kind"):
        row["stage2_boundary_kind"] = str(sample["stage2_boundary_kind"])
    if sample.get("stage2_boundary_path"):
        row["stage2_boundary_path"] = str(sample["stage2_boundary_path"])
    if sample.get("mask_boundary_edge"):
        row["mask_boundary_edge"] = str(sample["mask_boundary_edge"])
    if sample.get("mask_boundary_side"):
        row["mask_boundary_side"] = str(sample["mask_boundary_side"])
    row.update(replay)
    return row


def _metric_success(metric: str, replay: Mapping[str, Any], sample: Mapping[str, Any] | None = None) -> bool:
    if metric == "region_validity":
        return bool(replay["reach_avoid"])
    if metric == "boundary_tightness":
        return not bool(replay["reach_avoid"])
    if metric == "masking_exclusion":
        if sample and sample.get("mask_boundary_side"):
            expected = str(sample["mask_boundary_side"]) == "inside"
            return bool(replay["reach_avoid"]) is expected
        return not bool(replay["reach_avoid"])
    raise ValueError(f"unsupported metric: {metric}")


def _metric_category(metric: str) -> str:
    return {
        "region_validity": "Cat. I",
        "boundary_tightness": "Cat. II",
        "masking_exclusion": "Cat. III",
    }[metric]


def _failure_reason(row: Mapping[str, Any]) -> str:
    if row.get("reach_avoid"):
        return "still_reach_avoid"
    if not row.get("hazard"):
        return "no_hazard"
    if row.get("alarm_before_hazard"):
        return "alarm_before_hazard"
    return "other_no_reach_avoid"


def _summarize_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    success = sum(1 for row in rows if row["metric_success"])
    return {
        "n": n,
        "success": success,
        "failure": n - success,
        "rate": round(success / n, 6) if n else 0.0,
        "rate_pct": round(success / n * 100.0, 2) if n else 0.0,
        "ci95_pct": _wilson_ci_pct(success, n),
    }


def _summaries_by_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_metric: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_metric[str(row["metric"])].append(row)
    return {
        metric: _summarize_metric(by_metric.get(metric, []))
        for metric in ("region_validity", "boundary_tightness", "masking_exclusion")
    }


def _sample_cases(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    n_val: int,
    n_tight: int,
    n_mask: int,
    seed: int,
    boundary_abs_eps: float,
    boundary_source: str,
    mask_offset_fraction: float,
    mask_sampling_mode: str = "outside",
    stage2_paths: Sequence[str | Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    regions = [
        {"source_path": artifact["path"], "region": region}
        for artifact in artifacts
        for region in artifact["payload"].get("regions", [])
        if region.get("mask_slope_range")
    ]
    if not regions:
        return {"region_validity": [], "boundary_tightness": [], "masking_exclusion": []}
    stage2_records = _stage2_lower_records(artifacts, stage2_paths or []) if boundary_source == "stage2_lower" else []
    if boundary_source == "stage2_lower" and not stage2_records:
        raise ValueError("boundary_source=stage2_lower requires matching Stage 2 lower-bound rows")

    def choose() -> dict[str, Any]:
        return rng.choice(regions)

    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _ in range(int(n_val)):
        item = choose()
        samples["region_validity"].append({**item, "sample": _inside_sample(item["region"], rng)})
    for _ in range(int(n_tight)):
        if boundary_source == "stage2_lower":
            item = rng.choice(stage2_records)
            samples["boundary_tightness"].append(
                {**item, "sample": _stage2_lower_sample(item, rng, boundary_abs_eps)}
            )
        else:
            item = choose()
            samples["boundary_tightness"].append(
                {
                    **item,
                    "sample": _boundary_sample(
                        item["region"],
                        rng,
                        boundary_abs_eps,
                        boundary_source=boundary_source,
                    ),
                }
            )
    for _ in range(int(n_mask)):
        item = choose()
        if mask_sampling_mode == "boundary_accuracy":
            sample = _masking_boundary_sample(item["region"], rng, mask_offset_fraction)
        else:
            sample = _masking_sample(item["region"], rng, mask_offset_fraction)
        samples["masking_exclusion"].append(
            {**item, "sample": sample}
        )
    return dict(samples)


def _run_platform_cases(
    *,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    samples_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    workers: int = 1,
    batch_runner: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    work_items: list[tuple[str, Mapping[str, Any], Mapping[str, Any], dict[str, Any]]] = []
    tasks: list[dict[str, Any]] = []
    for metric, samples in samples_by_metric.items():
        for sample_record in samples:
            region = sample_record["region"]
            task = _dual_task(region, sample_record["sample"])
            work_items.append((metric, sample_record, region, task))
            tasks.append(task)

    if batch_runner:
        probes = [dict(item) for item in batch_runner(tasks, executor=executor, workers=int(workers))]
    else:
        probes = []
        for task in tasks:
            probes.append(dict(executor(task)))

    rows: list[dict[str, Any]] = []
    for (metric, sample_record, region, _task), probe in zip(work_items, probes):
        if hasattr(executor, "rules"):
            probe["_rules"] = getattr(executor, "rules")
        replay = classify_reach_avoid(probe, str(region["hazard_id"]), str(region["alarm_id"]))
        ok = _metric_success(metric, replay, sample_record["sample"])
        rows.append(
            _case_row(
                metric=metric,
                source_path=Path(sample_record["source_path"]),
                region=region,
                sample=sample_record["sample"],
                replay=replay,
                metric_success=ok,
            )
        )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _build_failure_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["metric_success"]:
            continue
        item = dict(row)
        item["category"] = _metric_category(str(row["metric"]))
        item["reason"] = _failure_reason(row)
        failures.append(item)
    counts: dict[str, int] = defaultdict(int)
    for item in failures:
        counts[str(item["category"])] += 1
    return {"total": len(failures), "counts_by_category": dict(sorted(counts.items())), "cases": failures}


def _group_artifacts_by_platform(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        platform = str(artifact["payload"].get("platform", "")).strip()
        if not platform:
            raise ValueError(f"Stage 3 artifact missing platform: {artifact['path']}")
        grouped[platform].append(artifact)
    return dict(grouped)


def _region_count(artifacts: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(artifact["payload"].get("regions", [])) for artifact in artifacts)


def _executor_for_platform(
    *,
    root: Path,
    platform: str,
    artifacts: Sequence[Mapping[str, Any]],
    executor_factory_map: Mapping[str, Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]]] | None,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Path]:
    regions = [region for artifact in artifacts for region in artifact["payload"].get("regions", [])]
    manifest_path = _resolve_manifest_path(root, platform, regions)
    if executor_factory_map:
        for key in (platform, platform.lower(), platform.upper()):
            factory = executor_factory_map.get(key)
            if factory:
                return factory(manifest_path), manifest_path
    return _default_executor(manifest_path), manifest_path


def _source_paths(root: Path, artifacts: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = []
    for artifact in artifacts:
        path = Path(artifact["path"])
        try:
            paths.append(path.relative_to(root).as_posix())
        except ValueError:
            paths.append(path.as_posix())
    return paths


def _build_region_type_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("platform", "")), str(row.get("bomega_mode", "")))].append(row)
    entries = []
    for (platform, mode), group_rows in sorted(groups.items()):
        region_ids = {str(row["region_id"]) for row in group_rows}
        entries.append(
            {
                "platform": platform,
                "region_type": mode,
                "region_count_sampled": len(region_ids),
                **_summaries_by_metric(group_rows),
            }
        )
    return {"entries": entries}


def run_rq3_evaluation(
    *,
    root: str | Path,
    region_paths: Sequence[str | Path] | None = None,
    n_val: int = DEFAULT_N_VAL,
    n_tight: int = DEFAULT_N_TIGHT,
    n_mask: int = DEFAULT_N_MASK,
    seed: int = DEFAULT_SEED,
    boundary_abs_eps: float = 0.01,
    boundary_source: str = "region",
    mask_offset_fraction: float = 0.05,
    mask_sampling_mode: str = "outside",
    workers: int = 1,
    stage2_paths: Sequence[str | Path] | None = None,
    executor_factory_map: Mapping[str, Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]]] | None = None,
    batch_runner: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    paths = [Path(path) for path in region_paths] if region_paths else _discover_region_paths(root)
    if not paths:
        raise FileNotFoundError(root / "results" / "stage3")
    artifacts = _load_region_artifacts(paths)
    resolved_stage2_paths = [Path(path) for path in stage2_paths] if stage2_paths else []
    grouped = _group_artifacts_by_platform(artifacts)
    rq3_dir = root / "results" / "rq3"

    all_rows: list[dict[str, Any]] = []
    platform_summaries = []
    manifest_paths: dict[str, str] = {}
    for platform, platform_artifacts in sorted(grouped.items()):
        executor, manifest_path = _executor_for_platform(
            root=root,
            platform=platform,
            artifacts=platform_artifacts,
            executor_factory_map=executor_factory_map,
        )
        manifest_paths[platform] = manifest_path.as_posix()
        samples = _sample_cases(
            artifacts=platform_artifacts,
            n_val=n_val,
            n_tight=n_tight,
            n_mask=n_mask,
            seed=seed,
            boundary_abs_eps=boundary_abs_eps,
            boundary_source=boundary_source,
            mask_offset_fraction=mask_offset_fraction,
            mask_sampling_mode=mask_sampling_mode,
            stage2_paths=resolved_stage2_paths or _discover_stage2_paths(root, platform_artifacts),
        )
        rows = _run_platform_cases(
            executor=executor,
            samples_by_metric=samples,
            workers=workers,
            batch_runner=batch_runner or _runtime_batch_runner,
        )
        for row in rows:
            row["platform"] = platform
        all_rows.extend(rows)
        platform_summaries.append(
            {
                "platform": platform,
                "region_count": _region_count(platform_artifacts),
                "source_stage3_regions": _source_paths(root, platform_artifacts),
                **_summaries_by_metric(rows),
            }
        )

    config = {
        "runtime_rule": RUNTIME_RULE,
        "tie_breaking": "same_step_alarm_and_hazard_counts_as_reach_avoid",
        "sampling_policy": SAMPLING_POLICY,
        "seed": int(seed),
        "n_val_per_platform": int(n_val),
        "n_tight_per_platform": int(n_tight),
        "n_mask_per_platform": int(n_mask),
        "boundary_abs_eps": float(boundary_abs_eps),
        "boundary_source": str(boundary_source),
        "mask_offset_fraction": float(mask_offset_fraction),
        "mask_sampling_mode": str(mask_sampling_mode),
        "workers": int(workers),
        "stage2_paths": [path.as_posix() for path in resolved_stage2_paths],
        "source_stage3_regions": _source_paths(root, artifacts),
        "manifest_paths": manifest_paths,
    }
    quality_summary = {
        "platforms": platform_summaries,
        "overall": _summaries_by_metric(all_rows),
    }
    failures = _build_failure_cases(all_rows)
    breakdown = _build_region_type_breakdown(all_rows)

    _write_json(rq3_dir / "rq3_eval_config.json", config)
    _write_json(rq3_dir / "rq3_quality_summary.json", quality_summary)
    _write_json(rq3_dir / "rq3_failure_cases.json", failures)
    _write_json(rq3_dir / "rq3_region_type_breakdown.json", breakdown)
    _write_jsonl(rq3_dir / "rq3_case_samples.jsonl", all_rows)
    _write_json(root / "results" / "manifests" / "rq3" / "run_manifest.json", config)
    _write_jsonl(
        root / "results" / "logs" / "rq3" / "events.jsonl",
        [{"stage": "rq3", "event": "write_summary", "case_count": len(all_rows), "seed": int(seed)}],
    )
    return {
        "config": config,
        "quality_summary": quality_summary,
        "failure_cases": failures,
        "region_type_breakdown": breakdown,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ3 closed-loop region replay.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--region-file", action="append", default=[])
    parser.add_argument("--stage2-file", action="append", default=[])
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    parser.add_argument("--n-tight", type=int, default=DEFAULT_N_TIGHT)
    parser.add_argument("--n-mask", type=int, default=DEFAULT_N_MASK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--boundary-abs-eps", type=float, default=0.01)
    parser.add_argument(
        "--boundary-source",
        choices=("region", "source_driver", "extreme_slope", "stage2_lower"),
        default="region",
    )
    parser.add_argument("--mask-offset-fraction", type=float, default=0.05)
    parser.add_argument("--mask-sampling-mode", choices=("outside", "boundary_accuracy"), default="outside")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_rq3_evaluation(
        root=args.root,
        region_paths=args.region_file or None,
        n_val=args.n_val,
        n_tight=args.n_tight,
        n_mask=args.n_mask,
        seed=args.seed,
        boundary_abs_eps=args.boundary_abs_eps,
        boundary_source=args.boundary_source,
        mask_offset_fraction=args.mask_offset_fraction,
        mask_sampling_mode=args.mask_sampling_mode,
        workers=args.workers,
        stage2_paths=args.stage2_file or None,
    )


if __name__ == "__main__":
    main()

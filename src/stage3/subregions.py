from __future__ import annotations

import argparse
import importlib
import json
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


HOLD_STEPS = 100
RUNTIME_RULE = "duration + 100"
REPRESENTATIVE_SAMPLES = 10
SAMPLE_DURATION_MOD = 50
REGION_DELTA_T = 50
REGION_DELTA_K = 0.01
REGION_GAMMA_MERGE = 0.10
DYNAMIC_SLOPE_TOL = 1e-5
DYNAMIC_GRID_SAMPLES = 10


def main(
    argv: Sequence[str] | None = None,
    *,
    create_executor: Callable[[Path], Callable[[Mapping[str, Any]], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    args = _parse_args(argv)
    base = _read_json(args.base_path)
    conditional = _read_json(args.conditional_path) if args.conditional_path else None
    extraction = _read_json(args.extraction_path)
    manifest = _read_json(args.manifest_path)
    executor = (create_executor or _load_executor)(args.manifest_path)

    points = _build_points(
        base,
        conditional,
        extraction,
        manifest,
        str(args.hazard_id),
        str(args.slope_search),
        int(args.representative_samples),
        int(args.sample_duration_mod),
    )
    process_manifest = args.manifest_path if create_executor is None else None
    raw_regions = _solve_regions(executor, points, args.workers, process_manifest)
    regions = _merge_regions(raw_regions, args.d_threshold)
    payload = _build_payload(args, regions)
    representative_payload = _build_payload(args, raw_regions)
    representative_payload["artifact"] = "representative_regions"
    _write_outputs(args.output_root, args.platform, args.hazard_driver, payload, representative_payload)
    return payload


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 3 alarm-masking subregions")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--hazard-id", required=True)
    parser.add_argument("--hazard-driver", required=True)
    parser.add_argument("--base-path", required=True, type=Path)
    parser.add_argument("--conditional-path", type=Path)
    parser.add_argument("--extraction-path", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--d-threshold", type=float, default=0.0)
    parser.add_argument("--slope-search", choices=["auto", "dynamic"], default="auto")
    parser.add_argument("--representative-samples", type=int, default=REPRESENTATIVE_SAMPLES)
    parser.add_argument("--sample-duration-mod", type=int, default=SAMPLE_DURATION_MOD)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_executor(manifest_path: Path) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    module_name = f"simulators.{Path(manifest_path).parent.name}.runtime_executor"
    module = importlib.import_module(module_name)
    return module.create_executor(Path(manifest_path))


def _build_points(
    base: Mapping[str, Any],
    conditional: Mapping[str, Any] | None,
    extraction: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fallback_hazard_id: str,
    slope_search: str,
    representative_samples: int,
    sample_duration_mod: int,
) -> list[dict[str, Any]]:
    cond_rows = _conditional_rows(conditional)
    points: list[dict[str, Any]] = []
    platform = str(base["platform"])
    base_hazard_id = str(base.get("hazard_id") or "").strip()
    base_hazard_driver = str(base.get("hazard_driver") or "").strip()
    if base_hazard_id:
        _validate_rule_id(base_hazard_id, "H")
    _validate_rule_id(fallback_hazard_id, "H")

    for boundary_index, boundary in enumerate(base.get("boundaries", [])):
        duration = int(boundary["duration"])
        runtime_steps = int(boundary["runtime_steps"])
        if runtime_steps != duration + HOLD_STEPS:
            raise ValueError("runtime_steps must equal duration + 100")
        intervals = boundary.get("target_intervals", [])
        if not intervals:
            continue
        if duration % int(sample_duration_mod) != 0:
            continue
        boundary_hazard_id = str(boundary.get("hazard_id") or base_hazard_id).strip()
        hazard_driver = str(boundary.get("hazard_driver") or base_hazard_driver).strip()
        boundary_alarm_id = str(boundary.get("alarm_id") or "").strip()
        if not hazard_driver:
            raise ValueError("target boundary missing hazard_driver")
        if boundary_hazard_id:
            _validate_rule_id(boundary_hazard_id, "H")
        direction = str(boundary["direction"])
        for interval_index, interval in enumerate(intervals):
            alarm_id = str(interval.get("alarm_id") or boundary_alarm_id).strip()
            if not alarm_id:
                raise ValueError("target interval missing alarm_id")
            _validate_rule_id(alarm_id, "A")
            hazard_ids = _hazard_ids_for_interval(base_hazard_id, boundary, interval)
            if not hazard_ids:
                hazard_ids = [fallback_hazard_id]
            hazard_id = hazard_ids[0]
            alarm_rule = _alarm_rule(extraction, alarm_id)
            masking_variable = str(alarm_rule["setpoint_var"])
            magnitude_range = [float(interval["rate_lo"]), float(interval["rate_hi"])]
            driver_range = _signed_range(direction, magnitude_range)
            cond = _match_conditional_row(
                cond_rows,
                duration,
                runtime_steps,
                magnitude_range,
                alarm_id,
                hazard_driver,
            )
            use_dynamic = slope_search == "dynamic" or cond is not None
            for sample_index, (sample_q, driver_amp) in enumerate(
                _sample_driver_amps(driver_range, int(representative_samples))
            ):
                points.append(
                    {
                        "point_id": f"{hazard_driver}_{boundary_index}_{interval_index}_{sample_index}",
                        "platform": platform,
                        "hazard_id": hazard_id,
                        "hazard_ids": hazard_ids,
                        "alarm_id": alarm_id,
                        "hazard_driver": hazard_driver,
                        "driver_direction": direction,
                        "masking_variable": masking_variable,
                        "alarm_var": str(alarm_rule["var"]),
                        "alarm_setpoint_var": str(alarm_rule["setpoint_var"]),
                        "alarm_threshold": float(alarm_rule["threshold"]),
                        "duration": duration,
                        "runtime_steps": runtime_steps,
                        "driver_amp_range": driver_range,
                        "driver_amp": driver_amp,
                        "sample_q": sample_q,
                        "k_int": _bucket_key(driver_amp),
                        "conditional": cond,
                        "bomega_mode": "coordinated" if use_dynamic else "static",
                        "slope_search_method": "dynamic" if use_dynamic else "static",
                    }
                )
    return points


def _conditional_rows(conditional: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not conditional:
        return []
    if "expanded_boundary" in conditional:
        raise ValueError("expanded_boundary[] is not a Stage 3 conditional input contract")
    rows: list[dict[str, Any]] = []
    for row in conditional.get("points", []):
        driver_range = [float(row["driver_amp_range"][0]), float(row["driver_amp_range"][1])]
        rows.append(
            {
                "point_id": str(row["point_id"]),
                "duration": int(row["duration"]),
                "runtime_steps": int(row["runtime_steps"]),
                "hazard_driver": str(row["hazard_driver"]),
                "hazard_id": str(row["hazard_id"]),
                "alarm_id": str(row["alarm_id"]),
                "driver_amp_range": driver_range,
                "driver_amp": sum(driver_range) / 2.0,
                "stable_slope_range": list(row["stable_slope_range"]),
                "extreme_slope_range": list(row["extreme_slope_range"]),
            }
        )
    return rows


def _match_conditional_row(
    rows: Sequence[Mapping[str, Any]],
    duration: int,
    runtime_steps: int,
    driver_range: Sequence[float],
    alarm_id: str,
    hazard_driver: str,
) -> Mapping[str, Any] | None:
    matches: list[Mapping[str, Any]] = []
    for row in rows:
        if int(row["duration"]) != duration or row["alarm_id"] != alarm_id:
            continue
        if int(row["runtime_steps"]) != runtime_steps:
            raise ValueError("conditional runtime_steps must match base runtime_steps")
        if row["hazard_driver"] != hazard_driver:
            continue
        if [float(row["driver_amp_range"][0]), float(row["driver_amp_range"][1])] == [
            float(driver_range[0]),
            float(driver_range[1]),
        ]:
            matches.append(row)
    if len(matches) > 1:
        ranges = {
            (
                tuple(float(value) for value in match["stable_slope_range"]),
                tuple(float(value) for value in match["extreme_slope_range"]),
            )
            for match in matches
        }
        if len(ranges) > 1:
            raise ValueError("ambiguous conditional rows after ignoring hazard_id")
    return dict(matches[0]) if matches else None


def _alarm_rule(extraction: Mapping[str, Any], alarm_id: str) -> dict[str, Any]:
    for rule in extraction.get("P", []):
        if rule.get("id") == alarm_id:
            var = str(rule.get("var", "")).strip()
            setpoint = str(rule.get("setpoint_var", "")).strip()
            threshold = rule.get("threshold")
            if not var or not setpoint or threshold is None:
                raise ValueError(f"alarm rule missing var/setpoint_var/threshold: {alarm_id}")
            return {"var": var, "setpoint_var": setpoint, "threshold": float(threshold)}
    raise KeyError(alarm_id)


def _validate_rule_id(rule_id: str, prefix: str) -> None:
    if not rule_id.startswith(f"{prefix}-"):
        raise ValueError(f"rule id must start with {prefix}-: {rule_id}")


def _hazard_ids_for_interval(
    base_hazard_id: str,
    boundary: Mapping[str, Any],
    interval: Mapping[str, Any],
) -> list[str]:
    ids: set[str] = set()
    for value in (
        interval.get("first_hazard_id"),
        boundary.get("hazard_id"),
        base_hazard_id,
    ):
        if value:
            ids.add(str(value).strip())
    first_hazard_ids = boundary.get("first_hazard_ids", [])
    if isinstance(first_hazard_ids, str):
        ids.update(part.strip() for part in first_hazard_ids.replace(",", "|").split("|") if part.strip())
    else:
        ids.update(str(value).strip() for value in first_hazard_ids if str(value).strip())
    for rule_id in ids:
        _validate_rule_id(rule_id, "H")
    return sorted(ids)


def _sample_driver_amps(driver_range: Sequence[float], representative_samples: int = REPRESENTATIVE_SAMPLES) -> list[tuple[float, float]]:
    lo, hi = float(driver_range[0]), float(driver_range[1])
    samples = max(1, int(representative_samples))
    if samples <= 1:
        return [(0.0, lo)]
    return [
        (index / (samples - 1), lo + index * (hi - lo) / (samples - 1))
        for index in range(samples)
    ]


def _signed_range(direction: str, magnitude_range: Sequence[float]) -> list[float]:
    lo, hi = float(magnitude_range[0]), float(magnitude_range[1])
    if direction == "pos":
        return [lo, hi]
    if direction == "neg":
        return [-hi, -lo]
    raise ValueError(f"unsupported direction: {direction}")


def _bucket_key(amp: float) -> int:
    return int(round(round(float(amp) / REGION_DELTA_K) * REGION_DELTA_K * 100))


def _solve_regions(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    points: Sequence[Mapping[str, Any]],
    workers: int,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    if int(workers) <= 1 or len(points) <= 1 or manifest_path is None:
        return [_solve_region(executor, point) for point in points]
    tasks = [(str(manifest_path), dict(point)) for point in points]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(_solve_region_process, tasks))


def _solve_region_process(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    manifest_path, point = item
    executor = _load_executor(Path(manifest_path))
    return _solve_region(executor, point)


def _solve_region(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
) -> dict[str, Any]:
    if point["slope_search_method"] == "dynamic":
        cond = point["conditional"]
        if cond:
            search_bounds = _registry_slope_bounds(executor, point)
            stable = _scan_mask_slope(executor, point, cond["stable_slope_range"], search_bounds)
            if _same_range(cond["stable_slope_range"], cond["extreme_slope_range"]):
                extreme = list(stable)
            else:
                extreme = _scan_mask_slope(executor, point, cond["extreme_slope_range"], search_bounds)
        else:
            search_bounds = _registry_slope_bounds(executor, point)
            seed_result = executor(_single_task(point, executor))
            hazard_step = seed_result.get("score", {}).get("first_hazard_step")
            if hazard_step is None or int(hazard_step) <= 1:
                stable = []
            else:
                seed_range = _static_slope_range(seed_result, point, executor)
                if seed_range:
                    stable = _scan_mask_slope(executor, point, seed_range, search_bounds)
                else:
                    stable = _scan_mask_slope(executor, point, search_bounds)
            extreme = list(stable)
    else:
        result = executor(_single_task(point, executor))
        stable = _static_slope_range(result, point, executor)
        extreme = list(stable)

    mask_range = _union_ranges(stable, extreme)
    return {
        "region_id": str(point["point_id"]),
        "hazard_id": str(point["hazard_id"]),
        "hazard_ids": list(point["hazard_ids"]),
        "alarm_id": str(point["alarm_id"]),
        "hazard_driver": str(point["hazard_driver"]),
        "driver_direction": str(point["driver_direction"]),
        "masking_variable": str(point["masking_variable"]),
        "driver_duration_range": [int(point["duration"]), int(point["duration"])],
        "driver_amp": float(point["driver_amp"]),
        "driver_amp_range": [float(point["driver_amp"]), float(point["driver_amp"])],
        "source_driver_amp_range": list(point["driver_amp_range"]),
        "sample_q": float(point["sample_q"]),
        "k_int": int(point["k_int"]),
        "stable_slope_range": stable,
        "extreme_slope_range": extreme,
        "mask_slope_range": mask_range,
        "bomega_mode": str(point["bomega_mode"]),
        "slope_search_method": str(point["slope_search_method"]),
        "runtime_steps": int(point["runtime_steps"]),
        "n_members": 1,
    }


def _scan_mask_slope(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
    bounds: Sequence[float],
    search_bounds: Sequence[float] | None = None,
) -> list[float]:
    seed_lower, seed_upper = _checked_slope_bounds(bounds)
    search_lower, search_upper = _checked_slope_bounds(search_bounds or bounds)
    lower = min(seed_lower, search_lower)
    upper = max(seed_upper, search_upper)

    seed = _find_mask_seed(executor, point, [seed_lower, seed_upper])
    if seed is None and (lower != seed_lower or upper != seed_upper):
        seed = _find_mask_seed(executor, point, [lower, upper])
    if seed is None:
        return []

    left = _expand_mask_edge(executor, point, lower, seed)
    right = _expand_mask_edge(executor, point, upper, seed)
    return [left, right] if left <= right else [right, left]


def _checked_slope_bounds(bounds: Sequence[float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if upper < lower:
        raise ValueError(f"invalid slope bounds: {bounds}")
    return lower, upper


def _find_mask_seed(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
    bounds: Sequence[float],
) -> float | None:
    lower, upper = float(bounds[0]), float(bounds[1])
    for slope in _seed_slope_candidates(lower, upper):
        if _mask_slope_success(executor, point, slope):
            return slope
    return None


def _seed_slope_candidates(lower: float, upper: float) -> list[float]:
    if upper == lower:
        return [lower]
    values = [(lower + upper) / 2.0]
    values.extend(
        lower + index * (upper - lower) / (DYNAMIC_GRID_SAMPLES - 1)
        for index in range(DYNAMIC_GRID_SAMPLES)
    )
    ordered: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(value, 12)
        if key not in seen:
            ordered.append(value)
            seen.add(key)
    return ordered


def _expand_mask_edge(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
    bound: float,
    pass_slope: float,
) -> float:
    if bound == pass_slope:
        return pass_slope
    if _mask_slope_success(executor, point, bound):
        return bound
    return _refine_mask_edge(executor, point, bound, pass_slope)


def _registry_slope_bounds(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
) -> list[float]:
    rate = abs(float(_injection_spec(executor, str(point["masking_variable"]))["rate"]))
    return [-rate, rate]


def _mask_slope_success(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
    slope: float,
) -> bool:
    return _reach_avoid_success(executor(_dual_task(point, slope)))


def _refine_mask_edge(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point: Mapping[str, Any],
    fail_slope: float,
    pass_slope: float,
) -> float:
    fail, passed = float(fail_slope), float(pass_slope)
    while abs(passed - fail) > DYNAMIC_SLOPE_TOL:
        midpoint = (passed + fail) / 2.0
        if _mask_slope_success(executor, point, midpoint):
            passed = midpoint
        else:
            fail = midpoint
    return passed


def _stop_ignore_predicates(executor: Callable[[Mapping[str, Any]], dict[str, Any]]) -> list[str]:
    rules = getattr(executor, "rules", {})
    alarms = rules.get("alarms", {}) if isinstance(rules, Mapping) else {}
    predicates: set[str] = set()
    for alarm in alarms.values():
        expr = str(alarm.get("expr", "")) if isinstance(alarm, Mapping) else ""
        predicates.update(re.findall(r"\b[A-Z][A-Z0-9_]*_TH\b", expr))
    return sorted(predicates)


def _single_task(
    point: Mapping[str, Any],
    executor: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    duration = int(point["duration"])
    runtime_args: dict[str, Any] = {"steps": duration + HOLD_STEPS, "stop_on_trip": True}
    if executor is not None:
        ignore = _stop_ignore_predicates(executor)
        if ignore:
            runtime_args["ignore_protection_predicates_for_stop"] = ignore
    return {
        "runtime_args": runtime_args,
        "injection": {
            "point": str(point["hazard_driver"]),
            "amp": float(point["driver_amp"]),
            "duration": duration,
            "t_start": 0,
            "role": "manip",
        },
    }


def _dual_task(point: Mapping[str, Any], slope: float) -> dict[str, Any]:
    duration = int(point["duration"])
    return {
        "runtime_args": {"steps": duration + HOLD_STEPS, "stop_on_trip": True},
        "injection": {
            "points": [
                {
                    "point": str(point["hazard_driver"]),
                    "amp": float(point["driver_amp"]),
                    "duration": duration,
                    "t_start": 0,
                    "role": "manip",
                },
                {
                    "point": str(point["masking_variable"]),
                    "amp": float(slope),
                    "duration": duration,
                    "t_start": 0,
                    "role": "mask",
                },
            ]
        },
    }


def _static_slope_range(
    result: Mapping[str, Any],
    point: Mapping[str, Any],
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[float]:
    if result.get("score", {}).get("first_hazard_step") is None:
        return []
    score = result.get("score", {})
    hazard_step = int(score.get("first_hazard_step") or 0)
    if hazard_step <= 1:
        return []
    alarm_var = str(point["alarm_var"])
    setpoint_var = str(point["alarm_setpoint_var"])
    threshold = float(point["alarm_threshold"])
    spec = _injection_spec(executor, str(point["masking_variable"]))
    value_range = spec.get("range")
    if not value_range or len(value_range) != 2:
        raise KeyError(f"{point['masking_variable']}.range")
    rate = float(spec["rate"])
    lower = -abs(rate)
    upper = abs(rate)
    rows = list(result.get("rows", []))
    if len(rows) < hazard_step:
        raise ValueError("missing pre-hazard trace rows")
    if setpoint_var not in rows[0]:
        raise ValueError(f"missing alarm setpoint trace column: {setpoint_var}")
    baseline = float(rows[0][setpoint_var])

    candidates = {lower, upper}
    for index, row in enumerate(rows[:hazard_step]):
        if alarm_var not in row:
            raise ValueError(f"missing alarm trace column: {alarm_var}")
        if setpoint_var not in row:
            raise ValueError(f"missing alarm setpoint trace column: {setpoint_var}")
        if index <= 0:
            continue
        eff = min(index, max(1, int(point["duration"]) - 1))
        measured = float(row[alarm_var])
        for value in (
            float(value_range[0]),
            float(value_range[1]),
            measured - threshold,
            measured + threshold,
        ):
            slope = (value - baseline) / eff
            if lower <= slope <= upper:
                candidates.add(slope)

    accepted: list[float] = []
    ordered = sorted(candidates)
    for left, right in zip(ordered, ordered[1:]):
        if _static_slope_feasible((left + right) / 2.0, rows[:hazard_step], point, baseline, value_range):
            accepted.extend([left, right])
    accepted.extend(
        slope
        for slope in ordered
        if _static_slope_feasible(slope, rows[:hazard_step], point, baseline, value_range)
    )
    return [min(accepted), max(accepted)] if accepted else []


def _static_slope_feasible(
    slope: float,
    rows: Sequence[Mapping[str, Any]],
    point: Mapping[str, Any],
    baseline: float,
    value_range: Sequence[float],
) -> bool:
    alarm_var = str(point["alarm_var"])
    threshold = float(point["alarm_threshold"])
    duration = max(2, int(point["duration"]))
    for index, row in enumerate(rows):
        eff = min(index, duration - 1)
        spoofed = _clamp(float(baseline) + float(slope) * eff, value_range)
        if abs(spoofed - float(row[alarm_var])) > threshold + 1e-12:
            return False
    return True


def _reach_avoid_success(result: Mapping[str, Any]) -> bool:
    score = result.get("score", {})
    hazard_step = score.get("first_hazard_step")
    if hazard_step is None:
        return False
    if "first_alarm_step" in score:
        alarm_step = score.get("first_alarm_step")
        return alarm_step is None or int(alarm_step) >= int(hazard_step)
    return not score.get("prehazard_alarm_rule_ids")


def _clamp(value: float, value_range: Sequence[float]) -> float:
    lower, upper = float(value_range[0]), float(value_range[1])
    return min(max(float(value), lower), upper)


def _injection_spec(
    executor: Callable[[Mapping[str, Any]], dict[str, Any]],
    point_name: str,
) -> Mapping[str, Any]:
    registry = getattr(executor, "registry", {})
    if isinstance(registry, Mapping) and point_name in registry:
        return registry[point_name]
    raise KeyError(point_name)


def _union_ranges(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    values = [*lhs, *rhs]
    if not values:
        return []
    return [min(values), max(values)]


def _same_range(lhs: Sequence[float], rhs: Sequence[float]) -> bool:
    return len(lhs) == 2 and len(rhs) == 2 and float(lhs[0]) == float(rhs[0]) and float(lhs[1]) == float(rhs[1])


def _merge_regions(
    regions: Sequence[Mapping[str, Any]],
    d_threshold: float,
) -> list[dict[str, Any]]:
    records = _aggregate_region_records(regions)
    if not records:
        return []

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    threshold = _merge_threshold(records, d_threshold)
    for left, right, distance in _neighbor_pairs(records):
        if distance <= threshold:
            union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(records)):
        groups.setdefault(find(index), []).append(index)

    merged: list[dict[str, Any]] = []
    for members in sorted(groups.values(), key=lambda group: (-len(group), group)):
        combined = _combine_region_group([records[index] for index in members], len(merged) + 1)
        if combined:
            merged.append(combined)
    return merged


def _aggregate_region_records(regions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for region in regions:
        if not region["mask_slope_range"]:
            continue
        key = _region_cell_key(region)
        current = records.get(key)
        if current is None:
            records[key] = dict(region)
            records[key]["hazard_ids"] = sorted(set(region.get("hazard_ids", [region["hazard_id"]])))
            records[key]["source_count"] = int(region.get("source_count", 1))
            continue
        current["hazard_ids"] = sorted(set(current["hazard_ids"]) | set(region.get("hazard_ids", [region["hazard_id"]])))
        current["driver_amp_range"] = _union_ranges(current["driver_amp_range"], region["driver_amp_range"])
        current["source_driver_amp_range"] = _union_ranges(
            current["source_driver_amp_range"], region["source_driver_amp_range"]
        )
        current["stable_slope_range"] = _intersect_ranges(current["stable_slope_range"], region["stable_slope_range"])
        current["extreme_slope_range"] = _intersect_ranges(current["extreme_slope_range"], region["extreme_slope_range"])
        current["mask_slope_range"] = _intersect_ranges(current["mask_slope_range"], region["mask_slope_range"])
        current["source_count"] = int(current.get("source_count", 1)) + int(region.get("source_count", 1))
    return sorted(records.values(), key=lambda item: (int(item["k_int"]), int(item["driver_duration_range"][0])))


def _region_cell_key(region: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(region["driver_duration_range"][0]),
        int(region["k_int"]),
        str(region["alarm_id"]),
        str(region["hazard_driver"]),
        str(region["driver_direction"]),
        str(region["masking_variable"]),
        str(region["bomega_mode"]),
        str(region["slope_search_method"]),
    )


def _merge_threshold(records: Sequence[Mapping[str, Any]], d_threshold: float) -> float:
    if d_threshold > 0:
        return float(d_threshold)
    widths = [
        float(region["mask_slope_range"][1]) - float(region["mask_slope_range"][0])
        for region in records
        if region["mask_slope_range"] and float(region["mask_slope_range"][1]) > float(region["mask_slope_range"][0])
    ]
    if not widths:
        return 0.0
    return REGION_GAMMA_MERGE * statistics.median(widths)


def _neighbor_pairs(records: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, float]]:
    buckets: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
    for index, region in enumerate(records):
        key = (
            int(region["k_int"]),
            str(region["alarm_id"]),
            str(region["hazard_driver"]),
            str(region["driver_direction"]),
            str(region["masking_variable"]),
            str(region["bomega_mode"]),
            str(region["slope_search_method"]),
        )
        buckets.setdefault(key, []).append((int(region["driver_duration_range"][0]), index))

    pairs: list[tuple[int, int, float]] = []
    for items in buckets.values():
        items.sort()
        for left, right in zip(items, items[1:]):
            if right[0] - left[0] != REGION_DELTA_T:
                continue
            distance = _slope_distance(records[left[1]]["mask_slope_range"], records[right[1]]["mask_slope_range"])
            pairs.append((left[1], right[1], distance))
    return pairs


def _slope_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1]))


def _combine_region_group(regions: Sequence[Mapping[str, Any]], region_index: int) -> dict[str, Any] | None:
    ordered = sorted(regions, key=lambda item: int(item["driver_duration_range"][0]))
    combined = dict(ordered[0])
    combined["region_id"] = f"R_{region_index:03d}"
    combined["hazard_ids"] = sorted({hazard for region in ordered for hazard in region.get("hazard_ids", [region["hazard_id"]])})
    combined["hazard_id"] = combined["hazard_ids"][0]
    combined["driver_duration_range"] = _union_many(region["driver_duration_range"] for region in ordered)
    combined["driver_amp_range"] = _union_many(region["driver_amp_range"] for region in ordered)
    combined["source_driver_amp_range"] = _union_many(region["source_driver_amp_range"] for region in ordered)
    combined["stable_slope_range"] = _intersect_many(region["stable_slope_range"] for region in ordered)
    combined["extreme_slope_range"] = _intersect_many(region["extreme_slope_range"] for region in ordered)
    combined["mask_slope_range"] = _intersect_many(region["mask_slope_range"] for region in ordered)
    if not combined["mask_slope_range"]:
        return None
    combined["runtime_steps"] = max(int(region["runtime_steps"]) for region in ordered)
    combined["n_members"] = len(ordered)
    combined["source_count"] = sum(int(region.get("source_count", 1)) for region in ordered)
    return combined


def _union_many(ranges: Sequence[Sequence[float]]) -> list[float]:
    values = [value for item in ranges for value in item]
    return [min(values), max(values)] if values else []


def _intersect_many(ranges: Sequence[Sequence[float]]) -> list[float]:
    values = [list(item) for item in ranges if item]
    if not values:
        return []
    lower = max(float(item[0]) for item in values)
    upper = min(float(item[1]) for item in values)
    return [lower, upper] if lower < upper else []


def _intersect_ranges(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    if not lhs or not rhs:
        return []
    lower = max(float(lhs[0]), float(rhs[0]))
    upper = min(float(lhs[1]), float(rhs[1]))
    return [lower, upper] if lower < upper else []


def _build_payload(args: argparse.Namespace, regions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "platform": str(args.platform),
        "hazard_id": str(args.hazard_id),
        "alarm_id": str(regions[0]["alarm_id"]) if regions else "",
        "hazard_driver": str(args.hazard_driver),
        "runtime_rule": RUNTIME_RULE,
        "seed": int(args.seed),
        "workers": int(args.workers),
        "d_threshold": float(args.d_threshold),
        "representative_samples": int(args.representative_samples),
        "sample_duration_mod": int(args.sample_duration_mod),
        "dynamic_slope_tol": DYNAMIC_SLOPE_TOL,
        "regions": list(regions),
    }


def _write_outputs(
    output_root: Path,
    platform: str,
    hazard_driver: str,
    payload: Mapping[str, Any],
    representative_payload: Mapping[str, Any],
) -> None:
    base = Path(output_root) / "results"
    region_dir = base / "stage3" / platform / hazard_driver
    log_dir = base / "logs" / "stage3" / platform / hazard_driver
    manifest_dir = base / "manifests" / "stage3" / platform / hazard_driver
    region_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    _write_json(region_dir / "coordinated_regions.json", payload)
    _write_json(region_dir / "representative_regions.json", representative_payload)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": "stage3",
        "platform": platform,
        "hazard_driver": hazard_driver,
        "event": "write_regions",
        "seed": payload["seed"],
        "workers": payload["workers"],
        "d_threshold": payload["d_threshold"],
        "region_count": len(payload["regions"]),
    }
    (log_dir / "events.jsonl").write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(
        manifest_dir / "run_manifest.json",
        {
            "stage": "stage3",
            "platform": platform,
            "hazard_driver": hazard_driver,
            "runtime_rule": RUNTIME_RULE,
            "seed": payload["seed"],
            "workers": payload["workers"],
            "d_threshold": payload["d_threshold"],
            "representative_samples": payload["representative_samples"],
            "sample_duration_mod": payload["sample_duration_mod"],
            "dynamic_slope_tol": payload["dynamic_slope_tol"],
            "region_count": len(payload["regions"]),
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

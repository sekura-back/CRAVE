from __future__ import annotations

import argparse
import importlib
import json
import re
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Callable


HOLD_STEPS = 100
DEFAULT_DURATION_START = 100
DEFAULT_DURATION_STOP = 1000
DEFAULT_DURATION_STEP = 5
DEFAULT_DURATION_ANCHOR_STEP = 100
DEFAULT_WORKERS = 16


def _require_rule_ids(hazard_id: str | None, alarm_id: str | None) -> None:
    if hazard_id is not None and not str(hazard_id).startswith("H-"):
        raise ValueError("hazard_id must start with H-")
    if alarm_id is not None and not str(alarm_id).startswith("A-"):
        raise ValueError("alarm_id must start with A-")


def _ids(value: Any) -> set[str]:
    if isinstance(value, str):
        return {part.strip() for part in value.replace(",", "|").split("|") if part.strip()}
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _first_hazard_step(probe: Mapping[str, Any], hazard_id: str) -> int | None:
    for index, row in enumerate(probe.get("rows", []) or []):
        if hazard_id in _ids(row.get("hazard_rule_ids", [])):
            return index
    score = probe.get("score", {})
    if isinstance(score, Mapping) and score.get("first_hazard_step") is not None:
        rules = probe.get("_rules", {})
        hazard = rules.get("hazards", {}).get(hazard_id) if isinstance(rules, Mapping) else None
        margin_col = hazard.get("margin_col") if isinstance(hazard, Mapping) else None
        if margin_col:
            for index, row in enumerate(probe.get("rows", []) or []):
                try:
                    if float(row[margin_col]) <= 0.0:
                        return index
                except (KeyError, TypeError, ValueError):
                    pass
    hazards = probe.get("hazards")
    if isinstance(hazards, Mapping) and hazards.get(hazard_id):
        try:
            if isinstance(score, Mapping) and score.get("first_hazard_step") is not None:
                return int(score.get("first_hazard_step"))
            return 0
        except (TypeError, ValueError):
            return 0
    return None


def _triggered_hazard_ids(probe: Mapping[str, Any]) -> set[str]:
    hazards = probe.get("hazards")
    if isinstance(hazards, Mapping):
        return {str(rule_id) for rule_id, hit in hazards.items() if hit}
    ids: set[str] = set()
    for row in probe.get("rows", []) or []:
        ids.update(_ids(row.get("hazard_rule_ids", [])))
    score = probe.get("score", {})
    if isinstance(score, Mapping) and score.get("first_hazard_step") is not None:
        rules = probe.get("_rules", {})
        hazards = rules.get("hazards", {}) if isinstance(rules, Mapping) else {}
        for rule_id, hazard in hazards.items():
            margin_col = hazard.get("margin_col") if isinstance(hazard, Mapping) else None
            if not margin_col:
                continue
            for row in probe.get("rows", []) or []:
                try:
                    if float(row[margin_col]) <= 0.0:
                        ids.add(str(rule_id))
                        break
                except (KeyError, TypeError, ValueError):
                    pass
    return {rule_id for rule_id in ids if rule_id.startswith("H-")}


def _first_triggered_hazard(probe: Mapping[str, Any]) -> tuple[str, int] | None:
    for index, row in enumerate(probe.get("rows", []) or []):
        ids = sorted(rule_id for rule_id in _ids(row.get("hazard_rule_ids", [])) if rule_id.startswith("H-"))
        if len(ids) == 1:
            return ids[0], index
        if len(ids) > 1:
            return None

    candidates: list[tuple[int, str]] = []
    rules = probe.get("_rules", {})
    hazards = rules.get("hazards", {}) if isinstance(rules, Mapping) else {}
    for rule_id, hazard in hazards.items():
        margin_col = hazard.get("margin_col") if isinstance(hazard, Mapping) else None
        if not margin_col:
            continue
        for index, row in enumerate(probe.get("rows", []) or []):
            try:
                if float(row[margin_col]) <= 0.0:
                    candidates.append((index, str(rule_id)))
                    break
            except (KeyError, TypeError, ValueError):
                pass
    if candidates:
        first_step = min(step for step, _ in candidates)
        first_ids = sorted(rule_id for step, rule_id in candidates if step == first_step and rule_id.startswith("H-"))
        return (first_ids[0], first_step) if len(first_ids) == 1 else None

    hazards = probe.get("hazards")
    if isinstance(hazards, Mapping):
        ids = sorted(str(rule_id) for rule_id, hit in hazards.items() if hit and str(rule_id).startswith("H-"))
        if len(ids) == 1:
            step = _first_hazard_step(probe, ids[0])
            return ids[0], step or 0
    return None


def _has_prehazard_alarm(probe: Mapping[str, Any], alarm_id: str, hazard_step: int) -> bool:
    if alarm_id in _ids(probe.get("prehazard_alarm_ids", [])):
        return True
    score = probe.get("score", {})
    if isinstance(score, Mapping) and alarm_id in _ids(score.get("prehazard_alarm_rule_ids", [])):
        return True
    for row in (probe.get("rows", []) or [])[:hazard_step]:
        if alarm_id in _ids(row.get("alarm_rule_ids", [])):
            return True
    rules = probe.get("_rules", {})
    alarm = rules.get("alarms", {}).get(alarm_id) if isinstance(rules, Mapping) else None
    margin_col = alarm.get("margin_col") if isinstance(alarm, Mapping) else None
    if margin_col:
        for row in (probe.get("rows", []) or [])[:hazard_step]:
            try:
                if float(row[margin_col]) <= 0.0:
                    return True
            except (KeyError, TypeError, ValueError):
                pass
    return False


def _prehazard_alarm_ids(probe: Mapping[str, Any], hazard_step: int) -> set[str]:
    ids = _ids(probe.get("prehazard_alarm_ids", []))
    score = probe.get("score", {})
    if isinstance(score, Mapping):
        ids.update(_ids(score.get("prehazard_alarm_rule_ids", [])))
    for row in (probe.get("rows", []) or [])[:hazard_step]:
        ids.update(_ids(row.get("alarm_rule_ids", [])))
    rules = probe.get("_rules", {})
    alarms = rules.get("alarms", {}) if isinstance(rules, Mapping) else {}
    for rule_id, alarm in alarms.items():
        margin_col = alarm.get("margin_col") if isinstance(alarm, Mapping) else None
        if not margin_col:
            continue
        for row in (probe.get("rows", []) or [])[:hazard_step]:
            try:
                if float(row[margin_col]) <= 0.0:
                    ids.add(str(rule_id))
            except (KeyError, TypeError, ValueError):
                pass
    return {rule_id for rule_id in ids if rule_id.startswith("A-")}


def _single_id(ids: set[str], label: str) -> str:
    if not ids:
        raise ValueError(f"no {label} found")
    if len(ids) > 1:
        raise ValueError(f"multiple {label}s found: {sorted(ids)}")
    return next(iter(ids))


def _classify_probe(probe: Mapping[str, Any], hazard_id: str | None, alarm_id: str | None) -> str:
    _require_rule_ids(hazard_id, alarm_id)
    first_hazard = _first_triggered_hazard(probe)
    if first_hazard is None:
        return "untarget"
    if hazard_id is None:
        hazard_id, hazard_step = first_hazard
    else:
        if first_hazard[0] != hazard_id:
            return "untarget"
        hazard_step = first_hazard[1]
    if alarm_id is None:
        alarms = _prehazard_alarm_ids(probe, hazard_step)
        if len(alarms) != 1:
            return "untarget"
        alarm_id = next(iter(alarms))
    return "target" if _has_prehazard_alarm(probe, alarm_id, hazard_step) else "untarget"


def _classify_with_ids(probe: dict[str, Any], hazard_id: str | None, alarm_id: str | None) -> str:
    state = _classify_probe(probe, hazard_id, alarm_id)
    if state != "target":
        return state
    first_hazard = _first_triggered_hazard(probe)
    resolved_hazard_id = hazard_id or (first_hazard[0] if first_hazard else None)
    if resolved_hazard_id is None:
        return "untarget"
    hazard_step = _first_hazard_step(probe, resolved_hazard_id)
    resolved_alarm_id = alarm_id or _single_id(_prehazard_alarm_ids(probe, hazard_step or 0), "prehazard alarm")
    probe["hazard_id"] = resolved_hazard_id
    probe["alarm_id"] = resolved_alarm_id
    return state


def _build_duration_layers(start: int, stop: int, step: int) -> list[list[int]]:
    if step <= 0:
        raise ValueError("duration_step must be positive")
    if stop < start:
        raise ValueError("duration range must be increasing")
    return [[duration for duration in range(int(start), int(stop) + 1, int(step))]]


def _duration_work_layers(values: list[int], anchor_step: int = DEFAULT_DURATION_ANCHOR_STEP) -> list[list[tuple[int, int | None, int | None]]]:
    if not values:
        return []
    if anchor_step <= 0:
        raise ValueError("duration_anchor_step must be positive")

    start = values[0]
    anchor_indices = [
        index
        for index, value in enumerate(values)
        if (int(value) - int(start)) % int(anchor_step) == 0
    ]
    if 0 not in anchor_indices:
        anchor_indices.insert(0, 0)
    if len(values) - 1 not in anchor_indices:
        anchor_indices.append(len(values) - 1)
    anchor_indices = sorted(set(anchor_indices))

    layers: list[list[tuple[int, int | None, int | None]]] = [
        [(index, None, None) for index in anchor_indices]
    ]
    segments = list(zip(anchor_indices, anchor_indices[1:]))
    while segments:
        layer: list[tuple[int, int | None, int | None]] = []
        next_segments: list[tuple[int, int]] = []
        for left_index, right_index in segments:
            if right_index - left_index <= 1:
                continue
            mid_index = (left_index + right_index) // 2
            layer.append((mid_index, left_index, right_index))
            next_segments.append((left_index, mid_index))
            next_segments.append((mid_index, right_index))
        if layer:
            layers.append(layer)
        segments = next_segments
    return layers


def _scan_points(lower_amp: float, upper_amp: float, coarse_span: float) -> list[float]:
    if upper_amp < lower_amp:
        raise ValueError("amp range must be increasing")
    if coarse_span <= 0:
        raise ValueError("coarse_span must be positive")
    steps = max(1, int(ceil((upper_amp - lower_amp) / coarse_span)))
    return [lower_amp + (upper_amp - lower_amp) * index / steps for index in range(steps + 1)]


def _state_key(probe: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    if probe.get("state_class") != "target":
        return "untarget", None, None
    return "target", str(probe.get("hazard_id")), str(probe.get("alarm_id"))


def _bisect_transition(probe: Callable[[float], dict[str, Any]], left: dict[str, Any], right: dict[str, Any], amp_tol: float) -> float:
    lo = dict(left)
    hi = dict(right)
    left_key = _state_key(lo)
    while hi["amp"] - lo["amp"] > amp_tol:
        mid = probe((lo["amp"] + hi["amp"]) / 2.0)
        if _state_key(mid) == left_key:
            lo = mid
        else:
            hi = mid
    return hi["amp"]


def _find_target_intervals(
    probe: Callable[[float], dict[str, Any]],
    lower_amp: float,
    upper_amp: float,
    coarse_span: float,
    amp_tol: float,
) -> list[dict[str, Any]]:
    probes = [probe(amp) for amp in _scan_points(lower_amp, upper_amp, coarse_span)]
    intervals: list[dict[str, Any]] = []
    current_lo = probes[0]["amp"]
    current_key = _state_key(probes[0])
    for left, right in zip(probes, probes[1:]):
        left_key = _state_key(left)
        right_key = _state_key(right)
        if left_key == right_key:
            continue
        boundary = _bisect_transition(probe, left, right, amp_tol)
        if current_key[0] == "target" and current_key[1] and current_key[2]:
            intervals.append(
                {
                    "rate_lo": current_lo,
                    "rate_hi": boundary,
                    "first_hazard_id": current_key[1],
                    "alarm_id": current_key[2],
                }
            )
        current_lo = boundary
        current_key = right_key
    if current_key[0] == "target" and current_key[1] and current_key[2]:
        intervals.append(
            {
                "rate_lo": current_lo,
                "rate_hi": upper_amp,
                "first_hazard_id": current_key[1],
                "alarm_id": current_key[2],
            }
        )
    return intervals


def _transition_points(
    intervals: list[Mapping[str, Any]],
    lower_amp: float | None = None,
    upper_amp: float | None = None,
    endpoint_tol: float = 1e-12,
) -> list[float]:
    points: set[float] = set()
    for interval in intervals:
        for value in (float(interval["rate_lo"]), float(interval["rate_hi"])):
            if lower_amp is not None and abs(value - float(lower_amp)) <= endpoint_tol:
                continue
            if upper_amp is not None and abs(value - float(upper_amp)) <= endpoint_tol:
                continue
            points.add(value)
    return sorted(points)


def _bracket_seeded_transition(
    probe: Callable[[float], dict[str, Any]],
    left_amp: float,
    right_amp: float,
    lower_amp: float,
    upper_amp: float,
    coarse_span: float,
    amp_tol: float,
) -> float | None:
    lo_amp = max(float(lower_amp), min(float(left_amp), float(right_amp)))
    hi_amp = min(float(upper_amp), max(float(left_amp), float(right_amp)))
    if hi_amp - lo_amp <= amp_tol:
        pad = max(float(coarse_span) / 2.0, amp_tol * 2.0)
        center = (lo_amp + hi_amp) / 2.0
        lo_amp = max(float(lower_amp), center - pad)
        hi_amp = min(float(upper_amp), center + pad)

    lo = probe(lo_amp)
    hi = probe(hi_amp)
    pad = max(float(coarse_span), hi_amp - lo_amp, amp_tol * 2.0)
    while _state_key(lo) == _state_key(hi):
        if lo_amp <= lower_amp and hi_amp >= upper_amp:
            return None
        lo_amp = max(float(lower_amp), lo_amp - pad)
        hi_amp = min(float(upper_amp), hi_amp + pad)
        lo = probe(lo_amp)
        hi = probe(hi_amp)
        pad *= 2.0
    return _bisect_transition(probe, lo, hi, amp_tol)


def _intervals_from_transitions(
    probe: Callable[[float], dict[str, Any]],
    transitions: list[float],
    lower_amp: float,
    upper_amp: float,
    amp_tol: float,
) -> list[dict[str, Any]]:
    points = [float(lower_amp)] + sorted({float(value) for value in transitions}) + [float(upper_amp)]
    intervals: list[dict[str, Any]] = []
    for left, right in zip(points, points[1:]):
        if right - left <= amp_tol:
            continue
        sample = probe((left + right) / 2.0)
        key = _state_key(sample)
        if key[0] == "target" and key[1] and key[2]:
            intervals.append(
                {
                    "rate_lo": left,
                    "rate_hi": right,
                    "first_hazard_id": key[1],
                    "alarm_id": key[2],
                }
            )
    return intervals


def _find_target_intervals_from_seeds(
    probe: Callable[[float], dict[str, Any]],
    lower_amp: float,
    upper_amp: float,
    coarse_span: float,
    amp_tol: float,
    left_intervals: list[Mapping[str, Any]],
    right_intervals: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left_points = _transition_points(left_intervals, lower_amp, upper_amp)
    right_points = _transition_points(right_intervals, lower_amp, upper_amp)
    if not left_points or len(left_points) != len(right_points):
        return _find_target_intervals(probe, lower_amp, upper_amp, coarse_span, amp_tol)

    transitions: list[float] = []
    for left_amp, right_amp in zip(left_points, right_points):
        transition = _bracket_seeded_transition(
            probe, left_amp, right_amp, lower_amp, upper_amp, coarse_span, amp_tol
        )
        if transition is None:
            return _find_target_intervals(probe, lower_amp, upper_amp, coarse_span, amp_tol)
        transitions.append(transition)
    return _intervals_from_transitions(probe, transitions, lower_amp, upper_amp, amp_tol)


def _duration_values(
    durations: Iterable[int] | None,
    duration_start: int | None,
    duration_stop: int | None,
    duration_step: int | None,
) -> tuple[list[int], list[list[int]]]:
    if durations is not None:
        values = sorted({int(value) for value in durations})
        return values, [values]
    if duration_start is None and duration_stop is None:
        duration_start = DEFAULT_DURATION_START
        duration_stop = DEFAULT_DURATION_STOP
    elif duration_start is None or duration_stop is None:
        raise ValueError("provide both duration_start and duration_stop")
    layers = _build_duration_layers(duration_start, duration_stop, duration_step or DEFAULT_DURATION_STEP)
    values = [duration for layer in layers for duration in layer]
    return values, layers


def _amp_range_from_registry(executor: Any, hazard_driver: str) -> tuple[float, float]:
    registry = getattr(executor, "registry", None)
    if not isinstance(registry, Mapping):
        raise ValueError("provide amp_magnitude_range or use an executor with registry")
    point = registry.get(hazard_driver)
    if not isinstance(point, Mapping):
        raise ValueError(f"hazard_driver not found in executor registry: {hazard_driver}")
    try:
        upper = abs(float(point["rate"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"registry point missing positive rate: {hazard_driver}") from exc
    if upper <= 0.0:
        raise ValueError(f"registry point missing positive rate: {hazard_driver}")
    return 0.0, upper


def _stop_ignore_predicates(executor: Any) -> list[str]:
    rules = getattr(executor, "rules", {})
    alarms = rules.get("alarms", {}) if isinstance(rules, Mapping) else {}
    predicates: set[str] = set()
    for alarm in alarms.values():
        expr = str(alarm.get("expr", "")) if isinstance(alarm, Mapping) else ""
        predicates.update(re.findall(r"\b[A-Z][A-Z0-9_]*_TH\b", expr))
    return sorted(predicates)


def _signed_amp(direction: str, amp: float) -> float:
    if direction == "pos":
        return float(amp)
    if direction == "neg":
        return -float(amp)
    raise ValueError(f"unsupported direction: {direction}")


def load_executor(manifest_path: str | Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    path = Path(manifest_path)
    module = importlib.import_module(f"simulators.{path.parent.name}.runtime_executor")
    return module.create_executor(path)


def _manifest_path(root: str | Path, platform: str) -> Path:
    path = Path(root) / "simulators" / str(platform) / "system_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found for platform {platform}: {path}")
    return path


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log(path: str | Path, event: Mapping[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **dict(event)}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _solve_duration_boundary(
    *,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    hazard_driver: str,
    hazard_id: str | None,
    alarm_id: str | None,
    duration: int,
    direction: str,
    lower_amp: float,
    upper_amp: float,
    coarse_span: float,
    amp_tol: float,
    stop_ignore_predicates: list[str],
    left_boundary: Mapping[str, Any] | None = None,
    right_boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_steps = int(duration) + HOLD_STEPS
    probe_count = 0

    def probe(amp: float) -> dict[str, Any]:
        nonlocal probe_count
        probe_count += 1
        task = {
            "runtime_args": {
                "steps": runtime_steps,
                "stop_on_trip": True,
                "ignore_protection_predicates_for_stop": stop_ignore_predicates,
            },
            "injection": {
                "point": hazard_driver,
                "amp": _signed_amp(direction, amp),
                "duration": int(duration),
                "t_start": 0,
                "role": "manip",
            },
        }
        payload = dict(executor(task))
        if hasattr(executor, "rules"):
            payload["_rules"] = getattr(executor, "rules")
        payload.update(
            {
                "amp": float(amp),
                "duration": int(duration),
                "direction": direction,
                "runtime_steps": runtime_steps,
            }
        )
        payload["state_class"] = _classify_with_ids(payload, hazard_id, alarm_id)
        return payload

    if left_boundary is not None and right_boundary is not None:
        intervals = _find_target_intervals_from_seeds(
            probe,
            lower_amp,
            upper_amp,
            coarse_span,
            amp_tol,
            list(left_boundary["target_intervals"]),
            list(right_boundary["target_intervals"]),
        )
    else:
        intervals = _find_target_intervals(
            probe, lower_amp, upper_amp, coarse_span, amp_tol
        )
    first_hazard_ids = sorted({str(item["first_hazard_id"]) for item in intervals})
    alarm_ids = sorted({str(item["alarm_id"]) for item in intervals})
    return {
        "hazard_driver": hazard_driver,
        "hazard_id": hazard_id or (first_hazard_ids[0] if len(first_hazard_ids) == 1 else None),
        "first_hazard_ids": first_hazard_ids,
        "alarm_id": alarm_id or (alarm_ids[0] if len(alarm_ids) == 1 else None),
        "duration": int(duration),
        "runtime_steps": runtime_steps,
        "direction": direction,
        "target_intervals": intervals,
        "target_interval_count": len(intervals),
        "lower_target_amp": intervals[0]["rate_lo"] if intervals else None,
        "upper_target_amp": intervals[-1]["rate_hi"] if intervals else None,
        "probe_count": probe_count,
    }


def _solve_duration_process_job(job: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    executor = load_executor(job["manifest_path"])
    boundary = _solve_duration_boundary(
        executor=executor,
        hazard_driver=str(job["hazard_driver"]),
        hazard_id=job.get("hazard_id"),
        alarm_id=job.get("alarm_id"),
        duration=int(job["duration"]),
        direction=str(job["direction"]),
        lower_amp=float(job["lower_amp"]),
        upper_amp=float(job["upper_amp"]),
        coarse_span=float(job["coarse_span"]),
        amp_tol=float(job["amp_tol"]),
        stop_ignore_predicates=list(job.get("stop_ignore_predicates", [])),
        left_boundary=job.get("left_boundary"),
        right_boundary=job.get("right_boundary"),
    )
    return int(job["index"]), boundary


def run_base_search(
    *,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    hazard_driver: str,
    hazard_id: str | None = None,
    alarm_id: str | None = None,
    durations: Iterable[int] | None = None,
    duration_start: int | None = None,
    duration_stop: int | None = None,
    duration_step: int | None = None,
    duration_anchor_step: int | None = None,
    amp_magnitude_range: tuple[float, float] | None = None,
    coarse_span: float = 0.05,
    amp_tol: float = 1e-4,
    directions: Iterable[str] = ("pos", "neg"),
    workers: int = DEFAULT_WORKERS,
    platform: str | None = None,
    process_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    _require_rule_ids(hazard_id, alarm_id)
    duration_list, _ = _duration_values(durations, duration_start, duration_stop, duration_step)
    work_layers = _duration_work_layers(
        duration_list, duration_anchor_step or DEFAULT_DURATION_ANCHOR_STEP
    )
    duration_layers = [
        [duration_list[index] for index, _, _ in layer]
        for layer in work_layers
    ]
    if amp_magnitude_range is None:
        amp_magnitude_range = _amp_range_from_registry(executor, hazard_driver)
    lower_amp, upper_amp = map(float, amp_magnitude_range)
    stop_ignore_predicates = _stop_ignore_predicates(executor)

    def solve_duration(
        duration: int,
        direction: str,
        left_boundary: Mapping[str, Any] | None = None,
        right_boundary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _solve_duration_boundary(
            executor=executor,
            hazard_driver=hazard_driver,
            hazard_id=hazard_id,
            alarm_id=alarm_id,
            duration=duration,
            direction=direction,
            lower_amp=lower_amp,
            upper_amp=upper_amp,
            coarse_span=coarse_span,
            amp_tol=amp_tol,
            stop_ignore_predicates=stop_ignore_predicates,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

    def solve_direction(direction: str) -> list[dict[str, Any]]:
        if not duration_list:
            return []
        cache: dict[int, dict[str, Any]] = {}

        def solve_item(item: tuple[int, int | None, int | None]) -> tuple[int, dict[str, Any]]:
            index, left_index, right_index = item
            if left_index is None or right_index is None:
                return index, solve_duration(duration_list[index], direction)
            return index, solve_duration(
                duration_list[index],
                direction,
                cache[left_index],
                cache[right_index],
            )

        for layer in work_layers:
            pending = [item for item in layer if item[0] not in cache]
            if not pending:
                continue
            if process_manifest_path is not None and workers and workers > 1 and len(pending) > 1:
                jobs = []
                for index, left_index, right_index in pending:
                    jobs.append(
                        {
                            "index": index,
                            "manifest_path": str(process_manifest_path),
                            "hazard_driver": hazard_driver,
                            "hazard_id": hazard_id,
                            "alarm_id": alarm_id,
                            "duration": duration_list[index],
                            "direction": direction,
                            "lower_amp": lower_amp,
                            "upper_amp": upper_amp,
                            "coarse_span": coarse_span,
                            "amp_tol": amp_tol,
                            "stop_ignore_predicates": stop_ignore_predicates,
                            "left_boundary": cache[left_index] if left_index is not None else None,
                            "right_boundary": cache[right_index] if right_index is not None else None,
                        }
                    )
                with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                    solved = list(pool.map(_solve_duration_process_job, jobs))
            elif workers and workers > 1 and len(pending) > 1:
                with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                    solved = list(pool.map(solve_item, pending))
            else:
                solved = [solve_item(item) for item in pending]
            for index, boundary in solved:
                cache[index] = boundary
        return [cache[index] for index in range(len(duration_list))]

    direction_list = tuple(directions)
    boundary_groups = [solve_direction(direction) for direction in direction_list]
    boundaries = [boundary for group in boundary_groups for boundary in group]
    boundaries.sort(
        key=lambda item: (
            item["hazard_driver"],
            item["duration"],
            item["direction"],
        )
    )
    result: dict[str, Any] = {
        "duration_layers": duration_layers,
        "boundaries": boundaries,
        "summary": {"probe_count": sum(int(boundary.get("probe_count", 0)) for boundary in boundaries)},
    }
    if platform is not None:
        result["platform"] = platform
    return result


def _output_paths(output_root: str | Path, platform: str, hazard_driver: str) -> tuple[Path, Path, Path]:
    root = Path(output_root)
    artifact_path = root / "results" / "stage2" / platform / hazard_driver / "boundary_results.json"
    log_path = root / "results" / "logs" / "stage2_base" / platform / hazard_driver / "events.jsonl"
    run_manifest_path = (
        root / "results" / "manifests" / "stage2_base" / platform / hazard_driver / "run_manifest.json"
    )
    return artifact_path, log_path, run_manifest_path


def run_base_from_platform(
    *,
    platform: str,
    hazard_driver: str,
    root: str | Path | None = None,
    output_root: str | Path | None = None,
    output_path: str | Path | None = None,
    direction_values: Iterable[str] = ("pos", "neg"),
    duration_start: int | None = None,
    duration_stop: int | None = None,
    duration_step: int | None = None,
    duration_anchor_step: int | None = None,
    coarse_span: float = 0.05,
    amp_tol: float = 1e-4,
    workers: int = DEFAULT_WORKERS,
    seed: int = 460,
) -> dict[str, Any]:
    repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    resolved_output_root = Path(output_root) if output_root is not None else repo_root
    manifest_path = _manifest_path(repo_root, platform)
    executor = load_executor(manifest_path)
    artifact_path, log_path, run_manifest_path = _output_paths(resolved_output_root, platform, hazard_driver)
    if output_path is not None:
        artifact_path = Path(output_path)

    _log(log_path, {"event": "start", "stage": "stage2_base", "platform": platform, "hazard_driver": hazard_driver})
    result = run_base_search(
        executor=executor,
        platform=platform,
        hazard_driver=hazard_driver,
        duration_start=duration_start,
        duration_stop=duration_stop,
        duration_step=duration_step,
        duration_anchor_step=duration_anchor_step,
        coarse_span=coarse_span,
        amp_tol=amp_tol,
        directions=tuple(direction_values),
        workers=workers,
        process_manifest_path=manifest_path,
    )
    write_base_output(artifact_path, result)
    _write_json(
        run_manifest_path,
        {
            "stage": "stage2_base",
            "platform": platform,
            "hazard_driver": hazard_driver,
            "seed": int(seed),
            "manifest_path": str(manifest_path),
            "artifact_path": str(artifact_path),
            "log_path": str(log_path),
            "duration_rule": "duration + 100",
        },
    )
    _log(log_path, {"event": "finish", "stage": "stage2_base", "boundary_count": len(result["boundaries"])})
    return result


def write_base_output(path: str | Path, result: Mapping[str, Any]) -> None:
    _write_json(path, result)


def _directions(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("provide at least one direction")
    return items


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--hazard-driver", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output-root")
    parser.add_argument("--output-path")
    parser.add_argument("--directions", type=_directions, default=("pos", "neg"))
    parser.add_argument("--duration-start", type=int)
    parser.add_argument("--duration-stop", type=int)
    parser.add_argument("--duration-step", type=int)
    parser.add_argument("--duration-anchor-step", type=int, default=DEFAULT_DURATION_ANCHOR_STEP)
    parser.add_argument("--coarse-span", type=float, default=0.05)
    parser.add_argument("--amp-tol", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=460)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_base_from_platform(
        platform=args.platform,
        hazard_driver=args.hazard_driver,
        root=args.root,
        output_root=args.output_root,
        output_path=args.output_path,
        direction_values=args.directions,
        duration_start=args.duration_start,
        duration_stop=args.duration_stop,
        duration_step=args.duration_step,
        duration_anchor_step=args.duration_anchor_step,
        coarse_span=args.coarse_span,
        amp_tol=args.amp_tol,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

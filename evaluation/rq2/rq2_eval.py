from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import src.stage2.base_search as base_search


HOLD_STEPS = 100
DEFAULT_PLATFORM = "tennessee_eastman"
DEFAULT_VARIABLE = "xmv_07"
DEFAULT_DURATION_RANGE = (100, 1000)
DEFAULT_DURATION_STEP = 5
DEFAULT_RATE_RANGE = (0.0, 1.0)
DEFAULT_SEED = 460
DEFAULT_HAZARD_ID = "H-TEP-SEPARATOR-LEVEL-LOW"
DEFAULT_ALARM_ID = "A-TEP-SEP-LEVEL-TRACK"
RECONSTRUCTION_POLICY = "target_sample_cell_union"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(indent=2, sort_keys=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in encoder.iterencode(payload):
            handle.write(chunk)
        handle.write("\n")


def _duration_values(duration_range: tuple[int, int], step: int) -> list[int]:
    start, stop = int(duration_range[0]), int(duration_range[1])
    if step <= 0:
        raise ValueError("duration_step must be positive")
    if stop < start:
        raise ValueError("duration_range must be increasing")
    return list(range(start, stop + 1, int(step)))


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
    rate_range: tuple[float, float],
) -> list[list[float]]:
    rate_lo, rate_hi = map(float, rate_range)
    clipped = []
    for lo, hi in intervals:
        left = max(rate_lo, float(lo))
        right = min(rate_hi, float(hi))
        if right < left:
            continue
        clipped.append((left, right))
    if not clipped:
        return []
    clipped.sort()
    merged: list[list[float]] = [[clipped[0][0], clipped[0][1]]]
    for lo, hi in clipped[1:]:
        current = merged[-1]
        if lo <= current[1]:
            current[1] = max(current[1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _interval_width(intervals: Iterable[Iterable[float]]) -> float:
    return sum(max(0.0, float(hi) - float(lo)) for lo, hi in intervals)


def _interval_overlap(a: list[list[float]], b: list[list[float]]) -> float:
    total = 0.0
    for alo, ahi in a:
        for blo, bhi in b:
            total += max(0.0, min(ahi, bhi) - max(alo, blo))
    return total


def _integrate_values(durations: list[int], values: Mapping[int, float]) -> float:
    if len(durations) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(durations, durations[1:]):
        area += (float(values.get(left, 0.0)) + float(values.get(right, 0.0))) * (right - left) / 2.0
    return area


def _single_value(values: Iterable[str], label: str) -> str:
    items = sorted({str(value) for value in values if str(value)})
    if len(items) != 1:
        raise ValueError(f"expected exactly one {label}, found {items}")
    return items[0]


def _boundary_intervals(
    payload: Mapping[str, Any],
    *,
    direction: str,
    rate_range: tuple[float, float],
) -> dict[int, list[list[float]]]:
    by_duration: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for boundary in payload.get("boundaries", []):
        if str(boundary.get("direction")) != direction:
            continue
        duration = int(boundary["duration"])
        for interval in boundary.get("target_intervals", []):
            by_duration[duration].append((float(interval["rate_lo"]), float(interval["rate_hi"])))
    return {
        duration: _merge_intervals(intervals, rate_range)
        for duration, intervals in by_duration.items()
    }


def _boundary_rule_ids(payload: Mapping[str, Any], direction: str) -> tuple[str, str, str]:
    variables: list[str] = []
    hazards: list[str] = []
    alarms: list[str] = []
    for boundary in payload.get("boundaries", []):
        if str(boundary.get("direction")) != direction:
            continue
        variables.append(str(boundary.get("hazard_driver", "")))
        alarm_id = boundary.get("alarm_id")
        if alarm_id:
            alarms.append(str(alarm_id))
        hazard_id = boundary.get("hazard_id")
        if hazard_id:
            hazards.append(str(hazard_id))
        for interval in boundary.get("target_intervals", []):
            if interval.get("first_hazard_id"):
                hazards.append(str(interval["first_hazard_id"]))
            if interval.get("alarm_id"):
                alarms.append(str(interval["alarm_id"]))
    return (
        _single_value(variables, "hazard driver"),
        _single_value(hazards, "target hazard"),
        _single_value(alarms, "base alarm"),
    )


def build_reference_summary(
    boundary_path: str | Path,
    *,
    duration_range: tuple[int, int] = DEFAULT_DURATION_RANGE,
    rate_range: tuple[float, float] = DEFAULT_RATE_RANGE,
    duration_step: int = DEFAULT_DURATION_STEP,
    direction: str = "pos",
    epsilon_k: float = 1e-4,
) -> dict[str, Any]:
    payload = _read_json(boundary_path)
    durations = _duration_values(duration_range, duration_step)
    intervals_by_duration = _boundary_intervals(payload, direction=direction, rate_range=rate_range)
    width_by_duration = {
        duration: _interval_width(intervals_by_duration.get(duration, []))
        for duration in durations
    }
    base_area = _integrate_values(durations, width_by_duration)
    total_area = (duration_range[1] - duration_range[0]) * (rate_range[1] - rate_range[0])
    variable, hazard_id, alarm_id = _boundary_rule_ids(payload, direction)
    boundary_points = [
        {
            "duration": duration,
            "intervals": intervals_by_duration.get(duration, []),
        }
        for duration in durations
        if intervals_by_duration.get(duration)
    ]
    return {
        "platform": str(payload.get("platform", DEFAULT_PLATFORM)),
        "variable": variable,
        "target_hazard": hazard_id,
        "base_alarm": alarm_id,
        "direction": direction,
        "duration_range": [int(duration_range[0]), int(duration_range[1])],
        "duration_step": int(duration_step),
        "duration_values": durations,
        "rate_range": [float(rate_range[0]), float(rate_range[1])],
        "epsilon_K": float(epsilon_k),
        "valid_duration_count": len(boundary_points),
        "boundary_points": boundary_points,
        "intervals_by_duration": {str(k): v for k, v in intervals_by_duration.items()},
        "base_area": base_area,
        "total_area": total_area,
        "base_area_pct": (base_area / total_area * 100.0) if total_area else None,
        "reference_boundary_source": str(Path(boundary_path)),
    }


def _linspace(lo: float, hi: float, count: int) -> list[float]:
    if count <= 1:
        return [float(lo)]
    return [float(lo) + (float(hi) - float(lo)) * index / (count - 1) for index in range(count)]


def _duration_sample_grid(durations: list[int], count: int) -> list[int]:
    if count >= len(durations):
        return list(durations)
    if count <= 1:
        return [durations[0]]
    indices = [round(index * (len(durations) - 1) / (count - 1)) for index in range(count)]
    return [durations[index] for index in sorted(set(indices))]


def _baseline_samples(
    *,
    method: str,
    budget: int,
    durations: list[int],
    rate_range: tuple[float, float],
    seed: int,
) -> list[tuple[int, float]]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if method == "uniform":
        duration_count = max(1, min(len(durations), int(math.floor(math.sqrt(budget)))))
        selected_durations = _duration_sample_grid(durations, duration_count)
        rate_count = max(1, int(math.ceil(budget / len(selected_durations))))
        rates = _linspace(rate_range[0], rate_range[1], rate_count)
        samples = [(duration, rate) for duration in selected_durations for rate in rates]
        return samples[:budget]
    if method == "random":
        rng = random.Random(int(seed))
        return [
            (rng.choice(durations), rng.uniform(float(rate_range[0]), float(rate_range[1])))
            for _ in range(budget)
        ]
    raise ValueError(f"unsupported baseline method: {method}")


def _cell_bounds(values: Iterable[float], domain: tuple[float, float]) -> dict[float, tuple[float, float]]:
    unique = sorted({float(value) for value in values})
    if not unique:
        return {}
    bounds: dict[float, tuple[float, float]] = {}
    for index, value in enumerate(unique):
        lo = float(domain[0]) if index == 0 else (unique[index - 1] + value) / 2.0
        hi = float(domain[1]) if index == len(unique) - 1 else (value + unique[index + 1]) / 2.0
        bounds[value] = (max(float(domain[0]), lo), min(float(domain[1]), hi))
    return bounds


def _target_sample_cell_intervals(
    sample_rows: list[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[int, list[list[float]]]:
    durations = [int(value) for value in reference["duration_values"]]
    duration_range = (float(reference["duration_range"][0]), float(reference["duration_range"][1]))
    rate_range = (float(reference["rate_range"][0]), float(reference["rate_range"][1]))
    duration_bounds = _cell_bounds((row["duration"] for row in sample_rows), duration_range)
    rate_bounds = _cell_bounds((row["rate"] for row in sample_rows), rate_range)
    predicted: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in sample_rows:
        if not row["target"]:
            continue
        d_lo, d_hi = duration_bounds[float(row["duration"])]
        r_lo, r_hi = rate_bounds[float(row["rate"])]
        for duration in durations:
            if d_lo <= duration <= d_hi:
                predicted[duration].append((r_lo, r_hi))
    return {
        duration: _merge_intervals(intervals, rate_range)
        for duration, intervals in predicted.items()
    }


def _probe_task(
    *,
    hazard_driver: str,
    duration: int,
    rate: float,
    stop_ignore_predicates: list[str],
) -> dict[str, Any]:
    runtime_args: dict[str, Any] = {
        "steps": int(duration) + HOLD_STEPS,
        "stop_on_trip": True,
    }
    if stop_ignore_predicates:
        runtime_args["ignore_protection_predicates_for_stop"] = stop_ignore_predicates
    return {
        "runtime_args": runtime_args,
        "injection": {
            "point": hazard_driver,
            "amp": float(rate),
            "duration": int(duration),
            "t_start": 0,
            "role": "manip",
        },
    }


def _is_target_probe(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    task: dict[str, Any],
    *,
    hazard_id: str,
    alarm_id: str,
) -> bool:
    probe = dict(executor(task))
    if hasattr(executor, "rules"):
        probe["_rules"] = getattr(executor, "rules")
    return base_search._classify_with_ids(probe, hazard_id, alarm_id) == "target"


def _chunks(items: list[tuple[int, float]], chunk_count: int) -> list[list[tuple[int, float]]]:
    if chunk_count <= 1:
        return [items]
    chunks: list[list[tuple[int, float]]] = [[] for _ in range(min(chunk_count, len(items)))]
    for index, item in enumerate(items):
        chunks[index % len(chunks)].append(item)
    return [chunk for chunk in chunks if chunk]


def _baseline_process_chunk(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    executor = base_search.load_executor(job["manifest_path"])
    stop_ignore_predicates = base_search._stop_ignore_predicates(executor)
    rows = []
    for sample in job["samples"]:
        duration = int(sample["duration"])
        rate = float(sample["rate"])
        task = _probe_task(
            hazard_driver=str(job["hazard_driver"]),
            duration=duration,
            rate=rate,
            stop_ignore_predicates=stop_ignore_predicates,
        )
        rows.append(
            {
                "duration": duration,
                "rate": rate,
                "target": _is_target_probe(
                    executor,
                    task,
                    hazard_id=str(job["hazard_id"]),
                    alarm_id=str(job["alarm_id"]),
                ),
            }
        )
    return rows


def _reference_intervals(reference: Mapping[str, Any]) -> dict[int, list[list[float]]]:
    return {
        int(duration): [[float(lo), float(hi)] for lo, hi in intervals]
        for duration, intervals in reference["intervals_by_duration"].items()
    }


def _lower_boundary(intervals_by_duration: Mapping[int, list[list[float]]]) -> dict[int, float]:
    return {
        int(duration): min(float(lo) for lo, _hi in intervals)
        for duration, intervals in intervals_by_duration.items()
        if intervals
    }


def _metrics_from_prediction(
    *,
    reference: Mapping[str, Any],
    predicted: Mapping[int, list[list[float]]],
) -> dict[str, Any]:
    durations = [int(value) for value in reference["duration_values"]]
    ref = _reference_intervals(reference)
    ref_widths = {duration: _interval_width(ref.get(duration, [])) for duration in durations}
    pred_widths = {duration: _interval_width(predicted.get(duration, [])) for duration in durations}
    overlap_widths = {
        duration: _interval_overlap(ref.get(duration, []), predicted.get(duration, []))
        for duration in durations
    }
    overreach_widths = {
        duration: max(0.0, pred_widths[duration] - overlap_widths[duration])
        for duration in durations
    }
    ref_area = _integrate_values(durations, ref_widths)
    pred_area = _integrate_values(durations, pred_widths)
    overlap_area = _integrate_values(durations, overlap_widths)
    overreach_area = _integrate_values(durations, overreach_widths)
    ref_lower = _lower_boundary(ref)
    pred_lower = _lower_boundary(predicted)
    matched = sorted(set(ref_lower) & set(pred_lower))
    boundary_mae = (
        sum(abs(pred_lower[duration] - ref_lower[duration]) for duration in matched) / len(matched)
        if matched
        else None
    )
    return {
        "predicted_area": pred_area,
        "reference_area": ref_area,
        "overlap_area": overlap_area,
        "overreach_area": overreach_area,
        "recovered_base_region_ratio": (overlap_area / ref_area) if ref_area else None,
        "overreach_ratio": (overreach_area / ref_area) if ref_area else None,
        "boundary_mae": boundary_mae,
        "boundary_mae_matched_duration_count": len(matched),
    }


def _sample_boundary_metrics(
    sample_rows: list[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    ref_lower = _lower_boundary(_reference_intervals(reference))
    target_rates: dict[int, list[float]] = defaultdict(list)
    for row in sample_rows:
        if row["target"]:
            target_rates[int(row["duration"])].append(float(row["rate"]))
    pred_lower = {
        duration: min(rates)
        for duration, rates in target_rates.items()
        if rates
    }
    matched = sorted(set(ref_lower) & set(pred_lower))
    return {
        "boundary_mae": (
            sum(abs(pred_lower[duration] - ref_lower[duration]) for duration in matched) / len(matched)
            if matched
            else None
        ),
        "boundary_mae_matched_duration_count": len(matched),
    }


def run_budget_baseline(
    *,
    method: str,
    budget: int,
    reference: Mapping[str, Any],
    executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    manifest_path: str | Path | None = None,
    workers: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    durations = [int(value) for value in reference["duration_values"]]
    rate_range = (float(reference["rate_range"][0]), float(reference["rate_range"][1]))
    hazard_driver = str(reference["variable"])
    hazard_id = str(reference["target_hazard"])
    alarm_id = str(reference["base_alarm"])
    samples = _baseline_samples(
        method=method,
        budget=int(budget),
        durations=durations,
        rate_range=rate_range,
        seed=int(seed),
    )
    if manifest_path is not None and workers > 1:
        jobs = [
            {
                "manifest_path": str(manifest_path),
                "hazard_driver": hazard_driver,
                "hazard_id": hazard_id,
                "alarm_id": alarm_id,
                "samples": [{"duration": duration, "rate": rate} for duration, rate in chunk],
            }
            for chunk in _chunks(samples, int(workers))
        ]
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            sample_rows = [row for chunk in pool.map(_baseline_process_chunk, jobs) for row in chunk]
    else:
        if executor is None:
            if manifest_path is None:
                raise ValueError("provide executor or manifest_path")
            executor = base_search.load_executor(manifest_path)
        stop_ignore_predicates = base_search._stop_ignore_predicates(executor)
        sample_rows = []
        for duration, rate in samples:
            task = _probe_task(
                hazard_driver=hazard_driver,
                duration=duration,
                rate=rate,
                stop_ignore_predicates=stop_ignore_predicates,
            )
            is_target = _is_target_probe(executor, task, hazard_id=hazard_id, alarm_id=alarm_id)
            sample_rows.append({"duration": int(duration), "rate": float(rate), "target": is_target})
    predicted = _target_sample_cell_intervals(sample_rows, reference)
    metrics = _metrics_from_prediction(reference=reference, predicted=predicted)
    metrics.update(_sample_boundary_metrics(sample_rows, reference))
    return {
        "method": method,
        "budget": int(budget),
        "seed": int(seed),
        "target_hits": sum(1 for row in sample_rows if row["target"]),
        "sample_count": len(sample_rows),
        "reconstruction_policy": RECONSTRUCTION_POLICY,
        "predicted_intervals_by_duration": {str(k): v for k, v in predicted.items()},
        "samples": sample_rows,
        **metrics,
    }


def _default_boundary_path(root: Path) -> Path:
    return root / "results" / "stage2" / DEFAULT_PLATFORM / DEFAULT_VARIABLE / "boundary_results.json"


def _manifest_path(root: Path) -> Path:
    return root / "simulators" / DEFAULT_PLATFORM / "system_manifest.json"


def _default_executor_factory(manifest_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return base_search.load_executor(manifest_path)


def _write_config(path: Path, *, reference: Mapping[str, Any], budget: int | None, seed: int) -> None:
    lines = [
        "# RQ2 Evaluation Config",
        "",
        f"- platform: `{reference['platform']}`",
        f"- variable: `{reference['variable']}`",
        f"- target_hazard: `{reference['target_hazard']}`",
        f"- base_alarm: `{reference['base_alarm']}`",
        f"- search_domain_T: `{reference['duration_range']}`",
        f"- search_domain_K: `{reference['rate_range']}`",
        "- runtime_rule: `duration + 100`",
        "- replay_oracle: Stage 2 base target, hazard plus pre-hazard alarm",
        f"- reconstruction_policy: `{RECONSTRUCTION_POLICY}`",
        f"- CRAVE_budget: `{budget}`",
        f"- baseline_budget: `{budget}`",
        f"- random_seed: `{seed}`",
        f"- reference_boundary_source: `{reference['reference_boundary_source']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_main_table(path: Path, *, reference: Mapping[str, Any], baselines: list[Mapping[str, Any]], budget: int | None) -> None:
    lines = [
        "| Method | Replay budget | Recovered reference region | Boundary error / precision | Overreach ratio |",
        "|---|---:|---:|---:|---:|",
        f"| CRAVE reference | {_fmt(budget)} | reference | {reference['epsilon_K']:.1e} | 0 |",
    ]
    for baseline in baselines:
        name = "Uniform grid" if baseline["method"] == "uniform" else "Random search"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(baseline["budget"]),
                    _fmt(baseline["recovered_base_region_ratio"]),
                    _fmt(baseline["boundary_mae"]),
                    _fmt(baseline["overreach_ratio"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_boundary_files(
    *,
    reference_path: str | Path,
    predicted_path: str | Path,
) -> dict[str, Any]:
    reference = build_reference_summary(reference_path)
    predicted = build_reference_summary(predicted_path)
    predicted_intervals = {
        int(duration): intervals
        for duration, intervals in predicted["intervals_by_duration"].items()
    }
    metrics = _metrics_from_prediction(reference=reference, predicted=predicted_intervals)
    return {
        "reference_path": str(reference_path),
        "predicted_path": str(predicted_path),
        "reference_area": reference["base_area"],
        "predicted_area": predicted["base_area"],
        **metrics,
    }


def run_rq2_evaluation(
    *,
    root: str | Path,
    boundary_path: str | Path | None = None,
    budget: int | None = None,
    seed: int = DEFAULT_SEED,
    executor_factory: Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]] = _default_executor_factory,
    run_baselines: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    root = Path(root)
    rq2_dir = root / "results" / "rq2"
    boundary_path = Path(boundary_path) if boundary_path is not None else _default_boundary_path(root)
    reference = build_reference_summary(boundary_path)
    reference["crave_budget"] = budget
    _write_json(rq2_dir / "te_crave_reference.json", reference)
    _write_config(rq2_dir / "rq2_eval_config.md", reference=reference, budget=budget, seed=seed)
    baselines: list[dict[str, Any]] = []
    if run_baselines:
        if budget is None:
            raise ValueError("provide budget before running baselines")
        for method in ("uniform", "random"):
            use_process_baseline = executor_factory is _default_executor_factory and workers > 1
            executor = None if use_process_baseline else executor_factory(_manifest_path(root))
            baseline = run_budget_baseline(
                method=method,
                budget=int(budget),
                executor=executor,
                manifest_path=_manifest_path(root) if use_process_baseline else None,
                workers=int(workers),
                reference=reference,
                seed=int(seed),
            )
            baselines.append(baseline)
            _write_json(rq2_dir / f"te_ablation_{method}_budget{int(budget)}.json", baseline)
    comparison = {
        "platform": reference["platform"],
        "variable": reference["variable"],
        "budget": budget,
        "reference": {
            "base_area": reference["base_area"],
            "base_area_pct": reference["base_area_pct"],
            "valid_duration_count": reference["valid_duration_count"],
            "epsilon_K": reference["epsilon_K"],
        },
        "baselines": baselines,
    }
    _write_json(rq2_dir / "te_ablation_results.json", comparison)
    _write_main_table(rq2_dir / "rq2_main_table.md", reference=reference, baselines=baselines, budget=budget)
    return {"reference": reference, "comparison": comparison}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ2 budget-matched base-region evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--boundary-path")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=base_search.DEFAULT_WORKERS)
    parser.add_argument("--no-baselines", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_rq2_evaluation(
        root=args.root,
        boundary_path=args.boundary_path,
        budget=args.budget,
        seed=args.seed,
        run_baselines=not args.no_baselines,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_searching_hazard_driving_regions.base_region_search import (  # noqa: E402
    _make_task,
    build_search_domain,
    classify_result,
)
from src.stage4_executor.generic_runtime_executor import (  # noqa: E402
    load_injection_registry,
    load_rules_from_manifest,
    load_simulation_module,
    simulate,
)


MANIFEST = ROOT / "simulators" / "tennessee_eastman" / "system_manifest.json"
REFERENCE_FILE = ROOT / "results" / "stage2" / "te_base_boundary.json"
REFERENCE_SUMMARY = ROOT / "results" / "rq2" / "te_results.json"
UNIFORM_FILE = ROOT / "results" / "rq2" / "te_ablation_uniform_budget935.json"
RANDOM_FILE = ROOT / "results" / "rq2" / "te_ablation_random_budget935.json"
SUMMARY_FILE = ROOT / "results" / "rq2" / "te_ablation_results.json"

POINT_NAME = "simulation_xmv_07"
REQUIRED_ALARMS = ["P-TEP-SEP-LEVEL-TRACK"]
ALARM_CAP = 1
AMP_MIN = 0.0
AMP_MAX = 1.0
DURATION_STEP = 5
HOLD_STEPS = 100
SIM_STEPS = 1100
BUDGET = 935
RANDOM_SEED = 43


def load_truth_intervals(reference_file: Path = REFERENCE_FILE) -> dict[int, tuple[float, float]]:
    payload = json.loads(reference_file.read_text(encoding="utf-8"))
    out: dict[int, tuple[float, float]] = {}
    for item in payload["boundaries"]:
        if item.get("direction") != "pos":
            continue
        lo = item.get("lower_target_amp")
        hi = item.get("upper_target_amp")
        if lo is None or hi is None:
            continue
        out[int(item["duration"])] = (float(lo), float(hi))
    return out


def build_uniform_samples(
    durations: list[int],
    amp_min: float,
    amp_max: float,
    budget: int,
) -> list[tuple[int, float]]:
    if not durations:
        return []
    base_per_duration = budget // len(durations)
    if base_per_duration <= 0:
        raise ValueError("budget must be at least the number of durations")
    remainder = budget - base_per_duration * len(durations)
    base_levels = [
        amp_min + (amp_max - amp_min) * i / max(1, base_per_duration - 1)
        for i in range(base_per_duration)
    ]
    samples = [(dur, round(level, 10)) for dur in durations for level in base_levels]
    extra_levels = [0.125, 0.375, 0.625, 0.875]
    if remainder:
        idxs = [
            round(i * (len(durations) - 1) / max(1, remainder - 1))
            for i in range(remainder)
        ]
        for i, idx in enumerate(idxs):
            samples.append((durations[idx], extra_levels[i % len(extra_levels)]))
    return samples


def build_random_samples(
    durations: list[int],
    amp_min: float,
    amp_max: float,
    budget: int,
    seed: int,
) -> list[tuple[int, float]]:
    rng = Random(seed)
    return [
        (durations[rng.randrange(len(durations))], rng.uniform(amp_min, amp_max))
        for _ in range(budget)
    ]


def summarize_method_stats(
    raw: list[dict[str, Any]],
    truth_intervals: dict[int, tuple[float, float]],
    duration_step: int,
    amp_max: float,
) -> dict[str, Any]:
    found_lower: dict[int, float] = {}
    for item in raw:
        if item.get("verdict") != "target":
            continue
        dur = int(item["duration"])
        rate = float(item["rate"])
        cur = found_lower.get(dur)
        if cur is None or rate < cur:
            found_lower[dur] = rate

    area = sum((amp_max - lo) * duration_step for lo in found_lower.values())
    errors = []
    for dur, found in sorted(found_lower.items()):
        truth = truth_intervals.get(dur)
        if truth is None:
            continue
        errors.append(found - truth[0])
    mae = sum(abs(x) for x in errors) / len(errors) if errors else None
    max_err = max(abs(x) for x in errors) if errors else None
    return {
        "discovered_duration_count": len(found_lower),
        "discovered_lower_per_duration": found_lower,
        "discovered_area": round(area, 4),
        "boundary_mae": round(mae, 4) if mae is not None else None,
        "boundary_max_error": round(max_err, 4) if max_err is not None else None,
    }


def _build_context() -> dict[str, Any]:
    registry = load_injection_registry(MANIFEST)
    rules = load_rules_from_manifest(MANIFEST)
    sim_module, sim_init_kwargs = load_simulation_module(MANIFEST)
    domain = build_search_domain(
        registry=registry,
        point_name=POINT_NAME,
        duration_min=100,
        duration_max=1000,
        duration_step=DURATION_STEP,
    )
    alarm_cols = [spec["margin_col"] for spec in rules.get("alarms", {}).values()]
    return {
        "registry": registry,
        "rules": rules,
        "sim_module": sim_module,
        "sim_init_kwargs": sim_init_kwargs,
        "domain": domain,
        "alarm_cols": alarm_cols,
    }


def _simulate_one(ctx: dict[str, Any], duration: int, rate: float) -> dict[str, Any]:
    task = _make_task(ctx["domain"], float(rate), int(duration), SIM_STEPS, HOLD_STEPS)
    result = simulate(
        task,
        registry=ctx["registry"],
        rules=ctx["rules"],
        sim_module=ctx["sim_module"],
        sim_init_kwargs=ctx["sim_init_kwargs"],
    )
    score = result.get("score", {})
    verdict = classify_result(
        {
            "status": result.get("status", "ok"),
            "first_hazard_step": score.get("first_hazard_step"),
            "prehazard_alarm_rule_ids_count": score.get("prehazard_alarm_rule_ids_count", 0),
            "alarm_rule_ids": score.get("prehazard_alarm_rule_ids", []),
        },
        alarm_cols=ctx["alarm_cols"],
        alarm_cap=ALARM_CAP,
        required_alarm_ids=REQUIRED_ALARMS,
    )
    return {
        "duration": int(duration),
        "rate": float(rate),
        "first_hazard_step": score.get("first_hazard_step"),
        "pre_count": score.get("prehazard_alarm_rule_ids_count", 0),
        "pre_ids": score.get("prehazard_alarm_rule_ids", []),
        "verdict": verdict,
    }


def run_method(
    method: str,
    samples: list[tuple[int, float]],
    out_file: Path,
    workers: int = 8,
) -> dict[str, Any]:
    ctx = _build_context()
    t0 = time.time()
    raw: list[dict[str, Any]] = [None] * len(samples)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_map = {
            pool.submit(_simulate_one, ctx, dur, rate): idx
            for idx, (dur, rate) in enumerate(samples)
        }
        for fut in as_completed(future_map):
            raw[future_map[fut]] = fut.result()
    elapsed = round(time.time() - t0, 2)
    counts = {
        "hazard_triggered": sum(1 for r in raw if r["first_hazard_step"] is not None),
        "target": sum(1 for r in raw if r["verdict"] == "target"),
        "unsafe": sum(1 for r in raw if r["verdict"] == "unsafe"),
        "safe": sum(1 for r in raw if r["verdict"] == "safe"),
    }
    out = {
        "method": method,
        "budget": len(samples),
        "elapsed_s": elapsed,
        "alarm_settings": {
            "ALARM_ACFEED_ABS": "1.0",
            "alarm_cap": ALARM_CAP,
            "required_alarm_ids": REQUIRED_ALARMS,
            "forbidden_alarm_ids": [],
        },
        "domain": {
            "duration_range": [100, 1000],
            "duration_step": DURATION_STEP,
            "amp_range": [AMP_MIN, AMP_MAX],
            "hold_steps": HOLD_STEPS,
            "seed": RANDOM_SEED if method == "random_search" else None,
        },
        "counts": counts,
        "rates": {
            "hazard_rate": round(counts["hazard_triggered"] / len(samples), 4),
            "target_rate": round(counts["target"] / len(samples), 4),
        },
        "raw": raw,
    }
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    truth_intervals = load_truth_intervals()
    durations = sorted(truth_intervals.keys())
    uniform_samples = build_uniform_samples(durations, AMP_MIN, AMP_MAX, BUDGET)
    random_samples = build_random_samples(durations, AMP_MIN, AMP_MAX, BUDGET, RANDOM_SEED)
    uniform = run_method("uniform_grid", uniform_samples, UNIFORM_FILE)
    random = run_method("random_search", random_samples, RANDOM_FILE)
    uniform_stats = summarize_method_stats(uniform["raw"], truth_intervals, DURATION_STEP, AMP_MAX)
    random_stats = summarize_method_stats(random["raw"], truth_intervals, DURATION_STEP, AMP_MAX)
    ref = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
    total_area = float(ref["parameter_space"]["total_area"])
    out = {
        "reference": {
            "budget": ref["rq2_table"]["total_sims"],
            "base_area": ref["rq2_table"]["base_area"],
            "base_area_pct": ref["rq2_table"]["base_area_pct"],
            "boundary_precision": 1e-4,
        },
        "uniform_grid": {
            "budget": BUDGET,
            "discovered_area": uniform_stats["discovered_area"],
            "discovered_area_pct": round(uniform_stats["discovered_area"] / total_area * 100.0, 4),
            "discovered_duration_count": uniform_stats["discovered_duration_count"],
            "boundary_mae": uniform_stats["boundary_mae"],
            "boundary_max_error": uniform_stats["boundary_max_error"],
            "target_hits": uniform["counts"]["target"],
            "elapsed_s": uniform["elapsed_s"],
        },
        "random_search": {
            "budget": BUDGET,
            "seed": RANDOM_SEED,
            "discovered_area": random_stats["discovered_area"],
            "discovered_area_pct": round(random_stats["discovered_area"] / total_area * 100.0, 4),
            "discovered_duration_count": random_stats["discovered_duration_count"],
            "boundary_mae": random_stats["boundary_mae"],
            "boundary_max_error": random_stats["boundary_max_error"],
            "target_hits": random["counts"]["target"],
            "elapsed_s": random["elapsed_s"],
        },
        "evaluation_policy": {
            "region_model": "upward_union_per_duration_from_min_target_rate",
            "outside_truth_not_counted_by_reference_comparison": True,
        },
    }
    SUMMARY_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

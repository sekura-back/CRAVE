from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable


HOLD_STEPS = 100
SLOPE_MAX = 0.12
SLOPE_GRID_SAMPLES = 31
SLOPE_BISECT_ROUNDS = 8
COOPERATIVE_SLOPE_FRACTIONS = (1.0, 0.75, 0.5, 0.25)
EXTREME_BAND_SAMPLES = 11
METHOD = "cooperative_dual_ramp_hold_boundary"


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


def resolve_masking_variable(extraction: Mapping[str, Any], alarm_id: str) -> str:
    for rule in extraction.get("P", []):
        if rule.get("id") != alarm_id:
            continue
        value = str(rule.get("setpoint_var", "")).strip()
        if not value:
            raise ValueError(f"alarm rule missing setpoint_var: {alarm_id}")
        return value
    raise KeyError(alarm_id)


def load_executor(manifest_path: str | Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    path = Path(manifest_path)
    module = importlib.import_module(f"simulators.{path.parent.name}.runtime_executor")
    return module.create_executor(path)


def _signed_amp(direction: str, magnitude: float) -> float:
    if direction == "pos":
        return float(magnitude)
    if direction == "neg":
        return -float(magnitude)
    raise ValueError(f"unsupported direction: {direction}")


def make_dual_task(
    *,
    driver_point: str,
    driver_amp: float,
    mask_point: str,
    mask_slope: float,
    duration: int,
) -> dict[str, Any]:
    duration = int(duration)
    return {
        "runtime_args": {"steps": duration + HOLD_STEPS},
        "injection": {
            "points": [
                {
                    "point": driver_point,
                    "amp": float(driver_amp),
                    "duration": duration,
                    "t_start": 0,
                    "role": "manip",
                },
                {
                    "point": mask_point,
                    "amp": float(mask_slope),
                    "duration": duration,
                    "t_start": 0,
                    "role": "mask",
                },
            ]
        },
    }


def probe_feasible(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    driver_point: str,
    driver_amp: float,
    mask_point: str,
    mask_slope: float,
    duration: int,
    driver_direction: str = "pos",
) -> tuple[bool, int | None]:
    result = executor(
        make_dual_task(
            driver_point=driver_point,
            driver_amp=_signed_amp(driver_direction, driver_amp),
            mask_point=mask_point,
            mask_slope=mask_slope,
            duration=duration,
        )
    )
    score = result.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("executor result missing score")
    hazard_step = score.get("first_hazard_step")
    feasible = hazard_step is not None and int(score.get("prehazard_alarm_rule_ids_count", 0)) == 0
    return feasible, int(hazard_step) if hazard_step is not None else None


def _slope_grid() -> list[float]:
    if SLOPE_GRID_SAMPLES < 2:
        return [0.0]
    return [SLOPE_MAX * index / (SLOPE_GRID_SAMPLES - 1) for index in range(SLOPE_GRID_SAMPLES)]


def find_slope_band(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    driver_point: str,
    driver_amp: float,
    mask_point: str,
    duration: int,
    driver_direction: str = "pos",
) -> tuple[float, float, int] | None:
    probes = _slope_grid()
    outcomes: list[bool] = []
    calls = 0
    for slope in probes:
        feasible, _hazard_step = probe_feasible(
            executor,
            driver_point=driver_point,
            driver_amp=driver_amp,
            driver_direction=driver_direction,
            mask_point=mask_point,
            mask_slope=slope,
            duration=duration,
        )
        calls += 1
        outcomes.append(feasible)

    best_start: int | None = None
    best_end: int | None = None
    current_start: int | None = None
    for index, feasible in enumerate(outcomes):
        if feasible and current_start is None:
            current_start = index
        if current_start is not None and (not feasible or index == len(outcomes) - 1):
            current_end = index if feasible else index - 1
            if best_start is None or (current_end - current_start) > (best_end - best_start):
                best_start = current_start
                best_end = current_end
            current_start = None

    if best_start is None or best_end is None:
        return None

    left = probes[best_start]
    right = probes[best_end]

    if best_start > 0:
        fail = probes[best_start - 1]
        ok = left
        for _ in range(SLOPE_BISECT_ROUNDS):
            mid = (fail + ok) / 2.0
            feasible, _hazard_step = probe_feasible(
                executor,
                driver_point=driver_point,
                driver_amp=driver_amp,
                driver_direction=driver_direction,
                mask_point=mask_point,
                mask_slope=mid,
                duration=duration,
            )
            calls += 1
            if feasible:
                ok = mid
            else:
                fail = mid
        left = ok

    if best_end < len(probes) - 1:
        ok = right
        fail = probes[best_end + 1]
        for _ in range(SLOPE_BISECT_ROUNDS):
            mid = (ok + fail) / 2.0
            feasible, _hazard_step = probe_feasible(
                executor,
                driver_point=driver_point,
                driver_amp=driver_amp,
                driver_direction=driver_direction,
                mask_point=mask_point,
                mask_slope=mid,
                duration=duration,
            )
            calls += 1
            if feasible:
                ok = mid
            else:
                fail = mid
        right = ok

    if right < left:
        return None
    return left, right, calls


def find_min_amp(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    driver_point: str,
    mask_point: str,
    slope: float,
    duration: int,
    amp_lo: float,
    amp_hi: float,
    amp_tol: float,
    max_bisect: int,
    hi_feasible: bool = False,
    driver_direction: str = "pos",
) -> tuple[float | None, int]:
    calls = 0
    if not hi_feasible:
        feasible_hi, _hazard_step = probe_feasible(
            executor,
            driver_point=driver_point,
            driver_amp=amp_hi,
            driver_direction=driver_direction,
            mask_point=mask_point,
            mask_slope=slope,
            duration=duration,
        )
        calls += 1
        if not feasible_hi:
            return None, calls
    lo = float(amp_lo)
    hi = float(amp_hi)
    for _ in range(int(max_bisect)):
        if hi - lo <= amp_tol:
            break
        mid = (lo + hi) / 2.0
        feasible, _hazard_step = probe_feasible(
            executor,
            driver_point=driver_point,
            driver_amp=mid,
            driver_direction=driver_direction,
            mask_point=mask_point,
            mask_slope=slope,
            duration=duration,
        )
        calls += 1
        if feasible:
            hi = mid
        else:
            lo = mid
    return hi, calls


def _pick_working_slope_and_candidate(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    driver_point: str,
    mask_point: str,
    duration: int,
    amp_cur: float,
    slope_min: float,
    slope_max: float,
    amp_lo: float,
    amp_tol: float,
    max_bisect: int,
    driver_direction: str = "pos",
) -> tuple[dict[str, Any] | None, int]:
    width = float(slope_max) - float(slope_min)
    if width < 0:
        return None, 0
    calls = 0
    candidates: list[dict[str, Any]] = []
    seen: set[float] = set()
    for fraction in COOPERATIVE_SLOPE_FRACTIONS:
        slope = float(slope_min) + width * float(fraction)
        key = round(slope, 12)
        if key in seen:
            continue
        seen.add(key)
        amp_next, amp_calls = find_min_amp(
            executor,
            driver_point=driver_point,
            mask_point=mask_point,
            slope=slope,
            duration=duration,
            amp_lo=amp_lo,
            amp_hi=amp_cur,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
            driver_direction=driver_direction,
        )
        calls += amp_calls
        if amp_next is None:
            continue
        candidates.append(
            {
                "working_slope": slope,
                "amp_next": amp_next,
            }
        )
    for candidate in sorted(candidates, key=lambda item: item["amp_next"]):
        next_band = find_slope_band(
            executor,
            driver_point=driver_point,
            driver_amp=float(candidate["amp_next"]),
            driver_direction=driver_direction,
            mask_point=mask_point,
            duration=duration,
        )
        if next_band is None:
            continue
        next_min, next_max, band_calls = next_band
        calls += band_calls
        candidate["next_band"] = (next_min, next_max)
        return candidate, calls
    return None, calls


def scan_extreme(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    driver_point: str,
    mask_point: str,
    duration: int,
    stable_amp: float,
    band: tuple[float, float],
    amp_lo: float,
    amp_tol: float,
    max_bisect: int,
    driver_direction: str = "pos",
) -> tuple[dict[str, Any] | None, int]:
    slope_min, slope_max = float(band[0]), float(band[1])
    if slope_max < slope_min:
        return None, 0
    if slope_max == slope_min or EXTREME_BAND_SAMPLES <= 1:
        slopes = [slope_min]
    else:
        slopes = [
            slope_min + (slope_max - slope_min) * index / (EXTREME_BAND_SAMPLES - 1)
            for index in range(EXTREME_BAND_SAMPLES)
        ]
    calls = 0
    best: dict[str, Any] | None = None
    for slope in slopes:
        amp_hi = stable_amp
        hi_feasible = False
        if best is not None:
            current_best_amp = float(best["extreme_amp"])
            if current_best_amp <= amp_lo:
                continue
            feasible, _hazard_step = probe_feasible(
                executor,
                driver_point=driver_point,
                driver_amp=current_best_amp,
                driver_direction=driver_direction,
                mask_point=mask_point,
                mask_slope=slope,
                duration=duration,
            )
            calls += 1
            if not feasible:
                continue
            amp_hi = current_best_amp
            hi_feasible = True
        amp_min, amp_calls = find_min_amp(
            executor,
            driver_point=driver_point,
            mask_point=mask_point,
            slope=slope,
            duration=duration,
            amp_lo=amp_lo,
            amp_hi=amp_hi,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
            hi_feasible=hi_feasible,
            driver_direction=driver_direction,
        )
        calls += amp_calls
        if amp_min is None:
            continue
        if best is None or amp_min < best["extreme_amp"]:
            best = {
                "extreme_amp": amp_min,
                "extreme_slope": slope,
                "extreme_slope_range": [slope, slope],
            }
    return best, calls


def expand_one_duration(
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    point: str,
    mask_point: str,
    duration: int,
    base_amp: float,
    amp_tol: float,
    max_bisect: int,
    driver_direction: str = "pos",
) -> tuple[dict[str, Any] | None, int]:
    band = find_slope_band(
        executor,
        driver_point=point,
        driver_amp=base_amp,
        driver_direction=driver_direction,
        mask_point=mask_point,
        duration=duration,
    )
    if band is None:
        return None, SLOPE_GRID_SAMPLES
    slope_min, slope_max, calls = band
    amp_lo = 0.0
    amp_cur = float(base_amp)
    best_amp = amp_cur
    best_band = (slope_min, slope_max)
    current_band = (slope_min, slope_max)
    for _ in range(int(max_bisect)):
        candidate, candidate_calls = _pick_working_slope_and_candidate(
            executor,
            driver_point=point,
            mask_point=mask_point,
            duration=duration,
            amp_cur=amp_cur,
            slope_min=current_band[0],
            slope_max=current_band[1],
            amp_lo=amp_lo,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
            driver_direction=driver_direction,
        )
        calls += candidate_calls
        if candidate is None:
            break
        amp_next = float(candidate["amp_next"])
        next_band = candidate["next_band"]
        if amp_next < best_amp:
            best_amp = amp_next
            best_band = next_band
        if amp_cur - amp_next < amp_tol:
            break
        amp_cur = amp_next
        current_band = next_band

    stable_amp = best_amp
    slope_min, slope_max = best_band
    extreme, extreme_calls = scan_extreme(
        executor,
        driver_point=point,
        mask_point=mask_point,
        duration=duration,
        stable_amp=stable_amp,
        band=(slope_min, slope_max),
        amp_lo=amp_lo,
        amp_tol=amp_tol,
        max_bisect=max_bisect,
        driver_direction=driver_direction,
    )
    calls += extreme_calls
    extreme_slope_range = (
        extreme["extreme_slope_range"] if extreme is not None else [slope_min, slope_max]
    )
    return {
        "stable_amp": stable_amp,
        "stable_slope_range": [slope_min, slope_max],
        "extreme_slope_range": extreme_slope_range,
        "extreme": extreme,
    }, calls


def _paths(output_root: str | Path, platform: str, hazard_driver: str) -> tuple[Path, Path]:
    root = Path(output_root)
    log_path = root / "results" / "logs" / "stage2_cond" / platform / hazard_driver / "events.jsonl"
    manifest_path = root / "results" / "manifests" / "stage2_cond" / platform / hazard_driver / "run_manifest.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path, manifest_path


def _log(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


def _require_rule_ids(hazard_id: str, alarm_id: str) -> None:
    if not str(hazard_id).startswith("H-") or not str(alarm_id).startswith("A-"):
        raise ValueError("Stage 2 rule IDs must use H-/A- prefixes")


def _base_context(base: Mapping[str, Any]) -> tuple[str, str, list[Mapping[str, Any]]]:
    boundaries = base.get("boundaries")
    if not isinstance(boundaries, list):
        raise ValueError("base boundary JSON missing boundaries[]")
    hazard_driver = str(base.get("hazard_driver") or "").strip()
    hazard_id = str(base.get("hazard_id") or "").strip()
    if boundaries:
        hazard_driver = hazard_driver or str(boundaries[0].get("hazard_driver", "")).strip()
        hazard_id = hazard_id or str(boundaries[0].get("hazard_id", "")).strip()
    if not hazard_driver or not hazard_id:
        raise ValueError("base boundary JSON missing hazard_driver or hazard_id")
    return hazard_driver, hazard_id, boundaries


def _conditional_process_job(job: Mapping[str, Any]) -> dict[str, Any]:
    executor = load_executor(job["manifest_path"])
    expansion, calls = expand_one_duration(
        executor,
        point=str(job["hazard_driver"]),
        mask_point=str(job["mask_point"]),
        duration=int(job["duration"]),
        base_amp=float(job["base_amp"]),
        amp_tol=float(job["amp_tol"]),
        max_bisect=int(job["max_bisect"]),
        driver_direction=str(job["direction"]),
    )
    if expansion is None:
        return {"point": None, "calls": int(calls)}
    direction = str(job["direction"])
    point = {
        "point_id": str(job["point_id"]),
        "hazard_driver": str(job["hazard_driver"]),
        "hazard_id": str(job["hazard_id"]),
        "alarm_id": str(job["alarm_id"]),
        "driver_direction": direction,
        "duration": int(job["duration"]),
        "runtime_steps": int(job["runtime_steps"]),
        "driver_amp_range": [float(value) for value in job["driver_amp_range"]],
        "coordinated_lower_amp": _signed_amp(direction, expansion["stable_amp"]),
        "coordinated_extreme_amp": _signed_amp(
            direction,
            (expansion["extreme"] or {}).get("extreme_amp", expansion["stable_amp"]),
        ),
        "stable_slope_range": expansion["stable_slope_range"],
        "extreme_slope_range": expansion["extreme_slope_range"],
        "masking_variable": str(job["mask_point"]),
    }
    return {"point": point, "calls": int(calls)}


def run_conditional_search(
    *,
    boundary_path: str | Path,
    extraction_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    output_root: str | Path,
    platform: str,
    direction: str = "pos",
    duration_step: int = 10,
    amp_tol: float = 0.005,
    max_bisect: int = 15,
    seed: int = 460,
    workers: int = 1,
    executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    del duration_step
    if direction not in {"pos", "neg"}:
        raise ValueError("stage2 conditional search only supports pos/neg direction")
    base = _read_json(boundary_path)
    extraction = _read_json(extraction_path)
    hazard_driver, hazard_id, boundaries = _base_context(base)
    log_path, run_manifest_path = _paths(output_root, platform, hazard_driver)
    _log(log_path, {"event": "start", "stage": "stage2_cond", "platform": platform, "hazard_driver": hazard_driver})
    points: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    probe_count = 0

    for boundary_index, boundary in enumerate(boundaries):
        if str(boundary.get("direction", direction)) != direction:
            continue
        boundary_hazard_id = boundary.get("hazard_id", hazard_id)
        duration = int(boundary["duration"])
        runtime_steps = int(boundary["runtime_steps"])
        if runtime_steps != duration + HOLD_STEPS:
            raise ValueError("runtime_steps must equal duration + 100")
        intervals = boundary.get("target_intervals")
        if not isinstance(intervals, list):
            raise ValueError("base boundary missing target_intervals[]")
        for interval_index, interval in enumerate(intervals):
            alarm_id = str(interval.get("alarm_id") or boundary.get("alarm_id", "")).strip()
            if not alarm_id:
                raise ValueError("base target interval missing alarm_id")
            mask_point = resolve_masking_variable(extraction, alarm_id)
            interval_hazard_id = str(interval.get("first_hazard_id") or boundary_hazard_id).strip()
            _require_rule_ids(interval_hazard_id, alarm_id)
            driver_amp_range = [float(interval["rate_lo"]), float(interval["rate_hi"])]
            jobs.append(
                {
                    "point_id": f"{hazard_driver}_{boundary_index}_{interval_index}",
                    "manifest_path": str(manifest_path),
                    "hazard_driver": str(boundary.get("hazard_driver", hazard_driver)),
                    "hazard_id": interval_hazard_id,
                    "alarm_id": alarm_id,
                    "direction": direction,
                    "duration": duration,
                    "runtime_steps": runtime_steps,
                    "driver_amp_range": driver_amp_range,
                    "base_amp": driver_amp_range[1],
                    "mask_point": mask_point,
                    "amp_tol": float(amp_tol),
                    "max_bisect": int(max_bisect),
                }
            )

    if executor is not None or int(workers) <= 1:
        local_executor = executor or load_executor(manifest_path)
        for job in jobs:
            expansion, calls = expand_one_duration(
                local_executor,
                point=str(job["hazard_driver"]),
                mask_point=str(job["mask_point"]),
                duration=int(job["duration"]),
                base_amp=float(job["base_amp"]),
                amp_tol=float(amp_tol),
                max_bisect=int(max_bisect),
                driver_direction=direction,
            )
            probe_count += calls
            if expansion is None:
                continue
            points.append(
                {
                    "point_id": str(job["point_id"]),
                    "hazard_driver": str(job["hazard_driver"]),
                    "hazard_id": str(job["hazard_id"]),
                    "alarm_id": str(job["alarm_id"]),
                    "driver_direction": direction,
                    "duration": int(job["duration"]),
                    "runtime_steps": int(job["runtime_steps"]),
                    "driver_amp_range": list(job["driver_amp_range"]),
                    "coordinated_lower_amp": _signed_amp(direction, expansion["stable_amp"]),
                    "coordinated_extreme_amp": _signed_amp(
                        direction,
                        (expansion["extreme"] or {}).get("extreme_amp", expansion["stable_amp"]),
                    ),
                    "stable_slope_range": expansion["stable_slope_range"],
                    "extreme_slope_range": expansion["extreme_slope_range"],
                    "masking_variable": str(job["mask_point"]),
                }
            )
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for result in pool.map(_conditional_process_job, jobs):
                probe_count += int(result["calls"])
                if result["point"] is not None:
                    points.append(result["point"])

    payload = {
        "platform": platform,
        "hazard_id": hazard_id,
        "hazard_driver": hazard_driver,
        "method": METHOD,
        "points": points,
        "summary": {"seed": int(seed), "probe_count": probe_count, "duration_rule": "duration + 100"},
    }
    _write_json(output_path, payload)
    _write_json(
        run_manifest_path,
        {
            "stage": "stage2_cond",
            "platform": platform,
            "hazard_driver": hazard_driver,
            "direction": direction,
            "seed": int(seed),
            "boundary_path": str(boundary_path),
            "extraction_path": str(extraction_path),
            "manifest_path": str(manifest_path),
            "output_path": str(output_path),
            "log_path": str(log_path),
            "method": METHOD,
            "duration_rule": "duration + 100",
        },
    )
    _log(log_path, {"event": "finish", "stage": "stage2_cond", "point_count": len(points), "probe_count": probe_count})
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-path", required=True)
    parser.add_argument("--extraction-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--direction", default="pos")
    parser.add_argument("--duration-step", type=int, default=10)
    parser.add_argument("--amp-tol", type=float, default=0.005)
    parser.add_argument("--max-bisect", type=int, default=15)
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_conditional_search(
        boundary_path=args.boundary_path,
        extraction_path=args.extraction_path,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        output_root=args.output_root,
        platform=args.platform,
        direction=args.direction,
        duration_step=args.duration_step,
        amp_tol=args.amp_tol,
        max_bisect=args.max_bisect,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

# Conditional Extension - Section 3.3.2

"""Conditional-extension workflow for Stage 2 hazard-region search.

The method implemented here is intended to remain adaptable across
platforms: it expands a base hazard-driving boundary by jointly searching
the manipulated variable and an alarm-masking hidden variable under the
same ramp-plus-hold injection rule.

In the current artifact, the platform-specific bindings still come from
the Boiler closed-loop environment.  Those bindings are intentionally kept
inside this single script so the Stage 2 method stays intact while the
hidden-point, alarm-margin, and controller-phase mappings can be replaced
for other platforms without redefining the conditional-extension workflow.
"""

from __future__ import annotations



import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import sys

import time

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple



_ROOT = Path(__file__).resolve().parents[2]

for _p in [
    str(_ROOT),
    str(_ROOT / "src"),
    str(_ROOT / "src" / "stage4_executor"),
    str(_ROOT / "simulators"),
]:

    if _p not in sys.path:

        sys.path.insert(0, _p)

PLATFORM_BINDING_MODE = "in_script_mappings"
PLATFORM_NAME = "boiler_ccs"
IGNORE_PREDS = ["BOILER_CUT_TH", "FUEL_CUT_TH", "WATER_CUT_TH"]
HOLD = 100
DIRECT_COND_SLOPE_MAX = 0.12
DIRECT_COND_GRID_SAMPLES = 31
DIRECT_COND_BISECT_ROUNDS = 8
COOPERATIVE_SLOPE_FRACTIONS = (1.0, 0.75, 0.5, 0.25)
EXTREME_BAND_SAMPLES = 11

# Boiler-specific bindings are centralized here so the Stage 2 method stays
# single-file and result-stable while keeping the platform surface explicit.
HIDDEN_POINT_ALARM_BINDINGS = [
    (("fuel", "steam", "pressure", "boiler"), "steam_setpoint", "alarm_boiler_cut_margin"),
    (("water",), "water_setpoint", "alarm_water_cut_margin"),
]

PHASE_FIELD_BINDINGS = [
    (("fuel_command", "fuel"), (), ("controller_fuel_command", "fuel_command")),
    (("water_pump", "water_pump_speed"), (), ("controller_water_pump", "water_pump_speed")),
    (("steam_setpoint",), ("steam_setpoint",), ("controller_steam_hslim", "steam_setpoint")),
    (("water_setpoint",), ("water_setpoint",), ("controller_water_setpoint", "water_setpoint")),
    (("boiler_setpoint",), (), ("controller_boiler_setpoint", "boiler_setpoint")),
]

@dataclass
class BoundaryPoint:
    point: str
    direction: str
    duration: int
    amplitude: float
    source_field: str = "lower_target_amp"
    boundary_type: str = "unknown"
    converged: bool = True


def _resolve_hidden_point_and_alarm(point: str, direction: str) -> Tuple[str, str]:
    del direction
    p = str(point).strip().lower()
    for point_tokens, hidden_point, alarm_col in HIDDEN_POINT_ALARM_BINDINGS:
        if any(token in p for token in point_tokens):
            return hidden_point, alarm_col
    return "steam_setpoint", "alarm_boiler_cut_margin"


def _resolve_phase_and_field(point: str) -> Optional[Tuple[str, str]]:
    p = str(point).strip().lower()
    for point_tokens, point_names, phase_field in PHASE_FIELD_BINDINGS:
        if p in point_names or any(token in p for token in point_tokens):
            return phase_field
    return None


def _resolve_phase_and_field_for_hidden(point: str) -> Optional[Tuple[str, str]]:
    return _resolve_phase_and_field(point)


def _load_current_boiler_sim():
    from simulators.boiler_ccs.simulation import ClosedLoopSim as CurrentClosedLoopSim

    return CurrentClosedLoopSim


@lru_cache(maxsize=1)
def _load_runtime_registry():
    from generic_runtime_executor import load_injection_registry

    manifest = _ROOT / "simulators" / "boiler_ccs" / "system_manifest.json"
    return load_injection_registry(manifest)


def _build_runtime_hook(point_specs, registry, runtime_args):
    from generic_runtime_executor import build_injection_hook

    return build_injection_hook(
        point_specs=point_specs,
        registry=registry,
        runtime_args=runtime_args,
    )


def _runtime_point_name(point: str, *, hidden: bool = False) -> str:
    p = str(point).strip().lower()
    if hidden:
        if p == "steam_setpoint":
            return "simulation_steam_setpoint"
        if p == "water_setpoint":
            return "simulation_water_setpoint"
        return f"simulation_{p}"
    if p.startswith("simulation_"):
        return point
    if p == "fuel_command":
        return "simulation_fuel_command"
    if p == "water_pump_speed":
        return "simulation_water_pump_speed"
    return point


def _run_dual_ramp_probe(
    bp: BoundaryPoint,
    hidden_point: str,
    hidden_slope: float,
    sim_steps: int,
    registry,
) -> Tuple[bool, int, Optional[int]]:
    """Run dual ramp+hold injections using artifact runtime executor bindings."""

    ClosedLoopSimCurrent = _load_current_boiler_sim()
    hook = _build_runtime_hook(
        point_specs=[
            {
                "point": _runtime_point_name(bp.point),
                "amp": float(bp.amplitude),
                "duration": int(bp.duration),
                "t_start": 0,
                "role": "manip",
            },
            {
                "point": _runtime_point_name(hidden_point, hidden=True),
                "amp": float(hidden_slope),
                "duration": int(bp.duration),
                "t_start": 0,
                "role": "mask",
            },
        ],
        registry=registry,
        runtime_args={"steps": sim_steps},
    )

    sim = ClosedLoopSimCurrent(Ts=0.1, enable_csv=False)
    try:
        trace, _ = sim.run(
            steps=sim_steps,
            injection_hook=hook,
            return_trace=True,
            stop_on_trip=True,
            ignore_protection_predicates_for_stop=IGNORE_PREDS,
            ignore_protection_predicates_for_enforcement=IGNORE_PREDS,
        )
    finally:
        sim.close()

    hazard_step = None
    for row in trace:
        if (
            row.get("hazard_pressure_triggered")
            or row.get("hazard_waterwall_triggered")
            or row.get("hazard_feedwater_triggered")
        ):
            hazard_step = int(row["t_step"])
            break

    alarm_types = set()
    limit = hazard_step if hazard_step is not None else len(trace)
    for row in trace[:limit]:
        if row.get("boiler_cut"):
            alarm_types.add("boiler_cut")
        if row.get("fuel_cut"):
            alarm_types.add("fuel_cut")
        if row.get("water_cut"):
            alarm_types.add("water_cut")

    return hazard_step is not None, len(alarm_types), hazard_step


def _has_direct_cond_feasible_slope(
    bp: BoundaryPoint,
    hidden_point: str,
    sim_steps: int,
    registry,
) -> Tuple[bool, Optional[float], int]:
    """Check whether any ramp+hold masking slope yields hazard without alarm."""

    slope_values = [
        i * DIRECT_COND_SLOPE_MAX / (DIRECT_COND_GRID_SAMPLES - 1)
        for i in range(DIRECT_COND_GRID_SAMPLES)
    ]
    probe_calls = 0
    for slope in slope_values:
        probe_calls += 1
        hazard, alarm_count, _ = _run_dual_ramp_probe(
            bp,
            hidden_point,
            slope,
            sim_steps,
            registry,
        )
        if hazard and alarm_count == 0:
            return True, slope, probe_calls
    return False, None, probe_calls


def solve_slope_band_for_amp(
    bp: BoundaryPoint,
    hidden_point: str,
    sim_steps: int,
    registry,
    *,
    slope_lo: float = 0.0,
    slope_hi: float = DIRECT_COND_SLOPE_MAX,
    grid_samples: int = DIRECT_COND_GRID_SAMPLES,
    bisect_rounds: int = DIRECT_COND_BISECT_ROUNDS,
) -> Optional[Tuple[float, float, int]]:
    """Return the RA-feasible slope interval for one amplitude probe."""

    if slope_hi <= slope_lo:
        return None
    if grid_samples <= 1:
        probes = [float(slope_lo)]
    else:
        probes = [
            float(slope_lo + i * (slope_hi - slope_lo) / (grid_samples - 1))
            for i in range(grid_samples)
        ]

    outcomes: List[bool] = []
    probe_calls = 0
    for slope in probes:
        probe_calls += 1
        hazard, alarm_count, _ = _run_dual_ramp_probe(
            bp,
            hidden_point,
            slope,
            sim_steps,
            registry,
        )
        outcomes.append(hazard and alarm_count == 0)

    best_start = None
    best_end = None
    cur_start = None
    for idx, ok in enumerate(outcomes):
        if ok and cur_start is None:
            cur_start = idx
        if (not ok or idx == len(outcomes) - 1) and cur_start is not None:
            cur_end = idx if ok and idx == len(outcomes) - 1 else idx - 1
            if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = None

    if best_start is None or best_end is None:
        return None

    left = probes[best_start]
    right = probes[best_end]

    if best_start > 0:
        left_fail = probes[best_start - 1]
        left_ok = left
        for _ in range(bisect_rounds):
            mid = 0.5 * (left_fail + left_ok)
            probe_calls += 1
            hazard, alarm_count, _ = _run_dual_ramp_probe(
                bp,
                hidden_point,
                mid,
                sim_steps,
                registry,
            )
            if hazard and alarm_count == 0:
                left_ok = mid
            else:
                left_fail = mid
        left = left_ok

    if best_end < len(probes) - 1:
        right_ok = right
        right_fail = probes[best_end + 1]
        for _ in range(bisect_rounds):
            mid = 0.5 * (right_ok + right_fail)
            probe_calls += 1
            hazard, alarm_count, _ = _run_dual_ramp_probe(
                bp,
                hidden_point,
                mid,
                sim_steps,
                registry,
            )
            if hazard and alarm_count == 0:
                right_ok = mid
            else:
                right_fail = mid
        right = right_ok

    if right <= left:
        return None
    return left, right, probe_calls


def find_min_amp_under_fixed_slope(
    *,
    point: str,
    direction: str,
    duration: int,
    hidden_point: str,
    slope_fixed: float,
    amp_lo: float,
    amp_hi: float,
    sim_steps: int,
    registry,
    amp_tol: float = 1e-4,
    max_bisect: int = 20,
) -> Tuple[Optional[float], int]:
    """Binary-search the minimum feasible amplitude under a fixed masking slope."""

    probe_calls = 0

    def _probe_amp(amp: float) -> bool:
        nonlocal probe_calls
        bp = BoundaryPoint(
            point=point,
            direction=direction,
            duration=duration,
            amplitude=float(amp),
            boundary_type="conditional",
        )
        probe_calls += 1
        hazard, alarm_count, _ = _run_dual_ramp_probe(
            bp,
            hidden_point,
            float(slope_fixed),
            sim_steps,
            registry,
        )
        return hazard and alarm_count == 0

    lo = float(amp_lo)
    hi = float(amp_hi)
    lo_ok = _probe_amp(lo)
    hi_ok = _probe_amp(hi)
    if not hi_ok:
        return None, probe_calls
    if lo_ok:
        return lo, probe_calls

    best = hi
    rounds = 0
    while hi - lo > amp_tol and rounds < max_bisect:
        mid = 0.5 * (lo + hi)
        rounds += 1
        if _probe_amp(mid):
            hi = mid
            best = mid
        else:
            lo = mid
    return best, probe_calls


def _pick_working_slope_and_candidate(
    *,
    point: str,
    direction: str,
    duration: int,
    amp_cur: float,
    slope_min: float,
    slope_max: float,
    hidden_point: str,
    amp_lo: float,
    sim_steps: int,
    registry,
    amp_tol: float,
    max_bisect: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Choose a working slope inside the current band that keeps the next band alive."""

    total_probe_calls = 0
    width = float(slope_max) - float(slope_min)
    if width <= 0:
        return None, total_probe_calls

    candidates = []
    seen = set()
    for frac in COOPERATIVE_SLOPE_FRACTIONS:
        slope = float(slope_min) + width * float(frac)
        key = round(slope, 12)
        if key in seen:
            continue
        seen.add(key)
        amp_next, probe_calls = find_min_amp_under_fixed_slope(
            point=point,
            direction=direction,
            duration=duration,
            hidden_point=hidden_point,
            slope_fixed=slope,
            amp_lo=amp_lo,
            amp_hi=amp_cur,
            sim_steps=sim_steps,
            registry=registry,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
        )
        total_probe_calls += probe_calls
        if amp_next is None:
            continue
        bp_next = BoundaryPoint(
            point=point,
            direction=direction,
            duration=duration,
            amplitude=float(amp_next),
            boundary_type="conditional",
        )
        next_band = solve_slope_band_for_amp(
            bp_next,
            hidden_point,
            sim_steps,
            registry,
        )
        if next_band is None:
            continue
        next_slope_min, next_slope_max, band_probe_calls = next_band
        total_probe_calls += band_probe_calls
        candidates.append({
            "working_slope": slope,
            "amp_next": float(amp_next),
            "next_band": (next_slope_min, next_slope_max),
        })

    if not candidates:
        return None, total_probe_calls

    best = min(candidates, key=lambda item: item["amp_next"])
    return best, total_probe_calls


def refine_extreme_amp_within_band(
    *,
    point: str,
    direction: str,
    duration: int,
    hidden_point: str,
    stable_amp: float,
    slope_min: float,
    slope_max: float,
    amp_lo: float,
    sim_steps: int,
    registry,
    amp_tol: float,
    max_bisect: int,
    band_samples: int = EXTREME_BAND_SAMPLES,
) -> Tuple[Optional[Dict[str, float]], int]:
    """Scan the final stable band to find the deepest extreme cond boundary point."""

    width = float(slope_max) - float(slope_min)
    if width <= 0 or band_samples <= 0:
        return None, 0

    total_probe_calls = 0
    best: Optional[Dict[str, float]] = None
    if band_samples == 1:
        slope_values = [float(slope_min)]
    else:
        slope_values = [
            float(slope_min + i * width / (band_samples - 1))
            for i in range(band_samples)
        ]

    for slope in slope_values:
        amp_min, probe_calls = find_min_amp_under_fixed_slope(
            point=point,
            direction=direction,
            duration=duration,
            hidden_point=hidden_point,
            slope_fixed=slope,
            amp_lo=amp_lo,
            amp_hi=float(stable_amp),
            sim_steps=sim_steps,
            registry=registry,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
        )
        total_probe_calls += probe_calls
        if amp_min is None:
            continue
        if best is None or amp_min < float(best["extreme_amp"]):
            best = {
                "extreme_amp": float(amp_min),
                "extreme_slope": float(slope),
            }

    return best, total_probe_calls


def _expand_duration_task(args: Tuple[str, str, int, float, int, float, int]) -> Dict[str, Any]:
    point, direction, duration, base_amp, sim_buffer, amp_tol, max_bisect = args
    return expand_single_boundary_point(
        point,
        direction,
        duration,
        base_amp,
        sim_buffer=sim_buffer,
        amp_tol=amp_tol,
        max_bisect=max_bisect,
    )
def expand_single_boundary_point(
    point: str,
    direction: str,
    duration: int,
    base_amp: float,
    *,

    sim_buffer: int = 100,

    search_lo_ratio: float = 0.3,
    amp_tol: float = 1e-4,
    max_bisect: int = 20,
) -> Dict[str, Any]:
    """Expand a boundary point by cooperative dual ramp+hold search."""

    sim_steps = duration + sim_buffer
    hidden_point, _ = _resolve_hidden_point_and_alarm(point, direction)
    registry = _load_runtime_registry()
    counters = {"dual_probe_calls": 0}

    def _probe_band(amp: float, boundary_type: str) -> Optional[Tuple[float, float]]:
        bp = BoundaryPoint(
            point=point,
            direction=direction,
            duration=duration,
            amplitude=amp,
            boundary_type=boundary_type,
        )
        band = solve_slope_band_for_amp(
            bp,
            hidden_point,
            sim_steps,
            registry,
        )
        if band is None:
            return None
        slope_min, slope_max, probe_calls = band
        counters["dual_probe_calls"] += probe_calls
        return slope_min, slope_max

    base_band = _probe_band(float(base_amp), "regular")
    if base_band is None:
        return {
            "duration": duration,
            "direction": direction,
            "base_amp": base_amp,
            "expanded_amp": None,
            "status": "no_hazard_or_alarm_free_slope_at_base",
            "dual_probe_calls": counters["dual_probe_calls"],
            "simulations_total": counters["dual_probe_calls"],
        }

    lo = float(base_amp) * search_lo_ratio
    amp_cur = float(base_amp)
    slope_min_cur, slope_max_cur = base_band
    best_amp = amp_cur
    best_band = base_band
    iteration_history: List[Dict[str, Any]] = []

    for iteration in range(1, max_bisect + 1):
        candidate, probe_calls = _pick_working_slope_and_candidate(
            point=point,
            direction=direction,
            duration=duration,
            amp_cur=amp_cur,
            slope_min=slope_min_cur,
            slope_max=slope_max_cur,
            hidden_point=hidden_point,
            amp_lo=lo,
            sim_steps=sim_steps,
            registry=registry,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
        )
        counters["dual_probe_calls"] += probe_calls
        iteration_record: Dict[str, Any] = {
            "iter": iteration,
            "amp": amp_cur,
            "slope_min": slope_min_cur,
            "slope_max": slope_max_cur,
            "amp_search_lo": lo,
            "amp_search_hi": amp_cur,
        }
        if candidate is None:
            iteration_history.append(iteration_record)
            break
        amp_next = float(candidate["amp_next"])
        next_slope_min, next_slope_max = candidate["next_band"]
        iteration_record["selected_working_slope"] = float(candidate["working_slope"])
        iteration_record["amp_next"] = amp_next
        if amp_cur - amp_next < amp_tol:
            best_amp = float(amp_next)
            best_band = (slope_min_cur, slope_max_cur)
            break
        iteration_record["next_slope_min"] = next_slope_min
        iteration_record["next_slope_max"] = next_slope_max
        iteration_history.append(iteration_record)
        if next_slope_max > slope_max_cur or amp_next < best_amp:
            best_amp = float(amp_next)
            best_band = (next_slope_min, next_slope_max)
            amp_cur = float(amp_next)
            slope_min_cur, slope_max_cur = next_slope_min, next_slope_max
            continue
        break

    simulations_total = counters["dual_probe_calls"]
    if best_amp < float(base_amp) - amp_tol:
        slope_min_best, slope_max_best = best_band
        stable_amp = best_amp
        extreme = {
            "extreme_amp": stable_amp,
            "extreme_slope": float(iteration_history[-1]["selected_working_slope"])
            if iteration_history and "selected_working_slope" in iteration_history[-1]
            else 0.5 * (slope_min_best + slope_max_best),
        }
        extreme_result, extreme_calls = refine_extreme_amp_within_band(
            point=point,
            direction=direction,
            duration=duration,
            hidden_point=hidden_point,
            stable_amp=stable_amp,
            slope_min=slope_min_best,
            slope_max=slope_max_best,
            amp_lo=lo,
            sim_steps=sim_steps,
            registry=registry,
            amp_tol=amp_tol,
            max_bisect=max_bisect,
        )
        counters["dual_probe_calls"] += extreme_calls
        simulations_total = counters["dual_probe_calls"]
        if extreme_result is not None:
            extreme = extreme_result
        return {
            "duration": duration,
            "direction": direction,
            "base_amp": base_amp,
            "expanded_amp": stable_amp,
            "expansion_ratio": stable_amp / float(base_amp),
            "stable_amp": stable_amp,
            "stable_slope_min": slope_min_best,
            "stable_slope_max": slope_max_best,
            "slope_min": slope_min_best,
            "slope_max": slope_max_best,
            "extreme_amp": extreme["extreme_amp"],
            "extreme_slope": extreme["extreme_slope"],
            "iteration_history": iteration_history,
            "final_iteration": len(iteration_history),
            "search_precision": amp_tol,
            "status": "ok",
            "search_method": "cooperative_dual_ramp_hold_boundary",
            "dual_probe_calls": counters["dual_probe_calls"],
            "simulations_total": simulations_total,
        }
    return {
        "duration": duration,
        "direction": direction,
        "base_amp": base_amp,
        "expanded_amp": None,
        "stable_amp": None,
        "stable_slope_min": slope_min_cur,
        "stable_slope_max": slope_max_cur,
        "slope_min": slope_min_cur,
        "slope_max": slope_max_cur,
        "extreme_amp": None,
        "extreme_slope": None,
        "iteration_history": iteration_history,
        "final_iteration": len(iteration_history),
        "search_precision": amp_tol,
        "status": "no_significant_expansion",
        "search_method": "cooperative_dual_ramp_hold_boundary",
        "dual_probe_calls": counters["dual_probe_calls"],
        "simulations_total": simulations_total,
    }




def run_expansion(
    *,
    boundary_path: str | Path,
    output_path: str | Path,
    direction: str = "pos",
    duration_step: int = 10,
    sim_buffer: int = 100,
    amp_tol: float = 0.005,
    max_bisect: int = 15,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run boundary expansion on all durations.



    Args:

        boundary_path: Path to CEGIS boundary_results.json

        output_path: Output path for expansion results

        direction: "pos" or "neg"

        duration_step: Only process every N-th duration (for speed)

        sim_buffer: Hold steps after ramp

        amp_tol: Bisection tolerance

        max_bisect: Max bisection iterations



    Returns:

        Summary dict

    """

    boundary_path = Path(boundary_path)

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)



    data = json.loads(boundary_path.read_text(encoding="utf-8"))
    point = data["point"]


    # Extract base boundaries

    base_boundaries = {}

    for b in data["boundaries"]:

        if b["direction"] == direction and b.get("lower_target_amp"):

            base_boundaries[b["duration"]] = b["lower_target_amp"]



    # Select durations to process

    all_durs = sorted(base_boundaries.keys())

    durs_to_process = [d for d in all_durs if d % duration_step == 0 or d == all_durs[0] or d == all_durs[-1]]



    print(f"[Expansion] {len(durs_to_process)} durations to process (step={duration_step})")

    t_start = time.time()



    expanded = []
    total_dual_probe_calls = 0
    total_simulations = 0
    tasks = [
        (point, direction, dur, base_boundaries[dur], sim_buffer, amp_tol, max_bisect)
        for dur in durs_to_process
    ]

    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            expanded = list(ex.map(_expand_duration_task, tasks))
    else:
        expanded = [_expand_duration_task(task) for task in tasks]

    expanded.sort(key=lambda r: int(r["duration"]))

    for i, result in enumerate(expanded):
        dur = int(result["duration"])
        base_amp = float(result["base_amp"])
        total_dual_probe_calls += int(result.get("dual_probe_calls", 0))
        total_simulations += int(result.get("simulations_total", 0))

        if (i + 1) % 10 == 0 or (i + 1) == len(expanded):
            status = result["status"]
            exp_amp = result.get("expanded_amp")
            ratio = result.get("expansion_ratio")
            if exp_amp:
                print(f"  [{i+1}/{len(durs_to_process)}] d={dur}: "
                      f"base={base_amp:.4f} -> expanded={exp_amp:.4f} (ratio={ratio:.3f})")
            else:
                print(f"  [{i+1}/{len(durs_to_process)}] d={dur}: {status}")



    t_end = time.time()



    # Build output

    ok_results = [r for r in expanded if r["status"] == "ok"]

    output_data = {

        "point": point,

        "direction": direction,

        "method": "cooperative_dual_ramp_hold_boundary",

        "base_boundary": [

            {"direction": direction, "duration": d, "lower_target_amp": base_boundaries[d]}

            for d in sorted(base_boundaries.keys())

        ],

        "expanded_boundary": [

            {

                "direction": r["direction"],

                "duration": r["duration"],

                "lower_target_amp": r["expanded_amp"],

                "regular_amp": r["base_amp"],

                "expansion_ratio": r["expansion_ratio"],

                "stable_amp": r["stable_amp"],

                "stable_slope_min": r["stable_slope_min"],

                "stable_slope_max": r["stable_slope_max"],

                "slope_min": r["slope_min"],

                "slope_max": r["slope_max"],

                "extreme_amp": r["extreme_amp"],

                "extreme_slope": r["extreme_slope"],

            }

            for r in ok_results

        ],

        "summary": {
            "total_processed": len(expanded),
            "expanded_ok": len(ok_results),
            "failed": len(expanded) - len(ok_results),
            "cond_points_found": len(ok_results),
            "dual_probe_calls": total_dual_probe_calls,
            "total_simulations": total_simulations,
            "time_s": round(t_end - t_start, 1),
        },
    }


    output_path.write_text(

        json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",

        encoding="utf-8",

    )

    print(f"\n[Expansion] Done: {len(ok_results)}/{len(expanded)} expanded, {t_end-t_start:.1f}s")

    print(f"Saved: {output_path}")

    return output_data["summary"]





if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Boundary expansion via direct dual ramp+hold search")
    p.add_argument("--boundary-path", required=True, help="CEGIS boundary_results.json")
    p.add_argument("--output-path", required=True, help="Output expansion results JSON")
    p.add_argument("--direction", default="pos")
    p.add_argument("--duration-step", type=int, default=10)
    p.add_argument("--amp-tol", type=float, default=0.005)
    p.add_argument("--max-bisect", type=int, default=15)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    run_expansion(
        boundary_path=args.boundary_path,
        output_path=args.output_path,
        direction=args.direction,
        duration_step=args.duration_step,
        amp_tol=args.amp_tol,
        max_bisect=args.max_bisect,
        workers=args.workers,
    )

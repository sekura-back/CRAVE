# Subregion Subdivision - Section 3.4.1

"""Unified Stage 3 main chain for boiler/TE-like platforms."""

from __future__ import annotations

import argparse

from concurrent.futures import ProcessPoolExecutor

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple



_HERE = Path(__file__).resolve()

_ROOT = _HERE.parents[2]

for _p in (_ROOT, _ROOT / "src", _ROOT / "src" / "stage4_executor",

           _ROOT / "simulators"):

    if str(_p) not in sys.path:

        sys.path.insert(0, str(_p))





# Boiler-specific constants.

ANCHOR_STEAM_SP = 23.74        # steam_setpoint default at nominal load

BOILER_CUT_TH = 1.5            # absolute MPa deviation

ANCHOR_FUEL = 157.6

SIM = 2200                     # boiler runs at Ts=0.1, hazards in 100-2000 step range
HOLD = 100

N_RATE_SAMPLES = 10

SPOOF_DUR_GRID = [50, 100, 200, 400, 800]

MAX_BISECT = 12

RATE_TOL = 0.0005
DYNAMIC_GRID_SAMPLES = 11
DYNAMIC_BISECT_ROUNDS = 8
COND_DIRECT_SLOPE_MAX = 0.12
COND_DIRECT_GRID_SAMPLES = 61
REGION_DELTA_T = 50
REGION_DELTA_K = 0.01
REGION_GAMMA_MERGE = 0.10
TE_SEP_LEVEL_SP = 50.0
TE_SEP_LEVEL_ALARM_ABS = 15.0
TE_SEP_LEVEL_RANGE = (0.0, 100.0)
TE_SEP_LEVEL_RATE_LIMIT = 2.0

BASE_STAGE3_FULL_PATH = _ROOT / "results" / "stage3" / "boiler_stage3_fuel_steam_full.json"
BASE_STAGE3_REGION_PATH = _ROOT / "results" / "stage3" / "boiler_stage3_subregion_2d_base.json"
COND_STAGE3_REFINE_PATH = _ROOT / "results" / "stage3" / "boiler_stage3_cond_slope_refine.json"
COMBINED_STAGE3_REGION_PATH = _ROOT / "results" / "stage3" / "boiler_stage3_combined_regions.json"
TE_STAGE3_FULL_PATH = _ROOT / "results" / "stage3" / "te_stage3_xmv07_sp3_full.json"
TE_STAGE3_REGION_PATH = _ROOT / "results" / "stage3" / "te_stage3_subregion_2d_base.json"


def get_stage3_platform_config(target: str) -> Dict:
    key = str(target).strip().lower()
    if key in {"boiler", "boiler_ccs"}:
        return {
            "target": "boiler",
            "system_id": "boiler_ccs",
            "manifest_path": _ROOT / "simulators" / "boiler_ccs" / "system_manifest.json",
            "base_results_path": _ROOT / "results" / "stage2" / "boiler_base_boundary.json",
            "conditional_results_path": _ROOT / "results" / "stage2" / "boiler_conditional_expansion.json",
            "stage3_full_path": BASE_STAGE3_FULL_PATH,
            "stage3_base_region_path": BASE_STAGE3_REGION_PATH,
            "stage3_cond_refine_path": COND_STAGE3_REFINE_PATH,
            "stage3_combined_region_path": COMBINED_STAGE3_REGION_PATH,
            "manip_point": "simulation_fuel_command",
            "mask_point": "simulation_steam_setpoint",
            "spoof_field": "steam_setpoint",
            "alarm_id": "P-BOILER-CUT-001",
            "sample_duration_mod": 50,
            "supports_conditional": True,
        }
    if key in {"te", "tennessee_eastman", "tep"}:
        return {
            "target": "te",
            "system_id": "tep_downs_vogel_1993",
            "manifest_path": _ROOT / "simulators" / "tennessee_eastman" / "system_manifest.json",
            "base_results_path": _ROOT / "results" / "stage2" / "te_base_boundary.json",
            "conditional_results_path": None,
            "stage3_full_path": TE_STAGE3_FULL_PATH,
            "stage3_base_region_path": TE_STAGE3_REGION_PATH,
            "stage3_cond_refine_path": None,
            "stage3_combined_region_path": None,
            "manip_point": "simulation_xmv_07",
            "mask_point": "simulation_sp_separator_level",
            "spoof_field": "setpoints_6",
            "alarm_id": "P-TEP-SEP-LEVEL-TRACK",
            "sample_duration_mod": 50,
            "supports_conditional": False,
        }
    raise ValueError(f"unsupported stage3 target: {target}")





def _load_boiler_sim():

    """Lazy-load the boiler ClosedLoopSim."""

    from simulators.boiler_ccs.simulation import ClosedLoopSim

    return ClosedLoopSim





def _build_hook(point_specs, registry, runtime_args):

    from generic_runtime_executor import build_injection_hook

    return build_injection_hook(point_specs=point_specs, registry=registry,

                                runtime_args=runtime_args)





def run_fuel_only(rate: float, duration: int, sim_steps: int,

                  registry) -> Tuple[List[float], Optional[int]]:

    """Run fuel_command-only attack, return (pressure trajectory, haz_step)."""

    ClosedLoopSim = _load_boiler_sim()

    hook = _build_hook(

        point_specs=[{"point": "simulation_fuel_command", "amp": rate,

                      "duration": duration, "t_start": 0, "role": "manip"}],

        registry=registry, runtime_args={"steps": sim_steps},

    )

    sim = ClosedLoopSim(Ts=0.1, enable_csv=False)

    # Boiler ClosedLoopSim has more nuanced run() args (ignore_protection_predicates).

    # We need the hazard to actually fire, so pass empty ignore lists.

    trace, meta = sim.run(

        steps=sim_steps,

        injection_hook=hook,

        return_trace=True,

        stop_on_trip=True,

        ignore_protection_predicates_for_stop=["BOILER_CUT_TH",

                                               "FUEL_CUT_TH",

                                               "WATER_CUT_TH"],

        ignore_protection_predicates_for_enforcement=["BOILER_CUT_TH",

                                                      "FUEL_CUT_TH",

                                                      "WATER_CUT_TH"],

    )

    sim.close()

    pressure = [float(r["main_steam_pressure"]) for r in trace]

    haz = None

    for r in trace:

        # 'trip_any' is the hazard trip flag in boiler's record dict.

        if r.get("hazard_pressure_triggered") or r.get("hazard_waterwall_triggered") \
                or r.get("hazard_feedwater_triggered"):

            haz = int(r["t_step"])

            break

    return pressure, haz


def run_fuel_with_steam(rate: float, duration: int, steam_slope: float,

                        registry, sim_steps: Optional[int] = None) -> Dict:

    """Run fuel + steam_setpoint ramp/hold and report reach-avoid outcome."""

    ClosedLoopSim = _load_boiler_sim()

    steps = int(sim_steps if sim_steps is not None else duration + 100)

    hook = _build_hook(

        point_specs=[{"point": "simulation_fuel_command", "amp": rate,

                      "duration": duration, "t_start": 0, "role": "manip"},

                     {"point": "simulation_steam_setpoint", "amp": steam_slope,

                      "duration": duration, "t_start": 0, "role": "mask"}],

        registry=registry, runtime_args={"steps": steps},

    )

    sim = ClosedLoopSim(Ts=0.1, enable_csv=False)

    trace, _ = sim.run(

        steps=steps,

        injection_hook=hook,

        return_trace=True,

        stop_on_trip=True,

        ignore_protection_predicates_for_stop=["BOILER_CUT_TH",

                                               "FUEL_CUT_TH",

                                               "WATER_CUT_TH"],

        ignore_protection_predicates_for_enforcement=["BOILER_CUT_TH",

                                                      "FUEL_CUT_TH",

                                                      "WATER_CUT_TH"],

    )

    sim.close()

    hazard_step = None

    alarm_step = None

    for row in trace:

        if hazard_step is None and (row.get("hazard_pressure_triggered")

                                    or row.get("hazard_waterwall_triggered")

                                    or row.get("hazard_feedwater_triggered")):

            hazard_step = int(row["t_step"])

        if alarm_step is None and (row.get("boiler_cut")

                                   or row.get("fuel_cut")

                                   or row.get("water_cut")):

            alarm_step = int(row["t_step"])

    hazard = hazard_step is not None

    alarm = alarm_step is not None

    reach_avoid = hazard and (alarm_step is None or alarm_step >= hazard_step)

    return {

        "hazard": hazard,

        "alarm": alarm,

        "reach_avoid": reach_avoid,

        "hazard_step": hazard_step,

        "alarm_step": alarm_step,

    }


def _dynamic_refine_band(rate: float, duration: int, static_band: Tuple[float, float],

                         registry) -> Optional[Tuple[float, float]]:

    """Shrink a static slope band using real dual-injection reach-avoid checks."""

    lo, hi = float(static_band[0]), float(static_band[1])

    if hi <= lo:

        return None

    if DYNAMIC_GRID_SAMPLES <= 1:

        probes = [lo]

    else:

        probes = [lo + i * (hi - lo) / (DYNAMIC_GRID_SAMPLES - 1)

                  for i in range(DYNAMIC_GRID_SAMPLES)]

    outcomes = [run_fuel_with_steam(rate, duration, slope, registry)["reach_avoid"]

                for slope in probes]

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

        for _ in range(DYNAMIC_BISECT_ROUNDS):

            mid = 0.5 * (left_fail + left_ok)

            if run_fuel_with_steam(rate, duration, mid, registry)["reach_avoid"]:

                left_ok = mid

            else:

                left_fail = mid

        left = left_ok

    if best_end < len(probes) - 1:

        right_ok = right

        right_fail = probes[best_end + 1]

        for _ in range(DYNAMIC_BISECT_ROUNDS):

            mid = 0.5 * (right_ok + right_fail)

            if run_fuel_with_steam(rate, duration, mid, registry)["reach_avoid"]:

                right_ok = mid

            else:

                right_fail = mid

        right = right_ok

    if right <= left:

        return None

    return left, right


def _direct_dynamic_band(rate: float, duration: int, registry,

                         slope_lo: float = 0.0,

                         slope_hi: float = COND_DIRECT_SLOPE_MAX) -> Optional[Tuple[float, float]]:

    """Directly scan ramp+hold slopes and keep the widest RA-feasible interval."""

    if slope_hi <= slope_lo:

        return None

    probes = [
        slope_lo + i * (slope_hi - slope_lo) / (COND_DIRECT_GRID_SAMPLES - 1)
        for i in range(COND_DIRECT_GRID_SAMPLES)
    ]

    outcomes = [run_fuel_with_steam(rate, duration, slope, registry)["reach_avoid"]
                for slope in probes]

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

        for _ in range(DYNAMIC_BISECT_ROUNDS):

            mid = 0.5 * (left_fail + left_ok)

            if run_fuel_with_steam(rate, duration, mid, registry)["reach_avoid"]:

                left_ok = mid

            else:

                left_fail = mid

        left = left_ok

    if best_end < len(probes) - 1:

        right_ok = right

        right_fail = probes[best_end + 1]

        for _ in range(DYNAMIC_BISECT_ROUNDS):

            mid = 0.5 * (right_ok + right_fail)

            if run_fuel_with_steam(rate, duration, mid, registry)["reach_avoid"]:

                right_ok = mid

            else:

                right_fail = mid

        right = right_ok

    if right <= left:

        return None

    return left, right


def solve_dynamic_point_bomega(
    *,
    rate: float,
    duration: int,
    registry,
    slope_lo: float = 0.0,
    slope_hi: float = COND_DIRECT_SLOPE_MAX,
    grid_samples: int = COND_DIRECT_GRID_SAMPLES,
    bisect_rounds: int = DYNAMIC_BISECT_ROUNDS,
) -> Dict:
    """Solve dynamic feasible slope segments for one fixed representative point."""

    if slope_hi <= slope_lo:
        return {
            "dynamic_segments": [],
            "dynamic_primary_band": None,
            "segment_count": 0,
            "probe_count": 0,
            "runtime_steps": int(duration) + HOLD,
        }

    runtime_steps = int(duration) + HOLD
    if grid_samples <= 1:
        probes = [float(slope_lo)]
    else:
        probes = [
            float(slope_lo + i * (slope_hi - slope_lo) / (grid_samples - 1))
            for i in range(grid_samples)
        ]

    outcomes = []
    probe_count = 0
    for slope in probes:
        probe_count += 1
        outcomes.append(
            bool(run_fuel_with_steam(rate, duration, slope, registry, sim_steps=runtime_steps)["reach_avoid"])
        )

    coarse_segments = []
    cur_start = None
    for idx, ok in enumerate(outcomes):
        if ok and cur_start is None:
            cur_start = idx
        if (not ok or idx == len(outcomes) - 1) and cur_start is not None:
            cur_end = idx if ok and idx == len(outcomes) - 1 else idx - 1
            coarse_segments.append((cur_start, cur_end))
            cur_start = None

    segments = []
    for start_idx, end_idx in coarse_segments:
        left = probes[start_idx]
        right = probes[end_idx]
        if start_idx > 0:
            left_fail = probes[start_idx - 1]
            left_ok = left
            for _ in range(bisect_rounds):
                mid = 0.5 * (left_fail + left_ok)
                probe_count += 1
                if run_fuel_with_steam(rate, duration, mid, registry, sim_steps=runtime_steps)["reach_avoid"]:
                    left_ok = mid
                else:
                    left_fail = mid
            left = left_ok
        if end_idx < len(probes) - 1:
            right_ok = right
            right_fail = probes[end_idx + 1]
            for _ in range(bisect_rounds):
                mid = 0.5 * (right_ok + right_fail)
                probe_count += 1
                if run_fuel_with_steam(rate, duration, mid, registry, sim_steps=runtime_steps)["reach_avoid"]:
                    right_ok = mid
                else:
                    right_fail = mid
            right = right_ok
        if right <= left:
            continue
        segments.append({
            "alpha_minus": float(left),
            "alpha_plus": float(right),
            "width": float(right - left),
            "grid_hit_count": int(end_idx - start_idx + 1),
            "refinement_rounds": int(bisect_rounds),
        })

    primary = None
    if len(segments) == 1:
        only = segments[0]
        primary = {
            "alpha_minus": only["alpha_minus"],
            "alpha_plus": only["alpha_plus"],
            "width": only["width"],
        }
    elif segments:
        widest = max(segments, key=lambda seg: float(seg["width"]))
        primary = {
            "alpha_minus": widest["alpha_minus"],
            "alpha_plus": widest["alpha_plus"],
            "width": widest["width"],
        }

    return {
        "dynamic_segments": segments,
        "dynamic_primary_band": primary,
        "segment_count": len(segments),
        "probe_count": probe_count,
        "runtime_steps": runtime_steps,
    }





def static_solve_steam_sp_rate(pressure: List[float], horizon: int,

                                spoof_dur: int) -> Optional[Tuple[float, float]]:

    """Static linear-feasibility solver for steam_setpoint ramp+hold.



    Constraint at every t in [0, horizon):

      |spoof(t) - pressure(t)| <= BOILER_CUT_TH

    where spoof(t) = ANCHOR_STEAM_SP + rate * min(t, spoof_dur-1).



    Returns (rate_lo, rate_hi) or None if infeasible.

    """

    if horizon <= 1:

        return None

    # t=0 admissibility:

    if abs(ANCHOR_STEAM_SP - pressure[0]) > BOILER_CUT_TH:

        return None

    rate_lo_max = -float("inf")

    rate_hi_min =  float("inf")

    for t in range(1, horizon):

        eff = min(t, spoof_dur - 1)

        if eff <= 0:

            continue

        # |ANCHOR + rate*eff - pressure[t]| <= TH

        # -> pressure[t] - TH - ANCHOR <= rate*eff <= pressure[t] + TH - ANCHOR

        rlo = (pressure[t] - BOILER_CUT_TH - ANCHOR_STEAM_SP) / eff

        rhi = (pressure[t] + BOILER_CUT_TH - ANCHOR_STEAM_SP) / eff

        if rlo > rate_lo_max: rate_lo_max = rlo

        if rhi < rate_hi_min: rate_hi_min = rhi

    if rate_lo_max > rate_hi_min:

        return None

    return rate_lo_max, rate_hi_min





def solve_steam_sp_at_point(rate: float, duration: int, registry,

                             sim_steps: int = SIM) -> Dict:

    """Compute steam_setpoint feasibility band at one (fuel_rate, fuel_dur) probe."""

    pressure, haz0 = run_fuel_only(rate, duration, sim_steps, registry)

    horizon = haz0 if haz0 is not None else len(pressure)

    spoof_dur = max(2, int(duration))



    bands_per_sd: List[Dict] = []

    any_feasible = False

    b = static_solve_steam_sp_rate(pressure, horizon, spoof_dur)
    dyn_payload = solve_dynamic_point_bomega(
        rate=rate,
        duration=duration,
        registry=registry,
    )
    dyn_band = dyn_payload.get("dynamic_primary_band")

    if dyn_band is None:

        bands_per_sd.append({"spoof_dur": spoof_dur, "feasible": False,

                             "steam_rate_lo": None, "steam_rate_hi": None,

                             "steam_band_width": None,

                             "hold_steps": horizon - spoof_dur,
                             "static_rate_lo": None if b is None else b[0],
                             "static_rate_hi": None if b is None else b[1],
                             "static_band_width": None if b is None else b[1] - b[0],
                             "band_source": "direct_dynamic_point_scan"})
 
    else:
 
        any_feasible = True
 
        bands_per_sd.append({"spoof_dur": spoof_dur, "feasible": True,
 
                             "steam_rate_lo": dyn_band["alpha_minus"], "steam_rate_hi": dyn_band["alpha_plus"],
 
                             "steam_band_width": dyn_band["width"],
 
                             "hold_steps": horizon - spoof_dur,
 
                             "static_rate_lo": None if b is None else b[0], "static_rate_hi": None if b is None else b[1],
 
                             "static_band_width": None if b is None else b[1] - b[0],
                             "band_source": "direct_dynamic_point_scan"})



    return {

        "fuel_rate": rate,

        "fuel_dur": duration,

        "haz_step_fuel_only": haz0,

        "horizon": horizon,

        "pressure_min": min(pressure),

        "pressure_max": max(pressure),

        "feasible": any_feasible,

        "bands": bands_per_sd,

        "anchor_steam_sp": ANCHOR_STEAM_SP,

        "boiler_cut_th": BOILER_CUT_TH,

    }


def solve_cond_steam_sp_at_point(rate: float, duration: int, registry) -> Dict:

    """Compute a cond-point slope band over the artifact runtime window."""
    sim_steps = int(duration) + HOLD

    pressure, haz0 = run_fuel_only(rate, duration, sim_steps, registry)

    horizon = haz0 if haz0 is not None else len(pressure)

    spoof_dur = max(2, int(duration))

    bands_per_sd: List[Dict] = []

    any_feasible = False

    band_source = "direct_dynamic_ra_scan"

    static_band = static_solve_steam_sp_rate(pressure, horizon, spoof_dur)

    dyn = None

    if static_band is not None:

        dyn = _dynamic_refine_band(rate, duration, static_band, registry)

        if dyn is not None:

            band_source = "static_then_dynamic_ra_shrink_same_duration"

    if dyn is None:

        dyn = _direct_dynamic_band(rate, duration, registry)

    if dyn is None:

        bands_per_sd.append({"spoof_dur": spoof_dur, "feasible": False,

                             "steam_rate_lo": None, "steam_rate_hi": None,

                             "steam_band_width": None,

                             "hold_steps": horizon - spoof_dur,

                             "static_rate_lo": None if static_band is None else static_band[0],

                             "static_rate_hi": None if static_band is None else static_band[1],

                             "static_band_width": None if static_band is None else static_band[1] - static_band[0],

                             "band_source": band_source})

    else:

        any_feasible = True

        bands_per_sd.append({"spoof_dur": spoof_dur, "feasible": True,

                             "steam_rate_lo": dyn[0], "steam_rate_hi": dyn[1],

                             "steam_band_width": dyn[1] - dyn[0],

                             "hold_steps": horizon - spoof_dur,

                             "static_rate_lo": None if static_band is None else static_band[0],

                             "static_rate_hi": None if static_band is None else static_band[1],

                             "static_band_width": None if static_band is None else static_band[1] - static_band[0],

                             "band_source": band_source})

    return {

        "fuel_rate": rate,

        "fuel_dur": duration,

        "haz_step_fuel_only": haz0,

        "horizon": horizon,

        "pressure_min": min(pressure),

        "pressure_max": max(pressure),

        "feasible": any_feasible,

        "bands": bands_per_sd,

        "anchor_steam_sp": ANCHOR_STEAM_SP,

        "boiler_cut_th": BOILER_CUT_TH,

    }





def find_rate_min_feasible(duration: int, rate_lo: float, rate_hi: float,

                            registry, *,

                            sim_steps: int = SIM,

                            max_bisect: int = MAX_BISECT,

                            rate_tol: float = RATE_TOL,
                            cfg: Optional[Dict] = None) -> Optional[Dict]:

    """Binary-search smallest fuel rate that admits feasible steam_setpoint spoof."""

    platform_cfg = cfg or get_stage3_platform_config("boiler")
    res_hi = solve_stage3_point_at_platform(platform_cfg, rate_hi, duration, registry, sim_steps)

    if not res_hi["feasible"]:

        return None

    res_lo = solve_stage3_point_at_platform(platform_cfg, rate_lo, duration, registry, sim_steps)

    if res_lo["feasible"]:

        return {**res_lo, "is_rate_min_feasible": True,

                "bisect_rounds": 0, "search_range": [rate_lo, rate_hi]}



    lo, hi = rate_lo, rate_hi

    last_feasible = res_hi

    rounds = 0

    while hi - lo > rate_tol and rounds < max_bisect:

        mid = 0.5 * (lo + hi)

        res = solve_stage3_point_at_platform(platform_cfg, mid, duration, registry, sim_steps)

        rounds += 1

        if res["feasible"]:

            hi = mid

            last_feasible = res

        else:

            lo = mid

    last_feasible["is_rate_min_feasible"] = True

    last_feasible["bisect_rounds"] = rounds

    last_feasible["search_range"] = [lo, hi]

    return last_feasible





def process_one_duration(d: int, rate_lo: float, rate_hi: float,

                          registry, cfg: Optional[Dict] = None) -> Dict:

    platform_cfg = cfg or get_stage3_platform_config("boiler")

    samples = []

    for i in range(N_RATE_SAMPLES):

        q = i / (N_RATE_SAMPLES - 1)

        r = rate_lo + q * (rate_hi - rate_lo)

        res = solve_stage3_point_at_platform(platform_cfg, r, d, registry)

        res["sample_q"] = q

        samples.append(res)

    rmf = find_rate_min_feasible(d, rate_lo, rate_hi, registry, cfg=platform_cfg)

    return {

        "duration": d,

        "base_rate_lo": rate_lo,

        "base_rate_hi": rate_hi,

        "samples": samples,

        "rate_min_feasible": rmf,

    }


def _process_one_duration_task(task: Tuple[int, float, float, Dict]) -> Dict:
    d, rate_lo, rate_hi, cfg = task
    from generic_runtime_executor import load_injection_registry

    registry = load_injection_registry(cfg["manifest_path"])
    return process_one_duration(d, rate_lo, rate_hi, registry, cfg)


def _sep_level_rate_is_static_feasible(
    rate: float,
    xmeas12: List[float],
    horizon: int,
    spoof_dur: int,
    anchor: float,
    alarm_abs: float,
    value_range: Tuple[float, float],
) -> bool:
    lo_value, hi_value = value_range
    for t in range(1, horizon):
        eff = min(t, spoof_dur - 1)
        if eff <= 0:
            continue
        sp = min(hi_value, max(lo_value, anchor + rate * eff))
        if abs(float(xmeas12[t]) - sp) > alarm_abs:
            return False
    return True


def _sep_level_clamped_static_band(
    xmeas12: List[float],
    horizon: int,
    spoof_dur: int,
    anchor: float,
    alarm_abs: float,
    value_range: Tuple[float, float] = TE_SEP_LEVEL_RANGE,
    rate_limit: float = TE_SEP_LEVEL_RATE_LIMIT,
) -> Optional[Tuple[float, float]]:
    lo_value, hi_value = value_range
    candidates = {-float(rate_limit), float(rate_limit)}
    for t in range(1, horizon):
        eff = min(t, spoof_dur - 1)
        if eff <= 0:
            continue
        band_lo = float(xmeas12[t]) - alarm_abs
        band_hi = float(xmeas12[t]) + alarm_abs
        for boundary in (lo_value, hi_value, band_lo, band_hi):
            candidates.add((boundary - anchor) / eff)

    sorted_candidates = sorted(c for c in candidates if -rate_limit <= c <= rate_limit)
    intervals: List[Tuple[float, float]] = []
    for left, right in zip(sorted_candidates, sorted_candidates[1:]):
        mid = (left + right) / 2.0
        if _sep_level_rate_is_static_feasible(
            mid, xmeas12, horizon, spoof_dur, anchor, alarm_abs, value_range
        ):
            intervals.append((left, right))

    for point in sorted_candidates:
        if _sep_level_rate_is_static_feasible(
            point, xmeas12, horizon, spoof_dur, anchor, alarm_abs, value_range
        ):
            intervals.append((point, point))

    if not intervals:
        return None
    return min(left for left, _ in intervals), max(right for _, right in intervals)


def solve_te_sp3_at_point(rate: float, duration: int, registry,
                          sim_steps: int = 1500) -> Dict:
    from simulators.tennessee_eastman.simulation import ClosedLoopSim
    from generic_runtime_executor import build_injection_hook

    sim_steps = int(duration) + HOLD
    anchor_sep_level_sp = TE_SEP_LEVEL_SP
    alarm_abs = TE_SEP_LEVEL_ALARM_ABS

    hook = build_injection_hook(
        point_specs=[{"point": "simulation_xmv_07", "amp": rate,
                      "duration": duration, "t_start": 0, "role": "manip"}],
        registry=registry,
        runtime_args={"steps": sim_steps},
    )
    sim = ClosedLoopSim(Ts=1.0, mode=1, random_seed=0)
    try:
        trace, meta = sim.run(
            steps=sim_steps,
            injection_hook=hook,
            return_trace=True,
            stop_on_trip=True,
        )
    finally:
        sim.close()

    xmeas12 = [float(row["xmeas_12"]) for row in trace]
    haz0 = meta.get("stop_step")
    horizon = min(int(haz0), len(xmeas12)) if haz0 is not None else len(xmeas12)
    spoof_dur = max(2, int(duration))

    if haz0 is None or horizon <= 1 or spoof_dur < 2:
        static_band = None
    else:
        band_lo0 = xmeas12[0] - alarm_abs
        band_hi0 = xmeas12[0] + alarm_abs
        if not (band_lo0 <= anchor_sep_level_sp <= band_hi0):
            static_band = None
        else:
            static_band = _sep_level_clamped_static_band(
                xmeas12,
                horizon,
                spoof_dur,
                anchor_sep_level_sp,
                alarm_abs,
            )

    dynamic_band = static_band

    bands_per_sd: List[Dict] = []
    if dynamic_band is None:
        bands_per_sd.append({
            "spoof_dur": spoof_dur,
            "feasible": False,
            "sep_level_rate_lo": None,
            "sep_level_rate_hi": None,
            "sep_level_band_width": None,
            "hold_steps": horizon - spoof_dur,
            "static_rate_lo": None if static_band is None else static_band[0],
            "static_rate_hi": None if static_band is None else static_band[1],
            "static_band_width": None if static_band is None else static_band[1] - static_band[0],
            "band_source": "sep_level_static_same_duration",
        })
    else:
        bands_per_sd.append({
            "spoof_dur": spoof_dur,
            "feasible": True,
            "sep_level_rate_lo": dynamic_band[0],
            "sep_level_rate_hi": dynamic_band[1],
            "sep_level_band_width": dynamic_band[1] - dynamic_band[0],
            "hold_steps": horizon - spoof_dur,
            "static_rate_lo": None if static_band is None else static_band[0],
            "static_rate_hi": None if static_band is None else static_band[1],
            "static_band_width": None if static_band is None else static_band[1] - static_band[0],
            "band_source": "sep_level_static_same_duration",
        })

    return {
        "xmv07_rate": rate,
        "xmv07_dur": duration,
        "haz_step_xmv07_only": haz0,
        "horizon": horizon,
        "xmeas12_min": min(xmeas12),
        "xmeas12_max": max(xmeas12),
        "feasible": dynamic_band is not None,
        "bands": bands_per_sd,
        "anchor_sep_level_sp": anchor_sep_level_sp,
        "alarm_threshold_abs": alarm_abs,
        "measured_var": "xmeas_12",
    }


def solve_stage3_point_at_platform(cfg: Dict, rate: float, duration: int, registry,
                                   sim_steps: int = SIM) -> Dict:
    if cfg["target"] == "te":
        return solve_te_sp3_at_point(rate, duration, registry)
    return solve_steam_sp_at_point(rate, duration, registry, sim_steps)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return out


def slope_distance(b1: Tuple[float, float], b2: Tuple[float, float]) -> float:
    return abs(b1[0] - b2[0]) + abs(b1[1] - b2[1])


def _pick_slope_interval(bands: List[Dict]) -> Optional[Tuple[float, float]]:
    feasible = [b for b in bands if b.get("feasible")]
    if not feasible:
        return None
    def _band_lo(b: Dict) -> float:
        if b.get("steam_rate_lo") is not None:
            return float(b["steam_rate_lo"])
        if b.get("sep_level_rate_lo") is not None:
            return float(b["sep_level_rate_lo"])
        return float(b["sp3_rate_lo"])

    def _band_hi(b: Dict) -> float:
        if b.get("steam_rate_hi") is not None:
            return float(b["steam_rate_hi"])
        if b.get("sep_level_rate_hi") is not None:
            return float(b["sep_level_rate_hi"])
        return float(b["sp3_rate_hi"])

    lo = min(_band_lo(b) for b in feasible)
    hi = max(_band_hi(b) for b in feasible)
    if hi <= lo:
        return None
    return lo, hi


def _amp_bucket(amp: float) -> float:
    return round(round(amp / REGION_DELTA_K) * REGION_DELTA_K, 2)


def _bucket_key(amp: float) -> int:
    return int(round(_amp_bucket(amp) * 100))


def _aggregate_probe_records(records: List[Dict]) -> List[Dict]:
    merged: Dict[Tuple[int, int], Dict] = {}
    for record in records:
        key = (
            int(record["duration"]),
            int(record["k_int"]),
            str(record.get("boundary_role", "default")),
        )
        current = merged.get(key)
        if current is None:
            merged[key] = dict(record)
            continue
        current["slope_min"] = max(float(current["slope_min"]), float(record["slope_min"]))
        current["slope_max"] = min(float(current["slope_max"]), float(record["slope_max"]))
        current["source_count"] = int(current.get("source_count", 1)) + int(record.get("source_count", 1))
        current["amp_actual_min"] = min(
            float(current.get("amp_actual_min", current["amp_actual"])),
            float(record.get("amp_actual_min", record["amp_actual"])),
        )
        current["amp_actual_max"] = max(
            float(current.get("amp_actual_max", current["amp_actual"])),
            float(record.get("amp_actual_max", record["amp_actual"])),
        )
    filtered = [
        record for record in merged.values()
        if float(record["slope_max"]) > float(record["slope_min"])
    ]
    return sorted(filtered, key=lambda r: (int(r["k_int"]), int(r["duration"])))


def _build_base_probe_records(full_data: Dict) -> List[Dict]:
    records: List[Dict] = []
    for per_duration in full_data["by_duration"]:
        duration = int(per_duration["duration"])
        for sample in per_duration["samples"]:
            band = _pick_slope_interval(sample["bands"])
            if band is None:
                continue
            if sample.get("fuel_rate") is not None:
                amp_actual = float(sample["fuel_rate"])
            else:
                amp_actual = float(sample["xmv07_rate"])
            k_int = _bucket_key(amp_actual)
            records.append({
                "duration": duration,
                "amp_actual": amp_actual,
                "amp_actual_min": amp_actual,
                "amp_actual_max": amp_actual,
                "amplitude": round(k_int / 100.0, 2),
                "k_int": k_int,
                "slope_min": float(band[0]),
                "slope_max": float(band[1]),
                "source_count": 1,
                "region_type": "base",
            })
    return _aggregate_probe_records(records)


def _build_cond_probe_records(cond_data: Dict, registry) -> List[Dict]:
    records: List[Dict] = []
    refine_records: List[Dict] = []
    for row in cond_data["expanded_boundary"]:
        duration = int(row["duration"])
        if duration % REGION_DELTA_T != 0:
            continue
        cond_amp = float(row.get("stable_amp", row["lower_target_amp"]))
        base_amp = float(row["regular_amp"])
        slope_min = row.get("stable_slope_min", row.get("slope_min"))
        slope_max = row.get("stable_slope_max", row.get("slope_max"))
        extreme_amp = row.get("extreme_amp")
        extreme_slope = row.get("extreme_slope")
        if slope_min is not None and slope_max is not None:
            band = (float(slope_min), float(slope_max))
            feasible = band[1] > band[0]
            result = {
                "haz_step_fuel_only": None,
                "horizon": int(duration) + HOLD,
                "bands": [],
            }
        else:
            result = solve_cond_steam_sp_at_point(cond_amp, duration, registry)
            band = _pick_slope_interval(result["bands"])
            feasible = band is not None
        refine_records.append({
            "duration": duration,
            "base_amp": base_amp,
            "cond_lower": cond_amp,
            "stable_amp": cond_amp,
            "expansion": round(base_amp - cond_amp, 6),
            "feasible": feasible,
            "haz_step_fuel_only": result["haz_step_fuel_only"],
            "horizon": result["horizon"],
            "slope_min": round(band[0], 6) if band else None,
            "slope_max": round(band[1], 6) if band else None,
            "slope_width": round(band[1] - band[0], 6) if band else None,
            "extreme_amp": None if extreme_amp is None else round(float(extreme_amp), 6),
            "extreme_slope": None if extreme_slope is None else round(float(extreme_slope), 6),
            "tightness": None if extreme_amp is None or cond_amp <= 0 else round(float(extreme_amp) / cond_amp, 6),
        })
        if not feasible:
            continue
        k_int = _bucket_key(cond_amp)
        records.append({
            "duration": duration,
            "amp_actual": cond_amp,
            "amp_actual_min": cond_amp,
            "amp_actual_max": cond_amp,
            "amplitude": round(k_int / 100.0, 2),
            "k_int": k_int,
            "slope_min": float(band[0]),
            "slope_max": float(band[1]),
            "source_count": 1,
            "region_type": "cond",
            "base_amp": base_amp,
            "stable_amp": cond_amp,
            "extreme_amp": None if extreme_amp is None else float(extreme_amp),
            "boundary_role": "stable",
            "force_singleton": False,
        })
        if extreme_amp is not None:
            extreme_amp_value = float(extreme_amp)
            extreme_result = solve_cond_steam_sp_at_point(extreme_amp_value, duration, registry)
            extreme_band = _pick_slope_interval(extreme_result["bands"])
            if extreme_band is None:
                continue
            extreme_k_int = _bucket_key(extreme_amp_value)
            records.append({
                "duration": duration,
                "amp_actual": extreme_amp_value,
                "amp_actual_min": extreme_amp_value,
                "amp_actual_max": extreme_amp_value,
                "amplitude": round(extreme_k_int / 100.0, 2),
                "k_int": extreme_k_int,
                "slope_min": float(extreme_band[0]),
                "slope_max": float(extreme_band[1]),
                "source_count": 1,
                "region_type": "cond",
                "base_amp": base_amp,
                "stable_amp": cond_amp,
                "extreme_amp": extreme_amp_value,
                "boundary_role": "extreme",
                "force_singleton": True,
            })

    COND_STAGE3_REFINE_PATH.write_text(
        json.dumps(refine_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return _aggregate_probe_records(records)


def _build_neighbor_pairs(records: List[Dict]) -> List[Tuple[int, int, float]]:
    by_k: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for idx, record in enumerate(records):
        if record.get("force_singleton"):
            continue
        by_k[int(record["k_int"])].append((int(record["duration"]), idx))
    pairs: List[Tuple[int, int, float]] = []
    for items in by_k.values():
        items.sort()
        for left, right in zip(items, items[1:]):
            if right[0] - left[0] != REGION_DELTA_T:
                continue
            li = left[1]
            ri = right[1]
            dv = slope_distance(
                (float(records[li]["slope_min"]), float(records[li]["slope_max"])),
                (float(records[ri]["slope_min"]), float(records[ri]["slope_max"])),
            )
            pairs.append((li, ri, dv))
    return pairs


def _derive_merge_threshold(records: List[Dict]) -> Tuple[float, float]:
    widths = [
        float(record["slope_max"]) - float(record["slope_min"])
        for record in records
        if float(record["slope_max"]) > float(record["slope_min"])
    ]
    width_median = statistics.median(widths) if widths else 0.0
    return REGION_GAMMA_MERGE * width_median, width_median


def _build_region_payload(records: List[Dict], region_type: str, method: str) -> Dict:
    pairs = _build_neighbor_pairs(records)
    d_merge, width_median = _derive_merge_threshold(records)
    uf = UnionFind(len(records))
    for left, right, dv in pairs:
        if dv <= d_merge:
            uf.union(left, right)

    groups = uf.groups()
    regions = []
    sizes = sorted((len(members) for members in groups.values()), reverse=True)
    d_values = [dv for _, _, dv in pairs]
    for members in sorted(groups.values(), key=lambda members: (-len(members), members)):
        ordered = sorted(members, key=lambda idx: int(records[idx]["duration"]))
        region_records = [records[idx] for idx in ordered]
        t_min = int(region_records[0]["duration"])
        t_max = int(region_records[-1]["duration"])
        amp_actual_min = min(
            float(record.get("amp_actual_min", record["amp_actual"]))
            for record in region_records
        )
        amp_actual_max = max(
            float(record.get("amp_actual_max", record["amp_actual"]))
            for record in region_records
        )
        slope_min = max(float(record["slope_min"]) for record in region_records)
        slope_max = min(float(record["slope_max"]) for record in region_records)
        if slope_max <= slope_min:
            continue
        corner_points = []
        for record in (region_records[0], region_records[-1]):
            corner = {
                "dur": int(record["duration"]),
                "amp": round(float(record.get("amp_actual", record["amp_actual_min"])), 6),
                "slope_min": round(float(record["slope_min"]), 6),
                "slope_max": round(float(record["slope_max"]), 6),
            }
            if not corner_points or corner_points[-1] != corner:
                corner_points.append(corner)
        regions.append({
            "region_id": f"R_{len(regions)+1:03d}",
            "region_type": region_type,
            "boundary_role": str(region_records[0].get("boundary_role", "default")),
            "n_members": len(region_records),
            "n_fine_pts": sum(int(record.get("source_count", 1)) for record in region_records),
            "t_range": [t_min, t_max],
            "k_range": [round(amp_actual_min, 6), round(amp_actual_max, 6)],
            "amp_actual_range": [
                round(amp_actual_min, 6),
                round(amp_actual_max, 6),
            ],
            "stable_amp_range": [
                round(min(float(record.get("stable_amp", record["amp_actual"])) for record in region_records), 6),
                round(max(float(record.get("stable_amp", record["amp_actual"])) for record in region_records), 6),
            ],
            "extreme_amp_range": [
                round(min(float(record.get("extreme_amp", record["amp_actual"])) for record in region_records), 6),
                round(max(float(record.get("extreme_amp", record["amp_actual"])) for record in region_records), 6),
            ],
            "n_corners": len(corner_points),
            "corner_points": corner_points,
            "B_omega": {
                "alpha_minus": round(slope_min, 6),
                "alpha_plus": round(slope_max, 6),
                "width": round(slope_max - slope_min, 6),
            },
        })

    return {
        "method": method,
        "config": {
            "delta_T": REGION_DELTA_T,
            "delta_K": REGION_DELTA_K,
            "gamma_merge": REGION_GAMMA_MERGE,
        },
        "thresholds": {
            "W_median": round(width_median, 6),
            "delta_merge": round(d_merge, 6),
            "D_median": round(statistics.median(d_values), 6) if d_values else 0.0,
        },
        "stats": {
            "initial_reps": len(records),
            "n_regions": len(regions),
            "n_pairs": len(pairs),
            "region_sizes_top10": sizes[:10],
        },
        "regions": regions,
    }


def _write_base_regions(full_data: Dict, cfg: Dict) -> Dict:
    base_payload = _build_region_payload(
        _build_base_probe_records(full_data),
        region_type="base",
        method="artifact_current_chain_T_only_merge_base",
    )
    base_payload["source_stage3_full"] = cfg["stage3_full_path"].relative_to(_ROOT).as_posix()
    base_payload["source_stage2_base"] = full_data["base_results_path"]
    cfg["stage3_base_region_path"].write_text(
        json.dumps(base_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return base_payload


def _write_combined_regions(full_data: Dict, base_payload: Dict, registry, cfg: Dict) -> Optional[Dict]:
    cond_path = cfg.get("conditional_results_path")
    if not cond_path:
        return None
    cond_data = json.loads(cond_path.read_text(encoding="utf-8"))
    cond_payload = _build_region_payload(
        _build_cond_probe_records(cond_data, registry),
        region_type="cond",
        method="artifact_current_chain_T_only_merge_cond",
    )

    combined_regions = []
    next_id = 1
    for source in (base_payload["regions"], cond_payload["regions"]):
        for region in source:
            region_copy = dict(region)
            region_copy["region_id"] = f"R_{next_id:03d}"
            combined_regions.append(region_copy)
            next_id += 1

    combined_payload = {
        "method": "T_only_merge_gamma_based_base_plus_cond",
        "config": {
            "delta_T": REGION_DELTA_T,
            "delta_K": REGION_DELTA_K,
            "gamma_merge": REGION_GAMMA_MERGE,
        },
        "stats": {
            "base_regions": len(base_payload["regions"]),
            "cond_regions": len(cond_payload["regions"]),
            "total_regions": len(combined_regions),
            "cond_d_merge": cond_payload["thresholds"]["delta_merge"],
            "cond_W_median": cond_payload["thresholds"]["W_median"],
        },
        "source_stage3_full": cfg["stage3_full_path"].relative_to(_ROOT).as_posix(),
        "source_stage2_base": full_data["base_results_path"],
        "source_stage2_conditional": cond_path.relative_to(_ROOT).as_posix(),
        "regions": combined_regions,
    }
    cfg["stage3_combined_region_path"].write_text(
        json.dumps(combined_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return combined_payload





def main(argv: Optional[List[str]] = None):

    """Unified Stage 3 main chain for boiler/TE-like platforms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="boiler", choices=["boiler", "te"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = get_stage3_platform_config(args.target)

    base_path = cfg["base_results_path"]
    base_path_label = base_path.relative_to(_ROOT).as_posix()
    out_path = cfg["stage3_full_path"]

    out_path.parent.mkdir(parents=True, exist_ok=True)



    base = json.loads(base_path.read_text(encoding="utf-8"))

    rows = []

    for b in base["boundaries"]:

        if b.get("direction") != "pos":

            continue

        rl = b.get("lower_target_amp")

        rh = b.get("upper_target_amp")

        if rl is None or rh is None:

            continue

        rows.append((int(b["duration"]), float(rl), float(rh)))



    sample_duration_mod = int(cfg.get("sample_duration_mod", 50))
    rows = [r for r in rows if r[0] % sample_duration_mod == 0]
    print(f"[stage3-{cfg['target']}] processing {len(rows)} durations "
          f"(subsampled @{sample_duration_mod}step) from {base_path.name}")



    from generic_runtime_executor import load_injection_registry
    manifest = cfg["manifest_path"]
    registry = load_injection_registry(manifest)



    t0 = time.time()

    all_results = []
    workers = max(1, int(args.workers))

    if cfg["target"] == "te" and workers > 1:
        tasks = [(d, rl, rh, cfg) for d, rl, rh in rows]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            mapped = pool.map(_process_one_duration_task, tasks)
            all_results = list(mapped)
        all_results.sort(key=lambda row: int(row["duration"]))
        for i, out in enumerate(all_results):
            d = int(out["duration"])
            rmf = out["rate_min_feasible"]
            samples = out.get("samples", [])
            f0 = samples[0]["feasible"] if samples else None
            f1 = samples[-1]["feasible"] if samples else None
            if rmf:
                rmf_rate_key = "fuel_rate" if "fuel_rate" in rmf else "xmv07_rate"
                rmf_str = f"rmf={rmf[rmf_rate_key]:.5f} (rounds={rmf['bisect_rounds']})"
            else:
                rmf_str = "rmf=None"
            print(f"  [{i+1}/{len(all_results)}] d={d:>4}  "
                  f"base=[{out['base_rate_lo']:.4f}, {out['base_rate_hi']:.4f}]  "
                  f"q=0 feas={f0}  q=1 feas={f1}  {rmf_str}")
    else:
        for i, (d, rl, rh) in enumerate(rows):

            per_t0 = time.time()

            out = process_one_duration(d, rl, rh, registry, cfg)

            all_results.append(out)

            per_dt = time.time() - per_t0

            rmf = out["rate_min_feasible"]

            f0 = out["samples"][0]["feasible"]

            f1 = out["samples"][-1]["feasible"]

            if rmf:
                rmf_rate_key = "fuel_rate" if "fuel_rate" in rmf else "xmv07_rate"
                rmf_str = f"rmf={rmf[rmf_rate_key]:.5f} (rounds={rmf['bisect_rounds']})"
            else:
                rmf_str = "rmf=None"

            print(f"  [{i+1}/{len(rows)}] d={d:>4}  "

                  f"base=[{rl:.4f}, {rh:.4f}]  "

                  f"q=0 feas={f0}  q=1 feas={f1}  {rmf_str}  ({per_dt:.1f}s)")



    elapsed = time.time() - t0

    out_data = {

        "system_id": cfg["system_id"],
        "manip_point": cfg["manip_point"],
        "alarm_id": cfg["alarm_id"],
        "spoof_field": cfg["spoof_field"],

        "sim_steps": SIM,

        "n_rate_samples": N_RATE_SAMPLES,

        "method": "static_then_dynamic_ra_shrink_same_duration",
        "base_results_path": base_path_label,

        "elapsed_s": round(elapsed, 1),

        "by_duration": all_results,

    }

    if cfg["target"] == "boiler":
        out_data["anchor_steam_sp"] = ANCHOR_STEAM_SP
        out_data["boiler_cut_threshold"] = BOILER_CUT_TH

    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2)

                        + "\n", encoding="utf-8")

    print(f"\n[stage3-{cfg['target']}] saved {out_path}")
    if cfg["target"] == "te" and workers > 1:
        print(f"[stage3-{cfg['target']}] skipped region writes for workers={workers} point-only validation")
    else:
        base_regions = _write_base_regions(out_data, cfg)
        combined_regions = _write_combined_regions(out_data, base_regions, registry, cfg)
        print(f"[stage3-{cfg['target']}] saved {cfg['stage3_base_region_path']} "
              f"({base_regions['stats']['n_regions']} regions)")
        if combined_regions is not None:
            print(f"[stage3-{cfg['target']}] saved {cfg['stage3_combined_region_path']} "
                  f"({combined_regions['stats']['total_regions']} regions)")
    print(f"[stage3-{cfg['target']}] elapsed {elapsed:.1f}s for {len(rows)} durations")





if __name__ == "__main__":

    main()


# Base Region Search - Section 3.3.1

"""Search Stage 2 base hazard-driving boundaries inside the artifact.

This single script still contains multiple boundary-search entrypoints for
artifact compatibility:

- ``run_boundary_search_target_intervals`` is the recommended artifact
  entrypoint for the current Stage 2 base-region search flow.
- ``run_boundary_search`` keeps the earlier coarse-bisect path available for
  backward-compatible reruns.
- ``run_boundary_search_cegis`` preserves the duration-refinement workflow
  used by older published/diagnostic outputs.
"""

from __future__ import annotations



import json
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


import sys



_STAGE4_DIR = str(Path(__file__).resolve().parents[1] / "stage4_executor")

if _STAGE4_DIR not in sys.path:

    sys.path.insert(0, _STAGE4_DIR)

_SRC_DIR = str(Path(__file__).resolve().parents[1])

if _SRC_DIR not in sys.path:

    sys.path.insert(0, _SRC_DIR)



from generic_runtime_executor import (

    load_injection_registry,

    load_rules_from_manifest,

    load_simulation_module,

    simulate,

    score_trace,

)





# ============================================================================

# Data Structures

# ============================================================================



@dataclass

class SearchDomain:

    """..."""

    point: str

    input_col: str

    default_value: float

    value_range: Tuple[float, float]

    amp_max: float

    duration_min: int = 100

    duration_max: int = 1000

    duration_step: int = 100

    t_start: int = 0





@dataclass

class BoundaryResult:
    """..."""

    point: str

    direction: str

    duration: int

    lower_safe_amp: Optional[float] = None

    lower_target_amp: Optional[float] = None

    upper_target_amp: Optional[float] = None

    upper_unsafe_amp: Optional[float] = None

    converged: bool = False
    rounds_used: int = 0


_WORKER_CTX: Dict[str, Any] = {}
PRIMARY_SEARCH_METHOD = "target_intervals"
POINT_CANONICAL_ALARM_SCOPE: Dict[str, List[str]] = {
    "simulation_xmv_04": ["P-TEP-ACFEED-TRACK"],
    "simulation_xmv_07": ["P-TEP-SEP-LEVEL-TRACK"],
    "simulation_xmv_08": ["P-TEP-STRIPPER-LEVEL-TRACK"],
}




# ============================================================================

# Search Domain Construction

# ============================================================================



def build_search_domain(

    *,

    registry: Mapping[str, Dict[str, Any]],

    point_name: str,

    duration_min: int = 100,

    duration_max: int = 1000,

    duration_step: int = 100,

    t_start: int = 0,

) -> SearchDomain:

    """..."""

    if point_name not in registry:

        raise ValueError(f"Point {point_name} not found in injection registry. Available: {list(registry.keys())}")

    spec = registry[point_name]

    rng = spec.get("range")

    value_range = tuple(rng) if rng and len(rng) == 2 else (0.0, 1.0)

    return SearchDomain(

        point=point_name,

        input_col=spec.get("field", point_name),

        default_value=spec.get("default_value", 0.0),

        value_range=value_range,

        amp_max=spec.get("rate", 1.0),

        duration_min=duration_min,

        duration_max=duration_max,

        duration_step=duration_step,

        t_start=t_start,

    )





# ============================================================================

# Classification

# ============================================================================



def classify_result(
    result: Mapping[str, Any],
    *,
    alarm_cols: List[str],
    alarm_cap: int,
    forbidden_alarm_ids: Optional[List[str]] = None,
    required_alarm_ids: Optional[List[str]] = None,
) -> str:
    """..."""

    if str(result.get("status", "")).strip() != "ok":

        return "unsafe"

    if result.get("first_hazard_step") is None:

        return "safe"

    alarm_count = int(result.get("prehazard_alarm_rule_ids_count", 0))

    if alarm_count > alarm_cap:
        return "unsafe"
    triggered_ids = result.get("alarm_rule_ids", [])
    if isinstance(triggered_ids, str):
        triggered_ids = [x.strip() for x in triggered_ids.replace(",", "|").split("|") if x.strip()]
    triggered_set = set(str(x) for x in triggered_ids)
    if required_alarm_ids:
        for rid in required_alarm_ids:
            if rid not in triggered_set:
                return "unsafe"
    # Check forbidden alarms
    if forbidden_alarm_ids:
        for fid in forbidden_alarm_ids:
            if fid in triggered_set:
                return "unsafe"
    return "target"


def scoped_alarm_view(
    *,
    point_name: str,
    score: Mapping[str, Any],
) -> Dict[str, Any]:
    required = POINT_CANONICAL_ALARM_SCOPE.get(str(point_name), [])
    ids = [str(x) for x in score.get("prehazard_alarm_rule_ids", [])]
    if not required:
        return {
            "first_hazard_step": score.get("first_hazard_step"),
            "prehazard_alarm_rule_ids": ids,
            "prehazard_alarm_rule_ids_count": len(set(ids)),
            "required_alarm_ids": [],
        }
    scoped_ids = [aid for aid in ids if aid in required]
    return {
        "first_hazard_step": score.get("first_hazard_step"),
        "prehazard_alarm_rule_ids": sorted(set(scoped_ids)),
        "prehazard_alarm_rule_ids_count": len(set(scoped_ids)),
        "required_alarm_ids": list(required),
    }


def _build_scan_points(amp_max: float, coarse_span: float) -> List[float]:
    coarse_span = max(1e-6, float(coarse_span))
    n_steps = max(1, int(math.ceil(float(amp_max) / coarse_span)))
    pts = [round(i * float(amp_max) / n_steps, 10) for i in range(n_steps + 1)]
    pts[0] = 0.0
    pts[-1] = float(amp_max)
    return sorted(set(pts))


def _bisect_transition(
    classify_amp: Callable[[float], str],
    lo: float,
    hi: float,
    cls_lo: str,
    cls_hi: str,
    cache: Dict[float, str],
    amp_tol: float,
) -> float:
    while (hi - lo) > amp_tol:
        mid = round((lo + hi) / 2.0, 10)
        cls_mid = cache.get(mid)
        if cls_mid is None:
            cls_mid = classify_amp(mid)
            cache[mid] = cls_mid
        if cls_mid == cls_lo:
            lo = mid
        else:
            hi = mid
    return hi


def find_target_intervals(
    *,
    classify_amp: Callable[[float], str],
    amp_max: float,
    coarse_span: float = 0.03125,
    amp_tol: float = 1e-4,
) -> Tuple[List[Dict[str, float]], int]:
    """Find all target intervals on the amplitude axis.

    This routine does not assume any monotone class ordering. It scans the
    full axis coarsely, then refines every class transition to amp_tol.
    """
    cache: Dict[float, str] = {}
    grid = _build_scan_points(float(amp_max), float(coarse_span))
    for amp in grid:
        cache[amp] = classify_amp(amp)

    sorted_pts = sorted(cache)
    segments: List[Dict[str, float | str]] = []
    current_lo = sorted_pts[0]
    current_class = cache[current_lo]

    for idx in range(len(sorted_pts) - 1):
        lo = sorted_pts[idx]
        hi = sorted_pts[idx + 1]
        cls_lo = cache[lo]
        cls_hi = cache[hi]
        if cls_lo == cls_hi:
            continue
        boundary = _bisect_transition(
            classify_amp=classify_amp,
            lo=lo,
            hi=hi,
            cls_lo=cls_lo,
            cls_hi=cls_hi,
            cache=cache,
            amp_tol=amp_tol,
        )
        segments.append({
            "class": current_class,
            "rate_lo": float(current_lo),
            "rate_hi": float(boundary),
        })
        current_lo = boundary
        current_class = cls_hi

    segments.append({
        "class": current_class,
        "rate_lo": float(current_lo),
        "rate_hi": float(amp_max),
    })

    target_intervals = [
        {"rate_lo": float(seg["rate_lo"]), "rate_hi": float(seg["rate_hi"])}
        for seg in segments
        if seg["class"] == "target"
    ]
    return target_intervals, len(cache)




# ============================================================================

# Coarse Bisect (Phase 1)

# ============================================================================



def run_coarse_bisect(
    *,

    simulate_fn: Callable,

    domain: SearchDomain,

    direction: str,

    duration: int,

    alarm_cols: List[str],
    alarm_cap: int,
    forbidden_alarm_ids: Optional[List[str]] = None,
    required_alarm_ids: Optional[List[str]] = None,
    steps: int = 1000,

    hold_steps: int = 100,

    max_probes: int = 40,

) -> Dict[str, Any]:

    """..."""

    sign = 1.0 if direction == "pos" else -1.0

    probes: List[Dict[str, Any]] = []

    best: Dict[str, Optional[float]] = {

        "lower_safe": None, "lower_target": None,

        "upper_target": None, "upper_unsafe": None,

    }

    probes_used = [0]



    def _probe(amp_abs: float) -> str:

        """Execute one probe and record result."""

        probes_used[0] += 1

        task = _make_task(domain, sign * amp_abs, duration, steps, hold_steps)

        result = simulate_fn(task)

        cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                              forbidden_alarm_ids=forbidden_alarm_ids,
                              required_alarm_ids=required_alarm_ids)
        probes.append({

            "amp_abs": amp_abs,

            "direction": direction,

            "class": cls,

            "first_hazard_step": result.get("first_hazard_step"),

            "prehazard_alarm_count": result.get("prehazard_alarm_rule_ids_count", 0),

            "alarm_rule_ids": "|".join(result.get("alarm_rule_ids", [])),

        })

        return cls



    # Phase 1: Initial scan to find class distribution

    # Probe at a few points to locate the target region

    scan_points = [domain.amp_max * f for f in [0.5, 0.25, 0.75, 0.125, 0.875]]

    for amp_abs in scan_points:

        if probes_used[0] >= max_probes:

            break

        cls = _probe(amp_abs)

        if cls == "safe":

            if best["lower_safe"] is None or amp_abs > best["lower_safe"]:

                best["lower_safe"] = amp_abs

        elif cls == "target":

            if best["lower_target"] is None or amp_abs < best["lower_target"]:

                best["lower_target"] = amp_abs

            if best["upper_target"] is None or amp_abs > best["upper_target"]:

                best["upper_target"] = amp_abs

        elif cls == "unsafe":

            if best["upper_unsafe"] is None or amp_abs < best["upper_unsafe"]:

                best["upper_unsafe"] = amp_abs



    # Phase 2: Binary search for lower boundary (safe ↔ target)

    if best["lower_target"] is not None:

        lo = best["lower_safe"] if best["lower_safe"] is not None else 0.0

        hi = best["lower_target"]

        while probes_used[0] < max_probes and (hi - lo) > 1e-4:

            mid = (lo + hi) / 2.0

            cls = _probe(mid)

            if cls == "safe":

                best["lower_safe"] = mid

                lo = mid

            else:  # target or unsafe

                if cls == "target":

                    if best["lower_target"] is None or mid < best["lower_target"]:

                        best["lower_target"] = mid

                hi = mid



    # Phase 3: Binary search for upper boundary (target ↔ unsafe)

    if best["upper_target"] is not None or best["lower_target"] is not None:

        upper_t = best["upper_target"] or best["lower_target"]

        lo = upper_t

        hi = best["upper_unsafe"] if best["upper_unsafe"] is not None else domain.amp_max

        while probes_used[0] < max_probes and (hi - lo) > 1e-4:

            mid = (lo + hi) / 2.0

            cls = _probe(mid)

            if cls == "unsafe":

                if best["upper_unsafe"] is None or mid < best["upper_unsafe"]:

                    best["upper_unsafe"] = mid

                hi = mid

            else:  # target or safe

                if cls == "target":

                    if best["upper_target"] is None or mid > best["upper_target"]:

                        best["upper_target"] = mid

                lo = mid



    return {**best, "probes": probes}





# ============================================================================

# CEGIS Refinement (Phase 2)

# ============================================================================



def run_cegis_refine(
    *,

    simulate_fn: Callable,

    domain: SearchDomain,

    direction: str,

    duration: int,

    initial_boundary: Dict[str, Any],

    alarm_cols: List[str],
    alarm_cap: int,
    forbidden_alarm_ids: Optional[List[str]] = None,
    required_alarm_ids: Optional[List[str]] = None,
    steps: int = 1000,

    hold_steps: int = 100,

    amp_tol: float = 0.01,

    max_rounds: int = 10,

) -> BoundaryResult:

    """..."""

    lower_safe = initial_boundary.get("lower_safe")

    lower_target = initial_boundary.get("lower_target")

    upper_target = initial_boundary.get("upper_target")

    upper_unsafe = initial_boundary.get("upper_unsafe")

    sign = 1.0 if direction == "pos" else -1.0

    rounds = 0



    for rounds in range(1, max_rounds + 1):

        lower_converged = True

        upper_converged = True



        # Refine lower boundary (safe <-> target)

        if lower_safe is not None and lower_target is not None:

            if abs(lower_target - lower_safe) > amp_tol:

                lower_converged = False

                mid = (lower_safe + lower_target) / 2.0

                task = _make_task(domain, sign * mid, duration, steps, hold_steps)

                result = simulate_fn(task)

                cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                                      forbidden_alarm_ids=forbidden_alarm_ids,
                                      required_alarm_ids=required_alarm_ids)
                if cls == "safe":

                    lower_safe = mid

                else:

                    lower_target = mid



        # Refine upper boundary (target <-> unsafe)

        if upper_target is not None and upper_unsafe is not None:

            if abs(upper_unsafe - upper_target) > amp_tol:

                upper_converged = False

                mid = (upper_target + upper_unsafe) / 2.0

                task = _make_task(domain, sign * mid, duration, steps, hold_steps)

                result = simulate_fn(task)

                cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                                      forbidden_alarm_ids=forbidden_alarm_ids,
                                      required_alarm_ids=required_alarm_ids)
                if cls == "target":

                    upper_target = mid

                elif cls == "unsafe":

                    upper_unsafe = mid

                else:  # safe (unexpected at upper boundary)

                    upper_target = mid



        if lower_converged and upper_converged:

            break



    return BoundaryResult(

        point=domain.point,

        direction=direction,

        duration=duration,

        lower_safe_amp=lower_safe,

        lower_target_amp=lower_target,

        upper_target_amp=upper_target,

        upper_unsafe_amp=upper_unsafe,

        converged=(lower_converged and upper_converged),

        rounds_used=rounds,

    )





def _make_task(domain: SearchDomain, amp: float, duration: int, steps: int,
               hold_steps: int = 100) -> Dict:
    """..."""

    effective_steps = min(steps, duration + max(1, int(hold_steps)))

    return {
        "injection": {
            "point": domain.point,

            "amp": amp,

            "duration": duration,

            "t_start": domain.t_start,

        },

        "runtime_args": {

            "steps": effective_steps,

            "early_stop_mode": "early_stop",

            "search_mode": "boundary_search",

            "injection_mode": "override_ramp",

        },
    }


def _ensure_worker_ctx(manifest_path: str) -> Dict[str, Any]:
    ctx = _WORKER_CTX.get(manifest_path)
    if ctx is None:
        manifest = Path(manifest_path)
        registry = load_injection_registry(manifest)
        rules = load_rules_from_manifest(manifest)
        sim_module, sim_init_kwargs = load_simulation_module(manifest)
        ctx = {
            "registry": registry,
            "rules": rules,
            "sim_module": sim_module,
            "sim_init_kwargs": sim_init_kwargs,
            "alarm_cols": [spec["margin_col"] for spec in rules.get("alarms", {}).values()],
        }
        _WORKER_CTX[manifest_path] = ctx
    return ctx


def _classify_point_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _ensure_worker_ctx(str(payload["manifest_path"]))
    manifest = Path(payload["manifest_path"])
    registry = ctx["registry"]
    domain = build_search_domain(
        registry=registry,
        point_name=str(payload["point_name"]),
        duration_min=int(payload["duration"]),
        duration_max=int(payload["duration"]),
        duration_step=1,
        t_start=int(payload.get("t_start", 0)),
    )
    sign = 1.0 if str(payload.get("direction", "pos")) == "pos" else -1.0
    duration = int(payload["duration"])
    steps = int(payload["steps"])
    hold_steps = int(payload["hold_steps"])
    alarm_cap = int(payload["alarm_cap"])
    forbidden_alarm_ids = payload.get("forbidden_alarm_ids")
    required_alarm_ids = payload.get("required_alarm_ids")

    probes: List[Dict[str, Any]] = []

    def classify_amp(amp_abs: float) -> str:
        task = _make_task(domain, sign * amp_abs, duration, steps, hold_steps)
        result = simulate(
            task,
            registry=ctx["registry"],
            rules=ctx["rules"],
            sim_module=ctx["sim_module"],
            sim_init_kwargs=ctx["sim_init_kwargs"],
        )
        rows = result.get("rows", [])
        scored = result.get("score", {})
        score_view = scoped_alarm_view(
            point_name=str(payload["point_name"]),
            score=scored,
        )
        cls = classify_result(
            {
                "status": result.get("status", "ok"),
                "first_hazard_step": score_view.get("first_hazard_step"),
                "prehazard_alarm_rule_ids_count": score_view.get("prehazard_alarm_rule_ids_count", 0),
                "alarm_rule_ids": score_view.get("prehazard_alarm_rule_ids", []),
            },
            alarm_cols=ctx["alarm_cols"],
            alarm_cap=alarm_cap,
            forbidden_alarm_ids=forbidden_alarm_ids,
            required_alarm_ids=score_view.get("required_alarm_ids") or required_alarm_ids,
        )
        probes.append({
            "amp_abs": float(amp_abs),
            "direction": str(payload.get("direction", "pos")),
            "class": cls,
            "first_hazard_step": score_view.get("first_hazard_step"),
            "prehazard_alarm_count": score_view.get("prehazard_alarm_rule_ids_count", 0),
            "alarm_rule_ids": "|".join(score_view.get("prehazard_alarm_rule_ids", [])),
            "duration": duration,
            "phase": "target_interval",
        })
        return cls

    target_intervals, n_simulations = find_target_intervals(
        classify_amp=classify_amp,
        amp_max=domain.amp_max,
        coarse_span=float(payload.get("coarse_span", 0.03125)),
        amp_tol=float(payload.get("amp_tol", 1e-4)),
    )
    boundary = {
        "direction": str(payload.get("direction", "pos")),
        "duration": duration,
        "lower_safe_amp": None,
        "lower_target_amp": None,
        "upper_target_amp": None,
        "upper_unsafe_amp": None,
        "target_intervals": target_intervals,
        "target_interval_count": len(target_intervals),
        "converged": True,
        "rounds_used": 0,
        "n_simulations": int(n_simulations),
    }
    if len(target_intervals) == 1:
        boundary["lower_target_amp"] = float(target_intervals[0]["rate_lo"])
        boundary["upper_target_amp"] = float(target_intervals[0]["rate_hi"])

    return {
        "manifest_path": str(manifest),
        "boundary": boundary,
        "probe_trace": probes,
    }




# ============================================================================

# Full Search Orchestrator

# ============================================================================



def run_boundary_search(
    *,

    manifest_path: Path,

    point_name: str,

    output_root: Path,

    duration_min: int = 100,

    duration_max: int = 1000,

    duration_step: int = 100,

    t_start: int = 0,

    steps: int = 1000,

    hold_steps: int = 100,
    alarm_cap: int = 2,
    forbidden_alarm_ids: Optional[List[str]] = None,
    required_alarm_ids: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
    amp_tol: float = 0.01,

    max_rounds: int = 10,

    workers: int = 4,

) -> Dict[str, Any]:

    """..."""

    #
    registry = load_injection_registry(manifest_path)

    rules = load_rules_from_manifest(manifest_path)

    sim_module, sim_init_kwargs = load_simulation_module(manifest_path)



    # Build search domain from registry

    domain = build_search_domain(

        registry=registry,

        point_name=point_name,

        duration_min=duration_min,

        duration_max=duration_max,

        duration_step=duration_step,

        t_start=t_start,

    )



    # Extract alarm margin columns from rules

    alarm_cols = [spec["margin_col"] for spec in rules.get("alarms", {}).values()]



    # Build simulate_fn using generic executor

    trace_dir = output_root / "traces"

    trace_dir.mkdir(parents=True, exist_ok=True)

    _sim_counter = [0]



    def simulate_fn(task: Dict) -> Dict:

        try:

            result = simulate(task, registry=registry, rules=rules,

                              sim_module=sim_module, sim_init_kwargs=sim_init_kwargs)

        except Exception as e:

            #
            return {"status": "error", "error": str(e),

                    "first_hazard_step": None, "prehazard_alarm_rule_ids_count": 99,

                    "prehazard_alarm_rule_ids": [], "alarm_rule_ids": [], "rows": []}

        # Score the trace

        rows = result.get("rows", [])

        scored = result.get("score", {})

        # Save trace CSV

        if rows:

            _sim_counter[0] += 1

            inj = task.get("injection", {})

            trace_name = f"sim_{_sim_counter[0]:04d}_d{inj.get('duration', 0)}_{inj.get('amp', 0):.6f}.csv"

            trace_path = trace_dir / trace_name

            with open(trace_path, "w", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

                writer.writeheader()

                writer.writerows(rows)

        # Return in the format classify_result expects

        return {

            "status": result.get("status", "ok"),

            "first_hazard_step": scored.get("first_hazard_step"),

            "prehazard_alarm_rule_ids_count": scored.get("prehazard_alarm_rule_ids_count", 0),

            "prehazard_alarm_rule_ids": scored.get("prehazard_alarm_rule_ids", []),

            "alarm_rule_ids": scored.get("prehazard_alarm_rule_ids", []),

            "rows": rows,

        }

    # Run search for each duration × direction

    results: List[BoundaryResult] = []

    durations = list(range(domain.duration_min, domain.duration_max + 1, domain.duration_step))

    search_directions = directions if directions else ["pos", "neg"]

    probe_trace: List[Dict[str, Any]] = []



    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)



    for dur in durations:

        for direction in search_directions:

            print(f"  [{point_name}] duration={dur} direction={direction} ...", flush=True)

            # Phase 1: coarse bisect

            initial = run_coarse_bisect(

                simulate_fn=simulate_fn,

                domain=domain,

                direction=direction,

                duration=dur,

                alarm_cols=alarm_cols,

                alarm_cap=alarm_cap,

                forbidden_alarm_ids=forbidden_alarm_ids,
                required_alarm_ids=required_alarm_ids,
                steps=steps,
                hold_steps=hold_steps,
            )

            probe_trace.extend(

                {**p, "duration": dur, "phase": "coarse"} for p in initial.get("probes", [])

            )



            # Phase 2: CEGIS refine

            boundary = run_cegis_refine(

                simulate_fn=simulate_fn,

                domain=domain,

                direction=direction,

                duration=dur,

                initial_boundary=initial,

                alarm_cols=alarm_cols,

                alarm_cap=alarm_cap,

                forbidden_alarm_ids=forbidden_alarm_ids,
                required_alarm_ids=required_alarm_ids,
                steps=steps,
                hold_steps=hold_steps,
                amp_tol=amp_tol,

                max_rounds=max_rounds,

            )

            results.append(boundary)



    # Save boundary results

    summary = {

        "point": point_name,

        "backend": "physics",

        "domain": {

            "amp_max": domain.amp_max,

            "default_value": domain.default_value,

            "value_range": list(domain.value_range),

            "duration_min": domain.duration_min,

            "duration_max": domain.duration_max,

            "duration_step": domain.duration_step,

        },

        "semantic_filters": {
            "forbidden_alarm_ids": forbidden_alarm_ids or [],
            "required_alarm_ids": required_alarm_ids or [],
        },
        "boundaries": [
            {

                "direction": r.direction,

                "duration": r.duration,

                "lower_safe_amp": r.lower_safe_amp,

                "lower_target_amp": r.lower_target_amp,

                "upper_target_amp": r.upper_target_amp,

                "upper_unsafe_amp": r.upper_unsafe_amp,

                "converged": r.converged,

                "rounds_used": r.rounds_used,

            }

            for r in results

        ],

    }

    (output_root / "boundary_results.json").write_text(

        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"

    )



    # Save probe trace CSV

    if probe_trace:

        csv_path = output_root / "probe_trace.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=list(probe_trace[0].keys()))

            writer.writeheader()

            writer.writerows(probe_trace)



    return summary


def run_boundary_search_target_intervals(
    *,
    manifest_path: Path,
    point_name: str,
    output_root: Path,
    duration_min: int = 100,
    duration_max: int = 1000,
    duration_step: int = 5,
    t_start: int = 0,
    steps: int = 1000,
    hold_steps: int = 100,
    alarm_cap: int = 2,
    forbidden_alarm_ids: Optional[List[str]] = None,
    required_alarm_ids: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
    amp_tol: float = 1e-4,
    coarse_span: float = 0.03125,
    workers: int = 16,
) -> Dict[str, Any]:
    """Search true target-vs-non-target interval boundaries per duration.

    Unlike the legacy monotone boundary search, this routine supports
    safe->unsafe->target or other non-monotone class layouts on the amplitude
    axis. Output always includes target_intervals; legacy lower/upper target
    fields are filled only when there is exactly one target interval.

    Recommended artifact entrypoint for current Stage 2 base-region search.
    """
    registry = load_injection_registry(manifest_path)
    domain = build_search_domain(
        registry=registry,
        point_name=point_name,
        duration_min=duration_min,
        duration_max=duration_max,
        duration_step=duration_step,
        t_start=t_start,
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    search_directions = directions if directions else ["pos", "neg"]
    durations = list(range(domain.duration_min, domain.duration_max + 1, domain.duration_step))
    tasks = [
        {
            "manifest_path": str(manifest_path),
            "point_name": point_name,
            "direction": direction,
            "duration": dur,
            "t_start": t_start,
            "steps": steps,
            "hold_steps": hold_steps,
            "alarm_cap": alarm_cap,
            "forbidden_alarm_ids": forbidden_alarm_ids,
            "required_alarm_ids": required_alarm_ids,
            "amp_tol": amp_tol,
            "coarse_span": coarse_span,
        }
        for dur in durations
        for direction in search_directions
    ]

    results: List[Dict[str, Any]] = []
    probe_trace: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_classify_point_worker, task) for task in tasks]
        for fut in as_completed(futures):
            item = fut.result()
            results.append(item["boundary"])
            probe_trace.extend(item["probe_trace"])

    results.sort(key=lambda x: (x["direction"], x["duration"]))
    summary = {
        "point": point_name,
        "backend": "physics",
        "method": "target_intervals",
        "domain": {
            "amp_max": domain.amp_max,
            "default_value": domain.default_value,
            "value_range": list(domain.value_range),
            "duration_min": domain.duration_min,
            "duration_max": domain.duration_max,
            "duration_step": domain.duration_step,
            "coarse_span": coarse_span,
            "amp_tol": amp_tol,
            "workers": int(workers),
        },
        "semantic_filters": {
            "forbidden_alarm_ids": forbidden_alarm_ids or [],
            "required_alarm_ids": required_alarm_ids or [],
        },
        "boundaries": results,
    }
    (output_root / "boundary_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if probe_trace:
        csv_path = output_root / "probe_trace.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(probe_trace[0].keys()))
            writer.writeheader()
            writer.writerows(probe_trace)
    return summary




# ============================================================================

#
# ============================================================================



def run_boundary_search_cegis(
    *,

    manifest_path: Path,

    point_name: str,

    output_root: Path,

    duration_min: int = 100,

    duration_max: int = 1000,

    duration_step_coarse: int = 100,

    duration_step_fine: int = 5,

    t_start: int = 0,

    steps: int = 1200,
    hold_steps: int = 100,

    alarm_cap: int = 2,

    forbidden_alarm_ids: Optional[List[str]] = None,
    required_alarm_ids: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
    amp_tol: float = 0.0001,

    max_rounds: int = 10,

    workers: int = 4,
    export_traces: bool = True,

) -> Dict[str, Any]:

    """..."""

    import time



    t_total_start = time.time()

    sim_count = [0]



    #
    registry = load_injection_registry(manifest_path)

    rules = load_rules_from_manifest(manifest_path)

    sim_module, sim_init_kwargs = load_simulation_module(manifest_path)



    domain = build_search_domain(

        registry=registry,

        point_name=point_name,

        duration_min=duration_min,

        duration_max=duration_max,

        duration_step=duration_step_fine,

        t_start=t_start,

    )



    alarm_cols = [spec["margin_col"] for spec in rules.get("alarms", {}).values()]

    search_directions = directions if directions else ["pos", "neg"]



    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    trace_dir = output_root / "traces"
    if export_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)
    sim_count_lock = Lock()



    # Build simulate_fn with trace export

    def simulate_fn(task: Dict) -> Dict:

        result = simulate(task, registry=registry, rules=rules,

                          sim_module=sim_module, sim_init_kwargs=sim_init_kwargs)

        with sim_count_lock:
            sim_count[0] += 1
            sim_idx = sim_count[0]

        rows = result.get("rows", [])

        scored = result.get("score", {})

        if export_traces and rows:

            inj = task.get("injection", {})

            trace_name = f"sim_{sim_idx:04d}_d{inj.get('duration', 0)}_{inj.get('amp', 0):.6f}.csv"

            with open(trace_dir / trace_name, "w", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

                writer.writeheader()

                writer.writerows(rows)

        return {

            "status": result.get("status", "ok"),

            "first_hazard_step": scored.get("first_hazard_step"),

            "prehazard_alarm_rule_ids_count": scored.get("prehazard_alarm_rule_ids_count", 0),

            "prehazard_alarm_rule_ids": scored.get("prehazard_alarm_rule_ids", []),

            "alarm_rule_ids": scored.get("prehazard_alarm_rule_ids", []),

            "rows": rows,

        }



    # Helper: run search for one (duration, direction) with optional warm start

    def _search_one(dur: int, direction: str, warm_start: Optional[Dict] = None,

                    neighbor_lower_safe: float = 0.0, neighbor_upper_unsafe: Optional[float] = None) -> BoundaryResult:

        if warm_start:

            # Validate warm start guesses before using them

            sign = 1.0 if direction == "pos" else -1.0

            validated = dict(warm_start)



            # Validate lower_safe: must actually be safe

            if validated.get("lower_safe") is not None:

                task = _make_task(domain, sign * validated["lower_safe"], dur, steps, hold_steps)

                result = simulate_fn(task)
                cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                                      forbidden_alarm_ids=forbidden_alarm_ids,
                                      required_alarm_ids=required_alarm_ids)
                if cls != "safe":

                    # Guess was wrong - search downward for real safe point

                    hi = validated["lower_safe"]

                    lo = neighbor_lower_safe  # Use neighbor's lower_safe as floor

                    validated["lower_target"] = hi  # current point is target/unsafe

                    found_safe = False

                    for _ in range(15):

                        mid = (lo + hi) / 2.0

                        task = _make_task(domain, sign * mid, dur, steps, hold_steps)

                        result = simulate_fn(task)
                        cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                                              forbidden_alarm_ids=forbidden_alarm_ids,
                                              required_alarm_ids=required_alarm_ids)
                        if cls == "safe":

                            validated["lower_safe"] = mid

                            lo = mid

                            found_safe = True

                        else:

                            if cls == "target" and (validated.get("lower_target") is None or mid < validated["lower_target"]):

                                validated["lower_target"] = mid

                            hi = mid

                        if found_safe and (hi - lo) < amp_tol:

                            break

                    if not found_safe:

                        validated["lower_safe"] = None



            # Validate upper_unsafe: must actually be unsafe

            if validated.get("upper_unsafe") is not None:

                task = _make_task(domain, sign * validated["upper_unsafe"], dur, steps, hold_steps)

                result = simulate_fn(task)
                cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                                      forbidden_alarm_ids=forbidden_alarm_ids,
                                      required_alarm_ids=required_alarm_ids)
                if cls != "unsafe":

                    # Guess was wrong - search upward for real unsafe point

                    lo = validated["upper_unsafe"]

                    hi = neighbor_upper_unsafe if neighbor_upper_unsafe is not None else domain.amp_max

                    validated["upper_target"] = lo  # current point is target/safe

                    found_unsafe = False

                    for _ in range(15):

                        mid = (lo + hi) / 2.0

                        task = _make_task(domain, sign * mid, dur, steps, hold_steps)

                        result = simulate_fn(task)
                        cls = classify_result(result, alarm_cols=alarm_cols, alarm_cap=alarm_cap,
                                              forbidden_alarm_ids=forbidden_alarm_ids,
                                              required_alarm_ids=required_alarm_ids)
                        if cls == "unsafe":

                            validated["upper_unsafe"] = mid

                            hi = mid

                            found_unsafe = True

                        else:

                            if cls == "target" and (validated.get("upper_target") is None or mid > validated["upper_target"]):

                                validated["upper_target"] = mid

                            lo = mid

                        if found_unsafe and (hi - lo) < amp_tol:

                            break

                    if not found_unsafe:

                        validated["upper_unsafe"] = None



            boundary = run_cegis_refine(

                simulate_fn=simulate_fn,

                domain=domain,

                direction=direction,

                duration=dur,

                initial_boundary=validated,

                alarm_cols=alarm_cols,
                alarm_cap=alarm_cap,
                forbidden_alarm_ids=forbidden_alarm_ids,
                required_alarm_ids=required_alarm_ids,
                steps=steps,
                amp_tol=amp_tol,
                max_rounds=max_rounds,
                hold_steps=hold_steps,
            )

        else:

            # Full search from scratch

            initial = run_coarse_bisect(

                simulate_fn=simulate_fn,

                domain=domain,

                direction=direction,

                duration=dur,

                alarm_cols=alarm_cols,
                alarm_cap=alarm_cap,
                forbidden_alarm_ids=forbidden_alarm_ids,
                required_alarm_ids=required_alarm_ids,
                steps=steps,
                hold_steps=hold_steps,
            )
            boundary = run_cegis_refine(
                simulate_fn=simulate_fn,

                domain=domain,

                direction=direction,

                duration=dur,

                initial_boundary=initial,

                alarm_cols=alarm_cols,
                alarm_cap=alarm_cap,
                forbidden_alarm_ids=forbidden_alarm_ids,
                required_alarm_ids=required_alarm_ids,
                steps=steps,
                hold_steps=hold_steps,
                amp_tol=amp_tol,
                max_rounds=max_rounds,
            )

        return boundary



    # Phase 1: Coarse search

    coarse_durations = list(range(duration_min, duration_max + 1, duration_step_coarse))

    # Ensure endpoints are included

    if coarse_durations[-1] != duration_max:

        coarse_durations.append(duration_max)



    # Store results indexed by (direction, duration)

    boundary_map: Dict[Tuple[str, int], BoundaryResult] = {}



    print(f"[CEGIS] Phase 1: Coarse search ({len(coarse_durations)} durations, step={duration_step_coarse})", flush=True)

    coarse_jobs = [(dur, direction) for dur in coarse_durations for direction in search_directions]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            future_map = {}
            for dur, direction in coarse_jobs:
                print(f"  [{point_name}] coarse d={dur} dir={direction} ...", flush=True)
                future_map[pool.submit(_search_one, dur, direction, None)] = (dur, direction)
            for fut in as_completed(future_map):
                dur, direction = future_map[fut]
                boundary_map[(direction, dur)] = fut.result()
    else:
        for dur, direction in coarse_jobs:
            print(f"  [{point_name}] coarse d={dur} dir={direction} ...", flush=True)
            boundary_map[(direction, dur)] = _search_one(dur, direction, warm_start=None)



    # Phase 2: Iterative refinement - insert midpoints between known durations

    current_step = duration_step_coarse

    while current_step > duration_step_fine:

        new_step = max(duration_step_fine, current_step // 2)

        # Find all gaps that need filling

        new_durations = []

        all_known = sorted(set(d for (_, d) in boundary_map.keys()))

        for i in range(len(all_known) - 1):

            d_lo, d_hi = all_known[i], all_known[i + 1]

            gap = d_hi - d_lo

            if gap > duration_step_fine:

                mid = d_lo + gap // 2

                # Round to nearest multiple of duration_step_fine

                mid = round(mid / duration_step_fine) * duration_step_fine

                if mid > d_lo and mid < d_hi and (mid not in all_known):

                    new_durations.append((mid, d_lo, d_hi))



        if not new_durations:

            break



        print(f"[CEGIS] Phase 2: Refine step {current_step}->{new_step}, inserting {len(new_durations)} durations", flush=True)



        refine_jobs = []
        for mid_dur, d_lo, d_hi in new_durations:
            for direction in search_directions:
                b_lo = boundary_map.get((direction, d_lo))
                b_hi = boundary_map.get((direction, d_hi))
                if b_lo and b_hi:
                    frac = (mid_dur - d_lo) / max(1, d_hi - d_lo)
                    warm = {}
                    for key in ("lower_safe", "lower_target", "upper_target", "upper_unsafe"):
                        amp_key = f"{key}_amp"
                        v_lo = getattr(b_lo, amp_key, None)
                        v_hi = getattr(b_hi, amp_key, None)
                        if v_lo is not None and v_hi is not None:
                            warm[key] = v_lo + frac * (v_hi - v_lo)
                        elif v_lo is not None:
                            warm[key] = v_lo
                        elif v_hi is not None:
                            warm[key] = v_hi
                        else:
                            warm[key] = None
                    neighbor_ls = min(
                        b_lo.lower_safe_amp if b_lo.lower_safe_amp is not None else 0.0,
                        b_hi.lower_safe_amp if b_hi.lower_safe_amp is not None else 0.0,
                    )
                    neighbor_uu = max(
                        b_lo.upper_unsafe_amp if b_lo.upper_unsafe_amp is not None else domain.amp_max,
                        b_hi.upper_unsafe_amp if b_hi.upper_unsafe_amp is not None else domain.amp_max,
                    )
                else:
                    warm = None
                    neighbor_ls = 0.0
                    neighbor_uu = None
                refine_jobs.append((mid_dur, direction, warm, neighbor_ls, neighbor_uu))

        if workers > 1:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                future_map = {}
                for mid_dur, direction, warm, neighbor_ls, neighbor_uu in refine_jobs:
                    print(f"  [{point_name}] refine d={mid_dur} dir={direction} (warm={warm is not None}) ...", flush=True)
                    future_map[pool.submit(
                        _search_one,
                        mid_dur,
                        direction,
                        warm,
                        neighbor_ls,
                        neighbor_uu,
                    )] = (mid_dur, direction)
                for fut in as_completed(future_map):
                    mid_dur, direction = future_map[fut]
                    boundary_map[(direction, mid_dur)] = fut.result()
        else:
            for mid_dur, direction, warm, neighbor_ls, neighbor_uu in refine_jobs:
                print(f"  [{point_name}] refine d={mid_dur} dir={direction} (warm={warm is not None}) ...", flush=True)
                boundary_map[(direction, mid_dur)] = _search_one(
                    mid_dur,
                    direction,
                    warm,
                    neighbor_ls,
                    neighbor_uu,
                )



        current_step = new_step



    # Collect all results sorted by duration

    results: List[BoundaryResult] = []

    for direction in search_directions:

        dir_results = [(d, b) for (dir_, d), b in boundary_map.items() if dir_ == direction]

        dir_results.sort(key=lambda x: x[0])

        results.extend(b for _, b in dir_results)



    total_time = time.time() - t_total_start



    # Save

    summary = {

        "point": point_name,

        "backend": "physics",

        "method": "cegis_duration_refine",

        "total_simulations": sim_count[0],

        "total_time_s": round(total_time, 2),

        "domain": {

            "amp_max": domain.amp_max,

            "default_value": domain.default_value,

            "value_range": list(domain.value_range),

            "duration_min": domain.duration_min,

            "duration_max": domain.duration_max,

            "duration_step_coarse": duration_step_coarse,

            "duration_step_fine": duration_step_fine,

        },

        "boundaries": [

            {

                "direction": r.direction,

                "duration": r.duration,

                "lower_safe_amp": r.lower_safe_amp,

                "lower_target_amp": r.lower_target_amp,

                "upper_target_amp": r.upper_target_amp,

                "upper_unsafe_amp": r.upper_unsafe_amp,

                "converged": r.converged,

                "rounds_used": r.rounds_used,

            }

            for r in results

        ],

    }

    (output_root / "boundary_results.json").write_text(

        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"

    )



    print(f"[CEGIS] Done: {sim_count[0]} simulations in {total_time:.1f}s", flush=True)

    return summary


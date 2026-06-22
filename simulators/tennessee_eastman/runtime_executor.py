# Runtime Executor - Section 3.3

"""..."""

from __future__ import annotations



import csv

import hashlib

import importlib

import importlib.util

import json

import os

import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass

from pathlib import Path

from typing import (

    Any, Callable, Dict, Iterable, List, Mapping,

    Optional, Sequence, Set, Tuple,

)





# ============================================================================

#
# ============================================================================





def load_injection_registry(manifest_path: Path) -> Dict[str, Dict[str, Any]]:

    """..."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    injection_points = manifest.get("injection_points", [])

    registry: Dict[str, Dict[str, Any]] = {}



    for ip in injection_points:

        name = str(ip["name"])

        rng = ip.get("range", [None, None])

        rate = float(ip.get("rate", 1.0))

        default = float(ip.get("default_value", 0.0))



        #
        if "phase" in ip and "field" in ip:

            phase = str(ip["phase"])

            field = str(ip["field"])

        else:

            #
            field = name

            for prefix in ("simulation_", "ctrl_out_", "ctrl_out_cmd_"):

                if field.startswith(prefix):

                    field = field[len(prefix):]

                    break

            phase = f"controller_{field}"



        registry[name] = {

            "phase": phase,

            "field": field,

            "trace_field": str(ip.get("trace_field", field)),

            "default_value": default,

            "range": [float(rng[0]), float(rng[1])] if rng and len(rng) == 2 else None,

            "rate": rate,

        }



    return registry





def load_rules_from_manifest(manifest_path: Path) -> Dict[str, Any]:

    """..."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))



    hazards: Dict[str, Dict[str, Any]] = {}

    for hr in manifest.get("hazard_rules", []):

        rule_id = hr["id"]

        var = hr.get("var", "")

        #
        if "margin_col" in hr:

            margin_col = hr["margin_col"]

        else:

            short = rule_id.lower().replace("h-", "").split("-")[0]

            margin_col = f"hazard_{short}_margin"

        hazards[rule_id] = {

            "margin_col": margin_col,

            "var": var,

            "threshold": hr.get("threshold"),

            "direction": hr.get("direction"),

        }



    alarms: Dict[str, Dict[str, Any]] = {}

    for ar in manifest.get("alarm_rules", []):

        rule_id = ar["id"]

        margin_col = ar.get("margin_col", f"alarm_{rule_id.lower()}_margin")

        alarms[rule_id] = {

            "margin_col": margin_col,

            "var": ar.get("var", ""),

            "threshold": ar.get("threshold"),

            "direction": ar.get("direction"),

        }



    return {"hazards": hazards, "alarms": alarms}





def load_simulation_module(manifest_path: Path) -> Tuple[Any, Dict[str, Any]]:

    """..."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    sim_spec = manifest["simulation"]

    module_name = sim_spec["module"]



    #
    manifest_dir = Path(manifest_path).resolve().parent

    parent_dir = str(manifest_dir.parent)

    if parent_dir not in sys.path:

        sys.path.insert(0, parent_dir)



    mod = importlib.import_module(module_name)

    class_name = sim_spec.get("class_name", "ClosedLoopSim")

    if not hasattr(mod, class_name):

        raise AttributeError(f"Module {module_name} does not define {class_name}")



    #
    pp_init = manifest.get("physical_process", {}).get("init_params", {})

    sim_init_kwargs: Dict[str, Any] = {}

    #
    for key in ("case_path", "gen_ids", "bus_ids"):

        if key in pp_init:

            val = pp_init[key]

            sim_init_kwargs[key] = val



    #
    #
    #
    if "Ts" in pp_init:

        sim_init_kwargs["__manifest_ts__"] = float(pp_init["Ts"])



    return mod, sim_init_kwargs





# ============================================================================

#
# ============================================================================





def _clamp(value: float, value_range: Optional[Sequence[float]]) -> float:

    """..."""

    if value_range is None or len(value_range) < 2:

        return float(value)

    lo, hi = float(value_range[0]), float(value_range[1])

    return float(max(lo, min(hi, float(value))))





def compute_run_key(task: Mapping[str, Any]) -> str:

    """..."""

    canonical = {

        "injection": task.get("injection", {}),

        "runtime_args": task.get("runtime_args", {}),

    }

    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()

    return "rk_" + hashlib.sha256(payload).hexdigest()[:24]





def _injection_amp(spec: Mapping[str, Any]) -> float:

    """..."""

    for key in ("amp", "delta", "delta_signed"):

        if key in spec:

            return float(spec[key])

    return 0.0





def _injection_duration(spec: Mapping[str, Any], runtime_args: Mapping[str, Any]) -> int:

    """..."""

    if "duration" in spec:

        return max(1, int(spec["duration"]))

    steps = int(runtime_args.get("steps", 200))

    t_start = int(spec.get("t_start", 0))

    return max(1, steps - max(0, t_start))





# ============================================================================

#
# ============================================================================





def _make_ramp_hold_hook(

    *,

    phase: str,

    field: str,

    amp_rate: float,

    t_start: int,

    duration: int,

    value_range: Optional[Sequence[float]] = None,

) -> Callable:

    """..."""

    t_end = t_start + duration

    state: Dict[str, Optional[float]] = {

        "t0": None, "anchor": None, "terminal": None,

    }



    def hook(t_step: int, current_phase: str, context: Dict[str, float]) -> Optional[Dict[str, float]]:

        if current_phase != phase:

            return None

        # capture simulation start step on first call (any phase first wins)

        if state["t0"] is None:

            state["t0"] = int(t_step)

        rel = int(t_step) - int(state["t0"])

        if rel < t_start:

            return None

        if field not in context:

            return None



        if state["anchor"] is None:

            state["anchor"] = float(context[field])



        anchor = state["anchor"]



        if rel >= t_end:

            if state["terminal"] is None:

                terminal_step = max(0, duration - 1)

                state["terminal"] = _clamp(anchor + amp_rate * terminal_step, value_range)

            return {field: state["terminal"]}



        local_step = rel - t_start

        injected = _clamp(anchor + amp_rate * local_step, value_range)

        if rel == t_end - 1:

            state["terminal"] = injected

        return {field: injected}



    return hook





def build_injection_hook(

    point_specs: Sequence[Mapping[str, Any]],

    registry: Dict[str, Dict[str, Any]],

    runtime_args: Mapping[str, Any],

) -> Optional[Callable]:

    """..."""

    hooks: List[Callable] = []



    for spec in point_specs:

        point = str(spec.get("point", "")).strip()

        role = str(spec.get("role", "manip")).strip().lower()

        if role == "hidden":

            continue  # hidden  hook ，

        if point not in registry:

            continue

        amp = _injection_amp(spec)

        if abs(amp) <= 0.0:

            continue



        reg = registry[point]

        t_start = max(0, int(spec.get("t_start", 0)))

        duration = _injection_duration(spec, runtime_args)

        value_range = spec.get("point_range") or reg.get("range")



        hooks.append(_make_ramp_hold_hook(

            phase=reg["phase"],

            field=reg["field"],

            amp_rate=amp,

            t_start=t_start,

            duration=duration,

            value_range=value_range,

        ))



    if not hooks:

        return None



    def merged_hook(t_step: int, current_phase: str, context: Dict[str, float]) -> Optional[Dict[str, float]]:

        result: Dict[str, float] = {}

        for fn in hooks:

            out = fn(t_step, current_phase, context)

            if out:

                result.update(out)

        return result or None



    return merged_hook





# ============================================================================

#
# ============================================================================





def annotate_rows(

    rows: Sequence[Mapping[str, Any]],

    rules: Mapping[str, Any],

) -> List[Dict[str, Any]]:

    """..."""

    hazards = rules.get("hazards", {})

    alarms = rules.get("alarms", {})

    out: List[Dict[str, Any]] = []



    for idx, raw in enumerate(rows):

        row = dict(raw)

        row.setdefault("t_step", idx)



        alarm_ids = []

        for rule_id, spec in alarms.items():

            margin_col = spec.get("margin_col", "")

            if margin_col and margin_col in row:

                try:

                    if float(row[margin_col]) <= 0.0:

                        alarm_ids.append(rule_id)

                except (ValueError, TypeError):

                    pass



        hazard_ids = []

        for rule_id, spec in hazards.items():

            margin_col = spec.get("margin_col", "")

            if margin_col and margin_col in row:

                try:

                    if float(row[margin_col]) <= 0.0:

                        hazard_ids.append(rule_id)

                except (ValueError, TypeError):

                    pass



        row["alarm_rule_ids"] = sorted(alarm_ids)

        row["hazard_rule_ids"] = sorted(hazard_ids)

        row["alarm_count"] = len(alarm_ids)

        row["hazard_count"] = len(hazard_ids)

        out.append(row)



    return out





def score_trace(

    rows: Sequence[Mapping[str, Any]],

    rules: Mapping[str, Any],

) -> Dict[str, Any]:

    """..."""

    annotated = annotate_rows(rows, rules)

    first_hazard_step: Optional[int] = None

    first_alarm_step: Optional[int] = None

    prehazard_alarm_ids: Set[str] = set()



    for idx, row in enumerate(annotated):

        hazard_ids = row.get("hazard_rule_ids", [])

        alarm_ids = row.get("alarm_rule_ids", [])



        if hazard_ids and first_hazard_step is None:

            first_hazard_step = idx

        if alarm_ids and first_alarm_step is None:

            first_alarm_step = idx

        if first_hazard_step is None:

            for aid in alarm_ids:

                prehazard_alarm_ids.add(aid)



    return {

        "status": "ok",

        "first_hazard_step": first_hazard_step,

        "first_alarm_step": first_alarm_step,

        "prehazard_alarm_rule_ids": sorted(prehazard_alarm_ids),

        "prehazard_alarm_rule_ids_count": len(prehazard_alarm_ids),

    }





# ============================================================================

#
# ============================================================================





def normalize_injection_points(injection: Mapping[str, Any]) -> List[Dict[str, Any]]:

    """..."""

    raw_points = injection.get("points")

    if isinstance(raw_points, list):

        points = []

        for item in raw_points:

            if isinstance(item, Mapping) and str(item.get("point", "")).strip():

                points.append(dict(item))

        return points



    point = str(injection.get("point", "")).strip()

    if not point:

        return []

    return [{

        "point": point,

        "amp": injection.get("amp", injection.get("delta", 0.0)),

        "duration": injection.get("duration"),

        "t_start": injection.get("t_start", 0),

        "role": injection.get("role", "manip"),

        "point_range": injection.get("point_range"),

    }]





# ============================================================================

#
# ============================================================================





def simulate(

    task: Mapping[str, Any],

    *,

    registry: Dict[str, Dict[str, Any]],

    rules: Dict[str, Any],

    sim_module: Any,

    sim_init_kwargs: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:

    """..."""

    ClosedLoopSim = getattr(sim_module, "ClosedLoopSim")

    injection = task.get("injection", {})

    runtime_args = task.get("runtime_args", {})



    steps = max(1, int(runtime_args.get("steps", 200)))

    #
    manifest_ts = None

    if sim_init_kwargs and "__manifest_ts__" in sim_init_kwargs:

        manifest_ts = float(sim_init_kwargs["__manifest_ts__"])

    default_dt = manifest_ts if manifest_ts is not None else 0.1

    dt = float(runtime_args.get("dt", runtime_args.get("Ts", default_dt)))

    stop_on_trip = bool(runtime_args.get("stop_on_trip", False))



    #
    point_specs = normalize_injection_points(injection)

    manip_specs = [s for s in point_specs if str(s.get("role", "manip")).lower() != "hidden"]

    hidden_specs = [s for s in point_specs if str(s.get("role", "")).lower() == "hidden"]



    #
    hook = build_injection_hook(manip_specs, registry, runtime_args)



    #
    init_kwargs = {"Ts": dt}

    if sim_init_kwargs:

        #
        for k, v in sim_init_kwargs.items():

            if not str(k).startswith("__"):

                init_kwargs[k] = v

    sim = ClosedLoopSim(**init_kwargs)

    try:

        trace, meta = sim.run(

            steps=steps,

            injection_hook=hook,

            return_trace=True,

            stop_on_trip=stop_on_trip,

        )

    finally:

        if hasattr(sim, "close"):

            sim.close()



    #
    scored = score_trace(trace, rules)



    #
    injection_echo = []

    for spec in point_specs:

        p = str(spec.get("point", ""))

        dur = _injection_duration(spec, runtime_args)

        reg = registry.get(p, {})

        injection_echo.append({

            "point": p,

            "amp": float(_injection_amp(spec)),

            "duration": dur,

            "t_start": int(spec.get("t_start", 0)),

            "role": str(spec.get("role", "manip")),

            "range": reg.get("range"),

        })



    return {

        "status": "ok",

        "rows": trace,

        "score": scored,

        "injection_echo": injection_echo,

        "meta": dict(meta) if isinstance(meta, Mapping) else {},

        "run_key": compute_run_key(task),

    }





# ============================================================================

#
# ============================================================================





def run_batch(

    tasks: Sequence[Mapping[str, Any]],

    *,

    registry: Dict[str, Dict[str, Any]],

    rules: Dict[str, Any],

    sim_module: Any,

    workers: int = 1,

) -> List[Dict[str, Any]]:

    """..."""

    if workers <= 1:

        results = []

        for task in tasks:

            try:

                result = simulate(task, registry=registry, rules=rules, sim_module=sim_module)

            except Exception as e:

                result = {"status": "error", "error": str(e), "run_key": compute_run_key(task)}

            results.append(result)

        return results



    #
    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as pool:

        futures = {}

        for idx, task in enumerate(tasks):

            f = pool.submit(simulate, task, registry=registry, rules=rules, sim_module=sim_module)

            futures[f] = idx

        for f in as_completed(futures):

            idx = futures[f]

            try:

                results[idx] = f.result()

            except Exception as e:

                results[idx] = {"status": "error", "error": str(e)}

    return results





# ============================================================================

#
# ============================================================================





def create_executor(manifest_path: Path):

    """..."""

    registry = load_injection_registry(manifest_path)

    rules = load_rules_from_manifest(manifest_path)

    sim_module, sim_init_kwargs = load_simulation_module(manifest_path)



    def bound_simulate(task: Mapping[str, Any]) -> Dict[str, Any]:

        return simulate(task, registry=registry, rules=rules,

                        sim_module=sim_module, sim_init_kwargs=sim_init_kwargs)



    bound_simulate.registry = registry

    bound_simulate.rules = rules

    bound_simulate.sim_module = sim_module

    bound_simulate.sim_init_kwargs = sim_init_kwargs

    return bound_simulate


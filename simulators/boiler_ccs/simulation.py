# Boiler Closed-Loop Simulation - Section 4.1

"""Closed-loop simulation environment with payload-driven injection support.



This module computes controller outputs in pipeline order and allows payload

replacement immediately after each output so downstream blocks in the same

cycle use the modified value.

"""

from __future__ import annotations



import csv

import json

from pathlib import Path

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple



try:

    from .controller import PLCController, AlarmResult

    from .physical_process import PhysicalProcess

except ImportError:  # pragma: no cover - direct script execution fallback

    from controller import PLCController, AlarmResult  # type: ignore

    from physical_process import PhysicalProcess  # type: ignore





TRIP_WATERWALL = 475.0

TRIP_PRESSURE = 28.3

TRIP_FEEDWATER = 184.8



HAZARD_RULES = [

    {

        "id": "H-WATERWALL-001",

        "kind": "threshold",

        "expr": "waterwall_temp > TRIP_WATERWALL",

        "var": "waterwall_temp",

        "threshold": 475.0,

        "direction": "upper",

        "component": "physical",

        "duration_s": 0.0,

    },

    {

        "id": "H-PRESSURE-001",

        "kind": "threshold",

        "expr": "main_steam_pressure > TRIP_PRESSURE",

        "var": "main_steam_pressure",

        "threshold": 28.3,

        "direction": "upper",

        "component": "physical",

        "duration_s": 0.0,

    },

    {

        "id": "H-FEEDWATER-001",

        "kind": "threshold",

        "expr": "feedwater_flow <= TRIP_FEEDWATER",

        "var": "feedwater_flow",

        "threshold": 184.8,

        "direction": "lower",

        "component": "physical",

        "duration_s": 0.0,

    },

]





REQUIRED_TRACE_KEYS = (

    "t_step",

    "t_time_s",

    "true_load",

    "coal_flow",

    "feedwater_flow",

    "main_steam_flow",

    "waterwall_temp",

    "main_steam_pressure",

    "load_output",

    "steam_setpoint",

    "boiler_setpoint",

    "fuel_command",

    "water_setpoint",

    "water_pump_speed",

    "phys_fuel_command",

    "phys_water_pump_speed",

    "trip_any",

)



BRANCH_TRACE_KEYS = (

    "br_load_rate_mode",

    "br_steam_hs_mode",

    "br_steam_hs_segment",

    "br_steam_rate_mode",

    "br_boiler_pid_deadband_hit",

    "br_boiler_pid_integral_on",

    "br_boiler_pid_td_on",

    "br_boiler_pid_sat_hi",

    "br_boiler_pid_sat_lo",

    "br_boiler_pid_ov_hi_clip",

    "br_boiler_pid_ov_lo_clip",

    "br_fuel_pt_mode",

    "br_fuel_pt_segment",

    "br_fuel_ti_mode",

    "br_fuel_ti_segment",

    "br_fuel_rate_mode",

    "br_fuel_pid_deadband_hit",

    "br_fuel_pid_integral_on",

    "br_fuel_pid_td_on",

    "br_fuel_pid_sat_hi",

    "br_fuel_pid_sat_lo",

    "br_fuel_pid_ov_hi_clip",

    "br_fuel_pid_ov_lo_clip",

    "br_waterk_hsc_mode",

    "br_waterk_hsc_segment",

    "br_waterk_floor_clip",

    "br_water_pt_mode",

    "br_water_pt_segment",

    "br_water_ti_mode",

    "br_water_ti_segment",

    "br_water_rate_mode",

    "br_water_pid_deadband_hit",

    "br_water_pid_integral_on",

    "br_water_pid_td_on",

    "br_water_pid_sat_hi",

    "br_water_pid_sat_lo",

    "br_water_pid_ov_hi_clip",

    "br_water_pid_ov_lo_clip",

)



M5_CONTROLLER_INTERNAL_TRACE_KEYS = (

    "load_rate_prev",

    "steam_hfop1_prev",

    "steam_hfop2_prev",

    "steam_hfop3_prev",

    "steam_hfop4_prev",

    "steam_rate_prev",

    "boiler_hfop1_prev",

    "boiler_pid_av0",

    "boiler_pid_err0",

    "boiler_pid_delta_err0",

    "boiler_pid_dk0",

    "fuel_hfop1_prev",

    "fuel_rate_prev",

    "fuel_pid_av0",

    "fuel_pid_err0",

    "fuel_pid_delta_err0",

    "fuel_pid_dk0",

    "water_hfop_prev",

    "water_rate_prev",

    "water_pid_av0",

    "water_pid_err0",

    "water_pid_delta_err0",

    "water_pid_dk0",

    "waterk_hfop1_prev",

)





InjectionHook = Callable[[int, str, Dict[str, float]], Optional[Dict[str, float]]]





FIELD_BINDINGS_BY_W: Dict[str, Tuple[str, str]] = {

    "TrueLoadCommand": ("controller_load_output", "load_output"),

    "MainSteamPressureSetpoint": ("controller_steam_hslim", "steam_setpoint"),

    "BoilerControlSetpoint": ("controller_boiler_setpoint", "boiler_setpoint"),

    "FuelControlInstruction": ("controller_fuel_command", "fuel_command"),

    "FeedwaterControlInstruction": ("controller_water_pump", "water_pump_speed"),

    "FeedwaterControlSetpoint": ("controller_water_setpoint", "water_setpoint"),

}





def baseline_u0(t_step: int, sensors: Mapping[str, float], ctrl_out: Mapping[str, float]) -> Dict[str, float]:

    """Return baseline applied commands (no external override)."""

    del t_step, sensors

    return {

        "load_output": float(ctrl_out["load_output"]),

        "steam_setpoint": float(ctrl_out["steam_setpoint"]),

        "boiler_setpoint": float(ctrl_out["boiler_setpoint"]),

        "fuel_command": float(ctrl_out["fuel_command"]),

        "water_setpoint": float(ctrl_out["water_setpoint"]),

        "water_pump_speed": float(ctrl_out["water_pump_speed"]),

    }





def _controller_fallback_binding(controller: str, w: str) -> Optional[Tuple[str, str]]:

    controller_l = controller.lower()

    w_l = w.lower()

    if "loadcommand" in controller_l or "trueloadcommand" in w_l:

        return ("controller_load_output", "load_output")

    if "mainsteampressuresetpoint" in controller_l or "mainsteampressuresetpoint" in w_l:

        return ("controller_steam_hslim", "steam_setpoint")

    if "boilmaster" in controller_l or "boilercontrolsetpoint" in w_l:

        return ("controller_boiler_setpoint", "boiler_setpoint")

    if "fuel" in controller_l or "fuelcontrolinstruction" in w_l:

        return ("controller_fuel_command", "fuel_command")

    if "feedwater" in controller_l:

        if "instruction" in w_l:

            return ("controller_water_pump", "water_pump_speed")

        return ("controller_water_setpoint", "water_setpoint")

    return None





def _effective_cut_any_for_stop(

    record: Mapping[str, object],

    *,

    ignored_protection_predicates: Set[str],

) -> bool:

    """Return cut flag after excluding ignored protection predicates."""

    check_boiler = "BOILER_CUT_TH" not in ignored_protection_predicates

    check_fuel = "FUEL_CUT_TH" not in ignored_protection_predicates

    check_water = "WATER_CUT_TH" not in ignored_protection_predicates

    return bool(

        (check_boiler and bool(record.get("boiler_cut", False)))

        or (check_fuel and bool(record.get("fuel_cut", False)))

        or (check_water and bool(record.get("water_cut", False)))

    )





def _active_event(strategy: Mapping[str, object], t_step: int) -> Optional[Tuple[int, Mapping[str, object]]]:

    events = strategy.get("E_w")

    if not isinstance(events, list):

        return None

    for idx, event in enumerate(events):

        if not isinstance(event, dict):

            continue

        t_s = int(event.get("t_s", -1))

        t_e = int(event.get("t_e", -1))

        if t_s <= int(t_step) < t_e:

            return idx, event

    return None





def _surrounding_events(

    strategy: Mapping[str, object], t_step: int

) -> Tuple[Optional[Tuple[int, Mapping[str, object]]], Optional[Tuple[int, Mapping[str, object]]]]:

    """Return (latest_ended_event, earliest_future_event) around current step."""

    events = strategy.get("E_w")

    if not isinstance(events, list):

        return None, None

    prev_event: Optional[Tuple[int, Mapping[str, object]]] = None

    next_event: Optional[Tuple[int, Mapping[str, object]]] = None

    cur = int(t_step)

    for idx, event in enumerate(events):

        if not isinstance(event, dict):

            continue

        t_s = int(event.get("t_s", -1))

        t_e = int(event.get("t_e", -1))

        if t_e <= t_s:

            continue

        if t_e <= cur:

            prev_event = (idx, event)

            continue

        if cur < t_s:

            next_event = (idx, event)

            break

    return prev_event, next_event





def _piecewise_integral(segments: Sequence[Mapping[str, object]], local_step: int) -> float:

    remain = max(0, int(local_step))

    offset = 0.0

    for seg in segments:

        seg_len = max(0, int(seg.get("len", 0)))

        seg_delta = float(seg.get("Delta", 0.0))

        take = min(remain, seg_len)

        offset += seg_delta * float(take)

        remain -= take

        if remain <= 0:

            break

    return offset





def _piecewise_mix_value(

    *,

    omega: str,

    segments: Sequence[Mapping[str, object]],

    baseline_now: float,

    baseline_anchor: float,

    local_step: int,

) -> float:

    remain = max(0, int(local_step))

    cur = float(baseline_anchor if omega == "override" else baseline_now)

    for seg in segments:

        if not isinstance(seg, Mapping):

            continue

        seg_len = max(0, int(seg.get("len", 0)))

        if seg_len <= 0:

            continue

        take = min(remain, seg_len)

        kind = str(seg.get("kind", seg.get("pi", "ramp"))).strip().lower()

        if kind in {"hold", "h"}:

            if omega == "override":

                cur = float(seg.get("value", cur))

            else:

                cur = float(cur + float(seg.get("offset", 0.0)))

        elif kind in {"ramp", "r"}:

            delta = float(seg.get("Delta", 0.0))

            cur = float(cur + delta * float(take))

        remain -= take

        if remain <= 0:

            break

    return float(cur)





def _apply_event_value(

    *,

    omega: str,

    event: Mapping[str, object],

    baseline_now: float,

    baseline_anchor: float,

    t_step: int,

) -> float:

    pi = str(event.get("pi", "hold"))

    theta = event.get("theta", {})

    if not isinstance(theta, Mapping):

        theta = {}

    t_s = int(event.get("t_s", t_step))

    local_step = max(0, int(t_step) - t_s)



    if pi == "hold":

        if omega == "override":

            return float(theta.get("value", baseline_now))

        return float(baseline_now + float(theta.get("offset", 0.0)))



    if pi == "ramp":

        delta = float(theta.get("Delta", 0.0))

        if omega == "override":

            return float(baseline_anchor + delta * float(local_step))

        return float(baseline_now + delta * float(local_step))



    if pi == "piecewise_ramp":

        segments = theta.get("segments", [])

        if not isinstance(segments, list):

            segments = []

        offset = _piecewise_integral(segments, local_step)

        if omega == "override":

            return float(baseline_anchor + offset)

        return float(baseline_now + offset)



    if pi == "piecewise_mix":

        segments = theta.get("segments", [])

        if not isinstance(segments, list):

            segments = []

        return _piecewise_mix_value(

            omega=omega,

            segments=[seg for seg in segments if isinstance(seg, Mapping)],

            baseline_now=float(baseline_now),

            baseline_anchor=float(baseline_anchor),

            local_step=local_step,

        )



    return float(baseline_now)





class PayloadInjectionPolicy:

    """Translate M3 strategy payload into phase-specific override dicts."""



    def __init__(self, strategies: Sequence[Mapping[str, object]]):

        self._strategies = [s for s in strategies if isinstance(s, Mapping)]

        self._anchor_by_event: Dict[Tuple[int, int, str], float] = {}

        self._ramp_terminal_value_by_event: Dict[Tuple[int, int, str], float] = {}



    @property

    def has_rules(self) -> bool:

        return bool(self._strategies)



    def _binding_for(self, strategy: Mapping[str, object]) -> Optional[Tuple[str, str]]:

        w = str(strategy.get("w", ""))

        if w in FIELD_BINDINGS_BY_W:

            return FIELD_BINDINGS_BY_W[w]

        controller = str(strategy.get("controller", ""))

        return _controller_fallback_binding(controller, w)



    def __call__(self, t_step: int, phase: str, context: Dict[str, float]) -> Dict[str, float]:

        overrides: Dict[str, float] = {}

        for strategy_idx, strategy in enumerate(self._strategies):

            binding = self._binding_for(strategy)

            if binding is None:

                continue

            bind_phase, field = binding

            if bind_phase != phase or field not in context:

                continue



            active = _active_event(strategy, t_step)

            if active is None:

                prev_event, next_event = _surrounding_events(strategy, t_step)

                if prev_event is not None and (next_event is None or int(t_step) < int(next_event[1].get("t_s", t_step + 1))):

                    prev_idx, prev = prev_event

                    if str(prev.get("pi", "hold")).strip().lower() == "ramp":

                        held_key = (strategy_idx, prev_idx, field)

                        held_value = self._ramp_terminal_value_by_event.get(held_key)

                        if held_value is not None:

                            overrides[field] = float(held_value)

                continue



            event_idx, event = active

            omega = str(strategy.get("omega", "override"))

            base_now = float(context.get(field, 0.0))



            anchor_key = (strategy_idx, event_idx, field)

            if anchor_key not in self._anchor_by_event:

                self._anchor_by_event[anchor_key] = base_now

            base_anchor = self._anchor_by_event[anchor_key]



            value = _apply_event_value(

                omega=omega,

                event=event,

                baseline_now=base_now,

                baseline_anchor=base_anchor,

                t_step=t_step,

            )

            overrides[field] = float(value)

            if str(event.get("pi", "hold")).strip().lower() == "ramp":

                t_e = int(event.get("t_e", int(t_step) + 1))

                if int(t_step) >= (t_e - 1):

                    self._ramp_terminal_value_by_event[(strategy_idx, event_idx, field)] = float(value)



        return overrides





class ClosedLoopSim:

    """Closed-loop simulator with optional CSV and neutral override hook."""



    def __init__(

        self,

        Ts: float = 0.1,

        init_state: Optional[Mapping[str, float]] = None,

        enable_csv: bool = False,

        csv_path: Optional[str] = None,

    ):

        self.Ts = float(Ts)

        self.enable_csv = bool(enable_csv)

        self.csv_path = Path(csv_path) if csv_path else Path("runs/closed_loop_trace.csv")



        self.controller = PLCController(Ts=self.Ts, init_state=None)

        self.process = PhysicalProcess(Ts=self.Ts, init_state=init_state)



        self._csv_file = None

        self._csv_writer: Optional[csv.DictWriter] = None

        self._next_step = 0

        self._state_min: Dict[str, float] = {}

        self._state_max: Dict[str, float] = {}

        self.reset(init_state=init_state)



    def _csv_fields(self) -> List[str]:

        return [

            "t_step",

            "t_time_s",

            "ctrl_loadgen_load_output",

            "ctrl_steam_main_steam_pressure_setpoint",

            "ctrl_boiler_boiler_setpoint",

            "ctrl_waterk_water_setpoint",

            "phys_fuel_command",

            "phys_water_pump_speed",

            "phys_true_load",

            "phys_coal_flow",

            "phys_feedwater_flow",

            "phys_main_steam_flow",

            "phys_waterwall_temp",

            "phys_main_steam_pressure",

            "alarm_boiler_cut_margin",

            "alarm_boiler_cut_triggered",

            "alarm_fuel_cut_margin",

            "alarm_fuel_cut_triggered",

            "alarm_water_cut_margin",

            "alarm_water_cut_triggered",

            "alarm_any_triggered",

            "hazard_waterwall_margin",

            "hazard_waterwall_triggered",

            "hazard_pressure_margin",

            "hazard_pressure_triggered",

            "hazard_feedwater_margin",

            "hazard_feedwater_triggered",

            "hazard_any_triggered",

        ]



    def _controller_internal_state_for_m5(self) -> Dict[str, float]:

        ctrl = self.controller

        return {

            "load_rate_prev": float(ctrl.loadGen.rate_limiter.prev_output),

            "steam_hfop1_prev": float(ctrl.steamControl.hfop1.av_0),

            "steam_hfop2_prev": float(ctrl.steamControl.hfop2.av_0),

            "steam_hfop3_prev": float(ctrl.steamControl.hfop3.av_0),

            "steam_hfop4_prev": float(ctrl.steamControl.hfop4.av_0),

            "steam_rate_prev": float(ctrl.steamControl.rate.prev_output),

            "boiler_hfop1_prev": float(ctrl.boilerControl.hfop1.av_0),

            "boiler_pid_av0": float(ctrl.boilerControl.HSV.AV_0),

            "boiler_pid_err0": float(ctrl.boilerControl.HSV.err_0),

            "boiler_pid_delta_err0": float(ctrl.boilerControl.HSV.delta_err_0),

            "boiler_pid_dk0": float(ctrl.boilerControl.HSV.dk_0),

            "fuel_hfop1_prev": float(ctrl.fuelControl.hfop1.av_0),

            "fuel_rate_prev": float(ctrl.fuelControl.rate.prev_output),

            "fuel_pid_av0": float(ctrl.fuelControl.hsvpid.AV_0),

            "fuel_pid_err0": float(ctrl.fuelControl.hsvpid.err_0),

            "fuel_pid_delta_err0": float(ctrl.fuelControl.hsvpid.delta_err_0),

            "fuel_pid_dk0": float(ctrl.fuelControl.hsvpid.dk_0),

            "water_hfop_prev": float(ctrl.waterControl.hfop.av_0),

            "water_rate_prev": float(ctrl.waterControl.rate.prev_output),

            "water_pid_av0": float(ctrl.waterControl.pid.AV_0),

            "water_pid_err0": float(ctrl.waterControl.pid.err_0),

            "water_pid_delta_err0": float(ctrl.waterControl.pid.delta_err_0),

            "water_pid_dk0": float(ctrl.waterControl.pid.dk_0),

            "waterk_hfop1_prev": float(ctrl.waterK.hfop1.av_0),

        }



    def _open_csv(self) -> None:

        if not self.enable_csv:

            return

        if self._csv_file is not None:

            self._csv_file.close()

            self._csv_file = None

            self._csv_writer = None



        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")

        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields())

        self._csv_writer.writeheader()



    def _close_csv(self) -> None:

        if self._csv_file is not None:

            self._csv_file.close()

            self._csv_file = None

            self._csv_writer = None



    def reset(self, init_state: Optional[Mapping[str, float]] = None) -> None:

        self.controller.reset(init_state=None)

        self.process.reset(init_state=init_state)

        self._next_step = 0



        state = self.process.get_state()

        self._state_min = {k: float(v) for k, v in state.items()}

        self._state_max = {k: float(v) for k, v in state.items()}

        self._open_csv()



    def _update_state_stats(self, state: Mapping[str, float]) -> None:

        for key, value in state.items():

            v = float(value)

            if key not in self._state_min:

                self._state_min[key] = v

                self._state_max[key] = v

            else:

                self._state_min[key] = min(self._state_min[key], v)

                self._state_max[key] = max(self._state_max[key], v)



    def _apply_hook(

        self,

        injection_hook: Optional[InjectionHook],

        t_step: int,

        phase: str,

        context: Mapping[str, float],

    ) -> Dict[str, float]:

        if injection_hook is None:

            return {}

        out = injection_hook(t_step, phase, dict(context))

        if out is None:

            return {}

        if not isinstance(out, dict):

            raise TypeError("injection_hook must return dict or None")

        return {str(k): float(v) for k, v in out.items()}



    def _apply_payload_replacement(

        self,

        target: Dict[str, float],

        *,

        allowed_keys: Sequence[str],

        overrides: Mapping[str, float],

    ) -> bool:

        """Apply payload replacement at one injection point.



        Returns True if at least one field is replaced, else False.

        """

        changed = False

        for key in allowed_keys:

            if key not in overrides:

                continue

            new_val = float(overrides[key])

            old_val = float(target.get(key, new_val))

            if old_val != new_val:

                changed = True

            target[key] = new_val

        return changed



    def step(self, t_step: int, injection_hook: Optional[InjectionHook] = None) -> Dict[str, object]:

        return self._step_with_ignore(

            t_step=t_step,

            injection_hook=injection_hook,

            ignore_protection_predicates_for_enforcement=None,

        )



    def _step_with_ignore(

        self,

        *,

        t_step: int,

        injection_hook: Optional[InjectionHook],

        ignore_protection_predicates_for_enforcement: Optional[Set[str]],

    ) -> Dict[str, object]:

        sensors = self.process.get_state()

        self._update_state_stats(sensors)

        ctrl_internal = self._controller_internal_state_for_m5()



        ctrl_work: Dict[str, float] = {}

        controller_modified = False



        # 1) Load command output.

        load_output = self.controller.loadGen.update(float(sensors["true_load"]))

        ctrl_work["load_output"] = float(load_output)

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "controller_load_output",

            {**sensors, **ctrl_work},

        )

        controller_modified = self._apply_payload_replacement(

            ctrl_work,

            allowed_keys=("load_output",),

            overrides=overrides,

        ) or controller_modified



        # 2) Steam pressure setpoint output, depends on possibly modified load_output.

        _hs_curve, steam_setpoint = self.controller.steamControl.update(

            float(ctrl_work["load_output"]),

            0.0,

            1800.0,

        )

        ctrl_work["steam_setpoint"] = float(steam_setpoint)

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "controller_steam_hslim",

            {**sensors, **ctrl_work},

        )

        controller_modified = self._apply_payload_replacement(

            ctrl_work,

            allowed_keys=("steam_setpoint",),

            overrides=overrides,

        ) or controller_modified



        # 3) Boiler command output, depends on steam_setpoint.

        boiler_setpoint = self.controller.boilerControl.update(

            float(sensors["main_steam_pressure"]),

            float(ctrl_work["steam_setpoint"]),

        )

        ctrl_work["boiler_setpoint"] = float(boiler_setpoint)

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "controller_boiler_setpoint",

            {**sensors, **ctrl_work},

        )

        controller_modified = self._apply_payload_replacement(

            ctrl_work,

            allowed_keys=("boiler_setpoint",),

            overrides=overrides,

        ) or controller_modified



        # 4) Fuel command output, depends on boiler_setpoint.

        fuel_command = self.controller.fuelControl.update(

            5.0,

            float(sensors["coal_flow"]),

            float(ctrl_work["boiler_setpoint"]),

        )

        ctrl_work["fuel_command"] = float(fuel_command)

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "controller_fuel_command",

            {**sensors, **ctrl_work},

        )

        controller_modified = self._apply_payload_replacement(

            ctrl_work,

            allowed_keys=("fuel_command",),

            overrides=overrides,

        ) or controller_modified



        # 5) Water flow setpoint output, depends on boiler_setpoint.

        water_setpoint = self.controller.waterK.update(float(ctrl_work["boiler_setpoint"]), 0.0)

        ctrl_work["water_setpoint"] = float(water_setpoint)

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "controller_water_setpoint",

            {**sensors, **ctrl_work},

        )

        controller_modified = self._apply_payload_replacement(

            ctrl_work,

            allowed_keys=("water_setpoint",),

            overrides=overrides,

        ) or controller_modified



        # 6) Water pump output, depends on water_setpoint.

        water_pump_speed = self.controller.waterControl.update(

            float(sensors["main_steam_flow"]),

            float(sensors["feedwater_flow"]),

            float(ctrl_work["water_setpoint"]),

        )

        ctrl_work["water_pump_speed"] = float(water_pump_speed)

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "controller_water_pump",

            {**sensors, **ctrl_work},

        )

        controller_modified = self._apply_payload_replacement(

            ctrl_work,

            allowed_keys=("water_pump_speed",),

            overrides=overrides,

        ) or controller_modified



        cmd_applied = baseline_u0(t_step, sensors, ctrl_work)



        # Injection point 2: actuators replacement.

        overrides = self._apply_hook(

            injection_hook,

            t_step,

            "actuators",

            {**sensors, **cmd_applied},

        )

        actuators_modified = self._apply_payload_replacement(

            cmd_applied,

            allowed_keys=tuple(cmd_applied.keys()),

            overrides=overrides,

        )



        trip_waterwall = float(sensors["waterwall_temp"]) > TRIP_WATERWALL

        trip_pressure = float(sensors["main_steam_pressure"]) > TRIP_PRESSURE

        trip_feedwater = float(sensors["feedwater_flow"]) <= TRIP_FEEDWATER

        trip_any = trip_waterwall or trip_pressure or trip_feedwater



        if (trip_waterwall + trip_pressure + trip_feedwater) > 1:

            trip_type = "multi"

        elif trip_waterwall:

            trip_type = "waterwall"

        elif trip_pressure:

            trip_type = "pressure"

        elif trip_feedwater:

            trip_type = "feedwater"

        else:

            trip_type = "none"



        boiler_alarm = self.controller.boilerControl.check_alarm(

            steam_setpoint=float(cmd_applied["steam_setpoint"]),

            main_steam_pressure=float(sensors["main_steam_pressure"]),

        )

        fuel_alarm = self.controller.fuelControl.check_alarm(

            coal_flow=float(sensors["coal_flow"]),

            fuel_command=float(cmd_applied["fuel_command"]),

        )

        water_alarm = self.controller.waterControl.check_alarm(

            water_setpoint=float(cmd_applied["water_setpoint"]),

            feedwater_flow=float(sensors["feedwater_flow"]),

        )



        boiler_cut = boiler_alarm.triggered

        fuel_cut = fuel_alarm.triggered

        water_cut = water_alarm.triggered

        alarm_boiler_cut_margin = boiler_alarm.margin

        alarm_fuel_cut_margin = fuel_alarm.margin

        alarm_water_cut_margin = water_alarm.margin

        alarm_any_triggered = boiler_cut or fuel_cut or water_cut

        alarm_rule_ids = ", ".join(

            a.alarm_id for a in (boiler_alarm, fuel_alarm, water_alarm) if a.triggered

        )

        hazard_waterwall_margin = TRIP_WATERWALL - float(sensors["waterwall_temp"])

        hazard_pressure_margin = TRIP_PRESSURE - float(sensors["main_steam_pressure"])

        hazard_feedwater_margin = float(sensors["feedwater_flow"]) - TRIP_FEEDWATER



        ignored_for_enforce = {

            str(x).strip().upper()

            for x in (ignore_protection_predicates_for_enforcement or set())

            if str(x).strip()

        }

        enforce_boiler_cut = bool(boiler_cut) and ("BOILER_CUT_TH" not in ignored_for_enforce)

        enforce_fuel_cut = bool(fuel_cut) and ("FUEL_CUT_TH" not in ignored_for_enforce)

        enforce_water_cut = bool(water_cut) and ("WATER_CUT_TH" not in ignored_for_enforce)



        fuel_applied = float(cmd_applied["fuel_command"])

        water_applied = float(cmd_applied["water_pump_speed"])







        if not trip_any:

            self.process.step(

                {

                    "fuel_command_0_applied": fuel_applied,

                    "water_pump_speed_0_applied": water_applied,

                },

                t_step=t_step,

            )



        boiler_pid_dbg = self.controller.boilerControl.last_debug.get("pid", {})

        if not isinstance(boiler_pid_dbg, Mapping):

            boiler_pid_dbg = {}

        fuel_dbg = self.controller.fuelControl.last_debug

        fuel_pid_dbg = fuel_dbg.get("pid", {}) if isinstance(fuel_dbg, Mapping) else {}

        if not isinstance(fuel_pid_dbg, Mapping):

            fuel_pid_dbg = {}

        steam_dbg = self.controller.steamControl.last_debug

        if not isinstance(steam_dbg, Mapping):

            steam_dbg = {}

        waterk_dbg = self.controller.waterK.last_debug

        if not isinstance(waterk_dbg, Mapping):

            waterk_dbg = {}

        water_dbg = self.controller.waterControl.last_debug

        water_pid_dbg = water_dbg.get("pid", {}) if isinstance(water_dbg, Mapping) else {}

        if not isinstance(water_pid_dbg, Mapping):

            water_pid_dbg = {}

        load_dbg = self.controller.loadGen.last_debug

        if not isinstance(load_dbg, Mapping):

            load_dbg = {}



        record: Dict[str, object] = {

            "t_step": int(t_step),

            "t_time_s": float(t_step) * self.Ts,

            "true_load": float(sensors["true_load"]),

            "coal_flow": float(sensors["coal_flow"]),

            "feedwater_flow": float(sensors["feedwater_flow"]),

            "main_steam_flow": float(sensors["main_steam_flow"]),

            "waterwall_temp": float(sensors["waterwall_temp"]),

            "main_steam_pressure": float(sensors["main_steam_pressure"]),

            "load_output": float(ctrl_work["load_output"]),

            "steam_setpoint": float(ctrl_work["steam_setpoint"]),

            "boiler_setpoint": float(ctrl_work["boiler_setpoint"]),

            "fuel_command": float(ctrl_work["fuel_command"]),

            "water_setpoint": float(ctrl_work["water_setpoint"]),

            "water_pump_speed": float(ctrl_work["water_pump_speed"]),

            "phys_fuel_command": float(fuel_applied),

            "phys_water_pump_speed": float(water_applied),

            "trip_any": bool(trip_any),

            "trip_type": trip_type,

            "trip_waterwall": bool(trip_waterwall),

            "trip_pressure": bool(trip_pressure),

            "trip_feedwater": bool(trip_feedwater),

            "boiler_cut": bool(boiler_cut),

            "fuel_cut": bool(fuel_cut),

            "water_cut": bool(water_cut),

            "ctrl_loadgen_load_output": float(ctrl_work["load_output"]),

            "ctrl_steam_main_steam_pressure_setpoint": float(ctrl_work["steam_setpoint"]),

            "ctrl_boiler_boiler_setpoint": float(ctrl_work["boiler_setpoint"]),

            "ctrl_waterk_water_setpoint": float(ctrl_work["water_setpoint"]),

            "phys_fuel_command": float(fuel_applied),

            "phys_water_pump_speed": float(water_applied),

            "phys_true_load": float(sensors["true_load"]),

            "phys_coal_flow": float(sensors["coal_flow"]),

            "phys_feedwater_flow": float(sensors["feedwater_flow"]),

            "phys_main_steam_flow": float(sensors["main_steam_flow"]),

            "phys_waterwall_temp": float(sensors["waterwall_temp"]),

            "phys_main_steam_pressure": float(sensors["main_steam_pressure"]),

            "alarm_boiler_cut_margin": float(alarm_boiler_cut_margin),

            "alarm_boiler_cut_triggered": bool(boiler_cut),

            "alarm_fuel_cut_margin": float(alarm_fuel_cut_margin),

            "alarm_fuel_cut_triggered": bool(fuel_cut),

            "alarm_water_cut_margin": float(alarm_water_cut_margin),

            "alarm_water_cut_triggered": bool(water_cut),

            "alarm_any_triggered": bool(alarm_any_triggered),

            "alarm_rule_ids": str(alarm_rule_ids),

            "hazard_waterwall_margin": float(hazard_waterwall_margin),

            "hazard_waterwall_triggered": bool(trip_waterwall),

            "hazard_pressure_margin": float(hazard_pressure_margin),

            "hazard_pressure_triggered": bool(trip_pressure),

            "hazard_feedwater_margin": float(hazard_feedwater_margin),

            "hazard_feedwater_triggered": bool(trip_feedwater),

            "hazard_any_triggered": bool(trip_any),

            "setpoints_modified": bool(controller_modified),

            "actuators_modified": bool(actuators_modified),

            "br_load_rate_mode": str(load_dbg.get("rate_mode", "")),

            "br_steam_hs_mode": str(steam_dbg.get("hs_hsc_mode", "")),

            "br_steam_hs_segment": int(steam_dbg.get("hs_hsc_segment", -1)),

            "br_steam_rate_mode": str(steam_dbg.get("rate_mode", "")),

            "br_boiler_pid_deadband_hit": bool(boiler_pid_dbg.get("deadband_hit", False)),

            "br_boiler_pid_integral_on": bool(boiler_pid_dbg.get("integral_on", False)),

            "br_boiler_pid_td_on": bool(boiler_pid_dbg.get("td_on", False)),

            "br_boiler_pid_sat_hi": bool(boiler_pid_dbg.get("sat_hi", False)),

            "br_boiler_pid_sat_lo": bool(boiler_pid_dbg.get("sat_lo", False)),

            "br_boiler_pid_ov_hi_clip": bool(boiler_pid_dbg.get("ov_hi_clip", False)),

            "br_boiler_pid_ov_lo_clip": bool(boiler_pid_dbg.get("ov_lo_clip", False)),

            "br_fuel_pt_mode": str(fuel_dbg.get("pt_hsc_mode", "")) if isinstance(fuel_dbg, Mapping) else "",

            "br_fuel_pt_segment": int(fuel_dbg.get("pt_hsc_segment", -1)) if isinstance(fuel_dbg, Mapping) else -1,

            "br_fuel_ti_mode": str(fuel_dbg.get("ti_hsc_mode", "")) if isinstance(fuel_dbg, Mapping) else "",

            "br_fuel_ti_segment": int(fuel_dbg.get("ti_hsc_segment", -1)) if isinstance(fuel_dbg, Mapping) else -1,

            "br_fuel_rate_mode": str(fuel_dbg.get("rate_mode", "")) if isinstance(fuel_dbg, Mapping) else "",

            "br_fuel_pid_deadband_hit": bool(fuel_pid_dbg.get("deadband_hit", False)),

            "br_fuel_pid_integral_on": bool(fuel_pid_dbg.get("integral_on", False)),

            "br_fuel_pid_td_on": bool(fuel_pid_dbg.get("td_on", False)),

            "br_fuel_pid_sat_hi": bool(fuel_pid_dbg.get("sat_hi", False)),

            "br_fuel_pid_sat_lo": bool(fuel_pid_dbg.get("sat_lo", False)),

            "br_fuel_pid_ov_hi_clip": bool(fuel_pid_dbg.get("ov_hi_clip", False)),

            "br_fuel_pid_ov_lo_clip": bool(fuel_pid_dbg.get("ov_lo_clip", False)),

            "br_waterk_hsc_mode": str(waterk_dbg.get("hsc_mode", "")),

            "br_waterk_hsc_segment": int(waterk_dbg.get("hsc_segment", -1)),

            "br_waterk_floor_clip": bool(waterk_dbg.get("floor_clip", False)),

            "br_water_pt_mode": str(water_dbg.get("pt_hsc_mode", "")) if isinstance(water_dbg, Mapping) else "",

            "br_water_pt_segment": int(water_dbg.get("pt_hsc_segment", -1)) if isinstance(water_dbg, Mapping) else -1,

            "br_water_ti_mode": str(water_dbg.get("ti_hsc_mode", "")) if isinstance(water_dbg, Mapping) else "",

            "br_water_ti_segment": int(water_dbg.get("ti_hsc_segment", -1)) if isinstance(water_dbg, Mapping) else -1,

            "br_water_rate_mode": str(water_dbg.get("rate_mode", "")) if isinstance(water_dbg, Mapping) else "",

            "br_water_pid_deadband_hit": bool(water_pid_dbg.get("deadband_hit", False)),

            "br_water_pid_integral_on": bool(water_pid_dbg.get("integral_on", False)),

            "br_water_pid_td_on": bool(water_pid_dbg.get("td_on", False)),

            "br_water_pid_sat_hi": bool(water_pid_dbg.get("sat_hi", False)),

            "br_water_pid_sat_lo": bool(water_pid_dbg.get("sat_lo", False)),

            "br_water_pid_ov_hi_clip": bool(water_pid_dbg.get("ov_hi_clip", False)),

            "br_water_pid_ov_lo_clip": bool(water_pid_dbg.get("ov_lo_clip", False)),

        }

        record.update(ctrl_internal)



        if self._csv_writer is not None:

            self._csv_writer.writerow({k: record.get(k) for k in self._csv_fields()})



        self._next_step = max(self._next_step, int(t_step) + 1)

        self.controller.last_output = {

            "load_output": float(ctrl_work["load_output"]),

            "steam_setpoint": float(ctrl_work["steam_setpoint"]),

            "boiler_setpoint": float(ctrl_work["boiler_setpoint"]),

            "fuel_command": float(ctrl_work["fuel_command"]),

            "water_setpoint": float(ctrl_work["water_setpoint"]),

            "water_pump_speed": float(ctrl_work["water_pump_speed"]),

        }

        return record



    def run(

        self,

        steps: int,

        injection_hook: Optional[InjectionHook] = None,

        return_trace: bool = True,

        stop_on_trip: bool = True,

        ignore_protection_predicates_for_stop: Optional[Sequence[str]] = None,

        ignore_protection_predicates_for_enforcement: Optional[Sequence[str]] = None,

    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:

        trace: List[Dict[str, object]] = []

        stop_reason = "steps_exhausted"

        stop_step: Optional[int] = None

        steps_run = 0

        ignored_for_stop = {

            str(x).strip().upper()

            for x in (ignore_protection_predicates_for_stop or [])

            if str(x).strip()

        }

        ignored_for_enforce = {

            str(x).strip().upper()

            for x in (ignore_protection_predicates_for_enforcement or [])

            if str(x).strip()

        }



        for _ in range(int(steps)):

            t_step = self._next_step

            record = self._step_with_ignore(

                t_step=t_step,

                injection_hook=injection_hook,

                ignore_protection_predicates_for_enforcement=ignored_for_enforce,

            )

            steps_run += 1

            if return_trace:

                trace.append(record)

            cut_any_for_stop = _effective_cut_any_for_stop(

                record,

                ignored_protection_predicates=ignored_for_stop,

            )

            if stop_on_trip and (bool(record["trip_any"]) or cut_any_for_stop):

                stop_reason = "trip" if bool(record["trip_any"]) else "cut"

                stop_step = int(record["t_step"])

                break



        meta: Dict[str, object] = {

            "stop_reason": stop_reason,

            "stop_step": stop_step,

            "steps_run": steps_run,

            "min": dict(self._state_min),

            "max": dict(self._state_max),

        }

        return trace, meta



    def close(self) -> None:

        self._close_csv()



    def __del__(self) -> None:  # pragma: no cover - defensive cleanup

        self._close_csv()





class SimRunner:

    """Wrapper used by M4 CEGIS to evaluate one M3 payload."""



    def __init__(

        self,

        Ts: float = 0.1,

        init_state: Optional[Mapping[str, float]] = None,

        controller_version: str = "m4_plc_1x",

    ):

        self.Ts = float(Ts)

        self.init_state = dict(init_state) if init_state is not None else None

        self.controller_version = controller_version



    def run_payload(

        self,

        payload: Mapping[str, object],

        *,

        steps: Optional[int] = None,

        enable_csv: bool = False,

        csv_path: Optional[Path] = None,

        return_trace: bool = True,

        stop_on_trip: bool = True,

        ignore_protection_predicates_for_stop: Optional[Sequence[str]] = None,

        ignore_protection_predicates_for_enforcement: Optional[Sequence[str]] = None,

    ) -> Dict[str, object]:

        payload_meta = payload.get("meta", {})

        payload_steps = int(payload_meta.get("T", 50)) if isinstance(payload_meta, Mapping) else 50

        n_steps = int(steps) if steps is not None else payload_steps



        strategies_obj = payload.get("strategies", [])

        strategies: List[Mapping[str, object]]

        if isinstance(strategies_obj, list):

            strategies = [s for s in strategies_obj if isinstance(s, Mapping)]

        else:

            strategies = []



        sim = ClosedLoopSim(

            Ts=self.Ts,

            init_state=self.init_state,

            enable_csv=bool(enable_csv),

            csv_path=str(csv_path) if csv_path is not None else None,

        )

        policy = PayloadInjectionPolicy(strategies)

        hook: Optional[InjectionHook] = policy if policy.has_rules else None

        trace, meta = sim.run(

            steps=n_steps,

            injection_hook=hook,

            return_trace=return_trace,

            stop_on_trip=stop_on_trip,

            ignore_protection_predicates_for_stop=ignore_protection_predicates_for_stop,

            ignore_protection_predicates_for_enforcement=ignore_protection_predicates_for_enforcement,

        )

        sim.close()



        trip_step = None

        trip_any = False

        cut_any = False

        for row in trace:

            if bool(row.get("trip_any", False)) and trip_step is None:

                trip_step = int(row.get("t_step", 0))

            trip_any = trip_any or bool(row.get("trip_any", False))

            cut_any = cut_any or bool(row.get("boiler_cut", False)) or bool(row.get("fuel_cut", False)) or bool(

                row.get("water_cut", False)

            )



        sim_summary = {

            "controller_version": self.controller_version,

            "steps": int(n_steps),

            "dt": float(self.Ts),

            "stop_reason": str(meta.get("stop_reason", "steps_exhausted")),

            "trip_any": bool(trip_any),

            "cut_any": bool(cut_any),

            "trip_step": trip_step,

        }

        return {

            "trace": trace,

            "meta": meta,

            "sim": sim_summary,

        }





def get_initial_baseline_cmds(sim: ClosedLoopSim, warmup_steps: int = 1) -> Dict[str, float]:

    """Run a short reset warmup and return baseline command snapshot."""

    sim.reset()

    last_record: Optional[Dict[str, object]] = None

    for step in range(max(1, int(warmup_steps))):

        last_record = sim.step(t_step=step, injection_hook=None)



    assert last_record is not None

    return {

        "fuel_command": float(last_record["fuel_command"]),

        "water_pump_speed": float(last_record["water_pump_speed"]),

        "steam_setpoint": float(last_record["steam_setpoint"]),

        "water_setpoint": float(last_record["water_setpoint"]),

    }





def _main_demo() -> None:

    sim = ClosedLoopSim(Ts=0.1, enable_csv=False)

    trace, meta = sim.run(steps=200, injection_hook=None, return_trace=True)



    for row in trace:

        for key in REQUIRED_TRACE_KEYS:

            if key not in row:

                raise AssertionError(f"missing key in trace row: {key}")

        for key in BRANCH_TRACE_KEYS:

            if key not in row:

                raise AssertionError(f"missing branch key in trace row: {key}")

        for key in M5_CONTROLLER_INTERNAL_TRACE_KEYS:

            if key not in row:

                raise AssertionError(f"missing M5 internal key in trace row: {key}")



    payload = {

        "meta": {"payload_id": "demo", "T": 50},

        "strategies": [

            {

                "controller": "MainSteamPressureSetpointController",

                "w": "MainSteamPressureSetpoint",

                "type": "SP",

                "omega": "override",

                "E_w": [{"t_s": 0, "t_e": 25, "pi": "hold", "theta": {"value": 28.0}}],

            }

        ],

        "constraints": {"static": []},

    }

    runner = SimRunner(Ts=0.1)

    out = runner.run_payload(payload, steps=50, return_trace=False)



    print(f"stop_reason={meta['stop_reason']} stop_step={meta['stop_step']} steps_run={meta['steps_run']}")

    print(f"min={meta['min']}")

    print(f"max={meta['max']}")

    print(json.dumps(out["sim"], ensure_ascii=False))





if __name__ == "__main__":

    _main_demo()




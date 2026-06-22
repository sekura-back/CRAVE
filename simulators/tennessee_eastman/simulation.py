# TE Closed-Loop Simulation - Section 4.1

"""TEP closed-loop simulation for the VeriPro Tennessee Eastman backend.


13-phase injection pipeline (12 controller_xmv_* + 1 actuators):

    controller_xmv_01  controls D Feed Flow

    controller_xmv_02  controls E Feed Flow

    controller_xmv_03  controls A Feed Flow

    controller_xmv_04  controls A+C Feed Flow

    controller_xmv_05  controls Compressor Recycle Valve

    controller_xmv_06  controls Purge Valve

    controller_xmv_07  controls Separator Pot Liquid Flow

    controller_xmv_08  controls Stripper Liquid Product Flow

    controller_xmv_09  controls Stripper Steam Valve

    controller_xmv_10  controls Reactor Cooling Water Flow

    controller_xmv_11  controls Condenser Cooling Water Flow

    controller_xmv_12  controls Agitator Speed

    actuators          final override layer applied to xmv_NN_applied



Hooks see {f"xmv_{NN}": value} during the matching ``controller_xmv_NN`` phase

and {f"xmv_{NN}_applied": value} at the ``actuators`` phase. Returning a dict

with those keys overrides the value before it reaches the physics.

"""

from __future__ import annotations



import csv

import os

import pickle

from pathlib import Path

from typing import (

    Any,

    Callable,

    Dict,

    List,

    Mapping,

    Optional,

    Sequence,

    Set,

    Tuple,

)



import numpy as np



try:

    from .controller import PLCController

    from .physical_process import PhysicalProcess, MEASUREMENT_KEYS, MV_KEYS

    from .tep.constants import SAFETY_LIMITS

except ImportError:  # pragma: no cover - direct-script fallback

    from controller import PLCController  # type: ignore

    from physical_process import PhysicalProcess, MEASUREMENT_KEYS, MV_KEYS  # type: ignore

    from tep.constants import SAFETY_LIMITS  # type: ignore





InjectionHook = Callable[[int, str, Dict[str, float]], Optional[Dict[str, float]]]

"""(t_step, phase, context_dict) -> override dict or None."""





# ---------------------------------------------------------------------------

# Setpoint injection: which entries of ``DecentralizedController.setpoints``

# are exposed as injection points.  Cascade-driven entries (Loop 13/14/15/

# 17/18/19/16-via-20) plus the few static SPs used for direct loops are all

# listed here; any field name in this dict can be overridden via the

# ``controller_setpoints`` phase.

# ---------------------------------------------------------------------------

SETPOINT_FIELDS: Dict[str, int] = {

    "setpoints_0": 0,    # D feed SP (cascade target of Loop 14)

    "setpoints_1": 1,    # E feed SP (cascade target of Loop 15)

    "setpoints_2": 2,    # A feed SP (cascade target of Loop 13)

    "setpoints_3": 3,    # A+C feed SP (cascade target of Loop 17)

    "setpoints_5": 5,    # Purge SP (cascade target of Loop 19)

    "setpoints_8": 8,    # Steam SP (cascade target of Loop 16)

    "setpoints_9": 9,    # Reactor CW SP (cascade target of Loop 18)

    "setpoints_15": 15,  # Stripper temp SP (cascade target of Loop 20)

}





# Phase names (must match the manifest's injection_points[].phase entries).

INJECTION_PHASES: Tuple[str, ...] = (

    "controller_setpoints",

) + tuple(

    f"controller_xmv_{i:02d}" for i in range(1, 13)

) + ("actuators",)





# ---------------------------------------------------------------------------

# Hazard thresholds (sourced from the bundled SafetyLimits dataclass so the

# definition stays in lockstep with PythonTEProcess.is_shutdown).

# ---------------------------------------------------------------------------

HAZARD_REACTOR_PRESSURE_MAX = float(SAFETY_LIMITS.reactor_pressure_max)

HAZARD_REACTOR_TEMP_MAX = float(SAFETY_LIMITS.reactor_temp_max)

HAZARD_REACTOR_LEVEL_MAX = float(SAFETY_LIMITS.reactor_level_max)

HAZARD_REACTOR_LEVEL_MIN = float(SAFETY_LIMITS.reactor_level_min)

HAZARD_SEPARATOR_LEVEL_MAX = float(SAFETY_LIMITS.separator_level_max)

HAZARD_SEPARATOR_LEVEL_MIN = float(SAFETY_LIMITS.separator_level_min)

HAZARD_STRIPPER_LEVEL_MAX = float(SAFETY_LIMITS.stripper_level_max)

HAZARD_STRIPPER_LEVEL_MIN = float(SAFETY_LIMITS.stripper_level_min)





def _env_float(name: str, default: float) -> float:

    raw = os.environ.get(name)

    if raw is None:

        return float(default)

    try:

        return float(raw)

    except ValueError:

        return float(default)





# Absolute alarm thresholds for tracked PV-SP deviations.

ALARM_DFEED_ABS = 350.0

ALARM_EFEED_ABS = 350.0

ALARM_AFEED_ABS = _env_float("VERIPRO_TE_ALARM_AFEED_ABS", 0.5)

ALARM_ACFEED_ABS = _env_float("VERIPRO_TE_ALARM_ACFEED_ABS", 0.7)

ALARM_RECYCLE_ABS = _env_float("VERIPRO_TE_ALARM_RECYCLE_ABS", 1.4)

ALARM_CW_ABS = _env_float("VERIPRO_TE_ALARM_CW_ABS", 2.5)

ALARM_PURGE_ABS = _env_float("VERIPRO_TE_ALARM_PURGE_ABS", 0.2)



DEFAULT_MODE1_SNAPSHOT = Path(__file__).with_name("te_steady_snapshot.pkl")





class _SnapshotUnpickler(pickle.Unpickler):

    """Map bundled snapshots saved before the canonical simulator package rename."""

    MODULE_ALIASES = {
        "close_loop" + "_TE.tep.python_backend": "simulators.tennessee_eastman.tep.python_backend",
    }

    def find_class(self, module: str, name: str) -> Any:

        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


def _load_snapshot_pickle(path: Path) -> Mapping[str, object]:

    with path.open("rb") as handle:

        return _SnapshotUnpickler(handle).load()


class ClosedLoopSim:
    """TEP closed-loop simulator with 13-phase injection pipeline."""



    def __init__(

        self,

        Ts: float = 1.0,

        *,

        mode: int = 1,

        random_seed: Optional[int] = None,

        init_state: Optional[Mapping[str, float]] = None,

        idv: Optional[Mapping[int, int]] = None,

        use_bundled_snapshot: bool = True,

        enable_csv: bool = False,

        csv_path: Optional[str] = None,

    ):

        self.Ts = float(Ts)

        self.mode = int(mode)

        self._random_seed = random_seed

        self._default_init_state = dict(init_state) if init_state is not None else None

        self._default_idv = dict(idv) if idv is not None else None

        self._use_bundled_snapshot = bool(use_bundled_snapshot)

        self._csv_enabled = bool(enable_csv)

        self._csv_file = None

        self._csv_writer: Optional[csv.DictWriter] = None

        self.process = PhysicalProcess(

            Ts=self.Ts,

            random_seed=random_seed,

            init_state=init_state,

            idv=idv,

        )

        self.controller = PLCController(

            Ts=self.Ts,

            mode=self.mode,

            init_state=init_state,

        )



        self._next_step = 0

        self._state_min: Dict[str, float] = {}

        self._state_max: Dict[str, float] = {}

        self._initialize_default_state(init_state=init_state, idv=idv)

        if self._csv_enabled:

            p = Path(csv_path) if csv_path else Path("tep_sim_trace.csv")

            p.parent.mkdir(parents=True, exist_ok=True)

            self._csv_file = open(p, "w", newline="", encoding="utf-8")



    # ---- helpers ---------------------------------------------------------



    def _apply_hook(

        self,

        t_step: int,

        phase: str,

        context: Dict[str, float],

        injection_hook: Optional[InjectionHook],

    ) -> Tuple[Dict[str, float], bool]:

        if injection_hook is None:

            return context, False

        overrides = injection_hook(int(t_step), str(phase), dict(context))

        if not overrides:

            return context, False

        if not isinstance(overrides, dict):

            raise TypeError("injection_hook must return dict or None")

        merged = dict(context)

        for k, v in overrides.items():

            merged[str(k)] = float(v)

        return merged, True



    def _update_minmax(self, state: Mapping[str, float]) -> None:

        for k, v in state.items():

            fv = float(v)

            if k not in self._state_min:

                self._state_min[k] = fv

                self._state_max[k] = fv

            else:

                if fv < self._state_min[k]:

                    self._state_min[k] = fv

                if fv > self._state_max[k]:

                    self._state_max[k] = fv



    def _write_csv_row(self, record: Dict[str, object]) -> None:

        if not self._csv_enabled or self._csv_file is None:

            return

        if self._csv_writer is None:

            self._csv_writer = csv.DictWriter(

                self._csv_file, fieldnames=list(record.keys())

            )

            self._csv_writer.writeheader()

        self._csv_writer.writerow(record)



    def _close_csv(self) -> None:

        if self._csv_file is not None:

            try:

                self._csv_file.close()

            except Exception:

                pass

            self._csv_file = None

            self._csv_writer = None



    def _should_use_bundled_snapshot(

        self,

        init_state: Optional[Mapping[str, float]],

        idv: Optional[Mapping[int, int]],

    ) -> bool:

        return (

            self._use_bundled_snapshot

            and self.mode == 1

            and init_state is None

            and not idv

        )



    def _load_bundled_snapshot(self) -> None:

        with DEFAULT_MODE1_SNAPSHOT.open("rb") as handle:

            snapshot = _SnapshotUnpickler(handle).load()
        self.load_snapshot(snapshot)

        # Keep the steady-state physical/controller values and controller

        # phase alignment from the saved operating point, but restart extrema

        # tracking from this bundled state.

        self._state_min.clear()

        self._state_max.clear()



    def _initialize_default_state(

        self,

        init_state: Optional[Mapping[str, float]],

        idv: Optional[Mapping[int, int]],

    ) -> None:

        if self._should_use_bundled_snapshot(init_state=init_state, idv=idv):

            self._load_bundled_snapshot()





    # ---- step ------------------------------------------------------------



    def step(

        self,

        t_step: int,

        injection_hook: Optional[InjectionHook] = None,

    ) -> Dict[str, object]:

        """One full Stage-4 cycle: controller -> 12 xmv hooks -> actuators -> physics."""

        self._next_step = int(t_step) + 1



        sensors = self.process.get_state()

        self._update_minmax(sensors)

        xmeas_arr = np.array(

            [float(sensors[k]) for k in MEASUREMENT_KEYS], dtype=np.float64

        )

        xmv_prev = np.array([float(sensors[k]) for k in MV_KEYS], dtype=np.float64)



        # 0) Setpoint-layer injection.  The hook sees the current values of

        # all exposed cascade SPs and may overwrite any subset before the

        # controller's PI loops read them.  This models attacks that

        # corrupt SP values via HMI / PLC compromise (e.g. Stuxnet-style

        # setpoint manipulation), which alarm checks based on |meas-SP|

        # cannot detect because the controller will faithfully drive meas

        # to the corrupted SP.

        sp_ctx: Dict[str, float] = {

            field: float(self.controller.setpoints[idx])

            for field, idx in SETPOINT_FIELDS.items()

        }

        sp_ctx, sp_modified = self._apply_hook(

            t_step, "controller_setpoints", sp_ctx, injection_hook

        )

        if sp_modified:

            for field, idx in SETPOINT_FIELDS.items():

                if field in sp_ctx:

                    self.controller.setpoints[idx] = float(sp_ctx[field])





        # 1) Controller cycle.

        new_xmv = self.controller.calculate(xmeas_arr, xmv_prev, int(t_step))

        new_xmv = np.asarray(new_xmv, dtype=np.float64)



        new_xmv[9] = float(np.clip(new_xmv[9], 0.0, 100.0))



        controller_xmv = {f"xmv_{i + 1:02d}": float(new_xmv[i]) for i in range(12)}



        # 2) Per-MV injection phases. Each phase shows {xmv_NN: value}; the

        # hook may overwrite the value or return None to keep it.

        setpoints_modified = bool(sp_modified)

        for i in range(12):

            phase = f"controller_xmv_{i + 1:02d}"

            key = f"xmv_{i + 1:02d}"

            ctx = {key: controller_xmv[key]}

            ctx, mod = self._apply_hook(t_step, phase, ctx, injection_hook)

            if mod:

                setpoints_modified = True

                controller_xmv[key] = float(ctx.get(key, controller_xmv[key]))



        # 3) Actuators phase: final override layer using xmv_NN_applied keys.

        actuator_ctx: Dict[str, float] = {

            f"{k}_applied": v for k, v in controller_xmv.items()

        }

        actuator_ctx, actuators_modified = self._apply_hook(

            t_step, "actuators", actuator_ctx, injection_hook

        )



        # 4) Clamp to valve range and drive the plant.

        u_applied: Dict[str, float] = {}

        for i in range(12):

            key = f"xmv_{i + 1:02d}_applied"

            v = float(actuator_ctx.get(key, controller_xmv[f"xmv_{i + 1:02d}"]))

            u_applied[key] = float(np.clip(v, 0.0, 100.0))

        new_state = self.process.step(u_applied, int(t_step))

        self._update_minmax(new_state)





        # ── Hazard margins (sign convention: margin > 0 = safe) ──────────

        reactor_pressure = float(new_state["xmeas_07"])

        reactor_temp = float(new_state["xmeas_09"])



        # Liquid levels are reported in % by xmeas, but the physical hazard

        # is defined on the volumetric holdup (vlr / vls / vlc in m^3).  We

        # read them from the bundled TEP backend so the margin column tracks

        # exactly the same quantity is_shutdown() checks.

        tp = self.process._process._teproc  # noqa: SLF001 - intentional access

        vlr_units = float(tp.vlr) / 35.3145

        vls_units = float(tp.vls) / 35.3145

        vlc_units = float(tp.vlc) / 35.3145



        h_pressure_margin = HAZARD_REACTOR_PRESSURE_MAX - reactor_pressure

        h_temp_margin = HAZARD_REACTOR_TEMP_MAX - reactor_temp

        h_reactor_lvl_hi_margin = HAZARD_REACTOR_LEVEL_MAX - vlr_units

        h_reactor_lvl_lo_margin = vlr_units - HAZARD_REACTOR_LEVEL_MIN

        h_sep_lvl_hi_margin = HAZARD_SEPARATOR_LEVEL_MAX - vls_units

        h_sep_lvl_lo_margin = vls_units - HAZARD_SEPARATOR_LEVEL_MIN

        h_strip_lvl_hi_margin = HAZARD_STRIPPER_LEVEL_MAX - vlc_units

        h_strip_lvl_lo_margin = vlc_units - HAZARD_STRIPPER_LEVEL_MIN



        # The simulator already clamps shutdown via PythonTEProcess.is_shutdown.

        h_shutdown = bool(self.process.is_shutdown())

        # Treat any margin <= 0 as a triggered hazard so Stage 4's score_trace

        # picks it up via the margin-column convention.

        all_h_margins = {

            "H-TEP-REACTOR-PRESSURE": h_pressure_margin,

            "H-TEP-REACTOR-TEMP": h_temp_margin,

            "H-TEP-REACTOR-LEVEL-HIGH": h_reactor_lvl_hi_margin,

            "H-TEP-REACTOR-LEVEL-LOW": h_reactor_lvl_lo_margin,

            "H-TEP-SEPARATOR-LEVEL-HIGH": h_sep_lvl_hi_margin,

            "H-TEP-SEPARATOR-LEVEL-LOW": h_sep_lvl_lo_margin,

            "H-TEP-STRIPPER-LEVEL-HIGH": h_strip_lvl_hi_margin,

            "H-TEP-STRIPPER-LEVEL-LOW": h_strip_lvl_lo_margin,

        }

        triggered_hazards = [hid for hid, m in all_h_margins.items() if float(m) <= 0.0]

        trip_any = bool(h_shutdown or triggered_hazards)

        if not triggered_hazards and h_shutdown:

            triggered_hazards.append("H-TEP-SHUTDOWN")

        trip_type = "multi" if len(triggered_hazards) > 1 else (

            triggered_hazards[0] if triggered_hazards else "none"

        )



        # ── Alarm margins: absolute |PV-SP| deviations on tracked control loops.

        sp = self.controller.setpoints  # numpy view

        # Indices follow DecentralizedController.setpoints layout.

        sp_d_feed = float(sp[0])      # XMEAS(2)

        sp_e_feed = float(sp[1])      # XMEAS(3)

        sp_a_feed = float(sp[2])      # XMEAS(1)

        sp_ac_feed = float(sp[3])     # XMEAS(4)

        sp_recycle_flow = float(sp[4])  # XMEAS(5)

        sp_purge_rate = float(sp[5])    # XMEAS(10)

        sp_cw = float(sp[9])            # XMEAS(21)



        meas_d_feed = float(new_state["xmeas_02"])

        meas_e_feed = float(new_state["xmeas_03"])

        meas_a_feed = float(new_state["xmeas_01"])

        meas_ac_feed = float(new_state["xmeas_04"])

        meas_recycle_flow = float(new_state["xmeas_05"])

        meas_purge_rate = float(new_state["xmeas_10"])

        meas_cw = float(new_state["xmeas_21"])



        alarm_thr_d = ALARM_DFEED_ABS

        alarm_thr_e = ALARM_EFEED_ABS

        alarm_thr_a = ALARM_AFEED_ABS

        alarm_thr_ac = ALARM_ACFEED_ABS

        alarm_thr_recycle = ALARM_RECYCLE_ABS

        alarm_thr_cw = ALARM_CW_ABS

        alarm_thr_purge = ALARM_PURGE_ABS



        alarm_margins = {

            "alarm_d_feed_track_margin": alarm_thr_d - abs(meas_d_feed - sp_d_feed),

            "alarm_e_feed_track_margin": alarm_thr_e - abs(meas_e_feed - sp_e_feed),

            "alarm_a_feed_track_margin": alarm_thr_a - abs(meas_a_feed - sp_a_feed),

            "alarm_ac_feed_track_margin": alarm_thr_ac - abs(meas_ac_feed - sp_ac_feed),

            "alarm_recycle_flow_track_margin":

                alarm_thr_recycle - abs(meas_recycle_flow - sp_recycle_flow),

            "alarm_cw_track_margin": alarm_thr_cw - abs(meas_cw - sp_cw),

            "alarm_purge_rate_track_margin":

                alarm_thr_purge - abs(meas_purge_rate - sp_purge_rate),

        }

        alarm_rule_map = {

            "alarm_d_feed_track_margin": "P-TEP-DFEED-TRACK",

            "alarm_e_feed_track_margin": "P-TEP-EFEED-TRACK",

            "alarm_a_feed_track_margin": "P-TEP-AFEED-TRACK",

            "alarm_ac_feed_track_margin": "P-TEP-ACFEED-TRACK",

            "alarm_recycle_flow_track_margin": "P-TEP-RECYCLE-FLOW-TRACK",

            "alarm_cw_track_margin": "P-TEP-CW-TRACK",

            "alarm_purge_rate_track_margin": "P-TEP-PURGE-RATE-TRACK",

        }

        triggered_alarm_ids: List[str] = []

        for col, m in alarm_margins.items():

            if float(m) <= 0.0:

                triggered_alarm_ids.append(alarm_rule_map[col])





        # ── Assemble trace record ────────────────────────────────────────

        record: Dict[str, object] = {

            "t_step": int(t_step),

            "t_time_s": float(t_step) * self.Ts,

        }

        # Full xmeas / xmv snapshot.

        for key in MEASUREMENT_KEYS:

            record[key] = float(new_state[key])

        for key in MV_KEYS:

            record[key] = float(new_state[key])

        # Applied valve commands (post-injection, post-clamp).

        for i in range(12):

            akey = f"xmv_{i + 1:02d}_applied"

            record[akey] = float(u_applied[akey])



        # Hazard fields.

        record["hazard_reactor_pressure_margin"] = float(h_pressure_margin)

        record["hazard_reactor_temp_margin"] = float(h_temp_margin)

        record["hazard_reactor_level_high_margin"] = float(h_reactor_lvl_hi_margin)

        record["hazard_reactor_level_low_margin"] = float(h_reactor_lvl_lo_margin)

        record["hazard_separator_level_high_margin"] = float(h_sep_lvl_hi_margin)

        record["hazard_separator_level_low_margin"] = float(h_sep_lvl_lo_margin)

        record["hazard_stripper_level_high_margin"] = float(h_strip_lvl_hi_margin)

        record["hazard_stripper_level_low_margin"] = float(h_strip_lvl_lo_margin)

        record["trip_any"] = bool(trip_any)

        record["trip_type"] = str(trip_type)

        record["shutdown"] = bool(h_shutdown)

        record["hazard_rule_ids"] = "|".join(triggered_hazards)



        # Alarm fields.

        for col, m in alarm_margins.items():

            record[col] = float(m)

        record["alarm_any_triggered"] = bool(triggered_alarm_ids)

        record["alarm_rule_ids"] = "|".join(triggered_alarm_ids)



        record["setpoints_modified"] = bool(setpoints_modified)

        record["actuators_modified"] = bool(actuators_modified)



        self._write_csv_row(record)

        return record





    # ---- run / reset / close --------------------------------------------



    def run(

        self,

        steps: int,

        injection_hook: Optional[InjectionHook] = None,

        return_trace: bool = True,

        stop_on_trip: bool = True,

        ignore_protection_predicates_for_stop: Optional[Sequence[str]] = None,

        ignore_protection_predicates_for_enforcement: Optional[Sequence[str]] = None,

    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:

        del ignore_protection_predicates_for_enforcement  # accepted for API symmetry

        ignored_for_stop: Set[str] = {

            str(x).strip().upper()

            for x in (ignore_protection_predicates_for_stop or [])

            if str(x).strip()

        }

        trace: List[Dict[str, object]] = []

        stop_reason = "steps_exhausted"

        stop_step: Optional[int] = None

        steps_run = 0



        for _ in range(int(steps)):

            t_step = self._next_step

            try:

                record = self.step(t_step=t_step, injection_hook=injection_hook)

            except Exception as exc:  # ANDES-style guard against numerical blowups

                stop_reason = f"runtime_error: {exc}"

                break

            steps_run += 1

            if return_trace:

                trace.append(record)

            trip = bool(record.get("trip_any", False))

            if ignored_for_stop and trip:

                # Recompute trip flag with ignored predicates removed.

                hazard_ids = {h for h in str(record.get("hazard_rule_ids", "")).split("|") if h}

                hazard_ids = {h for h in hazard_ids if h.upper() not in ignored_for_stop}

                trip = bool(hazard_ids) or bool(record.get("shutdown", False))

            if stop_on_trip and trip:

                stop_reason = "trip"

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



    def reset(self, init_state: Optional[Mapping[str, float]] = None) -> None:

        self.controller.reset(init_state=init_state)

        self.process.reset(init_state=init_state)

        self._next_step = 0

        self._state_min.clear()

        self._state_max.clear()

        self._initialize_default_state(init_state=init_state, idv=self._default_idv)



    def export_snapshot(self) -> Dict[str, object]:

        """Capture a restartable simulation snapshot.



        Includes the full physical process state, controller internal state,

        and top-level execution counters so a second simulator can continue

        from the exact same point without replaying warm-up transients.

        """

        return {

            "Ts": float(self.Ts),

            "mode": int(self.mode),

            "next_step": int(self._next_step),

            "state_min": dict(self._state_min),

            "state_max": dict(self._state_max),

            "process": self.process.export_snapshot(),

            "controller": self.controller.export_snapshot(),

        }



    def load_snapshot(self, snapshot: Mapping[str, object]) -> None:

        """Restore the simulator from export_snapshot()."""

        self.process.load_snapshot(snapshot["process"])  # type: ignore[index]

        self.controller.load_snapshot(snapshot["controller"])  # type: ignore[index]

        self._next_step = int(snapshot.get("next_step", 0))

        self._state_min = dict(snapshot.get("state_min", {}))

        self._state_max = dict(snapshot.get("state_max", {}))



    def close(self) -> None:

        self._close_csv()



    def __del__(self) -> None:  # pragma: no cover - defensive cleanup

        self._close_csv()


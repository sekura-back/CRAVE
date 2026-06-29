# TE Controller - Section 4.1

"""TEP PLC controller facade for the CRAVE Tennessee Eastman backend.


Wraps the bundled ``DecentralizedController`` (22 PI loops, identical to

temain_mod.f) and exposes the boiler-style interface:



    self.calculate(xmeas, xmv, t_step) -> new_xmv (12-vector)

    self.reset(init_state)



This is the single entry point used by ``simulation.ClosedLoopSim``.

"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional


import numpy as np



from .tep.controllers import DecentralizedController





DEFAULT_OPERATING_MODE = 1


ALARM_RULES = [
    {
        "id": "A-TEP-DFEED-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_02 - d_feed_sp| > 350.0",
        "var": "xmeas_02",
        "setpoint_var": "d_feed_sp",
        "threshold": 350.0,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-EFEED-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_03 - e_feed_sp| > 350.0",
        "var": "xmeas_03",
        "setpoint_var": "e_feed_sp",
        "threshold": 350.0,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-AFEED-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_01 - a_feed_sp| > 0.5",
        "var": "xmeas_01",
        "setpoint_var": "a_feed_sp",
        "threshold": 0.5,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-ACFEED-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_04 - ac_feed_sp| > 1.5",
        "var": "xmeas_04",
        "setpoint_var": "ac_feed_sp",
        "threshold": 1.5,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-RECYCLE-FLOW-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_05 - recycle_sp| > 1.4",
        "var": "xmeas_05",
        "setpoint_var": "recycle_sp",
        "threshold": 1.4,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-SEP-LEVEL-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_12 - separator_level_sp| > 10.0",
        "var": "xmeas_12",
        "setpoint_var": "separator_level_sp",
        "threshold": 10.0,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-STRIPPER-LEVEL-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_15 - stripper_level_sp| > 40.0",
        "var": "xmeas_15",
        "setpoint_var": "stripper_level_sp",
        "threshold": 40.0,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-REACTOR-LEVEL-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_08 - reactor_level_sp| > 14.5",
        "var": "xmeas_08",
        "setpoint_var": "reactor_level_sp",
        "threshold": 14.5,
        "component": "controller",
        "duration_s": 0.0,
    },
    {
        "id": "A-TEP-PURGE-RATE-TRACK",
        "kind": "deviation",
        "expr": "|xmeas_10 - purge_rate_sp| > 0.2",
        "var": "xmeas_10",
        "setpoint_var": "purge_rate_sp",
        "threshold": 0.2,
        "component": "controller",
        "duration_s": 0.0,
    },
]


class PLCController:
    """TEP decentralized PI controller stack."""



    def __init__(

        self,

        Ts: float = 1.0,

        *,

        mode: int = DEFAULT_OPERATING_MODE,

        init_state: Optional[Mapping[str, float]] = None,

    ):

        self.Ts = float(Ts)

        self.mode = int(mode)

        self._inner = DecentralizedController(mode=self.mode)

        self.last_output: Dict[str, float] = {}

        self.last_debug: Dict[str, Any] = {}

        if init_state is not None:

            self.last_debug["init_state_provided"] = True



    # ---- public API ------------------------------------------------------



    def reset(self, init_state: Optional[Mapping[str, float]] = None) -> None:

        self._inner.reset()

        self.last_output = {}

        self.last_debug = {}

        if init_state is not None:

            self.last_debug["init_state_provided"] = True



    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        t_step: int,
        hook: Optional[Callable[[str, Dict[str, float]], Optional[Dict[str, float]]]] = None,
    ) -> np.ndarray:
        """Run one controller cycle. Mirrors ``DecentralizedController.calculate``."""

        new_xmv = self._inner.calculate(xmeas, xmv, int(t_step), hook=hook)

        self.last_output = {f"xmv_{i + 1:02d}": float(new_xmv[i]) for i in range(12)}

        self.last_debug = dict(self._inner.last_outputs)

        return new_xmv



    @property

    def setpoints(self) -> np.ndarray:
        """Expose underlying setpoints array (Stage 5 may want to inspect)."""
        return self._inner.setpoints

    def export_snapshot(self) -> Dict[str, Any]:
        """Capture controller state needed to resume a simulation exactly."""
        ctrl_names = [
            "ctrl1", "ctrl2", "ctrl3", "ctrl4", "ctrl5", "ctrl6",
            "ctrl7", "ctrl8", "ctrl9", "ctrl10", "ctrl11",
            "ctrl13", "ctrl14", "ctrl15", "ctrl16", "ctrl17",
            "ctrl18", "ctrl19", "ctrl20", "ctrl22",
        ]
        controllers: Dict[str, Dict[str, float]] = {}
        for name in ctrl_names:
            ctrl = getattr(self._inner, name)
            controllers[name] = {
                "setpoint": float(ctrl.setpoint),
                "err_old": float(ctrl.err_old),
            }
        return {
            "mode": int(self.mode),
            "setpoints": self._inner.setpoints.copy(),
            "purge_flag": int(self._inner.purge_flag),
            "step_count": int(self._inner.step_count),
            "controllers": controllers,
        }

    def load_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore controller state from export_snapshot()."""
        self._inner.setpoints[:] = snapshot["setpoints"]
        self._inner.purge_flag = int(snapshot["purge_flag"])
        self._inner.step_count = int(snapshot["step_count"])
        for name, state in snapshot["controllers"].items():
            ctrl = getattr(self._inner, name)
            ctrl.setpoint = float(state["setpoint"])
            ctrl.err_old = float(state["err_old"])

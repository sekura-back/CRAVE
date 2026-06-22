# TE Controller - Section 4.1

"""TEP PLC controller facade for the VeriPro Tennessee Eastman backend.

Wraps the bundled ``DecentralizedController`` (22 PI loops, identical to
temain_mod.f) and exposes the boiler-style interface:

    self.calculate(xmeas, xmv, t_step) -> new_xmv (12-vector)
    self.reset(init_state)

This is the single entry point used by ``simulation.ClosedLoopSim``.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    from .tep.controllers import DecentralizedController
except ImportError:  # pragma: no cover - direct-script fallback
    from tep.controllers import DecentralizedController  # type: ignore


DEFAULT_OPERATING_MODE = 1


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
    ) -> np.ndarray:
        """Run one controller cycle. Mirrors ``DecentralizedController.calculate``."""
        new_xmv = self._inner.calculate(xmeas, xmv, int(t_step))
        self.last_output = {f"xmv_{i + 1:02d}": float(new_xmv[i]) for i in range(12)}
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

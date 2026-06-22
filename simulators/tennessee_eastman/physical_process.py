# TE Physical Process - Section 4.1

"""TEP physical process facade for the VeriPro Tennessee Eastman backend.

Exposes the same interface shape as the boiler physical-process backend:
  - get_state() -> Dict[str, float]
  - step(u_applied, t_step) -> Dict[str, float]
  - reset(init_state)

State variables (53 keys total):
  - xmeas_01..xmeas_41 (41 process measurements)
  - xmv_01..xmv_12 (current valve positions)

Control inputs (12 keys):
  - xmv_01_applied..xmv_12_applied (valve overrides 0-100%)

The underlying ``PythonTEProcess`` integrates one second per step
(dt = 1/3600 hours) using forward Euler, identical to ``TEPSimulator.step``.
"""
from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional, Sequence, Tuple

try:
    from .tep.python_backend import PythonTEProcess
except ImportError:  # pragma: no cover - direct-script fallback
    from tep.python_backend import PythonTEProcess  # type: ignore


# Measurement keys exposed by get_state(): names follow xmeas_NN with NN in 01..41.
MEASUREMENT_KEYS: Tuple[str, ...] = tuple(f"xmeas_{i:02d}" for i in range(1, 42))
# Valve / actuator keys: current position appears as both state (xmv_NN) and
# applied command (xmv_NN_applied) the way boiler exposes pump speed.
MV_KEYS: Tuple[str, ...] = tuple(f"xmv_{i:02d}" for i in range(1, 13))
APPLIED_KEYS: Tuple[str, ...] = tuple(f"{k}_applied" for k in MV_KEYS)

STATE_KEYS: Tuple[str, ...] = MEASUREMENT_KEYS + MV_KEYS
CONTROL_KEYS: Tuple[str, ...] = APPLIED_KEYS

# Steady-state baseline at mode 1 (Downs & Vogel base case). Only used as a
# fallback fill if a key happens to be missing from the live process state.
DEFAULT_NOMINAL: Dict[str, float] = {
    "xmv_01": 63.05263039,
    "xmv_02": 53.97970677,
    "xmv_03": 24.64355755,
    "xmv_04": 61.30192144,
    "xmv_05": 22.21,
    "xmv_06": 40.06374673,
    "xmv_07": 38.10034370,
    "xmv_08": 46.53415582,
    "xmv_09": 47.44573456,
    "xmv_10": 41.10581288,
    "xmv_11": 18.11349055,
    "xmv_12": 50.0,
}


class PhysicalProcess:
    """Pure-Python TEP wrapper conforming to PhysicalProcessProtocol.

    Parameters
    ----------
    Ts : float
        Wall-clock step in seconds (default 1.0). Accepts ``dt`` as a synonym.
    random_seed : Optional[int]
        Forwarded to PythonTEProcess; ``None`` keeps the Fortran-default seed.
    init_state : Optional[Mapping[str, float]]
        Optional caller-provided overlay applied on top of the steady state.
        XMV keys are applied directly. ``xmeas_08`` is also supported by
        back-projecting the requested reactor-level measurement onto the
        reactor liquid holdup state before the first integration step.
    idv : Optional[Mapping[int, int]]
        Disturbance injection schedule applied at reset, e.g. {4: 1} flips
        IDV(4) on (reactor cooling water step). Stage 4 hooks should not need
        this; it's exposed for compatibility with TEP fault-injection use.
    """

    def __init__(
        self,
        Ts: float = 1.0,
        *,
        random_seed: Optional[int] = None,
        init_state: Optional[Mapping[str, float]] = None,
        idv: Optional[Mapping[int, int]] = None,
        dt: Optional[float] = None,
    ):
        # The TEP backend uses hours internally; one Stage-4 step = Ts seconds.
        self.Ts = float(dt) if dt is not None else float(Ts)
        if self.Ts <= 0.0:
            raise ValueError("Ts must be positive")
        self._dt_hours = self.Ts / 3600.0
        self._random_seed = random_seed
        self._init_state_overlay = dict(init_state) if init_state else None
        self._idv_schedule = dict(idv) if idv else {}

        self._process = PythonTEProcess(random_seed=random_seed)
        self._process.initialize()
        self._apply_idv_schedule()
        self._apply_init_state_overlay()

        self.time_step = 0
        self._shutdown = False

    # ---- helpers ---------------------------------------------------------

    def _apply_idv_schedule(self) -> None:
        for idx, value in self._idv_schedule.items():
            self._process.set_idv(int(idx), int(value))

    def _apply_init_state_overlay(self) -> None:
        if not self._init_state_overlay:
            return
        for key, value in self._init_state_overlay.items():
            if not key.startswith("xmv_"):
                continue
            try:
                idx = int(key.split("_", 1)[1])
            except ValueError:
                continue
            if 1 <= idx <= 12:
                self._process.set_xmv(idx, float(value))

        if "xmeas_08" in self._init_state_overlay:
            self._apply_reactor_level_overlay(float(self._init_state_overlay["xmeas_08"]))

    def _apply_reactor_level_overlay(self, reactor_level_pct: float) -> None:
        """Back-project xmeas_08 onto the reactor liquid holdup state.

        xmeas_08 is derived from the reactor liquid volume:
            xmeas_08 = (vlr - 84.6) / 666.7 * 100

        We preserve the current liquid composition and specific energy, then
        rescale reactor liquid component holdups (yy[3:8]) and reactor total
        energy (yy[8]) so the initialized measurement matches the requested
        reactor level before the first simulation step.
        """
        yy = self._process.get_state()
        self._process.evaluate(0.0, yy)
        tp = self._process._teproc  # noqa: SLF001 - intentional internal access

        current_vlr = float(tp.vlr)
        if current_vlr <= 0.0:
            return

        target_vlr = 84.6 + float(reactor_level_pct) / 100.0 * 666.7
        if target_vlr <= 0.0:
            return

        scale = target_vlr / current_vlr
        yy = yy.copy()
        yy[3:8] *= scale
        yy[8] *= scale
        self._process.set_state(yy)
        self._process.evaluate(0.0, yy)

    # ---- public API ------------------------------------------------------

    def get_state(self) -> Dict[str, float]:
        """Return 41 xmeas + 12 xmv as a flat float dict."""
        xmeas = self._process.get_xmeas()
        xmv = self._process.get_xmv()
        out: Dict[str, float] = {}
        for i, key in enumerate(MEASUREMENT_KEYS):
            out[key] = float(xmeas[i])
        for i, key in enumerate(MV_KEYS):
            out[key] = float(xmv[i])
        return out

    def step(self, u_applied: Mapping[str, float], t_step: int) -> Dict[str, float]:
        """Advance one Euler step under forced valve commands."""
        del t_step  # not used; matches boiler interface
        if self._shutdown:
            # Match boiler facade: once shutdown, freeze and surface state.
            return self.get_state()

        for i, key in enumerate(MV_KEYS):
            applied_key = f"{key}_applied"
            if applied_key in u_applied:
                self._process.set_xmv(i + 1, float(u_applied[applied_key]))
            elif key in u_applied:
                # Allow callers to pass plain xmv_NN as well.
                self._process.set_xmv(i + 1, float(u_applied[key]))

        self._process.step(dt=self._dt_hours)
        self.time_step += 1
        if self._process.is_shutdown():
            self._shutdown = True
        return self.get_state()

    def export_snapshot(self) -> Dict[str, object]:
        """Capture a restartable snapshot of the full physical process state."""
        return {
            "Ts": float(self.Ts),
            "dt_hours": float(self._dt_hours),
            "time_step": int(self.time_step),
            "shutdown": bool(self._shutdown),
            "random_seed": self._random_seed,
            "init_state_overlay": dict(self._init_state_overlay) if self._init_state_overlay else None,
            "idv_schedule": dict(self._idv_schedule),
            "yy": self._process.yy.copy(),
            "yp": self._process.yp.copy(),
            "time": float(self._process.time),
            "g": float(self._process._g),  # noqa: SLF001 - internal RNG state
            "initialized": bool(self._process._initialized),  # noqa: SLF001
            "shutdown_backend": bool(self._process._shutdown),  # noqa: SLF001
            "xmeas": self._process._xmeas.copy(),  # noqa: SLF001
            "xmv": self._process._xmv.copy(),  # noqa: SLF001
            "idv": self._process._idv.copy(),  # noqa: SLF001
            "teproc": copy.deepcopy(self._process._teproc),  # noqa: SLF001
            "wlk": copy.deepcopy(self._process._wlk),  # noqa: SLF001
        }

    def load_snapshot(self, snapshot: Mapping[str, object]) -> None:
        """Restore the physical process from a prior export_snapshot()."""
        self._dt_hours = float(snapshot.get("dt_hours", self._dt_hours))
        self.time_step = int(snapshot.get("time_step", 0))
        self._shutdown = bool(snapshot.get("shutdown", False))
        self._init_state_overlay = dict(snapshot["init_state_overlay"]) if snapshot.get("init_state_overlay") else None
        self._idv_schedule = dict(snapshot.get("idv_schedule", {}))

        self._process.yy[:] = snapshot["yy"]  # type: ignore[index]
        self._process.yp[:] = snapshot["yp"]  # type: ignore[index]
        self._process.time = float(snapshot["time"])  # type: ignore[index]
        self._process._g = float(snapshot["g"])  # type: ignore[index]  # noqa: SLF001
        self._process._initialized = bool(snapshot["initialized"])  # type: ignore[index]  # noqa: SLF001
        self._process._shutdown = bool(snapshot["shutdown_backend"])  # type: ignore[index]  # noqa: SLF001
        self._process._xmeas[:] = snapshot["xmeas"]  # type: ignore[index]  # noqa: SLF001
        self._process._xmv[:] = snapshot["xmv"]  # type: ignore[index]  # noqa: SLF001
        self._process._idv[:] = snapshot["idv"]  # type: ignore[index]  # noqa: SLF001
        self._process._teproc = copy.deepcopy(snapshot["teproc"])  # type: ignore[index]  # noqa: SLF001
        self._process._wlk = copy.deepcopy(snapshot["wlk"])  # type: ignore[index]  # noqa: SLF001

    def is_shutdown(self) -> bool:
        return bool(self._shutdown or self._process.is_shutdown())

    def reset(self, init_state: Optional[Mapping[str, float]] = None) -> None:
        """Reset to steady state. Re-applies the disturbance schedule."""
        self._process = PythonTEProcess(random_seed=self._random_seed)
        self._process.initialize()
        self._apply_idv_schedule()
        if init_state is not None:
            self._init_state_overlay = dict(init_state)
        self._apply_init_state_overlay()
        self.time_step = 0
        self._shutdown = False

# Boiler Physical Process - Section 4.1

"""Simple gray-box physical process model for closed-loop simulation.

This model keeps the existing two-input / six-state interface but replaces the
mostly independent empirical recursions with weakly coupled first-order
dynamics around the current nominal operating point.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Mapping, Optional


STATE_KEYS = (
    "true_load",
    "coal_flow",
    "feedwater_flow",
    "main_steam_flow",
    "waterwall_temp",
    "main_steam_pressure",
)

DEFAULT_NOMINAL = {
    "true_load": 349.0,
    "coal_flow": 157.4,
    "feedwater_flow": 1065.7,
    "main_steam_flow": 1064.0,
    "waterwall_temp": 425.0,
    "main_steam_pressure": 23.7,
    "fuel_command": 157.6,
    "water_pump_speed": 2500.0,
}

DEFAULT_PARAMS = {
    "tau_coal": 2.0,
    "tau_feedwater": 0.3,
    "tau_steam_flow": 3.8,
    "tau_load": 4.5,
    "pump_to_feedwater_gain": 0.8,
    "coal_to_steam_gain": 3.5,
    "feedwater_to_steam_gain": 0.11,
    "pressure_restore": 0.28,
    "pressure_from_coal": 0.048,
    "pressure_from_steam": 0.0072,
    "pressure_from_feedwater": 0.02,
    "temp_restore": 0.10,
    "temp_from_coal": 0.104,
    "temp_from_feedwater": 0.0034,
    "temp_from_steam": 0.003,
    "load_from_steam": 0.24,
    "load_from_pressure": 6.2,
}


def physical_step_formula(
    state: Mapping[str, float],
    fuel_command: float,
    water_pump_speed: float,
    dt: float = 0.1,
) -> Dict[str, float]:
    """Pure one-step physical formula matching PhysicalProcess.step.

    Source: src/close_loop/pyhsicalprocess.py::PhysicalProcess.step
    This function is side-effect-free and does not depend on controller or
    PhysicalProcess instance state/history.
    """
    p = DEFAULT_PARAMS
    n = DEFAULT_NOMINAL
    ts = float(dt)

    coal_flow = float(state["coal_flow"])
    feedwater_flow = float(state["feedwater_flow"])
    main_steam_flow = float(state["main_steam_flow"])
    waterwall_temp = float(state["waterwall_temp"])
    main_steam_pressure = float(state["main_steam_pressure"])
    true_load = float(state["true_load"])

    fuel_input = float(fuel_command)
    pump_input = float(water_pump_speed)

    coal_dot = (fuel_input - coal_flow) / p["tau_coal"]

    feedwater_target = (
        n["feedwater_flow"]
        + p["pump_to_feedwater_gain"] * (pump_input - n["water_pump_speed"])
    )
    feedwater_dot = (feedwater_target - feedwater_flow) / p["tau_feedwater"]

    steam_target = (
        n["main_steam_flow"]
        + p["coal_to_steam_gain"] * (coal_flow - n["coal_flow"])
        + p["feedwater_to_steam_gain"] * (feedwater_flow - n["feedwater_flow"])
    )
    steam_dot = (steam_target - main_steam_flow) / p["tau_steam_flow"]

    pressure_dot = (
        -p["pressure_restore"] * (main_steam_pressure - n["main_steam_pressure"])
        + p["pressure_from_coal"] * (coal_flow - n["coal_flow"])
        + p["pressure_from_feedwater"] * (feedwater_flow - n["feedwater_flow"])
        - p["pressure_from_steam"] * (main_steam_flow - n["main_steam_flow"])
    )

    temp_dot = (
        -p["temp_restore"] * (waterwall_temp - n["waterwall_temp"])
        + p["temp_from_coal"] * (coal_flow - n["coal_flow"])
        - p["temp_from_feedwater"] * (feedwater_flow - n["feedwater_flow"])
        - p["temp_from_steam"] * (main_steam_flow - n["main_steam_flow"])
    )

    load_target = (
        n["true_load"]
        + p["load_from_steam"] * (main_steam_flow - n["main_steam_flow"])
        + p["load_from_pressure"] * (main_steam_pressure - n["main_steam_pressure"])
    )
    load_dot = (load_target - true_load) / p["tau_load"]

    return {
        "coal_flow": max(0.0, coal_flow + ts * coal_dot),
        "feedwater_flow": max(0.0, feedwater_flow + ts * feedwater_dot),
        "main_steam_flow": max(0.0, main_steam_flow + ts * steam_dot),
        "waterwall_temp": max(0.0, waterwall_temp + ts * temp_dot),
        "main_steam_pressure": max(0.0, main_steam_pressure + ts * pressure_dot),
        "true_load": max(0.0, true_load + ts * load_dot),
    }


class PhysicalProcess:
    """Unified physical process facade."""

    def __init__(self, Ts: float = 0.1, init_state: Optional[Mapping[str, float]] = None):
        self.Ts = float(Ts)
        self.time_step = 0
        self.history_len = 64
        self.outputs: Dict[str, Deque[float]] = {
            key: deque(maxlen=self.history_len) for key in STATE_KEYS
        }
        self.nominal = dict(DEFAULT_NOMINAL)
        self.params = dict(DEFAULT_PARAMS)
        self.reset(init_state=init_state)

    def _default_state(self) -> Dict[str, float]:
        return {
            "true_load": self.nominal["true_load"],
            "coal_flow": self.nominal["coal_flow"],
            "feedwater_flow": self.nominal["feedwater_flow"],
            "main_steam_flow": self.nominal["main_steam_flow"],
            "waterwall_temp": self.nominal["waterwall_temp"],
            "main_steam_pressure": self.nominal["main_steam_pressure"],
        }

    def reset(self, init_state: Optional[Mapping[str, float]] = None) -> None:
        """Reset process state histories."""
        state = self._default_state()
        if init_state:
            for key in STATE_KEYS:
                if key in init_state:
                    state[key] = float(init_state[key])

        self.time_step = 0
        for key in STATE_KEYS:
            self.outputs[key].clear()
            for _ in range(self.history_len):
                self.outputs[key].append(float(state[key]))

    def get_state(self) -> Dict[str, float]:
        return {key: float(self.outputs[key][-1]) for key in STATE_KEYS}

    def step(self, u_applied: Mapping[str, float], t_step: int) -> Dict[str, float]:
        """Advance one simulation step.

        Required input keys:
        - fuel_command_0_applied
        - water_pump_speed_0_applied
        """
        del t_step
        if "fuel_command_0_applied" not in u_applied:
            raise KeyError("u_applied missing fuel_command_0_applied")
        if "water_pump_speed_0_applied" not in u_applied:
            raise KeyError("u_applied missing water_pump_speed_0_applied")

        self.time_step += 1
        fuel_input = float(u_applied["fuel_command_0_applied"])
        pump_input = float(u_applied["water_pump_speed_0_applied"])

        state = self.get_state()
        next_state = physical_step_formula(
            state=state,
            fuel_command=fuel_input,
            water_pump_speed=pump_input,
            dt=self.Ts,
        )

        for key in STATE_KEYS:
            self.outputs[key].append(float(next_state[key]))

        return self.get_state()

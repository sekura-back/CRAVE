"""
Process controllers for the Tennessee Eastman Process.

This module now consolidates:
1. Controller base classes and registry utilities
2. Core decentralized/manual controllers
3. Example controller plugins
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from .constants import (
    DEFAULT_OPERATING_MODE,
    NUM_MANIPULATED_VARS,
    OPERATING_MODES,
)


ControllerHook = Optional[Callable[[str, Dict[str, float]], Optional[Dict[str, float]]]]


def _override_output(
    hook: ControllerHook,
    *,
    phase: str,
    field: str,
    value: float,
) -> float:
    if hook is None:
        return float(value)
    overrides = hook(str(phase), {field: float(value)})
    if not overrides:
        return float(value)
    if not isinstance(overrides, dict):
        raise TypeError("controller hook must return dict or None")
    return float(overrides.get(field, value))


class BaseController(ABC):
    """Abstract base class for all TEP controllers."""

    name: str = "base"
    description: str = "Base controller class"
    version: str = "1.0.0"
    controlled_mvs: Optional[List[int]] = None

    @abstractmethod
    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """Calculate new manipulated variable values."""
        pass

    @abstractmethod
    def reset(self):
        """Reset controller state to initial conditions."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get controller metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "controlled_mvs": self.controlled_mvs,
        }

    def set_parameter(self, name: str, value: Any):
        """Set a controller attribute if available."""
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise AttributeError(f"Unknown parameter: {name}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get tunable parameters (override in subclasses)."""
        return {}


@dataclass
class ControllerConfig:
    """Configuration for a registered controller."""

    controller_class: Type[BaseController]
    name: str
    description: str
    default_params: Dict[str, Any] = field(default_factory=dict)


class ControllerRegistry:
    """Registry for controller plugins."""

    _controllers: Dict[str, ControllerConfig] = {}

    @classmethod
    def register(
        cls,
        controller_class: Type[BaseController],
        name: str = None,
        description: str = None,
        default_params: Dict[str, Any] = None,
    ):
        if not issubclass(controller_class, BaseController):
            raise TypeError(
                f"Controller must inherit from BaseController, got {controller_class.__bases__}"
            )

        reg_name = name or controller_class.name
        reg_desc = description or controller_class.description
        cls._controllers[reg_name] = ControllerConfig(
            controller_class=controller_class,
            name=reg_name,
            description=reg_desc,
            default_params=default_params or {},
        )

    @classmethod
    def unregister(cls, name: str):
        if name in cls._controllers:
            del cls._controllers[name]

    @classmethod
    def get(cls, name: str) -> Type[BaseController]:
        if name not in cls._controllers:
            available = ", ".join(cls._controllers.keys())
            raise KeyError(
                f"Controller '{name}' not found. Available: {available}"
            )
        return cls._controllers[name].controller_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseController:
        config = cls._controllers.get(name)
        if config is None:
            available = ", ".join(cls._controllers.keys())
            raise KeyError(
                f"Controller '{name}' not found. Available: {available}"
            )
        params = {**config.default_params, **kwargs}
        return config.controller_class(**params)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._controllers.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        if name not in cls._controllers:
            raise KeyError(f"Controller '{name}' not found")
        config = cls._controllers[name]
        return {
            "name": config.name,
            "description": config.description,
            "class": config.controller_class.__name__,
            "default_params": config.default_params,
        }

    @classmethod
    def list_all_info(cls) -> List[Dict[str, Any]]:
        return [cls.get_info(name) for name in cls._controllers]

    @classmethod
    def clear(cls):
        cls._controllers.clear()


def register_controller(
    name: str = None,
    description: str = None,
    default_params: Dict[str, Any] = None,
):
    """Decorator to register a controller class."""

    def decorator(cls: Type[BaseController]) -> Type[BaseController]:
        ControllerRegistry.register(
            cls,
            name=name,
            description=description,
            default_params=default_params,
        )
        return cls

    return decorator


class CompositeController(BaseController):
    """Controller that combines multiple sub-controllers."""

    name = "composite"
    description = "Composite controller combining multiple sub-controllers"

    def __init__(self, fallback_controller: BaseController = None):
        self._sub_controllers: List[tuple] = []
        self._fallback = fallback_controller
        self._mv_assignment: Dict[int, BaseController] = {}

    def add_controller(
        self,
        controller: BaseController,
        mvs: List[int],
    ):
        mv_indices = [mv - 1 if mv >= 1 else mv for mv in mvs]

        for idx in mv_indices:
            if idx in self._mv_assignment:
                existing = self._mv_assignment[idx]
                raise ValueError(f"MV {idx + 1} already assigned to {existing.name}")

        self._sub_controllers.append((controller, mv_indices))
        for idx in mv_indices:
            self._mv_assignment[idx] = controller

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int,
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        for controller, mv_indices in self._sub_controllers:
            sub_xmv = controller.calculate(xmeas, xmv_new, step)
            for idx in mv_indices:
                xmv_new[idx] = sub_xmv[idx]

        if self._fallback is not None:
            fallback_xmv = self._fallback.calculate(xmeas, xmv_new, step)
            for i in range(NUM_MANIPULATED_VARS):
                if i not in self._mv_assignment:
                    xmv_new[i] = fallback_xmv[i]

        return xmv_new

    def reset(self):
        for controller, _ in self._sub_controllers:
            controller.reset()
        if self._fallback is not None:
            self._fallback.reset()

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["sub_controllers"] = [
            {"name": ctrl.name, "mvs": [i + 1 for i in mvs]}
            for ctrl, mvs in self._sub_controllers
        ]
        if self._fallback:
            info["fallback"] = self._fallback.name
        return info


class ControllerType(Enum):
    """Types of controllers available."""
    P = "proportional"
    PI = "proportional-integral"
    PID = "proportional-integral-derivative"


@dataclass
class PIController:
    """
    Proportional-Integral controller using velocity form.

    The velocity form calculates the change in output rather than
    the absolute output, which prevents integral windup.
    """
    setpoint: float
    gain: float
    taui: float = 0.0  # Reset time (hours). 0 = P-only control
    err_old: float = 0.0
    output_min: float = 0.0
    output_max: float = 100.0
    scale: float = 1.0  # Scaling factor for error normalization

    def calculate(
        self,
        measurement: float,
        current_output: float,
        dt: float
    ) -> float:
        """
        Calculate new controller output.

        Args:
            measurement: Current process measurement
            current_output: Current valve position (%)
            dt: Time step since last calculation (hours)

        Returns:
            New valve position (%)
        """
        # Calculate error (normalized)
        error = (self.setpoint - measurement) * self.scale

        # P-only or PI control
        if self.taui > 0:
            # PI controller (velocity form)
            dxmv = self.gain * ((error - self.err_old) + error * dt / self.taui)
        else:
            # P-only controller
            dxmv = self.gain * (error - self.err_old)

        # Update output
        new_output = current_output + dxmv

        # Apply limits
        new_output = np.clip(new_output, self.output_min, self.output_max)

        # Store error for next iteration
        self.err_old = error

        return new_output

    def calculate_change(
        self,
        measurement: float,
        dt: float
    ) -> float:
        """
        Calculate controller output change (dxmv) for cascade control.

        This method returns just the change in output (velocity form)
        without applying limits, for use in cascade controllers where
        the output adjusts a setpoint rather than a valve position.

        Args:
            measurement: Current process measurement
            dt: Time step since last calculation (hours)

        Returns:
            Change in output (dxmv)
        """
        # Calculate error (normalized)
        error = (self.setpoint - measurement) * self.scale

        # P-only or PI control
        if self.taui > 0:
            # PI controller (velocity form)
            dxmv = self.gain * ((error - self.err_old) + error * dt / self.taui)
        else:
            # P-only controller
            dxmv = self.gain * (error - self.err_old)

        # Store error for next iteration
        self.err_old = error

        return dxmv

    def reset(self):
        """Reset controller state."""
        self.err_old = 0.0


@dataclass
class CascadeController:
    """
    Cascade controller where primary controller adjusts secondary setpoint.

    Used for composition control loops in TEP.
    """
    primary: PIController
    secondary_setpoint_index: int  # Which setpoint to adjust
    secondary_scale: float = 1.0  # Scaling between primary output and secondary setpoint


@register_controller(
    name="decentralized",
    description="Decentralized multi-loop PI control system (18+ loops)"
)
class DecentralizedController(BaseController):
    """
    Decentralized multi-loop control system for TEP.

    Implements the control scheme from temain_mod.f with 18+ loops.
    """

    name = "decentralized"
    description = "Decentralized multi-loop PI control system (18+ loops)"
    version = "1.0.0"
    controlled_mvs = list(range(1, 12))  # Controls MVs 1-11

    def __init__(self, mode: int = DEFAULT_OPERATING_MODE):
        """
        Initialize the decentralized controller.

        Args:
            mode: Operating mode (1-6). Default is mode 1 (50/50 G/H, base rate).
        """
        # Store operating mode
        self._mode = mode

        # Initialize all controller loops
        self._init_controllers()

        # Control timing
        self.dt = 1.0 / 3600.0  # 1 second in hours
        self.step_count = 0

        # Setpoints array (for cascade controllers)
        self.setpoints = np.zeros(20)
        self._init_setpoints()

        # Purge valve flag for override control
        self.purge_flag = 0
        self.last_outputs: Dict[str, float] = {}

    def _init_setpoints(self):
        """Initialize setpoint values based on operating mode."""
        # Get mode configuration
        mode_config = OPERATING_MODES.get(self._mode)

        if mode_config is not None:
            # Use optimal setpoints from Ricker (1995)
            sp = mode_config.xmeas_setpoints
            self.setpoints[0] = sp.get(1, 3664.0)     # D Feed flow (XMEAS 2)
            self.setpoints[1] = sp.get(2, 4509.3)     # E Feed flow (XMEAS 3)
            self.setpoints[2] = sp.get(0, 0.25052)    # A Feed flow (XMEAS 1)
            self.setpoints[3] = sp.get(3, 9.3477)     # A+C Feed flow (XMEAS 4)
            self.setpoints[4] = 26.902                 # Recycle flow (not in mode spec)
            self.setpoints[5] = sp.get(9, 0.33712)    # Purge rate (XMEAS 10)
            self.setpoints[6] = sp.get(11, 50.0)      # Separator level (XMEAS 12)
            self.setpoints[7] = sp.get(14, 50.0)      # Stripper level (XMEAS 15)
            self.setpoints[8] = 230.31                 # Steam flow (varies by mode, keep default)
            self.setpoints[9] = 94.599                 # Reactor CW outlet temp
            self.setpoints[10] = 22.949                # Stripper underflow
            self.setpoints[11] = 2633.7                # Separator pressure
            self.setpoints[12] = 32.188                # Reactor feed A composition
            self.setpoints[13] = 6.8820                # Reactor feed D composition
            self.setpoints[14] = 18.776                # Reactor feed E composition
            self.setpoints[15] = sp.get(17, 65.731)   # Stripper temperature (XMEAS 18)
            self.setpoints[16] = sp.get(7, 75.000)    # Reactor level (XMEAS 8)
            self.setpoints[17] = sp.get(8, 120.40)    # Reactor temperature (XMEAS 9)
            self.setpoints[18] = 13.823                # Purge B composition
            self.setpoints[19] = 0.83570               # Product E composition
        else:
            # Default setpoints (Downs & Vogel base case)
            self.setpoints[0] = 3664.0    # D Feed flow
            self.setpoints[1] = 4509.3    # E Feed flow
            self.setpoints[2] = 0.25052   # A Feed flow
            self.setpoints[3] = 9.3477    # A+C Feed flow
            self.setpoints[4] = 26.902    # Recycle flow
            self.setpoints[5] = 0.33712   # Purge rate
            self.setpoints[6] = 50.0      # Separator level
            self.setpoints[7] = 50.0      # Stripper level
            self.setpoints[8] = 230.31    # Steam flow
            self.setpoints[9] = 94.599    # Reactor CW outlet temp
            self.setpoints[10] = 22.949   # Stripper underflow
            self.setpoints[11] = 2633.7   # Separator pressure
            self.setpoints[12] = 32.188   # Reactor feed A composition
            self.setpoints[13] = 6.8820   # Reactor feed D composition
            self.setpoints[14] = 18.776   # Reactor feed E composition
            self.setpoints[15] = 65.731   # Stripper temperature
            self.setpoints[16] = 75.000   # Reactor level
            self.setpoints[17] = 120.40   # Reactor temperature
            self.setpoints[18] = 13.823   # Purge B composition
            self.setpoints[19] = 0.83570  # Product E composition

    @property
    def mode(self) -> int:
        """Get current operating mode."""
        return self._mode

    def set_mode(self, mode: int):
        """
        Change the operating mode.

        This updates the controller setpoints to match the new mode.
        The process will transition to the new operating point over time.

        Args:
            mode: Operating mode (1-6)
                1: 50/50 G/H, Base Rate
                2: 10/90 G/H, Base Rate
                3: 90/10 G/H, Base Rate
                4: 50/50 G/H, Max Rate
                5: 10/90 G/H, Max Rate
                6: 90/10 G/H, Max Rate
        """
        if mode not in OPERATING_MODES:
            raise ValueError(f"Invalid mode {mode}. Must be 1-6.")

        self._mode = mode
        self._init_setpoints()

        # Reset controller error states for smooth transition
        for ctrl in [self.ctrl1, self.ctrl2, self.ctrl3, self.ctrl4, self.ctrl5,
                     self.ctrl6, self.ctrl7, self.ctrl8, self.ctrl9, self.ctrl10,
                     self.ctrl11, self.ctrl13, self.ctrl14, self.ctrl15, self.ctrl16,
                     self.ctrl17, self.ctrl18, self.ctrl19, self.ctrl20, self.ctrl22]:
            ctrl.err_old = 0.0

    def get_mode_info(self) -> dict:
        """Get information about the current operating mode."""
        mode_config = OPERATING_MODES.get(self._mode)
        if mode_config:
            return {
                "mode": self._mode,
                "name": mode_config.name,
                "g_h_ratio": mode_config.g_h_ratio,
                "production": mode_config.production,
            }
        return {"mode": self._mode, "name": "Unknown"}

    def _init_controllers(self):
        """Initialize all PI controllers."""
        # Controller 1: D Feed Flow (XMEAS 2 -> XMV 1)
        self.ctrl1 = PIController(
            setpoint=3664.0, gain=1.0, taui=0.0,
            scale=100.0/5811.0
        )

        # Controller 2: E Feed Flow (XMEAS 3 -> XMV 2)
        self.ctrl2 = PIController(
            setpoint=4509.3, gain=1.0, taui=0.0,
            scale=100.0/8354.0
        )

        # Controller 3: A Feed Flow (XMEAS 1 -> XMV 3)
        self.ctrl3 = PIController(
            setpoint=0.25052, gain=1.0, taui=0.0,
            scale=100.0/1.017
        )

        # Controller 4: A+C Feed Flow (XMEAS 4 -> XMV 4)
        self.ctrl4 = PIController(
            setpoint=9.3477, gain=1.0, taui=0.0,
            scale=100.0/15.25
        )

        # Controller 5: Recycle Flow (XMEAS 5 -> XMV 5)
        self.ctrl5 = PIController(
            setpoint=26.902, gain=-0.083, taui=1.0/3600.0,
            scale=100.0/53.0
        )

        # Controller 6: Purge Rate (XMEAS 10 -> XMV 6)
        self.ctrl6 = PIController(
            setpoint=0.33712, gain=1.22, taui=0.0,
            scale=100.0/1.0
        )

        # Controller 7: Separator Level (XMEAS 12 -> XMV 7)
        self.ctrl7 = PIController(
            setpoint=50.0, gain=-2.06, taui=0.0,
            scale=100.0/70.0
        )

        # Controller 8: Stripper Level (XMEAS 15 -> XMV 8)
        self.ctrl8 = PIController(
            setpoint=50.0, gain=-1.62, taui=0.0,
            scale=100.0/70.0
        )

        # Controller 9: Steam Flow (XMEAS 19 -> XMV 9)
        self.ctrl9 = PIController(
            setpoint=230.31, gain=0.41, taui=0.0,
            scale=100.0/460.0
        )

        # Controller 10: Reactor CW Temp (XMEAS 21 -> XMV 10)
        self.ctrl10 = PIController(
            setpoint=94.599, gain=-0.156*10, taui=1452.0/3600.0,
            scale=100.0/150.0
        )

        # Controller 11: Stripper Underflow (XMEAS 17 -> XMV 11)
        self.ctrl11 = PIController(
            setpoint=22.949, gain=1.09, taui=2600.0/3600.0,
            scale=100.0/46.0
        )

        # Controller 13: Reactor Feed A (cascade to setpoint 3)
        self.ctrl13 = PIController(
            setpoint=32.188, gain=18.0, taui=3168.0/3600.0,
            scale=100.0/100.0
        )

        # Controller 14: Reactor Feed D (cascade to setpoint 1)
        self.ctrl14 = PIController(
            setpoint=6.8820, gain=8.3, taui=3168.0/3600.0,
            scale=100.0/100.0
        )

        # Controller 15: Reactor Feed E (cascade to setpoint 2)
        self.ctrl15 = PIController(
            setpoint=18.776, gain=2.37, taui=5069.0/3600.0,
            scale=100.0/100.0
        )

        # Controller 16: Stripper Temp (cascade to setpoint 9)
        self.ctrl16 = PIController(
            setpoint=65.731, gain=1.69/10.0, taui=236.0/3600.0,
            scale=100.0/130.0
        )

        # Controller 17: Reactor Level (cascade to setpoint 4)
        self.ctrl17 = PIController(
            setpoint=75.000, gain=11.1/10.0, taui=3168.0/3600.0,
            scale=100.0/50.0
        )

        # Controller 18: Reactor Temp (cascade to setpoint 10)
        self.ctrl18 = PIController(
            setpoint=120.40, gain=2.83*10.0, taui=982.0/3600.0,
            scale=100.0/150.0
        )

        # Controller 19: Purge B (cascade to setpoint 6)
        self.ctrl19 = PIController(
            setpoint=13.823, gain=-83.2/5.0/3.0, taui=6336.0/3600.0,
            scale=100.0/26.0
        )

        # Controller 20: Product E (cascade to setpoint 16)
        self.ctrl20 = PIController(
            setpoint=0.83570, gain=-16.3/5.0, taui=12408.0/3600.0,
            scale=100.0/1.6
        )

        # Controller 22: Separator Pressure (backup for purge)
        self.ctrl22 = PIController(
            setpoint=2633.7, gain=-1.0*5.0, taui=1000.0/3600.0,
            scale=1.0
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        time_step: int,
        hook: ControllerHook = None,
    ) -> np.ndarray:
        """
        Calculate new manipulated variable values.

        Args:
            xmeas: Current measurements (41 elements, 0-indexed)
            xmv: Current manipulated variables (12 elements, 0-indexed)
            time_step: Current simulation step number

        Returns:
            Updated manipulated variables
        """
        xmv_new = xmv.copy()
        dt3 = 3.0 / 3600.0  # 3 seconds in hours
        dt900 = 900.0 / 3600.0  # 900 seconds in hours

        def emit_output(phase: str, field: str, value: float) -> float:
            overridden = _override_output(hook, phase=phase, field=field, value=value)
            self.last_outputs[field] = float(overridden)
            return float(overridden)

        # Main direct loops + cascades (every 3 seconds)
        if time_step % 3 == 0:
            # Update setpoints for cascade controllers
            self.ctrl1.setpoint = self.setpoints[0]
            self.ctrl2.setpoint = self.setpoints[1]
            self.ctrl3.setpoint = self.setpoints[2]
            self.ctrl4.setpoint = self.setpoints[3]
            self.ctrl9.setpoint = self.setpoints[8]

            # Controller 1: D Feed Flow
            xmv_new[0] = self.ctrl1.calculate(xmeas[1], xmv_new[0], dt3)

            # Controller 2: E Feed Flow
            xmv_new[1] = self.ctrl2.calculate(xmeas[2], xmv_new[1], dt3)

            # Controller 3: A Feed Flow
            xmv_new[2] = self.ctrl3.calculate(xmeas[0], xmv_new[2], dt3)

            # Controller 4: A+C Feed Flow
            xmv_new[3] = self.ctrl4.calculate(xmeas[3], xmv_new[3], dt3)

            # Controller 5: Recycle Flow
            xmv_new[4] = self.ctrl5.calculate(xmeas[4], xmv_new[4], dt3)

            # Controller 6: Purge Rate (with override logic)
            xmv_new[5] = self._purge_control(xmeas, xmv_new[5], dt3, hook=hook)

            # Controller 7: Separator Level
            xmv_new[6] = self.ctrl7.calculate(xmeas[11], xmv_new[6], dt3)

            # Controller 8: Stripper Level
            xmv_new[7] = self.ctrl8.calculate(xmeas[14], xmv_new[7], dt3)

            # Controller 9: Steam Flow
            xmv_new[8] = self.ctrl9.calculate(xmeas[18], xmv_new[8], dt3)

            # Controller 10: Reactor CW Temp
            xmv_new[9] = self.ctrl10.calculate(xmeas[20], xmv_new[9], dt3)

            # Controller 11: Stripper Underflow
            xmv_new[10] = self.ctrl11.calculate(xmeas[16], xmv_new[10], dt3)

            # Controller 16: Stripper Temp -> Steam setpoint (cascade)
            dxmv = self.ctrl16.calculate_change(xmeas[17], dt3)
            dxmv = emit_output(
                "controller_ctrl16_primary_output",
                "ctrl16_primary_output",
                dxmv,
            )
            self.setpoints[8] = emit_output(
                "controller_ctrl16_scaled_output",
                "ctrl16_scaled_output",
                self.setpoints[8] + dxmv * 460.0 / 100.0,
            )

            # Controller 17: Reactor Level -> A+C Feed setpoint (cascade)
            dxmv = self.ctrl17.calculate_change(xmeas[7], dt3)
            dxmv = emit_output(
                "controller_ctrl17_primary_output",
                "ctrl17_primary_output",
                dxmv,
            )
            self.setpoints[3] = emit_output(
                "controller_ctrl17_scaled_output",
                "ctrl17_scaled_output",
                self.setpoints[3] + dxmv * 15.25 / 100.0,
            )

            # Controller 18: Reactor Temp -> Reactor CW setpoint (cascade)
            dxmv = self.ctrl18.calculate_change(xmeas[8], dt3)
            dxmv = emit_output(
                "controller_ctrl18_primary_output",
                "ctrl18_primary_output",
                dxmv,
            )
            self.setpoints[9] = emit_output(
                "controller_ctrl18_scaled_output",
                "ctrl18_scaled_output",
                self.setpoints[9] + dxmv * 150.0 / 100.0,
            )
            self.ctrl10.setpoint = self.setpoints[9]

            # Controller 13: Reactor Feed A -> A Feed setpoint (cascade)
            dxmv = self.ctrl13.calculate_change(xmeas[22], dt3)
            dxmv = emit_output(
                "controller_ctrl13_primary_output",
                "ctrl13_primary_output",
                dxmv,
            )
            self.setpoints[2] = emit_output(
                "controller_ctrl13_scaled_output",
                "ctrl13_scaled_output",
                self.setpoints[2] + dxmv * 1.017 / 100.0,
            )

            # Controller 14: Reactor Feed D -> D Feed setpoint (cascade)
            dxmv = self.ctrl14.calculate_change(xmeas[25], dt3)
            dxmv = emit_output(
                "controller_ctrl14_primary_output",
                "ctrl14_primary_output",
                dxmv,
            )
            self.setpoints[0] = emit_output(
                "controller_ctrl14_scaled_output",
                "ctrl14_scaled_output",
                self.setpoints[0] + dxmv * 5811.0 / 100.0,
            )

            # Controller 15: Reactor Feed E -> E Feed setpoint (cascade)
            dxmv = self.ctrl15.calculate_change(xmeas[26], dt3)
            dxmv = emit_output(
                "controller_ctrl15_primary_output",
                "ctrl15_primary_output",
                dxmv,
            )
            self.setpoints[1] = emit_output(
                "controller_ctrl15_scaled_output",
                "ctrl15_scaled_output",
                self.setpoints[1] + dxmv * 8354.0 / 100.0,
            )

            # Controller 19: Purge B -> Purge Rate setpoint (cascade)
            dxmv = self.ctrl19.calculate_change(xmeas[29], dt3)
            dxmv = emit_output(
                "controller_ctrl19_primary_output",
                "ctrl19_primary_output",
                dxmv,
            )
            self.setpoints[5] = emit_output(
                "controller_ctrl19_scaled_output",
                "ctrl19_scaled_output",
                self.setpoints[5] + dxmv * 1.0 / 100.0,
            )
            self.ctrl6.setpoint = self.setpoints[5]

        # Slow loops (every 900 seconds = 15 minutes)
        if time_step % 900 == 0:
            # Controller 20: Product E -> Stripper Temp setpoint (cascade)
            dxmv = self.ctrl20.calculate_change(xmeas[37], dt900)
            dxmv = emit_output(
                "controller_ctrl20_primary_output",
                "ctrl20_primary_output",
                dxmv,
            )
            self.setpoints[15] = emit_output(
                "controller_ctrl20_scaled_output",
                "ctrl20_scaled_output",
                self.setpoints[15] + dxmv * 130.0 / 100.0,
            )
            self.ctrl16.setpoint = self.setpoints[15]

        for idx in range(12):
            field = f"xmv_{idx + 1:02d}_ctrl_output"
            phase = f"controller_xmv_{idx + 1:02d}_ctrl_output"
            xmv_new[idx] = emit_output(phase, field, xmv_new[idx])

        # Apply constraints
        for i in range(11):
            xmv_new[i] = np.clip(xmv_new[i], 0.0, 100.0)

        self.step_count = time_step

        return xmv_new

    def _purge_control(
        self,
        xmeas: np.ndarray,
        current_xmv6: float,
        dt: float,
        hook: ControllerHook = None,
    ) -> float:
        """
        Purge valve control with override logic.

        Implements the complex purge control from CONTRL6 in temain_mod.f.
        """
        sep_pressure = xmeas[12]  # Separator pressure
        high_output = _override_output(
            hook,
            phase="controller_purge_override_high_output",
            field="purge_override_high_output",
            value=100.0,
        )
        low_output = _override_output(
            hook,
            phase="controller_purge_override_low_output",
            field="purge_override_low_output",
            value=0.0,
        )
        normal_output = float(self.last_outputs.get("purge_normal_ctrl_output", current_xmv6))
        self.last_outputs["purge_override_high_output"] = float(high_output)
        self.last_outputs["purge_override_low_output"] = float(low_output)

        # High pressure override
        if sep_pressure >= 2950.0:
            self.purge_flag = 1
            self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
            return float(high_output)
        elif self.purge_flag == 1 and sep_pressure >= 2633.7:
            self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
            return float(high_output)
        elif self.purge_flag == 1 and sep_pressure < 2633.7:
            self.purge_flag = 0
            self.ctrl6.err_old = 0.0
            self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
            return 40.060

        # Low pressure override
        if sep_pressure <= 2300.0:
            self.purge_flag = 2
            self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
            return float(low_output)
        elif self.purge_flag == 2 and sep_pressure <= 2633.7:
            self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
            return float(low_output)
        elif self.purge_flag == 2 and sep_pressure > 2633.7:
            self.purge_flag = 0
            self.ctrl6.err_old = 0.0
            self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
            return 40.060

        # Normal control
        self.purge_flag = 0
        normal_output = self.ctrl6.calculate(xmeas[9], current_xmv6, dt)
        normal_output = _override_output(
            hook,
            phase="controller_purge_normal_ctrl_output",
            field="purge_normal_ctrl_output",
            value=normal_output,
        )
        self.last_outputs["purge_normal_ctrl_output"] = float(normal_output)
        return float(normal_output)

    def reset(self):
        """Reset all controllers to initial state."""
        self._init_setpoints()
        self._init_controllers()
        self.purge_flag = 0
        self.step_count = 0
        self.last_outputs = {}

    def get_parameters(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return {
            "mode": self._mode,
            "mode_info": self.get_mode_info(),
            "setpoints": self.setpoints.copy(),
            "purge_flag": self.purge_flag,
        }

    def set_setpoint(self, index: int, value: float):
        """
        Set a setpoint value.

        Args:
            index: Setpoint index (0-19)
            value: New setpoint value
        """
        if 0 <= index < len(self.setpoints):
            self.setpoints[index] = value


@register_controller(
    name="manual",
    description="Manual control - holds MVs at specified values"
)
class ManualController(BaseController):
    """
    Manual control mode - holds MVs at specified values.

    Useful for open-loop testing or when manual intervention is needed.
    """

    name = "manual"
    description = "Manual control - holds MVs at specified values"
    version = "1.0.0"
    controlled_mvs = list(range(1, 13))  # Can control all MVs

    def __init__(self, initial_values: np.ndarray = None):
        """
        Initialize manual controller.

        Args:
            initial_values: Initial MV values (default from steady state)
        """
        if initial_values is None:
            from .constants import INITIAL_STATES
            self.mv_values = INITIAL_STATES[38:50].copy()
        else:
            self.mv_values = initial_values.copy()

    def set_mv(self, index: int, value: float):
        """
        Set a manipulated variable.

        Args:
            index: MV index (1-12)
            value: Valve position (0-100%)
        """
        idx = index - 1 if index >= 1 else index
        if 0 <= idx < 12:
            self.mv_values[idx] = np.clip(value, 0.0, 100.0)

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        time_step: int
    ) -> np.ndarray:
        """
        Return the manually set MV values.

        Args:
            xmeas: Current measurements (ignored)
            xmv: Current MVs (ignored)
            time_step: Current step (ignored)

        Returns:
            Manual MV values
        """
        return self.mv_values.copy()

    def reset(self):
        """Reset to initial MV values."""
        from .constants import INITIAL_STATES
        self.mv_values = INITIAL_STATES[38:50].copy()

    def get_parameters(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return {
            "mv_values": self.mv_values.copy(),
        }


# =============================================================================
# PLUGIN CONTROLLERS (merged from controller_plugins.py)
# =============================================================================


@register_controller(
    name="reactor_temp",
    description="Single-loop reactor temperature control via cooling water"
)
class ReactorTemperatureController(BaseController):
    """Single-loop controller for reactor temperature."""

    name = "reactor_temp"
    description = "Single-loop reactor temperature control via cooling water"
    version = "1.0.0"
    controlled_mvs = [10]

    def __init__(
        self,
        setpoint: float = 120.40,
        gain: float = -1.56,
        taui: float = 0.403,
    ):
        self.setpoint = setpoint
        self.gain = gain
        self.taui = taui

        self._controller = PIController(
            setpoint=setpoint,
            gain=gain,
            taui=taui,
            scale=100.0 / 150.0,
            output_min=0.0,
            output_max=100.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            reactor_temp = xmeas[8]
            xmv_new[9] = self._controller.calculate(
                reactor_temp, xmv_new[9], dt
            )

        return xmv_new

    def reset(self):
        self._controller.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "setpoint": self.setpoint,
            "gain": self.gain,
            "taui": self.taui,
        }


@register_controller(
    name="separator_level",
    description="Single-loop separator level control"
)
class SeparatorLevelController(BaseController):
    """Single-loop controller for separator level."""

    name = "separator_level"
    description = "Single-loop separator level control"
    version = "1.0.0"
    controlled_mvs = [7]

    def __init__(
        self,
        setpoint: float = 50.0,
        gain: float = -2.06,
    ):
        self.setpoint = setpoint
        self.gain = gain

        self._controller = PIController(
            setpoint=setpoint,
            gain=gain,
            taui=0.0,
            scale=100.0 / 70.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            sep_level = xmeas[11]
            xmv_new[6] = self._controller.calculate(
                sep_level, xmv_new[6], dt
            )

        return xmv_new

    def reset(self):
        self._controller.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {"setpoint": self.setpoint, "gain": self.gain}


@register_controller(
    name="stripper_level",
    description="Single-loop stripper level control"
)
class StripperLevelController(BaseController):
    """Single-loop controller for stripper level."""

    name = "stripper_level"
    description = "Single-loop stripper level control"
    version = "1.0.0"
    controlled_mvs = [8]

    def __init__(
        self,
        setpoint: float = 50.0,
        gain: float = -1.62,
    ):
        self.setpoint = setpoint
        self.gain = gain

        self._controller = PIController(
            setpoint=setpoint,
            gain=gain,
            taui=0.0,
            scale=100.0 / 70.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            strip_level = xmeas[14]
            xmv_new[7] = self._controller.calculate(
                strip_level, xmv_new[7], dt
            )

        return xmv_new

    def reset(self):
        self._controller.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {"setpoint": self.setpoint, "gain": self.gain}


@register_controller(
    name="reactor_subsystem",
    description="Reactor subsystem control (temperature, level, cooling)"
)
class ReactorSubsystemController(BaseController):
    """Subsystem controller for the reactor section."""

    name = "reactor_subsystem"
    description = "Reactor subsystem control (temperature, level, cooling)"
    version = "1.0.0"
    controlled_mvs = [4, 10]

    def __init__(
        self,
        temp_setpoint: float = 120.40,
        level_setpoint: float = 75.0,
    ):
        self.temp_setpoint = temp_setpoint
        self.level_setpoint = level_setpoint

        self.cw_temp_setpoint = 94.599
        self._ctrl_cw_temp = PIController(
            setpoint=self.cw_temp_setpoint,
            gain=-1.56,
            taui=1452.0 / 3600.0,
            scale=100.0 / 150.0,
        )

        self._ctrl_reactor_temp = PIController(
            setpoint=temp_setpoint,
            gain=28.3,
            taui=982.0 / 3600.0,
            scale=100.0 / 150.0,
        )

        self.ac_feed_setpoint = 9.3477
        self._ctrl_ac_feed = PIController(
            setpoint=self.ac_feed_setpoint,
            gain=1.0,
            taui=0.0,
            scale=100.0 / 15.25,
        )

        self._ctrl_reactor_level = PIController(
            setpoint=level_setpoint,
            gain=1.11,
            taui=3168.0 / 3600.0,
            scale=100.0 / 50.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0

            dxmv = self._ctrl_reactor_temp.calculate_change(xmeas[8], dt)
            self.cw_temp_setpoint += dxmv * 150.0 / 100.0
            self._ctrl_cw_temp.setpoint = self.cw_temp_setpoint

            xmv_new[9] = self._ctrl_cw_temp.calculate(
                xmeas[20], xmv_new[9], dt
            )

            dxmv = self._ctrl_reactor_level.calculate_change(xmeas[7], dt)
            self.ac_feed_setpoint += dxmv * 15.25 / 100.0
            self._ctrl_ac_feed.setpoint = self.ac_feed_setpoint

            xmv_new[3] = self._ctrl_ac_feed.calculate(
                xmeas[3], xmv_new[3], dt
            )

        return xmv_new

    def reset(self):
        self.cw_temp_setpoint = 94.599
        self.ac_feed_setpoint = 9.3477
        self._ctrl_cw_temp.reset()
        self._ctrl_reactor_temp.reset()
        self._ctrl_ac_feed.reset()
        self._ctrl_reactor_level.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "temp_setpoint": self.temp_setpoint,
            "level_setpoint": self.level_setpoint,
            "cw_temp_setpoint": self.cw_temp_setpoint,
            "ac_feed_setpoint": self.ac_feed_setpoint,
        }


@register_controller(
    name="separator_subsystem",
    description="Separator subsystem control (level, pressure)"
)
class SeparatorSubsystemController(BaseController):
    """Subsystem controller for the separator section."""

    name = "separator_subsystem"
    description = "Separator subsystem control (level, pressure)"
    version = "1.0.0"
    controlled_mvs = [7, 11]

    def __init__(
        self,
        level_setpoint: float = 50.0,
    ):
        self.level_setpoint = level_setpoint

        self._ctrl_level = PIController(
            setpoint=level_setpoint,
            gain=-2.06,
            taui=0.0,
            scale=100.0 / 70.0,
        )

        self._ctrl_condenser = PIController(
            setpoint=22.949,
            gain=1.09,
            taui=2600.0 / 3600.0,
            scale=100.0 / 46.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            xmv_new[6] = self._ctrl_level.calculate(
                xmeas[11], xmv_new[6], dt
            )
            xmv_new[10] = self._ctrl_condenser.calculate(
                xmeas[16], xmv_new[10], dt
            )

        return xmv_new

    def reset(self):
        self._ctrl_level.reset()
        self._ctrl_condenser.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {"level_setpoint": self.level_setpoint}


@register_controller(
    name="feed_subsystem",
    description="Feed subsystem control (D, E, A, A+C feeds)"
)
class FeedSubsystemController(BaseController):
    """Subsystem controller for feed flow loops."""

    name = "feed_subsystem"
    description = "Feed subsystem control (D, E, A, A+C feeds)"
    version = "1.0.0"
    controlled_mvs = [1, 2, 3, 4]

    def __init__(
        self,
        d_feed_sp: float = 3664.0,
        e_feed_sp: float = 4509.3,
        a_feed_sp: float = 0.25052,
        ac_feed_sp: float = 9.3477,
    ):
        self.d_feed_sp = d_feed_sp
        self.e_feed_sp = e_feed_sp
        self.a_feed_sp = a_feed_sp
        self.ac_feed_sp = ac_feed_sp

        self._ctrl_d = PIController(
            setpoint=d_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 5811.0
        )
        self._ctrl_e = PIController(
            setpoint=e_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 8354.0
        )
        self._ctrl_a = PIController(
            setpoint=a_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 1.017
        )
        self._ctrl_ac = PIController(
            setpoint=ac_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 15.25
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            xmv_new[0] = self._ctrl_d.calculate(xmeas[1], xmv_new[0], dt)
            xmv_new[1] = self._ctrl_e.calculate(xmeas[2], xmv_new[1], dt)
            xmv_new[2] = self._ctrl_a.calculate(xmeas[0], xmv_new[2], dt)
            xmv_new[3] = self._ctrl_ac.calculate(xmeas[3], xmv_new[3], dt)

        return xmv_new

    def reset(self):
        self._ctrl_d.reset()
        self._ctrl_e.reset()
        self._ctrl_a.reset()
        self._ctrl_ac.reset()

    def set_setpoints(
        self,
        d_feed: float = None,
        e_feed: float = None,
        a_feed: float = None,
        ac_feed: float = None
    ):
        if d_feed is not None:
            self.d_feed_sp = d_feed
            self._ctrl_d.setpoint = d_feed
        if e_feed is not None:
            self.e_feed_sp = e_feed
            self._ctrl_e.setpoint = e_feed
        if a_feed is not None:
            self.a_feed_sp = a_feed
            self._ctrl_a.setpoint = a_feed
        if ac_feed is not None:
            self.ac_feed_sp = ac_feed
            self._ctrl_ac.setpoint = ac_feed

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "d_feed_sp": self.d_feed_sp,
            "e_feed_sp": self.e_feed_sp,
            "a_feed_sp": self.a_feed_sp,
            "ac_feed_sp": self.ac_feed_sp,
        }


@register_controller(
    name="product_quality",
    description="Product quality control via stripper temperature cascade"
)
class ProductQualityController(BaseController):
    """Product quality controller using cascade control."""

    name = "product_quality"
    description = "Product quality control via stripper temperature cascade"
    version = "1.0.0"
    controlled_mvs = [9]

    def __init__(
        self,
        product_e_setpoint: float = 0.8357,
    ):
        self.product_e_setpoint = product_e_setpoint
        self.stripper_temp_setpoint = 65.731
        self.steam_flow_setpoint = 230.31

        self._ctrl_product_e = PIController(
            setpoint=product_e_setpoint,
            gain=-3.26,
            taui=12408.0 / 3600.0,
            scale=100.0 / 1.6,
        )

        self._ctrl_stripper_temp = PIController(
            setpoint=self.stripper_temp_setpoint,
            gain=0.169,
            taui=236.0 / 3600.0,
            scale=100.0 / 130.0,
        )

        self._ctrl_steam = PIController(
            setpoint=self.steam_flow_setpoint,
            gain=0.41,
            taui=0.0,
            scale=100.0 / 460.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt3 = 3.0 / 3600.0
            self._ctrl_steam.setpoint = self.steam_flow_setpoint
            xmv_new[8] = self._ctrl_steam.calculate(
                xmeas[18], xmv_new[8], dt3
            )
            dxmv = self._ctrl_stripper_temp.calculate_change(xmeas[17], dt3)
            self.steam_flow_setpoint += dxmv * 460.0 / 100.0

        if step % 900 == 0:
            dt900 = 900.0 / 3600.0
            dxmv = self._ctrl_product_e.calculate_change(xmeas[37], dt900)
            self.stripper_temp_setpoint += dxmv * 130.0 / 100.0
            self._ctrl_stripper_temp.setpoint = self.stripper_temp_setpoint

        return xmv_new

    def reset(self):
        self.stripper_temp_setpoint = 65.731
        self.steam_flow_setpoint = 230.31
        self._ctrl_product_e.reset()
        self._ctrl_stripper_temp.reset()
        self._ctrl_steam.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "product_e_setpoint": self.product_e_setpoint,
            "stripper_temp_setpoint": self.stripper_temp_setpoint,
            "steam_flow_setpoint": self.steam_flow_setpoint,
        }


@register_controller(
    name="reactor_composition",
    description="Reactor feed composition control (A, D, E in feed)"
)
class ReactorCompositionController(BaseController):
    """Reactor feed composition controller."""

    name = "reactor_composition"
    description = "Reactor feed composition control (A, D, E in feed)"
    version = "1.0.0"
    controlled_mvs = [1, 2, 3]

    def __init__(
        self,
        comp_a_setpoint: float = 32.188,
        comp_d_setpoint: float = 6.882,
        comp_e_setpoint: float = 18.776,
    ):
        self.comp_a_setpoint = comp_a_setpoint
        self.comp_d_setpoint = comp_d_setpoint
        self.comp_e_setpoint = comp_e_setpoint

        self.a_feed_setpoint = 0.25052
        self.d_feed_setpoint = 3664.0
        self.e_feed_setpoint = 4509.3

        self._ctrl_comp_a = PIController(
            setpoint=comp_a_setpoint, gain=18.0,
            taui=3168.0 / 3600.0, scale=100.0 / 100.0
        )
        self._ctrl_comp_d = PIController(
            setpoint=comp_d_setpoint, gain=8.3,
            taui=3168.0 / 3600.0, scale=100.0 / 100.0
        )
        self._ctrl_comp_e = PIController(
            setpoint=comp_e_setpoint, gain=2.37,
            taui=5069.0 / 3600.0, scale=100.0 / 100.0
        )

        self._ctrl_a_feed = PIController(
            setpoint=self.a_feed_setpoint, gain=1.0,
            taui=0.0, scale=100.0 / 1.017
        )
        self._ctrl_d_feed = PIController(
            setpoint=self.d_feed_setpoint, gain=1.0,
            taui=0.0, scale=100.0 / 5811.0
        )
        self._ctrl_e_feed = PIController(
            setpoint=self.e_feed_setpoint, gain=1.0,
            taui=0.0, scale=100.0 / 8354.0
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt3 = 3.0 / 3600.0
            self._ctrl_a_feed.setpoint = self.a_feed_setpoint
            self._ctrl_d_feed.setpoint = self.d_feed_setpoint
            self._ctrl_e_feed.setpoint = self.e_feed_setpoint

            xmv_new[2] = self._ctrl_a_feed.calculate(xmeas[0], xmv_new[2], dt3)
            xmv_new[0] = self._ctrl_d_feed.calculate(xmeas[1], xmv_new[0], dt3)
            xmv_new[1] = self._ctrl_e_feed.calculate(xmeas[2], xmv_new[1], dt3)

        if step % 360 == 0:
            dt360 = 360.0 / 3600.0
            dxmv = self._ctrl_comp_a.calculate_change(xmeas[22], dt360)
            self.a_feed_setpoint += dxmv * 1.017 / 100.0

            dxmv = self._ctrl_comp_d.calculate_change(xmeas[25], dt360)
            self.d_feed_setpoint += dxmv * 5811.0 / 100.0

            dxmv = self._ctrl_comp_e.calculate_change(xmeas[26], dt360)
            self.e_feed_setpoint += dxmv * 8354.0 / 100.0

        return xmv_new

    def reset(self):
        self.a_feed_setpoint = 0.25052
        self.d_feed_setpoint = 3664.0
        self.e_feed_setpoint = 4509.3
        self._ctrl_comp_a.reset()
        self._ctrl_comp_d.reset()
        self._ctrl_comp_e.reset()
        self._ctrl_a_feed.reset()
        self._ctrl_d_feed.reset()
        self._ctrl_e_feed.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "comp_a_setpoint": self.comp_a_setpoint,
            "comp_d_setpoint": self.comp_d_setpoint,
            "comp_e_setpoint": self.comp_e_setpoint,
            "a_feed_setpoint": self.a_feed_setpoint,
            "d_feed_setpoint": self.d_feed_setpoint,
            "e_feed_setpoint": self.e_feed_setpoint,
        }


@register_controller(
    name="proportional_only",
    description="Simple proportional-only control for all loops"
)
class ProportionalOnlyController(BaseController):
    """Simple proportional-only controller for all MVs."""

    name = "proportional_only"
    description = "Simple proportional-only control for all loops"
    version = "1.0.0"
    controlled_mvs = list(range(1, 12))

    DEFAULT_CONFIG = {
        0: (1, 3664.0, 1.0, 100.0 / 5811.0),
        1: (2, 4509.3, 1.0, 100.0 / 8354.0),
        2: (0, 0.25052, 1.0, 100.0 / 1.017),
        3: (3, 9.3477, 1.0, 100.0 / 15.25),
        4: (4, 26.902, -0.083, 100.0 / 53.0),
        5: (9, 0.33712, 1.22, 100.0 / 1.0),
        6: (11, 50.0, -2.06, 100.0 / 70.0),
        7: (14, 50.0, -1.62, 100.0 / 70.0),
        8: (18, 230.31, 0.41, 100.0 / 460.0),
        9: (20, 94.599, -1.56, 100.0 / 150.0),
        10: (16, 22.949, 1.09, 100.0 / 46.0),
    }

    def __init__(self):
        self._controllers = {}
        for mv_idx, (meas_idx, sp, gain, scale) in self.DEFAULT_CONFIG.items():
            self._controllers[mv_idx] = PIController(
                setpoint=sp,
                gain=gain,
                taui=0.0,
                scale=scale,
            )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            for mv_idx, (meas_idx, _, _, _) in self.DEFAULT_CONFIG.items():
                xmv_new[mv_idx] = self._controllers[mv_idx].calculate(
                    xmeas[meas_idx], xmv_new[mv_idx], dt
                )

        return xmv_new

    def reset(self):
        for ctrl in self._controllers.values():
            ctrl.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            f"xmv{i+1}_setpoint": self._controllers[i].setpoint
            for i in self._controllers
        }


@register_controller(
    name="economic_mpc",
    description="Economic MPC-style controller (simplified demonstration)"
)
class EconomicMPCController(BaseController):
    """Simplified economic MPC-style controller."""

    name = "economic_mpc"
    description = "Economic MPC-style controller (simplified demonstration)"
    version = "1.0.0"
    controlled_mvs = list(range(1, 12))

    REACTOR_TEMP_MAX = 150.0
    REACTOR_TEMP_MIN = 100.0
    REACTOR_PRESSURE_MAX = 2900.0
    REACTOR_PRESSURE_MIN = 2700.0

    def __init__(
        self,
        production_rate_target: float = 1.0,
        product_e_target: float = 0.8357,
    ):
        self.production_rate_target = production_rate_target
        self.product_e_target = product_e_target

        self.base_setpoints = {
            "d_feed": 3664.0,
            "e_feed": 4509.3,
            "a_feed": 0.25052,
            "ac_feed": 9.3477,
            "recycle": 26.902,
            "purge": 0.33712,
            "sep_level": 50.0,
            "strip_level": 50.0,
            "steam": 230.31,
            "reactor_cw": 94.599,
            "cond_cw": 22.949,
        }

        self._init_controllers()

    def _init_controllers(self):
        self._controllers = {
            0: PIController(self.base_setpoints["d_feed"], 1.0, 0.0, scale=100.0/5811.0),
            1: PIController(self.base_setpoints["e_feed"], 1.0, 0.0, scale=100.0/8354.0),
            2: PIController(self.base_setpoints["a_feed"], 1.0, 0.0, scale=100.0/1.017),
            3: PIController(self.base_setpoints["ac_feed"], 1.0, 0.0, scale=100.0/15.25),
            4: PIController(self.base_setpoints["recycle"], -0.083, 1.0/3600.0, scale=100.0/53.0),
            5: PIController(self.base_setpoints["purge"], 1.22, 0.0, scale=100.0/1.0),
            6: PIController(self.base_setpoints["sep_level"], -2.06, 0.0, scale=100.0/70.0),
            7: PIController(self.base_setpoints["strip_level"], -1.62, 0.0, scale=100.0/70.0),
            8: PIController(self.base_setpoints["steam"], 0.41, 0.0, scale=100.0/460.0),
            9: PIController(self.base_setpoints["reactor_cw"], -1.56, 1452.0/3600.0, scale=100.0/150.0),
            10: PIController(self.base_setpoints["cond_cw"], 1.09, 2600.0/3600.0, scale=100.0/46.0),
        }
        self._meas_idx = {0: 1, 1: 2, 2: 0, 3: 3, 4: 4, 5: 9, 6: 11, 7: 14, 8: 18, 9: 20, 10: 16}

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        reactor_temp = xmeas[8]
        reactor_pressure = xmeas[6]

        safety_mode = False
        if reactor_temp > self.REACTOR_TEMP_MAX - 5:
            xmv_new[9] = min(xmv_new[9] + 2.0, 100.0)
            safety_mode = True
        elif reactor_temp < self.REACTOR_TEMP_MIN + 5:
            xmv_new[9] = max(xmv_new[9] - 2.0, 0.0)
            safety_mode = True

        if reactor_pressure > self.REACTOR_PRESSURE_MAX - 50:
            xmv_new[5] = min(xmv_new[5] + 5.0, 100.0)
            safety_mode = True

        if not safety_mode and step % 3 == 0:
            dt = 3.0 / 3600.0
            rate_factor = self.production_rate_target
            self._controllers[0].setpoint = self.base_setpoints["d_feed"] * rate_factor
            self._controllers[1].setpoint = self.base_setpoints["e_feed"] * rate_factor

            for mv_idx, ctrl in self._controllers.items():
                meas_idx = self._meas_idx[mv_idx]
                xmv_new[mv_idx] = ctrl.calculate(xmeas[meas_idx], xmv_new[mv_idx], dt)

        return xmv_new

    def reset(self):
        self._init_controllers()

    def set_production_rate(self, rate: float):
        self.production_rate_target = np.clip(rate, 0.5, 1.2)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "production_rate_target": self.production_rate_target,
            "product_e_target": self.product_e_target,
            "base_setpoints": self.base_setpoints.copy(),
        }


@register_controller(
    name="passthrough",
    description="Passthrough controller - holds MVs at current values"
)
class PassthroughController(BaseController):
    """Passthrough controller that holds MVs at current values."""

    name = "passthrough"
    description = "Passthrough controller - holds MVs at current values"
    version = "1.0.0"
    controlled_mvs = None

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        return xmv.copy()

    def reset(self):
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {}


def create_composite_controller(
    subsystems: List[str],
    fallback: str = "passthrough"
) -> CompositeController:
    """Create a composite controller from named subsystem controllers."""
    mv_assignments = {
        "reactor_temp": [10],
        "separator_level": [7],
        "stripper_level": [8],
        "reactor_subsystem": [4, 10],
        "separator_subsystem": [7, 11],
        "feed_subsystem": [1, 2, 3, 4],
        "product_quality": [9],
        "reactor_composition": [1, 2, 3],
    }

    fallback_ctrl = ControllerRegistry.create(fallback)
    composite = CompositeController(fallback_controller=fallback_ctrl)

    for name in subsystems:
        ctrl = ControllerRegistry.create(name)
        mvs = mv_assignments.get(name, ctrl.controlled_mvs or [])
        if mvs:
            composite.add_controller(ctrl, mvs)

    return composite

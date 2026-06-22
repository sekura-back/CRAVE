# Shared Protocols - Section 3

"""..."""

from __future__ import annotations



from typing import (

    Any,

    Callable,

    Dict,

    List,

    Mapping,

    Optional,

    Protocol,

    Sequence,

    Tuple,

    TypedDict,

    runtime_checkable,

)





# ============================================================================

#
# ============================================================================



InjectionHook = Callable[

    [int, str, Dict[str, float]],

    Optional[Dict[str, float]],

]

"""..."""





# ============================================================================

#
# ============================================================================





@runtime_checkable

class PhysicalProcessProtocol(Protocol):

    """..."""



    def get_state(self) -> Dict[str, float]:

        """..."""

        ...



    def step(

        self, u_applied: Mapping[str, float], t_step: int

    ) -> Dict[str, float]:

        """..."""

        ...



    def reset(

        self, init_state: Optional[Mapping[str, float]] = None

    ) -> None:

        """..."""

        ...





@runtime_checkable

class PLCControllerProtocol(Protocol):

    """..."""



    def reset(

        self, init_state: Optional[Mapping[str, float]] = None

    ) -> None:

        """..."""

        ...





@runtime_checkable

class ClosedLoopSimProtocol(Protocol):

    """..."""



    def run(

        self,

        steps: int,

        injection_hook: Optional[InjectionHook] = None,

        return_trace: bool = True,

        **kwargs: Any,

    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        """..."""

        ...



    def close(self) -> None:

        """..."""

        ...





# ============================================================================

#
# ============================================================================





class PhysicalProcessManifest(TypedDict):

    """..."""

    module: str

    class_name: str

    state_variables: List[str]

    control_inputs: List[str]

    init_params: Dict[str, Any]





class ControllerManifest(TypedDict):

    """..."""

    module: str

    class_name: str





class SimulationManifest(TypedDict):

    """..."""

    module: str

    class_name: str





class InjectionPointSpec(TypedDict):

    """..."""

    name: str

    default_value: float

    range: List[float]  # [min, max]

    rate: float  # 





class HazardRuleSpec(TypedDict):

    """..."""

    id: str

    var: str

    threshold: float

    direction: str  # "upper" | "lower"





class AlarmRuleSpec(TypedDict):

    """..."""

    id: str

    var: str

    threshold: float

    direction: str

    margin_col: str





class SystemManifest(TypedDict):

    """..."""

    system_id: str

    display_name: str

    physical_process: PhysicalProcessManifest

    controller: ControllerManifest

    simulation: SimulationManifest

    injection_points: List[InjectionPointSpec]

    hazard_rules: List[HazardRuleSpec]

    alarm_rules: List[AlarmRuleSpec]

    nominal_state: Dict[str, float]


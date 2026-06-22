# Boiler CCS Controller - Section 4.1

"""..."""

from __future__ import annotations



from dataclasses import dataclass

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple





#




@dataclass

class AlarmResult:

    """..."""

    triggered: bool

    alarm_id: str

    margin: float

    description: str





#


_NOMINAL = {

    "true_load": 349.0,

    "coal_flow": 157.4,

    "feedwater_flow": 1065.7,

    "main_steam_flow": 1064.0,

    "waterwall_temp": 425.0,

    "main_steam_pressure": 23.7,

    "fuel_command": 157.6,

    "water_pump_speed": 2500.0,

    "load_output": 349.0,

    "steam_setpoint": 23.7,

    "boiler_setpoint": 157.6,

    "water_setpoint": 1065.7,

}



#




def hscharc_eval(points: Sequence[Tuple[float, float]], x: float) -> float:

    """..."""

    if not points:

        return float(x)

    pts = sorted(points, key=lambda p: p[0])

    if x <= pts[0][0]:

        return float(pts[0][1])

    if x >= pts[-1][0]:

        return float(pts[-1][1])

    for i in range(len(pts) - 1):

        x1, y1 = pts[i]

        x2, y2 = pts[i + 1]

        if x1 <= x <= x2:

            if x2 == x1:

                return float(y1)

            return float(y1 + (y2 - y1) * (x - x1) / (x2 - x1))

    return float(pts[-1][1])





class Hfop:

    """..."""



    def __init__(self, TC: float, Ts: float, initial_output: float = 0.0):

        self.TC = float(TC)

        self.Ts = float(Ts)

        self.av_0 = float(initial_output)

        self.last_debug: Dict[str, Any] = {}



    def update(self, u: float) -> float:

        alpha = self.Ts / (self.TC + self.Ts) if (self.TC + self.Ts) > 0 else 1.0

        self.av_0 = (1.0 - alpha) * self.av_0 + alpha * float(u)

        return self.av_0



    def reset(self, initial_output: float = 0.0) -> None:

        self.av_0 = float(initial_output)





class HSLIMRate:

    """..."""



    def __init__(self, rate_limit: float, Ts: float, initial_output: float = 0.0):

        self.rate_limit = float(rate_limit)

        self.Ts = float(Ts)

        self.prev_output = float(initial_output)

        self.last_debug: Dict[str, Any] = {}



    def update(self, target: float) -> float:

        max_delta = self.rate_limit * self.Ts

        delta = float(target) - self.prev_output

        if abs(delta) > max_delta:

            delta = max_delta if delta > 0 else -max_delta

        self.prev_output = self.prev_output + delta

        self.last_debug = {"rate_mode": "limited" if abs(delta) >= max_delta * 0.999 else "free"}

        return self.prev_output



    def reset(self, initial_output: float = 0.0) -> None:

        self.prev_output = float(initial_output)





class HSVPID:

    """..."""



    def __init__(

        self,

        CP: float,

        TI: float,

        TD: float,

        OC: float,

        OT: float,

        OB: float,

        OV: float,

        Ts: float,

        initial_output: float = 0.0,

        deadband: float = 0.0,

    ):

        self.CP = float(CP)

        self.TI = float(TI) if TI > 0 else 1e9

        self.TD = float(TD)

        self.OC = float(OC)   # 

        self.OT = float(OT)   # 

        self.OB = float(OB)   # （）

        self.OV = float(OV)   # （， = -OB）

        self.Ts = float(Ts)

        self.deadband = float(deadband)



        self.AV_0 = float(initial_output)

        self.err_0 = 0.0

        self.delta_err_0 = 0.0

        self.dk_0 = 0.0

        self.last_debug: Dict[str, Any] = {}



    def update(self, PV: float, SP: float) -> float:

        err = float(SP) - float(PV)



        #
        deadband_hit = abs(err) < self.deadband

        if deadband_hit:

            err_eff = 0.0

        else:

            err_eff = err



        #
        delta_err = err_eff - self.err_0

        delta_delta_err = delta_err - self.delta_err_0



        dk = (

            self.CP * delta_err

            + self.CP * self.Ts / self.TI * err_eff

            + self.CP * self.TD / self.Ts * delta_delta_err

        )



        #
        ov_hi_clip = dk > self.OB

        ov_lo_clip = dk < self.OV

        if ov_hi_clip:

            dk = self.OB

        elif ov_lo_clip:

            dk = self.OV



        av_raw = self.AV_0 + dk



        #
        sat_hi = av_raw > self.OT

        sat_lo = av_raw < self.OC

        if sat_hi:

            av_raw = self.OT

        elif sat_lo:

            av_raw = self.OC



        #
        integral_on = not (sat_hi or sat_lo)

        td_on = self.TD > 0



        self.AV_0 = av_raw

        self.err_0 = err_eff

        self.delta_err_0 = delta_err

        self.dk_0 = dk



        self.last_debug = {

            "pid": {

                "deadband_hit": deadband_hit,

                "integral_on": integral_on,

                "td_on": td_on,

                "sat_hi": sat_hi,

                "sat_lo": sat_lo,

                "ov_hi_clip": ov_hi_clip,

                "ov_lo_clip": ov_lo_clip,

            }

        }

        return self.AV_0



    def reset(self, initial_output: float = 0.0) -> None:

        self.AV_0 = float(initial_output)

        self.err_0 = 0.0

        self.delta_err_0 = 0.0

        self.dk_0 = 0.0





#


#
_STEAM_HS_CURVE: Sequence[Tuple[float, float]] = [

    (0.0, 8.0),

    (100.0, 12.0),

    (200.0, 17.0),

    (300.0, 21.5),

    (349.0, 23.7),

    (400.0, 25.5),

    (500.0, 28.0),

]



#
_WATER_K_CURVE: Sequence[Tuple[float, float]] = [

    (0.0, 0.0),

    (50.0, 300.0),

    (100.0, 600.0),

    (157.6, 1065.7),

    (200.0, 1350.0),

    (250.0, 1700.0),

]





class LoadInstructionGenerator:

    """..."""



    def __init__(self, Ts: float, rate_limit: float = 5.0, initial_output: float = 349.0):

        self.Ts = float(Ts)

        self.rate_limiter = HSLIMRate(rate_limit=rate_limit, Ts=Ts, initial_output=initial_output)

        self.last_debug: Dict[str, Any] = {}



    def update(self, actual_load: float) -> float:

        #
        out = self.rate_limiter.update(float(actual_load))

        self.last_debug = self.rate_limiter.last_debug

        return out



    def reset(self, initial_output: float = 349.0) -> None:

        self.rate_limiter.reset(initial_output)





class MainSteamPressureController:

    """..."""



    def __init__(self, Ts: float):

        self.Ts = float(Ts)

        #
        self.hfop1 = Hfop(TC=2.0, Ts=Ts, initial_output=23.7)

        self.hfop2 = Hfop(TC=2.0, Ts=Ts, initial_output=23.7)

        self.hfop3 = Hfop(TC=2.0, Ts=Ts, initial_output=23.7)

        self.hfop4 = Hfop(TC=2.0, Ts=Ts, initial_output=23.7)

        self.rate = HSLIMRate(rate_limit=0.5, Ts=Ts, initial_output=23.7)

        self.last_debug: Dict[str, Any] = {}



    def update(self, load_output: float, pressure_bias: float, unit_pressure_sp: float) -> Tuple[float, float]:

        #
        hs = hscharc_eval(_STEAM_HS_CURVE, float(load_output))

        #
        h1 = self.hfop1.update(hs)

        h2 = self.hfop2.update(h1)

        h3 = self.hfop3.update(h2)

        h4 = self.hfop4.update(h3)

        #
        hslim = self.rate.update(h4)

        self.last_debug = {

            "hs_hsc_mode": "normal",

            "hs_hsc_segment": 0,

            "rate_mode": self.rate.last_debug.get("rate_mode", "free"),

        }

        return hs, hslim



    def reset(self, initial_output: float = 23.7) -> None:

        for hfop in (self.hfop1, self.hfop2, self.hfop3, self.hfop4):

            hfop.reset(initial_output)

        self.rate.reset(initial_output)





class BoilerMasterController:

    """..."""



    #
    BOILER_CUT_TH: float = 1.5

    """..."""



    ALARM_RULES = [

        {

            "id": "P-BOILER-CUT-001",

            "kind": "deviation",

            "expr": "|steam_setpoint - main_steam_pressure| > BOILER_CUT_TH",

            "ref_var": "steam_setpoint",

            "proc_var": "main_steam_pressure",

            "threshold": 1.5,

            "component": "boiler",

            "duration_s": 0.0,

        },

    ]



    def __init__(self, Ts: float):

        self.Ts = float(Ts)

        self.hfop1 = Hfop(TC=3.0, Ts=Ts, initial_output=157.6)

        #
        self.HSV = HSVPID(

            CP=100.0 / 46.0, TI=50.0, TD=0.0,

            OC=0.0, OT=250.0, OB=5.0, OV=-5.0,

            Ts=Ts, initial_output=157.6, deadband=0.3,

        )

        self.last_debug: Dict[str, Any] = {}

        self.last_alarm: AlarmResult = AlarmResult(

            triggered=False, alarm_id="", margin=self.BOILER_CUT_TH, description=""

        )



    def update(self, main_steam_pressure: float, steam_setpoint: float) -> float:

        #
        pv_filtered = self.hfop1.update(float(main_steam_pressure))

        boiler_cmd = self.HSV.update(PV=pv_filtered, SP=float(steam_setpoint))

        self.last_debug = self.HSV.last_debug

        return boiler_cmd



    def check_alarm(self, steam_setpoint: float, main_steam_pressure: float) -> AlarmResult:

        """..."""

        deviation = abs(float(steam_setpoint) - float(main_steam_pressure))

        margin = self.BOILER_CUT_TH - deviation

        triggered = deviation > self.BOILER_CUT_TH

        self.last_alarm = AlarmResult(

            triggered=triggered,

            alarm_id="A-BOILER-CUT" if triggered else "",

            margin=float(margin),

            description=(

                f"|steam_setpoint({steam_setpoint:.3f}) - "

                f"main_steam_pressure({main_steam_pressure:.3f})| = "

                f"{deviation:.3f} > {self.BOILER_CUT_TH}"

            ) if triggered else "",

        )

        return self.last_alarm



    def reset(self, initial_output: float = 157.6) -> None:

        self.hfop1.reset(23.7)  # PV filter: initial pressure

        self.HSV.reset(initial_output)

        self.last_alarm = AlarmResult(

            triggered=False, alarm_id="", margin=self.BOILER_CUT_TH, description=""

        )





class FuelController:

    """..."""



    #
    FUEL_CUT_TH: float = 43.0

    """..."""



    ALARM_RULES = [

        {

            "id": "P-FUEL-CUT-001",

            "kind": "deviation",

            "expr": "|coal_flow - fuel_command| > FUEL_CUT_TH",

            "ref_var": "fuel_command",

            "proc_var": "coal_flow",

            "threshold": 43.0,

            "component": "fuel",

            "duration_s": 0.0,

        },

    ]



    def __init__(self, Ts: float):

        self.Ts = float(Ts)

        self.hfop1 = Hfop(TC=0.5, Ts=Ts, initial_output=157.6)

        self.rate = HSLIMRate(rate_limit=10.0, Ts=Ts, initial_output=157.6)

        #
        self.hsvpid = HSVPID(

            CP=1.5, TI=30.0, TD=0.0,

            OC=0.0, OT=300.0, OB=2.0, OV=-2.0,

            Ts=Ts, initial_output=157.6, deadband=0.1,

        )

        self.last_debug: Dict[str, Any] = {}

        self.last_alarm: AlarmResult = AlarmResult(

            triggered=False, alarm_id="", margin=self.FUEL_CUT_TH, description=""

        )



    def update(self, feeder_count: float, coal_flow: float, boiler_setpoint: float) -> float:

        #
        rate_out = self.rate.update(float(boiler_setpoint))

        #
        pid_out = self.hsvpid.update(PV=float(coal_flow), SP=rate_out)

        fuel_cmd = self.hfop1.update(pid_out)

        self.last_debug = {

            "pt_hsc_mode": "normal",

            "pt_hsc_segment": 0,

            "ti_hsc_mode": "normal",

            "ti_hsc_segment": 0,

            "rate_mode": self.rate.last_debug.get("rate_mode", "free"),

            "pid": self.hsvpid.last_debug.get("pid", {}),

        }

        return fuel_cmd



    def check_alarm(self, coal_flow: float, fuel_command: float) -> AlarmResult:

        """..."""

        deviation = abs(float(coal_flow) - float(fuel_command))

        margin = self.FUEL_CUT_TH - deviation

        triggered = deviation > self.FUEL_CUT_TH

        self.last_alarm = AlarmResult(

            triggered=triggered,

            alarm_id="A-FUEL-CUT" if triggered else "",

            margin=float(margin),

            description=(

                f"|coal_flow({coal_flow:.3f}) - "

                f"fuel_command({fuel_command:.3f})| = "

                f"{deviation:.3f} > {self.FUEL_CUT_TH}"

            ) if triggered else "",

        )

        return self.last_alarm



    def reset(self, initial_output: float = 157.6) -> None:

        self.hfop1.reset(initial_output)

        self.rate.reset(initial_output)

        self.hsvpid.reset(initial_output)

        self.last_alarm = AlarmResult(

            triggered=False, alarm_id="", margin=self.FUEL_CUT_TH, description=""

        )





class WaterK:

    """..."""



    def __init__(self, Ts: float):

        self.Ts = float(Ts)

        self.hfop1 = Hfop(TC=20.0, Ts=Ts, initial_output=1065.7)

        self.last_debug: Dict[str, Any] = {}



    def update(self, boiler_setpoint: float, pressure_bias: float) -> float:

        water_sp = hscharc_eval(_WATER_K_CURVE, float(boiler_setpoint))

        water_sp = self.hfop1.update(water_sp)

        self.last_debug = {

            "hsc_mode": "normal",

            "hsc_segment": 0,

            "floor_clip": False,

        }

        return water_sp



    def reset(self, initial_output: float = 1065.7) -> None:

        self.hfop1.reset(initial_output)





class FeedwaterController:

    """..."""



    #
    WATER_CUT_TH: float = 150.0

    """..."""



    ALARM_RULES = [

        {

            "id": "P-WATER-CUT-001",

            "kind": "deviation",

            "expr": "|water_setpoint - feedwater_flow| > WATER_CUT_TH",

            "ref_var": "water_setpoint",

            "proc_var": "feedwater_flow",

            "threshold": 150.0,

            "component": "feedwater",

            "duration_s": 0.0,

        },

    ]



    def __init__(self, Ts: float):

        self.Ts = float(Ts)

        self.hfop = Hfop(TC=0.5, Ts=Ts, initial_output=2500.0)

        self.rate = HSLIMRate(rate_limit=200.0, Ts=Ts, initial_output=2500.0)

        #
        self.pid = HSVPID(

            CP=2.0, TI=20.0, TD=0.0,

            OC=0.0, OT=5000.0, OB=50.0, OV=-50.0,

            Ts=Ts, initial_output=2500.0, deadband=1.0,

        )

        self.last_debug: Dict[str, Any] = {}

        self.last_alarm: AlarmResult = AlarmResult(

            triggered=False, alarm_id="", margin=self.WATER_CUT_TH, description=""

        )



    def update(self, main_steam_flow: float, feedwater_flow: float, water_setpoint: float) -> float:

        #
        pid_out = self.pid.update(PV=float(feedwater_flow), SP=float(water_setpoint))

        rate_out = self.rate.update(pid_out)

        pump_speed = self.hfop.update(rate_out)

        self.last_debug = {

            "pt_hsc_mode": "normal",

            "pt_hsc_segment": 0,

            "ti_hsc_mode": "normal",

            "ti_hsc_segment": 0,

            "rate_mode": self.rate.last_debug.get("rate_mode", "free"),

            "pid": self.pid.last_debug.get("pid", {}),

        }

        return pump_speed



    def check_alarm(self, water_setpoint: float, feedwater_flow: float) -> AlarmResult:

        """..."""

        deviation = abs(float(water_setpoint) - float(feedwater_flow))

        margin = self.WATER_CUT_TH - deviation

        triggered = deviation > self.WATER_CUT_TH

        self.last_alarm = AlarmResult(

            triggered=triggered,

            alarm_id="A-WATER-CUT" if triggered else "",

            margin=float(margin),

            description=(

                f"|water_setpoint({water_setpoint:.3f}) - "

                f"feedwater_flow({feedwater_flow:.3f})| = "

                f"{deviation:.3f} > {self.WATER_CUT_TH}"

            ) if triggered else "",

        )

        return self.last_alarm



    def reset(self, initial_output: float = 2500.0) -> None:

        self.hfop.reset(initial_output)

        self.rate.reset(initial_output)

        self.pid.reset(initial_output)

        self.last_alarm = AlarmResult(

            triggered=False, alarm_id="", margin=self.WATER_CUT_TH, description=""

        )





#




class PLCController:

    """..."""



    def __init__(self, Ts: float = 0.1, init_state: Optional[Mapping[str, float]] = None):

        self.Ts = float(Ts)

        self.loadGen = LoadInstructionGenerator(Ts=Ts, initial_output=_NOMINAL["load_output"])

        self.steamControl = MainSteamPressureController(Ts=Ts)

        self.boilerControl = BoilerMasterController(Ts=Ts)

        self.fuelControl = FuelController(Ts=Ts)

        self.waterK = WaterK(Ts=Ts)

        self.waterControl = FeedwaterController(Ts=Ts)

        self.last_output: Dict[str, float] = {}

        self.reset(init_state=init_state)



    def reset(self, init_state: Optional[Mapping[str, float]] = None) -> None:

        nom = dict(_NOMINAL)

        if init_state:

            nom.update({k: float(v) for k, v in init_state.items()})



        self.loadGen.reset(initial_output=nom.get("load_output", nom["true_load"]))

        self.steamControl.reset(initial_output=nom.get("steam_setpoint", nom["main_steam_pressure"]))

        self.boilerControl.reset(initial_output=nom.get("boiler_setpoint", nom["fuel_command"]))

        self.fuelControl.reset(initial_output=nom.get("fuel_command", nom["fuel_command"]))

        self.waterK.reset(initial_output=nom.get("water_setpoint", nom["feedwater_flow"]))

        self.waterControl.reset(initial_output=nom.get("water_pump_speed", nom["water_pump_speed"]))

        self.last_output = {}


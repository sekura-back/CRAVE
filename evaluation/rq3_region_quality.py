"""Unified RQ3 region quality on the current artifact stage3 chain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "src" / "stage4_executor", ROOT / "simulators"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


HOLD = 100
ANCHOR_STEAM_SP = 23.74
REGIONS_FILE = ROOT / "results" / "stage3" / "boiler_stage3_combined_regions.json"
BASE_FILE = ROOT / "results" / "stage2" / "boiler_base_boundary.json"
OUT_FILE = ROOT / "results" / "rq3" / "boiler_results.json"
REGION_VALIDITY_FILE = ROOT / "results" / "rq3" / "region_validity.json"
REGION_VALIDITY_CASES_FILE = ROOT / "results" / "rq3" / "region_validity_cases.json"
TIGHTNESS_FILE = ROOT / "results" / "rq3" / "boiler_tightness.json"
TIGHTNESS_CASES_FILE = ROOT / "results" / "rq3" / "boiler_tightness_cases.json"
ALARM_ACCURACY_FILE = ROOT / "results" / "rq3" / "boiler_alarm_mask_accuracy.json"
ALARM_ACCURACY_CASES_FILE = ROOT / "results" / "rq3" / "boiler_alarm_mask_accuracy_cases.json"
IGNORE_PREDS = ["BOILER_CUT_TH", "FUEL_CUT_TH", "WATER_CUT_TH"]
COVERAGE_SEED = 460
TIGHTNESS_SEED = 460
ALARM_ACCURACY_SEED = 460
TIGHTNESS_ABS_EPS = 1e-2


def get_rq3_platform_config(target: str) -> Dict:
    key = str(target).strip().lower()
    if key in {"boiler", "boiler_ccs"}:
        return {
            "target": "boiler",
            "platform_label": "boiler",
            "manifest_path": ROOT / "simulators" / "boiler_ccs" / "system_manifest.json",
            "regions_file": ROOT / "results" / "stage3" / "boiler_stage3_combined_regions.json",
            "base_file": ROOT / "results" / "stage2" / "boiler_base_boundary.json",
            "out_file": ROOT / "results" / "rq3" / "boiler_results.json",
            "tightness_file": ROOT / "results" / "rq3" / "boiler_tightness.json",
            "tightness_cases_file": ROOT / "results" / "rq3" / "boiler_tightness_cases.json",
            "alarm_accuracy_file": ROOT / "results" / "rq3" / "boiler_alarm_mask_accuracy.json",
            "alarm_accuracy_cases_file": ROOT / "results" / "rq3" / "boiler_alarm_mask_accuracy_cases.json",
            "region_validity_file": ROOT / "results" / "rq3" / "boiler_region_validity.json",
            "region_validity_cases_file": ROOT / "results" / "rq3" / "boiler_region_validity_cases.json",
            "manip_point": "simulation_fuel_command",
            "mask_point": "simulation_steam_setpoint",
            "supports_conditional": True,
        }
    if key in {"te", "tennessee_eastman", "tep"}:
        return {
            "target": "te",
            "platform_label": "te",
            "manifest_path": ROOT / "simulators" / "tennessee_eastman" / "system_manifest.json",
            "regions_file": ROOT / "results" / "stage3" / "te_stage3_subregion_2d_base.json",
            "base_file": ROOT / "results" / "stage2" / "te_base_boundary.json",
            "out_file": ROOT / "results" / "rq3" / "te_results.json",
            "tightness_file": ROOT / "results" / "rq3" / "te_tightness.json",
            "tightness_cases_file": ROOT / "results" / "rq3" / "te_tightness_cases.json",
            "alarm_accuracy_file": ROOT / "results" / "rq3" / "te_alarm_mask_accuracy.json",
            "alarm_accuracy_cases_file": ROOT / "results" / "rq3" / "te_alarm_mask_accuracy_cases.json",
            "region_validity_file": ROOT / "results" / "rq3" / "te_region_validity.json",
            "region_validity_cases_file": ROOT / "results" / "rq3" / "te_region_validity_cases.json",
            "manip_point": "simulation_xmv_07",
            "mask_point": "simulation_sp_separator_level",
            "supports_conditional": False,
        }
    raise ValueError(f"unsupported rq3 target: {target}")


def _load_boiler_sim():
    from simulators.boiler_ccs.simulation import ClosedLoopSim

    return ClosedLoopSim


def _load_te_sim():
    from simulators.tennessee_eastman.simulation import ClosedLoopSim

    return ClosedLoopSim


def _build_hook(point_specs, registry, runtime_args):
    from generic_runtime_executor import build_injection_hook

    return build_injection_hook(
        point_specs=point_specs,
        registry=registry,
        runtime_args=runtime_args,
    )


def wilson_ci(p: float, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [
        round(max(0.0, centre - margin) * 100, 2),
        round(min(1.0, centre + margin) * 100, 2),
    ]


def _duration_choices(region: Dict) -> Sequence[int]:
    start, stop = int(region["t_range"][0]), int(region["t_range"][1])
    return list(range(start, stop + 1, 50))


def _load_base_map(base_file: Path = BASE_FILE) -> Dict[int, float]:
    base_j = json.loads(base_file.read_text(encoding="utf-8"))
    return {
        int(b["duration"]): float(b["lower_target_amp"])
        for b in base_j["boundaries"]
        if b.get("direction") == "pos"
        and b.get("lower_target_amp") is not None
        and int(b["duration"]) % 50 == 0
    }


def _load_regions(regions_file: Path = REGIONS_FILE) -> List[Dict]:
    regions_j = json.loads(regions_file.read_text(encoding="utf-8"))
    return [
        region for region in regions_j["regions"]
        if float(region["B_omega"]["width"]) > 0
    ]


def run_dual(
    fuel_amp: float,
    fuel_dur: int,
    steam_slope: float,
    registry,
    cfg: Optional[Dict] = None,
) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    sim_steps = int(fuel_dur) + HOLD
    if platform_cfg["target"] == "te":
        ClosedLoopSim = _load_te_sim()
        hook = _build_hook(
            point_specs=[
                {
                    "point": platform_cfg["manip_point"],
                    "amp": float(fuel_amp),
                    "duration": int(fuel_dur),
                    "t_start": 0,
                    "role": "manip",
                },
                {
                    "point": platform_cfg["mask_point"],
                    "amp": float(steam_slope),
                    "duration": int(fuel_dur),
                    "t_start": 0,
                    "role": "mask",
                },
            ],
            registry=registry,
            runtime_args={"steps": sim_steps},
        )
        sim = ClosedLoopSim(Ts=1.0, mode=1, random_seed=0)
        try:
            trace, _ = sim.run(
                steps=sim_steps,
                injection_hook=hook,
                return_trace=True,
                stop_on_trip=True,
            )
        finally:
            sim.close()

        hazard_step = None
        alarm_step = None
        for row in trace:
            if hazard_step is None and row.get("trip_any"):
                hazard_step = int(row["t_step"])
            if alarm_step is None and float(row.get("alarm_sep_level_track_margin", 1.0)) <= 0.0:
                alarm_step = int(row["t_step"])
    else:
        ClosedLoopSim = _load_boiler_sim()
        hook = _build_hook(
            point_specs=[
                {
                    "point": "simulation_fuel_command",
                    "amp": float(fuel_amp),
                    "duration": int(fuel_dur),
                    "t_start": 0,
                    "role": "manip",
                },
                {
                    "point": "simulation_steam_setpoint",
                    "amp": float(steam_slope),
                    "duration": int(fuel_dur),
                    "t_start": 0,
                    "role": "mask",
                },
            ],
            registry=registry,
            runtime_args={"steps": sim_steps},
        )
        sim = ClosedLoopSim(Ts=0.1, enable_csv=False)
        try:
            trace, _ = sim.run(
                steps=sim_steps,
                injection_hook=hook,
                return_trace=True,
                stop_on_trip=True,
                ignore_protection_predicates_for_stop=IGNORE_PREDS,
                ignore_protection_predicates_for_enforcement=IGNORE_PREDS,
            )
        finally:
            sim.close()

        hazard_step = None
        alarm_step = None
        for row in trace:
            if hazard_step is None and (
                row.get("hazard_pressure_triggered")
                or row.get("hazard_waterwall_triggered")
                or row.get("hazard_feedwater_triggered")
            ):
                hazard_step = int(row["t_step"])
            if alarm_step is None and (
                row.get("boiler_cut") or row.get("fuel_cut") or row.get("water_cut")
            ):
                alarm_step = int(row["t_step"])

    hazard = hazard_step is not None
    alarm = alarm_step is not None
    reach_avoid = hazard and (alarm_step is None or alarm_step >= hazard_step)
    return {
        "hazard": hazard,
        "alarm": alarm,
        "reach_avoid": reach_avoid,
        "hazard_step": hazard_step,
        "alarm_step": alarm_step,
    }


def make_coverage_samples(regions: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    weights = np.array([max(1, int(r.get("n_fine_pts", 1))) for r in regions], dtype=float)
    weights /= weights.sum()
    samples = []
    for _ in range(n):
        region = regions[int(rng.choice(len(regions), p=weights))]
        duration = int(rng.choice(_duration_choices(region)))
        amp = float(rng.uniform(region["k_range"][0], max(region["k_range"][0] + 1e-6, region["k_range"][1])))
        b = region["B_omega"]
        slope = float(rng.uniform(b["alpha_minus"], b["alpha_plus"]))
        samples.append({"dur": duration, "fuel_amp": amp, "steam_slope": slope})
    return samples


def make_tightness_samples(
    regions: List[Dict],
    base_map: Dict[int, float],
    n: int,
    seed: int = 123,
    cfg: Optional[Dict] = None,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    platform = str((cfg or {}).get("target", "boiler")).lower()
    cond_regions = [region for region in regions if region.get("region_type") == "cond"]
    extreme_cond_regions = [
        region for region in cond_regions if region.get("boundary_role") == "extreme"
    ]
    pool = extreme_cond_regions or cond_regions or regions
    samples = []
    for _ in range(n):
        region = pool[int(rng.integers(0, len(pool)))]
        duration = int(rng.choice(_duration_choices(region)))
        region_amp = float(region["k_range"][0])
        if region.get("region_type") == "cond":
            extreme_range = region.get("extreme_amp_range")
            if isinstance(extreme_range, list) and extreme_range:
                region_amp = float(extreme_range[1])
            delta = float(rng.uniform(0.0, TIGHTNESS_ABS_EPS))
            amp = max(0.001, region_amp - delta)
            tight_mode = "cond_below_extreme"
        else:
            base_lower = base_map[duration]
            delta = float(rng.uniform(0.05, 0.15))
            amp = max(0.001, base_lower * (1.0 - delta))
            tight_mode = "base"
        b = region["B_omega"]
        slope = float(rng.uniform(b["alpha_minus"], b["alpha_plus"]))
        samples.append({
            "dur": duration,
            "fuel_amp": amp,
            "steam_slope": slope,
            "tight_mode": tight_mode,
        })
    return samples


def make_alarm_accuracy_samples(
    regions: List[Dict],
    n: int,
    seed: int = ALARM_ACCURACY_SEED,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        region = regions[int(rng.integers(0, len(regions)))]
        duration = int(rng.choice(_duration_choices(region)))
        amp = float(rng.uniform(region["k_range"][0], max(region["k_range"][0] + 1e-6, region["k_range"][1])))
        b = region["B_omega"]
        width = max(1e-6, b["alpha_plus"] - b["alpha_minus"])
        xi = float(rng.uniform(0.04, 0.06))
        if rng.random() < 0.5:
            slope = b["alpha_minus"] - xi * width
        else:
            slope = b["alpha_plus"] + xi * width
        samples.append({"dur": duration, "fuel_amp": amp, "steam_slope": float(slope)})
    return samples


def run_batch(
    label: str,
    samples: Sequence[Dict],
    registry,
    cfg: Optional[Dict] = None,
    workers: int = 1,
) -> Tuple[Dict, List[Dict]]:
    t0 = time.time()
    def _run_one(sample: Dict) -> Dict:
        if cfg is None:
            result = run_dual(sample["fuel_amp"], sample["dur"], sample["steam_slope"], registry)
        else:
            result = run_dual(
                sample["fuel_amp"],
                sample["dur"],
                sample["steam_slope"],
                registry,
                cfg=cfg,
            )
        return {**sample, **result}

    if int(workers) > 1 and len(samples) > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            results = list(pool.map(_run_one, samples))
    else:
        results = [_run_one(sample) for sample in samples]
    elapsed = time.time() - t0
    n = len(results)
    if label == "cov":
        success = sum(1 for r in results if r["reach_avoid"])
        rate = success / n if n else 0.0
        summary = {
            "n": n,
            "ra": success,
            "rate": round(rate * 100, 2),
            "ci": wilson_ci(rate, n),
        }
    elif label == "tight":
        mode = results[0].get("tight_mode", "base") if results else "base"
        if mode == "cond_below_extreme":
            no_reach_avoid = sum(1 for r in results if not r["reach_avoid"])
            rate = no_reach_avoid / n if n else 0.0
            summary = {
                "n": n,
                "mode": "cond_below_extreme_no_ra",
                "no_reach_avoid": no_reach_avoid,
                "rate": round(rate * 100, 2),
                "ci": wilson_ci(rate, n),
            }
        else:
            no_hazard = sum(1 for r in results if not r["hazard"])
            rate = no_hazard / n if n else 0.0
            summary = {
                "n": n,
                "mode": "base_below_boundary_hazard",
                "no_hazard": no_hazard,
                "rate": round(rate * 100, 2),
                "ci": wilson_ci(rate, n),
            }
    elif label == "acc":
        no_reach_avoid = sum(1 for r in results if not r["reach_avoid"])
        rate = no_reach_avoid / n if n else 0.0
        summary = {
            "n": n,
            "no_reach_avoid": no_reach_avoid,
            "rate": round(rate * 100, 2),
            "ci": wilson_ci(rate, n),
        }
    else:
        success = sum(1 for r in results if r["reach_avoid"])
        rate = success / n if n else 0.0
        deviations = [
            abs(int(r["hazard_step"]) - int(r["dur"]))
            for r in results
            if r["hazard_step"] is not None
        ]
        fail_rows = [r for r in results if not r["reach_avoid"]]
        summary = {
            "n": n,
            "ra": success,
            "rate": round(rate * 100, 2),
            "ci": wilson_ci(rate, n),
            "f_nh": sum(1 for r in fail_rows if not r["hazard"]),
            "f_al": sum(
                1
                for r in fail_rows
                if r["hazard"]
                and r["alarm"]
                and r["alarm_step"] is not None
                and r["hazard_step"] is not None
                and int(r["alarm_step"]) < int(r["hazard_step"])
            ),
            "avg_dev": round(float(np.mean(deviations)) if deviations else 0.0, 1),
        }
    summary["elapsed_s"] = round(elapsed, 1)
    return summary, results


def _artifact_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_region_validity_output(summary: Dict, n_samples: int, seed: int, cfg: Optional[Dict] = None) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    return {
        "platform": platform_cfg["platform_label"],
        "metric": "region_validity",
        "source_stage3_regions": _artifact_rel(platform_cfg["regions_file"]),
        "source_stage2_base": _artifact_rel(platform_cfg["base_file"]),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "summary": summary,
    }


def build_region_validity_cases_output(
    cases: Sequence[Dict],
    summary: Dict,
    seed: int,
    cfg: Optional[Dict] = None,
) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    return {
        "platform": platform_cfg["platform_label"],
        "metric": "region_validity_cases",
        "source_stage3_regions": _artifact_rel(platform_cfg["regions_file"]),
        "source_stage2_base": _artifact_rel(platform_cfg["base_file"]),
        "seed": int(seed),
        "n_cases": int(len(cases)),
        "summary": summary,
        "cases": list(cases),
    }


def build_tightness_output(
    summary: Dict,
    n_samples: int,
    seed: int,
    abs_eps: float,
    cfg: Optional[Dict] = None,
) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    return {
        "platform": platform_cfg["platform_label"],
        "metric": "tightness",
        "source_stage3_regions": _artifact_rel(platform_cfg["regions_file"]),
        "source_stage2_base": _artifact_rel(platform_cfg["base_file"]),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "abs_eps": float(abs_eps),
        "summary": summary,
    }


def build_tightness_cases_output(
    cases: Sequence[Dict],
    summary: Dict,
    seed: int,
    abs_eps: float,
    cfg: Optional[Dict] = None,
) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    return {
        "platform": platform_cfg["platform_label"],
        "metric": "tightness_cases",
        "source_stage3_regions": _artifact_rel(platform_cfg["regions_file"]),
        "source_stage2_base": _artifact_rel(platform_cfg["base_file"]),
        "seed": int(seed),
        "abs_eps": float(abs_eps),
        "n_cases": int(len(cases)),
        "summary": summary,
        "cases": list(cases),
    }


def build_alarm_accuracy_output(summary: Dict, n_samples: int, seed: int, cfg: Optional[Dict] = None) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    return {
        "platform": platform_cfg["platform_label"],
        "metric": "alarm_mask_accuracy",
        "source_stage3_regions": _artifact_rel(platform_cfg["regions_file"]),
        "source_stage2_base": _artifact_rel(platform_cfg["base_file"]),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "summary": summary,
    }


def build_alarm_accuracy_cases_output(
    cases: Sequence[Dict],
    summary: Dict,
    seed: int,
    cfg: Optional[Dict] = None,
) -> Dict:
    platform_cfg = cfg or get_rq3_platform_config("boiler")
    return {
        "platform": platform_cfg["platform_label"],
        "metric": "alarm_mask_accuracy_cases",
        "source_stage3_regions": _artifact_rel(platform_cfg["regions_file"]),
        "source_stage2_base": _artifact_rel(platform_cfg["base_file"]),
        "seed": int(seed),
        "n_cases": int(len(cases)),
        "summary": summary,
        "cases": list(cases),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="boiler", choices=["boiler", "te"])
    ap.add_argument("--n-cov", type=int, default=500)
    ap.add_argument("--n-tight", type=int, default=200)
    ap.add_argument("--n-acc", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    from generic_runtime_executor import load_injection_registry

    cfg = get_rq3_platform_config(args.target)
    manifest = cfg["manifest_path"]
    registry = load_injection_registry(manifest)
    regions = _load_regions(cfg["regions_file"])
    base_map = _load_base_map(cfg["base_file"])

    print(f"Loaded {len(regions)} current {cfg['platform_label']} regions from {cfg['regions_file'].name}")

    cov_summary, cov_raw = run_batch(
        "cov",
        make_coverage_samples(regions, args.n_cov, seed=COVERAGE_SEED),
        registry,
        cfg=cfg,
        workers=args.workers,
    )
    print(f"[coverage] {cov_summary['ra']}/{cov_summary['n']} = {cov_summary['rate']}% {cov_summary['ci']}")

    tight_summary, tight_raw = run_batch(
        "tight",
        make_tightness_samples(regions, base_map, args.n_tight, seed=TIGHTNESS_SEED, cfg=cfg),
        registry,
        cfg=cfg,
        workers=args.workers,
    )
    tight_key = "ra" if "ra" in tight_summary else ("no_reach_avoid" if "no_reach_avoid" in tight_summary else "no_hazard")
    print(f"[tightness] {tight_summary[tight_key]}/{tight_summary['n']} = {tight_summary['rate']}% {tight_summary['ci']}")

    acc_summary, acc_raw = run_batch(
        "acc",
        make_alarm_accuracy_samples(regions, args.n_acc, seed=ALARM_ACCURACY_SEED),
        registry,
        cfg=cfg,
        workers=args.workers,
    )
    print(f"[alarm_accuracy] {acc_summary['no_reach_avoid']}/{acc_summary['n']} = {acc_summary['rate']}% {acc_summary['ci']}")

    out = {
        "platform": cfg["platform_label"],
        "source_stage3_regions": cfg["regions_file"].relative_to(ROOT).as_posix(),
        "source_stage2_base": cfg["base_file"].relative_to(ROOT).as_posix(),
        "coverage": cov_summary,
        "tightness": tight_summary,
        "alarm_accuracy": acc_summary,
    }

    if args.save_raw:
        out["raw"] = {
            "coverage": cov_raw,
            "tightness": tight_raw,
            "alarm_accuracy": acc_raw,
        }

    cfg["out_file"].parent.mkdir(parents=True, exist_ok=True)
    cfg["out_file"].write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    region_validity_out = build_region_validity_output(
        summary=cov_summary,
        n_samples=args.n_cov,
        seed=COVERAGE_SEED,
        cfg=cfg,
    )
    if args.save_raw:
        region_validity_out["raw"] = cov_raw
    cfg["region_validity_file"].write_text(
        json.dumps(region_validity_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    region_validity_cases_out = build_region_validity_cases_output(
        cases=cov_raw,
        summary=cov_summary,
        seed=COVERAGE_SEED,
        cfg=cfg,
    )
    cfg["region_validity_cases_file"].write_text(
        json.dumps(region_validity_cases_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tightness_out = build_tightness_output(
        summary=tight_summary,
        n_samples=args.n_tight,
        seed=TIGHTNESS_SEED,
        abs_eps=TIGHTNESS_ABS_EPS,
        cfg=cfg,
    )
    if args.save_raw:
        tightness_out["raw"] = tight_raw
    cfg["tightness_file"].write_text(
        json.dumps(tightness_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tightness_cases_out = build_tightness_cases_output(
        cases=tight_raw,
        summary=tight_summary,
        seed=TIGHTNESS_SEED,
        abs_eps=TIGHTNESS_ABS_EPS,
        cfg=cfg,
    )
    cfg["tightness_cases_file"].write_text(
        json.dumps(tightness_cases_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    alarm_accuracy_out = build_alarm_accuracy_output(
        summary=acc_summary,
        n_samples=args.n_acc,
        seed=ALARM_ACCURACY_SEED,
        cfg=cfg,
    )
    if args.save_raw:
        alarm_accuracy_out["raw"] = acc_raw
    cfg["alarm_accuracy_file"].write_text(
        json.dumps(alarm_accuracy_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    alarm_accuracy_cases_out = build_alarm_accuracy_cases_output(
        cases=acc_raw,
        summary=acc_summary,
        seed=ALARM_ACCURACY_SEED,
        cfg=cfg,
    )
    cfg["alarm_accuracy_cases_file"].write_text(
        json.dumps(alarm_accuracy_cases_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {cfg['out_file']}")
    print(f"Saved: {cfg['region_validity_file']}")
    print(f"Saved: {cfg['region_validity_cases_file']}")
    print(f"Saved: {cfg['tightness_file']}")
    print(f"Saved: {cfg['tightness_cases_file']}")
    print(f"Saved: {cfg['alarm_accuracy_file']}")
    print(f"Saved: {cfg['alarm_accuracy_cases_file']}")


if __name__ == "__main__":
    main()

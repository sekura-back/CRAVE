from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import evaluation.rq2.rq2_eval as rq2
import src.stage2.base_search as base_search


DEFAULT_SEED = 460
DEFAULT_DURATION_RANGE = (100, 1000)
DEFAULT_RATE_RANGE = (0.0, 1.0)
DEFAULT_BOUNDARY_PLATFORM = "tennessee_eastman"
DEFAULT_BOUNDARY_DRIVER = "xmv_07"
DEFAULT_BOUNDARY_REFERENCE = {"setting_id": "reference", "delta_T": 2, "epsilon_K": 1e-5}
BOUNDARY_SWEEP = [
    {"setting_id": "delta_T_10", "delta_T": 10, "epsilon_K": 1e-4},
    {"setting_id": "default", "delta_T": 5, "epsilon_K": 1e-4},
    {"setting_id": "delta_T_2", "delta_T": 2, "epsilon_K": 1e-4},
    {"setting_id": "epsilon_K_1e-3", "delta_T": 5, "epsilon_K": 1e-3},
    {"setting_id": "epsilon_K_1e-5", "delta_T": 5, "epsilon_K": 1e-5},
]
PARTITION_FACTORS = [0.25, 0.5, 1.0, 2.0]
PARTITION_MERGE_SETTINGS = [{"merge_setting": "no_merge", "delta_merge_factor": None}] + [
    {"merge_setting": f"factor_{factor:g}", "delta_merge_factor": factor}
    for factor in PARTITION_FACTORS
]
DEFAULT_PARTITION_VALIDATION_SAMPLES = 200
BOILER_PARTITION_DRIVERS = ("fuel_command", "water_pump_speed")


def _partition_merge_settings(factors: Sequence[float] | None = None) -> list[dict[str, float | str | None]]:
    values = PARTITION_FACTORS if factors is None else [float(value) for value in factors]
    return [{"merge_setting": "no_merge", "delta_merge_factor": None}] + [
        {"merge_setting": f"factor_{factor:g}", "delta_merge_factor": float(factor)}
        for factor in values
    ]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _boundary_setting(setting: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "setting_id": str(setting["setting_id"]),
        "platform": DEFAULT_BOUNDARY_PLATFORM,
        "hazard_driver": DEFAULT_BOUNDARY_DRIVER,
        "delta_T": int(setting["delta_T"]),
        "epsilon_K": float(setting["epsilon_K"]),
        "duration_range": list(DEFAULT_DURATION_RANGE),
        "rate_range": list(DEFAULT_RATE_RANGE),
    }


def _default_base_runner(root: Path, setting: Mapping[str, Any], output_path: Path, workers: int) -> dict[str, Any]:
    manifest_path = root / "simulators" / DEFAULT_BOUNDARY_PLATFORM / "system_manifest.json"
    executor = base_search.load_executor(manifest_path)
    result = base_search.run_base_search(
        executor=executor,
        platform=DEFAULT_BOUNDARY_PLATFORM,
        hazard_driver=DEFAULT_BOUNDARY_DRIVER,
        duration_start=DEFAULT_DURATION_RANGE[0],
        duration_stop=DEFAULT_DURATION_RANGE[1],
        duration_step=int(setting["delta_T"]),
        duration_anchor_step=base_search.DEFAULT_DURATION_ANCHOR_STEP,
        amp_magnitude_range=DEFAULT_RATE_RANGE,
        coarse_span=0.05,
        amp_tol=float(setting["epsilon_K"]),
        directions=("pos",),
        workers=int(workers),
        process_manifest_path=manifest_path,
    )
    _write_json(output_path, result)
    return result


def _reference_summary(path: Path) -> dict[str, Any]:
    return rq2.build_reference_summary(
        path,
        duration_range=DEFAULT_DURATION_RANGE,
        rate_range=DEFAULT_RATE_RANGE,
        duration_step=int(DEFAULT_BOUNDARY_REFERENCE["delta_T"]),
        direction="pos",
        epsilon_k=float(DEFAULT_BOUNDARY_REFERENCE["epsilon_K"]),
    )


def _project_to_reference_grid(
    predicted: Mapping[int, list[list[float]]],
    reference_durations: Sequence[int],
) -> dict[int, list[list[float]]]:
    if not predicted:
        return {}
    source_durations = sorted(int(duration) for duration in predicted)
    projected: dict[int, list[list[float]]] = {}
    for duration in reference_durations:
        nearest = min(source_durations, key=lambda value: (abs(value - int(duration)), value))
        intervals = predicted.get(nearest, [])
        if intervals:
            projected[int(duration)] = [[float(lo), float(hi)] for lo, hi in intervals]
    return projected


def _boundary_metrics(reference: Mapping[str, Any], predicted_payload: Mapping[str, Any]) -> dict[str, Any]:
    predicted = rq2._boundary_intervals(
        predicted_payload,
        direction="pos",
        rate_range=DEFAULT_RATE_RANGE,
    )
    projected = _project_to_reference_grid(
        predicted,
        [int(value) for value in reference["duration_values"]],
    )
    metrics = rq2._metrics_from_prediction(reference=reference, predicted=projected)
    metrics["duration_projection"] = "nearest_predicted_duration_on_reference_grid"
    return metrics


def run_boundary_sensitivity(
    *,
    root: str | Path,
    rq4_dir: Path,
    base_runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
    workers: int = base_search.DEFAULT_WORKERS,
) -> dict[str, Any]:
    root = Path(root)
    work_dir = rq4_dir / "work" / "boundary"
    runner = base_runner

    reference_setting = _boundary_setting(DEFAULT_BOUNDARY_REFERENCE)
    reference_path = work_dir / "reference" / "boundary_results.json"
    if runner is None:
        reference_payload = _default_base_runner(root, reference_setting, reference_path, workers)
    else:
        reference_payload = dict(runner(reference_setting, reference_path))
    if not reference_path.exists():
        _write_json(reference_path, reference_payload)
    reference = _reference_summary(reference_path)

    rows: list[dict[str, Any]] = []
    for raw_setting in BOUNDARY_SWEEP:
        setting = _boundary_setting(raw_setting)
        output_path = work_dir / setting["setting_id"] / "boundary_results.json"
        if runner is None:
            payload = _default_base_runner(root, setting, output_path, workers)
        else:
            payload = dict(runner(setting, output_path))
        if not output_path.exists():
            _write_json(output_path, payload)
        metrics = _boundary_metrics(reference, payload)
        row = {
            **setting,
            "replay_count": int(payload.get("summary", {}).get("probe_count", 0)),
            "reference_setting_id": reference_setting["setting_id"],
            "reference_output_path": str(reference_path),
            "output_path": str(output_path),
            **metrics,
        }
        rows.append(row)

    payload = {
        "schema_version": 1,
        "reference": reference,
        "rows": rows,
    }
    _write_json(rq4_dir / "rq4_boundary_sensitivity.json", payload)
    _write_csv(rq4_dir / "rq4_boundary_sensitivity.csv", rows)
    return payload


def _mask_widths_from_conditional(path: Path) -> list[float]:
    payload = _read_json(path)
    widths: list[float] = []
    for point in payload.get("points", []):
        for key in ("stable_slope_range", "extreme_slope_range"):
            values = point.get(key) or []
            if len(values) == 2 and float(values[1]) > float(values[0]):
                widths.append(float(values[1]) - float(values[0]))
    return widths


def _region_width(region: Mapping[str, Any]) -> float:
    values = region.get("mask_slope_range") or []
    if len(values) != 2:
        return 0.0
    return max(0.0, float(values[1]) - float(values[0]))


def _region_volume_proxy(region: Mapping[str, Any]) -> float:
    duration = region.get("driver_duration_range") or [0.0, 0.0]
    amp = region.get("driver_amp_range") or [0.0, 0.0]
    duration_width = max(0.0, float(duration[1]) - float(duration[0]))
    amp_width = max(0.0, float(amp[1]) - float(amp[0]))
    return duration_width * amp_width * _region_width(region)


def _representative_path(source_root: Path, driver: str) -> Path:
    return source_root / "results" / "stage3" / "boiler_ccs" / driver / "representative_regions.json"


def _partition_region_output_path(output_root: Path, driver: str) -> Path:
    return output_root / "results" / "stage3" / "boiler_ccs" / driver / "coordinated_regions.json"


def _original_merge_threshold(regions: Sequence[Mapping[str, Any]], fallback_width: float) -> float:
    from src.stage3 import subregions

    records = subregions._aggregate_region_records(regions)
    if records:
        return float(subregions._merge_threshold(records, 0.0))
    return float(fallback_width) * float(subregions.REGION_GAMMA_MERGE)


def _first_hazard_id(path: Path) -> str:
    payload = _read_json(path)
    if payload.get("hazard_id"):
        return str(payload["hazard_id"])
    for boundary in payload.get("boundaries", []):
        if boundary.get("hazard_id"):
            return str(boundary["hazard_id"])
        for interval in boundary.get("target_intervals", []):
            if interval.get("first_hazard_id"):
                return str(interval["first_hazard_id"])
    raise ValueError(f"missing hazard_id in {path}")


def _default_stage3_runner(root: Path, setting: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    from src.stage3 import subregions

    if setting["delta_merge"] is None:
        raise ValueError("no_merge partition sensitivity requires representative_regions.json")
    driver = str(setting["hazard_driver"])
    base_path = root / "results" / "stage2" / "boiler_ccs" / driver / "boundary_results.json"
    conditional_path = root / "results" / "stage2" / "boiler_ccs" / driver / "conditional_results.json"
    extraction_path = root / "results" / "stage1" / "boilerCCS" / "extraction.json"
    manifest_path = root / "simulators" / "boiler_ccs" / "system_manifest.json"
    return subregions.main(
        [
            "--platform",
            "boiler_ccs",
            "--hazard-id",
            _first_hazard_id(base_path),
            "--hazard-driver",
            driver,
            "--base-path",
            str(base_path),
            "--conditional-path",
            str(conditional_path),
            "--extraction-path",
            str(extraction_path),
            "--manifest-path",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--seed",
            str(DEFAULT_SEED),
            "--d-threshold",
            str(float(setting["delta_merge"])),
        ]
    )


def _validation_skipped() -> dict[str, Any]:
    return {"status": "skipped", "value": None, "reason": "partition_validation_samples is 0"}


def _partition_ra_validity(
    *,
    root: Path,
    region_path: Path,
    sample_count: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if int(sample_count) <= 0:
        return _validation_skipped()
    from evaluation.RQ3 import rq3_eval

    artifacts = rq3_eval._load_region_artifacts([region_path])
    rows = []
    for platform, platform_artifacts in rq3_eval._group_artifacts_by_platform(artifacts).items():
        executor, _manifest_path = rq3_eval._executor_for_platform(
            root=root,
            platform=platform,
            artifacts=platform_artifacts,
            executor_factory_map=None,
        )
        samples = rq3_eval._sample_cases(
            artifacts=platform_artifacts,
            n_val=int(sample_count),
            n_tight=0,
            n_mask=0,
            seed=int(seed),
            boundary_abs_eps=0.01,
            boundary_source="region",
            mask_offset_fraction=0.05,
        )
        rows.extend(
            rq3_eval._run_platform_cases(
                executor=executor,
                samples_by_metric=samples,
                workers=int(workers),
                batch_runner=rq3_eval._runtime_batch_runner,
            )
        )
    summary = rq3_eval._summarize_metric([row for row in rows if row["metric"] == "region_validity"])
    return {
        "status": "artifact",
        "source": str(region_path),
        "seed": int(seed),
        "workers": int(workers),
        **summary,
    }


def _partition_alarm_accuracy(
    *,
    root: Path,
    region_path: Path,
    sample_count: int,
    seed: int,
    workers: int,
    mask_offset_fraction: float,
) -> dict[str, Any]:
    if int(sample_count) <= 0:
        return _validation_skipped()
    from evaluation.RQ3 import rq3_eval

    artifacts = rq3_eval._load_region_artifacts([region_path])
    rows = []
    for platform, platform_artifacts in rq3_eval._group_artifacts_by_platform(artifacts).items():
        executor, _manifest_path = rq3_eval._executor_for_platform(
            root=root,
            platform=platform,
            artifacts=platform_artifacts,
            executor_factory_map=None,
        )
        samples = rq3_eval._sample_cases(
            artifacts=platform_artifacts,
            n_val=0,
            n_tight=0,
            n_mask=int(sample_count),
            seed=int(seed),
            boundary_abs_eps=0.01,
            boundary_source="region",
            mask_offset_fraction=float(mask_offset_fraction),
            mask_sampling_mode="boundary_accuracy",
        )
        rows.extend(
            rq3_eval._run_platform_cases(
                executor=executor,
                samples_by_metric=samples,
                workers=int(workers),
                batch_runner=rq3_eval._runtime_batch_runner,
            )
        )
    summary = rq3_eval._summarize_metric([row for row in rows if row["metric"] == "masking_exclusion"])
    return {
        "status": "artifact",
        "source": str(region_path),
        "seed": int(seed),
        "workers": int(workers),
        **summary,
    }


def _partition_row(
    *,
    setting: Mapping[str, Any],
    payload: Mapping[str, Any],
    source_cell_count: int,
    output_root: Path,
    ra_validity: Mapping[str, Any],
    alarm_accuracy: Mapping[str, Any],
) -> dict[str, Any]:
    regions = list(payload.get("regions", []))
    widths = [_region_width(region) for region in regions if _region_width(region) > 0]
    return {
        **setting,
        "region_count": len(regions),
        "retained_cell_ratio": (len(regions) / source_cell_count) if source_cell_count else None,
        "median_masking_width": statistics.median(widths) if widths else None,
        "vulnerability_volume_proxy": sum(_region_volume_proxy(region) for region in regions),
        "region_validity": dict(ra_validity),
        "ra_validity": dict(ra_validity),
        "ra_validity_rate": ra_validity.get("rate"),
        "ra_validity_n": ra_validity.get("n"),
        "alarm_accuracy": dict(alarm_accuracy),
        "alarm_accuracy_rate": alarm_accuracy.get("rate"),
        "alarm_accuracy_n": alarm_accuracy.get("n"),
        "masking_exclusion": dict(alarm_accuracy),
        "output_root": str(output_root),
        "source_path": str(payload.get("source_path", "")),
    }


def _reuse_representative_regions(
    representative_root: Path,
    setting: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any] | None:
    driver = str(setting["hazard_driver"])
    path = _representative_path(representative_root, driver)
    if not path.exists():
        return None
    from src.stage3 import subregions

    representative = _read_json(path)
    source = "representative_regions"
    if setting["delta_merge"] is None:
        regions = [dict(region) for region in representative.get("regions", []) if region.get("mask_slope_range")]
        source = "representative_regions_no_merge"
    else:
        regions = subregions._merge_regions(representative.get("regions", []), float(setting["delta_merge"]))
    payload = {
        "schema_version": 1,
        "artifact": "rq4_partition_sensitivity_regions",
        "source": source,
        "source_path": str(path),
        "platform": "boiler_ccs",
        "hazard_driver": driver,
        "merge_setting": str(setting["merge_setting"]),
        "delta_merge_factor": setting["delta_merge_factor"],
        "original_delta_merge": float(setting["original_delta_merge"]),
        "delta_merge": None if setting["delta_merge"] is None else float(setting["delta_merge"]),
        "regions": regions,
    }
    for key in ("hazard_id", "alarm_id", "runtime_rule", "seed", "workers"):
        if key in representative:
            payload[key] = representative[key]
    _write_json(
        _partition_region_output_path(output_root, driver),
        payload,
    )
    return payload


def run_partition_sensitivity(
    *,
    root: str | Path,
    rq4_dir: Path,
    stage3_runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
    partition_validation_samples: int = DEFAULT_PARTITION_VALIDATION_SAMPLES,
    partition_alarm_accuracy_samples: int | None = None,
    partition_representative_root: str | Path | None = None,
    partition_drivers: Sequence[str] | None = None,
    partition_merge_settings: Sequence[Mapping[str, Any]] | None = None,
    partition_mask_offset_fraction: float = 0.05,
    workers: int = base_search.DEFAULT_WORKERS,
) -> dict[str, Any]:
    root = Path(root)
    representative_root = Path(partition_representative_root) if partition_representative_root else root
    rows: list[dict[str, Any]] = []
    runner = stage3_runner

    selected_drivers = tuple(partition_drivers) if partition_drivers else BOILER_PARTITION_DRIVERS
    merge_settings = list(partition_merge_settings or PARTITION_MERGE_SETTINGS)
    alarm_accuracy_samples = (
        int(partition_validation_samples)
        if partition_alarm_accuracy_samples is None
        else int(partition_alarm_accuracy_samples)
    )

    for driver in selected_drivers:
        conditional_path = root / "results" / "stage2" / "boiler_ccs" / driver / "conditional_results.json"
        if not conditional_path.exists():
            continue
        widths = _mask_widths_from_conditional(conditional_path)
        if not widths:
            continue
        median_width = statistics.median(widths)
        representative_path = _representative_path(representative_root, driver)
        representative_payload = _read_json(representative_path) if representative_path.exists() else {}
        representative_regions = list(representative_payload.get("regions", []))
        source_cell_count = len([region for region in representative_regions if region.get("mask_slope_range")])
        if not source_cell_count:
            source_cell_count = len(_read_json(conditional_path).get("points", []))
        original_delta_merge = _original_merge_threshold(representative_regions, median_width)
        for merge_spec in merge_settings:
            factor = merge_spec["delta_merge_factor"]
            setting = {
                "platform": "boiler_ccs",
                "hazard_driver": driver,
                "merge_setting": str(merge_spec["merge_setting"]),
                "delta_merge_factor": None if factor is None else float(factor),
                "median_feasible_masking_width": median_width,
                "original_delta_merge": original_delta_merge,
                "delta_merge": None if factor is None else float(factor) * original_delta_merge,
            }
            output_root = rq4_dir / "work" / "partition" / driver / str(merge_spec["merge_setting"])
            reused = _reuse_representative_regions(representative_root, setting, output_root)
            payload = reused if reused is not None else (
                _default_stage3_runner(root, setting, output_root)
                if runner is None
                else dict(runner(setting, output_root))
            )
            region_path = _partition_region_output_path(output_root, driver)
            ra_validity = _partition_ra_validity(
                root=root,
                region_path=region_path,
                sample_count=int(partition_validation_samples),
                seed=DEFAULT_SEED,
                workers=int(workers),
            )
            alarm_accuracy = _partition_alarm_accuracy(
                root=root,
                region_path=region_path,
                sample_count=int(alarm_accuracy_samples),
                seed=DEFAULT_SEED,
                workers=int(workers),
                mask_offset_fraction=float(partition_mask_offset_fraction),
            )
            row = _partition_row(
                setting=setting,
                payload=payload,
                source_cell_count=source_cell_count,
                output_root=output_root,
                ra_validity=ra_validity,
                alarm_accuracy=alarm_accuracy,
            )
            row["source"] = str(payload.get("source", "stage3_runner"))
            rows.append(
                row
            )

    payload = {"schema_version": 1, "rows": rows}
    _write_json(rq4_dir / "rq4_partition_sensitivity.json", payload)
    _write_csv(rq4_dir / "rq4_partition_sensitivity.csv", rows)
    return payload


def _config_payload(
    *,
    run_boundary_sweep: bool,
    run_partition_sweep: bool,
    partition_validation_samples: int,
    partition_alarm_accuracy_samples: int,
    partition_representative_root: str | None,
    partition_drivers: Sequence[str],
    partition_merge_settings: Sequence[Mapping[str, Any]],
    partition_mask_offset_fraction: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rq": "RQ4",
        "default_setting": {
            "Delta_T": 5,
            "epsilon_K": 1e-4,
            "delta_merge": "1.0 * Stage 3 original effective merge threshold",
        },
        "boundary_sensitivity": {
            "enabled": bool(run_boundary_sweep),
            "platform": DEFAULT_BOUNDARY_PLATFORM,
            "hazard_driver": DEFAULT_BOUNDARY_DRIVER,
            "reason": "TE xmv_07 is the RQ2 canonical base-only variable and isolates boundary search.",
            "reference": dict(DEFAULT_BOUNDARY_REFERENCE),
            "sweep": [dict(setting) for setting in BOUNDARY_SWEEP],
        },
        "partition_sensitivity": {
            "enabled": bool(run_partition_sweep),
            "platform": "boiler_ccs",
            "hazard_drivers": list(partition_drivers),
            "merge_settings": [dict(setting) for setting in partition_merge_settings],
            "factor_reference": "Stage 3 original effective merge threshold from representative regions",
            "ra_validity_samples_per_setting": int(partition_validation_samples),
            "alarm_accuracy_samples_per_setting": int(partition_alarm_accuracy_samples),
            "mask_offset_fraction": float(partition_mask_offset_fraction),
            "primary_metric": "alarm_accuracy_rate",
            "representative_root": partition_representative_root,
            "writes_original_stage3_artifacts": False,
        },
    }


def run_rq4_evaluation(
    *,
    root: str | Path,
    rq4_dir: str | Path | None = None,
    run_boundary_sweep: bool = False,
    run_partition_sweep: bool = False,
    partition_validation_samples: int = DEFAULT_PARTITION_VALIDATION_SAMPLES,
    partition_alarm_accuracy_samples: int | None = None,
    partition_representative_root: str | Path | None = None,
    partition_drivers: Sequence[str] | None = None,
    partition_factors: Sequence[float] | None = None,
    partition_mask_offset_fraction: float = 0.05,
    base_runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
    stage3_runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
    workers: int = base_search.DEFAULT_WORKERS,
) -> dict[str, Any]:
    root = Path(root)
    rq4_dir = Path(rq4_dir) if rq4_dir else root / "results" / "rq4"
    resolved_partition_drivers = tuple(partition_drivers) if partition_drivers else BOILER_PARTITION_DRIVERS
    resolved_merge_settings = _partition_merge_settings(partition_factors)
    resolved_alarm_samples = (
        int(partition_validation_samples)
        if partition_alarm_accuracy_samples is None
        else int(partition_alarm_accuracy_samples)
    )
    config = _config_payload(
        run_boundary_sweep=run_boundary_sweep,
        run_partition_sweep=run_partition_sweep,
        partition_validation_samples=int(partition_validation_samples),
        partition_alarm_accuracy_samples=resolved_alarm_samples,
        partition_representative_root=str(partition_representative_root) if partition_representative_root else None,
        partition_drivers=resolved_partition_drivers,
        partition_merge_settings=resolved_merge_settings,
        partition_mask_offset_fraction=float(partition_mask_offset_fraction),
    )
    _write_json(rq4_dir / "rq4_eval_config.json", config)

    result: dict[str, Any] = {
        "config": config,
    }
    if run_boundary_sweep:
        result["boundary_sensitivity"] = run_boundary_sensitivity(
            root=root,
            rq4_dir=rq4_dir,
            base_runner=base_runner,
            workers=workers,
        )
    else:
        result["boundary_sensitivity"] = {"status": "skipped"}
    if run_partition_sweep:
        result["partition_sensitivity"] = run_partition_sensitivity(
            root=root,
            rq4_dir=rq4_dir,
            stage3_runner=stage3_runner,
            partition_validation_samples=int(partition_validation_samples),
            partition_alarm_accuracy_samples=resolved_alarm_samples,
            partition_representative_root=partition_representative_root,
            partition_drivers=resolved_partition_drivers,
            partition_merge_settings=resolved_merge_settings,
            partition_mask_offset_fraction=float(partition_mask_offset_fraction),
            workers=int(workers),
        )
    else:
        result["partition_sensitivity"] = {"status": "skipped"}
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ4 sensitivity evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--rq4-dir")
    parser.add_argument("--run-boundary-sweep", action="store_true")
    parser.add_argument("--run-partition-sweep", action="store_true")
    parser.add_argument("--run-all-sweeps", action="store_true")
    parser.add_argument("--partition-validation-samples", type=int, default=DEFAULT_PARTITION_VALIDATION_SAMPLES)
    parser.add_argument("--partition-alarm-accuracy-samples", type=int)
    parser.add_argument("--partition-driver", action="append", default=[])
    parser.add_argument("--partition-factor", action="append", type=float, default=[])
    parser.add_argument("--partition-mask-offset-fraction", type=float, default=0.05)
    parser.add_argument("--partition-representative-root")
    parser.add_argument("--workers", type=int, default=base_search.DEFAULT_WORKERS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    run_all_sweeps = bool(args.run_all_sweeps)
    return run_rq4_evaluation(
        root=args.root,
        rq4_dir=args.rq4_dir,
        run_boundary_sweep=bool(args.run_boundary_sweep or run_all_sweeps),
        run_partition_sweep=bool(args.run_partition_sweep or run_all_sweeps),
        partition_validation_samples=int(args.partition_validation_samples),
        partition_alarm_accuracy_samples=args.partition_alarm_accuracy_samples,
        partition_representative_root=args.partition_representative_root,
        partition_drivers=args.partition_driver or None,
        partition_factors=args.partition_factor or None,
        partition_mask_offset_fraction=float(args.partition_mask_offset_fraction),
        workers=int(args.workers),
    )


if __name__ == "__main__":
    main()

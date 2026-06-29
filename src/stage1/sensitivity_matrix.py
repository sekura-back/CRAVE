from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from platforms.variable_mapping import build_platform_variable_map
from stage1.output_layout import stage1_artifact_dir


_PLATFORM_DIRS = {
    "boiler": "boiler_ccs",
    "boiler_ccs": "boiler_ccs",
    "te": "tennessee_eastman",
    "tennessee_eastman": "tennessee_eastman",
}

_RUNTIME_EXECUTOR_MODULES = {
    "boiler_ccs": "simulators.boiler_ccs.runtime_executor",
    "tennessee_eastman": "simulators.tennessee_eastman.runtime_executor",
}


def _normalize_platform(platform: str) -> str:
    value = str(platform).strip().lower()
    try:
        return _PLATFORM_DIRS[value]
    except KeyError as exc:
        raise ValueError(f"unsupported platform: {platform}") from exc


def _platform_dir(root: Path, platform_name: str) -> Path:
    return Path(root) / "simulators" / platform_name


def _build_executor(root: Path, platform_name: str) -> Any:
    root_path = Path(root).resolve()
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    module = importlib.import_module(_RUNTIME_EXECUTOR_MODULES[platform_name])
    manifest_path = _platform_dir(root_path, platform_name) / "system_manifest.json"
    return module.create_executor(manifest_path)


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _actuator_rows(variable_map: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in variable_map:
        layer = str(record.get("layer", "")).strip().lower()
        roles = {str(role).strip().lower() for role in record.get("roles", [])}
        if layer != "actuator" or "injectable" not in roles:
            continue
        canonical_name = str(record.get("canonical_name", "")).strip()
        manifest_name = str(record.get("manifest_name", "")).strip()
        if not canonical_name or not manifest_name:
            raise ValueError(f"actuator row missing canonical or manifest name: {record}")
        rows.append(
            {
                "canonical_name": canonical_name,
                "manifest_name": manifest_name,
            }
        )
    return rows


def _hazard_columns(
    extraction: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, str]]]:
    cols: list[str] = []
    hazard_columns: dict[str, dict[str, str]] = {}
    for rule in extraction.get("H", []):
        hazard_id = str(rule.get("id", "")).strip()
        hazard_var = str(rule.get("var", "")).strip()
        jacobian_col = str(rule.get("jacobian_col") or hazard_var).strip()
        jacobian_kind = str(rule.get("jacobian_kind", "state")).strip().lower() or "state"
        if not hazard_id or not jacobian_col:
            continue
        cols.append(jacobian_col)
        hazard_columns[hazard_id] = {
            "col": jacobian_col,
            "kind": jacobian_kind,
            "var": hazard_var,
        }
    return _ordered_unique(cols), hazard_columns


def _requested_amp(spec: Mapping[str, Any]) -> float:
    rate = abs(float(spec.get("rate", 0.0)))
    if rate > 0.0:
        return rate
    value_range = spec.get("range")
    if isinstance(value_range, list) and len(value_range) == 2:
        lo, hi = value_range
        if lo is not None and hi is not None:
            width = abs(float(hi) - float(lo))
            if width > 0.0:
                return width / 100.0
    return 1.0


def _last_row(result: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = result.get("rows", [])
    if not rows:
        raise ValueError("hazard jacobian run returned empty trace")
    return rows[-1]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def build_hazard_jacobian(root, platform, extraction, duration=10):
    if int(duration) < 2:
        raise ValueError("hazard jacobian duration must be at least 2")

    root_path = Path(root)
    platform_name = _normalize_platform(str(platform))
    variable_map = build_platform_variable_map(root_path, platform_name)
    row_records = _actuator_rows(variable_map)
    cols, hazard_columns = _hazard_columns(extraction)
    executor = _build_executor(root_path, platform_name)
    total_steps = int(duration) + 100

    baseline = executor(
        {
            "injection": {"points": []},
            "runtime_args": {
                "steps": total_steps,
                "stop_on_trip": False,
            },
        }
    )
    baseline_last = _last_row(baseline)

    matrix: dict[str, dict[str, float]] = {}
    for row in row_records:
        canonical_name = row["canonical_name"]
        manifest_name = row["manifest_name"]
        spec = executor.registry[manifest_name]
        requested_amp = _requested_amp(spec)
        perturbed = executor(
            {
                "injection": {
                    "points": [
                        {
                            "point": manifest_name,
                            "amp": requested_amp,
                            "duration": int(duration),
                            "t_start": 0,
                            "role": "manip",
                        }
                    ]
                },
                "runtime_args": {
                    "steps": total_steps,
                    "stop_on_trip": False,
                },
            }
        )
        perturbed_last = _last_row(perturbed)
        matrix[canonical_name] = {
            col: (
                float(perturbed_last[col]) - float(baseline_last[col])
            )
            / requested_amp
            for col in cols
        }

    return {
        "platform": platform_name,
        "runtime_rule": "duration + 100",
        "duration": int(duration),
        "total_steps": total_steps,
        "rows": [row["canonical_name"] for row in row_records],
        "cols": cols,
        "hazard_columns": hazard_columns,
        "matrix": matrix,
    }


def write_hazard_jacobian_output(
    *,
    hazard_jacobian: Mapping[str, Any],
    output_root: Path,
    platform: str,
    seed: int,
) -> dict[str, Path]:
    platform_name = _normalize_platform(str(platform))
    artifact_dir = stage1_artifact_dir(platform)
    root = Path(output_root)
    artifact_path = root / "results" / "stage1" / artifact_dir / "sensitivity_matrix.json"
    log_path = root / "results" / "logs" / "stage1" / platform_name / "events.jsonl"
    manifest_path = (
        root
        / "results"
        / "manifests"
        / "stage1"
        / platform_name
        / "sensitivity_matrix_run_manifest.json"
    )
    artifact = {"seed": int(seed), **dict(hazard_jacobian)}
    _write_json(artifact_path, artifact)
    manifest = {
        "stage": "stage1",
        "artifact_kind": "hazard_jacobian",
        "platform": platform_name,
        "seed": int(seed),
        "artifact": str(artifact_path),
        "log": str(log_path),
    }
    _write_json(manifest_path, manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "stage": "stage1",
        "artifact_kind": "hazard_jacobian",
        "platform": platform_name,
        "seed": int(seed),
        "event": "stage1_hazard_jacobian_written",
        "message": "stage1 hazard jacobian artifact written",
        "artifact": str(artifact_path),
        "manifest": str(manifest_path),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return {"artifact": artifact_path, "log": log_path, "manifest": manifest_path}

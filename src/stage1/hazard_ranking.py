from __future__ import annotations

import heapq
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .output_layout import stage1_artifact_dir
from .sensitivity_matrix import (
    _normalize_platform,
    build_hazard_jacobian,
    write_hazard_jacobian_output,
)


_TE_LOOP_RE = re.compile(r"L(\d+)(?:->L(\d+))?")
_HFOP_OUTPUT_RE = re.compile(r"hfop(\d+)_output")


def _record_names(record: Mapping[str, Any]) -> list[str]:
    names = [
        str(record.get("canonical_name", "")).strip(),
        str(record.get("manifest_name", "")).strip(),
        str(record.get("runtime_field", "")).strip(),
    ]
    names.extend(str(alias).strip() for alias in record.get("aliases", []) if str(alias).strip())
    return [name for name in names if name]


def _graph_nodes(edges: list[Mapping[str, Any]]) -> set[str]:
    nodes: set[str] = set()
    for edge in edges:
        source = str(edge.get("x", "")).strip()
        target = str(edge.get("y", "")).strip()
        if source:
            nodes.add(source)
        if target:
            nodes.add(target)
    return nodes


def _is_public_setpoint_name(name: str) -> bool:
    value = str(name).strip().lower()
    return value.endswith("_sp") or value.endswith("_setpoint") or "setpoint" in value


def _controller_loop_ids(controller: str) -> tuple[str, str]:
    match = _TE_LOOP_RE.search(str(controller))
    if not match:
        return "", ""
    return str(match.group(1) or ""), str(match.group(2) or "")


def _controller_base_name(controller: str) -> str:
    value = str(controller).strip()
    return value.split("(", 1)[0].strip()


def _dedupe_names(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _controller_role_candidates(record: Mapping[str, Any]) -> list[str]:
    canonical_name = str(record.get("canonical_name", "")).strip().lower()
    description = str(record.get("description", "")).strip().lower()
    candidates: list[str] = []
    hfop_match = _HFOP_OUTPUT_RE.search(canonical_name)
    if hfop_match is not None:
        index = hfop_match.group(1)
        candidates.extend([f"hfop{index}.av_0", f"h{index}"])
    if canonical_name.endswith("_primary_output") or "primary pi block output" in description:
        candidates.append("dxmv")
    if canonical_name.endswith("_rate_output") or "rate limiter output" in description or "速率限制器输出" in description:
        candidates.extend(["rate_out", "rate.prev_output", "rate_limiter.prev_output", "hslim"])
    if canonical_name.endswith("_pid_output") or "pid output" in description or "pid输出" in description:
        candidates.extend(["pid_out", "boiler_cmd"])
    if "pv_filtered" in canonical_name:
        candidates.extend(["pv_filtered", "hfop1.av_0"])
    if canonical_name.endswith("_filter_output") or "filter output" in description or "滤波器输出" in description:
        candidates.extend(["fuel_cmd", "pump_speed", "water_sp", "hfop1.av_0", "hfop.av_0"])
    return _dedupe_names(candidates)


def _shared_controller_records(
    writable_records: list[Mapping[str, Any]],
    *,
    controller: str,
    layer: str,
    canonical_name: str,
) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    for record in writable_records:
        if str(record.get("canonical_name", "")).strip() == canonical_name:
            continue
        if str(record.get("layer", "")).strip().lower() != layer:
            continue
        if str(record.get("controller", "")).strip() != controller:
            continue
        matches.append(record)
    return matches


def _candidate_graph_anchors(
    record: Mapping[str, Any],
    writable_records: list[Mapping[str, Any]],
) -> list[str]:
    canonical_name = str(record.get("canonical_name", "")).strip()
    lower_name = canonical_name.lower()
    layer = str(record.get("layer", "")).strip().lower()
    controller = str(record.get("controller", "")).strip()
    candidates = list(_record_names(record))
    if canonical_name.endswith("_ctrl_output"):
        candidates.append(canonical_name[: -len("_ctrl_output")])
    controller_loop, target_loop = _controller_loop_ids(controller)
    if layer == "setpoint" and controller_loop:
        candidates.append(f"DecentralizedController.ctrl{controller_loop}.setpoint")
    if canonical_name.endswith("_primary_output"):
        loop = controller_loop
        if not loop:
            match = re.search(r"ctrl(\d+)_primary_output", canonical_name)
            loop = str(match.group(1)) if match is not None else ""
        if loop:
            candidates.append(f"DecentralizedController.ctrl{loop}.dxmv")
    if canonical_name.endswith("_scaled_output"):
        candidates.extend(
            name
            for peer in _shared_controller_records(
                writable_records,
                controller=controller,
                layer="setpoint",
                canonical_name=canonical_name,
            )
            for name in _record_names(peer)
        )
        if target_loop:
            candidates.append(f"DecentralizedController.ctrl{target_loop}.setpoint")
    if layer == "controller_output":
        if not (
            lower_name.endswith("_ctrl_output")
            or lower_name.endswith("_primary_output")
            or lower_name.endswith("_scaled_output")
        ):
            candidates.extend(
                name
                for peer in _shared_controller_records(
                    writable_records,
                    controller=controller,
                    layer="setpoint",
                    canonical_name=canonical_name,
                )
                for name in _record_names(peer)
            )
        candidates.extend(
            name
            for peer in _shared_controller_records(
                writable_records,
                controller=controller,
                layer="actuator",
                canonical_name=canonical_name,
            )
            for name in _record_names(peer)
        )
        base_name = _controller_base_name(controller)
        if base_name:
            candidates.extend(
                f"{base_name}.{suffix}"
                for suffix in _controller_role_candidates(record)
            )
    return _dedupe_names(candidates)


def _resolve_graph_anchor(
    record: Mapping[str, Any],
    writable_records: list[Mapping[str, Any]],
    graph_nodes: set[str],
    row_names: set[str],
) -> str:
    for candidate in _candidate_graph_anchors(record, writable_records):
        if candidate in row_names or candidate in graph_nodes:
            return candidate
    return ""


def _edge_weight(edge: Mapping[str, Any]) -> float:
    source = str(edge.get("x", "")).strip()
    target = str(edge.get("y", "")).strip()
    kind = str(edge.get("kind", "")).strip().lower()
    if kind and kind != "main_control_dependency":
        return 0.60
    joined = f"{source} {target}".lower()
    if source.endswith(".dxmv") and _is_public_setpoint_name(target):
        return 0.90
    if source.endswith(".setpoint") or target.endswith(".setpoint") or _is_public_setpoint_name(source):
        return 0.95
    if any(token in joined for token in ("hfop", "rate", "pid", "av_0", "prev_output", "err_old", "delta_err", "dk_")):
        return 0.80
    return 0.95


def _reverse_graph(edges: list[Mapping[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    reverse: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        source = str(edge.get("x", "")).strip()
        target = str(edge.get("y", "")).strip()
        if not source or not target:
            continue
        reverse.setdefault(target, []).append((source, _edge_weight(edge)))
    return reverse


def _best_control_scores(
    reverse_graph: Mapping[str, list[tuple[str, float]]],
    bottom_interface: str,
) -> dict[str, float]:
    best = {str(bottom_interface): 1.0}
    queue: list[tuple[float, str]] = [(-1.0, str(bottom_interface))]
    while queue:
        neg_score, node = heapq.heappop(queue)
        score = -neg_score
        if score < best.get(node, 0.0):
            continue
        for upstream, weight in reverse_graph.get(node, []):
            candidate = score * float(weight)
            if candidate <= best.get(upstream, 0.0):
                continue
            best[upstream] = candidate
            heapq.heappush(queue, (-candidate, upstream))
    return best


def _candidate_representative(
    items: list[Mapping[str, Any]],
    bottom_interface: str,
) -> dict[str, Any]:
    for item in items:
        if str(item.get("variable", "")).strip() == str(bottom_interface).strip():
            return dict(item)
    return dict(items[0]) if items else {}


def _candidate_outputs(
    rankings: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidate_rankings: dict[str, list[dict[str, Any]]] = {}
    candidate_variables: list[str] = []
    for hazard_id, ordered in rankings.items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in ordered:
            bottom_interface = str(item.get("bottom_interface", "")).strip()
            if not bottom_interface:
                continue
            grouped.setdefault(bottom_interface, []).append(item)
        candidates: list[dict[str, Any]] = []
        for bottom_interface, items in grouped.items():
            candidate = _candidate_representative(items, bottom_interface)
            if not candidate:
                continue
            candidate["rank"] = 0
            candidate["group_size"] = len(items)
            candidate["group_variables"] = [
                str(item.get("variable", "")).strip()
                for item in items
                if str(item.get("variable", "")).strip()
            ]
            candidate["source_rank"] = min(int(item.get("rank", 0)) for item in items)
            candidates.append(candidate)
        trimmed = candidates[:3]
        for index, item in enumerate(trimmed, start=1):
            item["rank"] = index
            candidate_variables.append(str(item.get("bottom_interface", "")).strip())
        candidate_rankings[str(hazard_id)] = trimmed
    return {
        "candidate_rankings": candidate_rankings,
        "candidate_variables": _dedupe_names(candidate_variables),
    }


def _fallback_rank_hazard_drivers(
    extraction: Mapping[str, Any],
    hazard_jacobian: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [str(row).strip() for row in hazard_jacobian.get("rows", []) if str(row).strip()]
    hazard_columns = hazard_jacobian.get("hazard_columns", {})
    matrix = hazard_jacobian.get("matrix", {})
    rankings: dict[str, list[dict[str, Any]]] = {}

    for hazard in extraction.get("H", []):
        hazard_id = str(hazard.get("id", "")).strip()
        hazard_var = str(hazard.get("var", "")).strip()
        direction = str(hazard.get("direction", "upper")).strip().lower()
        hazard_column = hazard_columns.get(hazard_id, {})
        jacobian_col = str(hazard_column.get("col") or hazard.get("jacobian_col") or hazard_var).strip()
        jacobian_kind = str(
            hazard_column.get("kind") or hazard.get("jacobian_kind", "state")
        ).strip().lower() or "state"
        ordered: list[dict[str, Any]] = []
        for row in rows:
            derivative = float(matrix[row][jacobian_col])
            input_direction = _input_direction_for_hazard(
                hazard_direction=direction,
                derivative=derivative,
                jacobian_kind=jacobian_kind,
            )
            ordered.append(
                {
                    "rank": 0,
                    "variable": row,
                    "hazard_var": hazard_var,
                    "jacobian_col": jacobian_col,
                    "jacobian_kind": jacobian_kind,
                    "direction": direction,
                    "derivative": derivative,
                    "input_direction": input_direction,
                    "score": abs(derivative),
                    "control_score": 1.0,
                    "physical_sensitivity": abs(derivative),
                    "bottom_interface": row,
                    "graph_anchor": row,
                }
            )
        ordered.sort(key=lambda item: item["score"], reverse=True)
        for index, item in enumerate(ordered, start=1):
            item["rank"] = index
        rankings[hazard_id] = ordered

    return {
        "platform": str(hazard_jacobian.get("platform", "")).strip(),
        "rankings": rankings,
        **_candidate_outputs(rankings),
    }


def _input_direction_for_hazard(
    *,
    hazard_direction: str,
    derivative: float,
    jacobian_kind: str,
) -> str:
    direction = str(hazard_direction).strip().lower()
    kind = str(jacobian_kind).strip().lower() or "state"
    if derivative == 0.0:
        return "neutral"
    if kind == "margin":
        return "increase" if derivative < 0.0 else "decrease"
    if direction == "upper":
        return "increase" if derivative > 0.0 else "decrease"
    if direction == "lower":
        return "decrease" if derivative > 0.0 else "increase"
    raise ValueError(f"unsupported hazard direction: {hazard_direction}")


def rank_hazard_drivers(
    extraction: Mapping[str, Any],
    hazard_jacobian: Mapping[str, Any],
) -> dict[str, Any]:
    writable_records = [
        record
        for record in extraction.get("W", [])
        if isinstance(record, Mapping)
        and str(record.get("canonical_name", "")).strip()
    ]
    if not writable_records or not extraction.get("G", {}).get("E"):
        return _fallback_rank_hazard_drivers(extraction, hazard_jacobian)

    rows = [str(row).strip() for row in hazard_jacobian.get("rows", []) if str(row).strip()]
    row_names = {row for row in rows if row}
    hazard_columns = hazard_jacobian.get("hazard_columns", {})
    matrix = hazard_jacobian.get("matrix", {})
    edges = [
        edge
        for edge in extraction.get("G", {}).get("E", [])
        if isinstance(edge, Mapping)
    ]
    graph_nodes = _graph_nodes(edges) | row_names
    reverse_graph = _reverse_graph(edges)
    best_control_scores = {
        row: _best_control_scores(reverse_graph, row)
        for row in rows
    }
    seen_variables = {
        str(record.get("canonical_name", "")).strip()
        for record in writable_records
    }
    for row in rows:
        if row in seen_variables:
            continue
        writable_records.append(
            {
                "canonical_name": row,
                "manifest_name": row,
                "runtime_field": row,
                "aliases": [],
                "layer": "actuator",
            }
        )
    graph_anchors = {
        str(record.get("canonical_name", "")).strip(): _resolve_graph_anchor(
            record,
            writable_records,
            graph_nodes,
            row_names,
        )
        for record in writable_records
    }
    rankings: dict[str, list[dict[str, Any]]] = {}

    for hazard in extraction.get("H", []):
        hazard_id = str(hazard.get("id", "")).strip()
        hazard_var = str(hazard.get("var", "")).strip()
        direction = str(hazard.get("direction", "upper")).strip().lower()
        hazard_column = hazard_columns.get(hazard_id, {})
        jacobian_col = str(hazard_column.get("col") or hazard.get("jacobian_col") or hazard_var).strip()
        jacobian_kind = str(
            hazard_column.get("kind") or hazard.get("jacobian_kind", "state")
        ).strip().lower() or "state"
        ordered: list[dict[str, Any]] = []
        for record in writable_records:
            variable = str(record.get("canonical_name", "")).strip()
            anchor = graph_anchors.get(variable, "")
            best_row = ""
            best_derivative = 0.0
            best_score = 0.0
            best_control_score = 0.0
            best_physical_sensitivity = 0.0
            for row in rows:
                derivative = float(matrix[row][jacobian_col])
                physical_sensitivity = abs(derivative)
                control_score = best_control_scores.get(row, {}).get(anchor, 0.0) if anchor else 0.0
                score = control_score * physical_sensitivity
                if score < best_score:
                    continue
                if score == best_score and best_row and row >= best_row:
                    continue
                best_row = row
                best_derivative = derivative
                best_score = score
                best_control_score = control_score
                best_physical_sensitivity = physical_sensitivity
            if not anchor or best_control_score == 0.0:
                best_row = ""
                best_derivative = 0.0
                best_score = 0.0
                best_physical_sensitivity = 0.0
            input_direction = _input_direction_for_hazard(
                hazard_direction=direction,
                derivative=best_derivative,
                jacobian_kind=jacobian_kind,
            )
            ordered.append(
                {
                    "rank": 0,
                    "variable": variable,
                    "hazard_var": hazard_var,
                    "jacobian_col": jacobian_col,
                    "jacobian_kind": jacobian_kind,
                    "direction": direction,
                    "derivative": best_derivative,
                    "input_direction": input_direction,
                    "score": best_score,
                    "control_score": best_control_score,
                    "physical_sensitivity": best_physical_sensitivity,
                    "bottom_interface": best_row,
                    "graph_anchor": anchor,
                }
            )
        ordered.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item["control_score"]),
                -float(item["physical_sensitivity"]),
                str(item["variable"]),
            )
        )
        for index, item in enumerate(ordered, start=1):
            item["rank"] = index
        rankings[hazard_id] = ordered

    return {
        "platform": str(hazard_jacobian.get("platform", "")).strip(),
        "rankings": rankings,
        **_candidate_outputs(rankings),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def write_hazard_ranking_output(
    *,
    hazard_ranking: Mapping[str, Any],
    output_root: Path,
    platform: str,
    seed: int,
) -> dict[str, Path]:
    platform_name = _normalize_platform(str(platform))
    artifact_dir = stage1_artifact_dir(platform)
    root = Path(output_root)
    artifact_path = root / "results" / "stage1" / artifact_dir / "hazard_ranking.json"
    log_path = root / "results" / "logs" / "stage1" / platform_name / "events.jsonl"
    manifest_path = (
        root
        / "results"
        / "manifests"
        / "stage1"
        / platform_name
        / "hazard_ranking_run_manifest.json"
    )
    artifact = {"seed": int(seed), **dict(hazard_ranking)}
    _write_json(artifact_path, artifact)
    manifest = {
        "stage": "stage1",
        "artifact_kind": "hazard_ranking",
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
        "artifact_kind": "hazard_ranking",
        "platform": platform_name,
        "seed": int(seed),
        "event": "stage1_hazard_ranking_written",
        "message": "stage1 hazard ranking artifact written",
        "artifact": str(artifact_path),
        "manifest": str(manifest_path),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return {"artifact": artifact_path, "log": log_path, "manifest": manifest_path}


def run_stage1_hazard_ranking(
    *,
    root: Path,
    platform: str,
    extraction: Mapping[str, Any],
    output_root: Path,
    seed: int,
    duration: int = 10,
) -> dict[str, Path]:
    hazard_jacobian = build_hazard_jacobian(root, platform, extraction, duration=duration)
    jacobian_paths = write_hazard_jacobian_output(
        hazard_jacobian=hazard_jacobian,
        output_root=output_root,
        platform=platform,
        seed=seed,
    )
    hazard_ranking = rank_hazard_drivers(extraction, hazard_jacobian)
    ranking_paths = write_hazard_ranking_output(
        hazard_ranking=hazard_ranking,
        output_root=output_root,
        platform=platform,
        seed=seed,
    )
    return {
        "jacobian_artifact": jacobian_paths["artifact"],
        "jacobian_log": jacobian_paths["log"],
        "jacobian_manifest": jacobian_paths["manifest"],
        "ranking_artifact": ranking_paths["artifact"],
        "ranking_log": ranking_paths["log"],
        "ranking_manifest": ranking_paths["manifest"],
    }

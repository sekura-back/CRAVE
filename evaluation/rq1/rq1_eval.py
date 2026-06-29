from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260627
HOLD_STEPS = 100
RUNTIME_RULE = "duration + 100"
DEFAULT_DURATION_VALUES = (100, 300, 600, 1000)
DEFAULT_SCAN_RATES = (0.25, 0.5, 1.0)
DEFAULT_DIRECTIONS = ("pos", "neg")
DEFAULT_RANK_TOP_K = 3

PLATFORM_CASES = {
    "boiler_ccs": {
        "display_name": "Boiler CCS",
        "stage1_dir": "boilerCCS",
        "short_name": "boiler",
        "manifest_platform": "boiler_ccs",
    },
    "tennessee_eastman": {
        "display_name": "TE",
        "stage1_dir": "TE",
        "short_name": "te",
        "manifest_platform": "tennessee_eastman",
    },
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in text.splitlines():
            handle.write(line + "\n")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in text.splitlines():
            handle.write(line + "\n")


def _stage1_paths(root: Path, platform: str) -> dict[str, Path]:
    case = PLATFORM_CASES[platform]
    stage1_dir = root / "results" / "stage1" / case["stage1_dir"]
    return {
        "extraction": stage1_dir / "extraction.json",
        "sensitivity": stage1_dir / "sensitivity_matrix.json",
        "ranking": stage1_dir / "hazard_ranking.json",
    }


def _load_stage1_inputs(root: Path, platforms: Sequence[str]) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for platform in platforms:
        paths = _stage1_paths(root, platform)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing Stage1 artifact(s) for {platform}: {missing}")
        inputs[platform] = {
            "paths": paths,
            "extraction": _read_json(paths["extraction"]),
            "sensitivity": _read_json(paths["sensitivity"]),
            "ranking": _read_json(paths["ranking"]),
        }
    return inputs


def _count_layers(w_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in w_rows:
        counts[str(row.get("layer", "unknown"))] += 1
    return dict(sorted(counts.items()))


def _sensitivity_shape(payload: Mapping[str, Any]) -> dict[str, int]:
    return {
        "rows": len(payload.get("rows", [])),
        "cols": len(payload.get("cols", [])),
    }


def build_stage1_counts(stage1_inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    platforms: dict[str, Any] = {}
    for platform, bundle in stage1_inputs.items():
        extraction = bundle["extraction"]
        ranking = bundle["ranking"]
        w_rows = list(extraction.get("W", []))
        platforms[platform] = {
            "display_name": PLATFORM_CASES[platform]["display_name"],
            "seed": extraction.get("seed", ranking.get("seed")),
            "D": len(extraction.get("D", {}).get("variables", [])),
            "G": len(extraction.get("G", {}).get("E", [])),
            "P": len(extraction.get("P", [])),
            "H": len(extraction.get("H", [])),
            "W": len(w_rows),
            "W_by_layer": _count_layers(w_rows),
            "sensitivity_matrix": _sensitivity_shape(bundle["sensitivity"]),
            "W_rank": len(ranking.get("candidate_variables", [])),
            "candidate_variables": list(ranking.get("candidate_variables", [])),
        }
    return {
        "schema_version": 1,
        "rq": "RQ1",
        "runtime_rule": RUNTIME_RULE,
        "platforms": platforms,
    }


def _ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.replace(",", "|").split("|") if part.strip()}
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    return {text} if text else set()


def _rules(probe: Mapping[str, Any], group: str) -> Mapping[str, Any]:
    rules = probe.get("_rules", {})
    if not isinstance(rules, Mapping):
        return {}
    value = rules.get(group, {})
    return value if isinstance(value, Mapping) else {}


def _first_hazard(probe: Mapping[str, Any]) -> tuple[int | None, list[str]]:
    rows = list(probe.get("rows", []) or [])
    for index, row in enumerate(rows):
        ids = sorted(rule_id for rule_id in _ids(row.get("hazard_rule_ids", [])) if rule_id.startswith("H-"))
        if ids:
            return index, ids
    candidates: list[tuple[int, str]] = []
    for rule_id, spec in _rules(probe, "hazards").items():
        margin_col = spec.get("margin_col") if isinstance(spec, Mapping) else None
        if not margin_col:
            continue
        for index, row in enumerate(rows):
            try:
                if float(row[margin_col]) <= 0.0:
                    candidates.append((index, str(rule_id)))
                    break
            except (KeyError, TypeError, ValueError):
                pass
    if candidates:
        first_step = min(index for index, _rule_id in candidates)
        return first_step, sorted(rule_id for index, rule_id in candidates if index == first_step)
    score = probe.get("score", {})
    if isinstance(score, Mapping) and score.get("first_hazard_step") is not None:
        return int(score["first_hazard_step"]), []
    return None, []


def _first_alarm_step(probe: Mapping[str, Any]) -> int | None:
    rows = list(probe.get("rows", []) or [])
    for index, row in enumerate(rows):
        if _ids(row.get("alarm_rule_ids", [])):
            return index
    candidates: list[int] = []
    for spec in _rules(probe, "alarms").values():
        margin_col = spec.get("margin_col") if isinstance(spec, Mapping) else None
        if not margin_col:
            continue
        for index, row in enumerate(rows):
            try:
                if float(row[margin_col]) <= 0.0:
                    candidates.append(index)
                    break
            except (KeyError, TypeError, ValueError):
                pass
    if candidates:
        return min(candidates)
    score = probe.get("score", {})
    if isinstance(score, Mapping) and score.get("first_alarm_step") is not None:
        return int(score["first_alarm_step"])
    return None


def _prehazard_alarm_ids(probe: Mapping[str, Any], hazard_step: int | None) -> set[str]:
    ids: set[str] = set()
    if hazard_step is not None:
        for row in list(probe.get("rows", []) or [])[: int(hazard_step)]:
            ids.update(rule_id for rule_id in _ids(row.get("alarm_rule_ids", [])) if rule_id.startswith("A-"))
        for rule_id, spec in _rules(probe, "alarms").items():
            margin_col = spec.get("margin_col") if isinstance(spec, Mapping) else None
            if not margin_col:
                continue
            for row in list(probe.get("rows", []) or [])[: int(hazard_step)]:
                try:
                    if float(row[margin_col]) <= 0.0:
                        ids.add(str(rule_id))
                        break
                except (KeyError, TypeError, ValueError):
                    pass
    score = probe.get("score", {})
    if isinstance(score, Mapping):
        ids.update(
            rule_id
            for rule_id in _ids(score.get("prehazard_alarm_rule_ids", []))
            if rule_id.startswith("A-")
        )
    return ids


def _classify_w_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    hazard_step, hazard_ids = _first_hazard(probe)
    first_alarm = _first_alarm_step(probe)
    prehazard = _prehazard_alarm_ids(probe, hazard_step)
    return {
        "hazard_reached": hazard_step is not None,
        "target_hazard": "|".join(hazard_ids),
        "first_hazard_time": hazard_step,
        "pre_hazard_alarm_count": len(prehazard),
        "pre_hazard_alarm_ids": sorted(prehazard),
        "first_alarm_time": first_alarm,
    }


def _manifest_path(root: Path, platform: str) -> Path:
    path = root / "simulators" / PLATFORM_CASES[platform]["manifest_platform"] / "system_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _default_executor(manifest_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module = importlib.import_module(f"simulators.{manifest_path.parent.name}.runtime_executor")
    return module.create_executor(manifest_path)


def _signed_rate(direction: str, rate: float) -> float:
    if direction == "pos":
        return float(rate)
    if direction == "neg":
        return -float(rate)
    raise ValueError(f"unsupported direction: {direction}")


def _probe_task(point: str, duration: int, amp: float) -> dict[str, Any]:
    return {
        "runtime_args": {"steps": int(duration) + HOLD_STEPS},
        "injection": {
            "point": str(point),
            "amp": float(amp),
            "duration": int(duration),
            "t_start": 0,
            "role": "manip",
        },
    }


def _writable_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("canonical_name", "")).strip()
    if not value:
        raise ValueError(f"W row missing canonical_name: {row}")
    return value


def _hazard_reached(row: Mapping[str, Any]) -> bool:
    return row.get("hazard_reached") is True or str(row.get("hazard_reached", "")).lower() == "yes"


def _target_hazard(row: Mapping[str, Any]) -> str:
    return str(row.get("target_hazard", "")).strip()


def _alarm_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(_ids(row.get("pre_hazard_alarm_ids", []))))


def _alarm_signatures_by_target(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[tuple[str, ...]]]:
    signatures: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        target = _target_hazard(row)
        signature = _alarm_signature(row)
        if _hazard_reached(row) and target and signature:
            signatures[target].add(signature)
    return signatures


def _stable_alarm_targets(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    return {
        target: next(iter(signatures))
        for target, signatures in _alarm_signatures_by_target(rows).items()
        if len(signatures) == 1
    }


def _best_hit(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    stable_targets = _stable_alarm_targets(rows)
    hits = [
        row
        for row in rows
        if _hazard_reached(row)
        and _target_hazard(row) in stable_targets
        and _alarm_signature(row) == stable_targets[_target_hazard(row)]
    ]
    if not hits:
        return None
    return min(
        hits,
        key=lambda row: (
            int(row["first_hazard_time"]) if row.get("first_hazard_time") not in ("", None) else 10**9,
            int(row["duration"]),
            abs(float(row["amp"])),
        ),
    )


def _best_hazard(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    hits = [row for row in rows if _hazard_reached(row)]
    if not hits:
        return None
    return min(
        hits,
        key=lambda row: (
            int(row["first_hazard_time"]) if row.get("first_hazard_time") not in ("", None) else 10**9,
            int(row["duration"]),
            abs(float(row["amp"])),
        ),
    )


def _alarm_ids_text(hit: Mapping[str, Any] | None) -> str:
    if hit is None:
        return ""
    return "|".join(str(item) for item in hit.get("pre_hazard_alarm_ids", []) or [])


def _alarm_signature_status(
    rows: Sequence[Mapping[str, Any]],
    hazard_hit: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if hazard_hit is None:
        return "", ""
    target = _target_hazard(hazard_hit)
    signatures = _alarm_signatures_by_target(rows).get(target, set())
    if not signatures:
        return "", ""
    return ("yes" if len(signatures) == 1 else "no", str(len(signatures)))


def _best_replay_parameter(hit: Mapping[str, Any] | None) -> str:
    if hit is None:
        return ""
    return json.dumps(
        {
            "direction": hit["direction"],
            "duration": hit["duration"],
            "amp": hit["amp"],
        },
        sort_keys=True,
    )


def _selected_variables(root: Path, platform: str) -> set[str]:
    paths = sorted((root / "results" / "stage2" / platform).glob("*/boundary_results.json"))
    selected = {path.parent.name for path in paths}
    stage3_aliases = {platform, PLATFORM_CASES[platform]["stage1_dir"]}
    upper_paths = sorted((root / "results" / "stage3").glob("*/*/coordinated_regions.json"))
    for path in upper_paths:
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("platform", "")).lower() in {item.lower() for item in stage3_aliases}:
            selected.update(str(region.get("hazard_driver", "")) for region in payload.get("regions", []))
    return {item for item in selected if item}


def _ranking_top_k_variables(ranking: Mapping[str, Any], top_k: int) -> list[str]:
    variables: list[str] = []
    rankings = ranking.get("candidate_rankings", {})
    if not isinstance(rankings, Mapping):
        return variables
    for group in rankings.values():
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            continue
        for entry in list(group)[: int(top_k)]:
            if not isinstance(entry, Mapping):
                continue
            variable = str(entry.get("variable", "")).strip()
            if variable:
                variables.append(variable)
    return _unique_ordered(variables)


def _unique_ordered(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _search_variables(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        str(row.get("writable_id", "")).strip()
        for row in rows
        if row.get("in_W_hd") == "yes" and row.get("selected_for_region_search") == "yes"
    )


def _recompute_w_hd_from_probes(payload: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    probes_by_point: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in updated.get("probe_rows", []) or []:
        probes_by_point[str(row.get("writable_id", ""))].append(row)

    rows: list[dict[str, Any]] = []
    for raw_row in updated.get("rows", []) or []:
        row = dict(raw_row)
        point = str(row.get("writable_id", ""))
        probes = probes_by_point.get(point, [])
        hazard_hit = _best_hazard(probes)
        w_hd_hit = _best_hit(probes)
        signature_stable, signature_count = _alarm_signature_status(probes, hazard_hit)
        row["hazard_reached"] = "yes" if hazard_hit else "no"
        row["target_hazard"] = "" if hazard_hit is None else hazard_hit.get("target_hazard", "")
        row["first_hazard_time"] = "" if hazard_hit is None else hazard_hit.get("first_hazard_time", "")
        row["pre_hazard_alarm_count"] = "" if hazard_hit is None else hazard_hit.get("pre_hazard_alarm_count", "")
        row["pre_hazard_alarm_ids"] = _alarm_ids_text(hazard_hit)
        row["alarm_signature_stable"] = signature_stable
        row["alarm_signature_count"] = signature_count
        row["first_alarm_time"] = "" if hazard_hit is None else hazard_hit.get("first_alarm_time", "")
        row["best_replay_parameter"] = _best_replay_parameter(w_hd_hit)
        row["in_W_hd"] = "yes" if w_hd_hit else "no"
        rows.append(row)

    w_hd = sorted(row["writable_id"] for row in rows if row.get("in_W_hd") == "yes")
    summary = dict(updated.get("summary", {}))
    summary["W_hd"] = len(w_hd)
    summary["W_hd_variables"] = w_hd
    updated["rows"] = rows
    updated["summary"] = summary
    updated["W_hd"] = w_hd
    updated["replay_confirmed_candidates"] = w_hd
    return updated


def _normalize_w_scan_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    updated = _recompute_w_hd_from_probes(payload) if payload.get("probe_rows") else dict(payload)
    rows = [dict(row) for row in updated.get("rows", []) or []]
    w_hd = sorted(str(row.get("writable_id", "")).strip() for row in rows if row.get("in_W_hd") == "yes")
    w_rank = sorted(str(row.get("writable_id", "")).strip() for row in rows if row.get("in_W_rank_top_k") == "yes")
    w_search = _search_variables(rows)
    selected = sorted(
        str(row.get("writable_id", "")).strip()
        for row in rows
        if row.get("selected_for_region_search") == "yes"
    )
    summary = {
        "W": len(rows),
        "W_rank_top_k": len(w_rank),
        "W_rank_top_k_variables": w_rank,
        "W_hd": len(w_hd),
        "W_hd_variables": w_hd,
        "W_search": len(w_search),
        "W_search_variables": w_search,
        "selected_variables": selected,
    }
    normalized = {
        "schema_version": updated.get("schema_version", 1),
        "platform": updated.get("platform", ""),
        "runtime_rule": updated.get("runtime_rule", RUNTIME_RULE),
        "duration_values": list(updated.get("duration_values", [])),
        "scan_rates": list(updated.get("scan_rates", [])),
        "directions": list(updated.get("directions", [])),
        "rank_top_k": updated.get("rank_top_k", DEFAULT_RANK_TOP_K),
        "manifest_path": updated.get("manifest_path", ""),
        "summary": summary,
        "W_hd": w_hd,
        "replay_confirmed_candidates": w_hd,
        "rows": rows,
        "probe_rows": list(updated.get("probe_rows", []) or []),
    }
    return normalized


def run_w_scan_for_platform(
    *,
    root: Path,
    platform: str,
    extraction: Mapping[str, Any],
    ranking: Mapping[str, Any],
    duration_values: Sequence[int],
    scan_rates: Sequence[float],
    directions: Sequence[str],
    rank_top_k: int,
    executor_factory: Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]] = _default_executor,
) -> dict[str, Any]:
    manifest_path = _manifest_path(root, platform)
    executor = executor_factory(manifest_path)
    selected = _selected_variables(root, platform)
    ranked_candidates = set(_ranking_top_k_variables(ranking, rank_top_k))
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for writable in extraction.get("W", []):
        point = _writable_id(writable)
        probe_rows: list[dict[str, Any]] = []
        if point in ranked_candidates:
            for direction in directions:
                for duration in duration_values:
                    for rate in scan_rates:
                        amp = _signed_rate(direction, float(rate))
                        task = _probe_task(point, int(duration), amp)
                        probe = dict(executor(task))
                        if hasattr(executor, "rules"):
                            probe["_rules"] = getattr(executor, "rules")
                        replay = _classify_w_probe(probe)
                        probe_rows.append(
                            {
                                "platform": platform,
                                "writable_id": point,
                                "direction": direction,
                                "duration": int(duration),
                                "amp": amp,
                                "runtime_steps": int(duration) + HOLD_STEPS,
                                "run_key": probe.get("run_key", ""),
                                **replay,
                            }
                        )
        detail_rows.extend(probe_rows)
        hazard_hit = _best_hazard(probe_rows)
        w_hd_hit = _best_hit(probe_rows)
        summary_rows.append(
            _w_summary_row(
                platform,
                writable,
                hazard_hit,
                w_hd_hit,
                probe_rows,
                duration_values,
                scan_rates,
                directions,
                selected,
                point in ranked_candidates,
            )
        )

    w_hd = sorted(row["writable_id"] for row in summary_rows if row["in_W_hd"] == "yes")
    payload = {
        "schema_version": 1,
        "platform": platform,
        "runtime_rule": RUNTIME_RULE,
        "duration_values": [int(value) for value in duration_values],
        "scan_rates": [float(value) for value in scan_rates],
        "directions": list(directions),
        "rank_top_k": int(rank_top_k),
        "manifest_path": str(manifest_path),
        "summary": {
            "W": len(summary_rows),
            "W_rank_top_k": len(ranked_candidates),
            "W_rank_top_k_variables": sorted(ranked_candidates),
            "W_hd": len(w_hd),
            "W_hd_variables": w_hd,
            "selected_variables": sorted(selected),
        },
        "W_hd": w_hd,
        "replay_confirmed_candidates": w_hd,
        "rows": summary_rows,
        "probe_rows": detail_rows,
    }
    return _normalize_w_scan_payload(payload)


def _w_summary_row(
    platform: str,
    writable: Mapping[str, Any],
    hazard_hit: Mapping[str, Any] | None,
    w_hd_hit: Mapping[str, Any] | None,
    probe_rows: Sequence[Mapping[str, Any]],
    duration_values: Sequence[int],
    scan_rates: Sequence[float],
    directions: Sequence[str],
    selected: set[str],
    in_rank_top_k: bool,
) -> dict[str, Any]:
    point = _writable_id(writable)
    signature_stable, signature_count = _alarm_signature_status(probe_rows, hazard_hit)
    return {
        "platform": platform,
        "writable_id": point,
        "layer": writable.get("layer", ""),
        "controller": writable.get("controller", ""),
        "direction_tested": "/".join(directions),
        "duration_domain": f"{min(duration_values)}..{max(duration_values)}",
        "slope_domain": f"{min(scan_rates)}..{max(scan_rates)}",
        "hazard_reached": "yes" if hazard_hit else "no",
        "target_hazard": "" if hazard_hit is None else hazard_hit.get("target_hazard", ""),
        "first_hazard_time": "" if hazard_hit is None else hazard_hit.get("first_hazard_time", ""),
        "pre_hazard_alarm_count": "" if hazard_hit is None else hazard_hit.get("pre_hazard_alarm_count", ""),
        "pre_hazard_alarm_ids": _alarm_ids_text(hazard_hit),
        "alarm_signature_stable": signature_stable,
        "alarm_signature_count": signature_count,
        "first_alarm_time": "" if hazard_hit is None else hazard_hit.get("first_alarm_time", ""),
        "best_replay_parameter": _best_replay_parameter(w_hd_hit),
        "in_W_rank_top_k": "yes" if in_rank_top_k else "no",
        "in_W_hd": "yes" if w_hd_hit else "no",
        "selected_for_region_search": "yes" if point in selected else "no",
    }


def build_eval_config(
    *,
    root: Path,
    platforms: Sequence[str],
    run_w_scan: bool,
    duration_values: Sequence[int],
    scan_rates: Sequence[float],
    directions: Sequence[str],
    rank_top_k: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rq": "RQ1",
        "stage1_root": str(root / "results" / "stage1"),
        "platforms": list(platforms),
        "seed": int(seed),
        "D_scope": "analysis-relevant semantic variables",
        "G_scope": "dependencies among D nodes",
        "P_granularity": "replay-oracle predicate granularity",
        "H_granularity": "replay-oracle predicate granularity",
        "W_scope": "threat-model writable attack entries",
        "candidate_naming": {
            "W_rank": "hazard_ranking.json candidate_variables",
            "W_rank_top_k": "union of the top-k hazard-ranking variables per hazard",
            "W_hd": "rank-top-k replay-confirmed hazard-driving variables with a stable non-empty pre-hazard alarm signature per target hazard",
            "W_search": "rank-top-k replay-confirmed variables selected for downstream region search",
        },
        "w_scan": {
            "source": "fresh_replay" if run_w_scan else "retained_artifact",
            "runtime_rule": RUNTIME_RULE,
            "rank_top_k": int(rank_top_k),
            "duration_values": [int(value) for value in duration_values],
            "scan_rates": [float(value) for value in scan_rates],
            "directions": list(directions),
            "alarm_silence_required": False,
        },
    }


def build_main_tables(
    stage1_counts: Mapping[str, Any],
    w_scan: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows = []
    for platform, counts in stage1_counts["platforms"].items():
        scan_summary = w_scan[platform]["summary"]
        w_total = int(counts["W"])
        w_rank = int(scan_summary["W_rank_top_k"])
        w_search = int(scan_summary["W_search"])
        all_pairs = w_total * (w_total - 1)
        search_pairs = w_search * (w_total - 1)
        reduction = round(all_pairs / search_pairs, 1) if search_pairs else 0.0
        rows.append(
            {
                "Platform": counts["display_name"],
                "Entries": f"{counts['D']}/{counts['G']}/{counts['P']}/{counts['H']}",
                "W": w_total,
                "W_rank": w_rank,
                "W_search": w_search,
                "W_rank_search": f"{w_rank}/{w_search}",
                "Pairs_all": all_pairs,
                "Pairs_search": search_pairs,
                "Pairs_all_search": f"{all_pairs}->{search_pairs}",
                "Reduction": f"{reduction:.1f}x",
                "W_rank_variables": scan_summary["W_rank_top_k_variables"],
                "W_search_variables": scan_summary["W_search_variables"],
            }
        )
    return {"writable_reduction": rows}


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(str(field).replace("|", "\\|") for field in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |")
    return lines


def write_main_tables(path: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    lines = ["# RQ1 Main Tables", ""]
    display_rows = [
        {
            "Platform": row["Platform"],
            "Entries": row["Entries"],
            "|W|": row["W"],
            "W kept rank/search": row["W_rank_search"],
            "Pairs all->search": row["Pairs_all_search"],
            "Reduction": row["Reduction"],
        }
        for row in tables["writable_reduction"]
    ]
    lines.extend(_markdown_table(display_rows))
    _write_text(path, "\n".join(lines))


def _scan_csv_fields() -> list[str]:
    return [
        "platform",
        "writable_id",
        "layer",
        "controller",
        "direction_tested",
        "duration_domain",
        "slope_domain",
        "hazard_reached",
        "target_hazard",
        "first_hazard_time",
        "pre_hazard_alarm_count",
        "pre_hazard_alarm_ids",
        "alarm_signature_stable",
        "alarm_signature_count",
        "first_alarm_time",
        "best_replay_parameter",
        "in_W_rank_top_k",
        "in_W_hd",
        "selected_for_region_search",
    ]


def write_w_scan_outputs(rq1_dir: Path, w_scan: Mapping[str, Any]) -> None:
    for platform, payload in w_scan.items():
        short = PLATFORM_CASES[platform]["short_name"]
        _write_csv(rq1_dir / f"rq1_w_scan_{short}.csv", payload["rows"], _scan_csv_fields())
        _write_json(rq1_dir / f"rq1_w_scan_{short}.json", payload)


def _load_existing_w_scan(rq1_dir: Path, platform: str) -> dict[str, Any]:
    short = PLATFORM_CASES[platform]["short_name"]
    path = rq1_dir / f"rq1_w_scan_{short}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing reusable RQ1 W scan artifact for {platform}")
    return _normalize_w_scan_payload(_read_json(path))


def _normalize_platforms(platforms: Sequence[str] | None) -> tuple[str, ...]:
    if not platforms:
        return ("boiler_ccs", "tennessee_eastman")
    resolved = []
    aliases = {"boiler": "boiler_ccs", "boilerCCS": "boiler_ccs", "TE": "tennessee_eastman", "te": "tennessee_eastman"}
    for platform in platforms:
        value = aliases.get(str(platform), str(platform))
        if value not in PLATFORM_CASES:
            raise ValueError(f"unsupported platform: {platform}")
        resolved.append(value)
    return tuple(dict.fromkeys(resolved))


def run_rq1_evaluation(
    *,
    root: str | Path,
    platforms: Sequence[str] | None = None,
    run_w_scan: bool = False,
    duration_values: Sequence[int] = DEFAULT_DURATION_VALUES,
    scan_rates: Sequence[float] = DEFAULT_SCAN_RATES,
    directions: Sequence[str] = DEFAULT_DIRECTIONS,
    rank_top_k: int = DEFAULT_RANK_TOP_K,
    seed: int = DEFAULT_SEED,
    executor_factory_map: Mapping[str, Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    resolved_platforms = _normalize_platforms(platforms)
    stage1_inputs = _load_stage1_inputs(root, resolved_platforms)
    rq1_dir = root / "results" / "rq1"

    config = build_eval_config(
        root=root,
        platforms=resolved_platforms,
        run_w_scan=run_w_scan,
        duration_values=duration_values,
        scan_rates=scan_rates,
        directions=directions,
        rank_top_k=rank_top_k,
        seed=seed,
    )
    stage1_counts = build_stage1_counts(stage1_inputs)

    w_scan: dict[str, Any] = {}
    for platform in resolved_platforms:
        if run_w_scan:
            factory = _executor_factory_for(platform, executor_factory_map)
            w_scan[platform] = run_w_scan_for_platform(
                root=root,
                platform=platform,
                extraction=stage1_inputs[platform]["extraction"],
                ranking=stage1_inputs[platform]["ranking"],
                duration_values=duration_values,
                scan_rates=scan_rates,
                directions=directions,
                rank_top_k=rank_top_k,
                executor_factory=factory,
            )
        else:
            payload = _load_existing_w_scan(rq1_dir, platform)
            if int(payload.get("rank_top_k", 0) or 0) != int(rank_top_k):
                raise ValueError(f"reusable RQ1 W scan for {platform} was not generated with rank_top_k={rank_top_k}")
            w_scan[platform] = payload
    if run_w_scan:
        write_w_scan_outputs(rq1_dir, w_scan)

    main_tables = build_main_tables(stage1_counts, w_scan)

    _write_json(rq1_dir / "rq1_eval_config.json", config)
    _write_json(rq1_dir / "rq1_stage1_counts.json", stage1_counts)
    _write_json(rq1_dir / "rq1_main_tables.json", main_tables)
    write_main_tables(rq1_dir / "rq1_main_tables.md", main_tables)

    return {
        "config": config,
        "stage1_counts": stage1_counts,
        "main_tables": main_tables,
        "w_scan": w_scan,
    }


def _executor_factory_for(
    platform: str,
    executor_factory_map: Mapping[str, Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]]] | None,
) -> Callable[[Path], Callable[[dict[str, Any]], dict[str, Any]]]:
    if executor_factory_map:
        for key in (platform, platform.lower(), PLATFORM_CASES[platform]["stage1_dir"]):
            factory = executor_factory_map.get(key)
            if factory:
                return factory
    return _default_executor


def _parse_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return items


def _parse_floats(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("provide at least one float")
    return items


def _parse_strings(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("provide at least one value")
    return items


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RQ1 writable-entry reduction table.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument("--run-w-scan", action="store_true")
    parser.add_argument("--duration-values", type=_parse_ints, default=DEFAULT_DURATION_VALUES)
    parser.add_argument("--scan-rates", type=_parse_floats, default=DEFAULT_SCAN_RATES)
    parser.add_argument("--directions", type=_parse_strings, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--rank-top-k", type=int, default=DEFAULT_RANK_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_rq1_evaluation(
        root=args.root,
        platforms=args.platform or None,
        run_w_scan=bool(args.run_w_scan),
        duration_values=args.duration_values,
        scan_rates=args.scan_rates,
        directions=args.directions,
        rank_top_k=int(args.rank_top_k),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

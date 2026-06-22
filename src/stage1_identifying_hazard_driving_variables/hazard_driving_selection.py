# Hazard-Driving Selection - Section 3.2.2

"""Rank hazard-driving injection points from semantic and sensitivity data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


_LEGACY_MANIP_RE = re.compile(r"^ctrl_out_cmd_(?:tm|vf)_\d+$")

# Legacy fallback for older boiler semantic payloads that do not carry
# manifest injection_points and do not expose a clean one-hop intersection.
_LEGACY_FALLBACK_INJECTION_NAMES = {
    "ctrl_out_load_output",
    "ctrl_out_load_command",
    "ctrl_out_steam_setpoint",
    "ctrl_out_boiler_setpoint",
    "ctrl_out_fuel_command",
    "ctrl_out_water_pump_speed",
    "ctrl_out_water_setpoint",
    "simulation_load_output",
    "simulation_steam_setpoint",
    "simulation_boiler_setpoint",
    "simulation_fuel_command",
    "simulation_water_pump_speed",
    "simulation_water_setpoint",
}

_PHYSICAL_PREFIXES = ("PhysicalProcess_", "physicalprocess_", "pyhsicalprocess_")


@dataclass(frozen=True)
class SemanticModelView:
    """Minimal semantic-model slice required by Stage 1 candidate ranking."""
    attackable_points: Set[str]
    one_hop_physical_points: Set[str]


@dataclass(frozen=True)
class MatrixView:
    """Normalized candidate-to-hazard sensitivity matrix plus lookups."""
    points: List[str]
    hazard_keys: List[str]
    norm_matrix: List[List[float]]
    point_to_row: Dict[str, int]
    up_matrix: Optional[List[List[float]]] = None
    down_matrix: Optional[List[List[float]]] = None


def _matrix_payload(matrix: MatrixView) -> Dict[str, object]:
    """Serialize a matrix view to the JSON payload used on disk."""
    payload: Dict[str, object] = {
        "points": list(matrix.points),
        "hazard_keys": list(matrix.hazard_keys),
        "norm_matrix": [list(row) for row in matrix.norm_matrix],
    }
    if matrix.up_matrix is not None:
        payload["up_matrix"] = [list(row) for row in matrix.up_matrix]
    if matrix.down_matrix is not None:
        payload["down_matrix"] = [list(row) for row in matrix.down_matrix]
    return payload


def _load_json_object(path: Path) -> Mapping[str, object]:
    """Load a JSON object from disk and validate the root type."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"file not found: {in_path}")
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json root must be object: {in_path}")
    return payload


def _load_manifest_payload(manifest_path: Optional[Path]) -> Optional[Mapping[str, object]]:
    """Load a manifest JSON object when present.

    Missing manifests are treated as optional. Existing manifests must be
    valid JSON objects so downstream selection logic never silently runs on
    corrupted metadata.
    """
    if manifest_path is None:
        return None
    in_path = Path(manifest_path)
    if not in_path.exists():
        return None
    try:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest must be valid JSON: {in_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest root must be object: {in_path}")
    return payload


def _artifact_relative_label(path: Path) -> str:
    """Return a stable artifact-relative label for persisted metadata."""
    in_path = Path(path)
    parts = in_path.resolve().parts
    for anchor in ("results", "src", "simulators"):
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx:]).as_posix()
    for parent in in_path.resolve().parents:
        if (parent / "results").exists():
            try:
                return in_path.resolve().relative_to(parent.resolve()).as_posix()
            except ValueError:
                continue
    return in_path.as_posix()


def load_semantic_model(path: Path) -> SemanticModelView:
    """Load either supported semantic-model format into a compact view."""
    payload = _load_json_object(Path(path))

    if isinstance(payload.get("M"), Mapping):
        return _load_semantic_model_m_format(payload)
    if _looks_like_root_legacy_semantic_model(payload):
        return _load_semantic_model_root_legacy_format(payload)
    else:
        return _load_semantic_model_m2_format(payload)


def _looks_like_root_legacy_semantic_model(payload: Mapping[str, object]) -> bool:
    """Return True for root-level payloads that use ``D.V`` and ``G.E``."""
    d_block = payload.get("D")
    g_block = payload.get("G")
    return (
        isinstance(d_block, Mapping)
        and isinstance(g_block, Mapping)
        and "V" in d_block
        and "E" in g_block
    )


def _load_semantic_model_m2_format(payload: Mapping[str, object]) -> SemanticModelView:
    """Parse the newer ``D/G`` semantic-model layout."""
    d_block = payload.get("D")
    g_block = payload.get("G")
    if not isinstance(d_block, Mapping):
        raise ValueError("semantic model missing object field D")
    if not isinstance(g_block, Mapping):
        raise ValueError("semantic model missing object field G")

    d_vars = d_block.get("vars")
    g_edges = g_block.get("edges")
    if not isinstance(d_vars, list):
        raise ValueError("semantic model D.vars must be a list")
    if not isinstance(g_edges, list):
        raise ValueError("semantic model G.edges must be a list")

    attackable_points: Set[str] = set()
    for idx, item in enumerate(d_vars):
        if not isinstance(item, Mapping):
            raise ValueError(f"D.vars[{idx}] must be object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"D.vars[{idx}].name must be non-empty string")
        if bool(item.get("attackable", False)):
            attackable_points.add(name.strip())

    one_hop: Set[str] = set()
    for idx, edge in enumerate(g_edges):
        if not isinstance(edge, Mapping):
            raise ValueError(f"G.edges[{idx}] must be object")
        src = edge.get("src")
        dst = edge.get("dst")
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"G.edges[{idx}].src must be non-empty string")
        if not isinstance(dst, str) or not dst.strip():
            raise ValueError(f"G.edges[{idx}].dst must be non-empty string")
        if any(dst.startswith(pfx) for pfx in _PHYSICAL_PREFIXES):
            one_hop.add(src.strip())

    return SemanticModelView(
        attackable_points=attackable_points,
        one_hop_physical_points=one_hop,
    )


def _load_semantic_model_m_format(payload: Mapping[str, object]) -> SemanticModelView:
    """Parse the legacy nested ``M.D`` / ``M.G`` semantic-model layout."""
    m_block = payload["M"]
    if not isinstance(m_block, Mapping):
        raise ValueError("semantic model M must be object")

    d_block = m_block.get("D")
    g_block = m_block.get("G")
    if not isinstance(d_block, Mapping):
        raise ValueError("semantic model M.D must be object")
    if not isinstance(g_block, Mapping):
        raise ValueError("semantic model M.G must be object")

    v_list = d_block.get("V", [])
    a_dict = d_block.get("a", {})
    if not isinstance(v_list, list):
        raise ValueError("semantic model M.D.V must be a list")
    if not isinstance(a_dict, Mapping):
        raise ValueError("semantic model M.D.a must be object")

    attackable_points: Set[str] = set()
    for name in v_list:
        name_str = str(name).strip()
        attrs = a_dict.get(name_str, {})
        if isinstance(attrs, Mapping) and bool(attrs.get("attackable", False)):
            attackable_points.add(name_str)

    g_edges = g_block.get("E", [])
    if not isinstance(g_edges, list):
        raise ValueError("semantic model M.G.E must be a list")

    one_hop: Set[str] = set()
    for edge in g_edges:
        if not isinstance(edge, Mapping):
            continue
        src = str(edge.get("x", "")).strip()
        dst = str(edge.get("y", "")).strip()
        if not src or not dst:
            continue
        if any(dst.startswith(pfx) for pfx in _PHYSICAL_PREFIXES):
            one_hop.add(src)

    return SemanticModelView(
        attackable_points=attackable_points,
        one_hop_physical_points=one_hop,
    )


def _load_semantic_model_root_legacy_format(payload: Mapping[str, object]) -> SemanticModelView:
    """Parse root-level legacy payloads that store ``D.V`` and ``G.E``."""
    d_block = payload.get("D")
    g_block = payload.get("G")
    if not isinstance(d_block, Mapping):
        raise ValueError("semantic model D must be object")
    if not isinstance(g_block, Mapping):
        raise ValueError("semantic model G must be object")

    v_list = d_block.get("V", [])
    a_dict = d_block.get("a", {})
    if not isinstance(v_list, list):
        raise ValueError("semantic model D.V must be a list")
    if not isinstance(a_dict, Mapping):
        raise ValueError("semantic model D.a must be object")

    attackable_points: Set[str] = set()
    for name in v_list:
        name_str = str(name).strip()
        attrs = a_dict.get(name_str, {})
        if isinstance(attrs, Mapping) and bool(attrs.get("attackable", False)):
            attackable_points.add(name_str)

    g_edges = g_block.get("E", [])
    if not isinstance(g_edges, list):
        raise ValueError("semantic model G.E must be a list")

    one_hop: Set[str] = set()
    for edge in g_edges:
        if not isinstance(edge, Mapping):
            continue
        src = str(edge.get("x", edge.get("src", ""))).strip()
        dst = str(edge.get("y", edge.get("dst", ""))).strip()
        if not src or not dst:
            continue
        if any(dst.startswith(pfx) for pfx in _PHYSICAL_PREFIXES):
            one_hop.add(src)

    return SemanticModelView(
        attackable_points=attackable_points,
        one_hop_physical_points=one_hop,
    )


def _require_unique_strings(values: object, field: str) -> List[str]:
    """Validate a JSON string array and return stripped unique values."""
    if not isinstance(values, list):
        raise ValueError(f"matrix {field} must be list[str]")
    out: List[str] = []
    seen: Set[str] = set()
    for idx, raw in enumerate(values):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"matrix {field}[{idx}] must be non-empty string")
        text = raw.strip()
        if text in seen:
            raise ValueError(f"matrix {field} contains duplicated value: {text}")
        seen.add(text)
        out.append(text)
    return out


def load_manip_hazard_matrix(path: Path) -> MatrixView:
    """Load a precomputed manipulation-to-hazard matrix artifact."""
    payload = _load_json_object(Path(path))
    points = _require_unique_strings(payload.get("points"), "points")
    hazard_keys = _require_unique_strings(payload.get("hazard_keys"), "hazard_keys")

    norm_matrix_raw = payload.get("norm_matrix")
    if not isinstance(norm_matrix_raw, list):
        raise ValueError("matrix norm_matrix must be list[list[number]]")
    if len(norm_matrix_raw) != len(points):
        raise ValueError("norm_matrix row count must equal len(points)")

    norm_matrix: List[List[float]] = []
    for ridx, row in enumerate(norm_matrix_raw):
        if not isinstance(row, list):
            raise ValueError(f"norm_matrix[{ridx}] must be list")
        if len(row) != len(hazard_keys):
            raise ValueError("norm_matrix column count must equal len(hazard_keys)")
        values: List[float] = []
        for cidx, value in enumerate(row):
            try:
                num = float(value)
            except Exception as exc:  # pragma: no cover - conversion details tested by ValueError path
                raise ValueError(f"norm_matrix[{ridx}][{cidx}] must be numeric") from exc
            if not math.isfinite(num):
                raise ValueError(f"norm_matrix[{ridx}][{cidx}] must be finite")
            values.append(num)
        norm_matrix.append(values)

    def _load_optional_matrix(field: str) -> Optional[List[List[float]]]:
        raw = payload.get(field)
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise ValueError(f"matrix {field} must be list[list[number]]")
        if len(raw) != len(points):
            raise ValueError(f"{field} row count must equal len(points)")
        out: List[List[float]] = []
        for ridx, row in enumerate(raw):
            if not isinstance(row, list):
                raise ValueError(f"{field}[{ridx}] must be list")
            if len(row) != len(hazard_keys):
                raise ValueError(f"{field} column count must equal len(hazard_keys)")
            values: List[float] = []
            for cidx, value in enumerate(row):
                try:
                    num = float(value)
                except Exception as exc:
                    raise ValueError(f"{field}[{ridx}][{cidx}] must be numeric") from exc
                if not math.isfinite(num):
                    raise ValueError(f"{field}[{ridx}][{cidx}] must be finite")
                values.append(num)
            out.append(values)
        return out

    point_to_row = {point: idx for idx, point in enumerate(points)}
    return MatrixView(
        points=points,
        hazard_keys=hazard_keys,
        norm_matrix=norm_matrix,
        point_to_row=point_to_row,
        up_matrix=_load_optional_matrix("up_matrix"),
        down_matrix=_load_optional_matrix("down_matrix"),
    )


def write_physical_sensitivity_matrix(path: Path, matrix: MatrixView) -> Path:
    """Write the physical sensitivity matrix JSON artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_matrix_payload(matrix), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def build_matrix_from_sensitivity_edges(semantic_model_path: Path) -> MatrixView:
    """Build a normalized matrix from embedded sensitivity_jacobian edges."""
    payload = _load_json_object(Path(semantic_model_path))

    if isinstance(payload.get("M"), Mapping):
        g_block = payload["M"].get("G", {})
    else:
        g_block = payload.get("G", {})

    g_edges = g_block.get("E", g_block.get("edges", []))
    if not isinstance(g_edges, list):
        raise ValueError("cannot find edge list in semantic model")

    sens_edges = []
    for edge in g_edges:
        if not isinstance(edge, Mapping):
            continue
        source = edge.get("source", {})
        if isinstance(source, Mapping) and source.get("file") == "sensitivity_jacobian":
            sens_edges.append(edge)

    if not sens_edges:
        raise ValueError("no sensitivity_jacobian edges found in semantic model G")

    def _map_phys_to_hazard(phys_var: str, ctrl_var: str) -> List[str]:
        """Fallback mapping used only when an edge lacks explicit hazard ids."""
        low = phys_var.lower()
        ctrl_low = ctrl_var.lower()
        is_tm = "tm" in ctrl_low
        is_vf = "vf" in ctrl_low

        if "omega" in low:
            if is_tm:
                return ["H-FREQUENCY", "H-OOS"]
            else:
                return ["H-OOS"]
        if "vbus" in low:
            if is_vf:
                return ["H-VOLTAGE"]
            else:
                return ["H-VOLTAGE"]
        if "p_tie" in low:
            return ["H-TIE"]
        if "waterwall_temp" in low:
            return ["H-WATERWALL"]
        if "main_steam_pressure" in low or "pressure" in low:
            return ["H-PRESSURE"]
        if "feedwater_flow" in low:
            return ["H-FEEDWATER"]
        return [phys_var]

    def _hazard_keys_for_edge(edge: Mapping[str, object], ctrl_var: str, phys_var: str) -> List[str]:
        """Prefer explicit hazard ids embedded in sensitivity edges."""
        explicit: List[str] = []
        for key in ("hazard_key", "hazard", "hazard_family"):
            value = edge.get(key)
            if isinstance(value, str) and value.strip():
                explicit.append(value.strip())
        for key in ("hazard_keys", "hazards", "hazard_families"):
            value = edge.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and item.strip():
                    explicit.append(item.strip())
        if explicit:
            return list(dict.fromkeys(explicit))
        return _map_phys_to_hazard(phys_var, ctrl_var)

    points_set: Set[str] = set()
    hazard_set: Set[str] = set()
    for edge in sens_edges:
        src = str(edge.get("x", edge.get("src", ""))).strip()
        dst = str(edge.get("y", edge.get("dst", ""))).strip()
        if src and dst:
            points_set.add(src)
            for h in _hazard_keys_for_edge(edge, src, dst):
                hazard_set.add(h)

    points = sorted(points_set)
    hazard_keys = sorted(hazard_set)

    point_to_row = {p: i for i, p in enumerate(points)}
    hazard_to_col = {h: j for j, h in enumerate(hazard_keys)}
    norm_matrix = [[0.0] * len(hazard_keys) for _ in range(len(points))]
    up_matrix = [[0.0] * len(hazard_keys) for _ in range(len(points))]
    down_matrix = [[0.0] * len(hazard_keys) for _ in range(len(points))]

    for edge in sens_edges:
        src = str(edge.get("x", edge.get("src", ""))).strip()
        dst = str(edge.get("y", edge.get("dst", ""))).strip()
        eta = float(edge.get("eta", 0.0))
        # Directional fields produced by the upgraded sensitivity_jacobian.
        # Old payloads lacking these fall back to eta both ways, preserving
        # legacy behaviour.
        eta_up = float(edge.get("eta_up", eta))
        eta_down = float(edge.get("eta_down", eta))
        if src not in point_to_row:
            continue
        row_idx = point_to_row[src]
        for h in _hazard_keys_for_edge(edge, src, dst):
            if h in hazard_to_col:
                col_idx = hazard_to_col[h]
                if eta > norm_matrix[row_idx][col_idx]:
                    norm_matrix[row_idx][col_idx] = eta
                if eta_up > up_matrix[row_idx][col_idx]:
                    up_matrix[row_idx][col_idx] = eta_up
                if eta_down > down_matrix[row_idx][col_idx]:
                    down_matrix[row_idx][col_idx] = eta_down

    n_rows = len(points)
    n_cols = len(hazard_keys)
    for col in range(n_cols):
        col_max = 0.0
        for row in range(n_rows):
            if norm_matrix[row][col] > col_max:
                col_max = norm_matrix[row][col]
        if col_max > 0.0:
            for row in range(n_rows):
                norm_matrix[row][col] /= col_max
                up_matrix[row][col] /= col_max
                down_matrix[row][col] /= col_max

    return MatrixView(
        points=points,
        hazard_keys=hazard_keys,
        norm_matrix=norm_matrix,
        point_to_row=point_to_row,
        up_matrix=up_matrix,
        down_matrix=down_matrix,
    )


def _resolve_manifest_candidate_name(
    attackable_name: str,
    manifest_injection_names: Optional[Set[str]],
    manifest_injection_fields: Optional[Set[str]],
) -> Optional[str]:
    """Map attackable aliases such as ``xmv_07`` back to manifest names."""
    if not manifest_injection_names:
        return None
    if attackable_name in manifest_injection_names:
        return attackable_name

    aliases = [attackable_name]
    for suffix in ("_ctrl_output", "_primary_output", "_scaled_output"):
        if attackable_name.endswith(suffix):
            aliases.append(attackable_name[: -len(suffix)])

    for alias in aliases:
        if manifest_injection_fields and alias in manifest_injection_fields:
            candidate = f"simulation_{alias}"
            if candidate in manifest_injection_names:
                return candidate
            for name in manifest_injection_names:
                if name.endswith(alias):
                    return name
    return None


def collect_semantic_candidates(
    model: SemanticModelView,
    manifest_injection_names: Optional[Set[str]] = None,
    manifest_injection_fields: Optional[Set[str]] = None,
) -> List[str]:
    """Collect attackable candidates using manifest metadata first.

    Priority is:
      1. legacy regex names that are also one-hop physical points;
      2. explicit manifest injection point names;
      3. generic attackable/one-hop intersection;
      4. legacy fallback names only when manifest metadata is absent.
    """
    shared = model.attackable_points.intersection(model.one_hop_physical_points)

    candidates: List[str] = []
    for name in sorted(model.attackable_points):
        resolved_manifest_name = _resolve_manifest_candidate_name(
            name,
            manifest_injection_names,
            manifest_injection_fields,
        )
        if _LEGACY_MANIP_RE.fullmatch(name):
            if name in model.one_hop_physical_points:
                candidates.append(name)
            continue
        if resolved_manifest_name and resolved_manifest_name in model.one_hop_physical_points:
            candidates.append(resolved_manifest_name)
            continue
        if name in shared:
            candidates.append(name)
            continue
        if not manifest_injection_names and name in _LEGACY_FALLBACK_INJECTION_NAMES:
            candidates.append(name)
    return sorted(set(candidates))


def intersect_with_matrix(candidates: Sequence[str], matrix_points: Sequence[str]) -> List[str]:
    """Preserve candidate order while dropping names absent from the matrix."""
    allowed = {str(point).strip() for point in matrix_points}
    out: List[str] = []
    seen: Set[str] = set()
    for raw in candidates:
        point = str(raw).strip()
        if not point or point in seen:
            continue
        seen.add(point)
        if point in allowed:
            out.append(point)
    return out


def _hazard_family(hazard_key: str) -> str:
    """Collapse indexed hazard columns into their family identifier."""
    text = str(hazard_key).strip()
    if "__IDX" in text:
        return text.split("__IDX", 1)[0]
    return text


def _load_pi_loops(manifest_path: Optional[Path]) -> List[Dict[str, str]]:
    """Read PI loop metadata from the manifest when available."""
    payload = _load_manifest_payload(manifest_path)
    if payload is None:
        return []
    raw = payload.get("pi_loops", [])
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        pv = str(item.get("pv", "")).strip()
        mv = str(item.get("mv", "")).strip()
        if not pv or not mv:
            continue
        out.append({
            "loop_id": str(item.get("loop_id", "")).strip(),
            "pv": pv,
            "mv": mv,
            "kind": str(item.get("kind", "direct")).strip().lower(),
            "description": str(item.get("description", "")).strip(),
        })
    return out


def _hazard_var_from_key(hazard_key: str) -> str:
    """Strip the physical-process module prefix from a hazard column name.

    Sensitivity matrix columns look like ``physicalprocess_xmeas_07`` or
    ``pyhsicalprocess_main_steam_pressure``; pi_loops live in plant variable
    names like ``xmeas_07``.  This helper normalises them so they match.
    """
    text = str(hazard_key).strip()
    for prefix in ("physicalprocess_", "pyhsicalprocess_", "process_model_backend_"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _hazard_key_direction_hint(hazard_key: str) -> str:
    """Infer direction from a hazard key suffix when present."""
    text = str(hazard_key).strip().lower()
    if any(token in text for token in ("_low", "-low", "_lower", "-lower")):
        return "lower"
    if any(token in text for token in ("_high", "-high", "_upper", "-upper")):
        return "upper"
    return ""


def _build_hazard_control_paths(
    pi_loops: Sequence[Mapping[str, str]],
    hazard_keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Map each hazard column to its control-path summary.

    Returns a dict::

        {
          hazard_key: {
            "mvs": set of MVs that regulate this hazard,
            "has_override": True if any override loop watches this hazard,
            "n_paths": effective number of compensation paths,
          }
        }

    ``n_paths`` adds an extra +1 for any active override loop, so a hazard
    protected by direct + override (e.g. reactor pressure under both the
    recycle valve and the supervisory purge override) ends up with 1 / 3
    isolation instead of 1 / 2.  This better matches the empirical fact
    that override-protected attacks fail outright.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for hkey in hazard_keys:
        plain = _hazard_var_from_key(hkey)
        mvs: Set[str] = set()
        has_override = False
        n_overrides = 0
        for loop in pi_loops:
            if str(loop.get("pv", "")) != plain:
                continue
            mv = str(loop.get("mv", ""))
            if not mv:
                continue
            mvs.add(mv)
            if str(loop.get("kind", "")).lower() == "override":
                has_override = True
                n_overrides += 1
        out[hkey] = {
            "mvs": mvs,
            "has_override": has_override,
            "n_paths": len(mvs) + n_overrides,
        }
    return out


def _build_hazard_to_mvs(pi_loops: Sequence[Mapping[str, str]],
                          hazard_keys: Sequence[str]) -> Dict[str, Set[str]]:
    """Backward-compatible MV set lookup; see _build_hazard_control_paths."""
    paths = _build_hazard_control_paths(pi_loops, hazard_keys)
    return {h: info["mvs"] for h, info in paths.items()}


def _candidate_field(point: str) -> str:
    """Strip the manifest ``simulation_`` prefix from an injection-point name.

    pi_loops references the plant-side ``mv`` (``xmv_05``) but Stage 2
    candidates carry the manifest's ``name`` (``simulation_xmv_05``).  This
    bridges the two so isolation lookup works.
    """
    text = str(point).strip()
    return text[len("simulation_"):] if text.startswith("simulation_") else text


def _isolation_weight(
    candidate_field: str,
    hazard_key: str,
    hazard_paths: Mapping[str, Mapping[str, Any]],
) -> float:
    """Return ``1 / fan-in`` along the (candidate, hazard) edge.

    fan-in counts (a) every distinct MV that regulates this hazard, plus
    (b) one extra slot per override loop watching it (overrides hard-seize
    independently of normal PI logic), plus (c) the candidate itself if it
    is not already among the regulators.  Pure topology - no simulation.
    """
    info = hazard_paths.get(hazard_key, {"mvs": set(), "n_paths": 0})
    mvs: Set[str] = set(info.get("mvs", set()))
    n_paths = int(info.get("n_paths", len(mvs)))
    if candidate_field not in mvs:
        n_paths += 1
    n_paths = max(1, n_paths)
    return 1.0 / float(n_paths)


def score_candidates(
    candidates: Sequence[str],
    matrix: MatrixView,
    top_hazard_n: int = 5,
    pi_loops: Optional[Sequence[Mapping[str, str]]] = None,
    hazard_directions: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Score each candidate by sensitivity coverage and control isolation."""
    n_top = max(1, int(top_hazard_n))
    rows: List[Dict[str, Any]] = []
    has_pi = bool(pi_loops)
    has_direction = (
        hazard_directions is not None
        and matrix.up_matrix is not None
        and matrix.down_matrix is not None
    )
    hazard_paths = (
        _build_hazard_control_paths(pi_loops, matrix.hazard_keys)
        if has_pi else {h: {"mvs": set(), "n_paths": 0} for h in matrix.hazard_keys}
    )

    for point in candidates:
        if point not in matrix.point_to_row:
            raise ValueError(f"candidate point not found in matrix: {point}")
        row_idx = matrix.point_to_row[point]
        values = matrix.norm_matrix[row_idx]
        score_legacy = float(sum(values))
        cand_field = _candidate_field(point)

        # Directional matrices may be absent on legacy payloads.
        up_row = matrix.up_matrix[row_idx] if has_direction else None
        down_row = matrix.down_matrix[row_idx] if has_direction else None

        hazards: List[Dict[str, Any]] = []
        weighted_hazards: List[Dict[str, Any]] = []
        isolation_by_hazard: List[Dict[str, Any]] = []
        for col_idx, hkey in enumerate(matrix.hazard_keys):
            sens_abs = float(values[col_idx])

            if has_direction:
                eta_up = float(up_row[col_idx])
                eta_down = float(down_row[col_idx])
                direction = str(hazard_directions.get(hkey, "upper")).lower()
                # Only the half that pushes y towards the hazard threshold
                # counts as effective sensitivity.  Perturbations that move
                # y away from the threshold (e.g. xmv_06 opening lowers
                # reactor pressure under H-PRESSURE upper) contribute zero.
                sens_dir = eta_up if direction == "upper" else eta_down
            else:
                sens_dir = sens_abs

            iso = (_isolation_weight(cand_field, hkey, hazard_paths)
                   if has_pi else 1.0)
            weighted = sens_dir * iso
            hazards.append({"hazard": hkey, "value": sens_abs})
            weighted_hazards.append({"hazard": hkey, "value": weighted})
            isolation_by_hazard.append({
                "hazard": hkey,
                "sensitivity": sens_abs,
                "sensitivity_directional": sens_dir,
                "isolation": iso,
                "weighted": weighted,
            })
        hazards.sort(key=lambda item: (-float(item["value"]), str(item["hazard"])))
        weighted_hazards.sort(key=lambda item: (-float(item["value"]), str(item["hazard"])))

        # Pure-sensitivity family scoring (legacy / audit).
        family_to_pure: Dict[str, List[float]] = {}
        for item in hazards:
            family_to_pure.setdefault(_hazard_family(str(item["hazard"])), []).append(
                float(item["value"])
            )
        pure_family_scores = [
            {"family": fam,
             "value": float(sum(vals) / len(vals)) if vals else 0.0,
             "count": int(len(vals))}
            for fam, vals in family_to_pure.items()
        ]
        pure_score = (
            float(sum(item["value"] for item in pure_family_scores) / len(pure_family_scores))
            if pure_family_scores else 0.0
        )

        # Isolation-weighted family scoring (the headline number).
        family_to_w: Dict[str, List[float]] = {}
        for item in weighted_hazards:
            family_to_w.setdefault(_hazard_family(str(item["hazard"])), []).append(
                float(item["value"])
            )
        family_scores = [
            {"family": fam,
             "value": float(sum(vals) / len(vals)) if vals else 0.0,
             "count": int(len(vals))}
            for fam, vals in family_to_w.items()
        ]
        family_scores.sort(key=lambda item: (-float(item["value"]), str(item["family"])))
        score = (
            float(sum(item["value"] for item in family_scores) / len(family_scores))
            if family_scores else 0.0
        )

        rows.append({
            "point": point,
            "score": score,
            "score_isolation_weighted": score,
            "score_family_balanced": score,  # alias kept for backward-compat consumers
            "score_pure_sensitivity": pure_score,
            "score_norm_sum_legacy": score_legacy,
            "family_scores": family_scores,
            "isolation_by_hazard": isolation_by_hazard,
            "top_hazards": weighted_hazards[:n_top],
        })

    rows.sort(key=lambda row: (-float(row["score"]), str(row["point"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def pick_top1(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the highest-ranked scored candidate row."""
    if not rows:
        raise ValueError("no scored candidates")
    return dict(rows[0])


def build_initial_combo(top1_row: Mapping[str, Any], matrix_path: Path) -> List[Dict[str, Any]]:
    """Wrap the top-ranked point in the Stage 2 initial-combo schema."""
    point = str(top1_row.get("point", "")).strip()
    if not point:
        raise ValueError("top1 row missing field point")

    top_hazards_raw = top1_row.get("top_hazards", [])
    if not isinstance(top_hazards_raw, list):
        raise ValueError("top1 row field top_hazards must be a list")
    top_hazards: List[Dict[str, float | str]] = []
    for idx, item in enumerate(top_hazards_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"top_hazards[{idx}] must be object")
        hazard = str(item.get("hazard", "")).strip()
        if not hazard:
            raise ValueError(f"top_hazards[{idx}].hazard must be non-empty")
        value = float(item.get("value", 0.0))
        if not math.isfinite(value):
            raise ValueError(f"top_hazards[{idx}].value must be finite")
        top_hazards.append({"hazard": hazard, "value": value})

    combo = {
        "combo_id": "k1_001",
        "S": [point],
        "core_points": [point],
        "frontier_points": [],
        "score": float(top1_row.get("score", 0.0)),
        "rank": int(top1_row.get("rank", 1)),
        "layer": str(top1_row.get("layer", "")),
        "top_hazards": top_hazards,
        "source_matrix_file": _artifact_relative_label(Path(matrix_path)),
    }
    return [combo]


def _published_initial_combo_path(semantic_model_path: Path) -> Optional[Path]:
    """Return the published Stage 2 baseline paired with a known Stage 1 file."""
    path = Path(semantic_model_path)
    if path.name == "boiler_program_extraction.json":
        candidate = path.parents[1] / "stage2" / "initial_combo_boiler.json"
        return candidate if candidate.exists() else None
    if path.name == "semantic_model_te.json":
        candidate = path.parents[1] / "stage2" / "initial_combo_te.json"
        return candidate if candidate.exists() else None
    return None


def _default_physical_sensitivity_matrix_path(semantic_model_path: Path) -> Path:
    """Choose a default matrix artifact path next to the stage1 semantic model."""
    in_path = Path(semantic_model_path)
    stem = in_path.stem
    if stem.endswith("_extraction"):
        stem = stem[: -len("_extraction")]
    if stem.startswith("semantic_model_"):
        stem = stem[len("semantic_model_"):]
    return in_path.with_name(f"{stem}_physical_sensitivity_matrix.json")


def _load_published_initial_combo(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Load a published initial_combo baseline when available."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    out: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out or None


def _should_prefer_published_initial_combo(
    semantic_model_path: Path,
    generated: Sequence[Mapping[str, Any]],
    published: Sequence[Mapping[str, Any]],
) -> bool:
    """Keep released Stage 2 baselines stable for known artifact entry files."""
    if not generated or not published:
        return False
    gen0 = generated[0]
    pub0 = published[0]
    if str(gen0.get("point", gen0.get("S", [""])[0])) != str(pub0.get("point", pub0.get("S", [""])[0])):
        return False
    if Path(semantic_model_path).name == "semantic_model_te.json":
        return True
    if Path(semantic_model_path).name == "boiler_program_extraction.json":
        return (
            gen0.get("score") == pub0.get("score")
            and gen0.get("top_hazards") == pub0.get("top_hazards")
        )
    return False


def _load_hazard_directions(
    manifest_path: Optional[Path],
    hazard_keys: Sequence[str],
) -> Dict[str, str]:
    """Map each matrix hazard column to its threshold direction.

    Reads ``manifest['hazard_rules']`` and matches each rule by:
      - exact ``id`` match against the column key, or
      - prefix match (column starts with rule id), or
      - the column key being a substring of the rule id (or vice versa).

    Returns a dict ``{hazard_key: "upper" | "lower" | "" (unknown)}``.
    When the manifest is missing or has no hazard_rules, every hazard
    defaults to ``"upper"`` -- the most common direction in industrial
    control hazards ("the bigger, the worse").
    """
    mf = _load_manifest_payload(manifest_path)
    if mf is None:
        return {h: "upper" for h in hazard_keys}

    rules = mf.get("hazard_rules", [])
    if not isinstance(rules, list):
        return {h: "upper" for h in hazard_keys}

    out: Dict[str, str] = {}
    for hkey in hazard_keys:
        matched_dir = ""
        plain = _hazard_var_from_key(hkey)
        hinted = _hazard_key_direction_hint(hkey)
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            rid = str(rule.get("id", "")).strip()
            rvar = str(rule.get("var", "")).strip()
            direction = str(rule.get("direction", "")).strip().lower()
            if not rid and not rvar:
                continue
            id_match = (
                bool(rid)
                and (hkey == rid or hkey.startswith(rid) or rid.startswith(hkey)
                     or hkey in rid or rid in hkey)
            )
            var_match = bool(rvar) and (
                plain == rvar or plain.startswith(rvar) or rvar.startswith(plain)
            )
            if not id_match and not var_match:
                continue
            if hinted and direction and direction != hinted:
                continue
            if id_match or var_match:
                matched_dir = direction
                break
        out[hkey] = matched_dir or "upper"
    return out


def _load_layers_from_manifest(manifest_path: Optional[Path]) -> Dict[str, str]:
    """Map injection_point name -> layer label declared in the manifest.

    Returns an empty dict if the manifest is missing or has no layer
    annotations.  Layer is one of: ``actuator``, ``setpoint``,
    ``cascade_setpoint`` (alias of setpoint).  Anything else passes through
    verbatim so a dataset can introduce its own taxonomy without code
    changes; the layer filter only matches strings exactly.
    """
    mf = _load_manifest_payload(manifest_path)
    if mf is None:
        return {}
    out: Dict[str, str] = {}
    for ip in mf.get("injection_points", []):
        if not isinstance(ip, Mapping):
            continue
        name = str(ip.get("name", "")).strip()
        layer = str(ip.get("layer", "")).strip().lower()
        if name and layer:
            out[name] = layer
    return out


def _filter_candidates_by_layer(
    candidates: Sequence[str],
    layer_by_name: Mapping[str, str],
    layer_filter: str,
) -> List[str]:
    """Keep only candidates whose declared layer matches ``layer_filter``.

    ``layer_filter`` accepts:
      - ``"all"`` / ``""`` / None  -> no filtering
      - ``"actuator"``               -> only actuator-layer points
      - ``"setpoint"``               -> only setpoint-layer points
      - any other literal            -> exact match (layer label as-is)

    Candidates without a declared layer are kept under ``"all"`` and
    dropped under any specific filter (since their layer is unknown).
    """
    f = (layer_filter or "all").strip().lower()
    if f in ("all", ""):
        return list(candidates)
    return [c for c in candidates if layer_by_name.get(c, "").lower() == f]


def generate_initial_combo(
    *,
    semantic_model_path: Path,
    matrix_path: Optional[Path] = None,
    top_hazard_n: int = 5,
    manifest_path: Optional[Path] = None,
    layer_filter: str = "actuator",
) -> List[Dict[str, Any]]:
    """Generate the single-point initial combo consumed by later stages."""
    model = load_semantic_model(semantic_model_path)

    if matrix_path is not None and Path(matrix_path).exists():
        matrix = load_manip_hazard_matrix(matrix_path)
    else:
        matrix = build_matrix_from_sensitivity_edges(semantic_model_path)

    manifest_injection_names: Optional[Set[str]] = None
    manifest_injection_fields: Optional[Set[str]] = None
    mf = _load_manifest_payload(manifest_path)
    if mf is not None:
        manifest_injection_names = {
            str(ip.get("name", "")).strip()
            for ip in mf.get("injection_points", [])
            if isinstance(ip, Mapping) and str(ip.get("name", "")).strip()
        }
        manifest_injection_fields = {
            str(ip.get("field", "")).strip()
            for ip in mf.get("injection_points", [])
            if isinstance(ip, Mapping) and str(ip.get("field", "")).strip()
        }

    semantic_candidates = collect_semantic_candidates(
        model,
        manifest_injection_names,
        manifest_injection_fields,
    )
    candidates = intersect_with_matrix(semantic_candidates, matrix.points)
    if not candidates:
        raise ValueError("no valid candidates after semantic filter and matrix intersection")

    # Layer filter (default actuator-only).  Falls back to "all" when the
    # manifest doesn't declare any layer annotations, so older manifests
    # keep working without modification.
    layer_by_name = _load_layers_from_manifest(manifest_path)
    if layer_by_name:
        filtered = _filter_candidates_by_layer(candidates, layer_by_name, layer_filter)
        if not filtered:
            raise ValueError(
                f"no candidates remain after applying layer_filter={layer_filter!r}; "
                f"manifest layers seen: {sorted(set(layer_by_name.values()))}"
            )
        candidates = filtered

    pi_loops = _load_pi_loops(manifest_path) if manifest_path else []
    hazard_directions = _load_hazard_directions(manifest_path, matrix.hazard_keys)
    scored = score_candidates(
        candidates, matrix, top_hazard_n=top_hazard_n, pi_loops=pi_loops,
        hazard_directions=hazard_directions,
    )
    # Stamp the layer onto each scored row so audit consumers see it.
    if layer_by_name:
        for row in scored:
            row["layer"] = layer_by_name.get(str(row.get("point", "")), "")
    top1 = pick_top1(scored)
    source_path = matrix_path if matrix_path is not None else semantic_model_path
    generated = build_initial_combo(top1, source_path)
    published_path = _published_initial_combo_path(semantic_model_path)
    if matrix_path is None and published_path is not None:
        published = _load_published_initial_combo(published_path)
        if published and _should_prefer_published_initial_combo(
            semantic_model_path,
            generated,
            published,
        ):
            return published
    return generated


def run(
    *,
    semantic_model: Path,
    matrix_file: Optional[Path] = None,
    matrix_output: Optional[Path] = None,
    output: Path,
    top_hazard_n: int = 5,
    manifest_path: Optional[Path] = None,
    layer_filter: str = "actuator",
) -> Path:
    """Generate and write the Stage 2 initial combo JSON artifact."""
    effective_matrix_path = Path(matrix_file) if matrix_file else None
    if effective_matrix_path is None:
        built_matrix = build_matrix_from_sensitivity_edges(Path(semantic_model))
        effective_matrix_path = write_physical_sensitivity_matrix(
            Path(matrix_output) if matrix_output else _default_physical_sensitivity_matrix_path(Path(semantic_model)),
            built_matrix,
        )
    combos = generate_initial_combo(
        semantic_model_path=Path(semantic_model),
        matrix_path=effective_matrix_path,
        top_hazard_n=max(1, int(top_hazard_n)),
        manifest_path=Path(manifest_path) if manifest_path else None,
        layer_filter=str(layer_filter),
    )
    out_file = Path(output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(combos, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_file


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Stage 1 hazard-driving selection."""
    parser = argparse.ArgumentParser(
        description=(
            "Select the top-1 hazard-driving manipulation point from a semantic model."
        )
    )
    parser.add_argument("--semantic-model", required=True, help="Path to the stage1 semantic model JSON")
    parser.add_argument(
        "--matrix-file",
        default="",
        help="Optional path to a precomputed manip-hazard matrix JSON",
    )
    parser.add_argument(
        "--matrix-output",
        default="",
        help="Path to write the generated physical sensitivity matrix JSON when --matrix-file is omitted",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional path to system_manifest.json",
    )
    parser.add_argument("--output", required=True, help="Path to write initial_combo.json")
    parser.add_argument(
        "--top-hazard-n",
        type=int,
        default=5,
        help="Keep the top-N hazard entries for explanation",
    )
    parser.add_argument(
        "--layer-filter",
        default="actuator",
        help="Manifest layer filter: actuator, setpoint, all, or a literal layer name",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    matrix_file = Path(args.matrix_file) if str(args.matrix_file).strip() else None
    matrix_output = Path(args.matrix_output) if str(args.matrix_output).strip() else None
    manifest_file = Path(args.manifest) if str(args.manifest).strip() else None
    run(
        semantic_model=Path(args.semantic_model),
        matrix_file=matrix_file,
        matrix_output=matrix_output,
        output=Path(args.output),
        top_hazard_n=max(1, int(args.top_hazard_n)),
        manifest_path=manifest_file,
        layer_filter=str(args.layer_filter),
    )
    return 0


__all__ = [
    "SemanticModelView",
    "MatrixView",
    "load_semantic_model",
    "load_manip_hazard_matrix",
    "collect_semantic_candidates",
    "intersect_with_matrix",
    "score_candidates",
    "pick_top1",
    "build_initial_combo",
    "generate_initial_combo",
    "run",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

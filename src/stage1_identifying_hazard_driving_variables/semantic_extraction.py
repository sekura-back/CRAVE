"""Extract semantic model M=(D,G,P,H) from Python control-program source.

Supports the boiler and Tennessee Eastman closed-loop targets.
P/H rules are read from declarative ALARM_RULES / HAZARD_RULES blocks.
D metadata uses workbook defaults first and manifest injection points as fallback.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import keyword
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import openpyxl


# ============================================================================
# Target specifications
# ============================================================================

def _artifact_root_from_module() -> Path:
    """Return the artifact root containing this script."""
    return Path(__file__).resolve().parents[2]


def _default_root() -> Path:
    """Use the artifact root as the default workspace root."""
    return _artifact_root_from_module()


def _default_output_path(root: Path, target: str) -> Path:
    """Return the artifact-aligned default Stage 1 semantic-model path."""
    root_path = Path(root)
    if target == "boiler":
        filename = "boiler_program_extraction.json"
    elif target == "TE":
        filename = "semantic_model_te.json"
    else:
        filename = f"semantic_model_{target}.json"
    return root_path / "results" / "stage1" / filename


def _artifact_relative_label(path: Path, root: Path) -> str:
    """Return an artifact-relative path label when possible."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


@dataclass(frozen=True)
class TargetPolicy:
    manifest_alias_map: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    blacklist_patterns: Tuple[str, ...] = ()
    phys_edge_whitelist: Set[Tuple[str, str]] = field(default_factory=set)
    sensitivity_source_alias_map: Mapping[str, str] = field(default_factory=dict)
    sensitivity_edge_whitelist: Set[Tuple[str, str]] = field(default_factory=set)
    required_extra_edges: Tuple[Tuple[str, str, str, str], ...] = ()

    @property
    def required_extra_edge_keys(self) -> Set[Tuple[str, str]]:
        return {(x, y) for x, y, _lam, _source in self.required_extra_edges}


@dataclass(frozen=True)
class TargetSpec:
    loop_file: str
    controllers_dir: str
    variable_table: str
    extra_scan_dirs: Tuple[str, ...] = ()
    policy: TargetPolicy = field(default_factory=TargetPolicy)


TE_MANIFEST_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "simulation_sp_purge": ("sp_purge_rate",),
    "simulation_sp_purge_rate": ("sp_purge_rate",),
    "simulation_sp_recycle_flow": ("sp_recycle_flow",),
    "simulation_sp_separator_level": ("sp_separator_level",),
    "simulation_sp_steam": ("sp_steam_flow",),
    "simulation_sp_steam_flow": ("sp_steam_flow",),
    "simulation_sp_cw": ("sp_reactor_cw_temp",),
    "simulation_sp_strip_temp": ("sp_stripper_temp",),
}


# ============================================================================
# Variable-table parsing
# ============================================================================

def _parse_range_str(s: Any) -> Optional[Tuple[float, float]]:
    """Parse a '[lo, hi]' or '[lo-hi]' range string."""
    if s is None:
        return None
    s = str(s).strip().strip("[]() ")
    if not s:
        return None
    if "," in s:
        parts = s.split(",")
        if len(parts) == 2:
            try:
                return (float(parts[0].strip()), float(parts[1].strip()))
            except ValueError:
                return None
    m = re.match(r"(-?[\d.]+)\s*[-–]\s*(-?[\d.]+)", s)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except ValueError:
            return None
    return None


def _parse_rate_str(s: Any) -> Optional[float]:
    """Parse a '[-r, r]' rate string and return max(abs)."""
    rng = _parse_range_str(s)
    if rng is None:
        return None
    return max(abs(rng[0]), abs(rng[1]))


def parse_variable_table(xlsx_path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse a variable-table workbook.

    Returns a {variable: {default, range, rate}} mapping. If the workbook path is
    empty or missing, callers fall back to ``system_manifest.json`` injection
    points as the metadata source.
    """
    if str(xlsx_path).strip() in ("", "."):
        return {}
    if not xlsx_path.exists():
        return {}
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if len(rows) < 2:
        return {}

    header = [str(h).strip().lower() if h else "" for h in rows[0]]
    col_name = -1
    col_default = -1
    col_range = -1
    col_rate = -1
    for i, h in enumerate(header):
        if h in ("variable", "variable_name", "name"):
            col_name = i
        elif h in ("default", "deafult"):
            col_default = i
        elif h == "range":
            col_range = i
        elif h == "rate":
            col_rate = i

    if col_name < 0:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows[1:]:
        if not row or col_name >= len(row) or not row[col_name]:
            continue
        name = str(row[col_name]).strip()
        entry: Dict[str, Any] = {}
        if col_default >= 0 and col_default < len(row) and row[col_default] is not None:
            try:
                entry["default"] = float(row[col_default])
            except (ValueError, TypeError):
                pass
        if col_range >= 0 and col_range < len(row) and row[col_range] is not None:
            rng = _parse_range_str(row[col_range])
            if rng is not None:
                entry["range"] = list(rng)
        if col_rate >= 0 and col_rate < len(row) and row[col_rate] is not None:
            rate = _parse_rate_str(row[col_rate])
            if rate is not None:
                entry["rate"] = rate
        result[name] = entry
    return result


# ============================================================================
# Declarative P/H extraction
# ============================================================================

def _extract_rules_from_class(tree: ast.AST, rule_attr_name: str) -> List[Dict[str, Any]]:
    """Extract the named declarative rule attribute from all classes in an AST."""
    rules: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == rule_attr_name:
                        try:
                            value = ast.literal_eval(stmt.value)
                            if isinstance(value, list):
                                rules.extend(value)
                        except (ValueError, TypeError):
                            pass
    return rules


def _extract_rules_from_module(tree: ast.AST, rule_attr_name: str) -> List[Dict[str, Any]]:
    """Extract the named declarative rule block from module scope."""
    rules: List[Dict[str, Any]] = []
    for stmt in ast.iter_child_nodes(tree):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == rule_attr_name:
                    try:
                        value = ast.literal_eval(stmt.value)
                        if isinstance(value, list):
                            rules.extend(value)
                    except (ValueError, TypeError):
                        pass
    return rules


def extract_alarm_rules(files: Sequence[Path]) -> List[Dict[str, Any]]:
    """Extract ALARM_RULES, the protection predicates P, from all files."""
    all_rules: List[Dict[str, Any]] = []
    for f in files:
        source = f.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(f))
        all_rules.extend(_extract_rules_from_class(tree, "ALARM_RULES"))
        all_rules.extend(_extract_rules_from_module(tree, "ALARM_RULES"))
    return all_rules


def extract_hazard_rules(files: Sequence[Path]) -> List[Dict[str, Any]]:
    """Extract HAZARD_RULES, the hazard predicates H, from all files."""
    all_rules: List[Dict[str, Any]] = []
    for f in files:
        source = f.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(f))
        all_rules.extend(_extract_rules_from_class(tree, "HAZARD_RULES"))
        all_rules.extend(_extract_rules_from_module(tree, "HAZARD_RULES"))
    return all_rules


def _load_rules_from_manifest(
    controllers_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fallback rule loader: read alarm_rules / hazard_rules from manifest.

    Used when the control program does not contain declarative
    ``ALARM_RULES`` / ``HAZARD_RULES`` blocks (e.g. third-party PI stacks
    such as the bundled TEP DecentralizedController). Returns ``([], [])``
    if the manifest is missing or malformed.
    """
    manifest_path = Path(controllers_dir) / "system_manifest.json"
    if not manifest_path.exists():
        return [], []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return [], []

    p_rules: List[Dict[str, Any]] = []
    for item in manifest.get("alarm_rules", []):
        if isinstance(item, Mapping):
            p_rules.append(dict(item))
    h_rules: List[Dict[str, Any]] = []
    for item in manifest.get("hazard_rules", []):
        if isinstance(item, Mapping):
            h_rules.append(dict(item))
    return p_rules, h_rules


def _load_hazard_ids_by_measurement(root: Path, target: str) -> Dict[str, List[str]]:
    """Map physical-process measurement names to manifest hazard ids."""
    spec = _get_target_spec(target)
    _p_rules, h_rules = _load_rules_from_manifest((root / spec.controllers_dir).resolve())
    out: Dict[str, List[str]] = {}
    for item in h_rules:
        if not isinstance(item, Mapping):
            continue
        hid = str(item.get("id", "")).strip()
        var = str(item.get("var", "")).strip()
        if not hid or not var:
            continue
        for name in (f"physicalprocess_{var}", f"pyhsicalprocess_{var}"):
            bucket = out.setdefault(name, [])
            if hid not in bucket:
                bucket.append(hid)
    return out


# ============================================================================
# Variable filtering
# ============================================================================

BUILTIN_NAMES = set(dir(builtins))
IGNORE_BASE_NAMES = {
    "self", "cls", "np", "pd", "torch", "math", "json", "Path",
    "List", "Dict", "Tuple", "Optional", "Any", "Sequence", "Iterable",
    "set", "list", "dict", "tuple", "float", "int", "bool", "str", "print",
    "Mapping", "Set", "Callable", "Union",
}

COMMON_BLACKLIST_PATTERNS: Tuple[str, ...] = (
    r"^record", r"^trace$", r"^meta$", r"^stop_", r"^steps_",
    r"^overrides$", r"^cmd_applied$", r"^ctrl_work$",
    r"^csv_", r"^_csv_", r"^_next_step$",
    r"^t_step$", r"^t_time_s$",
    r"^last_debug$", r".*_dbg$",
    r"^controller_modified$", r"^actuators_modified$", r"^setpoints_modified$",
    r"^injection_hook$", r"^hook$", r"^payload$",
    r"^strategies$", r"^strategy$", r"^strategy_idx$",
    r"^return_trace$", r"^stop_on_trip$", r"^enable_csv$",
    r"^ignored_for_", r"^enforce_",
    r"^_state_min$", r"^_state_max$",
    r"^changed$", r"^mod$", r"^trip_type$",
    r"^out$", r"^out_path$", r"^parser$", r"^args$",
    r"^row$", r"^rows$", r"^idx$", r"^key$", r"^value$",
    r"^result$", r"^output$", r"^runner$", r"^sim$",
    r".*TRACE_KEYS$", r".*_KEYS$",
    r".*last_record$", r".*last_output$",
    r".*_record_", r".*_records$",
    r"^controller_controllers_name$",
    r"^controller_ctrl$",
    r"^controller_ctrl_err_old$",
    r"^controller_ctrl_names$",
    r"^controller_ctrl_setpoint$",
    r"^controller_name$",
    r"^controller_state_err_old$",
    r"^controller_state_setpoint$",
)

TE_BLACKLIST_PATTERNS: Tuple[str, ...] = (
    r"^simulation_ALARM_.*$",
    r"^simulation_DEFAULT_MODE1_SNAPSHOT$",
    r"^simulation_Path_file_with_name$",
    r"^simulation_alarm_thr_purge$",
    r"^simulation_alarm_thr_recycle$",
    r"^simulation_alarm_thr_sep_level$",
    r"^simulation_env_float$",
    r"^simulation_handle$",
    r"^simulation_meas_purge_rate$",
    r"^simulation_meas_recycle_flow$",
    r"^simulation_meas_separator_level$",
    r"^simulation_name$",
    r"^simulation_new_state_xmeas_05$",
    r"^simulation_new_state_xmeas_10$",
    r"^simulation_new_state_xmeas_12$",
    r"^simulation_new_xmv_9$",
    r"^simulation_os_environ_get$",
    r"^simulation_pickle_load$",
    r"^simulation_raw$",
    r"^simulation_sep_dev_exceeded$",
    r"^simulation_snapshot$",
    r"^simulation_sp_4$",
    r"^simulation_sp_5$",
    r"^simulation_sp_6$",
    r"^simulation_sp_purge_rate$",
    r"^simulation_sp_recycle_flow$",
    r"^simulation_sp_separator_level$",
)


def _compile_blacklist_regexes(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


COMMON_BLACKLIST_RE = _compile_blacklist_regexes(COMMON_BLACKLIST_PATTERNS)


def _is_noise_symbol(name: str) -> bool:
    """Return whether a symbol name is noise."""
    if not name:
        return True
    root = name.split(".", 1)[0].split("[", 1)[0]
    if root in IGNORE_BASE_NAMES:
        return True
    if root in BUILTIN_NAMES:
        return True
    if keyword.iskeyword(root):
        return True
    return False


def _is_blacklisted(
    name: str,
    blacklist_res: Optional[Sequence[re.Pattern[str]]] = None,
) -> bool:
    """Return whether a variable name is blacklisted."""
    if blacklist_res is None:
        blacklist_res = COMMON_BLACKLIST_RE
    for rx in blacklist_res:
        if rx.match(name):
            return True
    return False


TE_PHYS_EDGE_WHITELIST: Set[Tuple[str, str]] = {
    ("simulation_sp_a_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_12"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_15"),
    ("simulation_sp_a_feed", "simulation_alarm_thr_a"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_08"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_12"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_15"),
    ("simulation_sp_ac_feed", "simulation_alarm_thr_ac"),
    ("simulation_sp_cw", "physicalprocess_xmeas_07"),
    ("simulation_sp_cw", "physicalprocess_xmeas_08"),
    ("simulation_sp_cw", "physicalprocess_xmeas_09"),
    ("simulation_sp_cw", "physicalprocess_xmeas_12"),
    ("simulation_sp_cw", "physicalprocess_xmeas_15"),
    ("simulation_sp_d_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_d_feed", "physicalprocess_xmeas_08"),
    ("simulation_sp_d_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_d_feed", "simulation_alarm_thr_d"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_08"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_12"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_15"),
    ("simulation_sp_e_feed", "simulation_alarm_thr_e"),
    ("simulation_sp_purge", "physicalprocess_xmeas_07"),
    ("simulation_sp_purge", "physicalprocess_xmeas_08"),
    ("simulation_sp_purge", "physicalprocess_xmeas_09"),
    ("simulation_sp_purge", "physicalprocess_xmeas_12"),
    ("simulation_sp_purge", "physicalprocess_xmeas_15"),
    ("simulation_sp_steam", "physicalprocess_xmeas_07"),
    ("simulation_sp_strip_temp", "physicalprocess_xmeas_07"),
    ("simulation_xmv_02", "physicalprocess_xmeas_07"),
    ("simulation_xmv_04", "physicalprocess_xmeas_07"),
    ("simulation_xmv_05", "physicalprocess_xmeas_07"),
    ("simulation_xmv_06", "physicalprocess_xmeas_07"),
    ("simulation_xmv_07", "physicalprocess_xmeas_07"),
    ("simulation_xmv_07", "physicalprocess_xmeas_12"),
    ("simulation_xmv_07", "physicalprocess_xmeas_15"),
    ("simulation_xmv_08", "physicalprocess_xmeas_07"),
    ("simulation_xmv_10", "physicalprocess_xmeas_07"),
    ("simulation_xmv_11", "physicalprocess_xmeas_07"),
}

TE_SENSITIVITY_SOURCE_ALIAS_MAP: Dict[str, str] = {
    "simulation_sp_purge_rate": "simulation_sp_purge",
    "simulation_sp_reactor_cw_temp": "simulation_sp_cw",
    "simulation_sp_steam_flow": "simulation_sp_steam",
}

TE_SENSITIVITY_EDGE_WHITELIST: Set[Tuple[str, str]] = {
    ("simulation_xmv_02", "physicalprocess_xmeas_07"),
    ("simulation_xmv_04", "physicalprocess_xmeas_07"),
    ("simulation_xmv_05", "physicalprocess_xmeas_07"),
    ("simulation_xmv_06", "physicalprocess_xmeas_07"),
    ("simulation_xmv_07", "physicalprocess_xmeas_07"),
    ("simulation_xmv_07", "physicalprocess_xmeas_12"),
    ("simulation_xmv_07", "physicalprocess_xmeas_15"),
    ("simulation_xmv_08", "physicalprocess_xmeas_07"),
    ("simulation_xmv_10", "physicalprocess_xmeas_07"),
    ("simulation_xmv_11", "physicalprocess_xmeas_07"),
    ("simulation_sp_d_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_d_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_d_feed", "physicalprocess_xmeas_08"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_08"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_12"),
    ("simulation_sp_e_feed", "physicalprocess_xmeas_15"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_12"),
    ("simulation_sp_a_feed", "physicalprocess_xmeas_15"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_07"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_09"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_08"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_12"),
    ("simulation_sp_ac_feed", "physicalprocess_xmeas_15"),
    ("simulation_sp_purge", "physicalprocess_xmeas_07"),
    ("simulation_sp_purge", "physicalprocess_xmeas_09"),
    ("simulation_sp_purge", "physicalprocess_xmeas_08"),
    ("simulation_sp_purge", "physicalprocess_xmeas_12"),
    ("simulation_sp_purge", "physicalprocess_xmeas_15"),
    ("simulation_sp_steam", "physicalprocess_xmeas_07"),
    ("simulation_sp_cw", "physicalprocess_xmeas_07"),
    ("simulation_sp_cw", "physicalprocess_xmeas_09"),
    ("simulation_sp_cw", "physicalprocess_xmeas_08"),
    ("simulation_sp_cw", "physicalprocess_xmeas_12"),
    ("simulation_sp_cw", "physicalprocess_xmeas_15"),
    ("simulation_sp_strip_temp", "physicalprocess_xmeas_07"),
}

TE_REQUIRED_EXTRA_EDGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("simulation_sp_a_feed", "simulation_alarm_thr_a", "data", "simulators/tennessee_eastman/simulation.py"),
    ("simulation_sp_ac_feed", "simulation_alarm_thr_ac", "data", "simulators/tennessee_eastman/simulation.py"),
    ("simulation_sp_d_feed", "simulation_alarm_thr_d", "data", "simulators/tennessee_eastman/simulation.py"),
    ("simulation_sp_e_feed", "simulation_alarm_thr_e", "data", "simulators/tennessee_eastman/simulation.py"),
    ("simulation_sp_strip_temp", "physicalprocess_xmeas_07", "data", "sensitivity_jacobian"),
)

DEFAULT_TARGET_POLICY = TargetPolicy(
    blacklist_patterns=COMMON_BLACKLIST_PATTERNS,
)

TARGET_POLICIES: Dict[str, TargetPolicy] = {
    "boiler": DEFAULT_TARGET_POLICY,
    "TE": TargetPolicy(
        manifest_alias_map=TE_MANIFEST_ALIAS_MAP,
        blacklist_patterns=COMMON_BLACKLIST_PATTERNS + TE_BLACKLIST_PATTERNS,
        phys_edge_whitelist=TE_PHYS_EDGE_WHITELIST,
        sensitivity_source_alias_map=TE_SENSITIVITY_SOURCE_ALIAS_MAP,
        sensitivity_edge_whitelist=TE_SENSITIVITY_EDGE_WHITELIST,
        required_extra_edges=TE_REQUIRED_EXTRA_EDGES,
    ),
}


TARGET_SPECS: Dict[str, TargetSpec] = {
    "boiler": TargetSpec(
        loop_file="simulators/boiler_ccs/simulation.py",
        controllers_dir="simulators/boiler_ccs",
        variable_table="",
        policy=TARGET_POLICIES["boiler"],
    ),
    "TE": TargetSpec(
        loop_file="simulators/tennessee_eastman/simulation.py",
        controllers_dir="simulators/tennessee_eastman",
        variable_table="",
        extra_scan_dirs=("simulators/tennessee_eastman/tep",),
        policy=TARGET_POLICIES["TE"],
    ),
}


def _get_target_policy(target: str) -> TargetPolicy:
    return _get_target_spec(target).policy


def _get_target_spec(target: str) -> TargetSpec:
    return TARGET_SPECS[target]


def _normalize_target_sensitivity_edges(
    edges: Sequence[Dict[str, Any]],
    policy: TargetPolicy,
) -> List[Dict[str, Any]]:
    if not policy.sensitivity_edge_whitelist:
        return [dict(edge) for edge in edges]
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for edge in edges:
        rec = dict(edge)
        src = policy.sensitivity_source_alias_map.get(str(rec.get("x", "")), str(rec.get("x", "")))
        dst = str(rec.get("y", ""))
        rec["x"] = src
        key2 = (src, dst)
        key3 = (src, dst, str(rec.get("lambda", "")))
        if key2 not in policy.sensitivity_edge_whitelist:
            continue
        if key3 in seen:
            continue
        seen.add(key3)
        out.append(rec)
    return out


def _append_target_required_edges(
    edges: Sequence[Dict[str, Any]],
    policy: TargetPolicy,
) -> List[Dict[str, Any]]:
    if not policy.required_extra_edges:
        return [dict(edge) for edge in edges]
    out = [dict(edge) for edge in edges]
    seen = {
        (
            str(edge.get("x", "")),
            str(edge.get("y", "")),
            str(edge.get("lambda", "")),
            str(edge.get("source", {}).get("file", "")) if isinstance(edge.get("source"), dict) else "",
        )
        for edge in out
    }
    for x, y, lam, source_file in policy.required_extra_edges:
        key = (x, y, lam, source_file)
        if key in seen:
            continue
        out.append(
            {
                "x": x,
                "y": y,
                "lambda": lam,
                "eta": 1.0,
                "epsilon": 1.0,
                "source": {"file": source_file, "line": 0},
            }
        )
        seen.add(key)
    return out


PHYS_PREFIXES = ("pyhsicalprocess_", "physicalprocess_", "process_model_backend_")


def _is_phys_var(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in PHYS_PREFIXES)


def _filter_target_physical_edges(
    edges: Sequence[Dict[str, Any]],
    policy: TargetPolicy,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for edge in edges:
        src = str(edge.get("x", ""))
        dst = str(edge.get("y", ""))
        if (src, dst) in policy.phys_edge_whitelist:
            out.append(edge)
            continue
        if _is_phys_var(src) or _is_phys_var(dst):
            continue
        out.append(edge)
    return out


# ============================================================================
# AST helper functions
# ============================================================================

EDGE_TYPES = {"data", "guard", "call"}


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _read_source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _sanitize_token(text: str) -> str:
    text = text.replace('"', "").replace("'", "")
    text = text.replace("[", "_").replace("]", "")
    text = text.replace(".", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def _get_expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_get_expr_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        base = _get_expr_name(node.value)
        if isinstance(node.value, ast.Name) and node.value.id in {
            "row", "controller_outputs", "physical_vars",
            "sensors", "cmd_applied", "ctrl_work",
        }:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return str(node.slice.value)
        if isinstance(node.slice, ast.Constant):
            return f"{base}[{node.slice.value}]"
        return f"{base}[{_safe_unparse(node.slice)}]"
    return _safe_unparse(node)


def _collect_var_refs(node: Optional[ast.AST]) -> Set[str]:
    refs: Set[str] = set()
    if node is None:
        return refs

    def walk_expr(expr: ast.AST) -> None:
        if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
            refs.add(_get_expr_name(expr))
            return
        for child in ast.iter_child_nodes(expr):
            walk_expr(child)

    walk_expr(node)
    return refs


def _iter_targets(target: ast.AST):
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_targets(elt)
    else:
        yield target


def _infer_type_from_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "real"
    return None


# ============================================================================
# Variable entries
# ============================================================================

@dataclass
class VariableEntry:
    name: str
    aliases: Set[str] = field(default_factory=set)
    type: Optional[str] = None
    scope: str = "local"
    attackable: bool = False
    default_value: Any = None
    rate: Optional[float] = None
    range_lower: Optional[float] = None
    range_upper: Optional[float] = None
    sources: Set[Tuple[str, int]] = field(default_factory=set)

    def to_json(self) -> Dict[str, Any]:
        rng = None
        if self.range_lower is not None or self.range_upper is not None:
            rng = [self.range_lower, self.range_upper]
        return {
            "aliases": sorted(self.aliases),
            "name": self.name,
            "type": self.type,
            "scope": self.scope,
            "attackable": self.attackable,
            "default_value": self.default_value,
            "rate": self.rate,
            "range": rng,
            "sources": [
                {"file": f, "line": ln}
                for f, ln in sorted(self.sources, key=lambda x: (x[0], x[1]))
            ],
        }


# ============================================================================
# AST collector
# ============================================================================

class SemanticCollector(ast.NodeVisitor):
    """Walk a Python source AST and collect variables plus dependency edges."""

    def __init__(self, root: Path, file_path: Path):
        self.root = root
        self.file_path = file_path
        self.file_rel = file_path.relative_to(root).as_posix()
        self.entity = _sanitize_token(file_path.stem)
        self.scope_stack: List[str] = []

        self.variables: Dict[str, VariableEntry] = {}
        self.edges: List[Dict[str, Any]] = []

    def _scope(self) -> str:
        return "global" if not self.scope_stack else "local"

    def _norm(self, alias: str) -> str:
        return f"{self.entity}_{_sanitize_token(alias)}"

    def _is_ignored(self, name: str) -> bool:
        return _is_noise_symbol(name) or _is_blacklisted(name)

    def _register_var(
        self, alias: str, node: ast.AST,
        inferred_type: Optional[str] = None, default_value: Any = None,
    ) -> Optional[str]:
        if not alias or self._is_ignored(alias):
            return None
        norm = self._norm(alias)
        if _is_blacklisted(norm):
            return None
        ent = self.variables.get(norm)
        if ent is None:
            ent = VariableEntry(name=norm)
            self.variables[norm] = ent
        ent.aliases.add(alias)
        ent.sources.add((self.file_rel, getattr(node, "lineno", 0)))
        if self._scope() == "global":
            ent.scope = "global"
        if ent.type is None and inferred_type is not None:
            ent.type = inferred_type
        if ent.default_value is None and default_value is not None:
            ent.default_value = default_value
        return norm

    def _register_edge(self, src_alias: str, dst_alias: str, edge_type: str, node: ast.AST) -> None:
        if edge_type not in EDGE_TYPES:
            return
        if self._is_ignored(src_alias) or self._is_ignored(dst_alias):
            return
        src = self._register_var(src_alias, node)
        dst = self._register_var(dst_alias, node)
        if not src or not dst or src == dst:
            return
        self.edges.append({
            "x": src, "y": dst, "lambda": edge_type,
            "eta": 1.0, "epsilon": 1.0,
            "source": {"file": self.file_rel, "line": getattr(node, "lineno", 0)},
        })

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> Any:
        default = node.value.value if isinstance(node.value, ast.Constant) else None
        rhs_refs = _collect_var_refs(node.value)
        value_type = _infer_type_from_value(default)
        for tgt_root in node.targets:
            for tgt in _iter_targets(tgt_root):
                t_alias = _get_expr_name(tgt)
                self._register_var(t_alias, node, inferred_type=value_type, default_value=default)
                for src in rhs_refs:
                    self._register_edge(src, t_alias, "data", node)
                if isinstance(node.value, ast.Call):
                    for src in _collect_var_refs(node.value):
                        self._register_edge(src, t_alias, "call", node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        for tgt in _iter_targets(node.target):
            t_alias = _get_expr_name(tgt)
            self._register_var(t_alias, node)
            self._register_edge(t_alias, t_alias, "data", node)
            for src in _collect_var_refs(node.value):
                self._register_edge(src, t_alias, "data", node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        test_refs = _collect_var_refs(node.test)
        body_targets: Set[str] = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    for elt in _iter_targets(t):
                        body_targets.add(_get_expr_name(elt))
        for src in test_refs:
            for dst in body_targets:
                self._register_edge(src, dst, "guard", node)
        self.generic_visit(node)


# ============================================================================
# Edge weight computation
# ============================================================================

def _infer_edge_eta(edge: Dict[str, Any]) -> float:
    src = str(edge.get("x", "")).lower()
    dst = str(edge.get("y", "")).lower()
    text = f"{src} {dst}"
    if str(edge.get("lambda", "")).lower() == "guard":
        return 1.0
    pid_tokens = ("pid", "integral", "lag", "last_", "washout")
    if any(tok in text for tok in pid_tokens):
        return 0.6
    rate_tokens = ("rate", "slew", "clip", "sat", "limit", "clamp")
    if any(tok in text for tok in rate_tokens):
        return 0.8
    return 1.0


def _raw_epsilon_score(edge: Dict[str, Any]) -> float:
    edge_type = str(edge.get("lambda", "data")).lower()
    return {"data": 1.0, "call": 0.8, "guard": 0.5}.get(edge_type, 1.0)


def _dedup_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for e in edges:
        key = (e.get("x"), e.get("y"), e.get("lambda"),
               e.get("source", {}).get("file"), e.get("source", {}).get("line"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _apply_semantic_edge_weights(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not edges:
        return []
    enriched: List[Dict[str, Any]] = []
    raw_by_dst_src: Dict[Tuple[str, str], float] = {}
    sum_by_dst: Dict[str, float] = defaultdict(float)

    for edge in edges:
        rec = dict(edge)
        src = str(rec.get("x", ""))
        dst = str(rec.get("y", ""))
        if not src or not dst:
            continue

        # Keep sensitivity-edge weights as generated.
        source_file = rec.get("source", {}).get("file", "") if isinstance(rec.get("source"), dict) else ""
        if source_file == "sensitivity_jacobian":
            enriched.append(rec)
            continue

        eta = _infer_edge_eta(rec)
        raw_eps = _raw_epsilon_score(rec)
        rec["eta"] = max(min(eta, 1.0), 1e-6)
        rec["_raw_eps"] = raw_eps
        enriched.append(rec)
        key = (dst, src)
        if raw_eps > raw_by_dst_src.get(key, 0.0):
            raw_by_dst_src[key] = raw_eps

    for (dst, _), raw in raw_by_dst_src.items():
        sum_by_dst[dst] += raw

    out: List[Dict[str, Any]] = []
    for rec in enriched:
        # Keep sensitivity edges directly.
        source_file = rec.get("source", {}).get("file", "") if isinstance(rec.get("source"), dict) else ""
        if source_file == "sensitivity_jacobian":
            out.append(rec)
            continue

        src = str(rec.get("x", ""))
        dst = str(rec.get("y", ""))
        denom = sum_by_dst.get(dst, 0.0)
        eps = raw_by_dst_src.get((dst, src), 0.0) / denom if denom > 0 else 1.0
        rec["epsilon"] = max(min(eps, 1.0), 1e-6)
        rec.pop("_raw_eps", None)
        out.append(rec)
    return out


# ============================================================================
# File collection
# ============================================================================

def _collect_files(loop_file: Path, controllers_dir: Path) -> List[Path]:
    """Collect Python source files to analyze, excluding physical-process models.

    Physical-process models are treated as black boxes and skipped by AST analysis.
    Their relationship to control code is captured through finite-difference sensitivity Jacobians.
    """
    # Physical-process filename patterns to exclude.
    _PHYS_FILE_PATTERNS = {
        "pyhsicalprocess.py",
        "physicalprocess.py",
        "physical_process.py",
        "process_model_backend.py",
        "process_model.py",
    }

    files: List[Path] = [loop_file]
    # Python files in the same directory, excluding physical-process modules.
    for p in sorted(controllers_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        if p.name.lower() in _PHYS_FILE_PATTERNS:
            continue
        if p.resolve() != loop_file.resolve():
            files.append(p)
    # Deduplicate.
    seen: Set[Path] = set()
    uniq: List[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


# ============================================================================
# Variable-table override
# ============================================================================

def _apply_variable_table(
    variables: Dict[str, VariableEntry],
    var_table: Dict[str, Dict[str, Any]],
) -> Set[str]:
    """Overlay variable-table metadata onto registered variables and return matched normalized names.

    Variables unmatched by the table are registered directly as attackable variables.
    """
    matched: Set[str] = set()
    for raw_name, table_entry in var_table.items():
        found = False
        # Find a match by scanning aliases for raw_name.
        for norm, ent in variables.items():
            if raw_name in ent.aliases or raw_name == norm:
                if "default" in table_entry:
                    ent.default_value = table_entry["default"]
                if "range" in table_entry:
                    ent.range_lower = table_entry["range"][0]
                    ent.range_upper = table_entry["range"][1]
                if "rate" in table_entry:
                    ent.rate = table_entry["rate"]
                ent.attackable = True
                matched.add(norm)
                found = True
                break

        if found:
            continue

        # Try suffix matching.
        sanitized = _sanitize_token(raw_name)
        for norm, ent in variables.items():
            if norm.endswith(f"_{sanitized}"):
                if "default" in table_entry:
                    ent.default_value = table_entry["default"]
                if "range" in table_entry:
                    ent.range_lower = table_entry["range"][0]
                    ent.range_upper = table_entry["range"][1]
                if "rate" in table_entry:
                    ent.rate = table_entry["rate"]
                ent.attackable = True
                matched.add(norm)
                found = True
                break

        if found:
            continue

        # No match: register a new variable directly.
        norm = raw_name
        ent = VariableEntry(name=norm)
        ent.aliases.add(raw_name)
        ent.scope = "global"
        ent.attackable = True
        if "default" in table_entry:
            ent.default_value = table_entry["default"]
        if "range" in table_entry:
            ent.range_lower = table_entry["range"][0]
            ent.range_upper = table_entry["range"][1]
        if "rate" in table_entry:
            ent.rate = table_entry["rate"]
        variables[norm] = ent
        matched.add(norm)

    return matched


def _apply_manifest_injection_points(
    variables: Dict[str, VariableEntry],
    controllers_dir: Path,
    policy: TargetPolicy,
) -> Set[str]:
    """Register manifest injection_points as attackable variables.

    Used when the static AST scan does not find a matching attackable
    candidate for an injection point declared in ``system_manifest.json``
    (typical for TE where xmv channels live inside a numpy array and
    never appear as bare identifiers in source).

    Each injection point's ``name`` becomes a variable with ``default``,
    ``range`` and ``rate`` lifted from the manifest entry.
    """
    manifest_path = Path(controllers_dir) / "system_manifest.json"
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    matched: Set[str] = set()
    for ip in manifest.get("injection_points", []):
        if not isinstance(ip, Mapping):
            continue
        name = str(ip.get("name", "")).strip()
        if not name:
            continue
        field = str(ip.get("field", "")).strip()
        candidates: List[str] = [name]
        if field:
            candidates.append(field)
        if name.startswith("simulation_"):
            candidates.append(name[len("simulation_"):])
        for alias in policy.manifest_alias_map.get(name, ()):
            candidates.append(alias)

        ent: Optional[VariableEntry] = variables.get(name)
        matched_name: Optional[str] = name if ent is not None else None
        if ent is None:
            for cand in candidates:
                if not cand:
                    continue
                sanitized = _sanitize_token(cand)
                for norm, cur in variables.items():
                    if (
                        cand == norm
                        or cand in cur.aliases
                        or norm.endswith(f"_{sanitized}")
                    ):
                        ent = cur
                        matched_name = norm
                        break
                if ent is not None:
                    break

        if ent is None:
            ent = VariableEntry(name=name)
            ent.aliases.add(name)
            ent.scope = "global"
            variables[name] = ent
            matched_name = name
        else:
            ent.aliases.add(name)
            if field:
                ent.aliases.add(field)
        ent.attackable = True
        if "default_value" in ip:
            try:
                ent.default_value = float(ip["default_value"])
            except (TypeError, ValueError):
                pass
        rng = ip.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                ent.range_lower = float(rng[0])
                ent.range_upper = float(rng[1])
            except (TypeError, ValueError):
                pass
        if "rate" in ip:
            try:
                ent.rate = float(ip["rate"])
            except (TypeError, ValueError):
                pass
        matched.add(str(matched_name or name))
    return matched


# ============================================================================
# Core build functions
# ============================================================================

def _merge_type(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    if current is None:
        return incoming
    if incoming is None:
        return current
    if current == incoming:
        return current
    numeric = {"bool", "int", "real"}
    if current in numeric and incoming in numeric:
        if "real" in (current, incoming):
            return "real"
        if "int" in (current, incoming):
            return "int"
        return "bool"
    return current


def _scan_files_for_vars_and_edges(
    root: Path, files: Sequence[Path],
) -> Tuple[Dict[str, VariableEntry], List[Dict[str, Any]]]:
    """Scan all files with AST and collect variables plus edges."""
    all_vars: Dict[str, VariableEntry] = {}
    all_edges: List[Dict[str, Any]] = []

    for file_path in files:
        source = _read_source_text(file_path)
        tree = ast.parse(source, filename=str(file_path))
        collector = SemanticCollector(root=root, file_path=file_path)
        collector.visit(tree)

        for name, ent in collector.variables.items():
            merged = all_vars.get(name)
            if merged is None:
                all_vars[name] = ent
                continue
            merged.aliases.update(ent.aliases)
            merged.sources.update(ent.sources)
            merged.type = _merge_type(merged.type, ent.type)
            if merged.scope != "global" and ent.scope == "global":
                merged.scope = "global"
            if merged.default_value is None and ent.default_value is not None:
                merged.default_value = ent.default_value

        all_edges.extend(collector.edges)

    return all_vars, all_edges


# ============================================================================
# Multi-step sensitivity Jacobian
# ============================================================================


def _build_sensitivity_config_from_manifest(root: Path, target: str) -> Optional[Dict[str, Any]]:
    """Build sensitivity configuration automatically from system_manifest.json.

    Used when SENSITIVITY_CONFIG has no hard-coded entry for the target system.
    Read the target manifest under simulators/{target}/system_manifest.json.
    """
    try:
        spec = _get_target_spec(target)
    except KeyError:
        return None

    controllers_dir = root / spec.controllers_dir
    manifest_path = controllers_dir / "system_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    pp = manifest.get("physical_process", {})
    control_inputs = pp.get("control_inputs", [])
    state_variables = pp.get("state_variables", [])
    hazard_rules = manifest.get("hazard_rules", [])
    injection_points = manifest.get("injection_points", [])
    init_params = pp.get("init_params", {})

    if not control_inputs or not state_variables:
        return None

    # u_vars / u_semantic_names: prefer manifest injection_points
    # field/name pairs, which are the precise mapping intended by the manifest. If injection
    # points are absent, fall back to inference from control_inputs.
    if injection_points:
        u_vars = [str(ip.get("field", "")) for ip in injection_points if ip.get("field")]
        u_semantic_names = [str(ip.get("name", "")) for ip in injection_points if ip.get("name")]
        # Require aligned lengths; missing entries are ignored.
        n = min(len(u_vars), len(u_semantic_names))
        u_vars = u_vars[:n]
        u_semantic_names = u_semantic_names[:n]
    else:
        u_vars = []
        for ci in control_inputs:
            name = str(ci)
            if name.endswith("_applied"):
                name = name[:-len("_applied")]
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                name = parts[0]
            u_vars.append(name)
        u_semantic_names = list(u_vars)

    # y_vars: physical variables referenced by hazard rules and present in state_variables.
    y_vars = []
    for hr in hazard_rules:
        var = hr.get("var", "")
        if var and var in state_variables and var not in y_vars:
            y_vars.append(var)
    if not y_vars:
        y_vars = list(state_variables)

    # y_semantic_names: physical-process module prefix plus variable name.
    pp_module = pp.get("module", "")
    pp_file_stem = pp_module.rsplit(".", 1)[-1] if "." in pp_module else pp_module
    y_semantic_names = [f"{pp_file_stem}_{yv}" for yv in y_vars]

    Ts = init_params.get("Ts", init_params.get("dt", 0.1))

    # Closed-loop sensitivity needs to span at least a few full PI response
    # cycles. We choose K so that K*Ts covers ~5 minutes of physical time --
    # long enough for the slowest dominant valve dynamics (TEP steam valve
    # at 120 s, boiler reheater dynamics at ~30 s, ANDES governors ~10 s)
    # to settle, but short enough to stay cheap in a 12-channel sweep.
    target_seconds = 300.0  # 5 min
    K = max(20, int(round(target_seconds / max(float(Ts), 1e-6))))

    return {
        "u_vars": u_vars,
        "y_vars": y_vars,
        "u_semantic_names": u_semantic_names,
        "y_semantic_names": y_semantic_names,
        "N": 1,
        "K": K,
        "Ts": float(Ts),
        "threshold_ratio": 1e-4,
    }


SENSITIVITY_CONFIG: Dict[str, Dict[str, Any]] = {
    "boiler": {
        "u_vars": ["fuel_command", "water_pump_speed"],
        "y_vars": ["waterwall_temp", "main_steam_pressure", "feedwater_flow"],
        "u_semantic_names": ["simulation_fuel_command", "simulation_water_pump_speed"],
        "y_semantic_names": [
            "pyhsicalprocess_waterwall_temp",
            "pyhsicalprocess_main_steam_pressure",
            "pyhsicalprocess_feedwater_flow",
        ],
        "N": 1,
        # Closed-loop horizon: Ts=0.1s × K=600 = 60 s, long enough to let
        # the boiler PI loops respond (rate-limit, integral wind-up).
        "K": 600,
        "Ts": 0.1,
        "threshold_ratio": 1e-4,
    },
}


def _get_epsilon(var_name: str, var_table: Dict[str, Dict[str, Any]]) -> float:
    """Compute perturbation magnitude: rate/10, falling back to 0.1% of range width."""
    entry = var_table.get(var_name, {})
    rate = entry.get("rate")
    if rate is not None and float(rate) > 0:
        return float(rate) / 10.0
    rng = entry.get("range")
    if rng is not None and len(rng) == 2:
        width = abs(float(rng[1]) - float(rng[0]))
        if width > 0:
            return width * 0.001
    return 1e-4


def _rollout_boiler(state_0: Dict[str, float], u: Dict[str, float], K: int, Ts: float) -> List[Dict[str, float]]:
    """Run a K-step rollout of the boiler physical model."""
    import sys as _sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from simulators.boiler_ccs.physical_process import physical_step_formula

    trajectory: List[Dict[str, float]] = []
    current = dict(state_0)
    for _ in range(K):
        current = physical_step_formula(
            state=current,
            fuel_command=float(u.get("fuel_command", 157.6)),
            water_pump_speed=float(u.get("water_pump_speed", 2500.0)),
            dt=Ts,
        )
        trajectory.append(dict(current))
    return trajectory


def _build_generic_rollout(root: Path, target: str, y_vars: List[str], u_vars: List[str]):
    """Build a generic rollout function by loading the physical process manifest.

    Returns:
        (rollout_fn, state_0, u_0), or (None, None, None) on failure.
    """
    return _build_closed_loop_rollout(root, target, y_vars, u_vars)


def _build_closed_loop_rollout(root: Path, target: str, y_vars: List[str], u_vars: List[str]):
    """Build a *closed-loop* rollout function for sensitivity computation.

    Unlike the original open-loop variant (which just held all controller
    outputs frozen at their steady-state values), this version drives the
    full ``ClosedLoopSim`` from the system manifest.  Each step the real
    controller stack runs against the perturbed plant, so the rollout
    captures how the PI / multi-loop control reacts to the injection.

    Perturbation semantics: for each u_var ``uv`` whose value in ``u``
    differs from the steady-state ``u_0[uv]``, we install an injection
    hook that overrides the corresponding manifest field at every step
    (constant override, the simplest "attack" the metric can see).

    Returns:
        (rollout_fn, state_0, u_0)  or  (None, None, None) on failure.
    """
    import importlib as _imp
    import sys as _sys

    try:
        spec = _get_target_spec(target)
    except KeyError:
        return None, None, None

    controllers_dir = root / spec.controllers_dir
    manifest_path = controllers_dir / "system_manifest.json"
    if not manifest_path.exists():
        return None, None, None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    sim_spec = manifest.get("simulation", {})
    sim_module_name = sim_spec.get("module", "")
    sim_class_name = sim_spec.get("class_name", "")
    pp_spec = manifest.get("physical_process", {})
    init_params = dict(pp_spec.get("init_params", {}))
    nominal_state = manifest.get("nominal_state", {})
    injection_points = manifest.get("injection_points", [])

    # Make the controllers package importable.
    parent = str(controllers_dir.resolve().parent)
    if parent not in _sys.path:
        _sys.path.insert(0, parent)

    try:
        sim_mod = _imp.import_module(sim_module_name)
        sim_cls = getattr(sim_mod, sim_class_name)
    except Exception:
        return None, None, None

    # Build (point -> (phase, field)) map from injection_points.
    phase_field_by_uvar: Dict[str, Tuple[str, str]] = {}
    for ip in injection_points:
        if not isinstance(ip, Mapping):
            continue
        field = str(ip.get("field", "")).strip()
        phase = str(ip.get("phase", "")).strip()
        if field and phase:
            phase_field_by_uvar[field] = (phase, field)

    # Initial physical-state and control-input snapshots.
    state_0 = {yv: float(nominal_state.get(yv, 0.0)) for yv in y_vars}
    inj_defaults = {ip["name"]: ip.get("default_value", 0.0)
                    for ip in injection_points if "name" in ip}
    inj_field_default = {str(ip.get("field", "")): ip.get("default_value", 0.0)
                         for ip in injection_points if ip.get("field")}

    u_0: Dict[str, float] = {}
    for uv in u_vars:
        if uv in nominal_state:
            u_0[uv] = float(nominal_state[uv])
            continue
        if uv in inj_field_default:
            try:
                u_0[uv] = float(inj_field_default[uv])
                continue
            except (TypeError, ValueError):
                pass
        matched_default: Optional[float] = None
        for inj_name, dv in inj_defaults.items():
            if uv == inj_name or uv in inj_name.lower():
                try:
                    matched_default = float(dv)
                except (TypeError, ValueError):
                    pass
                break
        u_0[uv] = matched_default if matched_default is not None else 0.0

    def _build_hook(perturb: Mapping[str, float]):
        """Constant-override hook: at every matching phase, overwrite ``field``
        with ``perturb[field]``. Returns None when there is nothing to override."""
        active = {f: float(v) for f, v in perturb.items()
                  if f in phase_field_by_uvar}
        if not active:
            return None
        # Group by phase to avoid scanning the whole dict each call.
        phase_payload: Dict[str, Dict[str, float]] = {}
        for field, value in active.items():
            phase, _ = phase_field_by_uvar[field]
            phase_payload.setdefault(phase, {})[field] = value

        def hook(t_step: int, current_phase: str, ctx):
            payload = phase_payload.get(str(current_phase))
            if not payload:
                return None
            return dict(payload)

        return hook

    def _closed_loop_rollout(
        s0: Dict[str, float], u: Dict[str, float], K: int, Ts: float,
    ) -> List[Dict[str, float]]:
        # Honour the manifest Ts when the caller passes one; otherwise
        # default to whatever init_params declares.
        kwargs = dict(init_params)
        kwargs["Ts"] = float(Ts) if Ts > 0 else float(init_params.get("Ts", 1.0))
        sim = sim_cls(**{k: v for k, v in kwargs.items()
                         if k in {"Ts", "case_path", "mode", "random_seed"}})
        # Build a perturbation map: only fields where the caller's u differs
        # from the steady-state u_0 are overridden each step.
        perturb: Dict[str, float] = {}
        for uv, v_nom in u_0.items():
            v_now = float(u.get(uv, v_nom))
            if abs(v_now - float(v_nom)) > 1e-12:
                perturb[uv] = float(v_now)
        hook = _build_hook(perturb)
        try:
            trace, _meta = sim.run(
                steps=int(K),
                injection_hook=hook,
                return_trace=True,
                stop_on_trip=False,
            )
        finally:
            try:
                sim.close()
            except Exception:
                pass
        # Cast to plain {y_var: float} dicts for the caller.  The trace already
        # carries the manifest's state_variables as keys.
        out: List[Dict[str, float]] = []
        for row in trace:
            out.append({yv: float(row.get(yv, 0.0)) for yv in y_vars})
        return out

    return _closed_loop_rollout, state_0, u_0


def compute_sensitivity_edges(
    target: str,
    root: Path,
    var_table: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute a multi-step sensitivity Jacobian and return semantic-model edges."""
    cfg = SENSITIVITY_CONFIG.get(target)
    if cfg is None:
        # Fallback: build configuration automatically from system_manifest.json.
        cfg = _build_sensitivity_config_from_manifest(root, target)
    if cfg is None:
        return []

    u_vars: List[str] = cfg["u_vars"]
    y_vars: List[str] = cfg["y_vars"]
    u_sem: List[str] = cfg["u_semantic_names"]
    y_sem: List[str] = cfg["y_semantic_names"]
    N: int = cfg["N"]
    K: int = cfg["K"]
    Ts: float = cfg["Ts"]
    threshold_ratio: float = cfg["threshold_ratio"]
    hazard_ids_by_measurement = _load_hazard_ids_by_measurement(root, target)

    # Determine nominal operating point using closed-loop rollout for every target.
    _rollout_fn, state_0_cl, u_0_cl = _build_closed_loop_rollout(
        root, target, y_vars, u_vars
    )
    if _rollout_fn is None:
        return []
    state_0 = state_0_cl
    u_0 = u_0_cl

    # Compute epsilon for each u_i.
    epsilons: Dict[str, float] = {}
    for uv in u_vars:
        table_name = uv
        if table_name not in var_table:
            for candidate in var_table:
                if candidate.endswith(uv) or uv in candidate:
                    table_name = candidate
                    break
        epsilons[uv] = _get_epsilon(table_name, var_table)

    # Jacobian computation with bidirectional +/-epsilon perturbations.
    #
    # For each (u_i, y_j), record epsilon-normalized directional values.
    #
    #   d_plus  = max_k  (y_pos_k  - y_base_k) / eps    when u_i += eps
    #   d_minus = max_k  (y_neg_k  - y_base_k) / eps    when u_i -= eps
    #
    # Direction semantics for Stage 2 direction-aware scoring:
    #   up_strength   = max(d_plus,  d_minus)            maximum upward effect on y across both perturbation directions
    #   down_strength = max(-d_plus_min, -d_minus_min)   maximum downward effect on y across both perturbation directions
    #
    # Where
    #   d_plus_min  = min_k (y_pos_k - y_base_k) / eps   (record the most negative value; its magnitude is the downward change)
    #   d_minus_min = min_k (y_neg_k - y_base_k) / eps   (same as above)
    n_u = len(u_vars)
    n_y = len(y_vars)
    w_matrix = [[0.0] * n_y for _ in range(n_u)]
    up_matrix = [[0.0] * n_y for _ in range(n_u)]
    down_matrix = [[0.0] * n_y for _ in range(n_u)]

    for _sample in range(N):
        baseline = _rollout_fn(state_0, u_0, K, Ts)

        for i, uv in enumerate(u_vars):
            eps_i = epsilons[uv]
            if eps_i <= 0:
                continue

            u_pert_pos = dict(u_0)
            u_pert_pos[uv] = float(u_0[uv]) + eps_i
            perturbed_pos = _rollout_fn(state_0, u_pert_pos, K, Ts)

            u_pert_neg = dict(u_0)
            u_pert_neg[uv] = float(u_0[uv]) - eps_i
            perturbed_neg = _rollout_fn(state_0, u_pert_neg, K, Ts)

            for j, yv in enumerate(y_vars):
                # Scan the full trajectory for four extrema.
                d_plus_max = 0.0    # Largest upward change from +epsilon.
                d_plus_min = 0.0    # Largest downward change from +epsilon, stored as the minimum negative value.
                d_minus_max = 0.0   # Largest upward change from -epsilon.
                d_minus_min = 0.0   # Largest downward change from -epsilon.
                for k in range(K):
                    y_base_k = float(baseline[k].get(yv, 0.0))
                    y_pos_k = float(perturbed_pos[k].get(yv, 0.0))
                    y_neg_k = float(perturbed_neg[k].get(yv, 0.0))
                    dp = (y_pos_k - y_base_k) / eps_i
                    dn = (y_neg_k - y_base_k) / eps_i
                    if dp > d_plus_max:
                        d_plus_max = dp
                    if dp < d_plus_min:
                        d_plus_min = dp
                    if dn > d_minus_max:
                        d_minus_max = dn
                    if dn < d_minus_min:
                        d_minus_min = dn
                up_strength = max(d_plus_max, d_minus_max, 0.0)
                down_strength = max(-d_plus_min, -d_minus_min, 0.0)
                # Overall absolute sensitivity for norm_matrix normalization.
                abs_strength = max(up_strength, down_strength)

                w_matrix[i][j] += abs_strength / N
                up_matrix[i][j] += up_strength / N
                down_matrix[i][j] += down_strength / N

    # Normalize up/down effects to a shared scale using w_max.
    w_max = 0.0
    for row in w_matrix:
        for val in row:
            if val > w_max:
                w_max = val
    if w_max <= 0.0:
        return []

    threshold = threshold_ratio * w_max

    edges: List[Dict[str, Any]] = []
    for i in range(n_u):
        for j in range(n_y):
            w_ij = w_matrix[i][j]
            if w_ij < threshold:
                continue
            eta = w_ij / w_max
            eta_up = min(up_matrix[i][j] / w_max, 1.0)
            eta_down = min(down_matrix[i][j] / w_max, 1.0)
            edge = {
                "x": u_sem[i],
                "y": y_sem[j],
                "lambda": "data",
                "eta": float(max(min(eta, 1.0), 1e-6)),
                "epsilon": float(max(min(eta, 1.0), 1e-6)),
                # eta_up: strongest upward push on y across perturbation directions.
                # eta_down: strongest downward pull on y across perturbation directions.
                "eta_up": float(eta_up),
                "eta_down": float(eta_down),
                "source": {"file": "sensitivity_jacobian", "line": 0},
            }
            hazard_ids = hazard_ids_by_measurement.get(y_sem[j], [])
            if hazard_ids:
                edge["hazard_keys"] = list(hazard_ids)
            edges.append(edge)

    return edges


def build_semantic_model(root: Path, target: str) -> Dict[str, Any]:
    """Build the complete semantic model M=(D,G,P,H)."""
    spec = _get_target_spec(target)
    policy = spec.policy
    blacklist_res = _compile_blacklist_regexes(policy.blacklist_patterns)
    warnings: List[str] = []
    loop_file = (root / spec.loop_file).resolve()
    controllers_dir = (root / spec.controllers_dir).resolve()
    variable_table_raw = spec.variable_table.strip()
    variable_table_path = (
        (root / variable_table_raw).resolve() if variable_table_raw else Path("")
    )

    # Collect files.
    files = _collect_files(loop_file=loop_file, controllers_dir=controllers_dir)
    for extra_dir in spec.extra_scan_dirs:
        extra_path = (root / extra_dir).resolve()
        if extra_path.exists():
            existing = {f.resolve() for f in files}
            for p in sorted(extra_path.glob("*.py")):
                if p.name != "__init__.py" and p.resolve() not in existing:
                    files.append(p.resolve())

    # AST scan.
    all_vars, all_edges = _scan_files_for_vars_and_edges(root, files)

    # Blacklist filtering.
    filtered_vars = {
        k: v for k, v in all_vars.items()
        if not _is_blacklisted(k, blacklist_res)
    }

    # Variable-table override
    var_table = parse_variable_table(variable_table_path)
    matched_table_vars = _apply_variable_table(filtered_vars, var_table)

    # Manifest injection-point overlay: register system_manifest.json injection_points as
    # attackable variables with default/range/rate. This is the metadata source for
    # backends without workbooks and also a boiler fallback.
    matched_manifest_vars = _apply_manifest_injection_points(
        filtered_vars, controllers_dir, policy,
    )

    # Extract P/H.
    p_rules = extract_alarm_rules(files)
    h_rules = extract_hazard_rules(files)

    # Manifest fallback: when controller source lacks declarative ALARM_RULES / HAZARD_RULES,
    # as in the TE third-party PI controller stack, read system_manifest.json
    # alarm_rules / hazard_rules blocks instead.
    if not p_rules or not h_rules:
        manifest_p, manifest_h = _load_rules_from_manifest(controllers_dir)
        if not p_rules and manifest_p:
            p_rules = manifest_p
        if not h_rules and manifest_h:
            h_rules = manifest_h

    # Edge processing: filter edges referencing removed variables, deduplicate, and weight.
    valid_vars = set(filtered_vars.keys())
    all_edges = [e for e in all_edges if e.get("x") in valid_vars and e.get("y") in valid_vars]

    # Filter physical-model internal edges and physical-model-to-controller feedback edges.
    all_edges = _filter_target_physical_edges(all_edges, policy)

    # Multi-step sensitivity Jacobian edges from controllers to the physical model.
    sensitivity_edges: List[Dict[str, Any]] = []
    try:
        sensitivity_edges = compute_sensitivity_edges(target, root, var_table)
        sensitivity_edges = _normalize_target_sensitivity_edges(
            sensitivity_edges,
            policy,
        )
        all_edges.extend(sensitivity_edges)
    except Exception as e:
        warnings.append(
            f"sensitivity Jacobian computation failed for target={target}: {type(e).__name__}: {e}"
        )

    all_edges = _append_target_required_edges(all_edges, policy)

    all_edges = _dedup_edges(all_edges)
    all_edges = _apply_semantic_edge_weights(all_edges)

    # Assemble.
    var_names = sorted(filtered_vars.keys())
    var_map = {name: filtered_vars[name].to_json() for name in var_names}

    return {
        "M": {
            "D": {"V": var_names, "a": var_map},
            "G": {"V": var_names, "E": all_edges},
            "P": p_rules,
            "H": h_rules,
        },
        "meta": {
            "target": target,
            "source_root": ".",
            "loop_file": spec.loop_file,
            "controllers_dir": spec.controllers_dir,
            "variable_table_path": spec.variable_table,
            "variable_table_entries": len(var_table),
            "variable_table_matched": len(matched_table_vars),
            "manifest_injection_matched": len(matched_manifest_vars),
            "warnings": warnings,
            "files_parsed": [
                _artifact_relative_label(p, root) for p in files
            ],
            "counts": {
                "variables": len(var_names),
                "edges": len(all_edges),
                "sensitivity_edges": len(sensitivity_edges),
                "protection_predicates": len(p_rules),
                "hazard_predicates": len(h_rules),
            },
        },
    }


# ============================================================================
# CLI entrypoint
# ============================================================================

def parse_args() -> argparse.Namespace:
    root = _default_root()
    parser = argparse.ArgumentParser(
        description="Extract semantic model M=(D,G,P,H) from Python closed-loop control code."
    )
    parser.add_argument("--target", type=str, choices=sorted(TARGET_SPECS.keys()),
                        required=True, help="Extraction target: boiler / TE")
    parser.add_argument("--root", type=str, default=str(root), help="Workspace root directory")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    parser.add_argument("--m2-output", type=str, default="", help="Optional M2-format output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    target = args.target

    output_path = Path(args.output) if args.output else _default_output_path(root, target)

    model = build_semantic_model(root=root, target=target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )

    counts = model["meta"]["counts"]
    print(f"[{target}] Semantic model written: {output_path}")
    print(f"  variables={counts['variables']}, edges={counts['edges']}, "
          f"P={counts['protection_predicates']}, H={counts['hazard_predicates']}")
    print(f"  variable_table_matched={model['meta']['variable_table_matched']}"
          f"/{model['meta']['variable_table_entries']}")


if __name__ == "__main__":
    main()

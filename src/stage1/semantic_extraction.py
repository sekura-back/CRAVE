from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from stage1.output_layout import stage1_artifact_dir


def _collect_source_files(paths: Path | Sequence[Path]) -> list[Path]:
    raw_paths = [paths] if isinstance(paths, Path) else list(paths)
    files: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        key = str(path.resolve())
        if key not in seen:
            files.append(path)
            seen.add(key)
    return files


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_sources(source_files: Sequence[Path]) -> list[tuple[Path, ast.AST]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path)))
        for path in source_files
    ]


def _validate_source_filename(path: Path, expected_name: str) -> None:
    if Path(path).name != expected_name:
        raise ValueError(f"expected {expected_name} source, got {path}")


def _target_name_matches(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _literal_rule_list(node: ast.AST | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    value = ast.literal_eval(node)
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _load_rules(
    source_trees: Sequence[tuple[Path, ast.AST]],
    rule_name: str,
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for _path, tree in source_trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(_target_name_matches(target, rule_name) for target in node.targets):
                    rules.extend(_literal_rule_list(node.value))
            elif isinstance(node, ast.AnnAssign) and _target_name_matches(node.target, rule_name):
                rules.extend(_literal_rule_list(node.value))
    return rules


def _rule_ids(rules: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(rule.get("id", "")).strip() for rule in rules if str(rule.get("id", "")).strip()}


def _ordered_rule_ids(rules: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(rule.get("id", "")).strip() for rule in rules]


def _duplicate_values(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _validate_rule_id_oracle(
    source_rules: Sequence[Mapping[str, Any]],
    manifest_rules: Sequence[Mapping[str, Any]],
    prefix: str,
) -> None:
    source_id_list = _ordered_rule_ids(source_rules)
    manifest_id_list = _ordered_rule_ids(manifest_rules)
    if any(not rule_id for rule_id in source_id_list + manifest_id_list):
        raise ValueError(f"rule id oracle mismatch for {prefix}: empty id")
    duplicates = _duplicate_values(source_id_list) | _duplicate_values(manifest_id_list)
    if duplicates:
        raise ValueError(f"duplicate rule id for {prefix}: {sorted(duplicates)}")
    source_ids = set(source_id_list)
    manifest_ids = set(manifest_id_list)
    if source_ids != manifest_ids:
        raise ValueError(f"rule id oracle mismatch for {prefix}")
    if any(not rule_id.startswith(f"{prefix}-") for rule_id in source_ids):
        raise ValueError(f"rule id oracle mismatch for {prefix}: invalid prefix")
    if prefix == "A" and any("alarm_id" in rule for rule in [*source_rules, *manifest_rules]):
        raise ValueError("rule id oracle mismatch for A: duplicate alarm_id alias")


def _merge_manifest_rule_metadata(
    source_rules: Sequence[Mapping[str, Any]],
    manifest_rules: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {
        str(rule.get("id", "")).strip(): dict(rule)
        for rule in manifest_rules
        if str(rule.get("id", "")).strip()
    }
    merged: list[dict[str, Any]] = []
    for rule in source_rules:
        item = dict(rule)
        manifest_rule = manifest_by_id.get(str(item.get("id", "")).strip(), {})
        for key, value in manifest_rule.items():
            item.setdefault(key, value)
        merged.append(item)
    return merged


def _source_terms(source_trees: Sequence[tuple[Path, ast.AST]]) -> set[str]:
    terms: set[str] = set()
    for _path, tree in source_trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                terms.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                terms.add(node.value)
    return terms


def _registry_record_names(record: Mapping[str, Any]) -> set[str]:
    names = {
        str(record.get("canonical_name", "")).strip(),
        str(record.get("runtime_field", "")).strip(),
        str(record.get("manifest_name", "")).strip(),
    }
    names.update(str(alias).strip() for alias in record.get("aliases", []) if str(alias).strip())
    return {name for name in names if name}


def _registry_name_map(
    variable_registry: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for record in variable_registry:
        canonical_name = str(record.get("canonical_name", "")).strip()
        if not canonical_name:
            continue
        for name in _registry_record_names(record):
            owner = name_map.get(name)
            if owner is not None and owner != canonical_name:
                raise ValueError(f"ambiguous registry variable name: {name}")
            name_map[name] = canonical_name
    return name_map


def _resolve_canonical_name(
    name_map: Mapping[str, str],
    name: str,
    context: str,
) -> str:
    value = str(name).strip()
    if not value:
        return ""
    try:
        return name_map[value]
    except KeyError as exc:
        raise ValueError(f"missing registry record for {context}: {value}") from exc


def _canonicalize_rule_variables(
    rules: Sequence[Mapping[str, Any]],
    variable_registry: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, str],
) -> list[dict[str, Any]]:
    name_map = _registry_name_map(variable_registry)
    canonical_rules: list[dict[str, Any]] = []
    for rule in rules:
        item = dict(rule)
        for key, context in contexts.items():
            if key in item:
                item[key] = _resolve_canonical_name(name_map, str(item[key]), context)
        canonical_rules.append(item)
    return canonical_rules


def _canonical_registry_names(variable_registry: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(record.get("canonical_name", "")).strip()
        for record in variable_registry
        if str(record.get("canonical_name", "")).strip()
    }


def _validate_alarm_setpoints_in_registry(
    predicates: Sequence[Mapping[str, Any]],
    variable_registry: Sequence[Mapping[str, Any]],
) -> None:
    canonical_names = _canonical_registry_names(variable_registry)
    for rule in predicates:
        setpoint = str(rule.get("setpoint_var", "")).strip()
        if setpoint and setpoint not in canonical_names:
            raise ValueError(f"missing registry record for alarm setpoint: {setpoint}")


def _build_d_payload(
    variable_registry: Sequence[Mapping[str, Any]],
    source_terms: set[str],
) -> dict[str, Any]:
    variables = [
        str(record["canonical_name"])
        for record in variable_registry
        if _registry_record_names(record) & source_terms
    ]
    return {"variables": sorted(set(variables)), "attributes": []}


def _node_terms(node: ast.AST) -> set[str]:
    terms: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            terms.add(child.id)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            terms.add(child.value)
    return terms


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in node.elts:
            names.extend(_target_names(item))
        return names
    return []


def _dict_key_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _iter_target_nodes(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, (ast.Tuple, ast.List)):
        nodes: list[ast.AST] = []
        for item in node.elts:
            nodes.extend(_iter_target_nodes(item))
        return nodes
    return [node]


def _expr_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def _unwrap_expr(node: ast.AST) -> ast.AST:
    current = node
    while (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Name)
        and current.func.id in {"float", "int", "bool", "str"}
        and len(current.args) == 1
        and not current.keywords
    ):
        current = current.args[0]
    return current


def _literal_int(node: ast.AST) -> int | None:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return None
    return int(value) if isinstance(value, int) else None


def _expr_candidates(node: ast.AST) -> list[str]:
    current = _unwrap_expr(node)
    if isinstance(current, ast.Name):
        return [current.id]
    if isinstance(current, ast.Attribute):
        value = _expr_text(current.value)
        if not value:
            return []
        return [f"{value}.{current.attr}"]
    if isinstance(current, ast.Subscript):
        base_names = _expr_candidates(current.value)
        candidates: list[str] = []
        if isinstance(current.slice, ast.Constant) and isinstance(current.slice.value, str):
            return [str(current.slice.value)]
        index = _literal_int(current.slice)
        if index is None:
            return [_expr_text(current)]
        for base in base_names:
            if not base:
                continue
            if base.startswith("xmeas") or base.endswith("xmeas"):
                candidates.extend([f"xmeas_{index + 1:02d}", f"xmeas[{index}]"])
                continue
            if base.startswith("xmv") or base.endswith("xmv") or base.endswith("xmv_new"):
                candidates.extend([f"xmv_{index + 1:02d}", f"xmv[{index}]"])
                continue
            if base.endswith("setpoints"):
                candidates.extend([f"setpoints_{index}", f"setpoints[{index}]"])
                continue
            candidates.append(f"{base}[{index}]")
        return candidates
    return []


def _resolve_expr_canonical_names(
    name_map: Mapping[str, str],
    node: ast.AST,
) -> set[str]:
    resolved: set[str] = set()
    for candidate in _expr_candidates(node):
        canonical = name_map.get(str(candidate).strip())
        if canonical:
            resolved.add(canonical)
    return resolved


def _call_method_name(node: ast.AST) -> str:
    current = _unwrap_expr(node)
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        return str(current.func.attr)
    return ""


def _call_controller_name(node: ast.AST) -> str:
    current = _unwrap_expr(node)
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        return _expr_text(current.func.value)
    return ""


def _passthrough_wrapper_value_node(node: ast.AST) -> ast.AST | None:
    current = _unwrap_expr(node)
    if not isinstance(current, ast.Call) or not isinstance(current.func, ast.Name):
        return None
    if current.func.id == "emit_output":
        if len(current.args) >= 3:
            return current.args[2]
    if current.func.id == "_override_output":
        for keyword in current.keywords:
            if keyword.arg == "value" and keyword.value is not None:
                return keyword.value
        if len(current.args) >= 4:
            return current.args[3]
    return None


def _local_import_paths(path: Path) -> list[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    candidates: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = str(node.module or "").strip()
        if node.level <= 0:
            continue
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        parts = [part for part in module.split(".") if part]
        target = base.joinpath(*parts) if parts else base
        for candidate in (target.with_suffix(".py"), target / "__init__.py"):
            if candidate.exists():
                candidates.append(candidate.resolve())
                break
    return candidates


def _discover_dependency_files(source_files: Sequence[Path]) -> list[Path]:
    pending = _collect_source_files(source_files)
    files: list[Path] = []
    seen: set[str] = set()
    while pending:
        path = Path(pending.pop(0)).resolve()
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        files.append(path)
        for candidate in _local_import_paths(path):
            candidate_key = str(candidate.resolve())
            if candidate_key not in seen:
                pending.append(candidate)
    return files


_INTERNAL_NOISE_NAMES = {
    "self",
    "dt",
    "dt3",
    "dt360",
    "dt900",
    "time_step",
    "t_step",
    "step_count",
    "i",
    "j",
    "k",
    "idx",
    "name",
    "state",
    "snapshot",
    "controllers",
    "ctrl_names",
    "nom",
    "mode_config",
    "sp",
    "np",
    "xmeas",
    "xmv",
    "new_xmv",
    "xmv_new",
}

_INTERNAL_NOISE_ATTRS = {
    "last_debug",
    "last_alarm",
    "last_output",
    "debug",
    "alarm_id",
    "description",
    "message",
    "triggered",
    "rate_limit",
    "Ts",
    "TC",
    "gain",
    "taui",
    "scale",
    "output_min",
    "output_max",
    "deadband",
    "CP",
    "TI",
    "TD",
    "OC",
    "OT",
    "OB",
    "OV",
}


def _attribute_parts(node: ast.AST) -> list[str]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return []


def _iter_function_defs(
    source_trees: Sequence[tuple[Path, ast.AST]],
) -> list[tuple[Path, str | None, ast.FunctionDef | ast.AsyncFunctionDef]]:
    items: list[tuple[Path, str | None, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for path, tree in source_trees:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                items.append((path, None, node))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        items.append((path, node.name, child))
    return items


def _class_method_index(
    source_trees: Sequence[tuple[Path, ast.AST]],
) -> dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    index: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for _path, class_name, node in _iter_function_defs(source_trees):
        if class_name is None:
            continue
        index.setdefault(class_name, {})[node.name] = node
    return index


def _constructor_name(node: ast.AST) -> str:
    current = _unwrap_expr(node)
    if not isinstance(current, ast.Call):
        return ""
    if isinstance(current.func, ast.Name):
        return current.func.id
    if isinstance(current.func, ast.Attribute):
        return current.func.attr
    return ""


def _self_attribute_name(node: ast.AST) -> str:
    parts = _attribute_parts(node)
    if len(parts) == 2 and parts[0] == "self":
        return parts[1]
    return ""


def _class_attr_type_map(
    source_trees: Sequence[tuple[Path, ast.AST]],
) -> dict[str, dict[str, str]]:
    class_methods = _class_method_index(source_trees)
    attr_types: dict[str, dict[str, str]] = {}
    for _path, class_name, node in _iter_function_defs(source_trees):
        if class_name is None:
            continue
        mapping = attr_types.setdefault(class_name, {})
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                ctor = _constructor_name(child.value)
                if ctor not in class_methods:
                    continue
                for target in child.targets:
                    for item in _iter_target_nodes(target):
                        attr_name = _self_attribute_name(item)
                        if attr_name:
                            mapping[attr_name] = ctor
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                ctor = _constructor_name(child.value)
                if ctor not in class_methods:
                    continue
                attr_name = _self_attribute_name(child.target)
                if attr_name:
                    mapping[attr_name] = ctor
    return attr_types


def _resolve_method_call_ref(
    node: ast.AST,
    current_class: str | None,
    class_attr_types: Mapping[str, Mapping[str, str]],
) -> tuple[str, str] | None:
    current = _unwrap_expr(node)
    if not isinstance(current, ast.Call) or not isinstance(current.func, ast.Attribute):
        return None
    parts = _attribute_parts(current.func)
    if len(parts) < 2:
        return None
    method_name = parts[-1]
    receiver = parts[:-1]
    if receiver[:2] == ["self", "controller"]:
        class_name = "PLCController"
        for attr in receiver[2:]:
            class_name = class_attr_types.get(class_name, {}).get(attr, "")
            if not class_name:
                return None
        return class_name, method_name
    if receiver[0] != "self" or current_class is None:
        return None
    class_name = current_class
    for attr in receiver[1:]:
        class_name = class_attr_types.get(class_name, {}).get(attr, "")
        if not class_name:
            return None
    return class_name, method_name


def _resolve_method_call_context(
    node: ast.AST,
    current_class: str | None,
    current_owner: str | None,
    class_attr_types: Mapping[str, Mapping[str, str]],
) -> tuple[str, str, str] | None:
    current = _unwrap_expr(node)
    if not isinstance(current, ast.Call) or not isinstance(current.func, ast.Attribute):
        return None
    parts = _attribute_parts(current.func)
    if len(parts) < 2:
        return None
    method_name = parts[-1]
    receiver = parts[:-1]
    if receiver[:2] == ["self", "controller"]:
        class_name = "PLCController"
        owner = ""
        for attr in receiver[2:]:
            next_class = class_attr_types.get(class_name, {}).get(attr, "")
            if not next_class:
                return None
            owner = next_class if not owner else f"{owner}.{attr}"
            class_name = next_class
        return class_name, method_name, owner or class_name
    if receiver[0] != "self" or current_class is None:
        return None
    class_name = current_class
    owner = current_owner or current_class
    for attr in receiver[1:]:
        next_class = class_attr_types.get(class_name, {}).get(attr, "")
        if not next_class:
            return None
        if attr.startswith("_") and owner == current_class:
            owner = next_class
        else:
            owner = f"{owner}.{attr}"
        class_name = next_class
    return class_name, method_name, owner


def _function_parameter_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    parameters = [*node.args.posonlyargs, *node.args.args]
    names = [arg.arg for arg in parameters]
    return [name for name in names if name != "self"]


def _function_used_parameter_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    parameter_names = set(_function_parameter_names(node))
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id in parameter_names
    }


def _reachable_internal_method_refs(
    source_trees: Sequence[tuple[Path, ast.AST]],
) -> tuple[
    set[tuple[str, str]],
    dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    dict[str, dict[str, str]],
]:
    class_methods = _class_method_index(source_trees)
    class_attr_types = _class_attr_type_map(source_trees)
    roots: set[tuple[str, str]] = set()
    for class_name, methods in class_methods.items():
        if "controller_step" in methods:
            roots.add((class_name, "controller_step"))
    if "PLCController" in class_methods and "calculate" in class_methods["PLCController"]:
        roots.add(("PLCController", "calculate"))
    for path, class_name, node in _iter_function_defs(source_trees):
        if path.name != "simulation.py":
            continue
        for child in ast.walk(node):
            current = _unwrap_expr(child)
            if not isinstance(current, ast.Call) or not isinstance(current.func, ast.Attribute):
                continue
            parts = _attribute_parts(current.func)
            if len(parts) < 3 or parts[:2] != ["self", "controller"]:
                continue
            ref = _resolve_method_call_ref(child, class_name, class_attr_types)
            if ref is None:
                continue
            callee_class, method_name = ref
            if method_name not in {"calculate", "update", "controller_step"}:
                continue
            if method_name in class_methods.get(callee_class, {}):
                roots.add(ref)
    seen: set[tuple[str, str]] = set()
    pending = list(roots)
    while pending:
        class_name, method_name = pending.pop()
        if (class_name, method_name) in seen:
            continue
        method = class_methods.get(class_name, {}).get(method_name)
        if method is None:
            continue
        seen.add((class_name, method_name))
        for child in ast.walk(method):
            ref = _resolve_method_call_ref(child, class_name, class_attr_types)
            if ref is None or ref in seen:
                continue
            callee_class, callee_method = ref
            if callee_method in class_methods.get(callee_class, {}):
                pending.append(ref)
    return seen, class_methods, class_attr_types


def _qualify_internal_name(owner: str, raw_name: str) -> str:
    name = str(raw_name).strip()
    if not name:
        return ""
    if name.startswith("self."):
        name = name[5:]
    base = name.split(".")[-1]
    if base in _INTERNAL_NOISE_NAMES or base in _INTERNAL_NOISE_ATTRS:
        return ""
    if base.startswith("last_") or (base.startswith("_") and base.lstrip("_").isupper()):
        return ""
    return f"{owner}.{name}"


def _collect_internal_expr_names(
    node: ast.AST,
    owner: str,
    name_map: Mapping[str, str],
    local_bindings: Mapping[str, set[str]],
) -> set[str]:
    current = _unwrap_expr(node)
    resolved = _resolve_expr_canonical_names(name_map, current)
    if resolved:
        return resolved
    if isinstance(current, ast.Name):
        if current.id in local_bindings:
            return set(local_bindings[current.id])
        name = _qualify_internal_name(owner, current.id)
        return {name} if name else set()
    if isinstance(current, ast.Attribute):
        parts = _attribute_parts(current)
        if parts and parts[0] == "self":
            name = _qualify_internal_name(owner, ".".join(parts[1:]))
            return {name} if name else set()
        return set()
    if isinstance(current, ast.Subscript):
        text = _expr_text(current)
        if text.startswith("self."):
            name = _qualify_internal_name(owner, text[5:])
            return {name} if name else set()
        return set()
    if isinstance(current, ast.Call):
        names: set[str] = set()
        for arg in current.args:
            names.update(_collect_internal_expr_names(arg, owner, name_map, local_bindings))
        for keyword in current.keywords:
            if keyword.value is not None:
                names.update(
                    _collect_internal_expr_names(
                        keyword.value,
                        owner,
                        name_map,
                        local_bindings,
                    )
                )
        return names
    names: set[str] = set()
    for child in ast.iter_child_nodes(current):
        names.update(_collect_internal_expr_names(child, owner, name_map, local_bindings))
    return names


def _collect_internal_target_names(
    node: ast.AST,
    owner: str,
    name_map: Mapping[str, str],
    *,
    emit_local_targets: bool,
    local_owner: str | None = None,
) -> set[str]:
    names: set[str] = set()
    for item in _iter_target_nodes(node):
        resolved = _resolve_expr_canonical_names(name_map, item)
        if resolved:
            names.update(resolved)
            continue
        if isinstance(item, ast.Name):
            if not emit_local_targets:
                continue
            name = _qualify_internal_name(local_owner or owner, item.id)
            if name:
                names.add(name)
        elif isinstance(item, ast.Attribute):
            parts = _attribute_parts(item)
            if parts and parts[0] == "self":
                name = _qualify_internal_name(owner, ".".join(parts[1:]))
                if name:
                    names.add(name)
        elif isinstance(item, ast.Subscript):
            text = _expr_text(item)
            if text.startswith("self."):
                name = _qualify_internal_name(owner, text[5:])
                if name:
                    names.add(name)
    return names


def _repeated_local_target_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    counts: dict[str, int] = {}
    for child in ast.walk(node):
        targets: list[ast.AST] = []
        if isinstance(child, ast.Assign):
            if _passthrough_wrapper_value_node(child.value) is not None:
                continue
            for target in child.targets:
                targets.extend(_iter_target_nodes(target))
        elif isinstance(child, ast.AnnAssign):
            if child.value is not None and _passthrough_wrapper_value_node(child.value) is not None:
                continue
            targets.extend(_iter_target_nodes(child.target))
        elif isinstance(child, ast.AugAssign):
            targets.extend(_iter_target_nodes(child.target))
        for target in targets:
            if isinstance(target, ast.Name):
                counts[target.id] = counts.get(target.id, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def _bind_local_target_names(
    node: ast.AST,
    binding_names: set[str],
    local_bindings: dict[str, set[str]],
) -> None:
    for item in _iter_target_nodes(node):
        if isinstance(item, ast.Name):
            local_bindings[item.id] = set(binding_names)


def _passthrough_local_target_names(
    node: ast.AST,
    local_bindings: Mapping[str, set[str]],
) -> set[str]:
    names: set[str] = set()
    for item in _iter_target_nodes(node):
        if isinstance(item, ast.Name):
            names.update(local_bindings.get(item.id, set()))
    return names


def _controller_internal_edges(
    source_trees: Sequence[tuple[Path, ast.AST]],
    variable_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    name_map = _registry_name_map(variable_registry)
    class_methods = _class_method_index(source_trees)
    class_attr_types = _class_attr_type_map(source_trees)
    edge_keys: set[tuple[str, str, str]] = set()
    visited: set[tuple[str, str, str]] = set()

    def call_bindings(
        call: ast.Call,
        callee_method: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: str,
        current_class: str,
        local_bindings: Mapping[str, set[str]],
    ) -> dict[str, set[str]]:
        bindings = {
            name: set()
            for name in _function_parameter_names(callee_method)
        }
        parameter_names = _function_parameter_names(callee_method)
        for index, parameter_name in enumerate(parameter_names):
            if index < len(call.args):
                bindings[parameter_name] = collect_expr_names(
                    call.args[index],
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                )
        for keyword in call.keywords:
            if keyword.arg is None or keyword.value is None or keyword.arg not in bindings:
                continue
            bindings[keyword.arg] = collect_expr_names(
                keyword.value,
                owner,
                current_class,
                name_map,
                local_bindings,
            )
        return bindings

    def collect_return_names(
        statements: Sequence[ast.stmt],
        owner: str,
        current_class: str,
        local_bindings: dict[str, set[str]],
        guard_deps: set[str],
        return_stack: set[tuple[str, str, str]],
    ) -> set[str]:
        names: set[str] = set()

        def merge_child_bindings(*children: Mapping[str, set[str]]) -> None:
            for name in {key for child in children for key in child}:
                merged: set[str] = set()
                for child in children:
                    merged.update(child.get(name, set()))
                if merged:
                    local_bindings[name] = merged

        emit_local_targets = "." not in owner
        for statement in statements:
            if isinstance(statement, ast.Assign):
                passthrough_value = _passthrough_wrapper_value_node(statement.value)
                deps = guard_deps | collect_expr_names(
                    statement.value,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                    return_stack,
                )
                for target in statement.targets:
                    target_names = (
                        _passthrough_local_target_names(target, local_bindings)
                        if passthrough_value is not None
                        else set()
                    )
                    if not target_names:
                        target_names = _collect_internal_target_names(
                            target,
                            owner,
                            name_map,
                            emit_local_targets=emit_local_targets,
                        )
                    binding_names = set(target_names) | set(deps) if target_names else set(deps)
                    _bind_local_target_names(target, binding_names, local_bindings)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                passthrough_value = _passthrough_wrapper_value_node(statement.value)
                deps = guard_deps | collect_expr_names(
                    statement.value,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                    return_stack,
                )
                target_names = (
                    _passthrough_local_target_names(statement.target, local_bindings)
                    if passthrough_value is not None
                    else set()
                )
                if not target_names:
                    target_names = _collect_internal_target_names(
                        statement.target,
                        owner,
                        name_map,
                        emit_local_targets=emit_local_targets,
                    )
                binding_names = set(target_names) | set(deps) if target_names else set(deps)
                _bind_local_target_names(statement.target, binding_names, local_bindings)
            elif isinstance(statement, ast.AugAssign):
                deps = guard_deps | collect_expr_names(
                    statement.value,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                    return_stack,
                )
                deps.update(
                    collect_expr_names(
                        statement.target,
                        owner,
                        current_class,
                        name_map,
                        local_bindings,
                        return_stack,
                    )
                )
                target_names = _collect_internal_target_names(
                    statement.target,
                    owner,
                    name_map,
                    emit_local_targets=emit_local_targets,
                )
                binding_names = set(target_names) | set(deps) if target_names else set(deps)
                _bind_local_target_names(statement.target, binding_names, local_bindings)
            elif isinstance(statement, ast.Return) and statement.value is not None:
                names.update(
                    guard_deps
                    | collect_expr_names(
                        statement.value,
                        owner,
                        current_class,
                        name_map,
                        local_bindings,
                        return_stack,
                    )
                )

            if isinstance(statement, ast.If):
                condition_deps = collect_expr_names(
                    statement.test,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                    return_stack,
                )
                body_bindings = dict(local_bindings)
                else_bindings = dict(local_bindings)
                names.update(
                    collect_return_names(
                        statement.body,
                        owner,
                        current_class,
                        body_bindings,
                        guard_deps | condition_deps,
                        set(return_stack),
                    )
                )
                names.update(
                    collect_return_names(
                        statement.orelse,
                        owner,
                        current_class,
                        else_bindings,
                        guard_deps | condition_deps,
                        set(return_stack),
                    )
                )
                merge_child_bindings(body_bindings, else_bindings)
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                names.update(
                    collect_return_names(
                        statement.body,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        set(return_stack),
                    )
                )
                names.update(
                    collect_return_names(
                        statement.orelse,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        set(return_stack),
                    )
                )
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                names.update(
                    collect_return_names(
                        statement.body,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        set(return_stack),
                    )
                )
            elif isinstance(statement, ast.Try):
                names.update(
                    collect_return_names(
                        statement.body,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        set(return_stack),
                    )
                )
                names.update(
                    collect_return_names(
                        statement.orelse,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        set(return_stack),
                    )
                )
                names.update(
                    collect_return_names(
                        statement.finalbody,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        set(return_stack),
                    )
                )
                for handler in statement.handlers:
                    names.update(
                        collect_return_names(
                            handler.body,
                            owner,
                            current_class,
                            dict(local_bindings),
                            set(guard_deps),
                            set(return_stack),
                        )
                    )
        return names

    def collect_expr_names(
        node: ast.AST,
        owner: str,
        current_class: str,
        name_map: Mapping[str, str],
        local_bindings: Mapping[str, set[str]],
        return_stack: set[tuple[str, str, str]] | None = None,
    ) -> set[str]:
        current = _unwrap_expr(node)
        passthrough_value = _passthrough_wrapper_value_node(current)
        if passthrough_value is not None:
            return collect_expr_names(
                passthrough_value,
                owner,
                current_class,
                name_map,
                local_bindings,
                return_stack,
            )
        resolved = _resolve_expr_canonical_names(name_map, current)
        if resolved:
            return resolved
        if isinstance(current, ast.Name):
            if current.id in local_bindings:
                return set(local_bindings[current.id])
            name = _qualify_internal_name(owner, current.id)
            return {name} if name else set()
        if isinstance(current, ast.Attribute):
            parts = _attribute_parts(current)
            if parts and parts[0] == "self":
                name = _qualify_internal_name(owner, ".".join(parts[1:]))
                return {name} if name else set()
            return set()
        if isinstance(current, ast.Subscript):
            text = _expr_text(current)
            if text.startswith("self."):
                name = _qualify_internal_name(owner, text[5:])
                return {name} if name else set()
            return set()
        if isinstance(current, ast.Call):
            names: set[str] = set()
            for arg in current.args:
                names.update(collect_expr_names(arg, owner, current_class, name_map, local_bindings, return_stack))
            for keyword in current.keywords:
                if keyword.value is not None:
                    names.update(
                        collect_expr_names(
                            keyword.value,
                            owner,
                            current_class,
                            name_map,
                            local_bindings,
                            return_stack,
                        )
                    )
            ref = _resolve_method_call_context(current, current_class, owner, class_attr_types)
            if ref is None:
                return names
            callee_class, callee_method_name, callee_owner = ref
            callee_method = class_methods.get(callee_class, {}).get(callee_method_name)
            if callee_method is None:
                return names
            key = (callee_class, callee_method_name, callee_owner)
            if return_stack is not None and key in return_stack:
                return names
            next_stack = set(return_stack or set())
            next_stack.add(key)
            names.update(
                collect_return_names(
                    callee_method.body,
                    callee_owner,
                    callee_class,
                    call_bindings(current, callee_method, owner, current_class, local_bindings),
                    set(),
                    next_stack,
                )
            )
            return names
        names: set[str] = set()
        for child in ast.iter_child_nodes(current):
            names.update(collect_expr_names(child, owner, current_class, name_map, local_bindings, return_stack))
        return names

    def visit_nested_calls(
        node: ast.AST,
        owner: str,
        current_class: str,
        local_bindings: Mapping[str, set[str]],
    ) -> None:
        for child in ast.walk(node):
            ref = _resolve_method_call_context(child, current_class, owner, class_attr_types)
            if ref is None:
                continue
            current = _unwrap_expr(child)
            if not isinstance(current, ast.Call):
                continue
            callee_class, callee_method_name, callee_owner = ref
            callee_method = class_methods.get(callee_class, {}).get(callee_method_name)
            if callee_method is None:
                continue
            visit_method(
                callee_class,
                callee_method_name,
                callee_owner,
                call_bindings(current, callee_method, owner, current_class, local_bindings),
            )

    def visit_block(
        statements: Sequence[ast.stmt],
        owner: str,
        current_class: str,
        local_bindings: dict[str, set[str]],
        guard_deps: set[str],
        repeated_locals: set[str],
    ) -> None:
        def merge_child_bindings(*children: Mapping[str, set[str]]) -> None:
            for name in {key for child in children for key in child}:
                merged: set[str] = set()
                for child in children:
                    merged.update(child.get(name, set()))
                if merged:
                    local_bindings[name] = merged

        emit_local_targets = "." not in owner
        for statement in statements:
            visit_nested_calls(statement, owner, current_class, local_bindings)
            if isinstance(statement, ast.Assign):
                passthrough_value = _passthrough_wrapper_value_node(statement.value)
                deps = guard_deps | collect_expr_names(
                    statement.value,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                )
                target_local_owner = owner
                current = _unwrap_expr(statement.value)
                if isinstance(current, ast.Call):
                    ref = _resolve_method_call_context(current, current_class, owner, class_attr_types)
                    if ref is not None:
                        target_names = {
                            item.id
                            for target in statement.targets
                            for item in _iter_target_nodes(target)
                            if isinstance(item, ast.Name) and item.id in repeated_locals
                        }
                        if target_names:
                            target_local_owner = ref[2]
                for target in statement.targets:
                    target_names = (
                        _passthrough_local_target_names(target, local_bindings)
                        if passthrough_value is not None
                        else set()
                    )
                    if not target_names:
                        target_names = _collect_internal_target_names(
                            target,
                            owner,
                            name_map,
                            emit_local_targets=emit_local_targets,
                            local_owner=target_local_owner,
                        )
                    _add_main_control_edges(edge_keys, deps, target_names)
                    _bind_local_target_names(target, target_names or deps, local_bindings)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                passthrough_value = _passthrough_wrapper_value_node(statement.value)
                deps = guard_deps | collect_expr_names(
                    statement.value,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                )
                target_local_owner = owner
                current = _unwrap_expr(statement.value)
                if isinstance(current, ast.Call):
                    ref = _resolve_method_call_context(current, current_class, owner, class_attr_types)
                    if ref is not None and isinstance(statement.target, ast.Name) and statement.target.id in repeated_locals:
                        target_local_owner = ref[2]
                target_names = (
                    _passthrough_local_target_names(statement.target, local_bindings)
                    if passthrough_value is not None
                    else set()
                )
                if not target_names:
                    target_names = _collect_internal_target_names(
                        statement.target,
                        owner,
                        name_map,
                        emit_local_targets=emit_local_targets,
                        local_owner=target_local_owner,
                    )
                _add_main_control_edges(edge_keys, deps, target_names)
                _bind_local_target_names(statement.target, target_names or deps, local_bindings)
            elif isinstance(statement, ast.AugAssign):
                deps = guard_deps | collect_expr_names(
                    statement.value,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                )
                deps.update(
                    collect_expr_names(
                        statement.target,
                        owner,
                        current_class,
                        name_map,
                        local_bindings,
                    )
                )
                target_names = _collect_internal_target_names(
                    statement.target,
                    owner,
                    name_map,
                    emit_local_targets=emit_local_targets,
                )
                _add_main_control_edges(edge_keys, deps, target_names)
                _bind_local_target_names(statement.target, target_names or deps, local_bindings)

            if isinstance(statement, ast.If):
                condition_deps = collect_expr_names(
                    statement.test,
                    owner,
                    current_class,
                    name_map,
                    local_bindings,
                )
                body_bindings = dict(local_bindings)
                else_bindings = dict(local_bindings)
                visit_block(
                    statement.body,
                    owner,
                    current_class,
                    body_bindings,
                    guard_deps | condition_deps,
                    repeated_locals,
                )
                visit_block(
                    statement.orelse,
                    owner,
                    current_class,
                    else_bindings,
                    guard_deps | condition_deps,
                    repeated_locals,
                )
                merge_child_bindings(body_bindings, else_bindings)
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                visit_block(
                    statement.body,
                    owner,
                    current_class,
                    dict(local_bindings),
                    set(guard_deps),
                    repeated_locals,
                )
                visit_block(
                    statement.orelse,
                    owner,
                    current_class,
                    dict(local_bindings),
                    set(guard_deps),
                    repeated_locals,
                )
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                visit_block(
                    statement.body,
                    owner,
                    current_class,
                    dict(local_bindings),
                    set(guard_deps),
                    repeated_locals,
                )
            elif isinstance(statement, ast.Try):
                visit_block(
                    statement.body,
                    owner,
                    current_class,
                    dict(local_bindings),
                    set(guard_deps),
                    repeated_locals,
                )
                visit_block(
                    statement.orelse,
                    owner,
                    current_class,
                    dict(local_bindings),
                    set(guard_deps),
                    repeated_locals,
                )
                visit_block(
                    statement.finalbody,
                    owner,
                    current_class,
                    dict(local_bindings),
                    set(guard_deps),
                    repeated_locals,
                )
                for handler in statement.handlers:
                    visit_block(
                        handler.body,
                        owner,
                        current_class,
                        dict(local_bindings),
                        set(guard_deps),
                        repeated_locals,
                    )

    def visit_method(
        class_name: str,
        method_name: str,
        owner: str,
        parameter_bindings: Mapping[str, set[str]],
    ) -> None:
        key = (class_name, method_name, owner)
        if key in visited:
            return
        method = class_methods.get(class_name, {}).get(method_name)
        if method is None:
            return
        visited.add(key)
        repeated_locals = _repeated_local_target_names(method)
        local_bindings = {
            name: set(parameter_bindings.get(name, set()))
            for name in _function_parameter_names(method)
        }
        visit_block(method.body, owner, class_name, local_bindings, set(), repeated_locals)

    for class_name, methods in class_methods.items():
        if "controller_step" in methods:
            visit_method(class_name, "controller_step", class_name, {})
    if "PLCController" in class_methods and "calculate" in class_methods["PLCController"]:
        visit_method("PLCController", "calculate", "PLCController", {})
    for path, class_name, node in _iter_function_defs(source_trees):
        if path.name != "simulation.py":
            continue
        for child in ast.walk(node):
            ref = _resolve_method_call_context(child, class_name, None, class_attr_types)
            if ref is None:
                continue
            current = _unwrap_expr(child)
            if not isinstance(current, ast.Call):
                continue
            callee_class, method_name, owner = ref
            if method_name not in {"update", "calculate", "controller_step"}:
                continue
            callee_method = class_methods.get(callee_class, {}).get(method_name)
            if callee_method is None:
                continue
            visit_method(
                callee_class,
                method_name,
                owner,
                call_bindings(current, callee_method, class_name or "", class_name or "", {}),
            )
    return [
        {"x": source, "y": target, "kind": kind}
        for source, target, kind in sorted(edge_keys)
    ]


def _edge_payload(source: str, target: str) -> dict[str, str]:
    return {"x": source, "y": target, "kind": "main_control_dependency"}


def _add_main_control_edges(
    edge_keys: set[tuple[str, str, str]],
    sources: set[str],
    targets: set[str],
) -> None:
    for target in targets:
        for source in sources:
            if source != target:
                edge_keys.add((source, target, "main_control_dependency"))


def _call_argument_nodes(
    call: ast.Call,
    callee_method: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> list[ast.AST]:
    if callee_method is None:
        return list(call.args)
    parameter_names = _function_parameter_names(callee_method)
    used_names = _function_used_parameter_names(callee_method)
    keyword_map = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None and keyword.value is not None
    }
    argument_nodes: list[ast.AST] = []
    for index, parameter_name in enumerate(parameter_names):
        if parameter_name not in used_names:
            continue
        if index < len(call.args):
            argument_nodes.append(call.args[index])
            continue
        node = keyword_map.get(parameter_name)
        if node is not None:
            argument_nodes.append(node)
    return argument_nodes or list(call.args)


def _extract_edges_from_statements(
    statements: Sequence[ast.stmt],
    current_class: str | None,
    name_map: Mapping[str, str],
    controller_setpoints: Mapping[str, set[str]],
    edge_keys: set[tuple[str, str, str]],
    change_bindings: dict[str, set[str]],
    class_methods: Mapping[str, Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    class_attr_types: Mapping[str, Mapping[str, str]],
) -> None:
    for statement in statements:
        if isinstance(statement, ast.Assign):
            value = statement.value
            method_name = _call_method_name(value)
            if method_name in {"update", "calculate"}:
                source_names: set[str] = set()
                current = _unwrap_expr(value)
                if isinstance(current, ast.Call):
                    callee_ref = _resolve_method_call_ref(value, current_class, class_attr_types)
                    callee_method = None
                    if callee_ref is not None:
                        callee_method = class_methods.get(callee_ref[0], {}).get(callee_ref[1])
                    if method_name == "calculate" and current.args:
                        source_nodes = _call_argument_nodes(current, callee_method)
                        if source_nodes:
                            source_names.update(_resolve_expr_canonical_names(name_map, source_nodes[0]))
                        source_names.update(
                            controller_setpoints.get(_call_controller_name(value), set())
                        )
                    else:
                        for arg in _call_argument_nodes(current, callee_method):
                            source_names.update(_resolve_expr_canonical_names(name_map, arg))
                target_names: set[str] = set()
                for target in statement.targets:
                    for item in _iter_target_nodes(target):
                        target_names.update(_resolve_expr_canonical_names(name_map, item))
                _add_main_control_edges(edge_keys, source_names, target_names)
            elif method_name == "calculate_change":
                current = _unwrap_expr(value)
                source_names: set[str] = set()
                if isinstance(current, ast.Call) and current.args:
                    source_names.update(_resolve_expr_canonical_names(name_map, current.args[0]))
                for target in statement.targets:
                    for item in _iter_target_nodes(target):
                        if isinstance(item, ast.Name):
                            change_bindings[item.id] = set(source_names)

            for target in statement.targets:
                for item in _iter_target_nodes(target):
                    if not (
                        isinstance(item, ast.Attribute)
                        and item.attr == "setpoint"
                    ):
                        continue
                    source_names = _resolve_expr_canonical_names(name_map, value)
                    if source_names:
                        controller_setpoints[_expr_text(item.value)] = set(source_names)
            value_sources: set[str] = set()
            for item in ast.walk(value):
                if isinstance(item, ast.Name):
                    value_sources.update(change_bindings.get(item.id, set()))
            if value_sources:
                target_names: set[str] = set()
                for target in statement.targets:
                    for item in _iter_target_nodes(target):
                        target_names.update(_resolve_expr_canonical_names(name_map, item))
                _add_main_control_edges(edge_keys, value_sources, target_names)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            if (
                isinstance(statement.target, ast.Attribute)
                and statement.target.attr == "setpoint"
            ):
                source_names = _resolve_expr_canonical_names(name_map, statement.value)
                if source_names:
                    controller_setpoints[_expr_text(statement.target.value)] = set(source_names)
            value_sources: set[str] = set()
            for item in ast.walk(statement.value):
                if isinstance(item, ast.Name):
                    value_sources.update(change_bindings.get(item.id, set()))
            if value_sources:
                target_names = _resolve_expr_canonical_names(name_map, statement.target)
                _add_main_control_edges(edge_keys, value_sources, target_names)
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.AST):
            if isinstance(statement.value, ast.BinOp):
                source_names: set[str] = set()
                for node in ast.walk(statement.value):
                    if isinstance(node, ast.Name):
                        source_names.update(change_bindings.get(node.id, set()))
                target_names = _resolve_expr_canonical_names(name_map, statement.target)
                _add_main_control_edges(edge_keys, source_names, target_names)

        nested_blocks: list[Sequence[ast.stmt]] = []
        if isinstance(statement, ast.If):
            nested_blocks.extend([statement.body, statement.orelse])
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            nested_blocks.extend([statement.body, statement.orelse])
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            nested_blocks.append(statement.body)
        elif isinstance(statement, ast.Try):
            nested_blocks.extend(
                [
                    statement.body,
                    statement.orelse,
                    statement.finalbody,
                    *[handler.body for handler in statement.handlers],
                ]
            )
        for block in nested_blocks:
            _extract_edges_from_statements(
                block,
                current_class,
                name_map,
                controller_setpoints,
                edge_keys,
                dict(change_bindings),
                class_methods,
                class_attr_types,
            )


def _ast_main_control_edges(
    source_trees: Sequence[tuple[Path, ast.AST]],
    variable_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    name_map = _registry_name_map(variable_registry)
    class_methods = _class_method_index(source_trees)
    class_attr_types = _class_attr_type_map(source_trees)
    edge_keys: set[tuple[str, str, str]] = set()
    for _path, class_name, node in _iter_function_defs(source_trees):
            controller_setpoints: dict[str, set[str]] = {}
            _extract_edges_from_statements(
                node.body,
                class_name,
                name_map,
                controller_setpoints,
                edge_keys,
                {},
                class_methods,
                class_attr_types,
            )
    return [
        {"x": source, "y": target, "kind": kind}
        for source, target, kind in sorted(edge_keys)
    ]


def _expand_dependency_terms(
    raw_terms: set[str],
    name_map: Mapping[str, str],
    local_deps: Mapping[str, set[str]],
) -> set[str]:
    expanded: set[str] = set()
    pending = list(raw_terms)
    seen: set[str] = set()
    while pending:
        term = pending.pop()
        if term in seen:
            continue
        seen.add(term)
        if term in name_map:
            expanded.add(name_map[term])
        elif term in local_deps:
            pending.extend(local_deps[term])
    return expanded


def _source_dependency_edges(
    source_trees: Sequence[tuple[Path, ast.AST]],
    variable_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    name_map = _registry_name_map(variable_registry)
    edge_keys: set[tuple[str, str]] = set()
    for _path, tree in source_trees:
        local_deps: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                deps = _expand_dependency_terms(_node_terms(node.value), name_map, local_deps)
                for target in node.targets:
                    for target_name in _target_names(target):
                        if target_name in name_map:
                            target_canonical = name_map[target_name]
                            edge_keys.update(
                                (dep, target_canonical)
                                for dep in deps
                                if dep != target_canonical
                            )
                        local_deps[target_name] = set(deps)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                deps = _expand_dependency_terms(_node_terms(node.value), name_map, local_deps)
                for target_name in _target_names(node.target):
                    if target_name in name_map:
                        target_canonical = name_map[target_name]
                        edge_keys.update(
                            (dep, target_canonical)
                            for dep in deps
                            if dep != target_canonical
                        )
                    local_deps[target_name] = set(deps)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    target_name = _dict_key_name(key)
                    if target_name not in name_map:
                        continue
                    target_canonical = name_map[target_name]
                    deps = _expand_dependency_terms(_node_terms(value), name_map, local_deps)
                    edge_keys.update(
                        (dep, target_canonical)
                        for dep in deps
                        if dep != target_canonical
                    )
    return [
        {"x": source, "y": target, "kind": "source_dependency"}
        for source, target in sorted(edge_keys)
    ]


def _build_g_payload(
    source_trees: Sequence[tuple[Path, ast.AST]],
    variable_registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    edge_keys = {
        (edge["x"], edge["y"], edge["kind"])
        for edge in _ast_main_control_edges(source_trees, variable_registry)
    }
    edge_keys.update(
        (edge["x"], edge["y"], edge["kind"])
        for edge in _controller_internal_edges(source_trees, variable_registry)
    )
    return {
        "E": [
            {"x": source, "y": target, "kind": kind}
            for source, target, kind in sorted(edge_keys)
        ]
    }


def _is_controller_output_var(record: Mapping[str, Any]) -> bool:
    layer = str(record.get("layer", "")).lower()
    roles = {str(role).lower() for role in record.get("roles", [])}
    return (
        layer in {"actuator", "setpoint", "cascade_setpoint", "controller_output"}
        and "injectable" in roles
    )


def _manifest_injection_points(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    points: dict[str, Mapping[str, Any]] = {}
    for item in manifest.get("injection_points", []):
        if isinstance(item, Mapping):
            name = str(item.get("name", item.get("field", ""))).strip()
            if name:
                points[name] = item
    return points


def _registry_by_manifest_name(
    variable_registry: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in variable_registry:
        name = str(record.get("manifest_name", "")).strip()
        if name:
            if name in records:
                raise ValueError(f"duplicate registry manifest_name: {name}")
            records[name] = record
    return records


def _validate_manifest_registry_alignment(
    points: Mapping[str, Mapping[str, Any]],
    registry_records: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = sorted(name for name in points if name not in registry_records)
    if missing:
        raise ValueError(f"missing registry record for injection point: {missing}")


def _validate_stage1_w_roots(
    points: Mapping[str, Mapping[str, Any]],
    registry_records: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, point in points.items():
        root = str(point.get("stage1_w_root", "")).strip()
        if root and root not in points:
            raise ValueError(f"unknown stage1_w_root for {name}: {root}")
        if root and not _is_controller_output_var(registry_records[root]):
            raise ValueError(f"stage1_w_root is not writable for {name}: {root}")


def _is_stage1_w_root_record(
    record: Mapping[str, Any],
    points: Mapping[str, Mapping[str, Any]],
) -> bool:
    manifest_name = str(record.get("manifest_name", "")).strip()
    point = points.get(manifest_name)
    if point is None:
        return False
    root = str(point.get("stage1_w_root", "")).strip()
    return not root or root == manifest_name


def _build_writable_subset(
    variable_registry: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    points = _manifest_injection_points(manifest)
    registry_records = _registry_by_manifest_name(variable_registry)
    _validate_manifest_registry_alignment(points, registry_records)
    _validate_stage1_w_roots(points, registry_records)
    records = [
        dict(record)
        for record in variable_registry
        if _is_controller_output_var(record)
        and _is_stage1_w_root_record(record, points)
    ]
    return sorted(records, key=lambda item: str(item.get("canonical_name", "")))


def _build_payload(
    source_trees: Sequence[tuple[Path, ast.AST]],
    variable_registry: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    predicates: Sequence[Mapping[str, Any]],
    hazards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    g_payload = _build_g_payload(source_trees, variable_registry)
    variables = sorted(
        {
            edge["x"]
            for edge in g_payload["E"]
        }
        | {
            edge["y"]
            for edge in g_payload["E"]
        }
        | {
            str(rule.get("var", "")).strip()
            for rule in [*predicates, *hazards]
            if str(rule.get("var", "")).strip()
        }
        | {
            str(rule.get("setpoint_var", "")).strip()
            for rule in predicates
            if str(rule.get("setpoint_var", "")).strip()
        }
    )
    d_payload = {"variables": variables, "attributes": []}
    return {
        "D": d_payload,
        "G": g_payload,
        "P": [dict(rule) for rule in predicates],
        "H": [dict(rule) for rule in hazards],
        "W": _build_writable_subset(variable_registry, manifest),
    }


def build_extraction(
    *,
    controller_path: Path,
    simulation_path: Path,
    variable_registry: Sequence[Mapping[str, Any]],
    manifest_path: Path | None,
    dependency_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    controller_source = Path(controller_path)
    simulation_source = Path(simulation_path)
    _validate_source_filename(controller_source, "controller.py")
    _validate_source_filename(simulation_source, "simulation.py")
    controller_trees = _parse_sources([controller_source])
    simulation_trees = _parse_sources([simulation_source])
    dependency_files = _discover_dependency_files(
        [controller_source, simulation_source, *(dependency_paths or [])]
    )
    dependency_trees = _parse_sources(dependency_files)
    manifest = _load_manifest(manifest_path)
    predicates = _load_rules(controller_trees, "ALARM_RULES")
    hazards = _load_rules(simulation_trees, "HAZARD_RULES")
    _validate_rule_id_oracle(predicates, manifest.get("alarm_rules", []), "A")
    _validate_rule_id_oracle(hazards, manifest.get("hazard_rules", []), "H")
    predicates = _merge_manifest_rule_metadata(predicates, manifest.get("alarm_rules", []))
    hazards = _merge_manifest_rule_metadata(hazards, manifest.get("hazard_rules", []))
    predicates = _canonicalize_rule_variables(
        predicates,
        variable_registry,
        {"var": "alarm variable", "setpoint_var": "alarm setpoint"},
    )
    hazards = _canonicalize_rule_variables(
        hazards,
        variable_registry,
        {"var": "hazard variable"},
    )
    _validate_alarm_setpoints_in_registry(predicates, variable_registry)
    return _build_payload(dependency_trees, variable_registry, manifest, predicates, hazards)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def write_stage1_output(
    *,
    extraction: Mapping[str, Any],
    output_root: Path,
    platform: str,
    seed: int,
) -> dict[str, Path]:
    platform_name = str(platform).strip().lower()
    artifact_dir = stage1_artifact_dir(platform)
    root = Path(output_root)
    artifact_path = root / "results" / "stage1" / artifact_dir / "extraction.json"
    log_path = root / "results" / "logs" / "stage1" / platform_name / "events.jsonl"
    manifest_path = root / "results" / "manifests" / "stage1" / platform_name / "run_manifest.json"
    artifact = {"seed": int(seed), **dict(extraction)}
    _write_json(artifact_path, artifact)
    manifest = {
        "stage": "stage1",
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
        "platform": platform_name,
        "seed": int(seed),
        "event": "stage1_output_written",
        "message": "stage1 extraction artifact written",
        "artifact": str(artifact_path),
        "manifest": str(manifest_path),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return {"artifact": artifact_path, "log": log_path, "manifest": manifest_path}


def run_stage1_extraction(
    *,
    controller_path: Path,
    simulation_path: Path,
    variable_registry: Sequence[Mapping[str, Any]],
    manifest_path: Path | None,
    output_root: Path,
    platform: str,
    seed: int,
    dependency_paths: Sequence[Path] | None = None,
) -> dict[str, Path]:
    extraction = build_extraction(
        controller_path=controller_path,
        simulation_path=simulation_path,
        dependency_paths=dependency_paths,
        variable_registry=variable_registry,
        manifest_path=manifest_path,
    )
    return write_stage1_output(
        extraction=extraction,
        output_root=output_root,
        platform=platform,
        seed=seed,
    )

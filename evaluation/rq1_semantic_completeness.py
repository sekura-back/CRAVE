# RQ1
"""Artifact-internal RQ1 summary for semantic completeness and attack surface."""

import json
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ARTIFACT_ROOT / "results" / "stage1"
STAGE2 = ARTIFACT_ROOT / "results" / "stage2"
RQ1_OUT = ARTIFACT_ROOT / "results" / "rq1"


SYSTEMS = {
    "boiler": {
        "program": STAGE1 / "boiler_program_extraction.json",
        "combo": STAGE2 / "initial_combo_boiler.json",
        "manifest": ARTIFACT_ROOT / "simulators" / "boiler_ccs" / "system_manifest.json",
        "out": RQ1_OUT / "boiler_rq1_summary.json",
        "published_expert_counts": {
            "variables_count": 227,
            "edges_count": 383,
            "writable_points_count": 21,
            "alarm_predicates_count": 3,
            "hazard_predicates_count": 3,
        },
    },
    "te": {
        "program": STAGE1 / "semantic_model_te.json",
        "combo": STAGE2 / "initial_combo_te.json",
        "manifest": ARTIFACT_ROOT / "simulators" / "tennessee_eastman" / "system_manifest.json",
        "out": RQ1_OUT / "te_rq1_summary.json",
        "published_expert_counts": {
            "variables_count": 1273,
            "edges_count": 1917,
            "writable_points_count": 63,
            "alarm_predicates_count": 7,
            "hazard_predicates_count": 8,
        },
    },
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_writable_points(root: dict) -> int:
    writable = root.get("W")
    if isinstance(writable, list):
        return len(writable)

    attrs = root.get("D", {}).get("a", {})
    if isinstance(attrs, dict):
        return sum(
            1
            for item in attrs.values()
            if isinstance(item, dict) and item.get("attackable") is True
        )
    return 0


def count_semantic_objects(model: dict) -> dict:
    root = model.get("M", model)
    return {
        "variables_count": len(root["D"]["V"]),
        "edges_count": len(root["G"]["E"]),
        "writable_points_count": count_writable_points(root),
        "alarm_predicates_count": len(root.get("P", [])),
        "hazard_predicates_count": len(root.get("H", [])),
    }


def build_ratio_components(program_counts: dict, expert_counts: dict) -> dict:
    ratios = {}
    for key in (
        "variables_count",
        "edges_count",
        "writable_points_count",
        "alarm_predicates_count",
        "hazard_predicates_count",
    ):
        expert_val = expert_counts[key]
        ratio = (program_counts[key] / expert_val) if expert_val else 0.0
        ratios[key] = min(ratio, 1.0)
    return ratios


def build_platform_summary(platform: str, cfg: dict) -> dict:
    program = load_json(cfg["program"])
    combo = load_json(cfg["combo"])
    manifest = load_json(cfg["manifest"])

    program_counts = count_semantic_objects(program)
    expert_counts = dict(cfg["published_expert_counts"])
    ratios = build_ratio_components(program_counts, expert_counts)
    c_sem = sum(ratios.values()) / len(ratios) * 100.0

    inj_pts = manifest.get("injection_points", [])
    actuator = sum(1 for p in inj_pts if p.get("layer") == "actuator")
    setpoint = sum(1 for p in inj_pts if p.get("layer") == "setpoint")
    attack_surface_reduction_pct = (
        round(100.0 * (len(inj_pts) - 1) / len(inj_pts), 2) if inj_pts else 0.0
    )
    top1 = combo[0] if combo else {}

    return {
        "system": platform,
        "source_policy": "artifact_internal_only",
        "expert_baseline_policy": "published_counts_embedded_in_script",
        "program_counts": program_counts,
        "expert_counts": expert_counts,
        "semantic_completeness_pct": round(c_sem, 2),
        "semantic_completeness_components": {
            key: round(val * 100.0, 2) for key, val in ratios.items()
        },
        "attack_surface": {
            "injection_points_count": len(inj_pts),
            "injection_points_actuator": actuator,
            "injection_points_setpoint": setpoint,
            "selected_hazard_driver": top1.get("S", [None])[0],
            "selected_hazard_driver_rank": int(top1.get("rank", 1)),
            "selected_hazard_driver_score": float(top1.get("score", 0.0)),
            "attack_surface_reduction_pct": attack_surface_reduction_pct,
        },
    }


def main():
    RQ1_OUT.mkdir(parents=True, exist_ok=True)

    summary = {}
    for platform, cfg in SYSTEMS.items():
        platform_summary = build_platform_summary(platform, cfg)
        summary[platform] = platform_summary
        cfg["out"].write_text(
            json.dumps(platform_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[rq1] saved {cfg['out']}")

    summary_path = RQ1_OUT / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[rq1] saved {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = ROOT / "results" / "stage2" / "boiler_base_boundary.json"
COND_FILE = ROOT / "results" / "stage2" / "boiler_conditional_expansion.json"
OUT_FILE = ROOT / "results" / "rq2" / "boiler_results.json"

K_PHYS = 5.0
DURATION_STEP = 5


def main() -> None:
    base = json.loads(BASE_FILE.read_text(encoding="utf-8"))
    cond = json.loads(COND_FILE.read_text(encoding="utf-8"))

    base_pts = [
        b for b in base["boundaries"]
        if b.get("direction") == "pos" and b.get("lower_target_amp") is not None
    ]
    cond_pts = [
        r for r in cond.get("expanded_boundary", [])
        if r.get("direction") == "pos" and r.get("lower_target_amp") is not None
    ]

    base_area = sum(
        (float(b.get("upper_target_amp", K_PHYS)) - float(b["lower_target_amp"])) * DURATION_STEP
        for b in base_pts
    )
    cond_area = sum(
        (float(r["regular_amp"]) - float(r["lower_target_amp"])) * DURATION_STEP
        for r in cond_pts
    )
    total_area = K_PHYS * (int(base_pts[-1]["duration"]) - int(base_pts[0]["duration"]))
    total_ratio = (base_area + cond_area) / total_area * 100.0 if total_area else 0.0

    base_sims = int(base.get("total_simulations", 0))
    cond_sims = int(cond.get("summary", {}).get("total_simulations", 0))
    total_sims = base_sims + cond_sims
    total_points = len(base_pts) + len(cond_pts)

    out = {
        "source": {
            "base_boundary": str(BASE_FILE.relative_to(ROOT)),
            "conditional_expansion": str(COND_FILE.relative_to(ROOT)),
        },
        "parameter_space": {
            "duration_range": [int(base_pts[0]["duration"]), int(base_pts[-1]["duration"])],
            "duration_step": DURATION_STEP,
            "k_phys": K_PHYS,
            "total_area": round(total_area, 4),
        },
        "rq2_table": {
            "total_sims": total_sims,
            "base_boundary_points": len(base_pts),
            "cond_boundary_points": len(cond_pts),
            "avg_sims_per_boundary": round(total_sims / total_points, 2) if total_points else 0.0,
            "base_area": round(base_area, 4),
            "cond_area": round(cond_area, 4),
            "base_area_pct": round(base_area / total_area * 100.0, 4) if total_area else 0.0,
            "cond_area_pct": round(cond_area / total_area * 100.0, 4) if total_area else 0.0,
            "hazard_region_pct": round(total_ratio, 4),
        },
        "search_stats": {
            "base_total_simulations": base_sims,
            "cond_total_simulations": cond_sims,
            "cond_points_found": int(cond.get("summary", {}).get("cond_points_found", len(cond_pts))),
            "cond_dual_probe_calls": int(cond.get("summary", {}).get("dual_probe_calls", 0)),
        },
    }

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

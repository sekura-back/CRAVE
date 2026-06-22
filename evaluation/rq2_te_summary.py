from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = ROOT / "results" / "stage2" / "te_base_boundary.json"
OUT_FILE = ROOT / "results" / "rq2" / "te_results.json"

K_PHYS = 1.0


def main() -> None:
    base = json.loads(BASE_FILE.read_text(encoding="utf-8"))
    domain = base["domain"]
    duration_step = int(domain.get("duration_step_fine", domain.get("duration_step", 5)))

    base_pts = [
        b for b in base["boundaries"]
        if b.get("direction") == "pos"
        and b.get("lower_target_amp") is not None
        and b.get("upper_target_amp") is not None
    ]
    null_pts = [
        b for b in base["boundaries"]
        if b.get("direction") == "pos" and b.get("lower_target_amp") is None
    ]

    base_area = sum(
        (float(b["upper_target_amp"]) - float(b["lower_target_amp"])) * duration_step
        for b in base_pts
    )
    total_area = K_PHYS * (
        int(domain["duration_max"]) - int(domain["duration_min"])
    )
    total_sims = int(base.get("total_simulations", sum(int(b.get("n_simulations", 0)) for b in base["boundaries"])))

    out = {
        "source": {
            "base_boundary": str(BASE_FILE.relative_to(ROOT)),
        },
        "parameter_space": {
            "duration_range": [
                int(domain["duration_min"]),
                int(domain["duration_max"]),
            ],
            "duration_step": duration_step,
            "k_phys": K_PHYS,
            "total_area": round(total_area, 4),
        },
        "rq2_table": {
            "total_sims": total_sims,
            "base_boundary_points": len(base_pts),
            "cond_boundary_points": 0,
            "base_area": round(base_area, 4),
            "cond_area": 0.0,
            "base_area_pct": round(base_area / total_area * 100.0, 4) if total_area else 0.0,
            "cond_area_pct": 0.0,
            "hazard_region_pct": round(base_area / total_area * 100.0, 4) if total_area else 0.0,
        },
        "search_stats": {
            "base_total_simulations": total_sims,
            "base_null_points": len(null_pts),
            "first_valid_duration": int(base_pts[0]["duration"]) if base_pts else None,
            "last_valid_duration": int(base_pts[-1]["duration"]) if base_pts else None,
        },
    }

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

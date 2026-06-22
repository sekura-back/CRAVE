# Evaluation Entry

"""Artifact evaluation orchestrator for the currently migrated RQ scripts."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

STEPS = [
    ("rq1", ["rq1_semantic_completeness.py"], False),
    ("rq2", ["rq2_boiler_summary.py", "rq2_te_summary.py", "rq2_te_ablation.py"], False),
    ("rq3", ["rq3_region_quality.py"], True),
]


def run_step(name: str, scripts: list, accepts_workers: bool, workers: int) -> bool:
    print(f"\n========== STEP {name} ==========")
    t0 = time.time()
    for script in scripts:
        cmd = [sys.executable, str(HERE / script)]
        if accepts_workers:
            cmd += ["--workers", str(workers)]
        print(f"$ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"  STEP {name} FAILED (rc={rc})")
            return False
    print(f"========== STEP {name} OK ({time.time()-t0:.1f}s) ==========")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only", default=None,
                        help="Comma-separated step names to run (others skipped)")
    parser.add_argument("--skip", default=None,
                        help="Comma-separated step names to skip")
    args = parser.parse_args()

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None
    skip_set = set(s.strip() for s in args.skip.split(",")) if args.skip else set()

    failures = []
    overall_t0 = time.time()
    for name, scripts, accepts_workers in STEPS:
        if only_set is not None and name not in only_set:
            print(f"-- skip {name} (not in --only)")
            continue
        if name in skip_set:
            print(f"-- skip {name} (--skip)")
            continue
        ok = run_step(name, scripts, accepts_workers, args.workers)
        if not ok:
            failures.append(name)

    overall = time.time() - overall_t0
    print(f"\n=== run_all done in {overall:.1f}s ===")
    if failures:
        print(f"FAILED steps: {failures}")
        sys.exit(1)
    print("All steps OK")


if __name__ == "__main__":
    main()

"""Artifact compatibility wrapper for Stage 2/3 generic runtime execution."""

from simulators.boiler_ccs.runtime_executor import (  # noqa: F401
    annotate_rows,
    build_injection_hook,
    compute_run_key,
    create_executor,
    load_injection_registry,
    load_rules_from_manifest,
    load_simulation_module,
    normalize_injection_points,
    run_batch,
    score_trace,
    simulate,
)

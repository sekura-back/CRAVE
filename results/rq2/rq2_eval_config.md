# RQ2 Evaluation Config

- platform: `tennessee_eastman`
- variable: `xmv_07`
- target_hazard: `H-TEP-SEPARATOR-LEVEL-LOW`
- base_alarm: `A-TEP-SEP-LEVEL-TRACK`
- search_domain_T: `[100, 1000]`
- search_domain_K: `[0.0, 1.0]`
- runtime_rule: `duration + 100`
- replay_oracle: Stage 2 base target, hazard plus pre-hazard alarm
- reconstruction_policy: `target_sample_cell_union`
- CRAVE_budget: `2939`
- baseline_budget: `2939`
- random_seed: `460`
- reference_boundary_source: `results\rq2\te_crave_safe_boundary.json`

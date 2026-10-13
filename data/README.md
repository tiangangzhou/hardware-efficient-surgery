# Data

Frozen inputs and published results read by the scripts in `scripts/`.

## `shuttling/` (Fig. 3, Table S2)

- `table_s2.csv` — X, Z and nonparallel X+Z round times and the parallel X+Z
  average per round (ms) for every lift, protocol and AOD-pair budget. Regenerated
  exactly by `scripts/shuttling/extract_timings.py` from the animation
  `lp-surgery-aod-polynomials.html`, whose embedded JavaScript schedulers compute
  all times (tested with Node v25).
- `space_overhead.json` — physical-qubit counts of the bottom panel of Fig. 3.
- `gadgets/l{9,15,33}_{gauging,edge_expansion}.npz` — the Graph-WY / Graph-IGND
  gadgets (merged check matrices and mapping-cone maps) for one canonical logical,
  rebuilt and checked by `scripts/shuttling/build_graph_gadgets.py`.

The three shuttling scripts compare their results with these files; `--write` regenerates
them.

## `ler/` (Fig. 4, SM figure and tables on logical error rates)

- `l33/counts.json` — shots and failures for the 16 points of Fig. 4. The points
  at p = 0.002–0.003 were sampled on a cluster in 18–280 shards of 25,000 shots and
  decoded in two stages (Relay-BP on every shot, BP-LSD later on the non-converged
  shots); `scripts/ler/l33/run_batched.py` re-implements this driver, and
  `run_circuits.py` runs the same decoder chain inside sinter.
  **The p = 0.0016 rows were recorded by hand from the run summary; no raw data were
  archived for them.**
- `l33/failure_masks.json` — every logical failure of the cluster points as an
  observable mistake mask, split into T1 (Relay-BP converged to a wrong correction)
  and T2 (BP-LSD fallback); the SM table of failures by decoder stage and type
  (`circuit_error_table.py`) is computed from it.
- `l33/code_capacity_shards.csv`, `l33/code_capacity_campaigns.json` — one row per
  shard of the code-capacity campaigns and their settings.
- `l9/parent.csv`, `l9/topup.csv`, `l9/provenance.json` — sinter statistics of the
  l = 9 campaign (40 tasks) and of its p = 0.001 extension, which replaces the
  parent rows of the same tasks.

The runs used numpy 2.2, scipy 1.15, relay-bp 0.2.1 and a development version of
qLDPC. With the versions pinned here the detector error models are unchanged
(identical l = 9 DEMs; identical mechanisms and probabilities at l = 33), but new
samples agree with the published counts statistically, not bit for bit.

## `codes/` (SM code examples and distances)

- `witnesses.json` — explicit minimum-weight logical operators of the three merged
  codes (qubit order of `algebraic_qec.examples.surgery_instances`).
- `qdistrnd/l{9,15,33}_{X,Z}.json` — the QDistRnd runs of the SM distance table
  (QDistRnd 0.9.5, GAP 4.15.1, seed 260920). The l = 33 Z search stopped at a
  seven-day timeout; its record is the last checkpoint.

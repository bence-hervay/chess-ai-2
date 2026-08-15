# SHSD Stage B — Research instrumentation

Experiment card: `reports/shsd/cards/stage_b_teacher_validation.md`
(written before implementation).

## Status

PASS — the instrumentation separately exposes model error, search
error, move-ordering error, and computational cost (§55 gate), verified
against the exact fc-tiny oracle across 3 seeds and two evaluators.

## Research question

Can the system generate trustworthy deep-search teacher labels with
full target provenance, measure shallow-to-deep residuals and best-move
stability, and attribute error to the model, the search budget, or the
move ordering — cheaply and reproducibly?

## Implementation

- `SearchResult` now records `(best move, value)` per completed
  iterative-deepening iteration (§18.2 best-move stability). One field;
  no behavioral change (all 88 pre-existing tests unchanged and green).
- `src/data.rs` (855 lines incl. ~300 of tests): `TeacherRecord` with
  `Provenance` (§19.9), positions stored as exact **action paths from
  the initial state** (replay reconstructs full state incl. repetition
  history; `position_key` alone cannot), deterministic hash-based
  position sampling with visit weights (§20.4), parallel labelling
  (shallow ladder + deep + counterfactual children §19.5), retrograde
  oracle join, and summary aggregation.
- `lab relabel` command + `RelabelConfig` (fully explicit,
  `deny_unknown_fields`), writing standard run artifacts
  (`resolved.toml`, `manifest.json`, `records.jsonl`, `summary.json`
  with §47.8 cost block).
- `parameter_ledger.json` (repo root): 15 entries covering every
  constant in production code paths touched so far, each classified
  learned / hyperparameter / mathematical / engineering (§13).
- `datasets/frozen/MANIFEST.json`: frozen-set registry — implicit
  exact hash-bucket val/test sets (fc-tiny, fc-small) and the first
  frozen file set `fc_full_shadow_v1.jsonl` (sha256-pinned, 300
  c03-labelled fc-full positions) as a constant shadow-search
  diagnostic for Stage C+.

## Deliberately excluded

Feature extraction and per-family feature-cost timing (lands with the
first `FeatureExtractor` in Stage C), structured evaluators, any search
behavior change (PVS/aspiration/quiescence/pruning), Stockfish-teacher
records through this path (Track B stays quarantined in `lab teacher`),
record sharding/compression (single JSONL files are ~1.5 MB at current
scales).

## Baselines

Zero evaluator (search-only, §9.2) vs the Phase-8 fc-tiny w64 selfplay
checkpoint (raw-model baseline, §9.3). Random trajectories are
evaluator-independent, so equal seeds label **identical positions** —
the comparison is paired by construction.

## Data

6 runs × ~780–880 distinct fc-tiny positions (of 83,947 reachable) from
600 random trajectories each, seeds {1,2,3}; oracle join rate **100%**
in all runs. Plus one fc-full run (300 positions from c03-champion
selfplay trajectories) for the residual instrument at target scale.

## Target provenance

All records `internal_search` with evaluator id (`"zero"` or checkpoint
path) embedded per record (§20.5); exact WDL/optimal sets joined from
the in-process retrograde solution where configured.

## Primary result

**Optimal-decision rate of the search label vs node budget, fc-tiny,
mean over 3 seeds (per-seed spread ≤ ±0.006):**

| Node budget | zero evaluator | w64 checkpoint |
|---:|---:|---:|
| 50 | 0.933 | 0.979 |
| 200 | 0.960 | 0.987 |
| 800 | 0.981 | 0.995 |
| 6,400 (deep) | 0.997 | 0.998 |

Monotone in budget for both evaluators in **every individual run**
(hypothesis 1), and the trained model beats zero at **every budget on
identical positions** (hypothesis 2) — at 50 nodes it removes 68% of
the decision error (6.7% → 2.1%). Counterfactual child-argmax optimal
rate: 0.988–0.994 across all runs.

## Secondary results

- **Move-ordering error is visible** (hypothesis 3): deep-best mean
  rank in the evaluator-induced root ordering 0.39–0.42 (zero,
  action-ID order) vs 0.18–0.24 (w64 policy); top-1 0.82–0.83 vs
  0.89–0.92.
- **Deep self-stability**: last-iteration-stable 0.92–0.93 (zero) vs
  0.96–0.97 (w64); mean best-move changes per search 0.13 vs 0.07 —
  a better evaluator also stabilizes search, measurably.
- **fc-tiny cannot exercise the value-residual instrument**: every
  non-mate value in all six runs is exactly 0. Two verified causes: the
  w64 model's value head is so draw-confident that
  `round(1000·(P(win)−P(loss))) = 0` everywhere, and deep searches
  exhaust the tiny tree (completed depths up to 512). The instrument
  correctly measures a true property of the drawish micro-game. Value
  residual work needs fc-small/fc-full/chess.
- **fc-full residual check** (c03 champion, 300 positions): nonzero
  residuals (mean |shallow−deep| 129–147 score units), best-move
  agreement with deep 0.38 (100 nodes) → 0.48 (1,600); ordering rank of
  deep best 7.49 mean, top-1 only 0.16 — large learned-move-ordering
  headroom (Stage F). **The 6,400-node fc-full teacher is itself
  unstable** (last-iteration stable 0.373, 1.73 best-move changes per
  search): §18.2's warning is quantified — Stage C must treat
  fixed-budget deep labels as noisy, and store stability with every
  record (it does).

## Statistical uncertainty

Binomial SE at n≈800, rate 0.98 is ≈0.005; every reported zero-vs-w64
gap at 50–800 nodes exceeds 4 SE. Monotonicity holds in all 6 runs
independently (sign test p = 2⁻⁶ per evaluator against random
ordering).

## Fixed-node / fixed-time result

Not applicable (no play-strength claim in this stage); cost measured
instead.

## CPU result

| Run | Wall | CPU | Utilization (8 threads) | Positions/s | Nodes |
|---|---:|---:|---:|---:|---:|
| fc-tiny zero (each) | ~1.1 s | ~6 s | 0.69–0.71 | ~730–770 | ~5.3–5.9 M |
| fc-tiny w64 (each) | ~5 s | ~34–37 s | 0.81–0.91 | ~150–172 | ~5.2–5.9 M |
| fc-full w64 | 28.1 s | 213.5 s | **0.95** | 10.7 | 4.8 M |

The tiny-zero runs are too short to amortize the serial retrograde
solve (≈0.4 s of the 1.1 s wall); the longer runs hit 0.91–0.95,
meeting the §66.7 target for the parallel stage. Peak RSS ≈ 60 MB
(tiny, incl. oracle) — memory is a non-issue at this scale.

## Correctness evidence

Five new lib tests (93 total green, `clippy -D warnings` clean):
path-replay round-trip on two rule families + illegal-path error;
record serde round-trip; sampling determinism and thread-independence
with dedup/weight/cap invariants; labelling thread-independence with
per-depth/well-formedness invariants; end-to-end oracle cross-check on
solved Connect-k 3×3 (full-width-budget deep labels must be exactly
optimal — they are, 100%, including child argmax).

## Ablations

Not applicable (instrumentation); the zero-vs-w64 pairing doubles as
the demonstration that the instrument separates its error sources.

## Failure cases

None blocking. Two honest limitations recorded: fc-tiny is useless for
value-residual work (see above), and utilization on sub-2-second runs
is bounded by the serial oracle solve.

## Learned parameter interpretation

None (no fitting in this stage).

## Complexity delta

- Production LOC: +~560 (data.rs minus tests) +~300 (lab.rs relabel) +6 (search.rs)
- Public types: +10 (record family: Provenance, SearchLabel, ChildLabel,
  TeacherRecord, TrajectorySpec, PositionSample; summary family:
  RelabelSummary, BudgetAgreement, OracleSummary, OracleJoinStats).
  Over the §14.1 guideline of 3 — declared in the experiment card: one
  record schema family and its aggregation, no behavioral abstractions,
  no traits, no plugin surface.
- Config keys: 1 new command config (12 required fields, 2 sub-enums)
- Dependencies: 0
- Permanent modes: 0 (relabel is one command, no optional forests)
- Deleted code: 0 (instrumentation stage)

## Parameter-provenance changes

`parameter_ledger.json` created (15 entries). New constants this stage:
`sample_salt` (engineering, arbitrary hash salt), `labelling_max_depth`
= 512 (engineering, safety bound). All record-affecting knobs are
required config fields recorded in `resolved.toml`.

## Decision

- **Retain**: record schema v1 (paths + provenance + stability +
  children), `lab relabel`, ledger, frozen-set manifest.
- **Reject**: nothing.
- **Revise later**: record sharding/compression when sets exceed ~100 MB.
- Selected recipe: none yet (no fitting).

## Next research question (Stage C)

Can a compact structured evaluator — learned piece/orientation counts,
piece-square/orientation terms, basic mobility, immediate
attacks/defences, promotion distance, side to move — beat the raw
sparse MLP per training position on exact Forward Chess data, and
provide positive search value? First step: the `FeatureExtractor`
boundary + the §9.4 structured linear baseline, fitted on fc-small
exact data (46.5M positions with per-child WDL), evaluated against the
frozen hash-bucket test sets and the fc-tiny/fc-small oracles.

## Exact reproducibility information

Commit: this one (see git log). Configs:
`configs/shsd/stage_b/relabel_fc_tiny_{zero,w64}_s{1,2,3}.toml`,
`relabel_fc_full_w64_s1.toml`. Runs:
`runs/20260815-2118*-relabel-fc-tiny-*` (6),
`runs/20260815-211948-relabel-fc-full-*`. Frozen:
`datasets/frozen/fc_full_shadow_v1.jsonl`
(sha256 878748c4…dd0edb). All runs deterministic given the config;
thread count changes wall-clock only (tested).

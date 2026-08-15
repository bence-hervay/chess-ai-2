# SHSD Stage F1 — Learned move ordering from exact data

Experiment card: `reports/shsd/cards/stage_f1_move_ordering.md`
(stage reordering justified there per §3.2: exact ranking targets are
free, the structured search had no ordering at all, and §59 predicts
ordering can out-value another static family).

## Status

PASS (with an honest ceiling note) — the ranker cuts nodes materially
and improves both fixed-node and (marginally) fixed-time play on
fc-small; the small match effect is a property of the saturated
benchmark, not of the mechanism, and the real target is the measured
fc-full ordering headroom.

## Research question

Does a compact learned move-ordering model reduce the nodes the
structured-evaluator search needs at fixed depth, without hurting
fixed-node or fixed-time decision quality?

## Implementation

- `FcMoveFeatures` (35 dims): moving/captured piece×orientation,
  promotion choice, gives-check (applied-board attack test), forward
  displacement, horizontal flag, destination attacked/defended, source
  attacked, destination relative rank, castle, en passant. Rule-level
  facts only; rotation-swap invariance property-tested on 300+ moves.
- `MoveRanker`: 35-weight linear scorer, pairwise logistic ranking loss
  (§18.5 exact optimal vs non-optimal), deterministic Adam + L2.
- `FcOrderedEvaluator`: structured value + optional ranker through the
  existing `Evaluator::policy_scores` hook — **zero search changes**;
  TT-move-first still applies on top. §69.4 property test: ordering
  never changes search values (30 random positions, full depth 5).
- `lab fit` model kind `ranker` (pairs from the exact child map, cap 8
  per position, deterministic by position key); `structured_match`
  accepts a ranker and a structured opponent.

## Baselines

Move-generation order (what the structured search used until now);
TT-move-first common to all arms; the MLP's policy-head ordering as
context (C1 probes).

## Data

fc-small exact; 100k train positions → 254,715 ranking pairs; val/test
50k/51k pairs; probes = 500 exact test-bucket positions. lr = 0.02
selected by val over {0.02, 0.1} (l2 = 1e-4 carried from C1); seeds
{1,2,3}.

## Primary result — nodes to fixed depth (same leaf model, 500 states)

| Depth | Nodes plain → ordered | Total ratio | Median per-state |
|---:|---|---:|---:|
| 4 | 98,072 → 78,923 | **0.805** | 0.873 |
| 6 | 494,265 → 357,718 | **0.724** | 0.807 |

Seed spread on the depth-6 ratio: 0.724–0.729. The hypothesis
threshold (≥25% at some depth) is met at depth 6.

## Secondary results

- **Ordering quality**: mean rank of the first exact-optimal move drops
  0.652 → 0.116; top-1 rate 0.750 → 0.926–0.930 (3 seeds). Pairwise
  test accuracy 0.804–0.807 (random floor 0.5).
- **Fixed-node decisions improve**: 400-node probe optimal rate
  0.982 → **0.990** (identical across seeds).
- **Matches** (300 pairs, v1@1M leaf both sides):
  - ordered vs plain, 400 nodes: 0.506 [0.500, 0.512];
  - ordered vs plain, 20 ms/move: 0.505 [0.501, 0.509] — the §35.6
    fixed-time requirement is met, marginally but with LCB > 0.5;
  - ordered vs mlp-w32@1M, 400 nodes: 0.480 [0.470, 0.490] (was 0.474
    without ordering — the C2 gap narrows slightly).
- **Learned ordering rules** (all discovered): queen-promotion strongly
  up, minor-promotions *down* (net negative with is_promotion);
  captures up — but capturing spent reversed minors down while
  capturing reversed queens/rooks up (coherent with the C1 value
  model); checks up; moving onto attacked squares down; fleeing
  attacked sources up; pawn advances up.

## Why the match effect is small (honest ceiling analysis)

fc-small at 400 nodes is nearly saturated: the plain search already
picks an exact-optimal move 98.2% of the time, so ordering can convert
at most ~1.8% of decisions, and paired-opening games mostly mirror.
The 27% node saving is ≈0.46 of a search doubling — worth single-digit
Elo at this saturation. The mechanism's value lives where ordering is
bad and depth is scarce: Stage B measured the fc-full champion's
policy ranking the deep-best move at mean 7.5 (top-1 16%). Transfer is
the next experiment.

## Statistical uncertainty

3 seeds, spreads quoted; match CIs are pair-level normal
approximations; the fixed-time LCB clears 0.5 by 1 count in the third
decimal — reported as marginal, not as a strong effect.

## CPU result

Ranker fit ≈ 90 s of a ~250 s run (tablebase passes dominate); probes
(500 states × {depth 4,6} × 2 + 400-node × 2) ≈ 80 s at 4 threads.
Matches: 600 games in 1–3 min. Move-feature extraction is one
`attacks()` scan + a board copy per move; the 20 ms/move match staying
above 0.5 shows the ordering cost does not eat its own gains.

## Correctness evidence

103 tests green; new: move-feature rotation-swap invariance,
hand-computed move features (capture/check/displacement/rank on the
Stage C test position, including Rc4+ discovered along the empty
file), ranker toy-rule learning + determinism + serde, and the
ordering-never-changes-values search property.

## Complexity delta

+3 public types (FcMoveFeatures, MoveRanker, RankPair) +
FcOrderedEvaluator; one fit model kind; one optional-with-default
config field on `structured_match` (kept optional so archived C2
configs replay — documented deviation from the all-required rule);
0 dependencies; 0 deletions.

## Parameter-provenance changes

Ledger: +`ranker_pair_cap` = 8 (engineering, deterministic-by-key
sampling bound). Ranker lr = 0.02 (val-selected, 2-point comparison,
runs `20260815-23*-move-ranker`), l2 carried from C1.

## Decision

- **Retain** the ranker as the production ordering for structured
  Forward Chess search: material node savings, better fixed-node
  decisions, no fixed-time regression, negligible complexity.
- **Next**: F2 — ordering (and value) at fc-full from teacher
  counterfactual records (`lab relabel` children, §19.5→§35 pipeline),
  where the measured ordering headroom is ~60× larger than what
  fc-small has left. This also exercises the Stage B instrumentation
  for its designed purpose: search-distillation without an oracle.

## Exact reproducibility information

Configs: `configs/shsd/stage_f/`. Runs: `runs/*fit-fc-small-move-ranker*`
(lr check + 3 seeds), `runs/*structmatch*` (3 matches). All fits
deterministic; movetime match wall-clock.

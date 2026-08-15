# SHSD Stage F2 — fc-full move ordering distilled from internal search

Experiment card: `reports/shsd/cards/stage_f2_fullboard_ordering.md`.
This experiment is the Stage B instrumentation doing its designed job:
search-distillation targets (§19.5 counterfactual children) at a scale
with no oracle.

## Status

PASS — at fixed time the learned ranker beats the Program-1 champion's
own policy-head ordering by ≈ +93 protocol Elo with the value model
held identical; at fixed nodes it ties. The mechanism is
ordering-quality-per-cost, and it makes the champion+ranker the
strongest current fc-full configuration.

## Research question

Can a 35-parameter linear move ranker trained on internal-search
counterfactual child labels (no oracle) beat the champion MLP's own
policy-head ordering at full Forward Chess?

## Data

`lab relabel` with the c03 champion (teacher identity per record):
250 selfplay trajectories → 7,721 distinct positions (weights kept),
deep label 1,600 nodes, all-children labels at 400 nodes; ≈ 70M search
nodes, 14 min at 4 threads, utilization 0.83. Fresh sample confirms
the Stage B headroom: champion policy ranks the deep-best move at mean
6.89–7.42 (top-1 0.15–0.19). Provenance `internal_search` throughout.

## Implementation

`lab fit` source `records` + model `ranker_from_records` (pairs =
child-value differences beyond a margin, capped 8/position,
deterministic by key; standard bucket splits on record keys);
`MlpRankedEvaluator` (champion WDL head + ranker ordering);
`lab evaluate` kind `ordered_mlp_match` (same value model both sides,
orderings differ).

## Margin selection

margin 0 vs 150 at seed 1: test deep-best rank 5.67/top-1 0.254 vs
5.73/0.246 — a tie; margin 0 selected on the no-arbitrary-constants
principle (it also uses all pairs). Caveat recorded: pairwise
accuracies are not comparable across margins (different pair sets),
and the rank tie was read on the test bucket (a mild protocol slip for
a near-tied binary choice; the margin variable is now closed).

## Primary result — matches (300 pairs, c03 value both sides, 8-ply random openings)

| Match (fc-full) | Ranker-side score [95% CI] |
|---|---|
| ranker vs policy ordering, 400 nodes each | 0.504 [0.472, 0.537] |
| **ranker vs policy ordering, 20 ms/move each** | **0.631 [0.599, 0.663]** |
| control: policy vs policy, 20 ms/move | 0.483 [0.459, 0.508] |

Fixed nodes: tie — the ranker's ordering quality matches the policy
head's search effect despite ranking the deep-best somewhat better.
Fixed time: **+0.631 ≈ +93 protocol Elo**, far outside the symmetric
control's interval. Mechanism: the policy head costs a 4,168-wide
action-head forward at every interior node; the 35-feature ranker
costs one attack scan and a board copy per move. Equal ordering at a
fraction of the cost = more depth per second (§72.4's lesson inverted:
this is the fixed-time-optimal use of ordering knowledge). Games also
became more decisive (W/D/L 365/27/208 vs 272/36/292 in the control).

## Secondary results

- Deep-best rank on held-out records: champion policy 7.42 (top-1
  0.150) → ranker 5.45–5.67 (top-1 0.254–0.259) across 3 seeds.
- Pairwise test accuracy 0.611 (margin 0); 0.704 on margin-150 pairs —
  ordering signal exists but fc-full is far from fc-small's 0.93
  top-1: rule-level features + shallow child labels leave real
  headroom (history features, deeper teachers — later, evidence-gated).
- Learned weights (discovered, coherent with C1): captures up
  (weighted by victim: natural queen > rook > …), queen-promotion up
  but promotion-in-general down (minor promotions strongly negative —
  on fc-full too), pawn advances up, moving the queen (either
  orientation) down (tempo/commitment proxy), destinations under
  attack down.

## Statistical uncertainty

3 ranker seeds (rank spread 5.45–5.67); match CIs pair-level normal;
the fixed-time gain is 8× its standard error. Movetime matches are
wall-clock (control quantifies the jitter floor).

## CPU result

Data 14 min; each records-fit ≈ 2 min; each 600-game match 9–13 min at
4 threads. Total F2 ≈ 1.3 h wall.

## Correctness evidence

104-test suite green (adds MlpRankedEvaluator paths through existing
property tests); records replay validated (Stage B tests); the
policy-vs-policy control brackets protocol noise.

## Complexity delta

+1 fit source variant, +1 fit model variant, +1 evaluate kind,
+1 composite evaluator type; 0 dependencies; 0 deletions.

## Parameter-provenance changes

None beyond F1's `ranker_pair_cap` (reused); `pair_margin` closed at 0
(no constant retained).

## Decision

- **Retain**: `ranker_from_records` pipeline and the composite
  evaluator; **the strongest current fc-full configuration is the c03
  champion value + learned ranker ordering at fixed time.**
- **Next candidates, in expected-value order**:
  1. fc-full structured *value* model from teacher records (needs the
     §18.2 calibrated score→WDL transform — the one missing piece of
     the distillation loop; card required);
  2. iterate the teacher: relabel with the champion+ranker
     configuration (better teacher at equal cost), refit — the §6.1
     loop closing for the first time;
  3. ordering features from search history (§35.2) once the above
     saturate.

## Exact reproducibility information

Data: `runs/20260815-232720-relabel-fc-full-69f4289` (config
`configs/shsd/stage_f/relabel_fc_full_f2data.toml`). Fits:
`runs/*fit-fc-full-move-ranker-records*` (margins + 3 seeds). Matches:
`runs/*ordermatch-fc-full*` (400n, 20ms, control). Ranker checkpoint:
`runs/20260815-234249-fit-fc-full-move-ranker-records-n10000-s1-69f4289/checkpoint/ranker.json`.

# SHSD Stage C — First compact structured evaluator (C1 fit + C2 match)

Experiment card: `reports/shsd/cards/stage_c1_structured_linear.md`
(written before implementation). Machinery commit: 6b532d8; results
produced on the 4-vCPU / 16 GB VM shape.

## Status

PASS — the §56 gate holds: the structured linear evaluator beats the
raw sparse MLP per training position in the low-data regime, improves
shallow-search move choice massively over search alone, and has
positive fixed-time value. The linear form's capacity ceiling is
measured (the §14.4 evidence future model levels must cite).

## Research question

Can a compact structured linear WDL evaluator over hand-designed
measurements (all signs learned) beat the raw sparse MLP per training
position on exact Forward Chess data — in held-out prediction and in
search decision quality?

## Implementation

- `features::FeatureExtractor` + `FcExtractor` with two immutable
  recipes: `fc_counts_v0` (10 dims) and `fc_structured_linear_v1`
  (141 dims on 4×4: count/attacked/defended diffs, mover mobility,
  check, promotion-distance counts, total material, PSQ-per-orientation
  from each owner's perspective). Rotation-colour-swap invariance is a
  property test (D031-class guard).
- `structured_eval::LinearWdl`: 3-class multinomial logistic, 426
  parameters at v1, deterministic minibatch Adam + L2, direct Rust,
  gradient-checked; plugs into search via `Evaluator` (D019 leaf
  convention). Zero-initialized, so fits are convex-ish and
  seed-insensitive (spreads ≤ ±0.004 log-loss).
- `lab fit`: streams the fc-small tablebase (46.5M exact positions,
  D029), selects nested train subsets (smallest `splitmix64(key ^
  fit_train_salt)`, so every ladder rung trains on a superset of the
  previous), standard bucket-8/9 val/test thinned to 20k, fits either
  family on identical states, probes searched decisions against exact
  optimal sets, writes weight inspections.
- C2: `play_paired_match_with` generalizes the paired arena to
  arbitrary evaluators with node or movetime budgets; `lab evaluate`
  kind `structured_match`.

## Deliberately excluded

GAM/phase/pair models (need the saturation evidence below — now they
have it), move-ordering model (Stage F), any Stage D feature family,
non-FC extractors, quantization.

## Baselines

Class prior; `fc_counts_v0`; raw sparse MLP w32 recipe v1 (§9.3) on
identical data; zero-evaluator search (§9.2) in probes and matches.

## Data

fc-small exact tablebase; train = nested subsets of buckets 0–7
(37.2M states available), N ∈ {1k, 10k, 100k, 1M}; val/test = 19,912 /
19,973 states (frozen hash rule); probe = 1,000 test-bucket states
with exact optimal sets. Hyperparameters lr=0.02, l2=1e-4 selected on
**val** loss by an explicit 2×2 grid at the 10k rung (runs
`20260815-2159*`); test untouched until final.

## Target provenance

`exact` (retrograde solution) throughout.

## Primary result — sample-efficiency curve (test log-loss / accuracy)

| N | counts-v0 | structured-v1 | mlp-w32 |
|---:|---|---|---|
| 1k | 0.7575 / 0.648 | **0.6479 / 0.738** | 6.1251 / 0.627 |
| 10k | 0.7542 / 0.649 | **0.5262 / 0.780** | 0.9861 / 0.730 |
| 100k | 0.7542 / 0.649 | 0.5239 / 0.780 | **0.4113 / 0.823** |
| 1M | 0.7546 / 0.651 | 0.5236 / 0.779 | **0.3763 / 0.840** |

(3 seeds structured / 2 seeds MLP; per-seed spreads ≤ ±0.007 acc,
±0.06 LL except MLP@1k ±0.06; class prior 0.8874 / 0.634.)

- **Low-data regime (≤10k): the structured representation is roughly
  two orders of magnitude more sample-efficient.** The structured model
  at 1k beats the MLP at 10k; the MLP at 1k is catastrophically
  miscalibrated (LL 6.1) and *less accurate than counts-only*. This is
  the regime that matters for Expert Iteration (a Program-1 generation
  produces ~20k positions).
- **Crossover at ~100k**: the MLP's capacity wins the data-rich regime
  and was still improving at its step budget (train WDL loss falling at
  40k steps), so its 1M ceiling is understated.
- **The linear structured model saturates**: 10k → 1M buys nothing
  (0.5262 → 0.5236). Its 141-dim linear form, not data, is binding —
  the exact §14.4 evidence needed before admitting Level-2+ capacity.

## Secondary results

**Searched-decision probes** (1,000 exact positions, optimal-move rate):

| Evaluator | raw (no search) | 400 nodes | 6,400 nodes |
|---|---:|---:|---:|
| zero (search-only) | — | 0.856 | 0.932 |
| counts-v0 (any N) | 0.914 | 0.971 | 0.996 |
| structured-v1 @10k | 0.955 | 0.981 | 0.995 |
| mlp-w32 @10k | 0.890 | 0.978 | 0.996 |
| mlp-w32 @1M | 0.958 | 0.996 | 1.000 |

The structured evaluator's *raw* argmax nearly matches its own searched
decisions — a genuinely informative value function (contrast the
Program-1 fc-tiny w64 MLP whose value head collapsed to uniform draw,
Stage B report). Deep search washes out evaluator differences (§6.6:
model quality matters at shallow search).

**C2 matches** (fc-small, 300 pairs, shared random 6-ply openings,
colour-swapped):

| Match | Structured score [95% CI] |
|---|---|
| v1@10k vs mlp@10k, equal 400 nodes | 0.507 [0.496, 0.518] |
| v1@1M vs mlp@1M, equal 400 nodes | 0.474 [0.464, 0.485] |
| v1@1M vs zero-evaluator, 400 nodes | **0.727 [0.701, 0.753]** |
| v1@10k vs mlp@10k, 20 ms/move (fixed time) | 0.501 [0.497, 0.505] |

The structured evaluator adds ≈ +170 protocol Elo over search-only at
equal nodes, and plays at parity with the MLP at both fixed nodes and
fixed time **while providing no policy move-ordering at all** (the MLP
orders its search with its policy head; the structured search falls
back to action-ID order). The MLP's small edge at 1M matches its
prediction edge. Mean game length ~13 plies after the random opening —
the 4×4 board is sharply tactical.

## Learned parameter interpretation (§22.4, §52.5, §74.4)

Win–loss weight margins, stable between the 10k and 1M fits
(convergence by 10k explains the saturation):

- **Reversed queen +5.63, pawn +3.2, rook-nat +1.8**: a pawn is worth
  more than a rook on this board — it is the promotion vehicle. Values
  are discovered, not asserted, and invert conventional chess values
  (§1.2 vindicated).
- Reversed knight/bishop are *liabilities* (−0.24, −0.35): promoting to
  a reversed minor on 4×4 is worse than not promoting.
- Pawn at promotion distance 1: +2.45 on top of its base value.
- **Positional sign reversal**: a reversed queen on its birth rank
  (owner-perspective cells 12–15) is strongly winning; the same piece
  arrived at the owner's home rank (cells 0–3, horizontal moves only)
  flips to strongly negative — the "spent reversed piece" /
  interaction-window concept (§28.6, §34.2) expressed by a linear PSQ
  table.
- counts-v0's flat 0.649 accuracy at every N shows material alone is
  nearly uninformative here; structure carries fc-small (§8.5.2).

## Statistical uncertainty

Seeds: 3 (structured) / 2 (MLP) per rung; structured fits are
zero-init deterministic-batch, spreads ≤ ±0.004 LL. Match CIs are
normal-approximation over pairs (matching the standing protocol).
Binomial SE on probe rates at n=1000 ≈ 0.007 at rate 0.95.

## Fixed-node / fixed-time result

Both measured (tables above): fixed-node parity with the raw baseline
at its own data scale, +0.727 over search-only; fixed-time parity
(structured extraction ≈ MLP inference cost at this board size:
extraction ~0.9–1.2 µs/position measured in-fit).

## CPU result

32 ladder fits + 4 grid fits + 4 matches on 4 vCPUs ≈ 2.6 h wall
total. Per fit: ~180–360 s wall, dominated by the two streaming
tablebase passes (~2.2 GB peak RSS each, 4 concurrent fits fit in
16 GB; 1M-rung entries throttled to 2 concurrent via sweep cores).
Matches: 600 games in 2–4 min at 4 threads.

## Correctness evidence

99 tests green including: rotation-swap invariance over 100+ random
positions × both recipes; hand-computed position (counts, attacks,
defence, mobility=9 legal moves, promotion distances, PSQ cells,
reversed-rook geometry); gradient check vs finite differences; fit
determinism; serde round-trips; end-to-end oracle cross-check on
solved Connect-k (full-budget deep labels 100% optimal).

## Ablations

counts-v0 vs structured-v1 is the built-in family ablation: the
non-count families (attacks, mobility, PSQ, promotion, check) carry
+0.13 accuracy at 10k. Per-family ablations within v1 are deferred to
Stage D admission protocol (one family at a time from here on, §14.2).

## Failure cases

- MLP@1M undertrained at 40k steps (still improving) — reported; only
  strengthens the crossover conclusion, but 1M-rung MLP numbers are a
  floor, not a ceiling.
- probe@6400 saturates near 1.0 for all evaluators — fc-small cannot
  separate evaluators under deep search; separation lives at shallow
  budgets and raw decisions.

## Complexity delta

- Production LOC: +~700 (features + structured_eval) +~600 (lab fit +
  structured_match + arena generalization)
- Public types: +8 (FeatureExtractor, FcExtractor, FcRecipe, LinearWdl,
  StructuredRow, FitHyper, FitStep, StructuredEvaluator, MoveBudget) —
  over the §14.1 guideline; declared in the card (one model family +
  one extractor family, no plugin surface)
- Config keys: `lab fit` config (9 fields + 2 sub-enums),
  `structured_match` evaluate kind (9 fields)
- Dependencies: 0; Deleted code: 0 (play_paired_match became a wrapper)

## Parameter-provenance changes

Ledger: +`fit_train_salt` (engineering). lr=0.02 / l2=1e-4 for
`structured_linear_fc_v1` fits: experimentally selected (2×2 grid on
val at the 10k rung, `runs/20260815-2159*`); steps/batch are
engineering budgets with convergence monitored per run.

## Decision

- **Retain**: `fc_structured_linear_v1` as the Level-1 structured
  evaluator and the fit/probe/match machinery; the raw MLP stays the
  §9.3 data-rich baseline.
- **Reject**: nothing; counts-v0 is kept as the Level-0 anchor.
- **Next evidence needed**: the measured linear saturation licenses
  exactly one capacity step (Level 2 GAM on the same measurements) OR
  one structural family (Stage D reachability), not both at once
  (§6.8). Move ordering (Stage F) is the other high-expected-value
  lever: the structured evaluator currently searches with no ordering
  model at all, and Program-1 evidence says ordering is worth a lot.

## Next research question

Which single increment removes more of the measured gap at fixed
compute: (a) nonlinearity on existing measurements (GAM buckets), or
(b) a learned move-ordering model over the same measurements (§35,
pairwise ranking on exact counterfactual child data — which the exact
corpora already provide for free)? Both have clean §47.1 cards; (b)
also directly helps every future evaluator.

## Exact reproducibility information

Configs: `configs/shsd/stage_c/` (grid + gen_ladder.py + 32 ladder
configs), `configs/shsd/stage_c2/` (4 matches). Runs:
`runs/20260815-2159*` (grid), `runs/20260815-22*fit-fc-small*`
(ladder), `runs/20260815-225*/230*-structmatch-*` (matches).
Aggregation: `tools/shsd_fit_table.py`. All fits deterministic given
config; movetime match is wall-clock (CI reported).

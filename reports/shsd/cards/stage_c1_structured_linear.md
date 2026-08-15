# Experiment: C1 — structured linear evaluator vs raw sparse MLP (sample efficiency, exact Forward Chess)

## Question

Does a compact structured **linear** WDL evaluator over hand-designed
measurements — counts, piece-square/orientation, mobility, immediate
attacks/defences, promotion distance, check; every sign and magnitude
learned (§6.2) — beat the raw sparse MLP per training position on exact
Forward Chess data, in held-out prediction and in search decision
quality?

## Hypothesis

The §1.2 thesis predicts: at small N (1k–100k training positions) the
structured linear model clearly beats the raw MLP on held-out log-loss
and on searched decisions; at N = 1M the MLP may close the gap
(capacity). A count-only Level-0 model (§17.1) sits far below both.

## Smallest implementation

- `features` module: `FeatureExtractor` trait + one Forward Chess
  extractor with two named, immutable recipes: `fc_counts_v0` (Level 0:
  piece-type/orientation count differences + side-to-move constants)
  and `fc_structured_linear_v1` (adds PSQ differences, mover mobility,
  in-check, attacked/defended counts by victim type, promotion-distance
  counts, total-material phase observable).
- `structured_eval` module: multinomial-logistic WDL model (3-class
  softmax, L2, deterministic seeded Adam), direct Rust (§11.4), serde
  checkpoint, `search::Evaluator` impl (leaf = round(1000·(P(win)−P(loss))),
  the D019 convention).
- `lab fit` command: streams the fc-small tablebase, builds
  hash-subsampled train sets and the standard thinned val/test buckets,
  fits either model family (structured | raw MLP recipe v1) on identical
  data, and probes searched decisions against the exact solution.

## Baselines

Class-prior log-loss; `fc_counts_v0`; raw sparse MLP w32 (recipe v1,
§9.3); zero-evaluator search row in the probe (§9.2).

## Primary metric

Held-out **test WDL log-loss vs N** (sample-efficiency curve) on
fc-small, N ∈ {1k, 10k, 100k, 1M}.

## Secondary metrics

Test accuracy; probe optimal-decision rate at raw (argmax over child
evaluations), 400, and 6,400 nodes on ≥1,000 sampled test-bucket
states; feature-extraction ns/position (§68.2); fit wall/CPU cost;
calibration (reliability by bucket) if selection is ambiguous.

## Fixed resources

Single-threaded fits, sweep-level parallelism on 8 vCPUs; fc-tiny
in-process solve validates the pipeline before any fc-small run.

## Independent variables

Model family (counts-v0 | structured-v1 | mlp-w32); N; seed.

## Controlled variables

Data subsets (hash-selected, identical across models and seeds),
val/test buckets (frozen convention), probe states and budgets, TT
size, recipe constants.

## Seeds

3 for structured fits (cheap), 2 for MLP fits (expensive), on every N.

## Predicted outcomes

### If hypothesis is supported

Structured linear wins at low N on both prediction and decisions →
Stage C2: integrate as a search evaluator, fixed-node and fixed-time
match evaluation vs the MLP champion (the remaining §56 gate items).

### If hypothesis is rejected

MLP ≥ structured at every N: inspect target quality, symmetry,
extraction, and optimization per §56 ("do not immediately add more
feature families"); test on Breakthrough exact data to separate
game-specific from generic failure.

### If result is ambiguous

Structured wins prediction but not decisions (or vice versa): add w64
MLP arm and a third seed; examine calibration and per-WDL-class errors
before any model change.

## Correctness risks

Perspective/rotation bugs in extraction (D031 class) — guarded by a
180°-rotation-swap invariance property test; probe join errors —
guarded by the full key→WDL map from the checksummed tablebase; Adam
determinism — gradient check + fixed-seed repeat test.

## Performance risks

MLP arm at N=1M needs tens of minutes single-threaded (sweep absorbs
it); tablebase streaming holds the key→WDL map (~1.5 GB) plus ≤1M
feature rows in memory — within 32 GB.

## Complexity budget

Two modules (features, structured_eval), one command + one config type,
two named recipes, zero new dependencies. New fit hyperparameters (lr,
L2, steps, batch) selected by a small explicit grid on the 10k rung
(val loss), recorded in the ledger.

## Removal criterion

If the structured representation is rejected under the gate rules, the
extractor shrinks to whichever families later pass individual admission
(§14.2); the fit machinery remains (it is Stage C/E infrastructure).

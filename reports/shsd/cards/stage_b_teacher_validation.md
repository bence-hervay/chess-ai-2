# Experiment: Stage B teacher-instrumentation validation (fc-tiny oracle check)

## Question

Does the new deep-search relabelling instrumentation produce trustworthy
teacher labels — specifically, do deeper internal searches agree
monotonically more with exact optimal play, and can one run's records
separately expose model error, search error, and move-ordering error?

## Hypothesis

On Forward Chess `tiny` (exactly solved, 84k positions):

1. decision accuracy of the search label (best action ∈ exact optimal
   set) increases monotonically with node budget for a fixed evaluator;
2. at equal budget, the trained Phase-8 w64 checkpoint evaluator is more
   accurate than the zero evaluator (model error visible);
3. the rank of the deep-best action in the evaluator's policy ordering
   is better (lower) for the trained evaluator (ordering error visible);
4. shallow→deep value residuals shrink as the shallow budget grows
   (search error visible).

## Smallest implementation

- `SearchResult` gains per-completed-depth `(depth, best move, value)`
  (best-move-stability evidence, §18.2);
- `src/data.rs`: typed teacher records with provenance (§19.9), path
  replay, deterministic trajectory sampling, parallel labelling;
- `lab relabel` command with in-process retrograde oracle join.

No search-behavior changes, no features, no training.

## Baselines

- ZeroEvaluator (search-only baseline §9.2);
- Phase-8 fc-tiny w64 selfplay checkpoint (raw-model baseline §9.3,
  `runs/20260814-201220-selfplay-fc-tiny-w64-g400-e10-s1-c9c1ef5/checkpoint`).

## Primary metric

Optimal-decision rate of the search label vs node budget
(monotonicity), per evaluator, on oracle-joined positions.

## Secondary metrics

Shallow→deep value residuals per budget; shallow/deep best-move
agreement; per-depth best-move stability; counterfactual child top-move
optimal rate; policy rank of the deep best action; positions/s, CPU
utilization, wall time.

## Fixed resources

8 threads, fc-tiny, ≈2,000 labelled positions per run, deep budget
6,400 nodes, shallow budgets {50, 200, 800}.

## Independent variables

Evaluator (zero | w64 checkpoint); node budget (via shallow ladder +
deep label); seed.

## Controlled variables

Game, trajectory policy (random), sampling rule, TT size (2^16), move
ordering (Natural), thread count (results must be thread-independent).

## Seeds

1, 2, 3 (cheap exact-game experiment → 3 seeds per §47.4).

## Predicted outcomes

### If hypothesis is supported

Stage B gate passes; instrumentation is trusted for Stage C fitting.

### If hypothesis is rejected

A non-monotone budget→accuracy curve or an inverted model comparison
means a search, replay, or labelling bug; stop and debug before any
feature work (§72.5 class failure).

### If result is ambiguous

Increase positions to 10k and add seeds; if the ambiguity is only in
low-budget regions, report and proceed (deep labels are what Stage C
consumes).

## Correctness risks

Path replay divergence (round-trip tested); oracle join misses from
history-dependent state components (join rate reported; fifty-move /
repetition caveat documented in D028/D029); duplicate positions
(deduped by key, visit weight kept, §20.4).

## Performance risks

Labelling is embarrassingly parallel; per-position cost bounded by
deep + shallow + children budgets. Report positions/s and utilization.

## Complexity budget

One new module (`data`), one new `lab` subcommand + one config type
(with two small sub-enums), zero new dependencies.

## Removal criterion

Instrumentation, not a strategy component: retained while the SHSD
program runs. The record schema is versioned; failed schema decisions
are migrated, not accreted.

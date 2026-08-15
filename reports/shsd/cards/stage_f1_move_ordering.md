# Experiment: F1 — learned move ordering from exact data (Forward Chess)

## Question

Does a compact learned move-ordering model reduce the nodes the
structured-evaluator search needs at fixed depth, without hurting (and
ideally improving) fixed-node and fixed-time decision quality?

## Why now (stage reordering justification, §3.2)

Stage C left the structured evaluator searching in raw action-ID order
(its `policy_scores` returns nothing), while the raw-MLP baseline
enjoys policy ordering — and still only tied. §59 says ordering may
out-value another static feature family; decision-tree Case 3 says
"train a separate move-ordering model" when prediction improves but
search efficiency hasn't. Exact corpora provide perfect ranking
targets free of teacher noise (§18.5: exact optimal vs non-optimal),
and Stage B measured huge ordering headroom at the target game
(c03's deep-best mean policy rank 7.5 on fc-full). Ordering
improvements compound into every later experiment.

## Hypothesis

A linear pairwise ranker over cheap rule-level + tactical state-action
features (§35.2), trained on exact optimal-vs-non-optimal pairs,
(1) ranks a game-theoretically optimal move first far more often than
action-ID order, (2) cuts alpha–beta nodes to fixed depth materially
(≥25%), and (3) does not reduce fixed-node decision quality.

## Smallest implementation

- `FcMoveFeatures` in `features::forward_chess`: sparse per-move
  features — moving piece×orientation, captured piece×orientation,
  promotion (+choice), direct-check-after-move, forward displacement,
  destination attacked/defended (pre-move occupancy), source attacked,
  destination relative rank. No history/TT features (search-state
  ordering already exists via TT move and stays).
- `structured_eval::MoveRanker`: linear scorer, pairwise logistic
  ranking loss, deterministic Adam (reusing the fit machinery style).
- `StructuredEvaluator` gains an optional ranker that implements
  `policy_scores` — zero search changes (the Evaluator hook already
  exists and TT-move-first still applies on top).
- `lab fit` gains a `ranker` model kind reusing the same exact stream,
  splits, and probes.

## Baselines

Action-ID order (current structured search); the raw-MLP policy head's
ordering (measured in C1 probes); TT-move-first is common to all arms.

## Primary metric

Nodes to complete fixed depths d = 4 and d = 6 (same leaf evaluator,
same positions), ranker ordering vs action-ID ordering: median ratio
over ≥500 probe states.

## Secondary metrics

Held-out pairwise ranking accuracy; rank of the first exact-optimal
move (mean, top-1 rate); fixed-node probe optimal-decision rate @400;
paired match ordered-vs-unordered at 400 nodes and at 20 ms/move;
per-move feature cost (ns).

## Fixed resources

fc-small exact data (train N = 100k positions, the saturation-free
rung for ranking), standard val/test buckets, 4 vCPUs.

## Independent variables

Ordering model (none | ranker); match budget kind.

## Controlled variables

Leaf evaluator (structured-v1 @1M checkpoint), TT size, probe states,
data splits, lr/l2 from the C1 grid unless val loss says otherwise (one
2-point lr check allowed, val only).

## Seeds

3 for ranker fits; matches at seed 1 (300 pairs).

## Predicted outcomes

### If hypothesis is supported

Adopt the ranker as the production ordering for structured search;
next step is transferring it to fc-full via teacher-record targets.

### If hypothesis is rejected

If ranking accuracy is high but nodes don't drop: inspect cutoff
structure (the §35.6 warning — predictive gain without search gain is
insufficient); check TT-move interaction; do not stack more features.

### If result is ambiguous

Nodes drop at d=6 but not d=4 (or vice versa): report the depth
dependence and evaluate at the match level before deciding.

## Correctness risks

Move-feature perspective bugs (rotation-swap invariance test extended
to move features); ordering must never change search *values* (§69.4:
ordering changes node counts only — property-tested).

## Performance risks

Per-move feature extraction at every node could eat the node savings —
measured; the check-after-move feature is one attacks() call per move.

## Complexity budget

One feature family (state-action), one model mechanism (pairwise
ranker), one config variant, zero dependencies, ≤3 public types.

## Removal criterion

Rejected if fixed-time match strength does not improve (§35.6): the
ranker types and fit path are deleted; the report stays.

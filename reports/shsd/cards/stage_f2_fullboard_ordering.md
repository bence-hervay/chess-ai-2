# Experiment: F2 — fc-full move ordering distilled from internal search

## Question

Does a move-ordering model trained on internal-search counterfactual
child labels (no oracle) beat the champion MLP's own policy-head
ordering at full Forward Chess — in deep-best ranking, and in
fixed-node / fixed-time match play?

## Hypothesis

Stage B measured the c03 champion's policy ranking the deep search's
best move at mean 7.5 (top-1 0.16) on fc-full. A 35-parameter linear
ranker over rule-level move features, trained on child-value pairs
from `lab relabel` records (§19.5 → §35), ranks the deep-best move
materially better (mean rank < 3, top-1 > 0.4) and wins a fixed-time
match against the same value model using policy-head ordering.

## Smallest implementation

- `lab relabel` run generates the data (existing command, zero code):
  c03-champion selfplay trajectories, deep 1,600, children at 400.
- `lab fit` gains `source = records` (pairs built directly from stored
  child labels with a value-difference margin; positions replayed from
  action paths; splits by the standard key-bucket rule). Exact-source
  logic untouched.
- `MlpRankedEvaluator`: champion MLP value + ranker ordering (the
  composite the match needs).
- `lab evaluate` kind `ordered_mlp_match`: MLP value with ranker-A vs
  the same MLP value with its own policy ordering (or ranker-B).

## Baselines

The champion's policy-head ordering (records store its deep-best rank
per position, measured at relabel time); generation order as context.

## Primary metric

Fixed-time match (20 ms/move, 300 pairs): MLP+ranker vs MLP+policy
ordering, same value model both sides. Retain per §35.6 only if the
score's LCB > 0.5 or clearly ties while node metrics improve.

## Secondary metrics

Mean rank / top-1 of the deep-best action under the ranker vs the
stored policy rank (held-out records); held-out pairwise accuracy;
fixed-node match at 400 nodes; cost of ordering per node (movetime
match captures it end-to-end).

## Fixed resources

~10k distinct fc-full positions (≈250 selfplay games at 400 nodes,
sample 1-in-2), children at 400 nodes ≈ 140M search nodes ≈ 15–20 min
at 4 threads. Teacher = c03 champion (§20.5 identity recorded per
record).

## Independent variables

Ordering source (ranker | policy head); pair margin {0, 150} on the
10k data (val pairwise + teacher-rank selection — noisy-label
robustness check).

## Controlled variables

Value model (c03 both sides), budgets, openings, seeds, TT size.

## Seeds

Ranker fits at 3 seeds; matches at seed 1 (300 pairs).

## Predicted outcomes

### If hypothesis is supported

Adopt learned ordering for fc-full structured search going forward;
next: fc-full structured *value* from teacher WDL labels (needs the
§18.2 calibrated transform — its own card).

### If hypothesis is rejected

If ranking improves but matches don't: the policy head's ordering may
encode search-history information the rule-level features lack —
analyze which moves it orders better (§35.6 discipline); consider
history/continuation features before more tactical ones.

### If result is ambiguous

Tie at fixed time with clear ranking gains: retain for the structured
track only (the MLP keeps its policy head), and re-test after the
fc-full structured value model exists.

## Correctness risks

Label noise: 400-node child values on fc-full are shallow; the margin
variable measures sensitivity. Replay drift: records replay on the
same code revision (tested in Stage B).

## Performance risks

Per-node ordering cost at fc-full (64-cell board copies per move) —
the fixed-time match is the honest end-to-end check.

## Complexity budget

One config source variant, one composite evaluator type, one evaluate
kind; 0 dependencies.

## Removal criterion

§35.6: no fixed-time gain (against the policy-ordering baseline) and
no node reduction → delete the records-source pair path and the
composite; keep the report.

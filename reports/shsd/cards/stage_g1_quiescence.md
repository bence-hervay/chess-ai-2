# Experiment: G1 — capture/promotion quiescence (§37.3 rung 2)

## Question

Does resolving captures and promotions at the leaves stabilize search
values — fewer best-move reversals across depths, calmer teacher
labels — and improve fixed-node and fixed-time strength?

## Observed failure mode (§14.3 requirement 1)

Measured, twice: fc-full deep labels (1,600–6,400 nodes) change their
best move during the final completed iteration in 63–67% of sampled
positions (Stage B, F2 data runs); fc-small match games average ~13
plies of sharp tactics. Full-width leaves evaluate positions
mid-exchange, so every depth step re-litigates hanging material — the
classic horizon effect, now quantified in this engine.

## Hypothesis

Capture+promotion quiescence at depth 0 (stand-pat + tactical-only
continuation, no checks yet) raises deep-label last-iteration
stability materially (0.37 → ≥0.55), does not hurt fixed-node decision
quality on exact boards, and wins a fixed-time match against the
non-quiescent configuration at fc-full.

## Smallest implementation

- `Searcher` gains a quiescence flag (constructor argument; the OFF
  mode is the §14.3 correctness-preserving reference and stays the
  default everywhere not under test). At depth 0: stand-pat eval;
  beta-cutoff; search captures (incl. en passant) and promotions only,
  ordered by the evaluator's policy scores when provided; bounded by a
  ledgered safety depth. If the mover is in check at a quiescence
  node, return the static eval (rung-2 limitation, documented; checks
  and evasions are rung 3, separate evidence).
- Quiescence nodes count against the node budget and are separately
  counted (§68.1).
- `quiescence: bool` required field on `relabel` and
  `ordered_mlp_match` configs (the two instruments this card uses).

## Baselines

Current full-width search (quiescence off) everywhere.

## Primary metric

Fixed-time match at fc-full (20 ms/move, 300 pairs): champion value +
F2 ranker, quiescence ON vs OFF.

## Secondary metrics

Deep-label last-iteration stability and mean best-move changes on a
fresh fc-full relabel sample (same seed/budgets as F2 data, quiescence
on); shallow(=deep-budget-matched) agreement; fixed-node match at 400
nodes; fc-small exact searched-decision rate @400 nodes with
quiescence on vs off (fit probes); quiescence node share; nodes/s.

## Fixed resources

fc-full relabel ~8k positions (≈15 min); two 600-game matches; one
fc-small fit-probe pair. 4 vCPUs.

## Independent variables

Quiescence on/off. Nothing else changes (§6.8).

## Controlled variables

Value model (c03), ranker (F2), budgets, openings, seeds, TT size.

## Seeds

Matches at seed 1 (300 pairs); relabel at seed 11 (same as F2 data for
paired comparison of stability metrics on identical positions).

## Predicted outcomes

### If hypothesis is supported

Quiescence ON becomes the production search mode for Forward Chess;
teacher pipelines regenerate labels with it (calmer targets for the
value-distillation step that follows).

### If hypothesis is rejected

Fixed-time loss despite stability gains → quiescence cost exceeds its
value at these budgets; keep OFF, record the cost curve, revisit after
cheaper evaluators. Stability unchanged → the instability is not
capture-driven; investigate check-driven horizon (rung 3) before
adopting anything.

### If result is ambiguous

Stability up, match a tie: retain for teacher generation only
(label-quality tool, §57-style "retained for training labels"), not
for play.

## Correctness risks

Quiescence must terminate (captures strictly consume material;
promotions bounded; ledgered depth cap as defense); the OFF mode must
be bit-identical to today's search (regression-tested); TT entries
from quiescent and full-width searches must not mix incompatibly
(quiescence nodes are not TT-stored in this rung — simplest sound
choice).

## Performance risks

Quiescence explosion on loaded fc-full positions — bounded by the
depth cap and measured via the quiescence-node counter.

## Complexity budget

One search mechanism with a reference mode, one constructor argument,
two config fields, one ledger entry (quiescence depth cap), ≤1 new
public type. No new dependencies.

## Removal criterion

§37.5: no horizon-error reduction, or fixed-node gains that fixed-time
costs erase → delete the quiescence path (keep the report).

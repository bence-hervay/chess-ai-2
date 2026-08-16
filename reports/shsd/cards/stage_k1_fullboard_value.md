# Experiment: K1 — calibrated value distillation at full Forward Chess

## Question

Can the structured linear evaluator learn full-board Forward Chess
value from quiescent deep-search labels (no oracle), and what does it
say about fc-full strategy (§8.5: piece values, material vs structure,
reversed-piece geometry)? Baseline question: how close does a
~700-parameter interpretable model get to the 378k-parameter champion
MLP's value head in actual stack play?

## The §18.2 calibration design

Deep search values are on the champion's leaf scale
(`round(1000·(P(win)−P(loss)))` backed through minimax) plus
mate-proven scores. To turn them into WDL training targets without
inventing constants:

1. `relabel` records gain the **terminal outcome of the trajectory
   that sampled the position** (§18.3 — the trajectory already plays
   to terminal; one Monte-Carlo outcome draw per record; ε-exploration
   noise documented as a caveat).
2. A tiny multinomial logistic **calibration model** maps
   `(deep value, mate flags)` → WDL probabilities, fitted on those
   outcomes (train bucket only).
3. The structured model trains on the **calibrated soft targets** of
   the deep values (soft cross-entropy); mate-proven labels collapse
   to (near-)certain classes through the same calibration fit.

Every mapping is fitted; no hand-chosen thresholds (§6.3).

## Smallest implementation

- `TeacherRecord.trajectory_outcome: Option<Wdl>` (serde-default None
  so all existing record files still parse).
- Soft-target generalization of the linear-WDL fitter (the hard-label
  path becomes a one-hot wrapper).
- `lab fit`: `structured_from_records` model kind — calibration fit +
  soft-target structured fit + weight inspection; held-out metrics
  against both the calibrated soft targets and the raw outcomes.
- Match arm: `structured_match` already accepts a structured
  checkpoint + ranker; quiescence flags added to it (same
  serde-default pattern).

## Baselines

Champion MLP value head (the incumbent, in the identical
ranker+quiescence stack); class prior; counts-v0 recipe on the same
targets (Level-0 anchor).

## Primary metric

Fixed-time match (20 ms/move, 300 pairs): structured-value stack vs
champion-value stack. The realistic hypothesis is NOT a win — C1
showed the MLP wins the data-rich regime — but the gap in Elo is the
headline number: how much play does interpretability cost at fc-full?

## Secondary metrics

Held-out soft-target log-loss/accuracy vs the counts-v0 and prior
floors; calibration reliability (predicted vs empirical outcome rates
by value bucket); **learned fc-full piece values and PSQ curves**
(§8.5, §74.4) including reversed-piece geometry at 8×8; extraction
ns/position at 8×8 (fixed-time viability data).

## Fixed resources

~15k positions from stack-teacher relabel (quiescence on, deep 1,600,
children off — value only, cheaper per position); fits are minutes;
one 600-game match.

## Independent variables

Value model (structured-v1 | counts-v0 | champion MLP); seed (3 for
fits).

## Controlled variables

Ranker + quiescence in all match arms; budgets; teacher identity;
splits by record key (standard buckets).

## Predicted outcomes

### If the structured stack lands within ~100 Elo of the champion

A 700-parameter model nearly matching a 378k-parameter network in
play would make the structured track the preferred substrate for
from-scratch loops (sample efficiency + interpretability); next step
would be Stage D families to close the rest.

### If the gap is large (≥250 Elo)

The linear form is far from sufficient at 8×8 — expected direction:
GAM/Level-2 on the measured residuals, or accept the MLP as the value
substrate and focus structured work on ordering/search-control.

### If calibration is poor (reliability off by ≥0.15)

Trajectory-outcome noise dominates — switch the calibration source to
greedy playouts (no exploration) before judging the value model.

## Correctness risks

Outcome perspective bugs (outcome recorded game-global, converted to
side-to-move at the record's position — tested); soft-target gradient
(finite-difference check extended).

## Complexity budget

One record field (optional), one fit model kind, one soft-target
fitter generalization, quiescence flags on `structured_match`;
0 dependencies.

## Removal criterion

If the structured value is not retained for any role (play, teaching,
or analysis) after Stage D, the records-value fit path is deleted; the
strategy analysis and report remain.

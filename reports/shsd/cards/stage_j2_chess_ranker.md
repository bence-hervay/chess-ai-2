# Experiment: J2 — learned chess move ordering, then the anchor

## Question

Does the F2 mechanism transfer to standard chess: a linear pairwise
ranker over rule-level chess move features, distilled from internal
teacher records, (a) beating the policy head's ordering at fixed time,
(b) supplying the cheap capture ordering that J1 showed quiescence
needs, and (c) finally moving the Phase-7 Stockfish anchor?

## Hypothesis

The chess policy head costs a 4,168-wide forward per interior node
(same pathology F2 measured at fc-full); a ~25-feature ranker matches
its ordering at a fraction of the cost → fixed-time self-match win.
With ranker-ordered quiescence, the J1 capture-tree explosion is tamed
enough that quiescence turns positive at chess, and the combined stack
beats the 10.33% anchor baseline.

## Smallest implementation

- `MoveFeatures<G>` trait (two implementations now exist — §10.3
  satisfied); `ChessMoveFeatures` (moving/captured piece, promotion
  choice, gives-check via cozy `checkers`, destination/source attack
  facts, relative rank, castle, en passant).
- `fit` records-ranker path generalized over the game via the trait.
- UCI options `Ranker` (path) and `Quiescence` (bool) +
  `ckptx:<dir>,<ranker|->,<qs|->` spec in `run_match.sh` (space-free
  fastchess `option.*` tokens).
- Teacher records: `lab relabel` on chess with the Phase-7 w64
  checkpoint (250 selfplay games @400 nodes, deep 1,600, children 400
  — the F2 protocol).

## Baselines

Policy-head ordering (the engine as measured in Phase 7); J1's halted
quiescence configuration as the cautionary reference.

## Primary metric

Self-match at 0.3 s/move (150 pairs): ranker-ordered engine vs
policy-ordered engine, quiescence off — the pure ordering effect. Then
the winning configuration vs Stockfish 17.1 @ 20 nodes (300 games,
frozen protocol) vs the 10.33% baseline.

## Secondary metrics

Deep-best rank on held-out records (ranker vs policy); depth reached
at 300 ms with ranker-ordered quiescence (J1's collapsed depth-1 as
the reference); termination causes (D027).

## Seeds / resources

Ranker fits 3 seeds; matches protocol seed 42; ~2.5 h total.

## Predicted outcomes

Supported → the anchor improves; the §51.6 gate item advances; chess
quiescence decision made on measurement. Rejected (ranker loses to
policy head at chess) → the policy head's chess ordering knowledge
exceeds rule-level features; record and stop at ordering (Case 10).
Ambiguous → extend games before concluding.

## Complexity budget

One trait (two impls), one extractor, one generalized fit path, two
UCI options, one script spec. No dependencies.

## Removal criterion

§35.6 unchanged: no fixed-time gain → the chess ranker path is
deleted; the report stays.

# SHSD J1 — quiescence at standard chess (negative for now, root-caused)

Experiment card: `reports/shsd/cards/stage_j1_chess_quiescence_anchor.md`.

## Status

NEGATIVE at this configuration — and fully root-caused. Rung-2
quiescence, a ≈+115 Elo win at fc-full, *loses* at standard chess at
0.3 s/move with the Phase-7 engine: the halted self-match stood at
0.431 for the quiescent side after 72 games. The §73 Case-8 pattern:
a mechanism's value is game-conditional, and the conditions are now
measured.

## What was measured

1. **Control**: from the starting position (no captures possible),
   quiescence on/off give identical depth (4/4) — plumbing sound.
2. **Cost source 1 — ordering price**: qsearch ordered tactical moves
   via `policy_scores`; for the chess MLP that is a 4,168-wide action
   head per quiescence node. Nodes/s 12k vs 39k without quiescence.
   (At fc-full G1 the ordering came from the 35-weight ranker — cheap —
   which masked this.)
3. **Cost source 2 — movegen price**: each quiescence node generated
   all ~35 legal moves to find ~2 captures. Fixed properly:
   `Game::tactical_moves` with a targeted cozy-chess implementation
   (move bitboards masked to enemy occupancy + en-passant + promotion
   rank), differential-tested against the filter on 300 positions.
   Nodes/s with both fixes: 293k (24× the broken configuration).
4. **The irreducible problem**: even at 293k nodes/s, the engine
   completes only depth 1 in 300 ms (vs depth 3–4 without quiescence).
   Open-position capture trees branch ~4 ways for up to ~8 plies;
   without *ordered* beta cutoffs the tree is ~4⁸. Real engines tame
   this with capture ordering (MVV-LVA) and pruning; we currently have
   only stand-pat cutoffs from a weak, miscalibrated eval (the Phase-7
   chess model's raw eval was never good — its strength was search).

## The cross-game law this measures (§8.4.7)

Quiescence value ≈ f(capture density, stand-pat quality, ordering
cost). Forward Chess: directional geometry keeps capture density low,
the c03 eval is well-calibrated, and a learned ranker orders cheaply →
+115 Elo. Chess: high density, weak eval, no cheap ordering → ~2–3
plies of full-width depth lost, net negative. Neither result
transfers to the other game — exactly the §16/§73 classification
discipline.

## Decision

- Chess quiescence **deferred**, not deleted: the prerequisite is a
  cheap *learned* capture ordering — the F2 ranker mechanism at chess
  (learned MVV-LVA, no hardcoded piece values), which is J2.
- The targeted `tactical_moves` generator is retained (correct,
  tested, needed regardless).
- qsearch keeps evaluator-hook ordering (the G1-validated FC
  configuration; FC behavior unchanged).
- The Phase-7 anchor re-measurement is postponed until the chess stack
  (ranker, then quiescence) is validated by self-matches.

## Reproducibility

Halted self-match log (72 games):
`runs/20260816-074253-match-j1-qs-selfmatch-712ad8e/`. Depth
measurements in this report were taken with
`go movetime 300` on the Italian-opening position; commands inline in
the session log. Differential test:
`games::chess::tests::tactical_moves_match_the_filtered_default`.

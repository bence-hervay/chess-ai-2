# SHSD Stage G1 — Capture/promotion quiescence (§37.3 rung 2)

Experiment card: `reports/shsd/cards/stage_g1_quiescence.md`. Observed
failure mode driving this stage: fc-full deep labels change their best
move in 63–67% of final iterations; fc-small games are ~13-ply
tactical brawls (Stage B, F2 data).

## Status

PASS — the largest single strength gain of the program so far:
≈ +115 protocol Elo at both fixed nodes and fixed time on fc-full, and
the combined SHSD stack (ranker + quiescence) beats the Program-1
champion configuration by ≈ +186 Elo with the identical value network.

## Research question

Does resolving captures and promotions at the horizon stabilize search
values and improve fixed-node and fixed-time strength?

## Implementation

- `Game::is_tactical` (rule-level fact, default `false`): captures,
  en passant, and promotions for Forward Chess and chess; captures for
  Breakthrough; nothing for Connect-k/Othello — quiescence is a
  provable no-op where no quiet/tactical distinction exists.
- `Searcher::set_quiescence`: at depth 0, stand-pat on the static
  evaluation, beta cutoff, then tactical moves only, ordered by the
  evaluator's policy scores; recursion bounded by a ledgered safety
  depth (32); quiescence nodes count against the node budget and are
  reported separately (`SearchResult::quiescence_nodes`). No TT
  interaction in this rung. Checks receive no special treatment yet
  (identical to the old leaf behavior in check; evasions are rung 3).
- OFF remains the default everywhere and is bit-identical to the
  pre-G1 search (the §14.3 reference mode; regression-covered by every
  pre-existing search test plus an explicit neutrality test).
- `quiescence` flags (serde-default false) on `relabel` and
  `ordered_mlp_match`; the paired arena takes per-side flags.

## Primary result — matches (300 pairs, c03 value + F2 ranker both sides)

| Match | Quiescence-side score [95% CI] |
|---|---|
| fc-full, fixed 400 nodes | **0.663 [0.626, 0.700]** (≈ +117 Elo) |
| fc-full, 20 ms/move | **0.658 [0.622, 0.695]** (≈ +114 Elo) |
| fc-small, fixed 400 nodes (mlp-w32@1M value) | 0.507 [0.502, 0.511] |

The fixed-node and fixed-time gains are equal: quiescence nodes are
cheap relative to what they buy, so nothing is lost converting the
gain to wall-clock. fc-small is saturated as always, yet still ticks
positive with LCB > 0.5.

**Combined stack vs Program 1** (ranker + quiescence vs policy
ordering + no quiescence, same value network, 20 ms/move):
**0.744 [0.712, 0.776] ≈ +186 protocol Elo.** The strongest Forward
Chess configuration is now: c03 champion value + learned ranker
ordering + capture/promotion quiescence.

## Secondary results

- **Teacher stability** (same generator settings as the F2 data, seed
  11, quiescence on; 7,079 positions — trajectories shift because the
  quiescent searches play differently, so this is an aggregate, not
  per-position, comparison): deep-label last-iteration stability
  **0.366 → 0.684**, mean best-move changes per search **1.27 → 0.52**
  — the card's ≥0.55 threshold is cleanly exceeded. Champion-policy
  deep-best rank is unchanged within sample noise (6.89 → 7.21), as
  expected: quiescence changes values, not the policy head.
- **Cost profile**: identical node budgets now take ~1.9× wall
  (labelling 4.5 vs 9.2 positions/s at ~70M nodes both) because
  quiescence nodes are evaluation-heavy — more model calls per node.
  The fixed-time match already nets this cost out: the calmer values
  are worth the extra evals even at 2× per-node cost.
- Hand-verified mechanism test: on a constructed poisoned-pawn
  position, the non-quiescent depth-1 search grabs the defended pawn
  (root value +400 with a material oracle); the quiescent search sees
  the recapture and declines it (+300, keeping the rook).

## Failure found and fixed during the experiment

The first match round returned *exactly* 0.5000 with perfectly
mirrored W/D/L (282/36/282) — the config carried the quiescence flags
but the arena never threaded them into its searchers, so both sides
played identically and every pair mirrored. Lesson recorded: a
perfectly mirrored paired result is the signature of a null A/B, not
a tie; the arena now takes explicit per-side quiescence flags, and the
first valid runs followed. (The one earlier movetime "result" from the
unwired build, 0.480, was jitter and is superseded.)

## Statistical uncertainty

Match CIs pair-level normal approximations; both headline gains are
>8 SE from 0.5. Single seed for matches (300 pairs each); the
fixed-node and fixed-time agreement and the stack match are mutually
consistent replications of the effect.

## CPU result

Each 600-game fc-full match 8–14 min at 4 threads; stability relabel
≈ 15–20 min. Quiescence keeps node budgets identical by construction
(quiescence nodes are counted), so fixed-node matches cost the same
wall time.

## Correctness evidence

105 tests green. New: quiescence neutrality on a no-tactical-moves
game (values, moves, and node counts identical to OFF); the
poisoned-pawn resolution test with exact expected scores; budget-abort
cleanliness with quiescence on; `is_tactical` exercised for FC
(capture/ep/promotion), chess, and Breakthrough.

## Complexity delta

+1 trait method with a default (rule-level fact), +1 search flag with
reference mode, +1 result field, per-side arena flags, 2 config fields
(serde-default); 1 ledger entry (`qs_max_depth` = 32, engineering);
0 dependencies; 0 deletions.

## Decision

- **Retain**: quiescence ON becomes the production search mode for
  Forward Chess play and teacher generation (label pipelines regenerate
  with it from here on). OFF stays as the reference mode in tests.
- **Next candidates**: (1) regenerate teacher records with the full
  stack (better teacher at equal cost) and refit the ranker — the
  §6.1 loop, now with a substantially stronger teacher;
  (2) rung 3 (checks/evasions in quiescence) only after measuring what
  instability remains; (3) the fc-full structured value model on the
  calmer labels.

## Exact reproducibility information

Configs: `configs/shsd/stage_g/`. Matches:
`runs/*ordermatch-fc-full*` / `*ordermatch-fc-small*` (post-fix runs).
Stability data: `runs/*relabel-fc-full*` (quiescence run, seed 11).
Ranker: F2 checkpoint. The unwired-flag runs are retained on disk as
the incident record.

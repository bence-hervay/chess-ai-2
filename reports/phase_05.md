# Phase 5 Report: Othello as an independent positional benchmark

## Status

PASS

## Research hypothesis

The unchanged generic stack learns a qualitatively different positional
game — delayed consequences, mobility/access effects, material
reversals — from board occupancy alone, with no Othello concepts
(corners, stable discs, mobility, frontier, parity, edges) anywhere in
the evaluator.

## Minimal implementation

`src/games/othello.rs` (parameterized even `width` × `height` ≥ 4):
bracket-and-flip placements in 8 directions; **pass as an explicit
move** (an auto-pass inside `make_move` would break negamax's
strict-alternation perspective flip — D023); game ends when neither
player can place; disc majority wins, equality draws. Passes cannot
repeat positions (two consecutive passes is terminal), so the state
graph stays acyclic. Flip records use a fixed-size undo array (no
per-node allocation). Features: occupancy only, side-to-move relative.
Action IDs: cell for placements plus one pass ID.

`GameSpec::Othello` + one `dispatch_game!` arm; `match_probe` gained a
required `opponent_checkpoint` (cross-run champion matches for the
data axis). Search, model, and training code: **zero changes**.

## Deliberately excluded

All §25-restricted concepts (corners, stable discs, mobility, frontier,
parity, edge bonuses); 8×8 (optional per plan; 6×6 scaling was the
gate this phase needed); bitboard optimizations.

## Correctness evidence

54 tests (fmt/clippy/debug/release clean). New:

- slow reference movegen (independent ray-scan predicate incl. pass
  legality) matches on 900 random states across three board shapes;
- make/unmake restores state and key exactly through flips and passes;
  recomputed keys and disc counts match incrementals;
- colour-swap + side-swap encodes identically (perspective features);
- multi-direction flip correctness on a constructed position;
- pass mechanics verified in real play (only-move passes leave the
  board unchanged, flip the side, opponent can always place, unmake
  restores) plus dead-board disc-count outcomes;
- termination with material-correct outcomes (300 random games);
- **4x4 solver agreement**: exact solver, exhaustive negamax, and
  full-depth alpha–beta agree that 4x4 Othello is a **first-player
  loss** (matching the known second-player-win result), 56,621
  reachable non-terminal states (W 28,536 / D 4,254 / L 23,831).

## Global learning or playing result

### Exact 4x4 (oracle labels only for evaluation)

Supervised ceiling (test split): the value head confirms the phase's
premise — occupancy-only WDL prediction is genuinely hard in Othello:

| Config | WDL acc | Action acc | Optimal mass | Regret |
|---|---:|---:|---:|---:|
| w64, 45k states | 0.8548 | 0.9740 | 0.9467 | 0.0363 |
| w128, 40k steps | 0.8531 | 0.9721 | 0.9565 | 0.0379 |

(Compare 99%+ WDL accuracy on Breakthrough/Connect-k: Othello's
material reversals make value classification far harder than move
choice — exactly the intended stress.)

Self-play recovery (no oracle labels; ε=0.10, w64, 600 nodes):

| Seed | Raw action acc | Searched acc (600) | Exploitability |
|---|---:|---:|---|
| 1 | 0.9420 | **1.0000** | 0 drops / 34 games |
| 2 | 0.9339 | **1.0000** | 0 / 34 |
| 3 | 0.9393 | **1.0000** | 0 / 34 |

Probe of the seed-1 checkpoint: accuracy 0.936 → 1.000 monotone in
budget (100% from 600 nodes); exploitability 4 → 0 drops; disagreement
analysis: deeper search **fixes 66 decisions and breaks 0** (1,000
states, budget 1 vs 5000).

### Strategic 6x6 (not exactly solvable; match promotion)

- Strength improves across promoted checkpoints: seed 2 ran 16
  generations with 9 promotions; champion-vs-generation-0 trajectory
  0.96 → **1.000 (160–0)**; seed 1 halted at generation 8 (0.938
  final). Mean game length ~30 plies.
- **Search scaling** (frozen seed-2 champion vs itself, 160 games per
  point): 60 → 0.287, 200 → 0.169, 600 → exactly 0.500 (78W/78L/4D
  control tie), 2400 → **0.856** (LCB 0.807).
- **Data scaling, head-to-head** (§25 "fixed-search performance
  improves with data"): the 400-games/generation champion beats the
  100-games/generation champion at equal 600-node search **0.750**
  (LCB 0.688, 148W/4D/48L over 200 games). The 100-game run also
  plateaued lower vs its own gen-0 (0.884 vs 0.938–1.000).

## Scaling result

Supervised data axis (4x4, w64): 11k → 22.5k → 45k states gives WDL
0.8025 → 0.8368 → 0.8548, regret 0.0595 → 0.0450 → 0.0363 — monotone,
not yet saturated (unlike Breakthrough 4x4). Width w64 → w128 at full
data is flat (0.855 vs 0.853): 4x4 Othello is data-limited, not
capacity-limited. Self-play and search axes above.

## CPU and memory result

4x4 runs: supervised ≤ 50 s (w64), self-play ~105 s for 12 generations.
6x6 self-play ~20 s/generation (400 games, ~30 plies each);
full runs 166–332 s. All probes minutes. Peak RSS < 300 MiB.

## Reproducibility

Commit 3a3d60d + this phase's commit; seeds explicit; configs
`configs/phase_05/` (incl. two sweep manifests); metadata for 14 runs
archived under `reports/phase_05/runs/`.

## Complexity delta

- Dependencies: 0. New game module ~380 production + ~250 test lines.
- Config surface: `GameSpec::Othello` (2 fields);
  `opponent_checkpoint` on match probes (now required, explicit).
- Search/model/training: unchanged (the §25 "same code used unchanged"
  criterion holds literally).

## Failures and anomalies

- First dead-board test positions were wrong (a corner disc still
  terminated a bracket); caught by the game's own `outcome`, fixed in
  the test, and the final positions verified by the reference movegen.
- Seed-3 4x4 self-play halted at generation 4 via the oracle tolerance
  gate — final quality identical to full-length runs (100% searched,
  0 exploitability), so the halt cost nothing.
- The tiny-budget value/policy asymmetry appears again (budget 10
  exploitability worse than budget 1 on 4x4), consistent with Phases
  3–4; gone by budget 100.

## Decision

- Promote phase: yes
- Selected configuration: unchanged stack; Othello rungs 4x4 exact /
  6x6 strategic; 8x8 deferred (optional per plan, nothing further to
  demonstrate at this phase's gate)
- Rejected alternatives: auto-pass state transitions (breaks negamax
  alternation); larger exact instances (6x4 not needed for the gate)
- Reason: every §25 criterion met — near-perfect exact play (100%
  searched decisions, 0 exploitability), monotone checkpoint
  progression at 6x6, head-to-head data-scaling proof at fixed search,
  literally unchanged learner code, and zero Othello-specific concepts

The pipeline has now demonstrated two independent styles of positional
learning (racing/blocking Breakthrough; flipping/mobility Othello) plus
the connection game family, all with one generic learner.

## Exact next phase

Phase 6 (standard chess integration, §26): integrate `cozy-chess` as
the rules backend behind the same `Game` trait (move legality,
repetition and 50-move handling need the trait's first real repetition
support), sparse feature encoding for chess state facts, UCI binary,
fixed-node and fixed-time strength measurement against reference
engines, supervised bootstrap policy decision, and the self-play loop
at chess scale. This is the phase where repetition handling enters the
search (documented as deferred since Phase 1).

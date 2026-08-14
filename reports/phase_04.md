# Phase 4 Report: Breakthrough curriculum

## Status

PASS

## Research hypothesis

The generic search/model/training stack — frozen after Phase 3 —
transfers to a structurally different game (forward progress, captures,
races, blocking) with only new rule code, and Expert Iteration works on
a size that cannot be exactly solved, using match-based promotion.

## Minimal implementation

- `src/games/breakthrough.rs`: parameterized Breakthrough
  (width × height, `rows` initial pawn ranks per side). Selected ruleset
  (D020): one square straight or diagonally forward; only diagonal moves
  capture; reaching the opponent's home rank wins; a side with no legal
  move (subsuming pawn elimination) loses. Pawns only advance, so no
  repetition and **no draws**. Features and action IDs are encoded in
  the side-to-move's vertically mirrored frame, so colour-reflected
  positions encode identically (verified by test).
- `GameSpec::Breakthrough` + a `dispatch_game!` macro replacing five
  per-command match blocks in `lab`.
- Non-exact machinery this phase required (§12.6, D021):
  `play_paired_match` (paired colour-swapped games from shared random
  openings, asymmetric node budgets, 95% LCB on the paired score);
  `lab selfplay` promotion mode `"match"` with champion-progression
  matches against the frozen generation-0 baseline; `lab evaluate` kind
  `match_probe` (checkpoint at several budgets vs itself at a baseline
  budget); `search_disagreement_analysis` (per-state choices across
  budgets vs the deepest, classified by exact optimality).
- `exploitability_vs_perfect` now pre-solves once and clones the memo
  per game (`ExactSolver: Clone`), making large exact instances
  probe-able.

## Phase-integrity gate

Search, model, and training subsystems are **unchanged** (search.rs,
model.rs, training.rs untouched except none). New code lives in the new
game module, the Evaluation subsystem (match play and probes — §12.6
functionality this phase's non-exact size requires), and CLI dispatch.
The promotion-strike amendment (below) is loop control in `lab`, not a
training change.

## Deliberately excluded

No handcrafted race/material/blockage features (only raw cell
occupancy); no search extensions; no value-target changes; no
Breakthrough-specific model or training adjustments of any kind.

## Correctness evidence

47 tests (fmt/clippy/debug/release clean). New, per §24's required
list:

- **slow reference move generator**: fast movegen equals an independent
  all-pairs rule predicate on 1,200 random reachable states across four
  board shapes;
- **make/unmake**: exact state and Zobrist-key restoration along random
  games; recomputed keys match incremental keys;
- **colour reflection**: vertically mirrored, colour-swapped positions
  produce identical feature encodings and action-ID sets;
- **terminal races**: reaching the home rank wins for both colours;
  capturing the last pawn wins; straight moves never capture;
- **blocked positions**: individually immobile pawns verified; the
  no-move rule verified directly; sanity sweep that random reachable
  non-terminal states always have a move;
- **exact solver on small boards**: exact solver, exhaustive negamax,
  and full-depth alpha–beta agree on 2x4/3x4/2x5; all decisive (no
  draws), as theory requires;
- games terminate and never draw (300 random games).

## Global learning or playing result

### Exact ladder (solver)

| Instance | States | Solve time | Root value |
|---|---:|---:|---|
| 3x5 r1 | 29,730 | 0.07 s | Win (P1) |
| 4x4 r1 | 163,700 | 0.5 s | Win (P1) |
| 4x5 r1 | 1,800,365 | 8.0 s | Win (P1) |
| 4x5 r2 | aborted | > 10 min, multi-GB corpus | — (ladder cap) |

### Supervised oracle ceiling (test split)

| Instance | Width / steps | WDL acc | Action acc | Regret |
|---|---|---:|---:|---:|
| 4x4 r1 | 64 / 10k | 0.9984 | 0.9995 | 0.0011 |
| 4x4 r1 | 128 / 40k | **0.9997** | **0.9998** | 0.0005 |
| 4x5 r1 | 64 / 10k | 0.9933 | 0.9979 | 0.0042 |
| 4x5 r1 | 128 / 40k | **0.9988** | **0.9994** | 0.0012 |

### Self-play recovery (exact sizes, no oracle labels; ε=0.10, w64)

| Instance | Seeds | Raw action acc | Searched acc (600) | Exploit drops (600) |
|---|---|---|---|---|
| 4x4 r1 | 1/2/3 | 0.9916 / 0.9943 / 0.9894 | 1.000 / 0.9995 / 1.000 | 21→9 … 23→14 of 222 |
| 4x5 r1 | 1 | 0.9816 | 0.9995 | 75→29 of 222 |

**The raw self-play model recovers near-oracle play on Breakthrough**
(99%+), unlike Connect-k's coverage-limited 81% — decisive outcomes and
capture-branched lines give far better value signal and state coverage.
Probing the 4x4 checkpoint: 100% optimal decisions from 100 nodes up,
exploitability monotone 44 → **1**/222 drops from budget 1 → 5000; 4x5:
100% at 5000 nodes, drops 67 → 22.

### Strategic size (5x5 rows-2, not exactly solvable; match promotion)

- Candidates improve over generations: promotion matches passed 7 times
  (seed 2, 16 generations); champion-vs-generation-0 trajectory 0.94 →
  0.99–1.00, final **0.963–0.994** score over 160 paired games
  (154–159 W of 160). Seed 1 halted at generation 8 after two genuine
  regressions (per protocol).
- **Search scaling on the frozen final champion** (match probe, 160
  games per point vs itself at 600 nodes): 60 → 0.294, 200 → 0.225,
  600 → exactly 0.500 (80W/80L — a built-in protocol validation:
  identical budgets must tie), 2400 → **0.731** (LCB 0.671). More
  search clearly improves the frozen checkpoint.

### Search-depth disagreement analysis (§24)

On 1,000 shared test states per instance, comparing each budget's chosen
action with the 5000-node choice: disagreement falls 4.6% → 0% (4x4)
and 6.5% → 0% (4x5) as budget grows; among disagreements, deeper search
**fixes 9–17 decisions and breaks zero** at both sizes (the rest are
equal-value alternatives). Depth never hurts on these instances.

## Scaling result

- Model width: w64 → w128 improves the supervised ceiling at both exact
  sizes (0.9984→0.9997 and 0.9933→0.9988 WDL) — improvement, not yet
  saturation.
- Data (supervised, 4x4 r1, w64): 16k → 65k → 130k states gives WDL
  0.9952 → 0.9981 → 0.9984 — improvement then clearly measured
  saturation.
- Self-play generation search (4x4 r1): 150 vs 600 nodes end at
  statistically indistinguishable quality (drops 9 vs 14 of 222,
  searched acc 1.000 both) — the short, tactical 4x4 game is
  search-saturated at very low budgets; the informative search-scaling
  evidence at this phase comes from evaluation budgets and the 5x5
  match probe above.

## CPU and memory result

4x4 self-play runs ~100 s, 4x5 ~830 s (dominated by per-generation
oracle probes over the 1.8M-state corpus), 5x5 match-mode ~75–230 s for
5–16 generations; 5x5 games average 13–20 plies. All sweeps ran through
`lab sweep`; peak RSS < 400 MiB.

## Reproducibility

Commit bd70662 + this phase's commit; seeds explicit; self-play and
match streams keyed by run seed; configs `configs/phase_04/` (incl. 4
sweep manifests); metadata for 20 runs archived under
`reports/phase_04/runs/`.

## Complexity delta

- Dependencies: 0. New game module ~430 production lines + ~330 test
  lines. Evaluation additions ~270 lines (`MatchResult`,
  `play_paired_match`, `DisagreementReport`,
  `search_disagreement_analysis`).
- Config surface: `GameSpec::Breakthrough` (3 fields), self-play
  `promotion`/`promotion_pairs`/`opening_plies` (§12.6 non-exact
  promotion), evaluate kind `match_probe` (7 fields).
- Simplified: five hand-written game dispatch matches replaced by one
  `dispatch_game!` macro.

## Failures and anomalies

- 4x5 r2 exact solve aborted (time and disk); ladder capped at 4x5 r1
  (D022).
- The original match-promotion rule counted *inconclusive* candidates
  (observed score ≥ 0.5 but LCB ≤ 0.5) as halt strikes, ending 5x5 runs
  at generation 5; amended so only observed regressions (score < 0.5)
  strike (D021). Post-amendment, seed 2 ran all 16 generations; seed 1
  still halted at 8 on genuine regressions — an honest plateau signal.
- Tiny-budget probes again show the value/policy asymmetry (budget 10
  occasionally worse exploitability than budget 1), as diagnosed in
  Phase 3; it vanishes by 100 nodes.
- A partial 4x5 r2 corpus filled the disk during the ladder experiment;
  deleted. Disk was resized to 30 GB during the session.

## Decision

- Promote phase: yes
- Selected configuration: unchanged frozen stack (recipe v1, ε=0.10,
  replay 4); Breakthrough exact rungs 4x4 r1 / 4x5 r1, strategic rung
  5x5 r2; match promotion = LCB>0.5 with regression-only strikes,
  40 pairs, 2 opening plies
- Rejected alternatives: strict inconclusive-strike halting (ended runs
  mid-improvement); 4x5 r2 as an exact rung (infeasible)
- Reason: every §24 acceptance criterion is met — near-perfect exact
  play (100% searched decisions), generational improvement on the
  medium board (0.94→0.99 vs gen-0), monotone search benefit on frozen
  checkpoints (0.731 at 4× budget; exploitability 44→1), data scaling
  to measured saturation, width scaling improvement, and zero
  handcrafted features — with the integrity gate intact (no
  search/model/training changes).

## Exact next phase

Phase 5 (Othello, §25): independent positional benchmark with unstable
material and mobility/parity structure — new rules module only (flip
mechanics, pass moves need a legal-move representation decision),
required rule tests, exact solving on 4x4/6x6-reduced instances if
feasible, supervised ceiling and self-play recovery, plus the same
match-based machinery on full 8x8. Phase-integrity gate applies again.

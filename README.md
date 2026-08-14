# Minimal CPU-First Self-Play Game AI

A Rust research system that learns deterministic, two-player, perfect-information,
zero-sum board games through search-guided self-play (Expert Iteration with
generic alpha-beta search), without game-specific strategic knowledge.

The governing specification is
[`minimal_cpu_first_self_play_game_ai_research_plan.md`](minimal_cpu_first_self_play_game_ai_research_plan.md).
The project progresses one validated phase at a time; each phase ends with a
report in [`reports/`](reports/).

## Building and testing

```bash
cargo build --release
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
```

## Usage

```bash
lab evaluate <config.toml>   # paired evaluation arena
```

Configurations live in `configs/`. Every run creates a self-contained
directory under `runs/` with the fully resolved configuration, an
environment manifest, metrics, and game records.

## Status

| Phase | Result | Report |
|---|---|---|
| 0 — Deterministic foundation | **PASS** | [reports/phase_00.md](reports/phase_00.md) |
| 1 — Exact solving and search correctness | **PASS** | [reports/phase_01.md](reports/phase_01.md) |
| 2 — Supervised oracle ceiling | **PASS** | [reports/phase_02.md](reports/phase_02.md) |
| 3 — Self-play recovery of exact strategy | **PASS** | [reports/phase_03.md](reports/phase_03.md) |
| 4 — Breakthrough curriculum | **PASS** | [reports/phase_04.md](reports/phase_04.md) |
| 5 — Othello benchmark | **PASS** | [reports/phase_05.md](reports/phase_05.md) |
| 6 — Standard chess | not started | — |

### Current state (2026-08-14)

Phases 0–5 complete and promoted. **Two independent styles of
positional learning demonstrated with one unchanged learner**:

- Othello (occupancy features only, pass moves, material reversals):
  4x4 searched self-play play is perfect (100% optimal, 0
  exploitability, all seeds) while occupancy-only value prediction is
  measurably hard (85% WDL acc vs 99%+ elsewhere — the intended
  stress). 6x6 champions reach 160-0 vs generation 0; 4x search wins
  0.856; the 4x-data champion wins 0.750 head-to-head at equal search.
- 4x4 Othello solved: first-player loss (56,621 states), matching
  literature.

Phase 4 (Breakthrough) established transfer with rule code only. **The frozen stack transfers to a new
game with rule code only** (Breakthrough, §24 integrity gate intact):

- Exact sizes: supervised ceiling 99.97% WDL accuracy; self-play
  recovery reaches **99%+ raw** and 100% searched optimal decisions —
  no Connect-k-style coverage plateau (decisive games, richer lines).
- Strategic 5x5 (not exactly solvable): match-based promotion (paired
  colour-swapped games, LCB gate) drives champions from 0.94 to 0.99
  score vs the generation-0 baseline; a frozen champion at 4× search
  beats itself at 1× with 73% (the equal-budget control ties exactly).
- Search-depth disagreement analysis: deeper search fixes decisions and
  breaks none, at both exact sizes.

Phase 3 established ExIt recovery on Connect-k. **Expert Iteration recovers exact
play without oracle labels** on the solved benchmark (4x4 k=4 gravity):

- The searched self-play agent reaches **100% oracle-optimal decisions**
  on 2,000 held-out states at 5,000-node search (99.6–99.9% at the
  600-node training budget, all 3 seeds), with exploitability vs
  perfect play down to 2/42 games.
- The diagnostic matrix isolates the one open gap: the *raw* self-play
  network plateaus at ~81% optimal actions because a generation of 400
  games covers only ~500 distinct states of the 107k corpus —
  a coverage property, not representation/optimization/search failure.
- Frozen mechanisms: ε=0.10 exploration (calibrated 3ε×3 seeds), FIFO
  replay window of 4 generations, oracle-metric promotion with halt-on-
  regression. Custom compiled inference at 6.4 µs/forward (10× the
  framework path, validated to 1e-4).
- Generation-search budget is the strongest recovery lever (final
  exploit drops 12→4→2 at 150/600/2400 nodes); evaluation budget scales
  monotonically to exact play.

Phase 2 established the supervised ceiling (99.4% held-out WDL
accuracy, 99.8% optimal actions at w128 — raw network ≈ oracle), with
monotone width/data scaling. Phase 1 established the search machinery
is mathematically correct before any learning exists:

- **100% agreement** between the memoized exact solver, plain exhaustive
  negamax, and the production alpha–beta search (iterative deepening,
  transposition table on/off, natural/reversed ordering) on all seven
  solved Connect-k instances — values and optimal actions.
- External anchors: tic-tac-toe solves to a draw with exactly 4,520
  reachable non-terminal states (the literature count), validating both
  the solver and Zobrist hashing.
- The transposition table cuts alpha–beta nodes 8–65x with identical
  results; node-budget searches deterministically return the last
  completed iteration.
- Exact solver throughput ~2.3M states/s; oracle rung fixed at 5x4 k=4
  gravity (3.1M-state corpus with WDL + optimal-action labels, 8.8 s).
- Phase 0 foundation: byte-identical game records across 1/2/4/8 worker
  threads; 1.13M random games/s at 8 workers.

Next: Phase 6 — standard chess: cozy-chess rules backend behind the
same Game trait, repetition handling, UCI binary, fixed-node and
fixed-time strength measurement, self-play at chess scale.

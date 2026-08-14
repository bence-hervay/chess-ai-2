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
| 3 — Search + learned evaluation | not started | — |

### Current state (2026-08-14)

Phases 0–2 complete and promoted. The supervised ceiling of the
architecture is established on a solved oracle instance:

- **Raw network ≈ oracle** on 4x4 k=4 gravity: 99.4% held-out WDL
  accuracy, 99.8% optimal-action accuracy, 0.0024 WDL levels lost per
  decision (w128, 40k steps, fixed recipe v1).
- **Monotone scaling** in width (16→128: 78%→99.1% WDL acc) and data
  (1/8→full: 90.5%→97.0% at w64), tight across seeds; residual error is
  a generalization gap, not an optimization failure.
- Deterministic training (identical seeds → identical loss traces),
  in-run verified checkpoints, `lab sweep` CPU-slot scheduler.

Phase 1 established the search machinery is mathematically correct
before any learning exists:

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

Next: Phase 3 — the trained network as leaf evaluator and move-ordering
policy inside alpha–beta; play-strength and exploitability measurement
against perfect play.

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
| 1 — Exact solving and search correctness | not started | — |

### Current state (2026-08-14)

Phase 0 complete and promoted. The foundation exists: a rule-facts-only
`Game` trait, parameterized Connect-k (gravity on/off), a paired
random-agent arena, and self-contained run directories with full
environment manifests. Key evidence:

- **Determinism**: four benchmark runs at 1/2/4/8 worker threads produced
  byte-identical 209 MB game-record files (2,000,000 games, same sha256).
- **Parallel scaling** on 4 physical cores / 8 SMT threads: 247k games/s
  (1 worker, 99.8% utilization) → 891k (4 workers, 90% efficiency) → 1.13M
  (8 workers, SMT). ~24M moves/s peak.
- 20 unit/property tests plus differential terminal-detection oracle;
  fmt/clippy/test clean in debug and release.

Next: Phase 1 — exact negamax solver, alpha–beta with iterative deepening
and transposition table, proven equivalent on small Connect-k instances.

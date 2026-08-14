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
| 0 — Deterministic foundation | in progress | — |

### Current state (2026-08-14)

Phase 0 implementation complete: `Game` trait, parameterized Connect-k
(width x height, target k, gravity on/off), random agent, paired arena with
deterministic per-game RNG streams, self-contained run directories, and
unit/property tests. Benchmarks pending.

# Architecture

One Cargo package. Five conceptual subsystems are allowed by the governing
plan (`Game`, `Search`, `Model`, `Training`, `Evaluation`); only the ones a
completed phase required exist.

## Current modules (Phase 0)

| Module | Role |
|---|---|
| `src/game.rs` | Core `Game` trait: rule-level facts only (moves, transitions, outcomes, hashing, raw features, stable action IDs). |
| `src/games/connect_k.rs` | Parameterized Connect-k (width, height, k, gravity), Zobrist hashing, incremental terminal detection. |
| `src/evaluation.rs` | Paired random-versus-random arena; one single-threaded game per rayon worker; RNG streams keyed by `(run_seed, pair, slot)` so results are independent of thread count. |
| `src/experiment.rs` | Self-contained run directories, environment manifests, CPU/RSS probes. |
| `src/bin/lab.rs` | Typed CLI. Phase 0 exposes only `lab evaluate <config>`. |

## Not yet present (deferred to their phases)

Search (`search.rs`), model (`model.rs`), training (`training.rs`), exact
solver, sweep scheduler, other games, UCI binary.

## Determinism policy

- Every source of randomness is a `ChaCha12Rng` derived from the run seed.
- Per-game streams depend only on `(run_seed, pair, slot)` — never on thread
  identity or scheduling.
- Zobrist keys come from a fixed constant seed, so position keys are stable
  across processes.
- Batches of game records are collected in pair order before being written.

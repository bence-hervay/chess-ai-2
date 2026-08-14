# Architecture

One Cargo package. The five conceptual subsystems allowed by the governing
plan (`Game`, `Search`, `Model`, `Training`, `Evaluation`) all exist as of
Phase 2; each appeared in the phase that required it.

## Current modules (Phase 2)

| Module | Role |
|---|---|
| `src/game.rs` | Core `Game` trait: rule-level facts only (moves, transitions, outcomes, hashing, raw features, stable action IDs). |
| `src/games/connect_k.rs` | Parameterized Connect-k (width, height, k, gravity), Zobrist hashing, incremental terminal detection, feature/action ID spaces. |
| `src/search.rs` | Exact memoized WDL solver + reachable-state enumeration (research/test oracle); production alpha–beta with iterative deepening, transposition table, deterministic node budgets. |
| `src/model.rs` | Sparse policy/value network in Burn (NdArray CPU backend): embedding-sum → 2×ReLU → WDL head + action-logit head. One capacity knob (`model_width`). |
| `src/training.rs` | Exact-corpus dataset (hash-stratified 80/10/10 splits), fixed training recipe v1 (Adam, batch 256, lr 1e-3, `L = L_wdl + L_policy`), deterministic single-threaded training loop. |
| `src/evaluation.rs` | Paired game arena (deterministic RNG per `(run_seed, pair, slot)`); oracle metrics for raw models (WDL accuracy/log-loss/Brier, optimal-action accuracy and mass, decision regret in WDL levels). |
| `src/experiment.rs` | Self-contained run directories, environment manifests, CPU/RSS probes. |
| `src/bin/lab.rs` | Typed CLI: `lab evaluate`, `lab solve`, `lab train`, `lab sweep` (CPU-slot scheduler over a JSONL manifest of runs). |

## Not yet present (deferred to their phases)

Search-integrated model evaluation (Phase 3), self-play training loop
(Phase 3+), other games (Breakthrough, Othello), UCI binary.

## Determinism policy

- Every source of randomness is a `ChaCha12Rng` derived from the run seed,
  except model initialization, which uses Burn's backend RNG seeded once
  per process from the run seed.
- Per-game streams depend only on `(run_seed, pair, slot)` — never on thread
  identity or scheduling.
- Zobrist keys come from a fixed constant seed, so position keys are stable
  across processes.
- Training is single-threaded within a run; parallelism happens at the
  sweep level (independent processes), so training runs are bit-reproducible.
- Burn's backend RNG is process-global: tests that touch it serialize on a
  shared lock; production code never shares a process between runs.
- Data-sweep subsets are selected by a fixed, seed-independent stream so
  seed sweeps compare identical datasets.

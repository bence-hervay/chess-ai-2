# Architecture

One Cargo package. The five conceptual subsystems allowed by the governing
plan (`Game`, `Search`, `Model`, `Training`, `Evaluation`) all exist as of
Phase 2; each appeared in the phase that required it.

## Current modules (Phase 5)

| Module | Role |
|---|---|
| `src/game.rs` | Core `Game` trait: rule-level facts only (moves, transitions, outcomes, hashing, raw features, stable action IDs). |
| `src/games/connect_k.rs` | Parameterized Connect-k (width, height, k, gravity), Zobrist hashing, incremental terminal detection, feature/action ID spaces. |
| `src/games/breakthrough.rs` | Parameterized Breakthrough (width, height, initial pawn rows); no draws; perspective-mirrored feature/action encoding. |
| `src/games/othello.rs` | Parameterized Othello (even boards ≥ 4); explicit pass moves keep negamax alternation valid; occupancy-only features. |
| `src/search.rs` | Exact memoized WDL solver + reachable-state enumeration (research/test oracle); production alpha–beta with iterative deepening, transposition table, deterministic node budgets, and `Evaluator`-supplied leaf values + policy move ordering. |
| `src/model.rs` | Sparse policy/value network in Burn (NdArray CPU backend): embedding-sum → 2×ReLU → WDL head + action-logit head. One capacity knob (`model_width`). `CompiledNet`: validated plain-array inference path (6.4 µs batch-1) used by search and self-play. |
| `src/training.rs` | Exact-corpus dataset (hash-stratified 80/10/10 splits), fixed training recipe v1 (Adam, batch 256, lr 1e-3, `L = L_wdl + L_policy`), deterministic warm-startable training loop, and deterministic self-play generation (frozen champion, ε-greedy exploration, §12.2 records). |
| `src/evaluation.rs` | Paired game arena (deterministic RNG per `(run_seed, pair, slot)`); oracle metrics for raw models; searched-decision metrics, exploitability vs perfect opposition, paired model-vs-model matches with LCB promotion gates, and search-depth disagreement analysis. |
| `src/experiment.rs` | Self-contained run directories, environment manifests, CPU/RSS probes. |
| `src/bin/lab.rs` | Typed CLI: `lab evaluate` (arena / oracle probe), `lab solve`, `lab train`, `lab selfplay` (Expert Iteration generations), `lab sweep` (CPU-slot scheduler over a JSONL manifest of runs). |

## Not yet present (deferred to their phases)

Chess (cozy-chess backend), UCI binary, time-based search budgets,
repetition handling (no current game can repeat).

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

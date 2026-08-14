# Phase 0 Report: Minimal deterministic foundation

## Status

PASS

## Research hypothesis

A minimal deterministic Rust skeleton — a rule-facts-only `Game` trait,
parameterized Connect-k, a random legal agent, a paired arena with seeded
per-game RNG streams, and self-contained run directories — can produce
byte-identical game records from a fixed seed regardless of worker-thread
count, and independent-game parallelism uses this machine's 8 logical CPUs
(4 physical cores, 2-way SMT) sensibly. This resolves the uncertainty of
whether the experimental foundation (interfaces, determinism discipline,
run storage, parallel harness) is trustworthy before any search or learning
code exists.

## Minimal implementation

- `src/game.rs`: `Game` trait (`State`/`Move`/`Undo`, `legal_moves`,
  `make_move`/`unmake_move`, `outcome`, `position_key`, `encode_features`,
  `action_id`), `Player`, `Outcome`. Rule-level facts only.
- `src/games/connect_k.rs`: Connect-k with width, height, k, gravity on/off;
  Zobrist hashing from a fixed constant seed; incremental terminal detection
  (scan only the four lines through the placed stone); perspective-normalized
  sparse features (`2*cell + own/opponent`); action IDs = column (gravity) or
  cell (no gravity).
- `src/evaluation.rs`: paired random-vs-random arena. One single-threaded
  game per rayon task; per-game `ChaCha12Rng` derived only from
  `(run_seed, pair, slot)` via splitmix64; records serialized to JSONL inside
  workers and emitted in deterministic pair order in batches of 8192 pairs.
- `src/experiment.rs`: run directories (`resolved.toml`, `manifest.json`,
  `metrics.jsonl`, `summary.json`, `games/games.jsonl`, `stdout.log`),
  environment manifest (git commit/dirty, rustc, Cargo.lock hash, CPU model,
  logical CPUs, RAM, build flags, seed, command, timestamps), CPU-seconds and
  peak-RSS probes from `/proc`.
- `src/bin/lab.rs`: `lab evaluate <config>` with a fully required
  `EvaluationConfig` (`seed`, `pairs`, `threads`, `[game]`); prints its
  thread plan before substantial work.

## Deliberately excluded

Neural networks, alpha–beta, exact solver, self-play training, sweep
scheduler, other games, a general board abstraction, `search.rs`/`model.rs`/
`training.rs` placeholder files, `tracing`, `criterion`, an Agent trait
(only one agent exists — the random agent is a function), and the
`solve`/`train`/`sweep`/`report` CLI commands (their phases have not begun).

## Correctness evidence

- Unit tests (18): parameter validation; gravity and no-gravity move
  generation; horizontal/vertical/diagonal wins; draw on full board;
  terminal states generate no moves; make/unmake restores state and position
  key exactly; incremental outcome matches a slow full-board reference scan
  along a full playout; features are side-to-move-relative (colour-swapped
  mirror positions encode identically); colour-swap win symmetry; stable
  action IDs; `legal_moves` does not mutate state; arena determinism across
  1 vs 4 threads (byte-identical JSONL); different seeds differ; recorded
  games are terminal with legal lengths; timestamp/FNV/probe correctness.
- Property tests (proptest, boards 2..=7 x 2..=6, both gravity modes):
  scripted playouts maintain invariants (moves distinct and legal, action
  IDs distinct, feature encoding deterministic), full unwind restores the
  initial state and every intermediate position key; playouts terminate
  within `w*h` plies.
- Differential test: incremental terminal detection vs slow reference scan.
- `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D
  warnings`, `cargo test`, `cargo test --release` all pass.

## Global learning or playing result

No learning exists yet by design. Behavioural evidence from the arena
(random vs random, 2,000,000 games, 7x6 k=4 gravity, seed 42):

- P1 wins 55.57%, P2 wins 44.18%, draws 0.25%, mean 21.3 plies — the
  expected first-mover advantage for random Connect Four play, identical
  across every thread count.
- All four benchmark runs (1/2/4/8 threads, same seed) produced
  byte-identical 209 MB `games.jsonl` files
  (sha256 `44bfb1fa…5a245e1`).
- No-gravity 5x5 k=4 also runs end-to-end (100,000 games: P1 52.9%,
  draws 6.5%).

## Scaling result

Model/data axes do not exist yet. Worker-scaling (fixed workload:
1,000,000 pairs, seed 42, 7x6 k=4):

| Workers | Games/s | Moves/s | Parallel efficiency vs 1 thread | Utilization |
|---:|---:|---:|---:|---:|
| 1 | 246,772 | 5.26M | 1.000 | 99.8% |
| 2 | 474,853 | 10.12M | 0.962 | 96.2% |
| 4 | 891,468 | 19.00M | 0.903 | 90.8% |
| 8 | 1,129,758 | 24.07M | 0.572 | 83.8% |

Interpretation: near-linear scaling across the 4 physical cores. The
8-worker point uses 2-way SMT (only 4 physical cores exist); 8 workers gain
a further 1.27x over 4 workers, which is normal SMT yield for this
compute-bound workload, not a harness defect. The honest per-physical-core
efficiency at 8 workers is 1,129,758 / (4 x 246,772) = 1.14 with SMT.

A measured bottleneck was found and fixed during this phase: JSONL
serialization initially ran serially in the record sink, capping 8-worker
utilization at 60.9%. Moving serialization into the workers (sink now only
writes bytes) raised it to ~84% and throughput from 891k to 1.13M games/s.

## CPU and memory result

See table above. Peak RSS ≤ 11.1 MiB in all runs (record batches are
streamed to disk in 8192-pair batches). Process CPU seconds measured from
`/proc/self/stat`; utilization = cpu_seconds / (wall x allocated threads).

## Reproducibility

- Git commit: 492dfc3 (benchmarks ran on this commit; report added after)
- Rust toolchain: rustc 1.97.1 (8bab26f4f 2026-07-14), pinned in
  `rust-toolchain.toml`
- Cargo.lock: committed; hash recorded per run in `manifest.json`
- Seeds: 42 (benchmark), 1 (smoke), 5 (no-gravity)
- Resolved configs: `configs/phase_00/` and each run's `resolved.toml`
- Run directories: metadata archived under `reports/phase_00/runs/`
  (5 runs; bulk `games.jsonl` files not committed — regenerable exactly
  from seed + config)

## Complexity delta

- Production LOC: ~1,050 added (plus ~380 test LOC), 0 removed (new repo)
- Public types: added 13 — `Player`, `Outcome`, `Game`, `ConnectK`,
  `ConnectKMove`, `ConnectKState`, `ConnectKUndo`, `GameSpec`, `GameRecord`,
  `ArenaCounters`, `ArenaSummary`, `Manifest`, `RunDir` (all required by the
  Phase 0 implement list: trait, one game, arena, run storage)
- Config keys: 8 — `seed`, `pairs`, `threads`, `game.{kind,width,height,k,gravity}`;
  all varied in this phase's runs (threads 1/2/4/8, gravity on/off, sizes)
- Dependencies: 7 production (`clap`, `serde`, `serde_json`, `toml`, `rand`,
  `rand_chacha`, `rayon`), 1 dev (`proptest`) — all from the plan's §6.3
  expected list; `serde_json` justified in DECISIONS.md D005
- Permanent algorithmic branches: gravity vs no-gravity move generation
  (a rule difference, not a mode flag)
- New executables: `lab` (authorized by §7)
- Files: 18 tracked

## Failures and anomalies

- Serial record serialization initially destroyed 8-worker scaling
  (60.9% utilization). Diagnosed by measurement, fixed by in-worker
  serialization, verified by re-measurement. No other anomalies.
- Draw rate at 7x6 k=4 under random play is only 0.25%, so the paired-arena
  draw path is exercised mostly by the 5x5 no-gravity instance (6.5% draws).

## Decision

- Promote phase: yes
- Selected configuration: n/a (no competing configurations; thread sweep is
  the deliverable)
- Rejected alternatives: serial-sink record writing (measured, replaced,
  deleted per one-way-evolution rule)
- Reason: all acceptance criteria met — tests pass, fixed seeds reproduce
  byte-identical records across thread counts, no unexplained
  nondeterminism, one game implementation, no model/search abstractions,
  parallel arena shows near-linear physical-core scaling with the complexity
  audit above.

## Exact next phase

Phase 1 builds the exact oracle and proves production search correct on
small Connect-k instances: exhaustive negamax, memoized solving, alpha–beta
pruning, iterative deepening, a transposition table, deterministic node
accounting, exact optimal-action enumeration, and an exact evaluation
corpus. Required experiments: exhaustive minimax vs alpha–beta agreement on
all tested reachable positions, transposition table on/off, natural vs
reversed move ordering, exact node counts, and solve-time scaling by board
size. No neural code.

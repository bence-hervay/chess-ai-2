# Phase 1 Report: Exact solving and search correctness

## Status

PASS

## Research hypothesis

A memoized exact negamax solver and the production alpha–beta search
(negamax, pruning, iterative deepening, transposition table, deterministic
node budgets) return identical game-theoretic results on small Connect-k
instances, with the transposition table and move ordering changing node
counts but never values. This resolves whether the search machinery is
mathematically trustworthy before any learned component enters it.

## Minimal implementation

`src/search.rs` (+~540 production lines):

- `Wdl` value type (`Loss < Draw < Win`) with perspective flip;
- `ExactSolver`: memoized exhaustive negamax over the reachable state DAG
  (no pruning), optimal-action-set enumeration (category-optimal moves),
  node/memo-hit counters;
- `enumerate_solved`: deterministic DFS enumeration of every reachable
  non-terminal state, deduplicated by position key — drives exact corpus
  generation;
- `exhaustive_negamax`: plain full-tree negamax, the memo-free reference;
- `Searcher`: production negamax + alpha–beta + iterative deepening +
  optional fixed-size always-replace transposition table (exact/lower/upper
  bounds, mate-distance-adjusted scores) + deterministic node budgets that
  return the last fully completed depth; move order is TT move first, then
  stable action-ID order (`MoveOrdering::Reversed` exists for controlled
  degradation experiments);
- score convention: forced win at ply p = `32000 - p`, draw = 0, leaf
  evaluations clamped to ±1000 (no model exists yet; experiments use a
  zero evaluator).

`lab solve <config>` runs one method (`exact`, `exhaustive`, `alpha_beta`,
`alpha_beta_tt`) on one game instance and records everything in a run
directory; `exact` also writes `corpus.jsonl` (key, ply, WDL, sparse
features, legal action IDs, optimal action IDs — all side-to-move
perspective).

Fixed during the phase: run-directory name collisions (two same-label runs
within one second silently overwrote each other; now suffixed uniquely).

## Deliberately excluded

Neural code, policy-based move ordering (no policy exists), null-move/LMR/
futility/aspiration/quiescence (per plan §11.5–11.6), parallel search,
TT size configuration (fixed 2^20 entries for solve runs), distance-to-
terminal in the exact solver (optional per plan; add when a phase needs it),
repetition handling (Connect-k cannot repeat positions; comes with games
that can).

## Correctness evidence

- Unit/property tests: 29 total (9 new search tests), all passing in debug
  and release; fmt and clippy clean.
- Exact-search tests:
  - tic-tac-toe (3x3 k=3, no gravity) solves to **Draw** by all four
    methods — external literature anchor;
  - enumeration visits exactly **4,520** non-terminal states on
    tic-tac-toe, matching both the literature (5,478 reachable minus 958
    terminal) and a full-state-dedup reference — this also validates
    Zobrist keying (no collisions, side-to-move included);
  - exhaustive negamax and alpha–beta (TT on/off × natural/reversed
    ordering) return byte-equal scores on 90 sampled reachable states
    across three instances;
  - search best moves always lie in the exact optimal-action set (sampled,
    both orderings);
  - depth-1 search with oracle leaf values always chooses an optimal root
    action;
  - node-budget searches reproduce exactly the value, move, and node count
    of an unbudgeted search truncated at the completed depth;
  - repeated searches are bit-identical (nodes, value, move);
  - an immediate win scores exactly `SCORE_WIN - 1` with the winning move.

## Global learning or playing result

No learning yet (by design). Global solving results — root values with
**100% agreement across every method, ordering, and TT setting** (7
instances, 21 runs):

| Instance | Root value | Optimal first actions |
|---|---|---|
| 3x3 k3 gravity | Draw | 0, 1, 2 (all) |
| 3x3 k3 free (tic-tac-toe) | Draw | all 9 |
| 4x3 k3 gravity | Win | all 4 |
| 4x3 k3 free | Win | 10 of 12 |
| 4x4 k3 gravity | Win | all 4 |
| 4x4 k4 gravity | Draw | all 4 |
| 5x4 k4 gravity | Draw | 1, 2, 3 (centre) |

## Scaling result

Model/data axes still do not exist. Search-cost results:

### Exact solver scaling by board size (single thread)

| Instance | Non-terminal states | Solver nodes | Solve time | Peak RSS |
|---|---:|---:|---:|---:|
| 3x3 k3 g | 505 | 2,441 | 0.001 s | 3 MiB |
| 3x3 k3 f | 4,520 | 36,864 | 0.011 s | 3 MiB |
| 4x3 k3 g | 4,631 | 28,272 | 0.011 s | 3 MiB |
| 4x4 k3 g | 23,930 | 155,447 | 0.067 s | 4 MiB |
| 4x3 k3 f | 79,563 | 861,700 | 0.177 s | 8 MiB |
| 4x4 k4 g | 134,289 | 743,442 | 0.263 s | 12 MiB |
| 5x4 k4 g | 3,100,379 | 20,615,635 | 8.81 s | 143 MiB |

~2.3M states/s solve throughput. **6x4 k4 gravity exceeds the modest local
budget**: its corpus write passed 4.8 GB and exhausted disk before
completing; the run was aborted and deleted. The oracle rung for later
phases is therefore at most 5x4 (3.1M-state corpus, 396 MB, 8.8 s).

### Method comparison (full solve of the initial position)

| Instance | Exhaustive nodes | AB (ID, no TT) | AB (ID + TT) | Exact solver nodes |
|---|---:|---:|---:|---:|
| 3x3 k3 g | 3,278 | 1,483 | 886 | 2,441 |
| 3x3 k3 f | 549,946 | — | 8,295 | 36,864 |
| 4x3 k3 g | 277,167 | — | 2,243 | 28,272 |
| 4x4 k4 g | (infeasible) | 211,714 | 25,516 | 743,442 |
| 5x4 k4 g | (infeasible) | 54,345,068 | 830,931 | 20,615,635 |

The transposition table cuts alpha–beta node counts by **8x** (4x4) to
**65x** (5x4) while returning identical values everywhere.

### Ordering experiment

From the initial position, natural and reversed orderings cost *exactly*
211,714 nodes each (4x4 k4, no TT). This is not a bug: Connect-k boards are
mirror-symmetric, so reversing column order explores the isomorphic
mirrored tree — a strong incidental determinism check. On asymmetric
sampled positions (8 states, 5x4 k4, no TT) the orderings genuinely
diverge per position (totals 62.9M natural vs 68.7M reversed) while every
value stays equal. With a zero evaluator both orderings are arbitrary, so
neither dominates; the ordering machinery matters once a learned policy
provides real move preferences (Phase 2+).

## CPU and memory result

Solve runs are single-threaded by design (parallelism remains at the
run level: the 17 small experiments ran under `xargs -P 7`). Largest run:
5x4 exact = 8.8 s solve + corpus streaming, 143 MiB peak RSS. TT runs:
27 MiB (2^20-entry table).

## Reproducibility

- Git commits: 492dfc3 (Phase 0 base), 64d3400 (experiments; report added
  after)
- Rust toolchain: rustc 1.97.1, pinned
- Seeds: none required (solving is deterministic; manifest seed = 0)
- Resolved configs: `configs/phase_01/` (22 files) and per-run
  `resolved.toml`
- Run directories: metadata archived under `reports/phase_01/runs/`
  (24 runs)

## Complexity delta

- Production LOC: ~800 added (search.rs ~540, lab.rs ~250, experiment.rs
  ~15), ~300 test LOC added
- Public types: +6 (`Wdl`, `ExactSolver`, `SolvedPosition`, `MoveOrdering`,
  `SearchResult`, `Searcher`), 0 new traits — all required by the phase's
  implement list
- Config keys: +2 (`method`, `ordering`) — both varied and compared above,
  exactly at the phase budget
- Dependencies: 0 added
- New modes retained: `exhaustive` method and `reversed` ordering stay as
  the standing correctness harness — every future game phase must rerun
  these equivalence experiments (plan §14.2–14.3, §24–25); they are
  reference oracles, not rejected experiment residue
- Files: +3 (search.rs, phase_01 configs, this report)

## Failures and anomalies

- 6x4 k4 exact solve aborted: corpus JSONL exhausted local disk at 4.8 GB
  (disk hit 100%; freed after deletion). Conclusion recorded above; if a
  later phase needs bigger oracle corpora, sampling or a compact binary
  format is the remedy — not attempted now (no current need).
- Run-directory collision bug found by its own effects (three experiment
  runs silently overwritten) and fixed with uniqueness suffixes; the
  clobbered configurations were rerun cleanly.
- Symmetric-board ordering equality initially looked like a broken
  experiment; diagnosed as mirror-tree isomorphism (see above).

## Decision

- Promote phase: yes
- Selected configuration: production search = alpha–beta + iterative
  deepening + TT (natural ordering); exact solver = memoized enumeration
- Rejected alternatives: none rejected — comparisons retained as harness
- Reason: 100% value and optimal-action agreement across methods on every
  tested instance; TT/ordering affect only node counts; node budgets and
  determinism verified; solver capacity mapped (oracle rung ≤ 5x4)

## Exact next phase

Phase 2 (supervised oracle ceiling): implement the sparse embedding-sum
policy/value network (Burn, CPU backend), WDL + policy cross-entropy
training on the exact corpus, deterministic training, checkpoint save/load,
and raw-model oracle evaluation. Required experiments: capacity sweep
(four widths), data sweep (four dataset sizes), three seeds on baseline and
best, a tiny-batch memorization test, and held-out generalization, with
WDL accuracy / log loss, optimal-action accuracy and mass, and decision
regret as metrics. The oracle instance will be chosen from the Phase 1
ladder (4x4 k4 g as primary; 5x4 k4 g if capacity allows).

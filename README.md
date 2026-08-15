# CPU-First Game AI Research Lab

A Rust research system for learning deterministic, two-player,
perfect-information, zero-sum board games (Connect-k, Breakthrough,
Othello, standard chess, Forward Chess) on CPU only.

The repository hosts two research programs, kept deliberately separate:

1. **Program 1 (completed baseline)** — zero-knowledge self-play
   (Expert Iteration, generic alpha–beta, raw sparse embedding-sum MLP),
   governed by
   [`minimal_cpu_first_self_play_game_ai_research_plan.md`](minimal_cpu_first_self_play_game_ai_research_plan.md).
   Phases 0–9, reports in [`reports/`](reports/) (`phase_*.md`). Its
   engines, solvers, and match protocols are the frozen baselines.
2. **Program 2 (current)** — structured-heuristic, search-distillation
   research: hand-designed *measurements*, learned *strategy* (compact
   structured evaluators, search distillation, learned move ordering and
   search control), governed by
   [`structured_heuristic_search_distillation_game_ai_research_program.md`](structured_heuristic_search_distillation_game_ai_research_program.md).
   Stage reports live in [`reports/shsd/`](reports/shsd/), starting with
   the [Stage A audit](reports/shsd/stage_a_audit.md) that maps which
   Program 1 components are reused.

## Building and testing

```bash
cargo build --release
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
```

## Usage

**Full user guide: [tools/README.md](tools/README.md).** The
highlights:

```bash
tools/fc_train.sh my_campaign            # chunked, resumable Forward Chess training
tools/fc_train.sh my_campaign --status   # progress; Ctrl-C safe, rerun to resume
tools/fc_rating.py --campaign campaigns/my_campaign   # approximate internal Elo curve
lab play --checkpoint campaigns/my_campaign/champion  # play the champion interactively
lab bench --checkpoint campaigns/my_campaign/champion --movetime-ms 2000  # nodes/s + depth on one core
tools/run_match.sh anchor ckpt:<dir> sf-nodes:20 150 st=0.3  # chess vs Stockfish (§27)
lab solve configs/phase_08/solve_small.toml           # exact solving + tablebase backup
tools/check.sh                           # fmt + clippy + full test suite
lab relabel configs/shsd/stage_b/relabel_fc_tiny_zero_s1.toml  # deep-search teacher records (SHSD)
lab evaluate/train/selfplay/teacher/sweep <config.toml>  # raw lab commands
```

Every `lab` subcommand has `--help`; every script has `-h`.
Configurations live in `configs/`. Every run creates a self-contained
directory under `runs/` with the fully resolved configuration, an
environment manifest, metrics, and game records; training campaigns
keep resumable snapshots under `campaigns/`. All of it is
deterministic given the config — thread count only changes wall-clock
time.

## Status

| Phase | Result | Report |
|---|---|---|
| 0 — Deterministic foundation | **PASS** | [reports/phase_00.md](reports/phase_00.md) |
| 1 — Exact solving and search correctness | **PASS** | [reports/phase_01.md](reports/phase_01.md) |
| 2 — Supervised oracle ceiling | **PASS** | [reports/phase_02.md](reports/phase_02.md) |
| 3 — Self-play recovery of exact strategy | **PASS** | [reports/phase_03.md](reports/phase_03.md) |
| 4 — Breakthrough curriculum | **PASS** | [reports/phase_04.md](reports/phase_04.md) |
| 5 — Othello benchmark | **PASS** | [reports/phase_05.md](reports/phase_05.md) |
| 6 — Standard chess integration | **PASS** | [reports/phase_06.md](reports/phase_06.md) |
| 7 — Measurable chess strength | **PASS** | [reports/phase_07.md](reports/phase_07.md) |
| 8 — Forward Chess rules | **PASS** | [reports/phase_08.md](reports/phase_08.md) |
| 9 — Forward Chess learning | in progress | [reports/phase_09_progress.md](reports/phase_09_progress.md) |

### Program 2 (SHSD) progress

| Stage | Result | Report |
|---|---|---|
| A — Recover and audit the existing system | **PASS** | [reports/shsd/stage_a_audit.md](reports/shsd/stage_a_audit.md) |
| B — Research instrumentation | **PASS** | [reports/shsd/stage_b_instrumentation.md](reports/shsd/stage_b_instrumentation.md) |

Stage B delivered provenance-typed deep-search teacher records
(`lab relabel`), validated against the exact fc-tiny oracle across 3
seeds and two evaluators (monotone budget→accuracy, paired model
comparison, ordering-error metrics, 100% oracle join), the
parameter-provenance ledger (`parameter_ledger.json`), and the frozen
evaluation-set registry (`datasets/frozen/`).

### Program 1 state (2026-08-15, frozen baseline)

Phase 9 is underway: **tabula-rasa full-board Forward Chess reached
+476 relative pool Elo in 56 generations** (400-node campaign to
+360, then an 800-node fork through the plateau; bootstrap CIs ±~90;
pool-relative only, never human-comparable). Search scaling is
monotone through 64× (0.988); w32 reproduces its compute-efficiency
win from chess. Deep-search self-play also flushed out a real rules
bug the differential tests could not see (D031: Black's rotated king
home is d8; stale castling rights + a u16 file wrap let a "castle"
capture the king) — fixed, regression-tested, and all tainted results
re-run. Training now runs through resumable/forkable campaign tooling
with cached Elo-curve evaluation (`tools/README.md`). The measured
compute ↔ strength profile — nodes/s and depth per core, the
2-second-move budget, memory, Elo per search doubling, training cost
per Elo, NNUE headroom — is in
[`reports/compute_strength_profile.md`](reports/compute_strength_profile.md).

Phases 0–8 complete and promoted. Phase 8 brought **Forward Chess in,
exactly**:

- `FORWARD_CHESS_RULES.md` implemented verbatim in `games::forward_chess`
  (directional attacks, orientation-reversing promotion), differential-
  tested against an independent reference generator on all rulesets.
- Reduced instances **solved**: tiny 3×4 and small 4×4 are both
  game-theoretic **draws** (84k and 46.5M reachable positions); every
  solve writes a compact, checksummed, write-then-verified tablebase
  (D029).
- The frozen stack ran unchanged: oracle-evaluated self-play reaches
  searched@600 test accuracy 0.967 (tiny) / 0.9755 (small); raw-policy
  coverage stall reproduced on a third game family (the open finding).

Phase 7 delivered the **first honest chess strength
measurement** (all Elo protocol-relative, never human/FIDE):

- Headline anchor: **10.33% over 300 games vs full Stockfish 17.1
  limited to 20 nodes/move** (−375 ± 43 logistic Elo under this
  protocol); 3.5% vs the UCI_Elo-1320 anchor; 99% vs random.
- Search scaling monotone across four budgets (0.07 → 0.935 vs the
  400-node control); data scaling strong early (+200 Elo gen 3→7) then
  an honestly-reported plateau; **w32 is the width optimum** at this
  compute scale, with the classic fixed-node → fixed-time inversion
  measured (w128 collapses at fixed time).
- Lazy-head inference split tripled self-play throughput; the first
  Stockfish anchor round was invalidated by 1–2 ms time-forfeits and
  re-run with a time margin — termination tags are now always checked.

Phase 6 integrated chess + UCI + the quarantined teacher diagnostic. **Chess is integrated end-to-end**:

- cozy-chess rules behind the unchanged Game trait (perft-exact,
  differential-tested, threefold + fifty-move draws), UCI engine that
  passes fastchess tournaments with zero crashes/illegal moves/timeouts.
- Tabula-rasa self-play works at chess scale: champions reach 0.900 vs
  generation 0; games shorten from 265-ply shuffle-draws to 114-ply
  decisive play; raw policy scores just 0.050 against its own 400-node
  searched self — search is the engine of strength.
- Stockfish diagnostic ceiling (teacher-assisted, quarantined): 80.1%
  teacher-WDL accuracy, 2.1x chance move agreement from 27k positions.

Phase 5 (Othello) demonstrated a second positional style. **Two independent styles of
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

Next: Phase 8 — Forward Chess rules and reduced exact games: the novel
target game's rules module and exactness ladder under the same
curriculum machinery.

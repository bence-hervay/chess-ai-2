# Decision log

Dated, append-only. Each entry records a binding technical decision and why.

## 2026-08-14 — D001: Move representation for Connect-k is the destination cell

Moves are destination cell indices even with gravity on. Legal-move
generation restricts gravity games to the lowest empty cell per column, and
`action_id` maps gravity moves to their column (7 stable actions for
Connect Four) and no-gravity moves to the cell index, matching the plan's
stable-action-ID rule. This keeps one move type across both modes.

## 2026-08-14 — D002: Per-game RNG streams keyed by (run_seed, pair, slot)

Each arena game builds a fresh `ChaCha12Rng` seeded with
`splitmix64(splitmix64(run_seed) ^ splitmix64(2*pair + slot))`. Streams never
depend on worker identity, so game records are byte-identical for any thread
count. Verified by a test running the same arena on 1 and 4 threads.

## 2026-08-14 — D003: Terminal detection is incremental with a differential oracle

`make_move` computes the outcome by scanning only the four lines through the
placed stone; `outcome()` is O(1). A slow full-board scan lives in the test
module as the correctness oracle (plan §14.2), not in production code.

## 2026-08-14 — D004: No third-party crates for time/CPU probes

Run manifests need timestamps, CPU seconds, and peak RSS. Rather than adding
`chrono`/`libc`, `experiment.rs` implements a 20-line civil-from-days UTC
formatter and reads `/proc/self/stat` (USER_HZ = 100 per proc(5)) and
`/proc/self/status`. Linux-only, matching the target environment.

## 2026-08-14 — D005: serde_json alongside toml

The plan mandates TOML configs but JSON manifests/metrics (`manifest.json`,
`metrics.jsonl`, `summary.json`). These are different problems (config vs
machine-readable run artifacts), so this does not violate the "no two crates
for the same problem" rule.

## 2026-08-14 — D006: Score convention and exactness argument for search

Forced win at ply p scores `32000 - p` (shorter wins preferred), draw 0,
leaf evaluations clamped to ±1000. Mate-distance is a function of position
(not path), so TT storage with the standard ply adjustment stays exact.
Alpha-beta returns the same root value as pure minimax for any leaf
scoring; with full-depth search all values are true game values. Verified
against exhaustive negamax on 90 sampled states across three instances.

## 2026-08-14 — D007: Exact solver is pruning-free; corpus keyed by Zobrist

The exact solver is memoized full-enumeration negamax over WDL categories
(memo-safe, no path dependence, no win-cutoff), because corpus generation
needs every reachable state solved anyway. Dedup by 64-bit Zobrist key is
validated against full-state dedup on tic-tac-toe (4,520 states, zero
collisions); the collision risk for larger instances is ~n²/2⁶⁵ and
accepted.

## 2026-08-14 — D008: Oracle rung capped at 5x4 k4 gravity

6x4 k4 exact solving exhausted local disk (corpus JSONL > 4.8 GB) and was
abandoned. 5x4 k4 g (3.1M non-terminal states, 396 MB corpus, 8.8 s solve,
143 MiB RSS) is the largest oracle instance within the plan's "modest
fixed budget". 4x4 k4 g (134k states) is the primary Phase 2 instance.

## 2026-08-14 — D009: `exhaustive` method and `reversed` ordering are kept

They are the standing correctness harness that every future game phase
must rerun (plan §14.2-14.3), not rejected experiment modes. Natural
ordering with TT-move-first remains the sole production path.

## 2026-08-14 — D010: Burn 0.21 (ndarray + autodiff) as the ML framework

Plan-mandated (§10.6). Features limited to std/ndarray/autodiff; no GPU
backends, no burn-train high-level loop (ours is ~60 lines and must be
exactly reproducible). Latest stable at adoption time; MSRV compatible.

## 2026-08-14 — D011: Training recipe v1

Adam, constant lr 1e-3, batch 256, L = CE(WDL) + CE(policy), policy
target uniform over optimal actions, illegal actions masked at -1e9,
single-threaded runs. Constants live in training.rs and change only via
named experiment. Chosen defaults, not tuned: Phase 2 sweeps showed
monotone scaling and seed stability, so no tuning experiment was needed.

## 2026-08-14 — D012: Burn's backend RNG is process-global

Model init draws from a global RNG; determinism holds per-process, not
per-thread. Consequences: training runs are one-per-process (sweep-level
parallelism), and tests that seed the backend serialize on a shared lock.

## 2026-08-14 — D013: Sweep scheduler admitted (plan §16.4)

Phase 2's capacity/data/seed sweeps are the first meaningful sweep, so
`lab sweep` exists now: JSONL manifest {command, config, cores},
CPU-slot scheduling, fail-loud summary. Manifest lines carry an explicit
command instead of config-type sniffing.

## 2026-08-14 — D014: 4x4 k4 g is the Phase 2+ oracle instance; w128 reference

134k states train in ~1-4 min at useful quality (99.4% test WDL accuracy
at w128/40k steps). 5x4 (3.1M states) remains available as a later
stress instance. w128 is the reference capacity; w64 the fast variant.

## 2026-08-14 — D015: Custom sparse inference path (plan §10.6 contingency)

Profiling showed Burn batch-1 inference at 63.6 us/forward (w128) — >95%
of projected self-play cost. CompiledNet extracts weights to plain
row-major arrays; 8-accumulator vectorized dots give 6.4 us (10x). A
mandatory test validates it against the framework to 1e-4 on 60 states.
Framework path remains canonical for training and batched evaluation.

## 2026-08-14 — D016: FIFO replay window (plan §12.5 escape clause)

Current-generation-only training measurably forgot across generations
(candidate val regret oscillated ±0.01 and the promotion gate halted
runs while exploitability was still improving). One fixed FIFO window,
`replay_generations = 4`, is now part of the self-play config. No other
replay machinery.

## 2026-08-14 — D017: Promotion rule v2

Promote when candidate val regret <= champion regret + 0.005 (fixed
noise tolerance); two consecutive rejections halt the run (§12.6 stop
and diagnose). The strict lexicographic rule halted on metric jitter at
plateau. Halts remain honest convergence signals, not errors.

## 2026-08-14 — D018: Exploration epsilon frozen at 0.10

Calibration (3 eps x 3 seeds, w64/400 games/600 nodes): raw-model
recovery improves monotonically with eps (0.20 best), but final
exploitability is unstable at 0.20 (drops 7/14/6 per seed) and reliably
best at 0.10 (4/3/5, searched accuracy 99.3-99.8%). Chose reliability
of exact-strategy recovery per §12.4; eps = 0.10 frozen.

## 2026-08-14 — D019: Search leaf value = WDL expectation x 1000

Leaf score = round(1000 * (P(win) - P(loss))), clamped to the eval
range; policy ordering = TT move, then descending policy logits (stable
sort keeps action-ID order on ties), then Reversed degradation if
configured. Search with too small a budget for one iteration returns
the first ordered move (the policy argmax), making node budget 1 behave
as raw-policy play.

## 2026-08-14 — D020: Breakthrough ruleset

Selected rules: pawns move one square straight or diagonally forward;
diagonal moves may capture, straight moves never; reaching the
opponent's home rank wins; a side with no legal move (subsuming loss of
all pawns) loses. No draws are possible (pawns only advance). rows =
initial pawn ranks per side is a game parameter (rows=1 for small exact
boards, rows=2 standard).

## 2026-08-14 — D021: Match-based promotion and probes for non-exact games

`lab selfplay` gains promotion = "oracle" | "match" (§12.6): match mode
plays paired games vs the frozen champion from shared random openings
(promotion_pairs pairs, opening_plies uniform-random plies, colour
swap); promote only when the candidate's 95% LCB exceeds 0.5. Champion
progression is measured per generation against the frozen generation-0
baseline. `lab evaluate` kind match_probe plays a checkpoint at several
budgets against itself at a baseline budget (search-scaling evidence
without an oracle). play_paired_match takes asymmetric budgets.

## 2026-08-14 — D022: Breakthrough exact ladder capped at 4x5 rows=1

Measured: 3x5r1 = 30k states (0.07s), 4x4r1 = 164k (0.5s), 4x5r1 = 1.8M
(8s, 75MB). 4x5r2 exceeded a 10-minute budget and multi-GB corpus;
aborted and deleted. Strategic (non-exact) size: 5x5 rows=2.
exploitability_vs_perfect now pre-solves once and clones the memo per
game (ExactSolver is Clone) so large exact instances stay probe-able.

## 2026-08-14 — D023: Othello ruleset and explicit pass moves

Selected rules: bracket-and-flip placements in 8 directions; a player
with no placement passes; the game ends when neither player can place
(disc majority wins, equality draws). The pass is an explicit move in
`legal_moves` — an auto-pass inside `make_move` would break negamax's
strict-alternation perspective flip. Passes cannot repeat positions
(two consecutive passes is terminal). Features are board occupancy
only, per §25 restrictions. `match_probe` gained a required
`opponent_checkpoint` field for cross-run champion comparisons (the
data-scaling axis on non-exact sizes).

## 2026-08-14 — D024: Chess wrapper design

cozy-chess 0.3 is the rules backend (plan-mandated). The wrapper adds a
hash history that resets on halfmove-clock resets for threefold
detection; draws = stalemate + fifty-move (via Board::status) +
threefold. Undo restores a full board copy (Board is small). Known
standard approximation: position_key is the board hash only, so TT
entries ignore repetition context. Features: 768 piece-square (side-to-
move rank-mirrored) + 4 castling + 8 en-passant files = 780. Actions:
from x to (4096, queen promotions included) + 72 underpromotion slots =
4168, perspective-mirrored. cozy-chess castling is king-takes-rook; UCI
translation lives in games::chess::parse_move_text.

## 2026-08-14 — D025: UCI engine behavior

Synchronous single-thread search: `stop` is a no-op; `go` without
limits (or `go infinite`) gets a 1 s default deadline so the engine
always answers. Time allocation: (time/30 + inc/2) x 0.8, floor 10 ms.
The search deadline is checked every 128 nodes (model evaluators cost
~0.2 ms/node; the original 4096-node interval lost all 20 games on
time in the first fastchess run).

## 2026-08-14 — D026: Stockfish diagnostic protocol (teacher-assisted)

`lab teacher`: corpus = seeded random legal trajectories sampled at 15%
per ply; labels = Stockfish 17 at fixed depth 8 (cp and bestmove); WDL
rule: win > +100 cp, loss < -100 cp, else draw; policy target = the
teacher's best move; recipe v1 training; 90/10 held-out split. The
checkpoint is never used as the tabula-rasa champion.

## 2026-08-14 — D027: Match-protocol lessons (Phase 7)

Initial SF-anchor calibrations were invalid: Stockfish overshoots
fastchess `st=` movetime by 1-2 ms and forfeited every game, making a
w64 MLP engine appear to beat UCI_Elo 3190. All matches now run with
`timemargin=100`; anchor results are only reported from post-margin
matches, and every reported match records termination causes. Honest
bracket at st=0.3: ~3% vs SF UCI_Elo 1320, ~6-19% vs full SF at 5-20
nodes (NNUE evaluation is that strong per node). Elo language follows
§27: scores and Fastchess logistic differences under this exact
protocol only.

## 2026-08-14 — D028: Forward Chess implementation decisions

- FORWARD_CHESS_RULES.md is authoritative; the module implements it
  exactly. Reduced rulesets are named (tiny/small/medium/full) with
  fixed rotationally-mirrored layouts, not free parameters.
- Pawn double-steps may not land on the promotion rank (uniform rule
  that keeps reduced boards coherent); double-steps and hence en
  passant thus exist only where geometry allows (medium/full).
- Forward Chess repeats, so the acyclic ExactSolver is guarded off;
  reduced instances are solved by `solve_retrograde`: forward
  reachability + backward induction, repetition-as-draw for
  unresolvable cycles, fifty-move rule not modelled in the oracle (the
  standard tablebase caveat). Cross-validated against ExactSolver on
  acyclic games and against real-rules optimal play-outs.
- One production movegen + an independent slow reference generator in
  tests (all-pairs predicate) satisfies the §28 slow/optimized
  sequence within the simplicity contract; the differential test runs
  all four rulesets.
- Rung sizing history: `small` was 4x5 (K+R+2P), then 4x4 and 3x4 when
  the 7.7 GB machine could not hold the position graph. Retried at 4x5
  after the RAM upgrade to 32 GB (cap 60M positions, ~330 B each), but
  4x5 measures at more than 60M reachable positions, so `small` is
  finally 4x4 K+R+P (mirrored), as FORWARD_CHESS_RULES.md §12 records.

## 2026-08-14 — D029: Retrograde artifacts (tablebase + sampled corpus)

The first `small` solve runs filled the 30 GB disk with the full JSONL
corpus, losing an in-flight edit to a swallowed write error (unclosed
file handle on a full disk). After the disk upgrade to 64 GB:

- Every retrograde solve writes `tablebase.bin`, a compact binary
  backup of the full solution: 16-byte header (magic/version/ruleset/
  dimensions/count), one record per position in discovery order (cells
  at 5 bits each, flags byte, optional en-passant byte, halfmove byte),
  FNV-1a-64 checksummed footer. Repetition history is path-dependent
  and never stored; key and outcome are recomputed on load. The reader
  validates structure (codes, padding, king counts, en-passant
  geometry) and checksum before trusting records — corrupt or
  adversarial input errors, never panics (prefix/bit-flip/fuzz tests).
  Written tablebases are immediately re-read and compared record by
  record against the in-memory solution.
- `corpus.jsonl` is capped at ~1M rows, subsampled deterministically by
  `splitmix64(position_key) % denominator == 0` past the cap; summary
  records the denominator and full WDL counts over all non-terminal
  positions.
- Self-play oracle promotion on forward chess builds its evaluation
  dataset from the retrograde solver (val/test buckets thinned to
  ~20k states each); `lab train` and oracle probes stay guarded off
  for loopy games.
- Solved instances: `tiny` root Draw, 83,947 reachable positions
  (0.5 s, 30 MiB). `small` root Draw, 46,549,591 reachable positions
  (44.8M non-terminal: 11.2M win / 28.4M draw / 5.2M loss; 560 s,
  15 GB peak, 559 MB tablebase) — under the 60M cap that 4x5
  exceeded, validating the D028 rung sizing.
- Selfplay probe soundness: `exploitability_vs_perfect` and the
  candidate enumeration inside `searched_decision_metrics` use the
  acyclic ExactSolver, which on loopy games is unsound (path-dependent
  repetition memoization) and can return empty optimal-move lists —
  this crashed the first fc-small selfplay mid-run, and invalidated
  the first fc-tiny run's searched/exploit numbers (both runs deleted
  and re-run). Forward chess selfplay now draws searched-decision
  candidates from the retrograde solution with identical bucket and
  selection-hash semantics (`retrograde_searched_candidates`), and
  reports exploitability as null: a perfect-opponent play-out would
  need the full solution resident for the whole run (~10 GB for
  small), which is not worth one metric. Oracle regret metrics were
  always retrograde-based and needed no change.

## 2026-08-14 — D030: Match-mode strike rule is UCB-based (Phase 9)

§12.6 halts a run after two consecutive regressions. The original
match-mode strike criterion (candidate score < 0.5) is noise-blind: at
a plateau, a candidate truly equal to the champion scores below 0.5
half the time, so two consecutive sub-0.5 draws — probability ~1/4 per
window — halt a healthy run. This fired on the first fc-full campaign
(candidates 0.433 and 0.475 over 60 games; UCB95 ≈ 0.54 and 0.58 —
indistinguishable from equal strength), and only by luck not earlier
(chunk 1 gen 5 scored 0.367, UCB95 ≈ 0.49, a genuine strike).

Strike criterion now mirrors the promotion gate symmetrically:
promote on `score_lcb95 > 0.5` (proof of improvement), strike on
`score_ucb95 < 0.5` (proof of regression), keep the champion
otherwise. True regressions still halt; plateau noise no longer does.
A plateaued campaign now simply finishes its chunk budget without
promotions, which the per-generation log and the rating curve make
visible. Oracle-mode strikes are unchanged (regret tolerance, D-rule
v2). Related campaign-boundary fix in the same investigation: the FIFO
replay window persists across chunks (`replay.jsonl` + `init_replay`);
without it a continuation chunk's first candidates train on a single
generation of data (~26 epochs on ~19k positions) and genuinely
regress.

## 2026-08-15 — D031: Black castles from d8; movegen defense-in-depth

Deep-search self-play on `full` crashed with a captured king. Root
cause chain: the 180° layout rotation puts Black's king on **d8** (e1
rotated), but rights-clearing assumed both kings home on file
`width/2` — so Black king moves never cleared Black's rights; a
rights-holding king reached b8; queenside castle generation computed
destination file 1−2 = −1, whose `as u16` cast wrapped through
`cell()` arithmetic to h7 — "castling" onto (and capturing) White's
king. Random-playout differential testing never caught it because
random games shed castling rights almost immediately; 800-node search
found it within ~50 generations of training.

Fixes, each independently sufficient to prevent king capture:
`king_home(owner)` derives the true rotated home (rights now die with
king moves for both colours); castle generation requires the king on
its exact home square and bounds-checks files before any u16 cast; and
`make_move` permanently asserts both kings survive every move,
panicking with a rendered board dump (the instrumentation that caught
this). Corpus gains black-castles-from-d8, rights-die-with-king-moves,
and phantom-castle-with-forced-stale-rights tests; RULES.md §12 now
spells out the d8 home. All fc-full training results produced before
this fix (first fc_full_w64 campaign and its n800 fork) are tainted by
phantom black castling and were discarded and re-run.

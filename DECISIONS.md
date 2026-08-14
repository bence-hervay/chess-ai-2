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

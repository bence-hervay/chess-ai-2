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

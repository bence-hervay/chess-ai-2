# Lab user guide

Everything runs from the repo root on the release binaries
(`cargo build --release` produces `target/release/lab` and
`target/release/uci`). Every `lab` subcommand has `--help`; every
script here has `-h`. All experiment outputs land under `runs/`
(git-ignored) with a `manifest.json`, `resolved.toml`, `summary.json`,
and streaming `stdout.log` per run.

**Determinism:** all lab machinery is deterministic given the config —
per-game RNG streams are keyed by `(seed, generation, game_index)` (or
`(seed, pair, slot)` in arenas/matches), so results are independent of
thread count, and reruns with the same config reproduce byte-identical
metrics. Threads only change wall-clock time. Default to
`threads = 7` on this 8-CPU machine (one core left for the OS/driver).

---

## Training campaigns (interruptible + resumable)

```
tools/fc_train.sh <name> [options]          # create or resume
tools/fc_train.sh <name> --status           # progress at a glance
tail -f campaigns/<name>/log.txt            # live monitoring
```

`fc_train.sh` runs Expert Iteration in **chunks** of generations; each
chunk is a normal `lab selfplay` run whose champion seeds the next
chunk (`init_checkpoint`). Interrupt at any time (Ctrl-C, VM restart —
anything): re-running the same command resumes from the last completed
chunk. A chunk killed mid-run is re-run from its start; completed
chunks are never touched.

Three ways to keep training without retraining from scratch:

- **Interrupted?** Rerun the same command — it resumes.
- **Finished, want more of the same?** `--chunks N` raises the target;
  the next chunk picks up the champion and its replay window.
- **Finished, want a different recipe** (deeper search, more games,
  new epsilon, ...)? Fork it:
  `tools/fc_train.sh <new-name> --from campaigns/<old> --nodes 800`.
  `--from` takes a campaign, a chunk snapshot, or a bare checkpoint;
  chunk 1 of the new campaign continues from that champion (plus its
  replay window when available). Width and game must match. Each
  campaign's recipe stays frozen, so provenance is never mixed; the
  fork's `baseline_gen0` anchor is the fork point, and
  `fc_rating.py --add parent=campaigns/<old>/champion` links the two
  curves in one pool.

Defaults are the Phase 7-calibrated recipe: full-board Forward Chess,
width 64, 6 chunks × 8 generations × 200 games, 400-node expert
search, match promotion (30 pairs, LCB gate), ε = 0.10, replay 4.
Chunk *c* uses `seed + c − 1`.

Snapshots (the campaign's backbone, and the rating-pool members):

```
campaigns/<name>/
  campaign.env        frozen parameters (resume reads these)
  log.txt             per-generation lines + chunk summaries
  baseline_gen0/      untrained random-init net = Elo anchor
  chunk_001..N/       config, summary, metrics, checkpoint/ per chunk
  champion/           latest champion (use for play/matches)
```

The FIFO replay window persists across chunks (`replay.jsonl` +
`init_replay`), so a chunked campaign trains like one long run —
without this, the first generations of a continuation chunk train on a
single generation of data and regress against the replay-trained
champion (observed on fc-full before the fix). The remaining per-chunk
approximation: each chunk's `vs gen0` progression line is against the
*chunk's own* starting champion; cross-chunk progress is what
`fc_rating.py` measures properly.

Watch for during a run: `promoted` vs `REJECTED` per generation, match
scores, mean plies, and generation wall-times (≈700% CPU during
self-play, 100% during the deterministic single-threaded training
segments). Two consecutive *regressions* (candidate score < 0.5) halt
the run by design (plan §12.6) — the driver then pauses the whole
campaign with a diagnosis pointer instead of blindly relaunching;
resuming is an explicit decision.

## Rating curve (approximate internal Elo)

```
tools/fc_rating.py --campaign campaigns/<name>
tools/fc_rating.py --campaign campaigns/<name> --round-robin   # denser
```

Plays deterministic paired matches between snapshots (gauntlet vs the
latest champion + the adjacent chain; results cached, so re-running
after new chunks only plays the new pairings), fits relative Elo by
maximum likelihood with bootstrap 95% CIs, and writes
`campaigns/<name>/rating/ratings.{md,csv}` plus an ASCII curve.

Ratings are **relative engine-pool Elo under this exact protocol** —
never comparable to human/FIDE Elo, and Forward Chess numbers are
never comparable to chess numbers (plan §29).

## Interactive play (Forward Chess)

```
./target/release/lab play                                  # 8×8, unlearned search
./target/release/lab play --checkpoint campaigns/<name>/champion --nodes 1200
./target/release/lab play --game fc-small --side black
```

Moves are coordinates (`a2a3`, `a7a8=Q`); `?` lists legal moves,
`hint` asks the engine, `undo` takes back a full move, `quit` exits.
`~` after a piece letter marks reversed orientation (promoted pieces).
`--side none` watches engine vs engine.

Standard chess is played through the separate UCI engine instead:
`./target/release/uci` works in any UCI GUI
(`setoption name Checkpoint value <dir>` points it at a model;
without one it plays unlearned search; `uci --random[=seed]` is the
uniform random baseline).

## Chess vs Stockfish (§27 protocol)

```
tools/run_match.sh <name> <engine1> <engine2> <pairs> <limit> [seed]
# engine spec: ckpt:<dir> | zero | random:<seed> | sf-elo:<n> | sf-nodes:<n>
# limit:       tc=<t>+<inc> | st=<sec> | nodes=<n>
tools/run_match.sh anchor ckpt:campaigns/chess/champion sf-nodes:20 150 st=0.3
```

Runs fastchess under the frozen protocol: one thread per engine,
ponder off, no tablebases, fixed hash, shared symmetric opening suite
(`tools/openings_4ply.epd`), colour-swapped pairs, fixed seed,
`timemargin=100` (Stockfish overshoots `st=` by 1–2 ms — without the
margin it forfeits everything on time; always check the termination
tags in the saved PGN before believing a result — see D027). Needs
fastchess at `~/tools/fastchess/fastchess`; the script prints install
instructions if missing. Stockfish itself: `/usr/games/stockfish`.

## Exact solving and tablebases

```
./target/release/lab solve configs/phase_08/solve_tiny.toml
./target/release/lab solve configs/phase_08/solve_small.toml   # ~10 min, 15 GB
```

Retrograde solves write a compact checksummed `tablebase.bin`
(write-then-verified record by record) and a ≤1M-row hash-sampled
`corpus.jsonl` (D029). Solved so far: tiny and small are both draws.

## Testing

```
tools/check.sh            # fmt + clippy + full release test suite
tools/check.sh --quick    # fmt + tests
```

Run before every push. The suite includes the Forward Chess rules
corpus, the movegen differential tests, retrograde solver
cross-validation, and the tablebase corruption/fuzz tests.

## Evaluating single checkpoints

- `lab evaluate` with `kind = "match_probe"`: one checkpoint vs
  another (or vs itself at a different node budget — the §29
  search-scaling sanity test: probe `node_budgets = [n, 4n, 16n, 64n]`
  against `baseline_nodes = n`).
- `lab evaluate` with `kind = "oracle_probe"`: exact-oracle metrics
  for acyclic games (guarded off for Forward Chess; use the selfplay
  oracle mode on fc-tiny/fc-small instead).
- `lab sweep <manifest.jsonl>`: run many configs with CPU-slot
  scheduling (lines of `{"command","config","cores"}`).

## Disk policy

`runs/` and `campaigns/` are git-ignored working data. When disk
fills: delete the largest, most cheaply reproducible things first
(match PGNs, superseded chunk snapshots, old runs); keep verified
tablebases and campaign champions. `df -h /` before big solves.

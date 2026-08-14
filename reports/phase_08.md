# Phase 8 Report: Forward Chess rules and reduced exact games

## Status

PASS

## Research hypothesis

The full Forward Chess rule set (directional movement *and attack*,
orientation-reversing promotion, horizontal castling, en passant,
repetition) can be implemented exactly from an authoritative
human-readable rules document, validated against an independent
reference generator and a hand-authored rules corpus, and its reduced
instances solved exactly — with the frozen search/model/training stack
running on the new game unchanged.

## Minimal implementation

- **`FORWARD_CHESS_RULES.md`** — authoritative rules (coordinates,
  orientation law, attack restriction, castling, en passant, check
  family, repetition, fifty-move, promotion choices and orientation
  reversal, reduced layouts §12). The module implements this document
  exactly and nothing else (D028).
- **`games::forward_chess`** — packed cell codes
  (piece × owner × orientation), incremental Zobrist keys, one
  production move generator; the independent slow reference generator
  (all `(from, to, promotion)` triples validated by separately written
  directional logic plus self-check simulation) lives in tests,
  satisfying the §28 slow/optimized sequence inside the simplicity
  contract.
- **`solve_retrograde`** — forward BFS over the reachable graph plus
  backward induction; repetition-as-draw for unresolvable cycles;
  fifty-move rule not modelled (the standard tablebase caveat, D028).
  Guarded so the acyclic `ExactSolver` can never run on loopy games.
- **Tablebase backups (D029)** — every retrograde solve writes a
  packed, checksummed `tablebase.bin` (5-bit cell codes; ~12 B per
  position vs ~154 B JSONL) and re-reads it record-by-record against
  the in-memory solution before reporting success; `corpus.jsonl` is
  capped at ~1M rows by deterministic position-key-hash sampling.
- **Retrograde oracle for self-play evaluation** — oracle promotion on
  forward chess builds val/test sets and searched-decision probe states
  from the retrograde solution (identical split-bucket and
  selection-hash semantics as the acyclic path); exploitability
  play-outs are reported as null for loopy games (D029).

## Deliberately excluded

- No separate search or learning implementation for the reduced game
  (plan §28): the same `Searcher`, model, and training recipe run
  untouched.
- No fifty-move modelling in the oracle (documented caveat).
- No perfect-opponent exploitability for loopy games: it would pin the
  ~10 GB solution in RAM for a whole run to keep one metric.
- No full-board Forward Chess self-play (that is Phase 9).

## Correctness evidence

- **Rules corpus** (hand-authored, §28): black queen on d2 attacks
  forward only; castling is horizontal and fully checked; en passant
  correctly timed; promotion flips orientation and attack direction;
  horizontal moves remain legal; king safety under oriented attacks;
  checkmate and stalemate; repetition and fifty-move draws.
- **Randomized differential test**: 250 random reachable states per
  ruleset × all four rulesets, production generator vs the independent
  all-pairs reference — no discrepancy; plus a 300-state × 3-ruleset
  orientation invariant (no generated move ever violates the
  directional law).
- **Retrograde solver**: value-per-position cross-validation against
  `ExactSolver` on acyclic games (Connect-K, Breakthrough);
  deterministic across runs; every position's value equals its best
  child value; optimal-vs-optimal play-out under the *real* rules
  (history, threefold, fifty-move live) realizes the solved root value.
- **Tablebase**: record round-trip across all four rulesets with
  en-passant coverage asserted; file round-trip on the solved tiny
  instance; rejection (error, never panic) of every strict prefix and
  every single-bit flip of a real file, crafted kingless / invalid-WDL
  / count-mismatch records, cross-ruleset loads, trailing garbage; 20k
  random-bytes fuzz. Every production write is verified by re-reading.
- 17 forward-chess tests + 3 retrograde solver tests; full suite 82.

## Global learning or playing result

### Reduced instances solved (rung 1 and 2)

| Instance | Root | Reachable positions | Solve cost |
|---|---|---:|---|
| tiny 3×4 (K+P) | **Draw** | 83,947 | 0.5 s, 30 MiB |
| small 4×4 (K+R+P) | **Draw** | 46,549,591 | 560 s, 15 GB peak |

Small non-terminal split: 11.19M win / 28.42M draw / 5.23M loss
(mover's perspective). The 4×5 two-pawn variant exceeds the 60M
position cap (D028 rung history). Both solutions are backed up as
verified tablebases (tiny 0.8 MiB, small 533 MiB).

### Oracle-evaluated self-play, tiny (12 gens, w64, 400 games/gen)

Raw-net val regret plateaus at ~0.13–0.14 from generation 0 while
training losses fall to zero — the known coverage limitation (raw
self-play distribution collapses onto repeated drawn lines; Phase 4/5
finding). Search repairs it: **searched@600 test accuracy 0.9670,
regret 0.0340**; raw test regret 0.1362. 128 s wall.

### Oracle-evaluated self-play, small (w64, 400 games/gen)

First self-play measured against a 46.5M-position exact oracle in this
project. Gens 0–7 promoted (val regret 0.2368 → 0.2345 at gen 5, then
drifting up); gens 8–9 rejected, and the frozen §12.6 halt mechanism
stopped the run — working exactly as designed on a brand-new game.
Final: **searched@600 test accuracy 0.9755, regret 0.0245**; raw test
regret 0.2377 (the same coverage plateau as tiny, at higher regret on
the harder game). 681 s wall / 1,096 s CPU including the in-run oracle
solve.

## CPU and memory result

- Small solve: 560 s single-threaded, 15.0 GB peak RSS (46.5M
  positions ≈ 330 B each, matching the D028 estimate), tablebase
  write-plus-verify included; artifacts 712 MB total (533 MiB
  tablebase + 178 MiB sampled corpus) vs ~10 GB for the uncapped
  JSONL that filled the old disk.
- Small self-play: startup oracle solve dominates (~9.5 min);
  generations then run at 15–20 s (the crashed pre-fix run spent
  29 min *per generation* re-enumerating the game acyclically —
  precollected retrograde probes removed that entirely).
- Tiny: solve 0.5 s / 30 MiB; 12-generation self-play 128 s.

## Reproducibility

- Configs under `configs/phase_08/` (solve_tiny/solve_small,
  sp_fc_tiny_oracle, sp_fc_small_oracle, sp_fc_tiny_smoke); all
  sampling (corpus rows, eval buckets, probe states) is deterministic
  by position-key hash, independent of thread count.
- Machine: 8 CPUs, 32 GB RAM, 64 GB disk (upgraded this phase from
  30 GB after full JSONL corpora filled it — see D029 and Failures).
- Commits: c9c1ef5 (core; solves and runs executed on it), 4e83865
  (probe soundness fix; re-runs), report added in the phase-completion
  commit.

## Complexity delta

~2,900 insertions, no new dependencies, no new abstractions beyond
`RetrogradeSolution` and the tablebase section:

- `games/forward_chess.rs` 1,939 lines (roughly half tests: reference
  generator, rules corpus, differential, tablebase adversarial suite);
- `search.rs` +232 (retrograde solver + tests);
- `bin/lab.rs` +327 (retrograde solve runner incl. tablebase
  write-verify and corpus sampling, dispatch, guards);
- `training.rs` +71 (retrograde eval dataset builder);
- `evaluation.rs` +62 (searched-metrics split, retrograde candidates).

Audit: the tablebase writer/reader is the only speculative-looking
piece; it is justified as the D029 backup mandate and is exercised by
every retrograde solve (write-then-verify), not dead code.

## Failures and anomalies

- **Disk-full loss of an in-flight edit**: the uncapped JSONL corpus
  of the first small solves filled the 30 GB disk; a Python-scripted
  edit died with a swallowed write error (unclosed handle on a full
  disk), and the stale binary refilled the disk on the next run.
  Response: 64 GB disk, compact verified tablebases, capped corpus
  (D029).
- **Acyclic-solver probes on a loopy game**: the first fc-small
  self-play crashed in generation 1 (`optimal[0]` on an empty
  optimal-move list) because exploitability and searched-decision
  candidate enumeration used `ExactSolver`, whose path-dependent
  repetition memoization is unsound on forward chess; it had also
  silently skewed the first tiny run's searched metrics (0.9575
  reported vs 0.9670 sound). Both runs deleted and re-run after the
  fix (D029).
- **Raw-policy coverage stall** (tiny ~0.14 val regret with loss → 0;
  small ~0.24, ending in a §12.6 two-rejection halt at generation 9):
  expected from Phases 4/5; carried as the open finding into Phase 9.

## Decision

Phase 8 acceptance criteria all hold: rules corpus passes; randomized
differential testing finds no discrepancy; reduced exact instances are
solved (both roots: Draw); model and search ran unchanged; features
encode no positional conclusions; full-board self-play has not begun.
Proceed to Phase 9 (Forward Chess learning and strength, §29).

## Exact next phase

Phase 9 — Forward Chess learning and strength (§29): tabula-rasa
training from random initialization, evaluation ladder anchored by the
reduced-board oracles, internal rating pool.

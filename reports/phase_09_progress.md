# Phase 9 progress — overnight campaign report (2026-08-15)

Status: IN PROGRESS (checkpoint report, not the phase gate; §29 items
still open are listed at the end).

## What ran

Tabula-rasa full-board Forward Chess (8×8), frozen recipe
(200 games/generation, ε = 0.10, replay 4, match promotion LCB > 0.5,
D030 UCB strike), all through the new campaign tooling
(`tools/fc_train.sh`, `tools/fc_rating.py`; see tools/README.md):

1. `fc_full_w64` — w64, 400-node expert, 4 chunks × 8 generations.
2. `fc_full_w64_n800` — fork of its champion at 800-node expert
   (`--from`, inheriting champion + replay window), 3 chunks until
   §12.6 halts.
3. `fc_w32` (2 chunks) and `fc_w128` (1 chunk) — width axis.

## Headline: the Elo curve (relative pool Elo, gen0 = random init)

Combined fit over both campaigns (uniform 400-node matches, 40 pairs
per pairing, bootstrap 95% CIs ≈ ±90; never human-comparable, and
Forward Chess Elo is never comparable to chess Elo):

| snapshot | gens | expert nodes | Elo |
|---|---:|---:|---:|
| gen0 | 0 | — | 0 |
| p1 | 8 | 400 | +201 |
| p2 | 16 | 400 | +303 |
| p3 | 24 | 400 | +364 |
| p4 | 32 | 400 | +359 (plateau) |
| c01 | 40 | 800 | +402 |
| c02 | 48 | 800 | +454 |
| **c03** | **56** | **800** | **+476** |

The 400-node recipe plateaued at ~+360 after 24 generations; forking
the champion to 800-node expert search broke the plateau for ~+117
more. Current best engine: `campaigns/fc_full_w64_n800/champion`.

## Search-scaling sanity (§29)

Champion at 4× / 16× / 64× nodes vs its 1× self: 0.775 / 0.838 /
**0.988** — monotone through 64×. (Pre-D031 rules saturated at 16×;
correct rules search-scale better.)

## Width axis (first pass)

- w32 (16 gens) vs w64 at 16 gens: **0.694** (LCB 0.586) — w32 ahead
  at equal generations.
- w32 (16 gens) vs w64 final (32 gens): **0.625** (LCB 0.529).
- w128 first chunk healthy (3 promotions, 0.842 vs start) but slower
  per generation. The chess Phase 7 finding — w32 is the
  compute-efficiency optimum at this scale — reproduces on Forward
  Chess. Recommendation: next long campaign at w32 with ≥800-node
  expert.

## Late-night addendum: w32 at 800 nodes

`fc_w32_n800` (fork of the w32 champion at 800-node expert) gained one
more chunk (0.683 vs start, 2 promotions), then went flat and halted
honestly under D030 in chunk 5 — w32's capacity ceiling arrives
quickly under deep search, completing the width story: **w32 wins the
early compute race, w64 sustains growth longer**. Best-vs-best,
w32-n800 champion vs w64-n800 champion at 400 nodes: **0.550 (LCB
0.456) over 80 games — statistical parity**, reached by the w32 line
at distinctly lower generation cost. The frozen pool for Phase 9
should carry both lines.

## The D031 find (the night's most valuable result)

Deep-search self-play exposed a rules bug that random-playout
differential testing structurally could not find: under the 180°
layout rotation Black's king starts on **d8** (rotated e1), but
rights-clearing assumed file width/2 for both colours — Black king
moves never cleared castling rights; a rights-holding king on b8
generated a queenside castle whose file 1−2 = −1 wrapped through a
u16 cast to h7, capturing White's king. 800-node search found and
exploited this within ~50 generations; the permanent `make_move`
king-invariant (added as instrumentation, kept as defense) produced
the exact board dump. Fixes: correct rotated king-home, exact-home +
file-bounds checks in castle generation, three new corpus tests,
RULES.md §12 clarification. All pre-fix fc-full results were
discarded; the clean re-run is both stronger (+469 vs +365 parent-pool
peak) and search-scales better — the phantom rules were actively
hurting deep search. Lesson recorded: search-driven self-play is a
rules fuzzer; the king invariant stays.

Also tonight, D030: the match-mode §12.6 strike now requires provable
regression (UCB95 < 0.5) after plateau noise (0.433/0.475 over 60
games) spuriously halted a healthy run. With it, the halts that did
fire were genuine (candidates at 0.275–0.35 with UCB < 0.5): after a
promotion, the 4-generation replay window is dominated by the older
champion's games, so successor candidates can genuinely regress — the
oscillation is the recipe's character at strong-champion regime, and
the gate correctly refuses those candidates while the champion keeps
its gains.

## Infrastructure delivered tonight (tools/README.md)

Chunked interrupt-safe campaigns with continuation (`--chunks`) and
recipe-changing forks (`--from`, champion + replay inheritance);
replay-window persistence (`replay.jsonl` / `init_replay`);
cached-match Elo curves with bootstrap CIs and cross-campaign pools
(`--add`, duplicate-name guard, `--skip-baseline`); halt-aware driver;
interactive play (`lab play`); §27 Stockfish matches at a stable
fastchess path; `tools/check.sh` gate.

## Reproducibility

Campaign parameters frozen in each `campaigns/*/campaign.env`; chunk c
uses seed base+c−1; all sampling and matches deterministic and
thread-count-independent. Rating artifacts:
`campaigns/fc_full_w64_n800/rating/ratings.{md,csv}`. Commits:
c81bf44 (D031 fix) through the commit adding this report.

## Still open for the Phase 9 gate (§29)

Full width/data sweeps with the standard reporting; fixed-time
evaluation; distribution-shift frozen set; adversarial disagreement
corpus; reduced-board oracle trend; formal frozen rating pool
(baseline / first competent / every champion / compute-scaled);
phase report with complexity audit.

# Experiment: J1 — quiescence at standard chess and the anchor re-measurement

## Question

Does capture/promotion quiescence (G1, already implemented for chess
via `is_tactical`) improve the standard-chess engine, and by how much
does it move the frozen Phase-7 Stockfish anchor — the §51.6 gate's
"measurable improvement over the previous baseline under a documented
protocol"?

## Hypothesis

Standard chess is more tactical than Forward Chess, so rung-2
quiescence is worth at least as much as the +114 fixed-time Elo
measured at fc-full: the self-match (same Phase-7 w64 checkpoint,
quiescence on vs off, 0.3 s/move) shows a large win, and the anchor
score vs Stockfish 17.1 @ 20 nodes/move rises clearly above the
baseline 10.33% (−375 ± 43 protocol Elo).

## Smallest implementation

- `uci` accepts a single-token `--qs=<checkpoint-dir>` argument
  (quiescence on; plain `<dir>` stays quiescence-off), re-applied
  after `ucinewgame` re-creates the searcher.
- `tools/run_match.sh` gains the `ckptqs:<dir>` engine spec.
- No search changes: the G1 mechanism is reused untouched.

## Baselines

The Phase-7 w64 anchor engine exactly as measured (same checkpoint,
same protocol: one thread, hash 16 for SF, `timemargin=100`, frozen
250-opening suite, colour-swapped pairs, seed 42, st=0.3).

## Primary metric

Score (and protocol Elo) vs Stockfish 17.1 limited to 20 nodes/move,
300 games at 0.3 s/move, vs the recorded 10.33% baseline.

## Secondary metrics

Self-match quiescence on-vs-off (300 games, same checkpoint both
sides); termination-cause counts (D027 discipline: no time forfeits);
game lengths.

## Fixed resources

fastchess concurrency 3 on 4 vCPUs; ~1.5 h total for both matches.

## Independent variables

Quiescence on/off only (§6.8).

## Controlled variables

Checkpoint (`runs/20260814-131835-selfplay-chess-w64-g200-e10-s1-c03fefc/checkpoint`),
protocol constants, opening suite, seed.

## Seeds

Protocol seed 42 (the frozen convention); single 300-game samples per
match (matching the Phase-7 headline sample size).

## Predicted outcomes

### If hypothesis is supported

Quiescence becomes the chess engine's production mode; the SHSD
program has measurably improved the standard-chess calibration; next
lever is a chess move ranker (F2's mechanism, needs chess move
features).

### If hypothesis is rejected

Self-match win but anchor flat → SF@20nodes punishes something other
than horizon blunders (e.g. mid-game strategy); analyze the PGNs
before touching anything else. Self-match flat → chess-specific
quiescence bug; the fc-full result says the mechanism works, so
suspect `is_tactical` for chess or time management.

### If result is ambiguous

Anchor improvement inside the old ±43 interval: extend to 600 games
before concluding.

## Correctness risks

Time forfeits (D027): the deadline check runs inside quiescence
(verified in G1 code); PGN termination tags checked before reporting.

## Performance risks

Quiescence's eval-heavy nodes halve nodes/s per wall-clock — already
netted out at fc-full fixed time; chess evals are cheaper relative to
movegen, so the cost share is smaller.

## Complexity budget

One argv convention, one script spec. Nothing else.

## Removal criterion

Anchor regression → quiescence stays FC-only; report retained.

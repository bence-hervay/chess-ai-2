# Phase 6 Report: Standard chess integration

## Status

PASS

## Research hypothesis

Chess rules and UCI compatibility can be added to the generic stack —
rules module + protocol binary only — without demanding chess strength
yet; the diagnostic matrix (teacher ceiling, search-vs-raw) separates
representation, optimization, self-play, and search limits at chess
scale.

## Minimal implementation

- `src/games/chess.rs`: `cozy-chess` 0.3 rules backend (plan-mandated
  dependency) behind the unchanged `Game` trait. Repetition support
  (deferred since Phase 1 — the first game that can repeat): hash
  history resetting on halfmove-clock resets; draws = stalemate +
  fifty-move + threefold. Features (allowed §26 facts only): 768
  piece-square (side-to-move rank-mirrored) + 4 castling + 8 en-passant
  files = 780. Actions: from×to (4096, queen promotions folded in) + 72
  underpromotion slots = 4168, perspective-mirrored. cozy's
  king-takes-rook castling is translated to/from standard UCI notation
  in one shared helper. Documented standard approximation:
  `position_key` = board hash (repetition context not keyed).
- `Searcher`: optional wall-clock deadline (§11.4), checked every 128
  nodes, alongside the untouched deterministic node budgets.
- `src/bin/uci.rs`: UCI engine (uci/isready/ucinewgame/setoption/
  position/go/stop/quit; nodes, depth, movetime, wtime+winc time
  allocation `(t/30 + inc/2)·0.8`; limitless `go` bounded at 1 s;
  malformed input ignored on stderr). Optional checkpoint argument;
  without one it searches with the zero evaluator (protocol tests).
- `lab teacher`: the §26 Stockfish diagnostic (D026) — seeded
  random-trajectory corpus, fixed Stockfish-17-depth-8 labels, recipe
  v1 training, held-out teacher agreement. Explicitly teacher-assisted;
  never the tabula-rasa champion.
- `GameSpec::Chess` with explicit guards: exact-solve, oracle-mode
  self-play, oracle training, and oracle probes all reject chess with
  clear errors (not exactly solvable).

## Deliberately excluded

Quiescence and all speculative pruning (§11.5–11.6 — the horizon
measurements belong to the standard-chess strength phase), Elo claims
(§26 forbids them at this stage), bitboard-native feature extraction,
pondering/async stop, insufficient-material adjudication (fifty-move
and threefold suffice for termination), the split lazy-head inference
optimization (identified, ~25–35% projected; no phase requirement).

## Correctness evidence

63 tests (fmt/clippy/debug/release clean). Chess-specific, per §26's
required lists:

- **perft**: five established positions (startpos to depth 4 =
  197,281; Kiwipete 97,862; positions 3–5) all exact;
- **randomized differential vs cozy-chess**: move lists identical on
  60 random games × 120 plies; make/unmake restores state and hash
  exactly;
- check/checkmate/stalemate outcomes; castling (both sides), en
  passant capture, all four promotions with distinct action IDs;
- **threefold repetition** and **fifty-move** detected exactly;
- colour-mirrored positions encode to identical features and action
  IDs;
- **search**: mate-in-1 and mate-in-3-plies found with exact mate
  distances (TT on and off); the deterministic rook-counting test
  evaluator makes search claim a repetition draw when losing;
  node-budget and TT invariants inherited from Phase 1 suites;
- **UCI compliance** (integration test driving the real binary):
  handshake, malformed/unknown commands ignored alive, standard
  castling notation accepted, bounded default search, `bestmove 0000`
  on terminal positions, clean exit.

## Global learning or playing result

### Tabula-rasa self-play at chess scale (match promotion, w64, 400 nodes, 8 generations)

- The loop runs end-to-end at chess scale: 200 games/generation,
  ~53k → 32k positions per generation as games shortened from a mean of
  **265 plies (random shuffle-draws) to 114 plies (decisive play)**.
- Champion progression vs generation 0: 0.75 → 0.88 → 0.94; final
  match **0.900** (LCB 0.860) over 120 games. 3 of 8 candidates
  promoted; the LCB gate and regression-only strikes behaved as
  designed.
- **Model plus search ≫ raw model** (frozen champion, self-match): raw
  policy (node budget 1) scores **0.050** against 400-node search;
  100 nodes scores 0.100; the equal-budget control ties exactly
  (26W/26L/28D); 1600 nodes beats 400 with **0.794** (LCB 0.720).
- **Fastchess tournaments** (protocol acceptance): 10 games zero-vs-zero
  and 20 games champion-vs-zero completed with **zero crashes, zero
  illegal moves, zero timeouts** (after the deadline-granularity fix);
  champion scored 12W/8D/0L (80%).

### Stockfish diagnostic ceiling (teacher-assisted, separate)

30,000 positions labelled by Stockfish 17 at fixed depth 8 (19 s at
1,557 pos/s over 7 workers); w128, 20k steps, recipe v1; 3,000 held
out: **teacher-WDL accuracy 80.1%**, **best-move agreement 19.3% vs
9.0% chance** (2.1×). Materially better than chance, far from the
teacher — with 27k training positions the representation is data-
limited, not broken (consistent with the Othello finding that value is
the hard head). This checkpoint is quarantined as teacher-assisted.

## CPU and memory result

Chess inference: ~180 µs/forward (w64) — the 4168-wide action head
dominates; the identified lazy-head split is the next optimization
lever if Phase 7 needs it. Self-play generation ~7 min (200 games, 7
workers); the full 8-generation run took 59 min wall / 5.3 CPU-hours.
UCI search at tc=5+0.1 holds time cleanly after the 128-node deadline
check. Teacher labelling 1,557 positions/s.

## Reproducibility

Commit 2327384 + this phase's commit; cozy-chess 0.3.4 pinned in
Cargo.lock; Stockfish 17 (apt `17-1build1`) at documented fixed depth;
fastchess v1.8.2-alpha; configs `configs/phase_06/`; metadata for the
self-play, teacher, and match-probe runs archived under
`reports/phase_06/runs/` with the champion-vs-zero PGN.

## Complexity delta

- Dependencies: +1 (`cozy-chess`, plan-mandated).
- New code: chess wrapper ~340 production + ~270 test lines; UCI binary
  ~250 lines; `lab teacher` ~230 lines; deadline support ~15 lines.
- Config surface: `GameSpec::Chess {}`; `TeacherConfig` (7 fields, all
  §26-mandated experiment parameters).
- Search/model/training subsystems: the only change is the optional
  search deadline (§11.4, a §26 implement item). Recipe v1 untouched.

## Failures and anomalies

- First fastchess run with the model: **all 20 games lost on time** —
  the 4096-node deadline check interval assumed cheap evaluators
  (~6 µs) while chess forwards cost ~180 µs, so the first check landed
  ~0.7 s past a 216 ms budget. Fixed: 128-node interval + 20%
  allocation margin; re-run had zero timeouts (D025).
- An unbounded `go` (no limits) hung the engine — caught by the UCI
  compliance test; fixed with a 1 s default deadline.
- Early generations produce 250+-ply shuffle-draws (expected for
  tabula-rasa chess); game length falling to 114 plies is itself a
  learning signal.
- `pkill` in a build script matched its own command line and killed the
  shell — harness annoyance, no repo impact.

## Decision

- Promote phase: yes
- Selected configuration: chess rungs use match promotion only (oracle
  paths guarded off); UCI engine with checkpoint argument; teacher
  diagnostic protocol frozen as D026
- Rejected alternatives: auto-pass-style silent repetition keys
  (documented approximation instead); async stop support (synchronous
  engine suffices for fastchess operation)
- Reason: all §26 acceptance criteria met — differential rule tests
  pass, UCI behaves correctly, fastchess completes tournaments cleanly,
  the diagnostic is materially above chance, search decisively
  strengthens the raw model, and no Elo claim is made

## Exact next phase

Phase 7 (measurable standard-chess strength, §27): scaled self-play
generations at chess, fixed-node and fixed-time match protocols against
reference opponents with statistically meaningful match sizes, and the
first defensible strength measurements. The lazy-head inference split
and generation-scale tuning are the expected performance work.

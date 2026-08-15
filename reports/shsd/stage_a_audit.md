# SHSD Stage A — Recover and audit the existing system

Program: `structured_heuristic_search_distillation_game_ai_research_program.md`
(the "SHSD program"). This is the first stage report of the new program;
all SHSD artifacts live under `reports/shsd/` and are deliberately kept
separate from the phase reports of the earlier zero-knowledge program
(`minimal_cpu_first_self_play_game_ai_research_plan.md`, phases 0–9).

Date: 2026-08-15. Audit revision: e10550b (clean tree + the program doc).

## Status

PASS — the existing system is recovered, reproducible, and measured; the
reuse map and Stage B instrumentation plan are below.

## 1. What the previous system is

One Rust package (`selfplay-lab`, ~11.5k LOC) implementing Expert
Iteration with a generic alpha–beta search and a raw sparse
embedding-sum MLP, phase-gated over Connect-k → Breakthrough → Othello →
standard chess → Forward Chess. All eleven prior phase gates 0–8 PASS;
phase 9 (Forward Chess learning) documented in progress. Every claim
below is backed by a committed report in `reports/` and re-verified
today where cheap.

### Reproducibility check (done today)

- `cargo build --release` — clean (1m10s).
- `cargo test --release` — 88/88 pass (unit + property + differential +
  UCI compliance).
- `lab bench` on the best checkpoint
  (`campaigns/fc_full_w64_n800/champion`, w64, 378,571 params) loads and
  reproduces the documented profile: 44.0k nodes/s at 400-node budget,
  depth 4.4 @ 1 s/move single core (report said 37–44k n/s, 4.8 @ 2 s).
- Determinism: byte-identical records across thread counts is a
  standing, tested property (RNG streams keyed by `(run_seed, pair,
  slot)`).

## 2. The "approximately 1000 Elo" baseline, precisely

The previous system never claimed absolute Elo. The documented
standard-chess calibration (Phase 7, commit 4deabb1, all
protocol-relative logistic Elo, fastchess, timemargin=100, 0.3 s/move,
250-opening symmetric suite):

| Anchor | Games | Score | Protocol Elo |
|---|---:|---:|---|
| Random legal mover | 100 | 99.0% | ≈ +800 |
| Stockfish 17.1 UCI_Elo=1320 | 200 | 3.5% | −576 ± 141 |
| **Stockfish 17.1 @ 20 nodes/move** | **300** | **10.33%** | **−375 ± 43** |

"≈1000 Elo" is a fair informal absolute-scale reading of this bracket
(e.g. 1320 − 576 ≈ 744 against the one absolute-ish knob, with the knob
itself under-calibrated at fast TC; +800 over random). **The
reproducible baseline for §51.6 ("demonstrate measurable improvement
over the previous ≈1000-Elo baseline") is therefore the protocol
result: 10.33% ± (43 Elo CI) over 300 games vs SF17.1@20nodes at
0.3 s/move**, produced by the w64 16-generation tabula-rasa champion
(`runs/20260814-131835-selfplay-chess-w64-g200-e10-s1-c03fefc/checkpoint`,
preserved). Engine binary hashes, PGNs, and logs are archived under
`reports/phase_07/matches/`.

## 3. How much strength came from search (§54 question 2)

Measured, not assumed:

- Raw policy (no search) scores **0.050** against its own 400-node
  searched self at chess — search is the engine of strength.
- Search scaling is monotone over 256× on Forward Chess and 64× on
  chess: **≈ +90–175 protocol Elo per node doubling (mean ≈ +134)**.
- SF17.1 with 1,000× fewer nodes/move is still +375 ahead → evaluation
  quality and move ordering, not node count, are the binding constraint.
  This is the direct empirical motivation for the SHSD program.

## 4. Where CPU time goes (profile)

From `reports/compute_strength_profile.md` (measured 2026-08-15, same
hardware class, spot-reverified today):

- **Evaluation is 66% of node cost at w64** (54% at w32, 75% at w128):
  37k nodes/s with model vs 109k nodes/s zero-evaluator floor (FC full).
- The zero-eval floor itself is ~10–50× below engine-grade movegen —
  the portable directional generator and per-node overheads are the
  second-largest cost after eval.
- Effective branching factor ≈ 10 out of ~30–40 legal moves with only
  TT + policy ordering (no PVS/aspiration/history/quiescence).
- Training generation (200 games + 2,000 Adam steps + 120 gate games):
  117–209 s on 7 workers depending on width; dominated by Adam steps +
  search, not trunk math.
- Playing RSS 11–17 MB; training a few hundred MB/worker; only exact
  solving is memory-hungry (46.5M-position retrograde solve peaked at
  15 GB).

## 5. What the previous evaluator learned (and did not)

- Supervised ceiling on exact games: raw net ≈ oracle (99.4% WDL,
  Connect-k; 99.97% Breakthrough) → fitting machinery is sound.
- Occupancy-only Othello value plateaued at 85% WDL — representation,
  not optimization, was limiting (intended stress test).
- At chess/FC scale the raw sparse representation is data-starved: w128
  loses to w32 at fixed data; raw-policy coverage stall reproduced on
  three game families (~500 distinct states per 400-game generation).
- Conclusion (already drawn by the old program's own reports, and the
  SHSD program's premise §1.1): the raw sparse representation is too
  statistically inefficient; search knowledge must be distilled into a
  structured evaluator.

## 6. Component inventory and reuse map

| Component | Verdict | Notes |
|---|---|---|
| `game.rs` `Game` trait | **Reuse as-is** | Rule-level facts only; explicitly forbids strategic conclusions — exactly the SHSD §6.2 boundary. New `FeatureExtractor` will be a separate trait (§10.3). |
| `games/forward_chess.rs` | **Reuse** | 2,192 lines, differential-tested, D031-hardened, king invariant in `make_move`. `FORWARD_CHESS_RULES.md` authoritative (§52.1 satisfied). |
| `games/chess.rs` (cozy-chess) | **Reuse** | Perft-exact; UCI binary passes compliance tests (§51.2 satisfied). |
| `games/{breakthrough,othello,connect_k}.rs` | **Reuse** | The §15 game ladder is already implemented (Breakthrough = §15.2; Othello = §15.3 choice already made and validated). |
| `search.rs` `Searcher` | **Reuse as §36 foundation** | Has: negamax, alpha–beta, iterative deepening, TT, deterministic node budgets, terminal/repetition handling, `Evaluator`-supplied leaf values + policy ordering, completed-depth reporting. Missing vs §36: PVS, aspiration windows, history-based ordering (added later, evidence-gated, each with reference-mode tests). No quiescence/pruning/extensions — clean slate for §37–§42. |
| `search.rs` `Evaluator` trait | **Reuse as-is** | `leaf_value(&mut self, game, state) -> i32` + optional `policy_scores` — precisely the plug-in point for structured evaluators. |
| `ExactSolver` + `solve_retrograde` + verified tablebases | **Reuse** | §9.5 oracle infrastructure done: FC tiny (84k) and small (46.5M, 559 MB checksummed tablebase) solved; exact corpora carry per-child WDL — ready-made §18.1/§19.5 targets. |
| `evaluation.rs` paired arena | **Reuse** | Deterministic paired colour-swapped matches, LCB gates, match probes — the §55 paired arena exists. |
| `experiment.rs` run dirs | **Reuse, extend** | Manifests, CPU/RSS probes exist; SHSD adds parameter ledger + provenance fields. |
| `model.rs` sparse MLP + `CompiledNet` | **Freeze as §9.3 raw-model baseline** | Keep one representative version + checkpoints. No further investment. Burn stays a dependency only while this baseline is kept runnable (§11.4: new structured models are direct Rust). |
| `training.rs` recipe v1 | **Reuse for baseline; new fitting code for structured models** | Multinomial logistic / GAM fitting (§46.1) is a direct implementation, not Burn. |
| `tools/` (fc_train.sh, fc_rating.py, run_match.sh, check.sh) | **Reuse** | Campaign/rating/match/check machinery is game-agnostic at the CLI level. |
| Old phase reports + decision log | **Preserve untouched** | D001–D031 remain binding where they describe still-live code. |

**Smallest reusable vertical path** (§76.7): `Game` trait →
`forward_chess`/`chess` rules → `Searcher` (alpha–beta + TT) →
`Evaluator` plug-in → paired arena → run directories. A new structured
evaluator only needs to implement `Evaluator` and can be measured
end-to-end (fixed-node and fixed-time) with zero new infrastructure.

## 7. Preserved baselines (§9)

| Baseline | Artifact |
|---|---|
| §9.1 Existing system | Phase 7 chess champion (w64 16-gen) + protocol; FC c03 champion (`campaigns/fc_full_w64_n800/champion`, +476 pool Elo) + w32-n800 line; all hashed/archived. |
| §9.2 Search-only | `ZeroEvaluator` in `search.rs` (measured: 109k n/s, depth 6.8 @ 2 s FC). |
| §9.3 Raw-model | The sparse MLP family above, frozen. |
| §9.5 Oracle | FC tiny/small tablebases; Breakthrough 4×4r1/4×5r1; Connect-k rungs; Othello 4×4. Stockfish 17.1 (`/usr/games/stockfish`) + fastchess v1.8.2 for the quarantined Track B teacher. |
| §9.6 Random | `--random` UCI baseline mode. |

## 8. Gaps: minimum Stage B instrumentation (§55)

Already present: deterministic node-limited search, completed-depth
reporting, position hashing/dedup, run manifests, paired arena, CPU
counters.

To build (the whole of Stage B, nothing more):

1. **Typed training records with target provenance** (§19.9, §18.2):
   one record format carrying position, WDL/score target, provenance
   enum (`exact | terminal | internal_search | stockfish_teacher | …`),
   and teacher metadata (depth, nodes, best-move stability, score gap,
   evaluator checkpoint).
2. **Deep-search teacher relabelling** command: sample positions from
   games/trajectories, relabel with deeper internal search, record
   shallow-vs-deep residuals (the §6.1 "search is a teacher" loop).
3. **Counterfactual child search** on sampled positions (top-k or all
   moves) producing move-rank records (§19.5) — exact games already
   have this via `child_wdl` in solver corpora.
4. **Frozen evaluation sets** (§20.2): carve immutable test suites from
   the exact corpora + a frozen self-play position set; record hashes.
5. **Parameter-provenance ledger** (§13): one repo-level
   `parameter_ledger.json` + per-run copies; seed it with the existing
   frozen constants (ε=0.10, replay 4, recipe v1, TT size, …) and their
   provenance classes from the decision log.
6. **Feature-cost measurement hook**: ns/feature-family timing in the
   (future) extractor path, reported per run.

Gate check (§55): with 1–4 the system can separately attribute model
error (held-out prediction), search error (shallow-vs-deep residual),
move-ordering error (rank of final best move), and cost (existing
counters + 6).

## 9. Deliberately NOT implemented yet (§76)

Reachability/arrival maps, commitment features, safe-mobility
decomposition, support graphs, any structured feature family (Stage C+
only, one at a time, behind the §14.2 admission protocol); PVS,
aspiration, history ordering, quiescence, LMR, futility, ProbCut, null
move, extensions, proof search (Stages F–H); GAM/phase/pair/pattern
models (evaluation ladder, evidence-gated); NNUE-style incremental
inference; MCTS; new games; any new dependency.

## 10. Risks and notes carried forward

- **D031 lesson**: deep-search self-play is a rules fuzzer; the
  `make_move` king invariant stays permanently.
- **D027 lesson**: always `timemargin=100` + check PGN termination tags
  in fastchess matches.
- The `small` FC tablebase (46.5M positions) is the single most
  valuable training/eval asset for early structured-evaluator work:
  exact WDL + exact per-child WDL at scale, zero teacher noise.
- Disk: 38 GB free; `runs/` holds 4.5 GB of regenerable solver corpora
  (tablebase.bin backups exist) — delete-on-pressure per the standing
  storage policy, no action needed now.
- VM is back to 8 vCPU / 32 GB (the 2-vCPU downsizing noted in earlier
  session memory is obsolete).

## Decision

- **Retain**: everything in the reuse map; no rewrite (§54 gate
  respected — the existing code is sound and measured).
- **Replace over time**: raw sparse evaluator as the production model
  (kept as baseline); ad-hoc training-record formats (superseded by
  provenance-typed records in Stage B).
- **Next**: Stage B — research instrumentation (the six items in §8),
  smallest implementation first: typed records + deep-search
  relabelling, validated on Forward Chess `tiny`/`small` where exact
  answers exist.

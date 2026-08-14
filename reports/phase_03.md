# Phase 3 Report: Self-play recovery of exact strategy

## Status

PASS

## Research hypothesis

Search-guided self-play (Expert Iteration) can recover the exact
strategy of a solved game without oracle labels during training — the
oracle is used only for evaluation. The four-condition diagnostic
matrix separates representation, optimization, search, and self-play
failure.

## Minimal implementation

- **Model in search** (§11.2): `Evaluator` trait (leaf value + policy
  scores) on `Searcher`; move order = TT move, then descending policy
  logits (stable sort keeps action-ID tie order), then optional
  `Reversed` degradation. Leaf value = `1000·(P(win) − P(loss))` from
  the WDL head. A budget too small for one completed iteration returns
  the policy argmax, so node budget 1 behaves as raw-policy play.
- **Compiled inference** (§10.6 contingency, D015): framework batch-1
  inference profiled at 63.6 µs/forward — >95% of projected self-play
  cost. `CompiledNet` (plain row-major weights, vectorized dots) runs
  the same forward in 6.4 µs (10×), validated against the framework to
  1e-4 by a mandatory test. Training and batched evaluation keep the
  framework path.
- **Self-play generations** (§12): `lab selfplay` — frozen champion,
  per-game deterministic RNG streams `(run_seed, generation, game)`,
  ε-greedy exploration sampling the apprentice policy (label stays the
  expert move), per-position records with all §12.2 fields written per
  generation, warm-started candidates (fresh Adam), promotion by oracle
  val metrics with a fixed noise tolerance (D017), halt after two
  consecutive strictly-worse candidates (§12.6 stop-and-diagnose),
  per-generation checkpoints, FIFO replay window of 4 generations
  (§12.5 escape clause, justified by measured forgetting — D016).
- **Evaluation** (§32): `searched_decision_metrics` (search-guided
  decisions on a hash-selected corpus sample), `exploitability_vs_perfect`
  (both colours from the initial state + all distinct ply-1/2 states,
  vs a solver-optimal opponent), and `lab evaluate` kind `oracle_probe`
  (raw + searched + exploitability across node budgets for any saved
  checkpoint).

## Deliberately excluded

Speculative pruning and quiescence (§11.5–11.6), value-target blending
(§12.3 initial rule kept; the controlled comparison was not needed —
the searched agent reached the threshold), prioritized/novelty replay,
training league, repetition handling (Connect-k cannot repeat),
parallel training (sweep-level parallelism only).

## Correctness evidence

38 tests passing (fmt/clippy/debug/release clean). New:

- policy ordering changes node counts but never values (sampled full
  searches with a deliberately non-natural policy);
- compiled inference matches the framework within 1e-4 on 60 states
  (mandated by §10.6) and its evaluator interface is cache-consistent;
- self-play generation is byte-identical across thread counts, expert
  and played actions are always legal, exploratory moves keep the
  expert label;
- three separate full runs of one config produced byte-identical final
  metrics (cross-process determinism, observed in the calibration
  sweep);
- depth-1-with-oracle-leaves, node-budget, and TT invariants from
  Phase 1 all hold with the trait-based evaluator.

## Global learning or playing result

Instance: 4x4 k=4 gravity (draw; 134k solved states; oracle used only
for evaluation). Diagnostic matrix (test split; "no search" = node
budget 1, search = 600 / 5000 nodes):

| Training labels | No search | Search 600 | Search 5000 |
|---|---|---|---|
| Exact oracle (Phase 2, w128) | 99.76% opt / regret 0.0024 | 100%, 0/42 exploit drops | 100%, 0/42 |
| Self-play (w128, seed 2) | 80.9% / 0.2123 | 99.75% / 4/42 drops | **100.0% / 2/42 drops** |

- **The searched self-play agent recovers exact play**: 100% oracle-
  optimal decisions on 2,000 held-out states at 5,000 nodes; 99.6–99.9%
  at the training budget (600). Residual exploitability 2/42 games
  (mean 0.048 levels) at 5,000 nodes.
- **The raw self-play model improves but is coverage-limited**: action
  accuracy 78% → 81–82% (regret 0.25 → 0.20–0.21) and plateaus. Cause
  is diagnosed, not silent: a generation of 400 games contains only
  ~500 distinct positions (of 107k corpus states), so states outside
  the self-play distribution stay unlearned. This is a property of the
  frozen exploration mechanism, not of representation (oracle-trained
  raw = 99.8%), optimization (windows are fully fitted), or search
  (search lifts both models to 100%).
- Exploitability declines with training: headline drops 8→4, 10→4,
  18→11 per seed; promotion halts fire at plateau as designed.

## Required one-time exploration experiment

ε ∈ {0.05, 0.10, 0.20} × 3 seeds (w64, 400 games, 600 nodes, 12 gens):

| ε | final raw regret (3 seeds) | searched acc | final exploit drops /42 |
|---|---|---|---|
| 0.05 | 0.227 / 0.226 / 0.233 | 0.9985 / 0.9945 / 0.9925 | 5 / 5 / 9 |
| 0.10 | 0.209 / 0.214 / 0.205 | 0.9970 / 0.9980 / 0.9930 | **4 / 3 / 5** |
| 0.20 | 0.200 / 0.206 / 0.196 | 0.9955 / 0.9970 / 0.9955 | 7 / **14** / 6 |

Raw recovery is monotone in ε (coverage), but final exploitability is
unstable at 0.20 and reliably best at 0.10. **ε = 0.10 chosen and
frozen** (D018) per §12.4's reliability criterion.

## Scaling results

- **Games per generation** (100/400/1600, w64): 100 is clearly worse
  (9 final drops, raw regret 0.222); 400 → 1600 is flat within noise
  (4 → 8 drops, regret 0.209 → 0.212). More identical-line games add
  little coverage; the binding constraint is distribution width, not
  volume.
- **Generation search nodes** (150/600/2400): monotone and the
  strongest lever — final drops 12 → 4 → 2, mean levels lost 0.286 →
  0.095 → 0.048, raw regret 0.213 → 0.209 → 0.200. A stronger expert
  yields better recovery.
- **Model width** (64/128 headline): equal searched quality within
  noise at this instance size; w64 suffices for 4x4.
- **Evaluation search nodes** (probe, self-play w128 champion):
  1 → 10 → 100 → 600 → 5000 nodes gives 81.5% → 93.0% → 98.4% → 99.75%
  → 100.0% optimal decisions. Monotone above 10. Anomaly at 10 nodes:
  exploitability worsens (25/42 vs 22/42 raw) although decision
  accuracy improves — a 2-ply search trusts the weak value head at
  leaves more than the well-distilled policy argmax. Known ExIt
  asymmetry (policy learns from search; value only from outcomes);
  diagnosable, disappears by 100 nodes.

## CPU and memory result

Self-play generation ~700 games/min at 600 nodes/move on 7 workers
(6.4 µs/forward compiled inference); a full 12-generation w64 run costs
~100 s wall, headline w128/16 gens ~255 s, peak RSS < 300 MiB. The 16
experiment runs totalled ~35 min through `lab sweep`. Exploitability +
searched probes ~4 s per generation.

## Reproducibility

- Commit: cbc3d4b + this phase's commit; toolchain rustc 1.97.1
- Seeds 1–3, all config-explicit; self-play streams keyed by
  `(run_seed, generation, game_index)`; three same-config runs produced
  byte-identical results
- Configs `configs/phase_03/` (23 files incl. two sweep manifests);
  per-run metadata under `reports/phase_03/runs/` (23 runs);
  per-generation self-play records and checkpoints in each run dir

## Complexity delta

- Dependencies: 0 added. New public API: `Evaluator`/`ZeroEvaluator`
  (search), `CompiledNet`/`ModelEvaluator` (model), `TrainRow`,
  `SelfPlayRecord`/`SelfPlayStats`/`generate_selfplay`/`train_steps`
  (training), `SearchedMetrics`/`ExploitabilityReport`/`CorpusSplit`
  + two functions (evaluation) — each maps to a §23 implement bullet.
- Config surface: `SelfPlayConfig` (10 fields — the §23 experiment axes
  plus loop mechanics), `oracle_probe` evaluate kind (5 fields),
  `replay_generations` (D016). `lab evaluate` configs are now tagged by
  `kind`.
- Production LOC: ~+1,050 across search/model/training/evaluation/lab;
  tests +~230. Recipe v1 unchanged.
- Removed/simplified: `train_supervised` now delegates to the shared
  `train_steps` core; no duplicated training loop.

## Failures and anomalies

- First trajectory runs halted at generation ~6 via the promotion gate:
  diagnosed as current-generation-only forgetting → FIFO window (D016)
  and tolerance rule v2 (D017). Post-fix runs improve monotonically
  until a genuine coverage plateau, then halt honestly.
- Raw-model full-corpus recovery plateaus at ~81% action accuracy —
  coverage-limited (measured: ~500 distinct states/generation), the
  central open finding for later phases; search closes the gap
  entirely.
- Exploitability regression at 10-node evaluation (value-head/policy
  asymmetry), explained above.
- ε=0.20 destabilizes final exploitability on one seed (14/42) — the
  reason ε=0.10 was frozen.

## Decision

- Promote phase: yes
- Selected configuration: ε=0.10, replay window 4, promotion tolerance
  0.005, gen budget 600+ (2400 when quality matters), w64 for 4x4-class
  instances; recipe v1 unchanged
- Rejected alternatives: ε=0.05 (slow), ε=0.20 (unstable exploitability);
  strict promotion rule (halted on noise); framework-only inference
  (10× slower, kept as training/reference path)
- Reason: all §23 acceptance criteria met — searched agent at 100%
  optimal decisions (5k nodes) / ≥99.3% at 600 on all seeds (threshold:
  99% on ≥2 of 3), exploitability declines and is reported alongside
  regret, data/search budgets scale as required, and every failure mode
  observed was diagnosed in this report

## Exact next phase

Phase 4 (Breakthrough curriculum, §24): parameterized Breakthrough
(forward moves, diagonal captures, back-rank/elimination terminal),
required rule tests (slow reference movegen, make/unmake, colour
reflection, races, blocked positions), exact solving on 4x4/5x5, then
the same supervised-ceiling and self-play-recovery experiments on the
largest practical exact size, plus first strategic (non-exact) size
with champion progression. Search/model/training code must not change
except for generic correctness bugs (§24 phase-integrity gate).

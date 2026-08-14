# Phase 2 Report: Supervised oracle ceiling

## Status

PASS

## Research hypothesis

The compact sparse policy/value architecture (embedding sum → two ReLU
layers → WDL + action-logit heads), trained supervised on the exact
corpus of a solved Connect-k instance, can represent near-oracle play,
with quality scaling monotonically in model width and data size and
stable across seeds. This measures the *representation and optimization
ceiling* of the exact architecture/recipe that later self-play phases
will use — separating "the model can't represent it" from "self-play
can't find it" before self-play exists.

## Minimal implementation

- `src/model.rs`: `PolicyValueNet` in Burn 0.21 (pure-Rust NdArray CPU
  backend, autodiff). Sparse feature IDs → embedding sum scaled by
  `1/√n` → two ReLU layers (width W) → 3 WDL logits + per-action logits.
  Single capacity knob `model_width`. Save/load via Burn's binary
  full-precision recorder.
- `src/training.rs`: exact-state dataset via in-process solve
  (hash-stratified 80/10/10 split; states are unique so splits cannot
  leak), fixed **recipe v1**: Adam, lr 1e-3, batch 256,
  `L = CE(wdl) + CE(policy)` with the policy target uniform over
  optimal actions and illegal actions masked; deterministic epoch
  shuffling (ChaCha12); single-threaded training.
- `src/evaluation.rs`: `evaluate_model_oracle` — WDL accuracy / log
  loss / Brier, optimal-action accuracy (argmax over legal), optimal
  mass, decision regret in WDL levels, accuracy stratified by state
  value.
- `src/bin/lab.rs`: `lab train <config>` (config: seed, model_width,
  train_positions, training_steps, game) writing per-step
  `metrics.jsonl`, periodic validation, an atomic checkpoint that is
  **reloaded and output-verified inside every run**, and full
  train/val/test metrics in `summary.json`. `lab sweep <manifest>`:
  CPU-slot scheduler over JSONL entries `{command, config, cores}`
  (first meaningful sweep, per plan §16.4).
- `enumerate_solved` now reports per-move child values (needed for
  regret labels); the exact corpus rows gained a `child_wdl` column.

Dependency added: `burn 0.21` (std, ndarray, autodiff), mandated by the
plan (§10.6). Everything else is unchanged.

## Deliberately excluded

Search-integrated evaluation and play-based exploitability (Phase 3
requirement), learning-rate/architecture tuning beyond the fixed recipe
(no experiment demanded it), training parallelism (runs are
single-threaded; the sweep supplies parallelism), Burn's high-level
`Learner`/dataset machinery (the loop is ~60 lines and fully
deterministic), quantization/distillation (Phase 8).

## Correctness evidence

- 35 tests passing (fmt/clippy/debug/release clean). New:
  - forward-pass shape and repeat-determinism checks;
  - Adam reduces a simple loss;
  - save/load reproduces outputs bit-exactly (also re-verified inside
    every training run on a validation probe batch);
  - dataset split disjointness/completeness (505 states on 3x3 — the
    Phase 1 count — with non-empty optimal sets everywhere);
  - **memorization test**: 300 steps on 32 states drives WDL loss
    below 0.05;
  - **determinism test**: two identical-seed trainings produce
    identical loss traces (exposed that Burn's backend RNG is
    process-global; tests serialize on a lock, production is one run
    per process).
- Training-time validation and final test metrics come from held-out
  splits never touched by the optimizer.

## Global learning or playing result

Oracle instance: 4x4 k=4 gravity (134,289 solved states; splits
107,335 / 13,432 / 13,522). Test-split quality of the raw network (no
search), fixed recipe:

| Metric (test split) | best (w128, 40k steps) |
|---|---|
| WDL accuracy | **0.9944** |
| WDL log loss | 0.0223 |
| Optimal-action accuracy | **0.9976** |
| Policy mass on optimal actions | 0.9845 |
| Decision regret (WDL levels/move) | **0.0024** |
| Action accuracy by state value [L, D, W] | 1.000 / 0.9977 / 0.9969 |

The supervised ceiling for this architecture on the oracle instance is
effectively optimal play: ~1 blunder per ~420 decisions in held-out
states, each costing one WDL level.

## Scaling result

Capacity sweep (full data, 10k steps, seed 1):

| width | params | WDL acc | action acc | regret |
|---:|---:|---:|---:|---:|
| 16 | 1,191 | 0.7804 | 0.8659 | 0.1476 |
| 32 | 3,399 | 0.8956 | 0.9142 | 0.0917 |
| 64 | 10,887 | 0.9695 | 0.9715 | 0.0295 |
| 128 | 38,151 | 0.9905 | 0.9953 | 0.0049 |

Data sweep (w64, 10k steps, seed 1) with train→test WDL-accuracy gap:

| train positions | WDL acc | action acc | regret | generalization gap |
|---:|---:|---:|---:|---:|
| 13,375 (1/8) | 0.9046 | 0.9488 | 0.0544 | +0.088 |
| 26,750 (1/4) | 0.9459 | 0.9578 | 0.0451 | +0.038 |
| 53,500 (1/2) | 0.9618 | 0.9714 | 0.0304 | +0.015 |
| 107,335 (full) | 0.9695 | 0.9715 | 0.0295 | +0.005 |

Both axes are monotone with no saturation up to w128/full data.
Extending training 10k→40k steps at w128 lifts test WDL accuracy
0.9905→0.9944 while the train split reaches 0.9995 — the residual error
is a genuine generalization gap, not an optimization failure.

Seed stability (3 seeds each): w64 test WDL acc 0.9623–0.9695
(spread 0.7pt), w128 0.9890–0.9905 (spread 0.15pt); action accuracy
even tighter. The recipe is stable.

## CPU and memory result

Single-threaded training throughput 19k–65k examples/s depending on
width and co-scheduled load; a w64/10k-steps run costs ~41 s alone.
Peak RSS ≤ 77 MiB per run. The 11-run sweep completed in 357 s on
8 logical CPUs with the new CPU-slot scheduler (zero failures).
Checkpoints are ~150 KB.

## Reproducibility

- Commit: e8c2276 (+ this phase's commit); toolchain rustc 1.97.1;
  Cargo.lock updated with burn 0.21.0 (hash in each manifest)
- Seeds: 1, 2, 3 (config-explicit); dataset subsets seed-independent
- Configs: `configs/phase_02/` (13 files + sweep manifest);
  per-run metadata archived under `reports/phase_02/runs/` (13 runs,
  including per-step `metrics.jsonl`)

## Complexity delta

- Dependency: +1 (`burn`, plan-mandated). Public types: `PolicyValueNet`,
  `ModelDims`, backend aliases, `Example`, `ExactDataset`, `Batch`,
  `TrainStepMetrics`, `OracleMetrics` — each maps 1:1 to a §22 implement
  bullet (network, dataset, training loop, oracle evaluation).
- Config surface: one new config type (`TrainConfig`) with the two
  phase-authorized experiment axes (`model_width`, `train_positions`)
  plus plan-required `seed`/`training_steps`; `lab sweep` manifests
  (§16.4, first meaningful sweep).
- Production LOC: ~80 (model) + ~230 (training) + ~120 (oracle eval)
  + ~330 (lab train/sweep); tests +~180.
- Recipe constants (batch 256, lr 1e-3) are named, documented as
  recipe v1, and changeable only via named experiment.

## Failures and anomalies

- Burn's backend RNG is process-global; concurrent RNG-touching tests
  were nondeterministic until serialized behind a lock. Production
  isolation is per-process, documented in ARCHITECTURE.md.
- An OOM kill during a full-parallel debug+release test compile on the
  7.5 GiB VM; build jobs are now capped (`-j 4`) in the workflow.
- The 3x3 smoke run showed 58% test WDL accuracy — a 43-state test
  split is noise, not signal; the phase's conclusions use the 4x4
  instance only.
- Debug-mode test time crept to 125 s dominated by a release-grade
  measurement test; it is now release-only (`#[cfg_attr(debug_assertions,
  ignore)]`), debug suite back to ~22 s.

## Decision

- Promote phase: yes
- Selected configuration: architecture as specified, recipe v1,
  **w128 as the reference capacity** for the 4x4 oracle instance
  (w64 acceptable when speed matters); training_steps is a
  quality/compute dial with no measured instability
- Rejected alternatives: none — all swept points behaved monotonically;
  nothing to remove
- Reason: all §22 acceptance criteria met — near-ceiling accuracy
  (99.4% WDL / 99.8% action), monotone capacity and data scaling, seed
  stability, verified checkpoints, and the remaining error diagnosed as
  a small generalization gap concentrated in decision-relevant (win/draw)
  states

## Exact next phase

Phase 3 (search + learned evaluation): integrate the trained network as
the leaf evaluator and move-ordering policy inside alpha–beta; verify
search-with-model consistency (§14 invariants hold with a model in the
loop); measure play strength versus raw policy and versus perfect play
(exploitability from the §32.3 start-position set), and search-depth
scaling of decision regret. This is the last phase before the self-play
loop closes.

# Structured-Heuristic, Search-Distillation Game AI Research Program

## A CPU-first Rust research and engineering specification for learning chess-like games and Forward Chess

---

# 0. Purpose of this document

This document is the governing research and engineering specification for the next version of the game-playing system.

It is intended to be given directly to a highly capable autonomous coding and research agent.

The agent should use this document to:

- inspect and preserve the useful parts of the existing implementation;
- formulate and execute controlled experiments;
- design compact, interpretable, automatically fitted evaluators;
- improve search efficiency without importing unjustified chess assumptions;
- test generality across several games;
- establish scaling behavior with model size, data, search, and CPU compute;
- obtain a reproducible standard-chess strength estimate;
- and ultimately build a strong Forward Chess engine.

This document deliberately does **not** prescribe every implementation detail or pre-commit every experiment. The agent is expected to exercise research judgment.

However, the agent must obey the methodological, simplicity, reproducibility, and code-quality constraints in this document.

The project should proceed through evidence-gated research stages. At the end of each substantial stage, the agent must produce a report and stop or explicitly justify the next experiment. It must not silently turn a failed experiment into a forest of new modes, parameters, or abstractions.

---

# 1. Starting evidence and revised research thesis

## 1.1 Existing result

A previous implementation was developed and tested using approximately:

- 8 vCPUs;
- 32 GB of memory;
- 32 GB of disk space;
- roughly one day of elapsed work and experimentation;
- a modest number of total CPU-hours.

That system:

- learned substantially better-than-random play;
- produced a functioning generic search-and-learning pipeline;
- reached approximately 1000 Elo in the standard-chess calibration used for that run;
- obtained much of its strength from tree search;
- and did not learn a sufficiently accurate or strategically rich value function from a raw sparse representation under the available compute budget.

Treat this as valuable evidence, not failure.

The result suggests that:

1. search is already capable of generating useful strategic information;
2. the original state representation was too statistically inefficient for the available data;
3. terminal outcomes and raw sparse features required too much rediscovery;
4. the next system should make structural relationships easier to express;
5. the engine should distill knowledge from its stronger searches;
6. the number of learned parameters should be reduced or structured;
7. search-control decisions should themselves be learned from direct search evidence where practical.

## 1.2 Revised central thesis

The new project is based on this principle:

> Do not hand-design the game’s strategy. Hand-design a compact vocabulary of meaningful measurements through which a strategy can be expressed, then learn the signs, magnitudes, nonlinearities, phase dependence, interactions, move preferences, and search-control parameters automatically.

Examples of acceptable measurements include:

- attack;
- defence;
- reachability;
- arrival time;
- mobility;
- support;
- connectivity;
- promotion distance;
- path robustness;
- commitment;
- escape capacity;
- lane structure;
- congestion;
- piece interaction;
- king escape geometry;
- structural volatility.

These are properties of a position, not claims about whether the properties are good or bad.

The system must generally **not** assume:

- conventional chess piece values;
- that advancement is good;
- that centralization is good;
- that mobility is always good;
- that material advantage is readily convertible;
- that exchanges favor the materially stronger side;
- that ordinary pawn-structure concepts transfer;
- that ordinary king-safety formulas transfer;
- or that a feature’s effect is linear.

The learner should discover these relationships from:

- exact solutions;
- deeper search;
- self-play;
- historical opponents;
- counterfactual move analysis;
- and controlled external teachers used only in clearly separated diagnostic tracks.

---

# 2. Agent role

You are acting as:

- a senior Rust systems engineer;
- a game-search researcher;
- a statistical experimentalist;
- a reinforcement-learning researcher;
- a CPU performance engineer;
- a software architect;
- and a ruthless simplifier.

Your task is not to maximize the number of implemented techniques.

Your task is to discover the **smallest system that exhibits reliable, reproducible, and compute-efficient strength scaling**.

You must optimize for:

1. correctness;
2. scientific validity;
3. diagnostic clarity;
4. sample efficiency;
5. fixed-time playing strength;
6. generality across related games;
7. reproducibility;
8. maintainable Rust code;
9. CPU utilization;
10. architectural simplicity;
11. raw peak strength.

Raw strength eventually matters, but unexplained strength is not sufficient. A more complicated engine that is slightly stronger but scientifically opaque, brittle, or impossible to tune is not necessarily an improvement.

---

# 3. How to use this specification

## 3.1 Evidence-gated progression

For every substantial research stage:

1. state the research question;
2. identify the current baseline;
3. propose the smallest experiment capable of answering the question;
4. identify the primary metric before running the experiment;
5. identify the expected result under each plausible hypothesis;
6. implement only the required change;
7. test correctness;
8. measure computational cost;
9. execute the experiment;
10. analyze uncertainty and failure cases;
11. perform ablations where needed;
12. decide whether to:
   - retain the change;
   - revise it;
   - reject it;
   - or gather more evidence;
13. delete rejected production paths;
14. record the decision;
15. stop before beginning a materially new research stage.

## 3.2 Agent discretion

The stages later in this document are a recommended dependency order, not an inflexible calendar.

The agent may reorder experiments when:

- an earlier experiment exposes a clear bottleneck;
- an exact game provides a cheaper route to the same evidence;
- implementation realities make a different ordering substantially simpler;
- or a result invalidates the assumptions of the proposed sequence.

Any reordering must be justified in the research log.

The agent must not use “autonomy” as permission to:

- add speculative abstraction;
- skip baselines;
- invent arbitrary constants;
- perform an untracked hyperparameter search;
- replace a failed method with several simultaneous changes;
- or continue into expensive experiments without an interpretable scaling signal.

---

# 4. Research grounding

The methodology is informed by several established lines of game-AI research.

Expert Iteration separates planning from generalization: search produces an improved policy, and a learned apprentice generalizes those search decisions so that subsequent searches improve. The original work demonstrated this in Hex and reported a tabula-rasa search agent that defeated the publicly released MoHex 1.0. ([arxiv.org](https://arxiv.org/abs/1705.08439?utm_source=chatgpt.com))

TDLeaf was developed specifically to combine temporal-difference learning with minimax search. KnightCap used this family of methods to learn evaluation parameters while playing online, with the published work reporting an increase from roughly 1650 to 2150 in 308 games; the authors also emphasized that diverse online opposition was important, so this result must not be treated as evidence that naive self-play alone is sufficient. ([researchportalplus.anu.edu.au](https://researchportalplus.anu.edu.au/en/publications/learning-to-play-chess-using-temporal-differences/?utm_source=chatgpt.com))

Earlier work also demonstrated that relative chess piece values can be learned through temporal differences applied to minimax search, rather than copied from human chess books. ([journals.sagepub.com](https://journals.sagepub.com/doi/pdf/10.3233/ICG-1997-20302?utm_source=chatgpt.com))

Giraffe is relevant because it attempted to learn evaluation, move-ordering, and search-control knowledge rather than relying only on a large manually designed chess evaluation. It is evidence that these components can be treated as separate learning problems. ([arxiv.org](https://arxiv.org/abs/1509.01549?utm_source=chatgpt.com))

Buro’s work on Logistello combined statistically fitted feature conjunctions with selective search. The generalized linear evaluation model was explicitly intended to make feature construction and weight assignment more data-driven, while ProbCut used shallow/deep search correlations to make probabilistic pruning decisions. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0004370201000935?utm_source=chatgpt.com))

Systematic short n-tuple networks in Othello showed that carefully structured local interactions can produce steep learning curves with very few parameters; one published experiment used only 288 weights. The exact result is game-specific, but the broader lesson is that compact, structured interactions can outperform poorly chosen large representations. ([arxiv.org](https://arxiv.org/abs/1406.1509?utm_source=chatgpt.com))

General spatial state-action features have also been developed and evaluated across 33 games, providing evidence that efficiently evaluated local patterns around candidate actions can improve search across diverse board geometries. ([arxiv.org](https://arxiv.org/abs/2201.06401?utm_source=chatgpt.com))

MCTS with implicit minimax backups improved performance in Kalah, Breakthrough, and Lines of Action by maintaining rollout estimates and heuristic minimax values separately. This makes it a credible diagnostic alternative if alpha–beta remains constrained by an unreliable evaluator, although it is not the default production direction here. ([arxiv.org](https://arxiv.org/abs/1406.0486?utm_source=chatgpt.com))

NNUE demonstrates that shallow, sparse, incrementally updated networks can support very fast CPU inference. The official Stockfish documentation emphasizes sparse inputs, minimal input changes between consecutive positions, low-precision inference, and incremental accumulators. These principles are relevant if a small residual model eventually justifies itself, but a large chess-specific NNUE architecture should not be imported prematurely. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html?utm_source=chatgpt.com))

Modern Stockfish search contains a dense collection of iterative deepening, aspiration, move histories, reductions, pruning, correction histories, transposition handling, and re-search logic. It should be treated as an inventory of experimentally validated search ideas, not as a code template to imitate indiscriminately. ([github.com](https://github.com/official-stockfish/Stockfish/blob/master/src%2Fsearch.cpp?utm_source=chatgpt.com))

For engine evaluation, Fastchess supports concurrent matches, repeated openings, fixed-node settings, SPRT, and detailed result reporting. Stockfish’s own documentation emphasizes that engine Elo is relative to the opponent pool, time control, openings, and other match conditions, and that reduced `UCI_Elo` play is produced by intentionally selecting suboptimal moves rather than by representing an absolute human rating. ([github.com](https://github.com/Disservin/fastchess?utm_source=chatgpt.com))

For the Rust implementation, current documentation describes Rayon as lightweight data parallelism, `rand_chacha` as providing deterministic portable generators, Criterion as statistics-oriented microbenchmarking, `cozy-chess` as a strongly typed and performance-oriented legal move-generation library, and Burn as providing CPU backends, autodifferentiation, and model storage. Use exact dependency versions selected at implementation time and commit `Cargo.lock`. ([docs.rs](https://docs.rs/crate/rayon/latest?utm_source=chatgpt.com))

---

# 5. Non-goals

Do not attempt to build:

- a clone of Stockfish;
- a clone of AlphaZero;
- a giant end-to-end sparse network;
- a general game-description language;
- a universal game-AI framework;
- a distributed cluster platform;
- a cloud orchestration service;
- a web dashboard;
- an opening-book system;
- a hand-authored Forward Chess strategy guide;
- a large library of manually assigned feature weights;
- a permanent collection of interchangeable evaluators;
- a hierarchy of search-strategy classes;
- an asynchronous actor–learner architecture before a synchronous design is demonstrably insufficient;
- a plugin registry;
- or an optimizer that automatically mutates arbitrary architecture choices.

The following are acceptable as research instruments but must remain separate from the primary result:

- Stockfish-labelled standard-chess positions;
- standard-chess opening suites used symmetrically for evaluation variance reduction;
- exact tablebases for reduced games;
- exact low-material Forward Chess subgames;
- teacher-assisted diagnostic models;
- alternative search methods used for disagreement analysis.

---

# 6. Non-negotiable scientific principles

## 6.1 Search is a teacher, not merely a runtime crutch

The previous implementation already demonstrated that search contains more knowledge than the static evaluator.

Therefore:

- deeper search must generate training targets;
- shallow-to-deep disagreements must be logged;
- best-move stability across depths must be measured;
- search residuals must be analyzed;
- and evaluator training must attempt to compress the useful work performed by search.

A stronger evaluator should then:

- improve shallow search;
- improve move ordering;
- reduce the number of nodes needed to reproduce deep-search decisions;
- and ideally improve fixed-time strength even if its evaluation is somewhat more expensive.

## 6.2 Measurements may be designed; strategic conclusions must be learned

It is acceptable to calculate:

- number of safe moves;
- minimum promotion distance;
- arrival-time difference;
- graph connectivity;
- path count;
- number of attackers;
- support lag;
- commitment cost;
- king escape width.

It is generally not acceptable to hardcode:

- safe mobility is worth 18;
- a rook is worth 5 pawns;
- advancement is positive;
- doubled pawns are bad;
- an open file deserves a bonus;
- exchanging while ahead is good.

The first category defines observables. The second category asserts strategy.

## 6.3 Every parameter must have provenance

Every numeric parameter must be classified as exactly one of:

### Learned parameter

Fitted from data through a documented objective.

Examples:

- evaluator weights;
- spline values;
- piece-relation tables;
- move-ordering weights;
- phase-gating weights;
- pruning-risk coefficients.

### Experimentally selected hyperparameter

Chosen from a small, explicit comparison with saved results.

Examples:

- regularization strength;
- number of spline knots;
- model capacity;
- learning rate;
- search budget;
- exploration rate.

### Mathematical or representational constant

Determined by the rules or by a formal invariant.

Examples:

- board dimensions;
- number of piece types;
- action-space encoding;
- symmetry transforms;
- exact terminal values.

### Engineering constant

A resource or storage choice with no strategic interpretation.

Examples:

- buffer size;
- number of records per output shard;
- checkpoint interval;
- memory cap.

The repository must contain a parameter-provenance ledger. Unclassified constants are defects.

## 6.4 Do not infer strategy from one metric

A feature or model change must not be accepted solely because it improves:

- training loss;
- static evaluation accuracy;
- best-move agreement;
- fixed-node Elo;
- fixed-time Elo;
- or nodes per second.

The relevant combination is:

1. predictive quality;
2. search utility;
3. fixed-node playing strength;
4. fixed-time playing strength;
5. computational cost;
6. robustness across seeds and games.

## 6.5 Use the cheapest valid source of evidence

Prefer, in order:

1. mathematical invariants;
2. exact solutions;
3. full-search equivalence tests;
4. deep-search labels;
5. fixed-position paired comparisons;
6. paired engine matches;
7. large self-play tournaments;
8. human impressions.

Do not use noisy game outcomes to optimize a parameter when direct shadow-search labels are available.

## 6.6 Separate model quality from search quality

Always measure:

- raw evaluator or raw policy;
- shallow search;
- medium search;
- deep search;
- fixed-node strength;
- fixed-time strength.

A model that appears strong only under very deep search may not itself be learning much.

A model that is accurate but too slow may lose practical strength.

## 6.7 Preserve negative results

Every rejected feature family or search technique must leave behind:

- its hypothesis;
- exact implementation version;
- configurations;
- result;
- computational cost;
- and reason for rejection.

Rejected production code should normally be removed.

The report should remain.

## 6.8 One substantive research change at a time

Do not simultaneously add:

- a new feature family;
- a new model family;
- a new target;
- and a new search heuristic.

When interaction experiments are eventually required, they must be explicitly identified as such.

---

# 7. Definition of success

The project is successful only if it satisfies scientific, playing-strength, engineering, and simplicity criteria.

## 7.1 Scientific success

The final report must establish:

- which structured feature families produce measurable value;
- which do not;
- how evaluator error changes with model capacity;
- how strength changes with training data;
- how strength changes with search budget;
- whether these trends are monotonic over a useful range;
- which resource is limiting at each stage;
- how much of the engine’s strength comes from evaluation versus search;
- whether learned search control improves efficiency;
- whether results transfer across games;
- and which Forward Chess concepts differ most strongly from standard chess.

## 7.2 Standard-chess success

The system must:

- operate through UCI;
- complete statistically controlled matches;
- obtain a reproducible relative strength estimate;
- report the uncertainty of that estimate;
- separate self-search-trained and Stockfish-teacher-assisted tracks;
- demonstrate measurable improvement over the previous approximately 1000-Elo baseline under a documented protocol;
- and show whether the improvement comes from better evaluation, better ordering, better search selectivity, or a combination.

No particular absolute Elo is mandatory.

A scientifically convincing 1300–1600 result from a compact structured model may be more valuable than an opaque and unstable higher number.

## 7.3 Forward Chess success

The system must:

- generate only legal moves;
- handle orientation reversal correctly;
- handle checks, castling, en passant, promotions, repetitions, and draws;
- improve against historical checkpoints;
- obtain a stable internal rating;
- show positive scaling with at least two of:
  - model capacity;
  - training data;
  - teacher-search budget;
  - evaluation-search budget;
- and demonstrate that at least several structural feature families transfer meaningfully to the target game.

## 7.4 Engineering success

The final implementation must:

- be written primarily in Rust;
- build through Cargo;
- use a pinned Rust toolchain;
- commit `Cargo.lock`;
- produce deterministic fixed-seed runs where expected;
- make effective use of the VM’s CPU allocation;
- avoid thread oversubscription;
- resume at safe boundaries;
- enforce memory and disk budgets;
- pass correctness tests;
- and keep experiment artifacts self-contained.

## 7.5 Simplicity success

At completion:

- there should be one production evaluator family;
- one production move-ordering model;
- one principal search implementation;
- one training pipeline;
- one experiment format;
- no speculative plugin architecture;
- no collection of obsolete modes;
- and no large set of parameters without clear provenance.

---

# 8. Principal research questions

The project should answer the following questions.

## 8.1 Representation

1. Can compact semantic measurements outperform the previous raw sparse representation per training position?
2. Which measurements transfer across games?
3. Which measurements are specific to directional movement?
4. Does a generalized additive model capture enough nonlinearity?
5. When are pair terms or local patterns necessary?
6. Is a residual network still useful after the semantic model is strong?

## 8.2 Learning

1. Are deep-search WDL targets more useful than terminal outcomes?
2. Are move-ranking targets more sample-efficient than absolute value targets?
3. Does TDLeaf-style updating improve online iteration?
4. How much historical-opponent diversity is required?
5. How much does disagreement mining help?
6. Does the model generalize to positions outside its self-play distribution?

## 8.3 Search

1. How much extra effective depth comes from learned move ordering?
2. Which quiet moves must quiescence include?
3. Can shallow/deep residuals safely fit ProbCut or futility margins?
4. Can late-move reductions be chosen by an error-risk model?
5. Does Forward Chess contain enough zugzwang-like structure to make null move unsafe?
6. When do proof-oriented subsearches outperform ordinary alpha–beta?
7. Does fixed-time strength benefit from each extra evaluator feature?

## 8.4 Scaling

1. Does larger evaluator capacity reduce held-out search error?
2. Does more self-generated data improve performance?
3. Does greater teacher-search depth improve the apprentice?
4. Does more evaluation search improve the final player?
5. Where does each axis saturate?
6. What is the compute-optimal balance between evaluator cost and search depth?
7. Are scaling slopes similar across games?

## 8.5 Forward Chess strategy

1. How state-dependent are piece values?
2. How important is material relative to structure?
3. How costly is irreversible advancement?
4. How important are support lag and escape capacity?
5. What makes a promotion valuable when the new piece reverses orientation?
6. How much does horizontal waiting-move capacity matter?
7. Which structures prevent conversion of material advantage?
8. How often do pieces become strategically irrelevant because they can no longer interact with the important region?

---

# 9. Required experimental baselines

Preserve and version the following baselines.

## 9.1 Existing-system baseline

Reconstruct, where possible:

- the previous best checkpoint;
- its exact code revision;
- its search budget;
- its model size;
- its training data;
- its standard-chess match protocol;
- its measured Elo or match result;
- its nodes per second;
- and its CPU-hour cost.

If exact reconstruction is impossible, document the gap and create a closest faithful baseline.

## 9.2 Search-only baseline

Use the strongest simple evaluator already available without new structural learning.

This baseline answers:

> How strong is search without the proposed learned knowledge?

## 9.3 Raw-model baseline

Retain one representative version of the previous raw sparse evaluator.

Do not continue investing in it unless it unexpectedly remains competitive.

It exists to answer:

> Does the structured representation improve sample efficiency?

## 9.4 Structured linear baseline

Use the smallest meaningful structured evaluator:

- learned piece-type/orientation counts;
- learned piece-square/orientation terms;
- elementary mobility;
- immediate attack and defence counts;
- promotion distance;
- side to move.

All weights must be fitted.

## 9.5 Oracle or teacher ceiling

On small games, train against exact WDL and optimal actions.

On standard chess, create a separate Stockfish-teacher diagnostic.

On Forward Chess, use:

- deeper internal search;
- exact reduced games;
- exact low-material positions;
- and historical champion disagreement.

## 9.6 Randomized controls

Where useful, compare against:

- shuffled labels;
- shuffled feature groups;
- random weights with matched variance;
- randomly ordered moves;
- and equal-cost dummy features.

These controls help identify accidental leakage or benchmark artifacts.

---

# 10. Minimal architecture

The architecture should correspond to intrinsic parts of the research pipeline.

## 10.1 Core modules

A reasonable module structure is:

```text
src/
  game/
  features/
  evaluation/
  search/
  data/
  training/
  arena/
  experiment/
  bin/
```

These modules represent real concepts:

- `game`: rules and legal transitions;
- `features`: position and state-action measurements;
- `evaluation`: learned value and policy models;
- `search`: exact and approximate game-tree search;
- `data`: typed training records and storage;
- `training`: fitting and optimization;
- `arena`: paired play and ratings;
- `experiment`: configuration, manifests, metrics, and reports;
- `bin`: explicit command-line entry points.

Do not create additional architectural layers unless a real current requirement cannot be represented here.

## 10.2 Static dispatch

Use static dispatch in hot loops.

A top-level game enum may dispatch CLI requests to monomorphized game-specific functions.

Do not use trait objects inside:

- move generation;
- make/unmake;
- leaf evaluation;
- feature extraction;
- or recursive search,

unless profiling demonstrates that the cost is immaterial and the simplification is substantial.

## 10.3 Core traits

Potential intrinsic traits include:

- `Game`;
- `FeatureExtractor`;
- `Evaluator`.

Do not create:

- `SearchStrategy`;
- `TrainingBackend`;
- `ExperimentProvider`;
- `ModelFactory`;
- `FeatureRegistryPlugin`;
- or similar extension abstractions while only one production implementation exists.

## 10.4 State and move typing

Use typed newtypes or enums for:

- player;
- square;
- rank;
- file;
- piece type;
- orientation;
- feature ID;
- action ID;
- node count;
- depth;
- WDL outcome;
- search score;
- checkpoint ID;
- experiment seed.

Do not pass semantically different quantities as unlabelled integers where Rust can distinguish them cheaply.

## 10.5 One package first

Prefer one Cargo package with multiple binaries.

Create a workspace only if at least one of these becomes true:

- independent crates have genuinely distinct dependency graphs;
- compile times become materially problematic;
- a reusable library is consumed by multiple external packages;
- or a separate GPL-bound integration must be isolated.

Do not split the project to create an appearance of modularity.

---

# 11. Rust and package-management requirements

## 11.1 Toolchain

At repository root:

- commit `rust-toolchain.toml`;
- pin a stable Rust release;
- commit `Cargo.lock`;
- record the compiler version in every run manifest;
- use release builds for all performance results;
- preserve portable CI builds;
- use native CPU targeting only for machine-specific performance runs.

## 11.2 Required checks

Run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
cargo doc --no-deps
```

Use additional tools only when justified:

- Criterion for microbenchmarks;
- property testing for move/state invariants;
- fuzzing for parsers and move sequences;
- Miri when unsafe or subtle aliasing is introduced;
- dependency auditing before releases.

## 11.3 Dependency policy

Likely acceptable dependencies include:

- `clap`;
- `serde`;
- `toml`;
- `serde_json`;
- `rand`;
- `rand_chacha`;
- `rayon`;
- `thiserror`;
- `tracing`;
- `tracing-subscriber`;
- `criterion`;
- a property-testing crate;
- `cozy-chess` for standard-chess reference or production rule handling;
- Burn only when autodifferentiated residual modelling is justified.

Do not add:

- an async runtime;
- a database;
- a web framework;
- a general workflow engine;
- multiple linear-algebra libraries;
- multiple ML frameworks;
- multiple logging frameworks;
- or unpinned Git dependencies.

## 11.4 Model implementation policy

For:

- linear models;
- generalized additive models;
- splines;
- pair tables;
- small ranking models;
- and compact search-control regressions,

prefer a direct Rust implementation.

A heavyweight tensor framework is not required for a few thousand parameters.

Use Burn or another autodifferentiation framework only when:

- a genuine small neural residual is introduced;
- manual gradients would add more complexity than the dependency;
- and the production inference path remains demonstrably efficient.

## 11.5 Unsafe code

Do not use `unsafe` until:

1. a profiler identifies a relevant bottleneck;
2. a safe implementation exists as a reference;
3. the expected gain is material;
4. safety invariants are documented;
5. property and differential tests cover the operation;
6. benchmarks confirm the gain.

---

# 12. Configuration discipline

## 12.1 Typed command-specific configurations

Do not create a universal configuration containing dozens of optional fields.

Use separate required types such as:

- `SolveConfig`;
- `TeacherDataConfig`;
- `FitConfig`;
- `SelfPlayConfig`;
- `ArenaConfig`;
- `SweepManifest`;
- `ProfileConfig`.

Every field must be required or supplied by a versioned named recipe.

## 12.2 Configuration admission rule

A field may become user-configurable only when:

- it represents a real current experimental variable;
- at least two values will be compared;
- the comparison will be reported;
- and changing the field does not create an incoherent unsupported mode.

## 12.3 Versioned recipes

Keep algorithmic defaults in named recipes such as:

```text
structured_eval_v1
structured_eval_phase_gam_v2
move_order_ranker_v1
forward_search_v3
```

A recipe is:

- immutable after publication;
- fully resolved in the run manifest;
- and associated with a decision record.

Do not silently mutate what a recipe name means.

## 12.4 No optional-argument forest

Avoid:

```text
enable_x
disable_y
use_old_z
experimental_mode
fallback_mode
legacy_model
alternate_search
```

An experiment should normally live:

- on a branch;
- behind a private development constant;
- or in a separate short-lived executable,

until evidence supports adoption.

Once adopted, remove the previous production path unless it remains an essential baseline.

---

# 13. Parameter-provenance ledger

Maintain a machine-readable and human-readable parameter ledger.

Suggested fields:

```text
name
component
type
current value
unit
allowed range
provenance class
selection experiment
date selected
code revision
sensitivity result
notes
```

Examples:

| Parameter | Class | Required evidence |
|---|---|---|
| Piece-type utility coefficient | Learned | Fitted training objective |
| Number of phase experts | Hyperparameter | Capacity comparison |
| Maximum search ply | Engineering | Safety/resource rationale |
| Promotion rank | Mathematical | Game rule |
| LMR risk threshold | Learned or selected | Shadow-search calibration |
| TT size | Engineering | Memory/performance benchmark |
| Exploration probability | Hyperparameter | Multi-seed learning comparison |

Every stage report must identify ledger changes.

---

# 14. Complexity governance

## 14.1 Complexity budget per experiment

A normal experiment should add no more than:

- one feature family;
- one model mechanism;
- one search mechanism;
- two configuration variables;
- one new production dependency;
- three public types.

Exceeding this budget requires explicit justification before implementation.

## 14.2 Feature admission protocol

A new feature family must have:

1. a precise definition;
2. an invariance specification;
3. a computational-cost estimate;
4. a hypothesis about what error it addresses;
5. a target source;
6. an ablation plan;
7. a fixed-node evaluation plan;
8. a fixed-time evaluation plan;
9. a removal criterion.

## 14.3 Search-technique admission protocol

A search technique must have:

1. an observed failure mode;
2. a direct diagnostic;
3. a minimally sufficient implementation;
4. a correctness-preserving reference mode;
5. a shadow-search or paired-search evaluation;
6. a node-efficiency metric;
7. a playing-strength metric;
8. a false-prune or re-search metric where relevant;
9. a rollback path.

## 14.4 Model-capacity admission protocol

A larger model is justified only when:

- the smaller model has measurable held-out residual structure;
- extra capacity improves a held-out target;
- and the gain survives fixed-time search evaluation.

Do not enlarge a model merely because training loss has not reached zero.

## 14.5 Deletion policy

Delete:

- rejected model variants;
- obsolete configuration modes;
- duplicate feature implementations;
- unselected optimizers;
- unused abstractions;
- stale benchmark code.

Retain:

- reference move generators;
- exact solvers;
- small correctness oracles;
- frozen baseline binaries;
- experiment reports;
- result datasets where storage allows.

---

# 15. Game and benchmark ladder

The benchmark suite should provide complementary evidence.

## 15.1 Connect-\(k\)

Role:

- exact correctness;
- search equivalence;
- target-sign validation;
- basic model scaling;
- cheap global learning tests.

Use parameterized sizes.

Do not spend significant future research effort here after the pipeline is validated.

## 15.2 Breakthrough

Role:

- irreversible forward movement;
- races;
- blocking;
- support;
- sacrifices;
- overextension;
- promotion-like progress;
- low-cost strategic testing.

Breakthrough is the most relevant intermediate game for directional geometry.

Use several board sizes:

- exact;
- partially exact;
- strategic.

## 15.3 Othello or Hex

Role:

- test non-material structural reasoning;
- test local patterns;
- test mobility and path concepts;
- ensure that the system is not merely learning capture arithmetic.

Choose one initially.

Othello is preferable for:

- alpha–beta;
- phase structure;
- mobility reversal;
- compact local patterns.

Hex is preferable for:

- connectivity;
- path robustness;
- cuts;
- global structure without material.

Do not maintain both unless they answer different unresolved questions.

## 15.4 Reduced Forward Chess

Role:

- exact testing of the target geometry;
- orientation reversal;
- commitment;
- support;
- king movement;
- promotion transition;
- repetition;
- structural evaluation.

Construct several reduced rulesets that preserve important mechanics rather than merely shrinking the board indiscriminately.

## 15.5 Chess-like reduced games

Optional candidates include:

- Gardner minichess;
- Los Alamos chess;
- custom reduced standard-chess material sets;
- limited-pawn-and-piece positions;
- endgame subsets.

Role:

- bridge from generic board games to full chess semantics;
- evaluate king safety;
- evaluate promotion;
- evaluate tactical stabilization;
- validate UCI-independent search behavior.

## 15.6 Standard chess

Role:

- external calibration;
- UCI integration;
- access to a strong teacher for diagnostics;
- comparison with conventional search behavior;
- relative Elo estimation.

Standard chess is not expected to be the ideal training domain for the final model. It is a demanding calibration domain.

## 15.7 Full Forward Chess

Role:

- final target;
- internal rating progression;
- structural discovery;
- scaling analysis;
- human and agent challenge matches.

---

# 16. Portability levels for features

Classify every feature family.

## Level U: universal board-game feature

Examples:

- legal mobility;
- local action pattern;
- connected components;
- arrival time;
- move entropy;
- support graph;
- path count.

## Level D: directional-board feature

Examples:

- forward reachability;
- commitment;
- blind wake;
- interaction-window closure;
- support frontier;
- horizontal waiting reserve.

## Level C: chess-family feature

Examples:

- check;
- king escape;
- castling rights;
- promotion;
- attack and defence by piece type;
- sliding-ray structure.

## Level F: Forward-Chess-specific feature

Examples:

- reversed promoted orientation;
- normal/reversed coverage complementarity;
- promotion transition shock;
- orientation-dependent attack asymmetry.

A feature that can be represented at a more general level should not be hardcoded at a more specific one.

---

# 17. Evaluation-model ladder

Advance through this ladder only when residual evidence justifies the next level.

## 17.1 Level 0: trivial learned baseline

Components:

- piece-type/orientation counts;
- side to move;
- basic terminal-distance term where exact;
- perhaps one learned piece-square table.

Purpose:

- verify fitting;
- verify symmetry;
- quantify how little a trivial evaluator knows.

## 17.2 Level 1: linear structured WDL model

Form:

\[
z(s)=b+\sum_i w_i x_i(s)
\]

Output:

- three-class WDL softmax;
- or advantage plus draw propensity.

Requirements:

- side-swap antisymmetry;
- orientation-aware features;
- no manually fixed strategic signs;
- regularization;
- calibrated probabilities.

Use this to establish:

- feature-family value;
- data efficiency;
- feature correlation;
- residual structure.

## 17.3 Level 2: generalized additive model

Form:

\[
z(s)=b+\sum_i g_i(x_i(s)).
\]

Each \(g_i\) may be:

- a regularized bucket table;
- a piecewise-linear spline;
- a monotonic curve only where monotonicity is mathematically justified.

This model can learn:

- saturation;
- thresholds;
- sign reversals;
- asymmetric tails;
- diminishing returns.

Examples:

- mobility may be valuable from 0 to 3 moves but less important afterward;
- advancement may be dangerous in the middle ranks and decisive near promotion;
- one safe king escape may matter much more than the fifth.

## 17.4 Level 3: soft phase mixture

Form:

\[
E(s)=\sum_p q_p(s)E_p(s),
\]

where \(q_p(s)\) is a small learned softmax over phase observables.

Potential phases:

- formation;
- contact;
- race;
- reversed-piece;
- low-mobility conversion.

The phase gate should remain small.

Do not create a separate manually maintained evaluator for each phase.

## 17.5 Level 4: selected pair interactions

Add a small number of:

\[
h_{ij}(x_i,x_j)
\]

or:

\[
w_{ij}x_ix_j.
\]

Candidate interactions:

- advancement × support;
- advancement × escape width;
- material × convertibility;
- promotion distance × interception margin;
- king danger × escape capacity;
- commitment × forcing status;
- mobility × congestion;
- reversed-piece count × rear exposure.

Interactions must be selected by:

- residual analysis;
- sparse regularization;
- forward selection;
- or a defined automatic search.

Do not manually add hundreds of plausible interactions.

## 17.6 Level 5: piece-relation and local-pattern model

Add:

- piece-pair relation tables;
- local n-tuples;
- source/destination state-action patterns;
- small pattern conjunctions.

Use systematic short patterns before large random patterns.

Tie symmetric entries.

Use low-rank factorization or sparsity where tables become large.

## 17.7 Level 6: shallow tree ensemble

A small boosted or oblivious-tree model may be tested over semantic features.

Admit it only when:

- additive residuals contain clear high-order thresholds;
- pair terms are insufficient;
- evaluation cost remains acceptable;
- and the resulting model remains inspectable.

Do not create a forest with thousands of trees.

## 17.8 Level 7: tiny semantic residual network

Form:

\[
E_{\text{final}}(s)=E_{\text{structured}}(s)+\lambda R(x(s)).
\]

Requirements:

- input semantic features, not raw board indicators;
- one or two small hidden layers;
- residual magnitude cap;
- explicit ablation;
- fixed-time search validation;
- quantized inference if retained.

## 17.9 Level 8: incrementally updated sparse residual

Consider an NNUE-like implementation only when:

- model inference is a measured search bottleneck;
- residual capacity improves fixed-node strength;
- fixed-time strength is limited by inference cost;
- feature deltas are sparse;
- and incremental update complexity is justified.

This is not the default destination.

---

# 18. Value targets and loss functions

## 18.1 Exact WDL

Use exact WDL whenever available.

For positions with multiple optimal actions:

- distribute policy target mass across all optimal actions;
- do not label one arbitrary action as uniquely correct.

## 18.2 Deep-search WDL

Use a calibrated transformation of the deepest trusted search result.

Store:

- completed depth;
- nodes;
- score;
- WDL;
- best-move stability;
- principal variation;
- score gap;
- search method;
- evaluator checkpoint.

Do not treat an unstable shallow search as certain ground truth.

## 18.3 Terminal outcomes

Retain eventual outcomes as:

- an independent target;
- a calibration source;
- protection against teacher self-confirmation.

Do not rely on terminal outcomes alone for long strategic credit assignment.

## 18.4 TDLeaf targets

Test principal-variation leaf updates when:

- full-game outcomes learn too slowly;
- deep-search labels are expensive;
- and online refinement is desirable.

Compare TDLeaf-style training against fixed offline deep-search distillation.

## 18.5 Pairwise move ranking

For moves \(a^+\) and \(a^-\), fit:

\[
S(s,a^+)>S(s,a^-).
\]

Use:

- deep best versus alternatives;
- final-depth best versus shallow best;
- cutoff-causing moves;
- exact optimal versus non-optimal moves;
- move score differences where reliable.

Pairwise ranking is the primary candidate for move ordering.

## 18.6 Residual targets

Fit:

\[
R(s)=V_{\text{deep}}(s)-V_{\text{current}}(s).
\]

Residual analysis identifies:

- missing features;
- phase-specific failures;
- systematic overconfidence;
- tactical horizon effects.

## 18.7 Search-control targets

Search-control models should predict direct events:

- full-depth re-search required;
- shallow cutoff was false;
- quiet move changed quiescent value;
- extension changed the best move;
- null cutoff failed verification;
- pruned move exceeded alpha;
- move caused a beta cutoff.

Do not optimize these decisions through whole-game outcomes until direct calibration is satisfactory.

---

# 19. Training-data sources

Use a controlled mixture rather than one undifferentiated replay buffer.

## 19.1 Exact positions

Sources:

- complete small-game state spaces;
- reduced Forward Chess;
- low-material exact sets;
- exact tactical positions.

Purpose:

- correctness;
- calibration;
- exploitability;
- exact feature analysis.

## 19.2 Current self-play

Purpose:

- on-policy relevance;
- current strategic distribution;
- outcome generation.

Risk:

- narrowness;
- self-confirming errors;
- avoidance of unfamiliar positions.

## 19.3 Historical checkpoint play

Play current candidates against:

- recent champions;
- older champions;
- search-scaled versions;
- structurally distinct earlier models.

Purpose:

- opponent diversity;
- prevent forgetting;
- expose cyclic weaknesses.

## 19.4 Deep-search relabelling

Take positions from:

- self-play;
- historical games;
- random trajectories;
- disagreement sets.

Relabel using deeper or wider search.

Purpose:

- stronger supervision;
- shallow-to-deep residual measurement;
- policy improvement.

## 19.5 Counterfactual child analysis

At sampled positions, search several legal moves rather than only the selected move.

Store:

- child WDL;
- move rank;
- structural deltas;
- score gap;
- required search cost.

Purpose:

- move-order training;
- action-feature learning;
- feature attribution;
- commitment valuation.

## 19.6 Disagreement mining

Oversample positions where:

- raw evaluator and search disagree;
- shallow and deep searches disagree;
- current and historical champions disagree;
- additive and pattern models disagree;
- alpha–beta and a diagnostic MCTS disagree;
- high material advantage has poor WDL;
- quiet moves outperform forcing moves;
- promotion choice is surprising;
- best move changes late.

## 19.7 Perturbed positions

Generate legal perturbations:

- alternate plausible moves;
- random legal moves after a real prefix;
- material-preserving relocations where legal;
- mirrored positions;
- near-frontier variations;
- near-promotion variations;
- positions with unusual orientation mixtures.

Do not train on arbitrary illegal synthetic boards.

## 19.8 Standard-chess teacher data

Maintain a separate diagnostic track using Stockfish.

Possible uses:

- test whether the semantic feature set can express ordinary chess knowledge;
- train move-ordering targets;
- estimate representation ceiling;
- compare internal search teacher quality with Stockfish.

Do not mix these labels into the primary Forward Chess result.

## 19.9 Exact source tracking

Every record must contain target provenance:

```text
exact
terminal
internal_search
stockfish_teacher
historical_match
counterfactual_search
shadow_search
```

This enables target-specific ablation.

---

# 20. Data splitting and leakage prevention

## 20.1 Split by trajectory or root family

Do not randomly split individual positions when adjacent positions from the same game would cross datasets.

Prefer splitting by:

- complete game;
- opening/root position;
- generation;
- exact-state connected region;
- or position-hash family.

## 20.2 Frozen test sets

Maintain immutable test sets for:

- exact positions;
- tactical positions;
- structural positions;
- promotion races;
- king danger;
- reversed-piece positions;
- standard-chess teacher comparison;
- Forward Chess champion disagreement.

Do not train on these sets.

## 20.3 Evaluation opening isolation

Standard-chess evaluation openings may not appear in teacher training or tuning datasets when practical.

Forward Chess internal evaluation starting positions should be frozen separately from self-play generation roots.

## 20.4 Duplicate control

Deduplicate by:

- canonical position hash;
- symmetry-normalized hash where valid;
- side-to-move;
- orientation;
- rule state.

Track repeated samples rather than silently discarding their frequency information.

## 20.5 Teacher versioning

Every teacher label must record:

- teacher checkpoint;
- search revision;
- search budget;
- configuration;
- build;
- machine class if timing matters.


---

# 21. Feature-family research program

The following families are ordered roughly by expected value. The agent should not implement all of them automatically.

Each family should pass the feature admission protocol.

---

# 22. Dynamic material and effective piece utility

## 22.1 Hypothesis

Nominal piece counts are insufficient because a piece’s value depends heavily on:

- whether it can still interact with relevant targets;
- whether it is mobile;
- whether it is supported;
- whether it can contribute before a promotion race resolves;
- whether it is trapped beyond support;
- whether it can attack from the required orientation;
- and whether material advantage can be converted.

## 22.2 Candidate measurements

### Counts

- count by piece type;
- count by owner;
- count by orientation;
- count of promoted reversed pieces;
- count differences;
- pairwise count combinations;
- last remaining piece of a type.

### Active material

- pieces with at least one safe move;
- pieces with at least one reachable enemy target;
- pieces able to influence either king;
- pieces able to influence promotion corridors;
- pieces in the main future-interaction component.

### Irrelevant or dead material

- pieces with no plausible future enemy interaction;
- pieces trapped behind friendly blockers;
- pieces that have passed all important targets;
- pieces whose safe reachability is empty;
- pieces unable to contribute before the estimated race horizon.

### Convertibility

- safe captures available;
- forced contact opportunities;
- favorable directional exchange opportunities;
- number of enemy pieces that cannot avoid eventual interaction;
- ability to convert material into promotion access;
- ability to simplify without losing race control.

## 22.3 Model alternatives

Compare:

1. fixed learned global piece value;
2. phase-dependent piece value;
3. piece value conditioned on mobility;
4. piece value conditioned on mobility and support;
5. decomposed effective utility;
6. pair-relation model.

## 22.4 Experiments

### Piece-value recovery

On standard chess:

- train only type-count values;
- compare learned order and ratios with search labels;
- treat this as a sanity test, not a source of Forward Chess values.

On Forward Chess:

- fit from exact reduced games;
- fit from deep-search residuals;
- compare values across phase and orientation.

### Counterfactual removal

For sampled non-king pieces:

- compare deep search before and after a legal or carefully constructed removal;
- use only as an auxiliary diagnostic;
- stratify by mobility, support, and rank.

### Fixed versus dynamic value

At equal parameter count where possible:

- compare global piece values with dynamic effective utility;
- examine positions with material advantage but poor deep-search WDL.

## 22.5 Acceptance

Retain dynamic material terms if they improve:

- held-out WDL;
- deep-move agreement;
- exact regret;
- and fixed-time match strength.

Reject or simplify them if the same gain is obtained by fewer reachability or mobility terms.

---

# 23. Piece-square, orientation, and phase geometry

## 23.1 Hypothesis

Square location matters, but its value is:

- piece-dependent;
- orientation-dependent;
- phase-dependent;
- nonlinear;
- and potentially non-monotonic with advancement.

## 23.2 Candidate coordinates

- absolute square;
- relative rank;
- relative file;
- distance from edge;
- distance from center;
- distance to friendly king;
- distance to enemy king;
- distance to promotion;
- distance ahead of friendly frontier;
- distance ahead of support frontier;
- distance behind enemy frontier;
- distance to nearest useful lane;
- distance to nearest friendly component.

## 23.3 Experiments

Compare:

- raw piece-square tables;
- rank/file factorization;
- phase-conditioned tables;
- orientation-conditioned tables;
- low-rank square embeddings;
- spline effects over relative rank.

Inspect learned curves rather than assuming expected signs.

## 23.4 Important diagnostic

Plot learned value versus rank for each:

- piece type;
- orientation;
- phase;
- support state.

Look specifically for:

- safe staging bands;
- overextension valleys;
- promotion spikes;
- reversed-piece asymmetry.

---

# 24. Directed reachability and arrival-time maps

## 24.1 Hypothesis

Forward-only movement changes the game’s geometry from static control to directed future access.

A compact reachability representation may therefore be one of the most sample-efficient ways to encode strategy.

## 24.2 Reachability variants

For piece \(p\), compute approximate sets:

\[
R_k(p)=\{\text{squares reachable within }k\text{ own moves}\}.
\]

Variants:

- empty-board reachability;
- friendly-blocker-aware reachability;
- current-occupancy reachability;
- safe reachability;
- robust multi-path reachability;
- attack reachability;
- occupancy reachability.

## 24.3 Summary measurements

- one-step mobility;
- two-step reachability volume;
- three-step relaxed reachability volume;
- safe reachability volume;
- reachable file count;
- furthest reachable rank;
- number of critical squares reachable;
- number of enemy targets in reachability cone;
- number of king-zone squares reachable;
- number of promotion-blocking squares reachable;
- overlap with friendly support;
- overlap with enemy attack arrival.

## 24.4 Arrival fields

For side \(c\) and square \(q\):

\[
\tau_c(q)=\min_p \text{estimated moves for }p\text{ to influence }q.
\]

Use differences:

\[
\Delta(q)=\tau_{\text{opponent}}(q)-\tau_{\text{us}}(q).
\]

Aggregate over:

- promotion corridors;
- king zones;
- gateways;
- frontier gaps;
- horizontal escape lanes;
- blockade squares;
- rear support squares.

## 24.5 Cost ladder

Test increasingly expensive versions:

1. one-step attacks and mobility;
2. relaxed two-step bitboard expansion;
3. blocker-aware two-step expansion;
4. short multi-source BFS;
5. robust path counts;
6. full dynamic flow only at root or in training.

Do not put the most expensive version into every leaf by default.

## 24.6 Experiments

- Fit each reachability level separately.
- Measure incremental predictive value.
- Measure nanoseconds per evaluation.
- Compare fixed-node and fixed-time strength.
- Test whether expensive maps are better as:
  - evaluator features;
  - move-ordering features;
  - root-only features;
  - or training-only teacher features.

## 24.7 Follow-up implications

If reachability is highly predictive but too slow:

- distill expensive reachability into cheaper local terms;
- cache by structural hash;
- incrementally update one-step summaries;
- evaluate expensive maps only near the root;
- or use them only to generate labels.

---

# 25. Commitment and irreversibility

## 25.1 Hypothesis

A forward move permanently removes access to part of the board.

The strategic cost of a move may therefore be approximated by the future options it destroys.

## 25.2 Candidate move measurements

For move \(m\) by piece \(p\):

\[
C_k(m)=
\sum_{q\in R_k^{\text{before}}(p)\setminus R_k^{\text{after}}(p)}
w(q).
\]

Measurements:

- lost reachable squares;
- lost safe squares;
- lost reachable files;
- lost king-defence squares;
- lost support squares;
- lost future enemy interactions;
- lost waiting moves;
- reduction in escape paths;
- change in reinforcement arrival time;
- change in enemy interception margin.

## 25.3 Position measurements

- total already-incurred commitment;
- number of pieces ahead of support frontier;
- total advancement beyond support;
- pieces with no safe horizontal exit;
- pieces with one remaining useful lane;
- pieces with only losing forward moves;
- pieces whose rear region is uncovered;
- pieces with negative support lag.

## 25.4 Point-of-no-return models

A continuous point-of-no-return score may use:

- safe exit count;
- support arrival;
- enemy attack arrival;
- future target count;
- reachable-region shrinkage;
- lane closure risk.

Do not implement a hard tactical rule unless exact evidence supports it.

## 25.5 Experiments

- Train commitment as a move-ordering feature first.
- Test it as a value feature after move-level evidence exists.
- Compare commitment deltas against deep-search move ranks.
- Examine quiet moves whose only apparent purpose is preserving optionality.
- Stratify by promotion proximity and forcing status.

## 25.6 Important interactions

Test:

- commitment × support;
- commitment × promotion threat;
- commitment × check;
- commitment × capture;
- commitment × escape width;
- commitment × phase.

A highly committing move can be winning when it forces promotion.

---

# 26. Safe mobility, optionality, and waiting moves

## 26.1 Hypothesis

Raw legal-move count conflates useful flexibility with unsafe or forced commitment.

## 26.2 Mobility decomposition

Measure:

- legal moves;
- captures;
- checks;
- promotions;
- horizontal moves;
- forward moves;
- safe moves;
- defended destination moves;
- moves surviving directional SEE;
- moves retaining future exits;
- moves preserving king defence;
- moves preserving support connectivity.

## 26.3 Destination diversity

Measure:

- distinct destination files;
- distinct rank bands;
- distinct structural regions;
- distinct enemy targets;
- entropy of destinations;
- number of independent lanes.

## 26.4 Mobility resilience

Estimate:

- mobility after opponent’s best shallow reply;
- minimum mobility over top opponent replies;
- number of pieces immobilizable in one move;
- difference between current and worst-reply mobility.

## 26.5 Waiting resources

Measure:

- safe horizontal moves;
- reversible horizontal pairs;
- low-commitment moves;
- non-repeating waiting states;
- pieces capable of waiting;
- whether every legal move incurs major commitment;
- difference in waiting reserves between players.

## 26.6 Experiments

- Compare raw versus safe mobility.
- Compare count versus nonlinear spline.
- Test waiting-reserve features in low-mobility exact positions.
- Construct or mine positions where passing would be desirable.
- Evaluate null-move prediction errors against waiting resources.

## 26.7 Expected implication

If waiting-reserve features strongly predict null-move failures, they should be used primarily as search-control features even if their direct evaluation value is modest.

---

# 27. Attack, defence, tactical stability, and directional SEE

## 27.1 Hypothesis

Search cannot efficiently learn positional structure while repeatedly being surprised by basic directional tactics.

The engine needs a reliable, orientation-aware tactical substrate.

## 27.2 Attack and defence measurements

- attacked pieces;
- defended pieces;
- attacker counts;
- defender counts;
- multiply attacked pieces;
- undefended pieces;
- underdefended pieces;
- sole defenders;
- overloaded defenders;
- attacks on promotion paths;
- attacks on king escapes;
- attacks from reversed pieces;
- rear or blind-side attacks.

## 27.3 Directional SEE alternatives

### Learned-utility SEE

Use current dynamic piece utilities.

### Local exact tactical exchange search

Search captures, recaptures, promotions, and checks in a localized exchange sequence.

### Exchange classifier

Predict:

- favorable;
- likely favorable;
- unclear;
- likely unfavorable;
- unfavorable.

### Deep-search exchange table

Condition on:

- attacker type and orientation;
- victim type and orientation;
- attackers;
- defenders;
- rank;
- king relation;
- promotion proximity.

## 27.4 Tactical-structure measurements

- forks;
- multi-threats;
- discovered attacks;
- sole-blocker removal;
- support articulation attacks;
- attacks that close escape corridors;
- moves that create one-way interactions;
- pieces unable to respond to rear attacks.

## 27.5 Experiments

- Use SEE first for move ordering.
- Validate against local exact tactical searches.
- Measure capture-order improvement.
- Measure quiescence stability.
- Test whether SEE-based hanging features improve static WDL.
- Examine false positives involving promotion and orientation reversal.

---

# 28. Promotion races and orientation-changing promotion

## 28.1 Hypothesis

Promotion value is determined by:

- arrival time;
- interception;
- support;
- path robustness;
- tempo;
- king effects;
- and the utility of the reversed promoted piece.

A conventional promotion bonus is inadequate.

## 28.2 Race measurements

- legal moves to promotion;
- unobstructed distance;
- safe distance;
- blocker count;
- contested path squares;
- support arrival;
- enemy interception arrival;
- promotion with check;
- promotion while capturing;
- path alternatives;
- move-order parity;
- en-passant effect.

## 28.3 Interception margin

\[
I(p)=
\tau_{\text{opponent intercept}}
-
\tau_{\text{promotion}}.
\]

Compute:

- relaxed margin;
- blocker-aware margin;
- safe-path margin;
- side-to-move-adjusted margin;
- margin after forcing checks.

## 28.4 Path robustness

- viable path count;
- node-disjoint path count;
- edge-disjoint path count;
- minimum blocking cut;
- paths remaining after best defensive block;
- path support redundancy.

## 28.5 Promotion transition measurements

Before and after promotion:

- mobility change;
- safe reachability change;
- attack coverage change;
- rear coverage change;
- support graph change;
- king pressure change;
- promotion-defence change;
- congestion change.

## 28.6 Reversed-piece measurements

- safe horizontal exits;
- useful backward-direction reach;
- ability to protect advanced friendly pieces from behind;
- coverage complementarity with normal pieces;
- orientation diversity;
- rear-zone coverage;
- trapping probability;
- distance to relevant targets.

## 28.7 Experiments

- Exact reduced promotion subgames.
- Promotion-choice comparison.
- Global promotion bonus versus transition model.
- Reversed-piece table versus ordinary piece table.
- Search-depth stability around promotions.
- Feature ablation on positions containing reversed pieces.

---

# 29. Directional king safety

## 29.1 Hypothesis

King safety is governed by directional attack arrival and irreversible escape geometry rather than ordinary shelter formulas.

## 29.2 Measurements

### Immediate safety

- current checks;
- safe checks available to opponent;
- legal replies;
- capture responses;
- block responses;
- attacked king-ring squares.

### Escape geometry

- legal king moves;
- safe king moves;
- safe horizontal moves;
- safe reachable region within \(k\) moves;
- distinct escape corridors;
- bottleneck width;
- number of disjoint escape paths;
- enemy closure time;
- friendly reinforcement time.

### King commitment

- safe region lost by an advance;
- inability to return to rear sanctuary;
- number of future forced advances;
- exposure to reversed pieces;
- distance from support graph.

### Castling

- destination escape geometry;
- rook activation;
- preservation of horizontal options;
- loss or gain of rear coverage;
- castling-right option value.

## 29.3 Experiments

- Train king features on standard-chess teacher data as a representation diagnostic.
- Refit completely for Forward Chess.
- Compare static king-ring counts with arrival-time features.
- Mine positions where deeper search changes evaluation because of a quiet king move.
- Test king-corridor features as extension and quiescence triggers.

---

# 30. Support graph and structural robustness

## 30.1 Hypothesis

A formation’s value depends on the robustness and timing of support, not only current defence counts.

## 30.2 Graph construction

Nodes:

- friendly pieces.

Possible edges:

- immediate defence;
- one-move reinforcement;
- two-move reinforcement;
- support surviving one advance;
- support requiring an open lane;
- support supplied by a reversed piece.

## 30.3 Measurements

- edge count;
- average indegree;
- undefended node count;
- singly defended node count;
- mutually supporting pairs;
- connected components;
- largest component;
- support-chain depth;
- distance of advanced pieces from main component;
- number of pieces supported from behind;
- support disappearing after advance;
- cycle count;
- reversed-orientation support cycles.

## 30.4 Articulation and robustness

For each important piece:

- components created if removed;
- advanced pieces disconnected;
- king defenders disconnected;
- promotion paths disconnected;
- support edges lost.

Global:

- number of articulation pieces;
- number of independent support paths;
- minimum cut to disconnect front from rear;
- robustness under one capture.

## 30.5 Support lag

\[
L(p)=
\tau_{\text{friendly reinforcement}}(p)
-
\tau_{\text{enemy attack}}(p).
\]

Use:

- minimum;
- average;
- worst advanced piece;
- number of pieces with positive lag;
- lag weighted by dynamic utility.

## 30.6 Experiments

- Current defence counts versus temporal support.
- Cheap connectivity versus articulation.
- Support features as value versus move-ordering deltas.
- Cost-benefit comparison of exact graph recomputation versus incremental approximation.
- Exact reduced-game analysis of overextended formations.

---

# 31. Frontier, lanes, congestion, wakes, and blockades

## 31.1 Frontier measurements

For each file or lane:

- foremost friendly piece;
- foremost enemy piece;
- gap between fronts;
- friendly support depth;
- contact status;
- locked status.

Aggregate:

- mean advancement;
- frontier variance;
- adjacent-file roughness;
- protrusion count;
- protected protrusions;
- unsupported protrusions;
- gaps;
- blocked lanes.

## 31.2 Support frontier

Define the furthest region with reliable friendly support.

Measure:

- pieces beyond it;
- total excess distance;
- escape capacity of those pieces;
- enemy arrival to those pieces;
- promotion potential.

## 31.3 Lane measurements

- open lane;
- semi-open lane;
- occupied lane;
- arrival-time advantage;
- entrance control;
- exit control;
- useful-piece access;
- promotion access;
- king access;
- alternative lane count.

## 31.4 Congestion

- friendly blockers ahead;
- pieces with all forward paths blocked by friends;
- sliders blocked by friendly pieces;
- stacked pieces;
- dependency chains;
- dependency cycles;
- moves that unblock multiple pieces;
- own pieces occupying king escapes;
- own pieces blocking promotion paths.

## 31.5 Blind wake

For advanced pieces, define the region they can no longer attack or occupy.

Measure:

- wake area;
- uncovered wake area;
- enemy occupancy in wakes;
- enemy arrival advantage to wakes;
- friendly reversed-piece coverage;
- king squares in exposed wakes;
- support routes crossing wakes.

## 31.6 Blockade strength

- wall thickness;
- minimum captures required to cross;
- alternative crossings;
- support behind blocker;
- blocker mobility;
- sacrificial breakthrough cost;
- horizontal bypass availability.

## 31.7 Experiments

- Start with cheap frontier and congestion terms.
- Add wake measurements after directional geometry is validated.
- Use graph-cut calculations only on sampled roots or training labels initially.
- Examine whether expensive blockade features can be distilled into local patterns.

---

# 32. Conversion, fortress, and breakthrough potential

## 32.1 Hypothesis

Material advantage is valuable only if the stronger side can force contact, promotion, penetration, or simplification.

## 32.2 Convertibility measurements

- forced-contact opportunities;
- unavoidable attacks;
- enemy pieces with no retreat;
- promotion threats requiring response;
- checking sequences forcing structural concessions;
- lanes where interaction cannot be avoided;
- material able to reach the active region before the race horizon.

## 32.3 Fortress measurements

- invasion-path cut size;
- attacker orientation mismatch;
- overdefended blockers;
- low attacker safe mobility;
- sufficient defender waiting moves;
- sealed king corridor;
- attackers past relevant interaction windows;
- low conversion liquidity.

## 32.4 Breakthrough measurements

- sacrificial lane-opening moves;
- minimum dynamic utility sacrificed to open a path;
- threats after opening;
- independent promotion threats;
- defender reinforcement time;
- promotion with tempo;
- reachability gain from removing a friendly blocker.

## 32.5 Sacrificial unblock value

For friendly piece \(p\):

\[
B(p)=
\text{friendly reachability after removal}
-
\text{friendly reachability before removal}.
\]

Use only as a measurement.

Do not automatically recommend sacrificing pieces with high \(B(p)\).

## 32.6 Experiments

- Mine high-material but low-WDL positions.
- Compare convertibility terms against pure dynamic material.
- Analyze positions where exchanging worsens the stronger side’s WDL.
- Train fortress classification from deep-search draw outcomes.
- Test whether fortress features improve draw calibration rather than advantage score.

---

# 33. Local patterns and state-action features

## 33.1 Hypothesis

Many useful tactical and structural interactions are local around:

- a moved piece;
- its source;
- its destination;
- a king;
- a promotion square;
- a gateway;
- or a blockade.

## 33.2 Pattern types

- adjacent pairs;
- horizontal triples;
- forward diagonals;
- 2×2 windows;
- 3×2 oriented windows;
- king-zone patterns;
- promotion-zone patterns;
- source/destination windows;
- slider-ray source changes;
- support-chain fragments.

## 33.3 Pattern contents

Each square may encode:

- empty;
- own piece type and orientation;
- opponent piece type and orientation;
- attacked by us;
- attacked by opponent;
- safe;
- critical region membership.

Control table size carefully.

## 33.4 State-action deltas

For candidate moves:

- mobility change;
- support change;
- commitment change;
- king-safety change;
- promotion-margin change;
- lane opening;
- wake exposure;
- future-interaction change;
- attack and defence change;
- congestion change.

## 33.5 Automatic pattern selection

Generate candidate short patterns systematically.

Select through:

- held-out ranking gain;
- residual correlation;
- L1 regularization;
- group sparsity;
- forward stagewise selection;
- boosting.

Do not retain all generated patterns.

## 33.6 Experiments

- Compare state-only versus state-action patterns.
- Compare random long n-tuples with systematic short tuples.
- Compare pattern evaluation cost.
- Measure move-ordering gain before using patterns in static value.
- Tie symmetric patterns and test orientation-specific exceptions.

---

# 34. More speculative structural ideas

These should not be implemented until high-priority families have been evaluated.

## 34.1 Dominated reachability basin

Fraction of a piece’s reachable region where enemy arrival precedes friendly support.

## 34.2 Interaction-window half-life

Estimated number of advances before two pieces can no longer meaningfully interact.

## 34.3 Defender-to-threat matching

Construct a bipartite graph between threats and available defenders.

Measure:

- maximum matching;
- unmatched threats;
- overloaded defenders;
- defensive slack.

## 34.4 Minimum-cut promotion defence

Minimum number or dynamic utility of defensive interventions required to block all promotion paths.

## 34.5 Independent-lane decomposition

Detect future-interaction components that behave as nearly separate races.

## 34.6 Combinatorial tempo decomposition

Estimate the urgency or move value of independent local races.

## 34.7 Support-graph spectral summaries

Potentially measure bottlenecks or fragility through graph eigenvalues.

## 34.8 Evaluator disagreement uncertainty

Use disagreement between:

- additive structured model;
- local-pattern model;
- or current and historical models

as a search-extension and data-selection signal.

These ideas are research backlog items, not default requirements.

---

# 35. Move-ordering research program

A separate move-ordering model is a high-priority component.

## 35.1 Objective

At equal final search depth and correctness, minimize:

- nodes searched before cutoff;
- total nodes;
- time;
- failed low/high re-searches.

## 35.2 Candidate inputs

### Rule-level

- moving piece type;
- orientation;
- source and destination;
- rank displacement;
- file displacement;
- capture;
- captured type and orientation;
- check;
- promotion;
- castling;
- horizontal move;
- irreversible move.

### Tactical

- directional SEE class;
- destination attacked;
- destination defended;
- newly attacked pieces;
- abandoned defenders;
- discovered attacks;
- safe check;
- promotion threat.

### Structural deltas

- commitment;
- safe mobility;
- support;
- lane access;
- promotion margin;
- king escape;
- frontier roughness;
- congestion;
- future interaction;
- waiting reserve.

### Search history

- TT move;
- previous PV move;
- quiet history;
- capture history;
- killer status;
- countermove;
- continuation history;
- previous cutoff depth;
- shallow score;
- previous iteration rank.

## 35.3 Targets

- final best move;
- final move rank;
- score difference from best;
- beta-cutoff success;
- nodes consumed before cutoff;
- whether full-depth re-search was required.

## 35.4 Model

Begin with:

- linear ranker;
- additive ranker;
- history tables.

Add local state-action patterns only if residual evidence supports them.

## 35.5 Evaluation

Report:

- top-1 best-move recall;
- top-3 recall;
- mean rank of final best move;
- nodes to first cutoff;
- total nodes;
- effective depth;
- fixed-node match strength;
- fixed-time match strength.

## 35.6 Acceptance

A move-ordering model should normally be retained only when:

- it reduces node count materially;
- or increases depth;
- and improves fixed-time play.

Predictive ranking gain without search gain is insufficient.

---

# 36. Search foundation

The production search should begin with:

- negamax;
- alpha–beta;
- iterative deepening;
- principal-variation search;
- transposition table;
- aspiration windows;
- terminal detection;
- repetition handling;
- deterministic node budgets;
- learned and historical move ordering.

Each mechanism must have a correctness-preserving reference mode in tests.

---

# 37. Quiescence and leaf stabilization

## 37.1 Problem

Capture-only quiescence may miss decisive quiet structural moves.

Potentially critical quiet moves include:

- blocking promotion;
- preserving the only support link;
- opening a horizontal escape;
- preventing an interaction window from closing;
- preserving a king corridor;
- avoiding irreversible trapping;
- clearing a friendly blocker;
- stopping a rear invasion.

## 37.2 Structural instability features

- check;
- immediate capture;
- promotion;
- one-move promotion threat;
- hanging piece;
- sole defender under attack;
- support articulation threatened;
- near-zero promotion interception margin;
- near-zero king corridor width;
- high shallow move-value spread;
- shallow/deep disagreement;
- low safe mobility;
- rapidly closing interaction window.

## 37.3 Experiment ladder

1. no quiescence;
2. captures and promotions;
3. captures, promotions, and checks;
4. add exact mandatory promotion blocks;
5. add top-ranked structural quiet move;
6. learned inclusion classifier.

## 37.4 Shadow labels

For sampled quiet candidates:

- run stabilized search with and without the move class;
- record value change;
- best-move change;
- WDL change;
- cost.

Train:

\[
P(\text{quiet move is quiescence-critical}\mid z).
\]

## 37.5 Acceptance

Retain structural quiescence only if it:

- reduces horizon errors;
- improves fixed-node play;
- and survives fixed-time cost.

---

# 38. Late-move reductions

## 38.1 Baseline

Start with a simple depth-and-move-index reduction table.

Do not import a highly tuned chess formula.

## 38.2 Risk features

- remaining depth;
- move index;
- node type;
- in check;
- move-order score;
- history score;
- capture/check/promotion;
- directional SEE;
- commitment;
- structural volatility;
- low mobility;
- support articulation change;
- promotion margin;
- king danger;
- horizontal waiting move.

## 38.3 Shadow-search labels

For sampled reduced moves:

- reduced result;
- full-depth result;
- whether it became best;
- whether it exceeded alpha;
- score error;
- WDL error;
- re-search necessity.

Fit:

\[
P(\text{full-depth search needed}\mid z).
\]

## 38.4 Output

The learned or calibrated controller may choose:

- no reduction;
- one ply;
- two plies;
- a small maximum supported by evidence.

Do not create a continuous unrestricted reduction model.

## 38.5 Evaluation

- false reduction rate;
- re-search rate;
- nodes saved;
- best-move agreement;
- fixed-node Elo;
- fixed-time Elo;
- performance by game and phase.

---

# 39. Futility, razoring, and lazy evaluation

## 39.1 Principle

Margins should come from empirical shallow-to-deep residuals, not conventional piece values.

For residual:

\[
\epsilon=V_{\text{deep}}-V_{\text{static}},
\]

fit conservative quantiles by:

- depth;
- phase;
- tactical state;
- mobility;
- king danger;
- promotion proximity;
- orientation mixture.

## 39.2 Pruning rule

Prune only when:

- static value;
- plus conservative maximum residual;
- plus any omitted feature bound

cannot enter the search window.

## 39.3 Guards

Disable or weaken pruning for:

- checks;
- promotions;
- promotion threats;
- low mobility;
- high commitment;
- reversed-piece tactics;
- king danger;
- evaluator disagreement;
- uncertain phase;
- unfamiliar feature distribution.

## 39.4 Lazy evaluator

Evaluate feature tiers:

1. cheap counts and basic geometry;
2. attacks, mobility, support;
3. reachability and graph terms.

Fit high-confidence bounds on omitted tiers.

Skip later tiers only when the search result cannot change.

---

# 40. ProbCut

## 40.1 Principle

Fit the conditional relation between shallow and deep search.

Potential model:

\[
V_d=aV_s+b+\epsilon,
\]

conditioned on:

- shallow depth;
- target depth;
- phase;
- tactical class;
- mobility;
- uncertainty;
- orientation mixture;
- promotion proximity.

## 40.2 Required calibration

Estimate:

- residual mean;
- variance;
- tail behavior;
- false cutoff probability;
- calibration by score region.

## 40.3 Evaluation

- nodes saved;
- false cutoffs;
- deep best-move agreement;
- fixed-node strength;
- fixed-time strength.

ProbCut should be introduced only after the evaluator and shallow search are sufficiently calibrated.


---

# 41. Null-move pruning

## 41.1 Default stance

Treat null move as dangerous in Forward Chess.

Forward-only movement and limited waiting resources may produce many zugzwang-like positions.

## 41.2 Candidate safety features

- safe horizontal waiting moves;
- low-commitment move count;
- safe mobility;
- all-moves commitment;
- phase;
- promotion race;
- king danger;
- locked front;
- material count;
- reversed-piece presence;
- null-counterfactual discrepancy.

## 41.3 Validation

For sampled null cutoffs:

- run verification search;
- record false cutoff;
- train a safety classifier.

Do not use null move without verification until false-cutoff behavior is understood.

## 41.4 Acceptance

Retain only if:

- node savings are substantial;
- false-cutoff rate is controlled;
- and Forward Chess strength improves.

A successful standard-chess result does not establish Forward Chess safety.

---

# 42. Extensions

## 42.1 Singular extensions

Candidate when one move is demonstrably much stronger than alternatives at a shallower search.

Possible triggers:

- unique non-losing move;
- only promotion block;
- sole king escape;
- only move preserving support articulation;
- singular TT move;
- large score gap.

## 42.2 Structural extensions

Potential triggers:

- promotion race within one tempo;
- interaction window about to close;
- support bridge capture;
- king corridor nearly closed;
- high commitment threshold crossing;
- large evaluator disagreement;
- high shallow/deep instability.

## 42.3 Shadow labels

Compare depth \(d\) and \(d+1\):

- best-move change;
- WDL change;
- score change;
- extension usefulness.

Train a small horizon-risk model.

## 42.4 Constraints

- tightly cap total extensions;
- avoid interacting extension rules initially;
- measure explosion in node count;
- retain only extensions with positive fixed-time effect.

---

# 43. Proof-oriented subsearch and exact solving

## 43.1 Candidate goals

Use proof-number or DFPN-style subsearch for binary questions:

- forced mate;
- forced promotion;
- promotion prevention;
- forced capture of critical blocker;
- forced escape;
- forced loss avoidance.

## 43.2 Invocation

Invoke only when:

- a tactical predicate is detected;
- branching is low;
- and the binary result has high strategic value.

## 43.3 Exact low-material tablebases

Generate for selected material sets.

Include:

- owner;
- piece type;
- orientation;
- side to move;
- relevant rights;
- repetition or move-count state where required.

Use tablebases for:

- exact play;
- training;
- calibration;
- promotion-value analysis;
- dynamic material analysis;
- search verification.

## 43.4 Progress-layer decomposition

Investigate whether positions can be grouped by a directional progress signature.

Vertical moves may transition between progress layers while horizontal moves remain inside a layer.

Potentially:

- solve horizontal strongly connected components;
- then propagate values between progress layers.

This is speculative and should begin only as a reduced-game research experiment.

---

# 44. Correction histories

## 44.1 Purpose

Capture systematic residuals not represented by the main evaluator.

Potential coarse keys:

- pawn/front structure;
- orientation layout;
- king squares;
- reversed-piece count;
- open-lane mask;
- recent continuation pattern.

Update bounded residual estimates:

\[
C[h]\leftarrow(1-\eta)C[h]
+
\eta(V_{\text{search}}-V_{\text{static}}).
\]

## 44.2 Constraints

- very few correction tables;
- semantically defined keys;
- bounded correction magnitude;
- minimum sample count;
- variance tracking;
- explicit ablation.

Do not create dozens of opaque history arrays.

---

# 45. Alternative search diagnostics

## 45.1 MCTS with implicit minimax values

Test only if:

- alpha–beta remains highly evaluator-sensitive;
- rollout or value estimates are informative;
- and strategic positions show persistent shallow/deep instability.

Possible role:

- teacher ensemble;
- disagreement detector;
- root search;
- Breakthrough benchmark.

Do not immediately support MCTS as a second production search.

## 45.2 Search ensemble

Use a more expensive diagnostic search combining:

- deeper alpha–beta;
- alternate evaluator;
- optional MCTS.

Retain positions with disagreement.

The ensemble may produce better training data without becoming the runtime engine.

---

# 46. Learning and optimization methods

Use different methods for different parameter classes.

## 46.1 Evaluator weights

Preferred methods:

- multinomial logistic regression;
- regularized gradient descent;
- L-BFGS for smooth moderate-size models;
- coordinate descent for sparse linear models;
- gradient-based spline fitting;
- group-sparse optimization;
- pairwise ranking losses.

## 46.2 Feature selection

Preferred methods:

- held-out residual correlation;
- L1 regularization;
- group lasso;
- forward stagewise selection;
- boosting;
- hierarchical sparsity;
- low-rank factorization.

An interaction should normally require its parent features.

## 46.3 TD refinement

Use TDLeaf-style or residual temporal-difference refinement when:

- offline teacher labels leave systematic trajectory errors;
- online adaptation is desired;
- or terminal calibration remains poor.

Compare it against fixed-data refitting.

## 46.4 Search-control models

Fit directly from shadow data:

- logistic classifiers;
- quantile regressions;
- small calibrated lookup tables;
- shallow trees only when needed.

## 46.5 SPSA

Use SPSA only for a modest number of parameters whose effects are visible mainly through playing strength or total search efficiency.

Candidate parameters:

- LMR offsets;
- history bonuses;
- aspiration widths;
- extension thresholds;
- time-management constants;
- ProbCut confidence.

SPSA is designed to estimate gradients in noisy black-box objectives using a small number of perturbed evaluations, making it relevant to game-program tuning. ([sciweavers.org](https://www.sciweavers.org/publications/universal-parameter-optimisation-games-based-spsa?utm_source=chatgpt.com))

## 46.6 CLOP

Use CLOP for a small smooth set of noisy black-box parameters if SPSA is unstable or local quadratic behavior appears plausible. CLOP was designed for noisy game-engine parameter optimization. ([remi-coulom.fr](https://www.remi-coulom.fr/CLOP/?utm_source=chatgpt.com))

## 46.7 Evolutionary optimization

Use CMA-ES or a genetic method only when:

- fewer than roughly a few dozen strongly interacting parameters are involved;
- direct labels are unavailable;
- parallel match evaluation is cheap enough;
- and a simpler optimizer has failed.

Research has shown that selective-search parameters can be automatically tuned to competitive values, but this does not justify evolutionary optimization for every parameter. ([arxiv.org](https://arxiv.org/abs/1009.0550?utm_source=chatgpt.com))

## 46.8 Hyperparameter optimization restrictions

Do not launch a generic black-box optimizer over:

- architecture;
- feature families;
- search techniques;
- replay policies;
- and training recipes simultaneously.

Use human-readable research choices at the family level and automated fitting within the chosen family.

---

# 47. Experiment design

## 47.1 Required experiment card

Before implementation, write:

```markdown
# Experiment: <name>

## Question

## Hypothesis

## Smallest implementation

## Baselines

## Primary metric

## Secondary metrics

## Fixed resources

## Independent variables

## Controlled variables

## Seeds

## Predicted outcomes

### If hypothesis is supported

### If hypothesis is rejected

### If result is ambiguous

## Correctness risks

## Performance risks

## Complexity budget

## Removal criterion
```

## 47.2 Primary metric

Choose one primary decision metric.

Examples:

- exact decision regret;
- held-out deep-search WDL loss;
- nodes required to match teacher move;
- fixed-time paired Elo;
- false-prune rate at a fixed node-saving target.

Secondary metrics may explain the result but should not replace it after observing the data.

## 47.3 Common random numbers

For paired stochastic comparisons:

- use identical initial positions;
- identical colour assignment;
- matched random seeds;
- matched exploration streams where possible.

This reduces noise.

## 47.4 Replication

Use:

- at least three seeds for cheap exact and intermediate experiments;
- at least two seeds for expensive pilot experiments;
- a confirmation run for major conclusions;
- a larger game count until the reported interval is adequate.

## 47.5 Sequential testing

Use SPRT for routine candidate promotion after the evaluation system is stable.

Do not use an early stopped SPRT result as the only final effect estimate.

For final reporting, run a fixed or appropriately planned sample and report the resulting confidence interval.

## 47.6 Multiple comparisons

When testing many features:

- use a development set for screening;
- a validation set for selection;
- a frozen test set for the final claim;
- and a paired tournament for practical confirmation.

Do not repeatedly inspect and tune against the final test set.

## 47.7 Negative controls

Include negative controls when leakage is plausible.

## 47.8 Cost accounting

Every experiment reports:

- wall time;
- core-hours;
- search nodes;
- training examples;
- model evaluations;
- peak memory;
- disk written;
- energy proxy where available;
- and result per unit compute.

---

# 48. Scaling-law program

Raw strength on simple games is secondary to discovering reliable scaling.

## 48.1 Primary axes

Let:

- \(P\): effective model parameters;
- \(D\): unique or weighted training positions;
- \(T\): teacher-search effort per labelled position;
- \(S\): evaluation-search nodes per move;
- \(C\): total core-hours.

## 48.2 Model capacity definition

For structured models, capacity may be varied through:

- number of feature families;
- number of spline knots;
- number of phase experts;
- number of pair terms;
- number of local patterns;
- rank of factorizations;
- residual width.

Report both:

- raw parameter count;
- active parameters per evaluation.

## 48.3 Data definition

Report:

- total positions;
- unique positions;
- unique game roots;
- exact versus search-labelled fraction;
- on-policy versus disagreement fraction;
- target provenance;
- total teacher nodes.

## 48.4 Search definition

Report:

- nominal node budget;
- completed depth;
- selective depth;
- model evaluations;
- quiescence nodes;
- proof-subsearch nodes;
- TT hit rate;
- nodes per second.

## 48.5 Sweep order

Do not begin with a large full factorial design.

For each stable game and model family:

### Capacity sweep

Hold data and search constant.

Use logarithmically spaced capacity values.

### Data sweep

Hold capacity and search constant.

Use logarithmically spaced dataset sizes.

### Teacher sweep

Hold apprentice capacity and evaluation search constant.

Vary teacher nodes or depth.

### Evaluation-search sweep

Freeze a checkpoint.

Vary evaluation nodes or time.

### Local interaction study

After one-dimensional curves are understood, test a small grid near the promising region.

## 48.6 Scaling models

For a local nonsaturated region:

\[
M=
c+
\alpha\log P+
\beta\log D+
\gamma\log T+
\delta\log S.
\]

For exact regret:

\[
R=
R_\infty+
aP^{-\alpha}+
bD^{-\beta}+
cT^{-\gamma}+
dS^{-\delta}.
\]

These are descriptive local fits, not universal laws.

## 48.7 Required plots

For each major game:

- held-out loss versus parameters;
- exact regret versus parameters;
- strength versus data;
- strength versus teacher compute;
- strength versus evaluation search;
- nodes per second versus capacity;
- fixed-node strength versus capacity;
- fixed-time strength versus capacity;
- strength versus total core-hours;
- variance across seeds;
- residuals of the scaling fit.

## 48.8 Monotonicity requirements

A major stage should not be considered successful merely because the largest run is strongest.

Report:

- number of adjacent capacity increases that improve;
- uncertainty of each difference;
- saturation;
- regressions;
- seed interactions.

## 48.9 Compute-optimal frontier

For a fixed wall-clock budget, compare combinations of:

- smaller model and deeper search;
- larger model and shallower search;
- more teacher compute and less dataset breadth;
- more data and weaker teacher.

The final production choice should lie near an empirically observed compute-efficient frontier.

---

# 49. Exact-game metrics

For exact positions, use:

## 49.1 WDL accuracy

Necessary but insufficient.

## 49.2 Cross-entropy and calibration

Measure:

- log loss;
- Brier score;
- reliability by probability bucket.

## 49.3 Optimal-action mass

\[
m(s)=\sum_{a\in A^*(s)}\pi(a\mid s).
\]

## 49.4 Decision regret

Measure the game-theoretic loss caused by the chosen action.

## 49.5 Exploitability

Play against a perfect opponent from:

- the start;
- principal opening states;
- stratified sampled states.

## 49.6 Stratification

By:

- depth to terminal;
- branching factor;
- WDL;
- number of optimal actions;
- phase;
- mobility;
- promotion proximity;
- training frequency.

---

# 50. Non-exact game metrics

Use:

- paired match score;
- relative Elo;
- confidence interval;
- pentanomial outcome counts where applicable;
- historical champion performance;
- raw evaluator performance;
- fixed-node performance;
- fixed-time performance;
- best-move stability;
- search depth;
- nodes per second;
- crash and illegal-move count.

---

# 51. Standard-chess evaluation

## 51.1 Separate tracks

### Track A: self-search structured learning

Allowed:

- exact reduced chess;
- internal self-play;
- internal deeper search;
- terminal outcomes;
- structural features.

Not allowed:

- Stockfish-labelled training;
- human game databases;
- opening-book knowledge;
- standard endgame-tablebase deployment.

### Track B: Stockfish-teacher diagnostic

Allowed:

- Stockfish WDL;
- Stockfish best moves;
- multiple teacher budgets.

Purpose:

- estimate representation ceiling;
- validate fitting;
- identify whether internal search is the limiting teacher.

Never silently mix tracks.

## 51.2 UCI support

Support at minimum:

```text
uci
isready
ucinewgame
position startpos moves ...
position fen ...
go nodes ...
go depth ...
go movetime ...
stop
quit
```

## 51.3 Match protocol

For reported results:

- one engine thread unless thread scaling is under test;
- fixed hash;
- pondering disabled;
- tablebases disabled;
- same time control;
- same starting-position set;
- colours reversed for each start;
- fixed random seed;
- all games saved;
- binaries and checkpoints hashed.

## 51.4 Opening positions

Using a frozen opening suite for evaluation is allowed to reduce variance.

The learned engine must not contain an opening book.

Do not present results from the standard start alone as robust Elo.

## 51.5 Stockfish anchors

Reduced `UCI_Elo` settings may be used as coarse anchors.

Report:

- exact Stockfish version;
- exact options;
- time control;
- threads;
- hash;
- openings;
- score;
- interval.

Do not call the resulting number FIDE Elo.

## 51.6 Required comparisons

- existing approximately 1000-Elo baseline;
- search-only baseline;
- raw sparse evaluator;
- structured evaluator without advanced search;
- structured evaluator with search;
- teacher-assisted diagnostic;
- self-search-trained model;
- same model at multiple search budgets.

---

# 52. Forward Chess evaluation

## 52.1 Rules specification

Maintain a reviewed `FORWARD_CHESS_RULES.md`.

It must define:

- coordinate directions;
- orientation;
- allowed vertical displacement;
- horizontal moves;
- attack restrictions;
- checks;
- castling;
- en passant;
- promotions;
- orientation reversal;
- repetition;
- draw rules;
- stalemate;
- move-count rules.

## 52.2 Internal rating pool

Include:

- random player;
- search-only baseline;
- previous raw evaluator;
- first structured evaluator;
- every major champion;
- selected deeper-search versions;
- selected ablation models.

## 52.3 Paired starting positions

Create a frozen diverse set containing:

- ordinary opening positions;
- quiet structures;
- early contact;
- advanced pieces;
- low mobility;
- promotion races;
- reversed promoted pieces;
- material imbalances;
- closed lanes;
- exposed kings;
- repetition candidates.

Play both colours.

## 52.4 Reduced exact suite

Maintain exact or near-exact suites covering:

- simple promotion races;
- reversed-piece tactics;
- king escape;
- commitment;
- blockades;
- support chains;
- waiting-move positions.

## 52.5 Human-readable analysis

For major champions, output feature decompositions for selected positions.

The decomposition should show:

- measured feature values;
- learned contributions;
- phase weights;
- move-order scores;
- search result by depth.

This is diagnostic, not a requirement that the evaluator be perfectly human-readable.

---

# 53. Recommended research stages

These stages are deliberately broader than a fully scripted implementation plan.

The agent should choose the smallest useful experiment within each stage.

---

# 54. Stage A — Recover and audit the existing system

## Questions

- What exactly produced the existing strength?
- How much came from search?
- What did the evaluator learn?
- Where did CPU time go?
- Which existing components are correct and reusable?

## Required work

- recover baseline revision and checkpoint;
- reproduce representative results;
- profile move generation, search, evaluation, training, and I/O;
- measure raw evaluator accuracy;
- measure search scaling;
- measure evaluator contribution;
- inspect parameter count and data volume;
- inspect code complexity and configuration surface.

## Required outputs

- baseline report;
- architecture map;
- performance profile;
- reproducibility gaps;
- list of reusable components;
- list of components requiring replacement;
- preserved baseline binary or checkpoint.

## Gate

Do not rewrite the repository merely because the new plan differs.

Retain sound code.

Replace only what evidence identifies as inadequate.

---

# 55. Stage B — Build the research instrumentation

## Questions

- Can the system generate trustworthy teacher labels?
- Can it measure shallow-to-deep residuals?
- Can it track target provenance and computational cost?
- Can it run reproducible paired experiments?

## Required capabilities

- deterministic node-limited search;
- completed-depth reporting;
- deep-search teacher records;
- counterfactual child search;
- position hashing and deduplication;
- typed dataset records;
- frozen evaluation sets;
- run manifests;
- paired arena;
- feature-cost measurement;
- shadow-search hooks;
- CPU utilization counters.

## Gate

No major feature research begins until the instrumentation can distinguish:

- model error;
- search error;
- move-ordering error;
- and computational cost.

---

# 56. Stage C — First compact structured evaluator

## Initial candidate families

- learned type/orientation counts;
- learned piece-square/orientation terms;
- basic mobility;
- immediate attacks and defences;
- promotion distance;
- side to move;
- simple phase observables.

## Candidate models

- linear WDL;
- generalized additive WDL;
- small phase mixture.

## Required comparisons

- previous raw sparse model;
- search-only baseline;
- linear structured model;
- additive structured model;
- terminal targets;
- deep-search targets;
- deep-search plus terminal calibration.

## Primary objective

Demonstrate superior sample efficiency and evaluator contribution.

## Gate

Proceed when the structured model:

- improves held-out teacher prediction;
- improves shallow-search move choice;
- and provides positive fixed-time value.

If it does not:

- inspect target quality;
- inspect symmetry;
- inspect feature extraction;
- test exact games;
- do not immediately add more feature families.

---

# 57. Stage D — Directed geometry and structural value

## Candidate priority

1. reachability;
2. commitment;
3. safe mobility;
4. support lag;
5. promotion interception;
6. reversed-piece utility;
7. king escape;
8. frontier and congestion.

## Method

Add one family at a time.

For each:

- fit;
- ablate;
- profile;
- test on Breakthrough;
- test on reduced Forward Chess;
- confirm on full Forward Chess when affordable.

## Gate

Retain only families with net search value.

A feature may be:

- retained at all leaves;
- retained only at root;
- retained only in move ordering;
- retained only for training labels;
- or rejected.

---

# 58. Stage E — Interactions and local patterns

## Trigger

Begin only when additive residual analysis shows systematic interactions.

## Candidate order

1. selected scalar pair interactions;
2. piece-pair relation tables;
3. state-action deltas;
4. systematic short local patterns;
5. small boosted model;
6. tiny residual network.

## Primary question

What is the smallest interaction mechanism that removes the important residual?

## Gate

Do not adopt a larger mechanism when a smaller one gives equivalent fixed-time strength.

---

# 59. Stage F — Learned move ordering

## Work

- collect final move ranks;
- fit compact action ranker;
- integrate with TT and history;
- measure node savings;
- compare generic versus Forward-specific features;
- inspect ranking calibration.

## Gate

Require:

- lower mean rank of final best move;
- fewer nodes;
- and improved fixed-time strength.

This stage may produce a larger gain than another static feature family.

---

# 60. Stage G — Tactical stabilization

## Candidate sequence

1. directional SEE;
2. capture/check/promotion quiescence;
3. mandatory structural quiet moves;
4. learned quiescence inclusion;
5. tactical correction history.

## Primary question

Can leaf values be made stable enough that deeper search does not repeatedly reverse obvious tactical conclusions?

## Gate

Measure:

- shallow/deep value reversal;
- tactical false positives;
- nodes;
- fixed-time strength.


---

# 61. Stage H — Learned selective search

## Candidate sequence

1. calibrated LMR;
2. empirical futility;
3. lazy evaluation;
4. ProbCut;
5. extensions;
6. cautious null move;
7. proof-oriented subsearch.

The agent may reorder this list based on the measured bottleneck.

## Required discipline

Every technique must use shadow or verification data before large match tuning.

## Gate

Adopt only techniques with positive fixed-time value and controlled error.

---

# 62. Stage I — Cross-game generality

## Purpose

Determine which ideas genuinely transfer.

## Minimum comparison

Use:

- Breakthrough;
- one independent structural game;
- reduced chess-like game;
- reduced Forward Chess.

## Questions

- Which features work across games?
- Which require directional movement?
- Which are chess-family-specific?
- Which search controllers transfer?
- Do scaling slopes remain positive?
- Does the architecture remain simple when adding a game?

## Architectural gate

Adding a new game should mainly add:

- rule code;
- feature adapters where intrinsically necessary;
- tests;
- configurations.

It should not restructure the training and search system.

---

# 63. Stage J — Standard-chess calibration

## Work

- UCI integration;
- standard correctness suite;
- teacher diagnostic;
- self-search structured training;
- Fastchess matches;
- fixed-node and fixed-time scaling;
- comparison with previous baseline.

## Key interpretation

A low standard-chess Elo does not necessarily invalidate Forward Chess progress.

However, standard chess remains valuable for detecting:

- weak tactical stabilization;
- poor move ordering;
- uncalibrated values;
- inadequate king safety;
- inability to exploit material;
- and excessive evaluation cost.

## Gate

Produce a reproducible relative strength estimate and a decomposition of gains.

---

# 64. Stage K — Full Forward Chess research loop

## Work

- train from scratch or from the selected structured baseline;
- maintain champion pool;
- generate deep teacher labels;
- mine disagreements;
- fit feature families;
- evaluate search scaling;
- evaluate data scaling;
- evaluate capacity scaling;
- inspect learned signs and phase effects;
- maintain exact reduced-suite performance.

## Required outputs

- internal rating curve;
- scaling plots;
- feature ablations;
- dynamic piece-utility analysis;
- structural case studies;
- compute efficiency;
- comparison with search-only and raw-model baselines.

---

# 65. Stage L — Scale and production hardening

## Trigger

Begin only when:

- the model learns predictably;
- larger data budgets help;
- search budget helps;
- CPU scaling is measured;
- and a small canary reproduces expected behavior.

## Work

- freeze recipe;
- freeze code;
- freeze evaluation suites;
- profile target VM;
- run canary;
- run capacity/data/search sweep;
- select compute-optimal model;
- run confirmation;
- generate final report.

---

# 66. Full-core CPU utilization

## 66.1 Resource detection

At process start, record:

- logical CPU count;
- physical core count if detectable;
- CPU model;
- SIMD features;
- NUMA topology if relevant;
- memory;
- disk;
- thread environment variables.

## 66.2 Self-play

Default:

- one independent game per worker;
- one search thread per game;
- immutable shared evaluator;
- per-worker mutable search state;
- deterministic independent RNG stream;
- bounded output buffer.

Use approximately all allocated logical CPUs unless hyperthread scaling is negative.

Measure both physical-core and logical-core scaling where possible.

## 66.3 Teacher relabelling

Parallelize positions independently.

Use static or work-stealing allocation depending on position-cost variance.

Avoid a serial producer bottleneck.

## 66.4 Training

During batch fitting:

- stop self-play workers;
- allocate all intended cores to fitting;
- explicitly control library thread counts;
- avoid nested Rayon and BLAS pools;
- measure parallel efficiency.

For simple linear and additive models, parallelize:

- feature extraction;
- gradient accumulation;
- objective evaluation;
- residual analysis.

## 66.5 Evaluation

Run independent paired games concurrently.

Each game receives the exact intended engine thread allocation.

Do not let every engine process use all VM cores.

## 66.6 Sweeps

Implement a small CPU-slot scheduler only when needed.

A sweep manifest should explicitly list:

```text
config
seed
requested cores
memory budget
expected output
```

The scheduler should:

- detect free slots;
- launch jobs whose allocation fits;
- capture exit status;
- avoid oversubscription;
- stop on systemic failure;
- resume completed manifests safely.

Do not build a daemon or remote service.

## 66.7 Utilization

Estimate:

\[
U=
\frac{\text{process CPU seconds}}
{\text{wall seconds}\times\text{allocated logical CPUs}}.
\]

Report for:

- self-play;
- relabelling;
- training;
- evaluation;
- sweeps.

A reasonable target for long embarrassingly parallel stages is at least roughly 80–90%, subject to memory bandwidth and I/O constraints.

## 66.8 Parallel efficiency

\[
E_N=
\frac{\text{throughput at }N}
{N\times\text{throughput at }1}.
\]

Measure:

```text
1, 2, 4, 8, 16, ...
```

Do not claim success from utilization alone. A fully utilized machine can still be doing unnecessary work.

---

# 67. Memory and disk discipline

The initial machine class may have only 32 GB of memory and disk.

## 67.1 Dataset storage

Use:

- compact typed binary records;
- sharded files;
- streaming readers;
- optional lightweight compression;
- position deduplication;
- target provenance;
- retention policies.

Do not store full search trees.

## 67.2 Retention

Keep permanently:

- run manifest;
- resolved configuration;
- summary metrics;
- selected checkpoints;
- final evaluation games;
- frozen test sets;
- decision reports.

Allow deletion or compression of:

- redundant intermediate teacher shards;
- rejected candidate checkpoints;
- verbose per-node logs;
- temporary feature matrices.

## 67.3 Bounded pipelines

Use bounded producer-consumer queues if needed.

Backpressure must prevent memory growth.

Do not add async Rust merely to implement a bounded pipeline.

## 67.4 Memory budgeting

Before a large run, estimate:

- evaluator weights;
- transposition tables per worker;
- search stacks;
- training batches;
- dataset buffers;
- OS reserve.

Avoid swap.

---

# 68. Performance engineering

## 68.1 Required counters

Record:

- move generations;
- make/unmake operations;
- search nodes;
- quiescence nodes;
- proof nodes;
- model evaluations;
- feature-family evaluations;
- TT probes and hits;
- beta cutoffs;
- LMR re-searches;
- false-prune samples;
- games;
- positions;
- training steps;
- bytes read and written;
- CPU seconds;
- wall seconds;
- peak RSS.

## 68.2 Feature cost

For every feature family:

- nanoseconds per full calculation;
- incremental update cost;
- percentage of leaf time;
- cache behavior where measurable;
- fixed-node gain;
- fixed-time gain.

## 68.3 Incremental state

Incrementally maintain cheap features where correctness remains clear:

- piece counts;
- piece-square contribution;
- frontier summaries;
- basic mobility where practical;
- support counts;
- Zobrist key;
- phase observables;
- local pattern activations.

Keep a slow full recomputation for differential testing.

## 68.4 Tiered evaluation

Potential tiers:

### Tier 0

- terminal;
- counts;
- piece-square;
- immediate promotion;
- basic mobility.

### Tier 1

- attacks;
- safe mobility;
- support;
- commitment;
- frontier;
- king ring.

### Tier 2

- arrival maps;
- graph robustness;
- path diversity;
- flow or cut features.

Use empirically calibrated bounds for lazy skipping.

## 68.5 Quantization

After model selection:

- test integer weights;
- fixed-point splines;
- compact pair tables;
- SIMD-friendly layout.

Require:

- prediction comparison;
- move-order comparison;
- fixed-node comparison;
- fixed-time comparison.

---

# 69. Correctness testing

## 69.1 Unit tests

Cover:

- move encoding;
- feature values;
- symmetry;
- WDL transforms;
- model serialization;
- configuration parsing;
- search bounds;
- table updates.

## 69.2 Property tests

- make/unmake restores state;
- make/unmake restores hash;
- all legal moves preserve invariants;
- illegal moves are rejected;
- feature recomputation equals incremental features;
- side swap negates advantage;
- valid symmetry preserves WDL;
- action IDs remain stable.

## 69.3 Differential tests

For every custom game:

- slow reference generator;
- optimized generator;
- randomized reachable positions.

For standard chess:

- trusted move generator;
- perft;
- random legal trajectories;
- FEN and move parsing.

For Forward Chess:

- slow rule-direct implementation;
- optimized orientation-aware implementation;
- hand-authored rule corpus;
- randomized differential testing.

## 69.4 Search equivalence

On small positions:

- exhaustive minimax versus alpha–beta;
- TT off versus on;
- ordering changes result only in node count;
- selective search versus reference full search on sampled positions;
- exact leaves produce exact root decisions.

## 69.5 Model tests

- deterministic fixed-seed initialization;
- finite outputs;
- gradient check on small models;
- checkpoint round-trip;
- symmetry;
- quantized versus floating tolerance;
- batch versus single-position equivalence.

## 69.6 Global learning tests

A small training run must demonstrate:

- loss reduction;
- improvement over initialization;
- exact regret reduction;
- or stronger paired play.

This is required before stage promotion.

## 69.7 Performance regression tests

Maintain microbenchmarks for:

- move generation;
- make/unmake;
- feature extraction;
- evaluation;
- TT;
- search on fixed positions.

Do not gate every commit on noisy microbenchmark thresholds, but investigate material regressions.

---

# 70. Experiment artifacts

Every run directory should contain:

```text
resolved.toml
manifest.json
metrics.jsonl
summary.json
parameter_ledger.json
checkpoint/
games/
datasets/
profiles/
report.md
stdout.log
stderr.log
```

The manifest must include:

- Git commit;
- dirty state;
- Rust toolchain;
- Cargo.lock hash;
- dependency versions;
- CPU;
- RAM;
- OS;
- build flags;
- thread allocation;
- seed;
- model parameter count;
- active feature count;
- exact command;
- start/end times;
- exit code.

---

# 71. Required stage report

```markdown
# Stage Report: <name>

## Status

PASS / FAIL / PARTIAL / INCONCLUSIVE

## Research question

## Prior evidence

## Hypothesis

## Implementation

## Deliberately excluded

## Baselines

## Data

## Target provenance

## Primary result

## Secondary results

## Statistical uncertainty

## Scaling behavior

| Axis | Values | Result | Monotonic? | Interpretation |
|---|---:|---:|---|---|
| Capacity | | | | |
| Data | | | | |
| Teacher search | | | | |
| Evaluation search | | | | |

## Fixed-node result

## Fixed-time result

## CPU result

| CPUs | Throughput | Efficiency | Utilization | Memory |
|---:|---:|---:|---:|---:|

## Feature or search cost

## Correctness evidence

## Ablations

## Failure cases

## Learned parameter interpretation

## Complexity delta

- Production LOC:
- Public types:
- Config keys:
- Dependencies:
- Permanent modes:
- New parameters:
- Deleted code:

## Parameter-provenance changes

## Decision

- Retain:
- Reject:
- Revise:
- Additional evidence:
- Selected recipe:

## Next research question

## Exact reproducibility information
```

---

# 72. Failure diagnosis

## 72.1 Structured model cannot fit exact strategy

Investigate:

- missing observables;
- symmetry bugs;
- insufficient nonlinear capacity;
- action-ID errors;
- optimization;
- data splits.

Do not add search complexity.

## 72.2 Structured model fits exact data but not deep-search data

Investigate:

- teacher instability;
- phase ambiguity;
- missing interactions;
- tactical noise;
- target calibration;
- distribution breadth.

## 72.3 Deep-search prediction improves but playing strength does not

Investigate:

- ranking versus calibration;
- errors concentrated on critical positions;
- model evaluation cost;
- move ordering;
- search horizon;
- teacher bias.

## 72.4 Fixed-node strength improves but fixed-time strength declines

Investigate:

- expensive feature families;
- incremental updates;
- quantization;
- tiered evaluation;
- smaller capacity;
- root-only use.

Do not automatically reject the feature. Determine its compute-optimal use.

## 72.5 More search makes play worse

Investigate:

- search bug;
- TT bounds;
- repetition;
- value instability;
- quiescence;
- selective-search unsoundness;
- evaluator perspective;
- draw handling.

This is a critical failure.

## 72.6 More data does not help

Investigate:

- saturation;
- duplicate positions;
- narrow self-play;
- poor labels;
- excessive model bias;
- optimization;
- data imbalance;
- stale teacher.

## 72.7 Larger models do not help

Investigate:

- insufficient data;
- regularization;
- optimization;
- feature bottleneck;
- model cost;
- teacher noise.

Do not assume still larger models will fix the problem.

## 72.8 Self-play collapses or cycles

Use:

- historical opponents;
- exact calibration;
- checkpoint pool;
- disagreement mining;
- broader starts;
- candidate promotion.

Do not immediately build a league framework.

## 72.9 Search selectivity gives unstable gains

Investigate:

- direct calibration;
- false-prune tails;
- phase guards;
- interaction with move ordering;
- interaction with quiescence;
- test-set leakage.

## 72.10 CPU utilization is low

Profile:

- serial data production;
- output locks;
- shared TT contention;
- nested thread pools;
- tiny work units;
- memory bandwidth;
- I/O;
- worker imbalance.

Do not build distributed infrastructure as the first response.

## 72.11 Standard chess remains weak

Determine whether the limiting factor is:

- tactics;
- king safety;
- conversion;
- move ordering;
- evaluator capacity;
- search pruning;
- or training labels.

Do not import a large standard-chess handcrafted evaluator into Forward Chess.

## 72.12 Forward Chess piece values remain unstable

This may be a real property.

Investigate:

- phase dependence;
- orientation;
- convertibility;
- interaction-component membership;
- promotion proximity;
- mobility;
- support.

Do not force one global value.

---

# 73. Decision tree for the agent

Use the following reasoning pattern.

## Case 1

**Exact supervised performance is poor.**

Then:

- improve representation or fitting;
- do not spend more on self-play.

## Case 2

**Exact supervised performance is strong, but self-search distillation is poor.**

Then:

- inspect teacher quality;
- improve data selection;
- compare targets;
- test TDLeaf or ranking;
- do not increase model size first.

## Case 3

**Prediction improves, but search nodes do not decrease.**

Then:

- train a separate move-ordering model;
- inspect whether value accuracy matters at searched leaves;
- measure ranking rather than only WDL.

## Case 4

**Fixed-node play improves, fixed-time play declines.**

Then:

- profile;
- tier features;
- incrementally update;
- quantize;
- reduce capacity;
- move expensive features to root or teacher only.

## Case 5

**Search depth helps, but evaluator capacity does not.**

Then:

- feature vocabulary is likely limiting;
- inspect residuals;
- add one justified structural family.

## Case 6

**Evaluator capacity helps, but more data does not.**

Then:

- inspect data diversity and label noise;
- improve disagreement mining;
- deepen the teacher;
- deduplicate.

## Case 7

**More data helps, but larger models do not.**

Then:

- remain with the small model;
- do not add capacity without residual evidence.

## Case 8

**A feature helps standard chess but not Forward Chess.**

Then:

- classify it as chess-specific;
- do not force transfer;
- investigate the analogous directional measurement.

## Case 9

**A feature helps Breakthrough and Forward Chess.**

Then:

- classify it as directional and prioritize refinement.

## Case 10

**A feature improves prediction but has no match gain.**

Then:

- inspect critical-position weighting;
- evaluator cost;
- search interaction;
- and calibration.

Reject it if no practical mechanism emerges.

---

# 74. Final deliverables

The final project should contain:

## 74.1 Engine

- standard chess UCI binary;
- Forward Chess binary or protocol;
- selected structured evaluator;
- learned move-ordering model;
- selected search-control mechanisms;
- reproducible checkpoints.

## 74.2 Research artifacts

- baseline reconstruction;
- feature-family reports;
- search-technique reports;
- scaling reports;
- exact-game results;
- standard-chess rating report;
- Forward Chess internal rating report;
- CPU scaling report;
- parameter-provenance ledger;
- complexity history.

## 74.3 Datasets

- frozen exact suites;
- frozen teacher test sets;
- disagreement suites;
- Forward Chess structural suites;
- standard-chess diagnostic set;
- match starting positions.

## 74.4 Analysis

- learned dynamic piece values;
- phase curves;
- feature contributions;
- material-versus-structure analysis;
- promotion transition analysis;
- commitment analysis;
- search-error analysis;
- compute-optimal frontier.

## 74.5 Reproduction

A technically capable person should be able to:

- build the project;
- run a small exact benchmark;
- fit a compact evaluator;
- reproduce one scaling point;
- run a standard-chess match;
- run a Forward Chess match;
- and regenerate the main reports.

---

# 75. Final satisfaction checklist

The project should not be declared complete until:

- [ ] The previous system’s result has been reconstructed or carefully approximated.
- [ ] Search and evaluator contributions have been separated.
- [ ] The structured evaluator beats the raw sparse baseline per unit data.
- [ ] At least one compact model exhibits positive capacity scaling.
- [ ] At least one game exhibits positive data scaling.
- [ ] At least one checkpoint exhibits positive search scaling.
- [ ] Fixed-node and fixed-time conclusions are both reported.
- [ ] Dynamic piece values outperform or clarify fixed piece values.
- [ ] Directed reachability has been tested.
- [ ] Commitment has been tested.
- [ ] Safe mobility and waiting resources have been tested.
- [ ] Promotion and reversed-piece features have been tested.
- [ ] Support or frontier structure has been tested.
- [ ] Move ordering has been trained separately.
- [ ] Tactical leaf stabilization is validated.
- [ ] Any retained pruning method has direct error calibration.
- [ ] Reduced Forward Chess passes exact or oracle tests.
- [ ] Full Forward Chess produces only legal games.
- [ ] Standard chess has a reproducible relative rating.
- [ ] Forward Chess has a stable internal rating pool.
- [ ] Large runs use the VM’s cores efficiently.
- [ ] Every parameter has provenance.
- [ ] Every retained feature family has an ablation.
- [ ] Rejected production paths have been deleted.
- [ ] The final selected system remains understandable.

---

# 76. Initial instruction to the agent

Begin with **Stage A: recover and audit the existing system**.

Do not begin by rewriting the architecture.

Before changing production code:

1. inspect the repository;
2. locate previous reports, checkpoints, and configurations;
3. identify the approximate 1000-Elo standard-chess result;
4. identify how much of that strength came from search;
5. profile the existing implementation;
6. enumerate existing public types, modules, configuration keys, dependencies, model parameters, and search mechanisms;
7. identify the smallest reusable vertical path;
8. propose the minimum instrumentation required for Stage B;
9. list everything that will deliberately not be implemented yet;
10. produce an audit report.

The first implementation change should be the smallest one required to make the existing baseline reproducible and measurable.

Do not begin by adding:

- reachability graphs;
- commitment features;
- a new neural network;
- MCTS;
- ProbCut;
- LMR tuning;
- a sweep scheduler;
- or a new game.

Establish the evidence base first.

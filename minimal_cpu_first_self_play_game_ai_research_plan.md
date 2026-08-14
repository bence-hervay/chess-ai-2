# Minimal CPU-First Self-Play Game AI

## Research Plan, Engineering Contract, Evaluation Methodology, and Phase-by-Phase Agent Prompt

This entire document is intended to be given to a capable coding and research agent. Treat it as the governing specification for the project.

The project must progress **one validated phase at a time**. Do not implement features belonging to later phases. At the end of each phase, produce the required report and stop. Continue only after the phase’s acceptance criteria have been met.

---

# 1. Agent role

You are acting as:

- a senior Rust systems engineer;
- a game-search researcher;
- a reinforcement-learning experimentalist;
- a performance engineer for multi-core CPU systems;
- and, importantly, a ruthless simplifier.

Your job is not merely to build a game-playing program. Your job is to build the **smallest defensible experimental system** that can answer the following research question:

> Can a compact model, trained from exact solutions and then search-guided self-play, learn increasingly sophisticated deterministic board games in a way that scales predictably with model size, training data, search effort, and CPU compute—and ultimately produce measurable strength in ordinary chess and strong play in Forward Chess?

You must prioritize:

1. correctness;
2. interpretable experiments;
3. reproducibility;
4. monotonic scaling;
5. CPU efficiency;
6. architectural simplicity;
7. raw playing strength, in that order.

Raw strength matters eventually, but early phases exist primarily to prove that the system learns for the right reasons and improves when more resources are supplied.

---

# 2. Mission

Build a Rust program that supports a progression of deterministic, two-player, alternating-turn, perfect-information, zero-sum games:

1. parameterized Connect-\(k\);
2. parameterized Breakthrough;
3. parameterized Othello;
4. standard chess;
5. Forward Chess.

The same core search, model, training, experiment, and evaluation code must be reused across games.

Only these components may be game-specific:

- legal state representation;
- legal move generation;
- move application and reversal;
- terminal and draw rules;
- raw rule-state encoding;
- stable action identifiers;
- parsing and display.

The following must **not** be game-specific:

- optimizer;
- self-play algorithm;
- model training loop;
- search implementation;
- checkpoint promotion;
- experiment logging;
- scaling analysis;
- statistical evaluation;
- CPU resource allocation.

The final system should be able to:

- solve sufficiently small instances exactly;
- learn exact strategies from oracle data;
- recover exact or nearly exact play through self-play;
- exhibit measurable model-, data-, and search-scaling;
- produce meaningful strength estimates in ordinary chess;
- train Forward Chess without handcrafted positional heuristics;
- make effective use of a VM containing dozens of logical CPU cores.

---

# 3. Non-goals

Do **not** attempt to build any of the following:

- a complete reproduction of Stockfish;
- a complete reproduction of AlphaZero;
- a general game-description language;
- a distributed cluster platform;
- a web service;
- a GUI;
- an opening-book system;
- an endgame-tablebase runtime for chess;
- a plugin architecture;
- a generic callback framework;
- an asynchronous actor–learner system;
- multiple interchangeable search engines;
- multiple interchangeable machine-learning frameworks;
- a permanent zoo of model architectures;
- a hierarchy of managers, controllers, factories, providers, registries, or backends;
- game-specific handcrafted positional evaluation terms.

Do not implement MCTS merely because it is common in reinforcement-learning papers. This project will initially use one search path only.

Do not implement optional future modes. Future ideas belong in the phase report, not in production code.

---

# 4. Fixed methodological direction

## 4.1 Primary method

Use **policy/value Expert Iteration with generic alpha–beta search**.

The loop is:

1. freeze the current model;
2. use it inside search to generate stronger decisions;
3. play self-play games using those searched decisions;
4. train the next model to:
   - predict game outcomes;
   - imitate the search-selected actions;
5. evaluate the candidate against exact oracles or the previous champion;
6. promote it only if it demonstrably improves;
7. repeat.

This is inspired by Expert Iteration: tree search acts as the expert, while a learned apprentice generalizes the expert’s decisions and then improves future searches. The original Expert Iteration work demonstrated this planning/generalization decomposition in Hex and reported a tabula-rasa search agent that defeated the publicly released MoHex 1.0. ([arxiv.org](https://arxiv.org/abs/1705.08439))

Giraffe is also relevant evidence: it used learned evaluation, move ordering, and search-selection components to acquire much of its chess knowledge through self-play rather than a conventional large handcrafted evaluation function. ([arxiv.org](https://arxiv.org/abs/1509.01549))

This project is not required to reproduce either system exactly.

## 4.2 Why alpha–beta rather than AlphaZero-style MCTS

Alpha–beta is the default because:

- it is especially effective in deterministic, alternating, zero-sum games;
- it scales strongly with improved move ordering;
- it is naturally CPU-friendly;
- it can use very small value networks;
- it does not require large batches of expensive neural inference;
- exact minimax and approximate search share almost all their logic;
- it gives a direct route to conventional chess evaluation.

The initial implementation must contain:

- negamax;
- alpha–beta pruning;
- iterative deepening;
- a transposition table;
- policy-based move ordering;
- deterministic node accounting.

It must initially exclude:

- null-move pruning;
- late-move reductions;
- futility pruning;
- singular extensions;
- aspiration windows;
- quiescence search;
- parallel search within one game.

Those may be considered only when a named benchmark demonstrates a specific deficiency.

## 4.3 Learned knowledge

The learned model may receive only **raw state facts**:

- which cell contains which piece or marker;
- owner or colour;
- piece orientation where orientation is part of the rules;
- side to move;
- castling or comparable legal rights;
- en-passant state;
- other rule-state bits strictly required to determine legal continuation.

The model must not receive handcrafted concepts such as:

- material score;
- mobility;
- king safety;
- pawn structure;
- passed pawns;
- centre control;
- connected components;
- piece-square bonuses;
- attack counts;
- space advantage;
- tempo bonuses;
- manually assigned piece values.

The model is expected to discover useful regularities from search-generated and outcome-generated targets.

---

# 5. Hard simplicity contract

These rules are binding.

## 5.1 The only core concepts

Production code may contain only these top-level conceptual subsystems:

1. `Game`
2. `Search`
3. `Model`
4. `Training`
5. `Evaluation`

Experiment storage and the command-line interface are supporting infrastructure, not additional algorithmic subsystems.

## 5.2 Abstraction admission rule

Do not introduce a public abstraction until at least two concrete, currently used implementations require it.

Examples:

- Do not introduce a `SearchStrategy` trait when only alpha–beta exists.
- Do not introduce a `ModelBackend` trait when only one ML framework exists.
- Do not introduce an `ExperimentStore` interface when runs are stored on the local filesystem.
- Do not introduce a distributed worker interface before multiple machines are actually used.
- Do not introduce game plugins; static Rust modules are sufficient.

## 5.3 Configuration admission rule

A new experiment configuration field may be added only when:

1. it corresponds to a real experimental variable;
2. the current phase will run at least two values of it;
3. the phase report will compare those values;
4. its meaning is intrinsic to the experiment.

Do not expose constants merely because they might be useful later.

Avoid boolean mode flags. A boolean usually creates two permanent code paths without providing enough information about why either path exists.

## 5.4 One-way evolution

When an experiment proves that implementation B should replace implementation A:

- migrate to B;
- preserve any useful A benchmark result;
- delete A from the production path;
- do not retain `mode = "old" | "new"`.

Small reference implementations may remain under tests when they provide a correctness oracle.

## 5.5 Change budget

For each phase, unless explicitly authorized by this document:

- add at most one new public trait;
- add at most three new public structs or enums;
- add at most two experiment configuration fields;
- add at most one production dependency;
- do not add an async runtime;
- do not add a database;
- do not add network services;
- do not add a new executable unless the phase explicitly calls for it.

## 5.6 Complexity audit

Every phase report must state:

- production lines added and removed;
- public types added and removed;
- configuration keys added and removed;
- dependencies added and removed;
- permanent algorithmic branches added;
- files created;
- the justification for each architectural increase.

A phase cannot be considered complete merely because tests pass. Unjustified complexity is a phase failure.

---

# 6. Technology and package-management requirements

## 6.1 Language and toolchain

Use stable Rust.

At repository creation:

- add `rust-toolchain.toml`;
- pin a specific stable toolchain;
- commit `Cargo.lock`;
- record the Rust version in every experiment manifest;
- use Rust 2021 or the newest stable edition supported by the pinned toolchain;
- compile CI builds portably;
- compile final VM performance builds with native CPU targeting.

Use:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
```

The project must build and run entirely through Cargo.

## 6.2 Package structure

Start as **one Cargo package**, not a workspace.

Recommended layout:

```text
Cargo.toml
Cargo.lock
rust-toolchain.toml
README.md
ARCHITECTURE.md
DECISIONS.md

src/
  lib.rs
  game.rs
  search.rs
  model.rs
  training.rs
  evaluation.rs
  experiment.rs

  games/
    mod.rs
    connect_k.rs
    breakthrough.rs
    othello.rs
    chess.rs
    forward_chess.rs

  bin/
    lab.rs
    uci.rs

configs/
  phase_00/
  phase_01/
  ...

manifests/
reports/
```

`uci.rs` must not be created before the chess phase.

Do not split the project into multiple crates merely to make the directory tree look architectural.

## 6.3 Dependency policy

Expected dependencies, introduced only when needed:

- `clap` for a typed CLI;
- `serde` and `toml` for explicit experiment configuration;
- `rand` and `rand_chacha` for reproducible randomness;
- `rayon` for process-local data parallelism;
- `thiserror` for typed errors;
- `tracing` and `tracing-subscriber` for structured local logging;
- one ML framework;
- `cozy-chess` for standard-chess rules and move generation;
- `proptest` and `criterion` as development dependencies.

Rayon is a lightweight work-stealing data-parallelism library designed to turn independent iterator work into CPU-parallel work while retaining Rust’s data-race guarantees. ([docs.rs](https://docs.rs/crate/rayon/latest))

Use **Burn with its current pure-Rust CPU backend** as the initial ML framework. Burn’s current documentation provides a pure-Rust CPU backend as well as autodifferentiation and model-storage support. Do not support multiple Burn backends in the application unless a later phase explicitly requires it. ([docs.rs](https://docs.rs/burn/latest/burn/index.html))

For standard chess, initially wrap `cozy-chess` rather than implementing standard chess move generation from scratch. It provides strongly typed board types, legal move generation, bitboards, and incrementally maintained Zobrist hashing suitable for engine use. ([docs.rs](https://docs.rs/cozy-chess/latest/cozy_chess/))

## 6.4 Forbidden package practices

Do not:

- use unpinned Git dependencies on a main branch;
- depend on two crates that solve the same problem;
- introduce feature flags for hypothetical users;
- make every dependency optional;
- use `unsafe` without a benchmark showing a material need;
- add a metrics server;
- add a cloud experiment service;
- add a workflow engine;
- add a configuration framework beyond typed TOML deserialization.

---

# 7. Command-line surface

The `lab` binary may expose only these commands:

```text
lab solve <config>
lab train <config>
lab evaluate <config>
lab sweep <manifest>
lab report <run-directory>
```

Before the phase requiring each command, it may be absent.

The late-stage `uci` binary should expose the trained chess engine through the standard UCI protocol.

Do not add command aliases, nested mode trees, interactive menus, or a general job-control language.

---

# 8. Typed configuration

Do not create one enormous configuration object full of `Option<T>` fields.

Use a separate required configuration type for each command:

- `SolveConfig`
- `TrainConfig`
- `EvaluationConfig`
- `SweepManifest`

A resolved training configuration should contain approximately:

```toml
game = "breakthrough_6x6"
seed = 2

model_width = 128
search_nodes = 4096

self_play_games = 20000
training_steps = 10000

threads = 32
```

The exact format may use small nested sections if that improves clarity, but every field should be required and semantically meaningful.

The following should remain fixed in a versioned training recipe rather than becoming routine knobs:

- number of hidden layers;
- activation function;
- optimizer family;
- loss weighting;
- replay policy;
- initialization scheme;
- normalization convention;
- exploration mechanism.

Changing one of those requires a named experiment and a decision entry.

Every run directory must contain a **fully resolved configuration** with no implicit or environment-dependent fields.

---

# 9. Core game interface

Use static dispatch inside the search and training loops. A top-level enum may dispatch CLI choices to generic functions.

A suitable conceptual interface is:

```rust
pub trait Game: Send + Sync + 'static {
    type State: Clone + Send + Sync;
    type Move: Copy + Eq + Ord + Send + Sync;
    type Undo;

    fn initial_state(&self) -> Self::State;

    fn side_to_move(&self, state: &Self::State) -> Player;

    fn legal_moves(
        &self,
        state: &Self::State,
        moves: &mut Vec<Self::Move>,
    );

    fn make_move(
        &self,
        state: &mut Self::State,
        mv: Self::Move,
    ) -> Self::Undo;

    fn unmake_move(
        &self,
        state: &mut Self::State,
        mv: Self::Move,
        undo: Self::Undo,
    );

    fn outcome(
        &self,
        state: &Self::State,
        history: &PositionHistory,
    ) -> Option<Outcome>;

    fn position_key(&self, state: &Self::State) -> u64;

    fn encode_features(
        &self,
        state: &Self::State,
        features: &mut Vec<FeatureId>,
    );

    fn action_id(
        &self,
        state: &Self::State,
        mv: Self::Move,
    ) -> ActionId;
}
```

This is illustrative rather than mandatory syntax.

## 9.1 Interface restrictions

The game interface must not expose:

- material values;
- move-quality hints;
- capture ordering;
- positional features;
- search extensions;
- game-specific evaluation scores.

`encode_features` must encode facts, not conclusions.

## 9.2 Perspective normalization

Encode positions from the perspective of the player to move wherever practical.

For example:

- “own rook on square 12” and “opponent rook on square 51”;
- not permanently “white rook” and “black rook” unless the game rules make absolute colour necessary.

This reduces the burden on the model without injecting strategic knowledge.

## 9.3 Stable action IDs

Each legal move must map to a stable integer action ID.

Examples:

- Connect-\(k\): destination column or square;
- Othello: destination square;
- Breakthrough: from-square × destination-square;
- chess: from-square × destination-square × promotion code;
- Forward Chess: the same broad mapping as chess.

Action IDs are a rule-level representation, not a heuristic.

---

# 10. Model specification

## 10.1 Fixed model family

Use one compact sparse policy/value network family.

For a position with active sparse feature IDs \(f_1,\ldots,f_n\):

1. look up an embedding for every active feature;
2. sum the embeddings;
3. apply a fixed two-layer MLP;
4. produce:
   - three WDL logits;
   - a hidden state used to score legal actions.

Conceptually:

\[
x = \frac{1}{\sqrt{n}}\sum_{i=1}^{n} E_{f_i}
\]

\[
h_1 = \operatorname{ReLU}(W_1x+b_1)
\]

\[
h_2 = \operatorname{ReLU}(W_2h_1+b_2)
\]

\[
\ell_{\mathrm{WDL}} = W_vh_2+b_v
\]

For legal action \(a\):

\[
\ell_a = A_a^\top h_2 + c_a
\]

Only legal-action logits need to be calculated.

## 10.2 Capacity knob

The primary model-capacity parameter is `model_width`.

The number of hidden layers remains fixed.

Initial scaling widths should be log-spaced, for example:

```text
32, 64, 128, 256
```

For tiny games, smaller values may be used to prevent immediate saturation.

Do not add choices for:

- activation type;
- arbitrary depth;
- residual blocks;
- attention;
- convolution;
- transformer layers;
- mixture-of-experts.

## 10.3 WDL head

Use three outputs:

- win;
- draw;
- loss.

The scalar search value is:

\[
v = p(\mathrm{win}) - p(\mathrm{loss})
\]

This representation works for games with and without draws.

## 10.4 Policy head

Train the policy head to imitate the strongest move found by search.

For exact positions:

- divide policy target mass uniformly across all game-theoretically optimal moves.

For approximate self-play search:

- use the best move from the last completed iterative-deepening search;
- ties at the same completed depth may share target mass;
- do not fabricate a large pseudo-probability distribution from incomplete alpha–beta bounds.

## 10.5 Loss

Use:

\[
L = L_{\mathrm{WDL}} + L_{\mathrm{policy}}
\]

where both terms are ordinary cross-entropy losses.

Do not add auxiliary heads during the initial research program.

Do not train on handcrafted centipawn values.

## 10.6 Efficient inference path

Initially use the ML framework for training and inference.

At the performance-gating phase, profile search.

Only if model inference accounts for a material fraction of search time should a packed, single-position inference implementation be added.

The packed implementation must:

- consume the same trained weights;
- agree numerically with framework inference within a documented tolerance;
- replace the framework path inside search;
- not become a separate configurable model backend.

## 10.7 Optional NNUE-like incrementality

Do not implement incremental accumulators initially.

An NNUE-like accumulator may be added only if:

1. packed inference is already implemented;
2. model evaluation remains a major measured bottleneck;
3. make/unmake operations expose exact sparse feature deltas;
4. a benchmark shows a material end-to-end search improvement.

The motivation is sound: NNUE architectures exploit sparse inputs, small changes between consecutive positions, and very cheap CPU inference. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html))

Do not copy Stockfish’s chess-specific input architecture unless a later chess-only experiment explicitly justifies it.

---

# 11. Search specification

## 11.1 Exact solver

Implement an exact memoized negamax solver for sufficiently small acyclic or finitely repeated games.

It should return:

- WDL value;
- complete set of optimal actions where feasible;
- optional distance-to-terminal for diagnostics.

The exact solver is a testing and research oracle. It is not the production search engine.

## 11.2 Approximate search

Implement:

- negamax;
- alpha–beta pruning;
- iterative deepening;
- transposition table;
- terminal-state detection;
- repetition handling;
- policy-based move ordering;
- deterministic node limits.

Move order:

1. transposition-table best move;
2. descending learned policy score;
3. stable action-ID order for ties.

Search value at leaves comes from the model’s WDL expectation.

## 11.3 Node budgets

Training and most controlled experiments should use node budgets rather than wall-clock budgets.

At each move:

1. perform iterative deepening;
2. stop when the node budget is exhausted;
3. return the result from the last fully completed depth;
4. record incomplete-depth work separately.

This makes algorithmic experiments reproducible.

## 11.4 Time budgets

Use wall-clock time controls for practical chess-engine strength tests.

Report both:

- fixed-node strength;
- fixed-time strength.

A larger model may improve quality per node while reducing nodes per second. Both effects matter.

## 11.5 No speculative pruning

Until a specific benchmark requires otherwise, do not add:

- null-move pruning;
- futility pruning;
- late-move reductions;
- razoring;
- singular extensions;
- aspiration windows;
- game-specific tactical extensions.

## 11.6 Quiescence policy

Do not implement quiescence search during the exact and intermediate-game phases.

At the standard-chess phase, first measure:

- frequency of large one-ply evaluation reversals;
- blunder rate near fixed-depth horizons;
- improvement from one additional full search ply;
- search time spent evaluating immediately unstable positions.

Only if these measurements identify a clear horizon problem may a minimal chess-specific leaf-stabilization experiment be run.

If adopted, it must be described as a chess search optimization and must not alter the learned input features.

---

# 12. Self-play and Expert Iteration

## 12.1 Synchronous generations

Use synchronous generations.

For generation \(g\):

1. freeze champion model \(M_g\);
2. generate a fixed number of self-play games using \(M_g\);
3. create a training dataset;
4. warm-start candidate \(M_{g+1}\) from \(M_g\);
5. perform a fixed number of optimizer updates;
6. evaluate \(M_{g+1}\);
7. promote or reject;
8. checkpoint.

Do not update model weights while a self-play generation is being produced.

## 12.2 Training examples

For every selected training position, store:

- sparse state features;
- legal action IDs;
- search-selected expert action or optimal-action set;
- eventual game WDL from that player’s perspective;
- search node count;
- completed search depth;
- model/checkpoint identifier;
- game/ruleset identifier.

## 12.3 Value target

The initial value target is the final game outcome.

Do not initially blend:

- terminal outcome;
- leaf evaluation;
- searched root score;
- TD target;
- Stockfish evaluation.

If terminal outcome learning proves too slow on an exactly evaluable benchmark, run one controlled comparison:

1. terminal outcome only;
2. completed-search root value plus terminal outcomes.

Choose one, document it, and delete the losing production path.

## 12.4 Exploration

The expert search selects its best move.

During training self-play:

- with probability \(1-\epsilon\), play the expert move;
- with probability \(\epsilon\), sample a legal move from the apprentice policy;
- the policy target remains the expert move, not the exploratory move.

Run a one-time exact-game calibration over a small set such as:

```text
epsilon = 0.05, 0.10, 0.20
```

Select the value that gives the most reliable and sample-efficient recovery of exact strategy across seeds. Freeze it afterward.

Do not add an elaborate exploration schedule unless a later benchmark proves it necessary.

## 12.5 Replay

Begin with data from the current generation only.

If exact-game metrics reveal measurable catastrophic forgetting, add one fixed FIFO window covering a small number of recent generations.

Do not add:

- prioritized replay;
- novelty replay;
- opponent-conditioned replay;
- multiple replay classes;
- reservoir sampling plus recent sampling;
- arbitrary replay mixing coefficients.

## 12.6 Candidate promotion

For exact games, promote candidates according to oracle metrics.

For non-exact games:

- play paired matches against the current champion;
- use identical starting positions with colours swapped;
- promote only when the candidate’s lower confidence bound exceeds the required threshold.

A candidate that fails promotion must not trigger automatic hyperparameter mutation. Stop and diagnose.

## 12.7 Historical checkpoints

Retain historical champions for evaluation.

Do not use a training league initially.

A historical opponent league may be considered only if:

- candidate-versus-champion results become cyclic;
- performance against older champions regresses materially;
- the evidence is included in a phase report.

---

# 13. Experiment and run storage

Every run must have a self-contained directory:

```text
runs/<timestamp>-<game>-<short-git-sha>/
  resolved.toml
  manifest.json
  metrics.jsonl
  summary.json
  checkpoint/
  games/
  stdout.log
  report.md
```

The manifest must include:

- Git commit;
- dirty-working-tree status;
- Rust toolchain;
- Cargo.lock hash;
- operating system;
- CPU model;
- logical CPU count;
- allocated thread count;
- RAM;
- build flags;
- model parameter count;
- seed;
- exact command;
- start and completion times;
- exit status.

Write checkpoints atomically:

1. write temporary file;
2. flush;
3. rename into place.

A run must be resumable at generation boundaries.

Do not implement mid-game distributed recovery.

---

# 14. Testing methodology

The project needs four distinct forms of testing.

## 14.1 Unit and property tests

Run in ordinary CI.

Required properties include:

- legal moves never mutate the source state;
- make/unmake restores byte-equivalent or field-equivalent state;
- make/unmake restores the original position key;
- all generated moves are legal;
- applying a legal move preserves game invariants;
- terminal positions generate no illegal continuations;
- side-to-move perspective negation is consistent;
- position encodings are deterministic;
- action IDs are stable;
- serialization round-trips;
- model checkpoint loading reproduces outputs;
- fixed seed plus fixed node budget produces deterministic search.

Use property-based testing for state transitions.

## 14.2 Differential tests

Maintain a slow, obviously correct reference implementation for small custom games.

For standard chess:

- compare legal moves and outcomes with `cozy-chess`;
- run standard perft suites;
- compare random legal trajectories;
- verify FEN parsing and move application.

For Forward Chess:

- implement a slow reference move generator before optimizing it;
- compare the optimized generator against the reference on randomized reachable positions.

## 14.3 Search equivalence tests

On small positions:

- exhaustive minimax and alpha–beta must return identical values;
- alpha–beta with and without transposition tables must return identical values;
- move ordering may alter node count but not result;
- model leaves replaced by exact oracle values must produce optimal root actions;
- node-budget stopping must return the last complete iteration;
- repetition and draw adjudication must be independently tested.

## 14.4 Global learning tests

These are essential and are not ordinary local unit tests.

A global learning test should:

1. initialize a model from a fixed seed;
2. generate data or self-play from scratch;
3. train for a fixed small budget;
4. evaluate against an exact oracle;
5. verify that the resulting regret is below a broad but meaningful threshold.

Run:

- a short deterministic version on important pull requests;
- a three-seed version before phase promotion;
- a larger version as a scheduled or explicit benchmark.

Do not make every small commit wait for a long stochastic training test.

---

# 15. Performance methodology

## 15.1 Required counters

Record:

- game states generated;
- legal moves generated;
- search nodes;
- leaf model evaluations;
- transposition-table probes;
- transposition-table hits;
- self-play games;
- self-play positions;
- training examples;
- optimizer steps;
- wall-clock seconds;
- process CPU seconds;
- peak resident memory;
- bytes read and written.

## 15.2 Throughput metrics

Report:

- move generations per second;
- make/unmake pairs per second;
- model evaluations per second;
- search nodes per second;
- self-play positions per core-hour;
- completed games per core-hour;
- training examples per core-second;
- evaluation games per core-hour.

## 15.3 CPU utilization

Define approximate utilization as:

\[
U =
\frac{\text{process CPU seconds}}
{\text{wall seconds}\times\text{allocated logical CPUs}}
\]

Report this for:

- self-play;
- training;
- evaluation;
- sweeps.

Aim for:

- at least 85% during long self-play;
- at least 80% during long tournament evaluation;
- high utilization during training, subject to memory-bandwidth limits.

A low utilization result is not automatically an architecture problem. Profile before changing the design.

## 15.4 Parallel efficiency

Measure at:

```text
1, 2, 4, 8, 16, ... logical CPUs
```

Define:

\[
E_N =
\frac{\text{throughput with }N\text{ CPUs}}
{N\times\text{single-CPU throughput}}
\]

Report both throughput and efficiency.

Do not claim good scaling merely because total throughput increased.

---

# 16. Multi-core execution policy

## 16.1 Self-play

The default unit of CPU parallelism is **one independent game per worker**.

For a VM with \(C\) logical CPUs:

- reserve zero or one core for orchestration and operating-system overhead;
- use approximately \(C-1\) one-threaded self-play workers;
- share the frozen model immutably;
- give each worker an independent deterministic RNG stream derived from the run seed;
- give each worker its own search stack and transposition table.

Do not parallelize a single alpha–beta tree during the early phases.

Independent-game parallelism is easier to test, more deterministic, and generally scales better for training-data production.

## 16.2 Training

During the training stage:

- stop self-play workers;
- allow the ML backend to use the full allocated CPU count;
- avoid nested Rayon and BLAS thread pools;
- explicitly set thread counts;
- log the resolved thread settings.

The initial system should alternate between:

1. full-core self-play;
2. full-core training;
3. full-core evaluation.

Do not build a simultaneous asynchronous pipeline merely to keep every thread busy during every second.

## 16.3 Evaluation

For internal game evaluation:

- run independent paired games concurrently;
- use one search thread per playing agent;
- allocate approximately one active CPU per concurrent game.

For UCI chess evaluation, use Fastchess.

Fastchess supports concurrent matches, fixed-node or time controls, colour-repeated openings, SPRT, pentanomial reporting, CPU affinity, and detailed PGN statistics. ([github.com](https://github.com/Disservin/fastchess/blob/master/man.md))

## 16.4 Sweeps

Do not implement a scheduler until the first meaningful sweep.

When required, `lab sweep` must:

- read an explicit list of fully resolved run configurations;
- allow each run to request a fixed number of CPU slots;
- launch child processes while total allocated slots do not exceed available CPUs;
- capture exit status and run directory;
- stop launching new runs after a systemic failure;
- contain no database, daemon, web UI, retry policy language, or remote execution support.

Do not implement a Cartesian-product configuration DSL. Generate the explicit manifest before running it.

## 16.5 Oversubscription

Never allow:

- every self-play worker to spawn its own full Rayon pool;
- every engine in a tournament to use all machine threads;
- BLAS threads multiplied by experiment-process concurrency;
- a sweep to ignore per-run thread allocation.

Every process must print its thread plan before starting substantial work.

---

# 17. Scaling-law methodology

Raw strength on tiny games is not the principal result. The principal result is whether additional resources produce reliable gains.

Study three primary axes:

1. model parameters \(P\);
2. generated training positions \(D\);
3. search nodes per move \(S\).

## 17.1 Do not begin with a full factorial sweep

For each new game, choose a baseline configuration and perform three one-dimensional sweeps:

### Model sweep

Hold \(D\) and \(S\) constant. Vary:

```text
P1, 2P1, 4P1, 8P1
```

Usually this corresponds to four model widths.

### Data sweep

Hold \(P\) and \(S\) constant. Vary:

```text
D1, 4D1, 16D1, 64D1
```

### Search sweep

Use one frozen checkpoint. Vary:

```text
S1, 4S1, 16S1, 64S1
```

After those sweeps, run only a small local grid around the most promising region.

## 17.2 Seeds

Use:

- at least three seeds for exact and intermediate games;
- at least three seeds for important hyperparameter decisions;
- two seeds for expensive chess pilot runs;
- one large run only after smaller runs establish stability;
- a second confirmation run before making a strong final claim.

## 17.3 Metrics for exact games

Use:

- game-theoretic decision regret;
- optimal-move accuracy;
- policy mass on optimal moves;
- WDL accuracy;
- WDL log loss;
- Brier score;
- exploitability against perfect play;
- start-position result;
- performance stratified by depth and branching factor.

Do not use Elo as the primary exact-game metric.

## 17.4 Metrics for non-exact games

Use:

- paired match score;
- Elo difference with confidence interval;
- fixed-node performance;
- fixed-time performance;
- strength against current champion;
- strength against historical champions;
- raw model performance without search;
- performance at multiple evaluation search budgets.

## 17.5 Scaling fits

For a local empirical strength region, begin with:

\[
M =
c + \alpha\log_2 P +
\beta\log_2 D +
\gamma\log_2 S
\]

where \(M\) is Elo or another approximately linear strength metric.

For exact regret \(R\), consider:

\[
R =
R_\infty +
aP^{-\alpha} +
bD^{-\beta} +
cS^{-\gamma}
\]

Do not add interaction terms until residual plots show a systematic need.

Do not call four points a universal scaling law. Report:

- fitted slope;
- confidence interval;
- residuals;
- monotonicity;
- observed range;
- likely saturation.

## 17.6 Required scaling conclusions

For every promoted game phase, answer:

1. Does more search reliably improve the same checkpoint?
2. Does more self-play data reliably improve a fixed model?
3. Does a larger model improve at fixed node count?
4. Does the larger model still improve at fixed wall-clock time?
5. Where does each curve begin to saturate?
6. How much variance exists across seeds?
7. Which resource is currently limiting?

## 17.7 Fixed-node versus fixed-time

Always distinguish:

### Fixed-node evaluation

Measures decision quality per searched state.

### Fixed-time evaluation

Measures practical engine strength, including inference speed and move-generation efficiency.

A model that improves fixed-node strength but loses fixed-time strength is not necessarily a failed model. It may be too large for the available compute.

---

# 18. Selecting benchmark instances

For each parameterized game, create three rungs.

## 18.1 Oracle rung

Choose the largest instance that the exact solver can solve on a local machine within a modest fixed budget.

This provides:

- complete WDL labels;
- complete optimal-action labels;
- exact regret;
- exploitability.

## 18.2 Scaling rung

Choose a harder instance where:

- the exact solver can still label a large evaluation sample;
- a weak model is measurably imperfect;
- stronger models can improve without immediately reaching 100%;
- runs remain cheap enough for several seeds.

## 18.3 Strategic rung

Choose the next size where full solution is not required.

Use:

- champion matches;
- deeper search;
- historical checkpoints;
- fixed evaluation positions.

Do not choose one fixed board size forever. Increase difficulty when all configurations saturate, and reduce it when no configuration learns.

---

# 19. Phase overview

| Phase | Primary result |
|---|---|
| 0 | Correct, deterministic Rust game and experiment skeleton |
| 1 | Exact solver and validated alpha–beta search |
| 2 | Compact model can fit exact strategy |
| 3 | Self-play recovers exact strategy and exhibits scaling |
| 4 | Breakthrough demonstrates forward-race and positional learning |
| 5 | Othello demonstrates independent positional learning |
| 6 | Correct standard-chess and UCI integration |
| 7 | Measurable chess strength and chess scaling curves |
| 8 | Correct Forward Chess implementation and reduced exact oracles |
| 9 | Forward Chess self-play, champion ladder, and scaling |
| 10 | Frozen many-vCPU production experiment |

---

# 20. Phase 0 — Minimal deterministic foundation

## Objective

Create the smallest correct executable experimental skeleton.

## Implement

- one Cargo package;
- typed CLI;
- `Game` trait;
- parameterized Connect-\(k\);
- random legal agent;
- paired arena;
- deterministic seeded RNG;
- run directories and resolved configs;
- unit/property tests;
- basic throughput counters.

Support:

- width;
- height;
- target \(k\);
- gravity on or off.

## Do not implement

- neural networks;
- alpha–beta;
- exact solver;
- self-play training;
- sweep scheduler;
- general board abstraction;
- other games.

## Required tests

- move-generation invariants;
- make/unmake;
- terminal detection;
- hashing;
- colour swap;
- gravity;
- no-gravity mode;
- deterministic matches from fixed seed.

## Required performance result

Measure random-versus-random games with:

```text
1, 2, 4, 8, ... workers
```

Verify that independent game execution uses the available cores sensibly.

## Acceptance criteria

- all tests pass;
- fixed seeds reproduce identical game records;
- no unexplained nondeterminism;
- no more than one game implementation;
- no model/search abstractions;
- parallel arena shows useful scaling;
- phase report contains the complexity audit.

---

# 21. Phase 1 — Exact solving and search correctness

## Objective

Build an exact oracle and prove the production search is mathematically correct on small games.

## Implement

- exhaustive negamax;
- memoization;
- alpha–beta pruning;
- iterative deepening;
- transposition table;
- deterministic node counts;
- exact optimal-action enumeration;
- exact evaluation corpus generation.

## Required experiments

For several Connect-\(k\) sizes:

1. exhaustive minimax versus alpha–beta;
2. transposition table off versus on;
3. natural ordering versus deliberately reversed ordering;
4. exact node counts;
5. solve-time scaling by board size.

## Required result

Alpha–beta and exhaustive minimax must agree on every tested reachable position.

The transposition table and move ordering may reduce work but may never alter the result.

## Acceptance criteria

- 100% value agreement;
- 100% optimal-action agreement;
- zero incorrect start-state solutions;
- table on state count, node count, solve time, and memory;
- exact solver remains a separate test/research utility;
- no neural code added.

---

# 22. Phase 2 — Supervised oracle ceiling

## Objective

Determine whether the intended model can represent exact strategy before attempting self-play.

## Implement

- sparse policy/value network;
- WDL loss;
- policy loss;
- exact-state dataset;
- deterministic training;
- model save/load;
- raw-model oracle evaluation.

## Dataset construction

If all states fit comfortably:

- enumerate all reachable states;
- stratify train/validation/test by position hash;
- ensure no duplicate states cross splits.

If the game is larger:

- sample by:
  - ply;
  - WDL class;
  - branching factor;
  - distance to terminal.

## Required experiments

### Capacity sweep

Train at four model widths.

### Data sweep

Train with four dataset sizes.

### Seed sweep

Use at least three seeds for the baseline and likely best configuration.

### Memorization test

Overfit a tiny fixed batch.

### Generalization test

Evaluate on held-out states and held-out game trajectories.

## Required metrics

- WDL accuracy;
- WDL log loss;
- optimal-action accuracy;
- policy mass on all optimal actions;
- game-theoretic regret of raw model play;
- parameter count;
- training examples per second;
- CPU utilization.

## Acceptance criteria

On at least one nontrivial exact instance:

- WDL accuracy is near the achievable ceiling;
- optimal-action mass is high;
- larger models improve until saturation;
- more data improves until saturation;
- results are stable across seeds;
- checkpoint save/load is numerically stable;
- the agent can state whether remaining error is caused by model capacity, data, or optimization.

Do not proceed to self-play while the supervised ceiling is poor.

---

# 23. Phase 3 — Self-play recovery of exact strategy

## Objective

Show that search-guided self-play can recover the strategy without oracle labels during training.

The oracle remains available only for evaluation.

## Implement

- synchronous self-play generations;
- expert search action labels;
- terminal WDL labels;
- warm-started candidates;
- candidate promotion;
- one fixed exploration mechanism;
- generation-level checkpointing.

## Required diagnostic matrix

Run these four conditions:

| Training labels | Evaluation search |
|---|---|
| Exact oracle | No search |
| Exact oracle | Learned model plus search |
| Self-play | No search |
| Self-play | Learned model plus search |

This separates:

- representation failure;
- optimization failure;
- search failure;
- self-play failure.

## Required one-time exploration experiment

Compare a small fixed set of exploration probabilities.

Choose the best robust value and freeze it.

## Required scaling experiments

Measure:

- model width;
- self-play positions;
- search nodes during data generation;
- search nodes during evaluation.

## Acceptance criteria

On the exact benchmark:

- self-play reliably improves from random initialization;
- the searched agent approaches exact play;
- the raw model also improves;
- exploitability declines with training;
- larger data budgets improve results;
- larger evaluation search budgets improve results;
- at least two of three seeds reach the required oracle threshold;
- failures are diagnosable rather than silent.

A suitable default threshold is at least 99% oracle-optimal decisions on the principal held-out exact corpus, but the report must include regret and exploitability rather than only top-1 accuracy.

---

# 24. Phase 4 — Breakthrough curriculum

## Objective

Test forward progress, blocking, races, sacrifices, and positional structure without chess complexity.

## Implement

Parameterized Breakthrough with:

- configurable board width and height;
- ordinary forward moves;
- diagonal captures;
- terminal back-rank or elimination rule as defined by the selected ruleset.

Do not change search, model, or training unless a generic correctness bug is discovered.

## Curriculum

Suggested ladder:

1. 4×4 exact;
2. 5×5 exact or heavily sampled oracle;
3. 6×6 strategic;
4. larger board only if needed for scaling headroom.

The exact cutoff is determined empirically by solver cost.

## Required tests

- slow reference move generator;
- make/unmake;
- colour reflection;
- terminal races;
- blocked positions;
- exact solver on small boards.

## Required experiments

- supervised oracle ceiling on the largest practical exact size;
- self-play recovery on that size;
- self-play on the next size;
- model-width sweep;
- data sweep;
- search sweep;
- champion progression;
- search-depth disagreement analysis.

## Phase-integrity gate

Adding Breakthrough should primarily touch:

- `games/breakthrough.rs`;
- test data;
- configurations;
- reports.

If model, search, or training require structural changes, the report must explain which generic requirement was absent and why it could not have been represented through the existing interfaces.

## Acceptance criteria

- exact small-board play is near-perfect;
- medium-board candidates improve over generations;
- more search improves a frozen checkpoint;
- more self-play improves fixed model capacity;
- model scaling exhibits either improvement or a clearly measured saturation;
- no handcrafted race, material, or blockage features are introduced.

---

# 25. Phase 5 — Othello as an independent positional benchmark

## Objective

Show that the system learns a qualitatively different positional game where immediate piece count can be misleading.

## Implement

Parameterized Othello:

- 4×4 exact;
- 6×6 strategic;
- 8×8 optional only after 6×6 scaling is established.

## Why Othello

Othello provides:

- delayed positional consequences;
- mobility and access effects;
- material reversals;
- a simple action representation;
- exact small-board evaluation;
- a meaningful medium-board strategic challenge.

The point is not to build a specialist Othello engine. The point is to test whether the generic learner discovers useful nonlocal structure.

## Restrictions

Do not encode:

- corners;
- stable discs;
- mobility;
- frontier discs;
- parity;
- edge bonuses.

Only encode board occupancy and side to move.

## Required experiments

Repeat the same experimental template used for Breakthrough.

Compare:

- raw model;
- model plus search;
- deeper search;
- larger model;
- more self-play.

## Acceptance criteria

- exact 4×4 play is near-perfect;
- 6×6 strength improves across promoted checkpoints;
- fixed-search performance improves with data;
- the same model/search/trainer code is used unchanged;
- no Othello strategic concepts appear in the evaluator.

If this phase passes cleanly, the pipeline has demonstrated more than one style of positional learning.

---

# 26. Phase 6 — Standard chess integration

## Objective

Introduce chess rules and UCI compatibility without yet demanding high chess strength.

## Implement

- standard chess wrapper using `cozy-chess`;
- raw chess feature encoding;
- stable chess action mapping;
- repetition and draw handling;
- UCI executable;
- fixed-node and fixed-time search;
- standard PGN/FEN/UCI conversion as needed;
- Fastchess-compatible operation.

## Raw chess features

Allowed:

- own/opponent piece type and square;
- side to move;
- castling rights;
- en-passant file or square;
- halfmove state only if required for the implemented draw rule.

Forbidden:

- material values;
- mobility;
- attack maps as model inputs;
- piece-square tables;
- pawn categories;
- king-safety terms.

## Required correctness tests

- established perft positions;
- randomized differential testing against `cozy-chess`;
- make/unmake and hash restoration;
- check, checkmate, and stalemate;
- castling;
- en passant;
- all promotions;
- repetition;
- fifty-move rule if supported;
- UCI compliance smoke test;
- illegal command handling.

## Required search tests

- exact low-piece tactical positions;
- mate-in-\(n\);
- model replaced with a deterministic test evaluator;
- transposition-table consistency;
- node-limit consistency.

## Stockfish diagnostic ceiling

Run a **separate diagnostic experiment**:

1. generate a position corpus from legal chess trajectories;
2. label positions using a fixed, documented Stockfish search budget;
3. train the model to predict Stockfish WDL and best move;
4. measure the supervised ceiling.

This is not the main self-play result.

Do not use this teacher-trained checkpoint as the tabula-rasa system’s champion unless the experiment is explicitly labelled teacher-assisted.

The purpose is to distinguish:

- an incapable representation;
- an incapable optimizer;
- insufficient self-play;
- insufficient search.

## Acceptance criteria

- chess rules pass differential tests;
- UCI process behaves correctly;
- Fastchess can complete a tournament without crashes or illegal moves;
- supervised diagnostic performance is materially better than chance;
- model plus search is stronger than raw model;
- no claim of chess Elo is made yet unless the match protocol is already statistically meaningful.

---

# 27. Phase 7 — Measurable standard-chess strength

## Objective

Produce a statistically meaningful strength estimate and establish chess scaling.

## Training tracks

Maintain two clearly separated tracks:

### Track A: tabula-rasa

- self-play only;
- no human games;
- no Stockfish labels;
- no opening book;
- no chess tablebases.

### Track B: diagnostic teacher-assisted

- Stockfish-labelled positions;
- used to estimate model and optimizer ceiling;
- not confused with tabula-rasa results.

Do not silently mix data between tracks.

## Chess baselines

Evaluate against:

1. random legal play;
2. raw policy play;
3. model plus shallow search;
4. previous champions;
5. reduced-strength Stockfish;
6. full Stockfish under tightly limited conditions.

## Elo terminology

Do not report a result as human or FIDE Elo.

Stockfish’s own documentation emphasizes that Elo is relative to match conditions, opponent pool, opening set, and time control. It also explains that `UCI_Elo` and `Skill Level` weaken Stockfish by intentionally selecting suboptimal moves. Treat those settings as coarse anchors, not an absolute human scale. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html))

Use terms such as:

- “engine-pool Elo”;
- “Fastchess logistic Elo difference”;
- “score against Stockfish UCI_Elo anchor”;
- “relative Elo under this exact protocol.”

## Required match protocol

For every reported match:

- use one thread per engine unless explicitly testing thread scaling;
- disable pondering;
- disable tablebases;
- keep hash fixed and documented;
- use a fixed time control;
- use the same opening/start-position suite;
- repeat each start with colours reversed;
- use a fixed random seed;
- save all games;
- record engine binaries and checkpoints;
- report crashes and illegal moves.

An external opening suite is allowed for **evaluation variance reduction**. It is not part of the learned engine and must be applied symmetrically.

## Match sizes

Use approximately:

- 50–100 pairs for smoke testing;
- 200–500 pairs for rough estimates;
- more games until the target confidence interval is reached for reported results.

Do not select a game count merely because it is round.

For a meaningful late-stage estimate, target an Elo confidence interval no wider than roughly 50–75 Elo. A tighter interval is preferable.

## Fastchess

Use Fastchess for:

- paired starts;
- concurrent games;
- PGN output;
- rating reporting;
- pentanomial statistics;
- SPRT for candidate-versus-champion testing.

Stockfish’s Fishtest infrastructure also demonstrates the importance of large paired game samples and statistically controlled progression tests. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/stockfish-wiki/Regression-Tests.html))

## Fixed-node chess evaluation

Use fixed-node evaluation for:

- candidate-versus-candidate comparisons;
- search-scaling curves;
- evaluator quality per node;
- deterministic regression tests.

Do not treat equal node count between different engines as equal compute. Engine node definitions and per-node work differ.

## Fixed-time chess evaluation

Use fixed-time evaluation for:

- practical strength;
- comparison with Stockfish;
- model-size tradeoffs;
- end-to-end CPU efficiency.

## Required chess scaling experiments

For the tabula-rasa track:

1. evaluate one checkpoint at four search budgets;
2. train four data-budget points at fixed model width;
3. train four model widths at fixed data budget;
4. compare fixed-node and fixed-time performance;
5. use at least two seeds for the principal pilot result.

## Acceptance criteria

- the engine obtains a measurable nontrivial score against at least one Stockfish anchor;
- a rating or score interval can be reported honestly;
- additional evaluation search improves strength;
- additional training data improves strength over a meaningful range;
- model scaling is understood at both fixed nodes and fixed time;
- the candidate beats previous champions consistently;
- no opening or endgame knowledge is embedded in the engine;
- the phase report clearly separates tabula-rasa and teacher-assisted performance.

No particular absolute Elo is mandatory. The scientific requirement is a reproducible, statistically meaningful rating and a positive scaling trend.

---

# 28. Phase 8 — Forward Chess rules and reduced exact games

## Objective

Implement Forward Chess correctly before training it at full size.

## Human-readable rules file

Create `FORWARD_CHESS_RULES.md`.

It must explicitly define:

- board coordinates and rank direction;
- legal movement direction for each orientation;
- attack direction;
- horizontal movement;
- castling;
- en passant;
- check;
- checkmate;
- stalemate;
- repetition;
- move-count draw rules;
- promotion choices;
- orientation reversal after promotion.

This file must be reviewed before any expensive Forward Chess training.

## Core directional rule

For ordinary-orientation pieces:

- a white piece may not move or attack to a lower rank;
- a black piece may not move or attack to a higher rank;
- zero rank displacement remains possible where the ordinary piece geometry allows horizontal movement.

After promotion:

- the promoted piece receives reversed orientation;
- its permitted movement and attack direction reverse accordingly;
- orientation is therefore separate from colour.

The attack restriction is real, not merely a movement restriction. For example, the rule corpus must include the stated property that a black queen on `d2` does not attack backward squares such as `d3` or `e3`.

## Representation

Forward Chess feature IDs must include:

- owner/colour;
- piece type;
- square;
- orientation;
- side to move;
- castling rights;
- en-passant state.

Do not encode positional conclusions.

## Implementation sequence

1. slow reference board;
2. reference legal move generator;
3. hand-authored rules corpus;
4. optimized board;
5. optimized move generator;
6. randomized differential test;
7. reduced exact solver;
8. only then full-board training.

## Reduced instances

Create reduced Forward Chess rulesets that preserve:

- directional attacks;
- kings;
- promotions;
- reversed promoted orientation;
- horizontal movement;
- at least one sliding piece;
- eventual repetitions where practical.

Use several rungs:

1. tiny exact board;
2. larger exact or sampled board;
3. medium strategic board;
4. full 8×8 game.

Do not create a separate search or learning implementation for the reduced game.

## Required exact tests

- no generated move violates orientation;
- attack maps obey orientation;
- horizontal queen/rook/king moves remain legal when otherwise legal;
- kings do not move into oriented attacks;
- castling is horizontal and correctly checked;
- en passant is correctly timed;
- promotions flip orientation;
- promoted attack direction flips;
- slow and optimized generators agree;
- exact search returns stable WDL results.

## Acceptance criteria

- the full rules corpus passes;
- randomized differential testing finds no discrepancy over a large sample;
- reduced exact instances can be solved;
- model and search work unchanged;
- Forward Chess raw feature encoding contains no strategy features;
- full-board self-play has not yet begun.

---

# 29. Phase 9 — Forward Chess learning and strength

## Objective

Train Forward Chess from scratch, establish internal strength, and verify scaling.

## Initial training

Begin from random initialization.

Do not transfer the normal-chess model into the primary result.

After a strong scratch baseline exists, one isolated transfer experiment may compare:

- random initialization;
- initialization from ordinary chess-compatible embeddings.

That experiment must remain an ablation and must not become a required mode.

## Evaluation ladder

Use:

1. exact reduced-board oracle;
2. raw model;
3. model plus search;
4. deeper search using the same model;
5. current champion;
6. historical champions;
7. simple unlearned search baseline;
8. optional human matches much later.

## Internal rating pool

Maintain a frozen pool containing:

- initial random baseline;
- first competent checkpoint;
- every promoted champion;
- selected compute-scaled versions of champions.

Run paired round-robin or gauntlet matches.

Report relative engine-pool Elo with uncertainty.

Do not report Forward Chess Elo as though it were comparable with ordinary chess.

## Search-scaling sanity test

For every major champion, evaluate at:

```text
1×, 4×, 16×, 64×
```

search nodes.

A healthy evaluator should ordinarily gain strength with deeper search. If it does not:

- investigate search correctness;
- examine horizon effects;
- examine evaluator calibration;
- inspect transposition-table behaviour;
- inspect repetition handling;
- inspect policy move ordering.

Do not immediately add new search heuristics.

## Training scaling

Run:

- model-width sweep;
- self-play-data sweep;
- search-teacher-budget sweep;
- fixed-time evaluation.

Use the same reporting methodology as Breakthrough, Othello, and chess.

## Distribution-shift evaluation

Create a frozen set of Forward Chess positions from:

- early games;
- middle games;
- near-promotion positions;
- positions containing reversed promoted pieces;
- low-material positions;
- repeated-position candidates;
- unusual but legal random trajectories.

Evaluate every champion on the same set.

## Adversarial disagreement corpus

For sampled positions:

1. compare the candidate’s preferred move with a much deeper search;
2. retain positions with large disagreement;
3. categorize failures;
4. do not manually encode the observed categories;
5. use them as a frozen evaluation set.

## Acceptance criteria

- full-board self-play remains legal and stable;
- candidate promotion produces a clear historical progression;
- more search improves strength;
- more self-play improves strength over a measured range;
- model scaling is understood;
- reduced exact exploitability improves;
- full-core data generation scales well;
- no manual structural Forward Chess evaluation terms have been introduced.

---

# 30. Phase 10 — Many-vCPU production run

## Objective

Run a large experiment only after the system has already demonstrated predictable behaviour.

The expensive VM must extend an existing curve. It must not be the first environment in which the system is tested.

## Freeze before launching

Before the large run:

- freeze the Git commit;
- freeze `Cargo.lock`;
- freeze resolved configuration;
- freeze evaluation suites;
- freeze initial checkpoint;
- freeze seeds;
- freeze promotion criteria;
- build the release binary;
- run all tests;
- complete a small canary run on the target VM.

Do not edit code during the production run.

## VM preflight

Record:

- logical CPU count;
- physical core count if available;
- CPU model;
- SIMD capabilities;
- NUMA topology;
- RAM;
- disk space;
- sustained single-core search speed;
- 1/2/4/8/... worker scaling.

Build with native CPU targeting for the specific VM:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The `cozy-chess` documentation specifically notes that native CPU feature selection can materially improve engine move-generation performance. ([docs.rs](https://docs.rs/cozy-chess/latest/cozy_chess/))

## Canary run

Run approximately 1% of the intended budget using:

- the exact production binary;
- the exact production config;
- the exact checkpoint;
- the exact output location structure.

Verify:

- checkpoints;
- restart;
- full-core utilization;
- memory stability;
- disk-growth projection;
- deterministic seed assignment;
- generation completion;
- evaluation completion;
- report generation.

## Production resource policy

During self-play:

- use nearly all logical CPUs as independent workers;
- use one thread per search;
- divide total transposition-table memory across workers;
- keep aggregate memory below a conservative fraction of RAM;
- avoid swap.

During training:

- stop workers;
- use the full CPU allocation for batch training;
- verify no nested thread-pool oversubscription.

During evaluation:

- use concurrent independent matches;
- preserve paired starts and fixed seeds.

## Checkpointing

Checkpoint:

- after every completed training generation;
- after candidate evaluation;
- before promotion;
- after promotion.

A restart must resume from the last complete generation.

## Production stop conditions

Stop the run if:

- illegal moves occur;
- NaNs appear;
- candidate strength collapses across two consecutive generations;
- CPU utilization remains unexpectedly low after a short diagnostic period;
- memory grows without bound;
- checkpoint writes fail;
- data corruption is detected;
- evaluation no longer reproduces known baselines.

Do not allow a large run to continue merely because compute has already been purchased.

---

# 31. Chess Elo and Stockfish evaluation instructions

## 31.1 UCI implementation minimum

Support:

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

Initially support one engine thread.

Accept common harmless UCI options gracefully, but do not claim parallel search support before it exists.

## 31.2 Stockfish configuration

For reduced-strength anchors:

```text
UCI_LimitStrength = true
UCI_Elo = selected anchor
Threads = 1
Ponder = false
MultiPV = 1
SyzygyPath = empty
```

For full-strength constrained anchors:

```text
UCI_LimitStrength = false
Threads = 1
Ponder = false
MultiPV = 1
SyzygyPath = empty
```

Keep hash and time control fixed.

## 31.3 Opening protocol

Use a frozen set of starting positions.

For every position:

1. candidate plays White;
2. candidate plays Black.

Randomize pair order but preserve pairing.

Do not tune the model against the evaluation opening set.

## 31.4 Reported statistics

Report:

- wins;
- draws;
- losses;
- paired pentanomial counts where available;
- score;
- Elo difference;
- confidence interval;
- game count;
- pair count;
- time control;
- opening set;
- engine thread count;
- average nodes;
- average NPS;
- crashes;
- illegal moves.

## 31.5 Candidate-versus-champion testing

Use SPRT for routine chess development only after the engine is stable.

A reasonable development question is:

```text
H0: candidate is no better than champion
H1: candidate is at least a modest Elo improvement
```

Do not use SPRT as a substitute for reporting the eventual fixed-sample rating.

## 31.6 Interpretation restrictions

Never claim:

- “the engine is FIDE 1800”;
- “the engine is grandmaster strength”;
- “the engine gained exactly 40 universal Elo.”

Use:

> Under the documented Fastchess protocol, the engine scored X against the selected Stockfish anchor, corresponding to an estimated relative Elo of Y ± Z.

---

# 32. Exact-game evaluation instructions

For every exact-game checkpoint, evaluate on a fixed oracle corpus.

## 32.1 Decision regret

For state \(s\), let \(V^\*(s)\) be its game-theoretic value, and let \(a\) be the selected action.

Measure:

\[
r(s)=V^\*(s)-V^\*(T(s,a))
\]

with appropriate player-perspective negation.

For WDL categories, also report levels lost:

- winning move retained: 0;
- win reduced to draw: 1;
- win reduced to loss: 2;
- draw reduced to loss: 1.

## 32.2 Optimal-action mass

If \(A^\*(s)\) is the set of optimal actions:

\[
m(s)=\sum_{a\in A^\*(s)}\pi(a\mid s)
\]

This avoids penalizing the model for distributing probability among several equally optimal moves.

## 32.3 Exploitability

Play the candidate against perfect opposition from:

- the normal initial state;
- all principal opening states;
- a stratified state sample.

Report expected result and frequency of avoidable losses.

## 32.4 Stratification

Break results down by:

- WDL class;
- ply;
- remaining game length;
- branching factor;
- number of optimal actions;
- frequency under candidate self-play;
- whether the position appeared in training.

---

# 33. Intermediate-game evaluation instructions

For Breakthrough and Othello:

## 33.1 Frozen baseline pool

Include:

- random agent;
- untrained model;
- shallow-search untrained model;
- first promoted champion;
- selected later champions;
- same checkpoint at several search budgets.

## 33.2 Paired starts

Where the game has first-player advantage:

- create a frozen set of opening perturbations or intermediate starts;
- play both colours;
- aggregate by pair.

## 33.3 Promotion

A candidate may be promoted when:

- its paired score confidence interval is positive;
- it does not regress materially against older champions;
- its exact-subgame metrics do not regress;
- it introduces no correctness or performance failure.

## 33.4 Generality criterion

A phase is not considered successful merely because the program wins.

It must demonstrate that:

- the existing generic learner works;
- no strategic feature code was added;
- scaling trends remain interpretable;
- search improvements come from learned policy/value quality rather than benchmark-specific rules.

---

# 34. Sweep methodology and core allocation

## 34.1 Explicit manifests

A sweep manifest is an explicit list:

```json
{"config":"configs/run_001.toml","cores":1}
{"config":"configs/run_002.toml","cores":1}
{"config":"configs/run_003.toml","cores":4}
```

Do not implement nested sweep syntax.

## 34.2 Scheduling

Use a simple CPU-slot scheduler:

1. detect available logical CPUs;
2. maintain count of free slots;
3. launch the next run whose core request fits;
4. reclaim slots on completion;
5. collect status;
6. terminate cleanly on interrupt.

## 34.3 Large sweeps

For many small runs:

- prefer one core per run;
- run many independent seeds/configurations concurrently.

For a small number of large runs:

- allocate several cores per run only if the training backend or workload has demonstrated parallel efficiency.

## 34.4 Successive allocation

Do not spend the largest budget on every configuration.

Use:

1. smoke budget;
2. medium budget for surviving configurations;
3. large budget only for configurations that preserve their advantage.

This is experimental triage, not a permanent hyperparameter-optimization controller.

The promotion decision should remain explicit in the report.

---

# 35. Global research gates before expensive compute

The large VM is forbidden until all of the following are true:

- [ ] Exact solver agrees with exhaustive minimax.
- [ ] Search with exact leaves always chooses optimal moves.
- [ ] The model can fit held-out exact strategy.
- [ ] Self-play recovers exact strategy from random initialization.
- [ ] More self-play data improves at least one non-saturated exact benchmark.
- [ ] More search improves the same checkpoint.
- [ ] Larger models show an interpretable capacity curve.
- [ ] Breakthrough training improves across generations.
- [ ] Othello training improves across generations.
- [ ] Standard chess passes differential and UCI tests.
- [ ] Standard chess obtains a statistically measurable result.
- [ ] Forward Chess passes slow-versus-fast differential testing.
- [ ] Reduced Forward Chess can be solved exactly.
- [ ] Full-board Forward Chess self-play produces only legal games.
- [ ] CPU utilization and parallel efficiency have been measured.
- [ ] A target-VM canary completes successfully.
- [ ] The large run extends an existing scaling curve.

---

# 36. Failure diagnosis matrix

## Search with oracle leaves fails

Likely causes:

- perspective sign error;
- terminal score error;
- make/unmake corruption;
- transposition-table bound error;
- repetition error;
- node-budget return error.

Do not change the model.

## Supervised exact model fails

Likely causes:

- insufficient representation;
- insufficient capacity;
- optimizer bug;
- target encoding bug;
- data leakage or split error;
- policy action-ID mismatch.

Do not launch self-play.

## Supervised model succeeds but self-play fails

Likely causes:

- inadequate exploration;
- insufficient search teacher;
- self-play state-distribution collapse;
- outcome target variance;
- candidate promotion bug;
- catastrophic forgetting.

Run controlled diagnostics. Do not add a new architecture.

## Raw model improves but search does not

Likely causes:

- search bug;
- poor policy move ordering;
- bad value calibration;
- horizon effects;
- TT collisions or incorrect bounds;
- repetition handling.

## Search improves with nodes but training does not improve

Likely causes:

- poor targets;
- insufficient data diversity;
- optimizer instability;
- too little model capacity;
- self-play games dominated by draws;
- training/evaluation mismatch.

## Larger models improve fixed-node but regress fixed-time

Interpretation:

- capacity helps;
- inference cost is too high;
- find the compute-optimal width;
- consider packed or incremental inference only after profiling.

## CPU utilization is low

Inspect:

- too few independent games;
- thread-pool oversubscription;
- locks around model access;
- serialized dataset writes;
- tiny work units;
- memory bandwidth;
- transposition-table contention;
- ML backend thread settings.

Do not immediately build distributed infrastructure.

---

# 37. Agent implementation workflow

For every phase:

## Step 1: Restate the phase hypothesis

Write one paragraph answering:

> What specific uncertainty will this phase resolve?

## Step 2: Inspect current state

Report:

- current modules;
- current public types;
- current config keys;
- current dependencies;
- current passing benchmarks;
- current failing benchmark, if any.

## Step 3: Propose the smallest vertical slice

The proposal must include:

- files to change;
- public API changes;
- config changes;
- tests;
- benchmark;
- expected evidence.

Do not present multiple large architecture alternatives unless the phase is explicitly a comparison experiment.

## Step 4: Implement

Keep the change narrowly scoped.

Prefer:

- functions over stateless structs;
- enums over boolean combinations;
- plain data over manager objects;
- direct calls over registries;
- explicit loops over generic frameworks;
- static dispatch in performance-sensitive code.

## Step 5: Run local correctness tests

Required:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
```

## Step 6: Run behavioural benchmark

Every phase must conclude with a global or strategic benchmark, not merely unit tests.

## Step 7: Run performance benchmark

Record throughput, CPU utilization, and memory.

## Step 8: Produce phase report

Use the template below.

## Step 9: Stop

Do not begin the next phase in the same work session unless explicitly instructed.

---

# 38. Required phase-report template

```markdown
# Phase N Report: <title>

## Status

PASS / FAIL / PARTIAL

## Research hypothesis

<What was being tested?>

## Minimal implementation

<What was added?>

## Deliberately excluded

<What was not added, and why?>

## Correctness evidence

- Unit tests:
- Property tests:
- Differential tests:
- Exact-search tests:

## Global learning or playing result

<Main behavioural evidence>

## Scaling result

| Axis | Values | Metric trend | Confidence | Interpretation |
|---|---:|---:|---:|---|
| Model size | | | | |
| Training data | | | | |
| Search | | | | |

## CPU and memory result

| CPUs | Throughput | Parallel efficiency | Utilization | Peak memory |
|---:|---:|---:|---:|---:|

## Reproducibility

- Git commit:
- Rust toolchain:
- Cargo.lock hash:
- Seeds:
- Resolved configs:
- Run directories:

## Complexity delta

- Production LOC:
- Public types:
- Config keys:
- Dependencies:
- New modes:
- Files:

## Failures and anomalies

<Include negative results.>

## Decision

- Promote phase: yes/no
- Selected configuration:
- Rejected alternatives:
- Reason:

## Exact next phase

<One paragraph describing, but not implementing, the next phase.>
```

---

# 39. Rules for AI-agent behaviour

The following instructions are mandatory.

1. **Do not implement future flexibility.**
2. **Do not create an abstraction for a single implementation.**
3. **Do not add a configuration option without running a comparison.**
4. **Do not retain rejected experiments as runtime modes.**
5. **Do not add a new algorithm before diagnosing the current algorithm.**
6. **Do not optimize before profiling.**
7. **Do not claim strength without a confidence interval.**
8. **Do not call an experiment reproducible if its full resolved configuration was not saved.**
9. **Do not use opening books or chess tablebases in the learned engine.**
10. **Do not introduce handcrafted positional features.**
11. **Do not silently use Stockfish-labelled data in the tabula-rasa track.**
12. **Do not run an expensive experiment that is not an extension of a smaller measured curve.**
13. **Do not continue after a failed phase gate.**
14. **Do not hide negative scaling or bad seeds.**
15. **Do not tune against the test set.**
16. **Do not use averages without showing seed variation.**
17. **Do not create a web dashboard.**
18. **Do not add async Rust.**
19. **Do not build a distributed system for a single VM.**
20. **Delete code that no longer serves the selected methodology.**

Names such as these are presumptively forbidden unless their necessity is demonstrated:

```text
Manager
Controller
Factory
Registry
Provider
Plugin
Backend
Orchestrator
Coordinator
Service
StrategyFactory
PipelineBuilder
```

A type should be named after the intrinsic entity it represents.

---

# 40. Definition of project success

The project is successful when all of the following are true.

## Scientific success

- Small games are solved and evaluated exactly.
- The model can fit exact strategies.
- Search-guided self-play recovers exact or near-exact play.
- More data produces measurable gains before saturation.
- More search produces measurable gains.
- Model-size effects are understood.
- The same methodology works in multiple structurally different games.
- Ordinary chess receives a reproducible relative strength estimate.
- Forward Chess receives a stable internal rating and scaling analysis.

## Engineering success

- Core code is Rust.
- Cargo fully manages builds and dependencies.
- The code is typed, deterministic, and testable.
- Game rules are isolated from learning and search.
- Runs are reproducible from saved manifests.
- The project scales across dozens of CPU cores.
- Expensive runs are resumable at generation boundaries.
- Performance regressions are measurable.
- No unnecessary distributed infrastructure exists.

## Simplicity success

- There is one production search algorithm.
- There is one production model family.
- There is one training loop.
- There is one evaluation methodology per game class.
- Configuration contains only real experimental variables.
- Rejected ideas are deleted rather than hidden behind modes.
- Every major component corresponds to an intrinsic part of the problem.
- The repository remains understandable to one technically capable person.

---

# 41. Initial instruction to the agent

Begin with **Phase 0 only**.

Before writing code:

1. restate the Phase 0 hypothesis;
2. propose the minimal repository layout;
3. list the exact dependencies required for Phase 0;
4. list everything deliberately deferred;
5. identify the global benchmark that will prove Phase 0 successful.

Then implement Phase 0, run its tests and benchmarks, produce the required report, and stop.

Do not begin exact solving, neural-network code, self-play, Breakthrough, Othello, chess, or Forward Chess during Phase 0.

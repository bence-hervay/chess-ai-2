//! `lab` — the experiment command-line surface.
//!
//! Commands (each appears in the phase that requires it):
//! - `lab evaluate <config>`: paired evaluation arena (Phase 0);
//! - `lab solve <config>`: exact solving and search-correctness
//!   experiments (Phase 1);
//! - `lab train <config>`: supervised training on exact corpora (Phase 2);
//! - `lab sweep <manifest>`: CPU-slot scheduling of many runs (Phase 2);
//! - `lab selfplay <config>`: synchronous Expert Iteration (Phase 3);
//! - `lab evaluate` kind `oracle_probe`: checkpoint quality with and
//!   without search at several node budgets (Phase 3).

use burn::module::{AutodiffModule, Module as _};
use burn::prelude::Backend as _;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use clap::{Parser, Subcommand};
use selfplay_lab::data::{
    collect_positions, join_retrograde_oracle, label_positions, summarize, OracleJoinStats,
    RelabelSummary, TrajectorySpec,
};
use selfplay_lab::evaluation::{
    evaluate_model_oracle, exploitability_vs_perfect, play_paired_match,
    retrograde_searched_candidates, search_disagreement_analysis, searched_decision_metrics,
    searched_decision_metrics_on, CorpusSplit, OracleMetrics,
};
use selfplay_lab::evaluation::{run_random_arena, ArenaSummary};
use selfplay_lab::experiment::{
    collect_manifest, peak_rss_bytes, process_cpu_seconds, unix_seconds, Manifest, RunDir,
};
use selfplay_lab::features::forward_chess::{FcExtractor, FcRecipe};
use selfplay_lab::features::FeatureExtractor as _;
use selfplay_lab::game::Game;
use selfplay_lab::games::breakthrough::Breakthrough;
use selfplay_lab::games::chess::Chess;
use selfplay_lab::games::connect_k::ConnectK;
use selfplay_lab::games::forward_chess::{read_tablebase_with, write_tablebase, ForwardChess};
use selfplay_lab::games::othello::Othello;
use selfplay_lab::games::GameSpec;
use selfplay_lab::model::{
    CompiledNet, InferBackend, ModelDims, ModelEvaluator, PolicyValueNet, TrainBackend,
};
use selfplay_lab::search::{
    enumerate_solved, exhaustive_negamax, solve_retrograde, Evaluator, ExactSolver, MoveOrdering,
    Searcher, Wdl, ZeroEvaluator,
};
use selfplay_lab::training::{
    build_exact_dataset, build_retrograde_dataset, generate_selfplay, make_batch, splitmix64,
    train_steps, train_supervised, TrainRow, BATCH_SIZE, EVAL_EVERY,
};
use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Construct the concrete game named by a [`GameSpec`] and evaluate the
/// same generic body for it. Both games expose inherent
/// `feature_count()` / `action_count()` methods, so bodies may use them.
macro_rules! dispatch_game {
    ($spec:expr, $game:ident, $body:expr) => {
        match $spec {
            GameSpec::ConnectK {
                width,
                height,
                k,
                gravity,
            } => {
                let $game = ConnectK::new(*width, *height, *k, *gravity)?;
                $body
            }
            GameSpec::Breakthrough {
                width,
                height,
                rows,
            } => {
                let $game = Breakthrough::new(*width, *height, *rows)?;
                $body
            }
            GameSpec::Othello { width, height } => {
                let $game = Othello::new(*width, *height)?;
                $body
            }
            GameSpec::Chess {} => {
                let $game = Chess::new();
                $body
            }
            GameSpec::ForwardChess { ruleset } => {
                let $game = ForwardChess::new(*ruleset);
                $body
            }
        }
    };
}

/// Transposition-table size used by solve experiments (2^20 entries).
const SOLVE_TT_LOG2: u32 = 20;

#[derive(Parser)]
#[command(name = "lab", about = "Minimal CPU-first self-play game AI lab")]
struct Cli {
    #[command(subcommand)]
    command: LabCommand,
}

#[derive(Subcommand)]
enum LabCommand {
    /// Run a paired evaluation arena from a TOML configuration.
    Evaluate {
        /// Path to an evaluation configuration file.
        config: PathBuf,
    },
    /// Solve a game instance exactly or run a search-correctness experiment.
    Solve {
        /// Path to a solve configuration file.
        config: PathBuf,
    },
    /// Train the policy/value network on an exact corpus.
    Train {
        /// Path to a training configuration file.
        config: PathBuf,
    },
    /// Run synchronous Expert Iteration self-play generations.
    Selfplay {
        /// Path to a self-play configuration file.
        config: PathBuf,
    },
    /// Stockfish diagnostic ceiling (teacher-assisted; plan §26).
    Teacher {
        /// Path to a teacher configuration file.
        config: PathBuf,
    },
    /// Deep-search relabelling: sample positions from trajectories and
    /// label them with shallow, deep, and counterfactual child searches,
    /// with target provenance and optional exact-oracle join (SHSD
    /// program §18.2, §19.5, §55).
    Relabel {
        /// Path to a relabel configuration file.
        config: PathBuf,
    },
    /// Fit a structured evaluator (or the raw-MLP baseline) on exact
    /// Forward Chess data and probe its searched decisions against the
    /// exact solution (SHSD program §56).
    Fit {
        /// Path to a fit configuration file.
        config: PathBuf,
    },
    /// Run a manifest of experiment configs with CPU-slot scheduling.
    Sweep {
        /// Path to a sweep manifest (JSONL: {"command","config","cores"}).
        manifest: PathBuf,
    },
    /// Play Forward Chess interactively against a checkpoint (or the
    /// zero-evaluator search when no checkpoint is given). Moves are
    /// typed as coordinates, e.g. `a2a3` or `a7a8=Q`; other commands:
    /// `?` (list legal moves), `hint`, `undo`, `quit`. Standard chess
    /// is played through the separate `uci` binary in any UCI GUI.
    Play {
        /// Ruleset: fc-tiny | fc-small | fc-medium | fc-full.
        #[arg(long, default_value = "fc-full")]
        game: String,
        /// Checkpoint directory (`model.bin` + `model.json`); omit to
        /// play against unlearned search.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Engine node budget per move.
        #[arg(long, default_value_t = 600)]
        nodes: u64,
        /// Your colour: white | black | none (none = engine vs engine).
        #[arg(long, default_value = "white")]
        side: String,
    },
    /// Benchmark a checkpoint's search on one thread: nodes/second,
    /// achieved iterative-deepening depth, and time per move at each
    /// node budget, plus an optional fixed wall-clock movetime probe
    /// (e.g. how many nodes fit in 2 s on one core). Positions are
    /// drawn deterministically from self-play with the same checkpoint;
    /// node-budget results are deterministic, movetime results are not.
    Bench {
        /// Game: fc-tiny | fc-small | fc-medium | fc-full | chess.
        #[arg(long, default_value = "fc-full")]
        game: String,
        /// Checkpoint directory (`model.bin` + `model.json`); omit to
        /// bench unlearned search (zero evaluator: movegen and search
        /// overhead only).
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Node budgets to benchmark.
        #[arg(long, default_value = "400,800,1600,6400", value_delimiter = ',')]
        nodes: Vec<u64>,
        /// Fixed wall-clock budget per move to probe, in milliseconds
        /// (0 = skip).
        #[arg(long, default_value_t = 0)]
        movetime_ms: u64,
        /// Number of benchmark positions.
        #[arg(long, default_value_t = 60)]
        positions: usize,
        /// Seed for the position-generating self-play games.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

/// Fully explicit configuration for `lab evaluate`, dispatched on `kind`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum EvaluateConfig {
    /// Paired random-versus-random arena (Phase 0).
    RandomArena(EvaluationConfig),
    /// Oracle-based probe of a saved checkpoint: raw metrics plus
    /// searched decisions and exploitability at several node budgets
    /// (Phase 3 diagnostic matrix).
    OracleProbe(ProbeConfig),
    /// Search-scaling probe without an oracle: the checkpoint at each
    /// probed node budget plays paired matches against itself at a
    /// fixed baseline budget (Phase 4, non-exact sizes).
    MatchProbe(MatchProbeConfig),
}

/// Match-probe parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchProbeConfig {
    /// Checkpoint directory containing `model.bin` and `model.json`.
    checkpoint: PathBuf,
    /// Opponent checkpoint; pass the same path to probe search scaling
    /// of a single model, or a different champion for cross-run
    /// comparisons (e.g. the data-scaling axis).
    opponent_checkpoint: PathBuf,
    /// Node budgets to probe against the baseline.
    node_budgets: Vec<u64>,
    baseline_nodes: u64,
    pairs: u64,
    opening_plies: u32,
    seed: u64,
    threads: usize,
    game: GameSpec,
}

/// Paired-arena parameters. Every field is required.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationConfig {
    seed: u64,
    /// Number of game pairs (two games per pair, colours swapped).
    pairs: u64,
    /// Worker threads for independent games.
    threads: usize,
    game: GameSpec,
}

/// Oracle-probe parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeConfig {
    /// Checkpoint directory containing `model.bin` and `model.json`.
    checkpoint: PathBuf,
    /// Node budgets to probe (1 behaves as raw-policy play).
    node_budgets: Vec<u64>,
    /// Cap on test states for searched-decision metrics.
    searched_sample: usize,
    threads: usize,
    game: GameSpec,
}

/// Solving/search method under measurement.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SolveMethod {
    /// Memoized exhaustive negamax; solves every reachable state and
    /// writes the exact corpus.
    Exact,
    /// Plain negamax without memoization or pruning (tiny boards only).
    Exhaustive,
    /// Production iterative-deepening alpha-beta without a transposition
    /// table.
    AlphaBeta,
    /// Production iterative-deepening alpha-beta with a transposition
    /// table.
    AlphaBetaTt,
    /// Reachable-graph retrograde analysis for repetition-capable games
    /// (repetition-as-draw convention; fifty-move rule not modelled).
    Retrograde,
}

impl SolveMethod {
    fn label(self) -> &'static str {
        match self {
            SolveMethod::Exact => "exact",
            SolveMethod::Exhaustive => "exhaustive",
            SolveMethod::AlphaBeta => "ab",
            SolveMethod::AlphaBetaTt => "abtt",
            SolveMethod::Retrograde => "retro",
        }
    }
}

/// Fully explicit configuration for `lab solve`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SolveConfig {
    method: SolveMethod,
    ordering: MoveOrdering,
    game: GameSpec,
}

/// One row of the exact evaluation corpus (side-to-move perspective).
#[derive(Serialize)]
struct CorpusRow {
    key: u64,
    ply: u32,
    wdl: Wdl,
    features: Vec<u32>,
    legal: Vec<u32>,
    child_wdl: Vec<Wdl>,
    optimal: Vec<u32>,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        LabCommand::Evaluate { config } => evaluate(&config),
        LabCommand::Solve { config } => solve(&config),
        LabCommand::Train { config } => train(&config),
        LabCommand::Selfplay { config } => selfplay(&config),
        LabCommand::Teacher { config } => teacher(&config),
        LabCommand::Relabel { config } => relabel(&config),
        LabCommand::Fit { config } => fit(&config),
        LabCommand::Sweep { manifest } => sweep(&manifest),
        LabCommand::Play {
            game,
            checkpoint,
            nodes,
            side,
        } => play(&game, checkpoint.as_deref(), nodes, &side),
        LabCommand::Bench {
            game,
            checkpoint,
            nodes,
            movetime_ms,
            positions,
            seed,
        } => bench(
            &game,
            checkpoint.as_deref(),
            &nodes,
            movetime_ms,
            positions,
            seed,
        ),
    };
    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

fn read_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

/// Shared run lifecycle: directory, resolved config, initial manifest.
fn start_run<C: Serialize>(
    label: &str,
    config: &C,
    seed: u64,
    threads: usize,
) -> Result<(RunDir, Manifest), String> {
    let manifest = collect_manifest(seed, threads);
    let run_dir = RunDir::create(Path::new("runs"), label, &manifest.git_commit)
        .map_err(|e| format!("creating run directory: {e}"))?;
    run_dir
        .write_text(
            "resolved.toml",
            &toml::to_string_pretty(config).expect("serializable"),
        )
        .map_err(|e| e.to_string())?;
    run_dir
        .write_json("manifest.json", &manifest)
        .map_err(|e| e.to_string())?;
    Ok((run_dir, manifest))
}

fn finish_run(run_dir: &RunDir, mut manifest: Manifest, log: &str) -> Result<(), String> {
    run_dir
        .write_text("stdout.log", log)
        .map_err(|e| e.to_string())?;
    manifest.end_unix_seconds = Some(unix_seconds());
    manifest.exit_status = "success".to_string();
    run_dir
        .write_json("manifest.json", &manifest)
        .map_err(|e| e.to_string())
}

fn say(line: String, log: &mut String) {
    println!("{line}");
    log.push_str(&line);
    log.push('\n');
}

// ---------------------------------------------------------------------------
// lab evaluate
// ---------------------------------------------------------------------------

/// Per-run performance metrics derived from the counters and clocks.
#[derive(Clone, Debug, Serialize)]
struct RunMetrics {
    wall_seconds: f64,
    cpu_seconds: f64,
    /// cpu_seconds / (wall_seconds * allocated threads).
    utilization: f64,
    games_per_second: f64,
    moves_per_second: f64,
    peak_rss_bytes: u64,
}

fn evaluate(config_path: &Path) -> Result<(), String> {
    match read_config::<EvaluateConfig>(config_path)? {
        EvaluateConfig::RandomArena(config) => run_arena_command(config),
        EvaluateConfig::OracleProbe(config) => probe(config),
        EvaluateConfig::MatchProbe(config) => match_probe(config),
    }
}

fn match_probe(config: MatchProbeConfig) -> Result<(), String> {
    if config.node_budgets.is_empty() || config.pairs == 0 {
        return Err("node_budgets and pairs must be non-empty/positive".into());
    }
    let label = format!("matchprobe-{}", config.game.label());
    let (run_dir, manifest) = start_run(&label, &config, config.seed, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    let (net, dims, _max_features) = load_checkpoint(&config.checkpoint)?;
    let (opponent_net, opponent_dims, _) = load_checkpoint(&config.opponent_checkpoint)?;
    let mut manifest = manifest;
    manifest.model_parameter_count = net.num_params() as u64;
    let compiled = CompiledNet::from_net(&net, dims);
    let opponent = CompiledNet::from_net(&opponent_net, opponent_dims);
    let started = Instant::now();
    let mut reports = Vec::new();
    dispatch_game!(&config.game, game, {
        for &budget in &config.node_budgets {
            let result = play_paired_match(
                &game,
                &compiled,
                &opponent,
                config.pairs,
                config.opening_plies,
                budget,
                config.baseline_nodes,
                config.seed ^ budget,
                config.threads,
            );
            say(
                format!(
                    "budget {budget} vs baseline {}: score {:.3} (lcb {:.3}), \
                     +{} ={} -{} over {} games",
                    config.baseline_nodes,
                    result.score,
                    result.score_lcb95,
                    result.candidate_wins,
                    result.draws,
                    result.candidate_losses,
                    result.games,
                ),
                &mut log,
            );
            reports.push(serde_json::json!({
                "node_budget": budget,
                "baseline_nodes": config.baseline_nodes,
                "match": result,
            }));
        }
    });
    run_dir
        .write_json(
            "summary.json",
            &serde_json::json!({
                "config": config,
                "parameter_count": manifest.model_parameter_count,
                "budgets": reports,
                "wall_seconds": started.elapsed().as_secs_f64(),
            }),
        )
        .map_err(|e| e.to_string())?;
    finish_run(&run_dir, manifest, &log)
}

fn run_arena_command(config: EvaluationConfig) -> Result<(), String> {
    if config.pairs == 0 {
        return Err("pairs must be positive".into());
    }
    if config.threads == 0 {
        return Err("threads must be positive".into());
    }

    let (run_dir, manifest) =
        start_run(&config.game.label(), &config, config.seed, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    say(
        format!(
            "thread plan: {} arena workers on {} logical CPUs; \
             one single-threaded game per worker",
            config.threads, manifest.logical_cpus
        ),
        &mut log,
    );

    let (arena, metrics) = dispatch_game!(&config.game, game, run_arena(&game, &config, &run_dir)?);

    say(
        format!(
            "{} games: P1 {} / draws {} / P2 {} (agent A points {:.1}), mean plies {:.1}",
            arena.games,
            arena.p1_wins,
            arena.draws,
            arena.p2_wins,
            arena.agent_a_points,
            arena.total_plies as f64 / arena.games.max(1) as f64,
        ),
        &mut log,
    );
    say(
        format!(
            "throughput: {:.0} games/s, {:.0} moves/s; wall {:.2}s, cpu {:.2}s, \
             utilization {:.1}%, peak RSS {:.1} MiB",
            metrics.games_per_second,
            metrics.moves_per_second,
            metrics.wall_seconds,
            metrics.cpu_seconds,
            100.0 * metrics.utilization,
            metrics.peak_rss_bytes as f64 / (1024.0 * 1024.0),
        ),
        &mut log,
    );
    finish_run(&run_dir, manifest, &log)
}

fn run_arena<G: Game>(
    game: &G,
    config: &EvaluationConfig,
    run_dir: &RunDir,
) -> Result<(ArenaSummary, RunMetrics), String> {
    run_dir.create_subdir("games").map_err(|e| e.to_string())?;
    let games_file = std::fs::File::create(run_dir.path().join("games/games.jsonl"))
        .map_err(|e| format!("creating games file: {e}"))?;
    let mut writer = std::io::BufWriter::new(games_file);
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();
    let mut sink_error: Option<std::io::Error> = None;
    let arena = run_random_arena(game, config.pairs, config.seed, config.threads, |lines| {
        if sink_error.is_none() {
            if let Err(e) = writer.write_all(lines.as_bytes()) {
                sink_error = Some(e);
            }
        }
    });
    if let Some(e) = sink_error {
        return Err(format!("writing game records: {e}"));
    }
    writer
        .flush()
        .map_err(|e| format!("flushing game records: {e}"))?;
    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    let metrics = RunMetrics {
        wall_seconds,
        cpu_seconds,
        utilization: cpu_seconds / (wall_seconds * config.threads as f64),
        games_per_second: arena.games as f64 / wall_seconds,
        moves_per_second: arena.counters.moves_played as f64 / wall_seconds,
        peak_rss_bytes: peak_rss_bytes().unwrap_or(0),
    };
    run_dir
        .append_jsonl("metrics.jsonl", &[&metrics])
        .map_err(|e| e.to_string())?;
    run_dir
        .write_json(
            "summary.json",
            &serde_json::json!({ "arena": arena, "metrics": metrics }),
        )
        .map_err(|e| e.to_string())?;
    Ok((arena, metrics))
}

// ---------------------------------------------------------------------------
// lab solve
// ---------------------------------------------------------------------------

fn solve(config_path: &Path) -> Result<(), String> {
    let config: SolveConfig = read_config(config_path)?;
    if matches!(config.game, GameSpec::Chess {}) {
        return Err("chess cannot be exactly solved or enumerated".into());
    }
    if matches!(config.game, GameSpec::ForwardChess { .. })
        && config.method != SolveMethod::Retrograde
    {
        return Err(
            "forward chess positions can repeat; only the retrograde method is sound".into(),
        );
    }
    let label = format!("solve-{}-{}", config.method.label(), config.game.label());
    let (run_dir, manifest) = start_run(&label, &config, 0, 1)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    say("thread plan: 1 solver thread".to_string(), &mut log);

    if config.method == SolveMethod::Retrograde {
        // Retrograde is forward-chess-specific: it writes the packed
        // tablebase backup, whose format is tied to that module.
        let GameSpec::ForwardChess { ruleset } = &config.game else {
            return Err("the retrograde method is only for forward_chess".into());
        };
        run_solve_retrograde(&ForwardChess::new(*ruleset), &config, &run_dir, &mut log)?;
        return finish_run(&run_dir, manifest, &log);
    }
    dispatch_game!(&config.game, game, {
        let max_depth = game.cell_count();
        run_solve(&game, max_depth, &config, &run_dir, &mut log)?;
    });
    finish_run(&run_dir, manifest, &log)
}

fn run_solve<G: Game>(
    game: &G,
    max_depth: u32,
    config: &SolveConfig,
    run_dir: &RunDir,
    log: &mut String,
) -> Result<(), String> {
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();
    let mut state = game.initial_state();

    let summary = match config.method {
        SolveMethod::Exact => {
            let mut solver = ExactSolver::new();
            let mut root_optimal = Vec::new();
            let root = solver.optimal_moves(game, &mut state, &mut root_optimal);
            let root_actions: Vec<u32> = root_optimal
                .iter()
                .map(|&m| game.action_id(&state, m))
                .collect();
            let solve_seconds = started.elapsed().as_secs_f64();

            // Corpus: every reachable non-terminal state with exact labels.
            let corpus_file = std::fs::File::create(run_dir.path().join("corpus.jsonl"))
                .map_err(|e| format!("creating corpus file: {e}"))?;
            let mut writer = std::io::BufWriter::new(corpus_file);
            let mut corpus_states = 0u64;
            let mut wdl_counts = [0u64; 3];
            let mut write_error: Option<std::io::Error> = None;
            enumerate_solved(game, &mut solver, |position| {
                corpus_states += 1;
                wdl_counts[position.value as usize] += 1;
                let mut features = Vec::new();
                game.encode_features(position.state, &mut features);
                let row = CorpusRow {
                    key: game.position_key(position.state),
                    ply: position.ply,
                    wdl: position.value,
                    features,
                    legal: position
                        .legal
                        .iter()
                        .map(|&m| game.action_id(position.state, m))
                        .collect(),
                    child_wdl: position.child_values.to_vec(),
                    optimal: position
                        .optimal_moves()
                        .map(|m| game.action_id(position.state, m))
                        .collect(),
                };
                if write_error.is_none() {
                    let line = serde_json::to_string(&row).expect("serializable row");
                    if let Err(e) = writer
                        .write_all(line.as_bytes())
                        .and_then(|()| writer.write_all(b"\n"))
                    {
                        write_error = Some(e);
                    }
                }
            });
            if let Some(e) = write_error {
                return Err(format!("writing corpus: {e}"));
            }
            writer
                .flush()
                .map_err(|e| format!("flushing corpus: {e}"))?;

            say(
                format!(
                    "exact: root {root:?}, optimal root actions {root_actions:?}; \
                     {corpus_states} reachable non-terminal states \
                     (W {} / D {} / L {}), {} solver nodes, {} memo hits",
                    wdl_counts[Wdl::Win as usize],
                    wdl_counts[Wdl::Draw as usize],
                    wdl_counts[Wdl::Loss as usize],
                    solver.nodes,
                    solver.memo_hits,
                ),
                log,
            );
            serde_json::json!({
                "method": config.method,
                "ordering": config.ordering,
                "root_wdl": root,
                "optimal_root_actions": root_actions,
                "solve_seconds": solve_seconds,
                "corpus_states": corpus_states,
                "wdl_counts": {
                    "win": wdl_counts[Wdl::Win as usize],
                    "draw": wdl_counts[Wdl::Draw as usize],
                    "loss": wdl_counts[Wdl::Loss as usize],
                },
                "solved_states": solver.solved_states(),
                "nodes": solver.nodes,
                "memo_hits": solver.memo_hits,
            })
        }
        SolveMethod::Retrograde => unreachable!("dispatched to run_solve_retrograde in solve()"),
        SolveMethod::Exhaustive => {
            let mut nodes = 0u64;
            let value = exhaustive_negamax(game, &mut state, 0, &mut nodes);
            let root = Wdl::from_solved_score(value);
            say(
                format!("exhaustive: root {root:?} (score {value}), {nodes} nodes"),
                log,
            );
            serde_json::json!({
                "method": config.method,
                "ordering": config.ordering,
                "root_wdl": root,
                "score": value,
                "nodes": nodes,
            })
        }
        SolveMethod::AlphaBeta | SolveMethod::AlphaBetaTt => {
            let tt = (config.method == SolveMethod::AlphaBetaTt).then_some(SOLVE_TT_LOG2);
            let mut searcher: Searcher<G> = Searcher::new(tt, config.ordering);
            let result = searcher.search(game, &mut state, max_depth, u64::MAX, &mut ZeroEvaluator);
            let root = Wdl::from_solved_score(result.value);
            let best_action = result.best_move.map(|m| game.action_id(&state, m));
            let (tt_probes, tt_hits) = searcher.tt_stats();
            say(
                format!(
                    "{}: root {root:?} (score {}), best action {best_action:?}, \
                     completed depth {}, {} nodes, TT {tt_probes} probes / {tt_hits} hits",
                    config.method.label(),
                    result.value,
                    result.completed_depth,
                    result.nodes,
                ),
                log,
            );
            serde_json::json!({
                "method": config.method,
                "ordering": config.ordering,
                "root_wdl": root,
                "score": result.value,
                "best_action": best_action,
                "completed_depth": result.completed_depth,
                "nodes": result.nodes,
                "nodes_at_completed_depth": result.nodes_at_completed_depth,
                "tt_probes": tt_probes,
                "tt_hits": tt_hits,
            })
        }
    };

    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    let peak_rss = peak_rss_bytes().unwrap_or(0);
    say(
        format!(
            "wall {wall_seconds:.3}s, cpu {cpu_seconds:.2}s, peak RSS {:.1} MiB",
            peak_rss as f64 / (1024.0 * 1024.0)
        ),
        log,
    );
    let mut summary = summary;
    summary["wall_seconds"] = serde_json::json!(wall_seconds);
    summary["cpu_seconds"] = serde_json::json!(cpu_seconds);
    summary["peak_rss_bytes"] = serde_json::json!(peak_rss);
    run_dir
        .write_json("summary.json", &summary)
        .map_err(|e| e.to_string())
}

/// Retrograde solve for forward chess: solve the reachable graph, back
/// up the full solution as a packed `tablebase.bin` (verified by
/// re-reading every record), and write the JSONL corpus, subsampled
/// deterministically by position-key hash past `CORPUS_ROW_CAP` rows.
fn run_solve_retrograde(
    game: &ForwardChess,
    config: &SolveConfig,
    run_dir: &RunDir,
    log: &mut String,
) -> Result<(), String> {
    const CORPUS_ROW_CAP: u64 = 1_000_000;
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();
    // Position cap sized for this machine's RAM (~330 bytes per
    // position across states, hash index, and both edge maps; 60M ~ 20
    // GB of the 32 GB machine); exceeding it fails gracefully instead
    // of invoking the OOM killer.
    let solution =
        solve_retrograde(game, 60_000_000).map_err(|e| format!("retrograde solve: {e}"))?;
    let solve_seconds = started.elapsed().as_secs_f64();

    let tb_path = run_dir.path().join("tablebase.bin");
    let tb_file =
        std::fs::File::create(&tb_path).map_err(|e| format!("creating tablebase: {e}"))?;
    let mut tb_writer = std::io::BufWriter::new(tb_file);
    let tablebase_bytes = write_tablebase(game, &solution.states, &solution.values, &mut tb_writer)
        .map_err(|e| format!("writing tablebase: {e}"))?;
    drop(tb_writer);
    let mut tb_reader =
        std::fs::File::open(&tb_path).map_err(|e| format!("reopening tablebase: {e}"))?;
    let verified = read_tablebase_with(game, &mut tb_reader, |index, state, wdl| {
        match solution.states.get(index).zip(solution.values.get(index)) {
            Some((expected, &value)) if value == wdl && expected.identical_core(&state) => Ok(()),
            _ => Err(format!("tablebase verification mismatch at record {index}")),
        }
    })
    .map_err(|e| format!("verifying tablebase: {e}"))?;
    if verified as usize != solution.states.len() {
        return Err(format!(
            "tablebase verification count {verified} != {}",
            solution.states.len()
        ));
    }

    let nonterminal = solution
        .states
        .iter()
        .filter(|s| game.outcome(s).is_none())
        .count() as u64;
    let denominator = nonterminal.div_ceil(CORPUS_ROW_CAP).max(1);
    if denominator > 1 {
        say(
            format!(
                "corpus: keeping 1/{denominator} of {nonterminal} non-terminal positions \
                 (deterministic by position-key hash)"
            ),
            log,
        );
    }
    let corpus_file = std::fs::File::create(run_dir.path().join("corpus.jsonl"))
        .map_err(|e| format!("creating corpus file: {e}"))?;
    let mut writer = std::io::BufWriter::new(corpus_file);
    let mut wdl_counts = [0u64; 3];
    let mut corpus_states = 0u64;
    let mut features = Vec::new();
    let mut legal = Vec::new();
    for (index, state) in solution.states.iter().enumerate() {
        if game.outcome(state).is_some() {
            continue;
        }
        let value = solution.values[index];
        wdl_counts[value as usize] += 1;
        let key = game.position_key(state);
        if denominator > 1 && !splitmix64(key).is_multiple_of(denominator) {
            continue;
        }
        corpus_states += 1;
        game.encode_features(state, &mut features);
        game.legal_moves(state, &mut legal);
        let child_wdl = solution.child_values(index);
        let row = CorpusRow {
            key,
            ply: 0,
            wdl: value,
            features: features.clone(),
            legal: legal.iter().map(|&m| game.action_id(state, m)).collect(),
            optimal: legal
                .iter()
                .zip(&child_wdl)
                .filter(|(_, &v)| v == value)
                .map(|(&m, _)| game.action_id(state, m))
                .collect(),
            child_wdl,
        };
        let line = serde_json::to_string(&row).expect("serializable row");
        writer
            .write_all(line.as_bytes())
            .and_then(|()| writer.write_all(b"\n"))
            .map_err(|e| format!("writing corpus: {e}"))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flushing corpus: {e}"))?;

    let root = solution.values[0];
    say(
        format!(
            "retrograde: root {root:?}; {} reachable positions \
             ({nonterminal} non-terminal: W {} / D {} / L {}); \
             {corpus_states} corpus rows; tablebase {tablebase_bytes} bytes, verified",
            solution.states.len(),
            wdl_counts[Wdl::Win as usize],
            wdl_counts[Wdl::Draw as usize],
            wdl_counts[Wdl::Loss as usize],
        ),
        log,
    );
    let mut summary = serde_json::json!({
        "method": config.method,
        "ordering": config.ordering,
        "root_wdl": root,
        "positions": solution.states.len(),
        "nonterminal_states": nonterminal,
        "corpus_states": corpus_states,
        "corpus_sample_denominator": denominator,
        "solve_seconds": solve_seconds,
        "tablebase_bytes": tablebase_bytes,
        "tablebase_verified": true,
        "wdl_counts": {
            "win": wdl_counts[Wdl::Win as usize],
            "draw": wdl_counts[Wdl::Draw as usize],
            "loss": wdl_counts[Wdl::Loss as usize],
        },
    });
    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    let peak_rss = peak_rss_bytes().unwrap_or(0);
    say(
        format!(
            "wall {wall_seconds:.3}s, cpu {cpu_seconds:.2}s, peak RSS {:.1} MiB",
            peak_rss as f64 / (1024.0 * 1024.0)
        ),
        log,
    );
    summary["wall_seconds"] = serde_json::json!(wall_seconds);
    summary["cpu_seconds"] = serde_json::json!(cpu_seconds);
    summary["peak_rss_bytes"] = serde_json::json!(peak_rss);
    run_dir
        .write_json("summary.json", &summary)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// lab train
// ---------------------------------------------------------------------------

/// Fully explicit configuration for `lab train` (supervised, Phase 2).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainConfig {
    seed: u64,
    /// Single capacity knob: embedding and hidden width.
    model_width: usize,
    /// Number of training-split positions to use (data-scaling axis).
    /// Selection is deterministic and independent of `seed`.
    train_positions: usize,
    /// Optimizer steps at the fixed recipe batch size.
    training_steps: u64,
    game: GameSpec,
}

fn train(config_path: &Path) -> Result<(), String> {
    let config: TrainConfig = read_config(config_path)?;
    if matches!(config.game, GameSpec::Chess {}) {
        return Err("supervised oracle training needs an exactly solvable game".into());
    }
    if matches!(config.game, GameSpec::ForwardChess { .. }) {
        return Err(
            "forward chess is loopy; `lab train` uses the acyclic exact solver and is not \
             wired to the retrograde oracle"
                .into(),
        );
    }
    if config.model_width == 0 || config.train_positions == 0 || config.training_steps == 0 {
        return Err("model_width, train_positions, training_steps must be positive".into());
    }
    let label = format!(
        "train-{}-w{}-n{}-s{}",
        config.game.label(),
        config.model_width,
        config.train_positions,
        config.seed
    );
    let (run_dir, mut manifest) = start_run(&label, &config, config.seed, 1)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    say(
        "thread plan: 1 training thread (sweep-level parallelism only)".to_string(),
        &mut log,
    );

    dispatch_game!(&config.game, game, {
        let dims = ModelDims {
            feature_count: game.feature_count(),
            action_count: game.action_count(),
            width: config.model_width,
        };
        run_train(&game, dims, &config, &run_dir, &mut manifest, &mut log)?;
    });
    finish_run(&run_dir, manifest, &log)
}

fn run_train<G: Game>(
    game: &G,
    dims: ModelDims,
    config: &TrainConfig,
    run_dir: &RunDir,
    manifest: &mut Manifest,
    log: &mut String,
) -> Result<(), String> {
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();

    // Dataset: enumerate + solve in-process (deterministic, self-contained).
    let dataset = build_exact_dataset(game);
    say(
        format!(
            "dataset: {} train / {} val / {} test states, max {} features",
            dataset.train.len(),
            dataset.val.len(),
            dataset.test.len(),
            dataset.max_features
        ),
        log,
    );
    if config.train_positions > dataset.train.len() {
        return Err(format!(
            "train_positions {} exceeds train split size {}",
            config.train_positions,
            dataset.train.len()
        ));
    }
    // Deterministic, seed-independent subset so data sweeps compare the
    // same subset across training seeds.
    let subset: Vec<TrainRow> = {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut order: Vec<usize> = (0..dataset.train.len()).collect();
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(0x0D47_45E7);
        order.shuffle(&mut rng);
        order
            .into_iter()
            .take(config.train_positions)
            .map(|i| TrainRow::from(&dataset.train[i]))
            .collect()
    };
    let solve_seconds = started.elapsed().as_secs_f64();
    say(
        format!(
            "training on {} positions (dataset built in {solve_seconds:.1}s)",
            subset.len()
        ),
        log,
    );

    // Training with periodic validation metrics.
    let metrics_path = run_dir.path().join("metrics.jsonl");
    let metrics_file =
        std::fs::File::create(&metrics_path).map_err(|e| format!("creating metrics file: {e}"))?;
    let mut metrics_writer = std::io::BufWriter::new(metrics_file);
    let mut last_val: Option<OracleMetrics> = None;
    let mut log_lines: Vec<String> = Vec::new();
    let net = train_supervised(
        dims,
        &subset,
        dataset.max_features,
        config.seed,
        config.training_steps,
        |step_metrics, net| {
            let mut line = serde_json::to_value(step_metrics).expect("serializable");
            if step_metrics.step % EVAL_EVERY == 0 || step_metrics.step == config.training_steps {
                let val_metrics =
                    evaluate_model_oracle(&net.valid(), &dataset.val, dims, dataset.max_features);
                log_lines.push(format!(
                    "step {}: wdl_loss {:.4}, policy_loss {:.4}; val: wdl_acc {:.4}, \
                     action_acc {:.4}, optimal_mass {:.4}, regret {:.4}",
                    step_metrics.step,
                    step_metrics.wdl_loss,
                    step_metrics.policy_loss,
                    val_metrics.wdl_accuracy,
                    val_metrics.action_accuracy,
                    val_metrics.optimal_mass,
                    val_metrics.mean_regret_levels,
                ));
                line["val"] = serde_json::to_value(&val_metrics).expect("serializable");
                last_val = Some(val_metrics);
            }
            let _ = writeln!(metrics_writer, "{line}");
        },
    );
    metrics_writer
        .flush()
        .map_err(|e| format!("flushing metrics: {e}"))?;
    for line in log_lines {
        say(line, log);
    }
    let train_wall = started.elapsed().as_secs_f64() - solve_seconds;

    // Checkpoint: save, reload, verify bit-identical outputs on a probe.
    run_dir
        .create_subdir("checkpoint")
        .map_err(|e| e.to_string())?;
    let device = Default::default();
    let inference_net = net.valid();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = run_dir.path().join("checkpoint/model");
    inference_net
        .clone()
        .save_file(&model_path, &recorder)
        .map_err(|e| format!("saving checkpoint: {e}"))?;
    run_dir
        .write_json(
            "checkpoint/model.json",
            &serde_json::json!({
                "dims": dims,
                "max_features": dataset.max_features,
                "seed": config.seed,
                "training_steps": config.training_steps,
                "train_positions": config.train_positions,
            }),
        )
        .map_err(|e| e.to_string())?;
    let restored = PolicyValueNet::<InferBackend>::new(dims, &device)
        .load_file(&model_path, &recorder, &device)
        .map_err(|e| format!("reloading checkpoint: {e}"))?;
    let probe_rows: Vec<TrainRow> = dataset.val.iter().take(64).map(TrainRow::from).collect();
    let probe: Vec<&TrainRow> = probe_rows.iter().collect();
    let probe_batch = make_batch::<InferBackend>(&probe, dims, dataset.max_features, &device);
    let (a_wdl, a_act) = inference_net.forward(
        probe_batch.feature_ids.clone(),
        probe_batch.feature_mask.clone(),
    );
    let (b_wdl, b_act) = restored.forward(probe_batch.feature_ids, probe_batch.feature_mask);
    if a_wdl.to_data().to_vec::<f32>().unwrap() != b_wdl.to_data().to_vec::<f32>().unwrap()
        || a_act.to_data().to_vec::<f32>().unwrap() != b_act.to_data().to_vec::<f32>().unwrap()
    {
        return Err("checkpoint reload changed model outputs".into());
    }
    say("checkpoint saved and reload-verified".to_string(), log);

    // Final evaluation on all three splits.
    let train_metrics =
        evaluate_model_oracle(&restored, &dataset.train, dims, dataset.max_features);
    let val_metrics = evaluate_model_oracle(&restored, &dataset.val, dims, dataset.max_features);
    let test_metrics = evaluate_model_oracle(&restored, &dataset.test, dims, dataset.max_features);
    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    let peak_rss = peak_rss_bytes().unwrap_or(0);
    let params = restored.parameter_count();
    manifest.model_parameter_count = params as u64;
    let examples_per_second =
        (config.training_steps * BATCH_SIZE as u64) as f64 / train_wall.max(1e-9);
    say(
        format!(
            "test: wdl_acc {:.4}, log_loss {:.4}, action_acc {:.4}, optimal_mass {:.4}, \
             regret {:.4} ({} params, {:.0} ex/s)",
            test_metrics.wdl_accuracy,
            test_metrics.wdl_log_loss,
            test_metrics.action_accuracy,
            test_metrics.optimal_mass,
            test_metrics.mean_regret_levels,
            params,
            examples_per_second,
        ),
        log,
    );
    say(
        format!(
            "wall {wall_seconds:.1}s (train {train_wall:.1}s), cpu {cpu_seconds:.1}s, \
             peak RSS {:.0} MiB",
            peak_rss as f64 / (1024.0 * 1024.0)
        ),
        log,
    );
    run_dir
        .write_json(
            "summary.json",
            &serde_json::json!({
                "config": config,
                "parameter_count": params,
                "dataset": {
                    "train": dataset.train.len(),
                    "val": dataset.val.len(),
                    "test": dataset.test.len(),
                    "used_train_positions": subset.len(),
                },
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "last_periodic_val": last_val,
                "examples_per_second": examples_per_second,
                "dataset_seconds": solve_seconds,
                "train_seconds": train_wall,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "peak_rss_bytes": peak_rss,
            }),
        )
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// lab sweep
// ---------------------------------------------------------------------------

/// One line of a sweep manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SweepEntry {
    /// `lab` subcommand to run: "train", "solve", "evaluate",
    /// "selfplay", "fit", or "relabel".
    command: String,
    config: PathBuf,
    /// CPU slots this run occupies while active.
    cores: usize,
}

fn sweep(manifest_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(manifest_path)
        .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
    let entries: Vec<SweepEntry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("manifest line {l:?}: {e}")))
        .collect::<Result<_, _>>()?;
    if entries.is_empty() {
        return Err("sweep manifest is empty".into());
    }
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    for entry in &entries {
        if entry.cores == 0 || entry.cores > total_cores {
            return Err(format!(
                "entry {} requests {} cores (machine has {total_cores})",
                entry.config.display(),
                entry.cores
            ));
        }
        if !matches!(
            entry.command.as_str(),
            "train" | "solve" | "evaluate" | "selfplay" | "fit" | "relabel"
        ) {
            return Err(format!("unknown sweep command {:?}", entry.command));
        }
    }
    let exe = std::env::current_exe().map_err(|e| format!("finding lab binary: {e}"))?;
    println!(
        "sweep: {} runs on {total_cores} cores from {}",
        entries.len(),
        manifest_path.display()
    );

    struct Active {
        index: usize,
        child: std::process::Child,
        cores: usize,
    }
    let mut pending: std::collections::VecDeque<(usize, SweepEntry)> =
        entries.into_iter().enumerate().collect();
    let mut active: Vec<Active> = Vec::new();
    let mut free = total_cores;
    let mut failures = 0usize;
    let started = Instant::now();

    while !pending.is_empty() || !active.is_empty() {
        while let Some((index, entry)) = pending.front().cloned() {
            if entry.cores > free {
                break;
            }
            pending.pop_front();
            let child = std::process::Command::new(&exe)
                .arg(&entry.command)
                .arg(&entry.config)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| format!("spawning {}: {e}", entry.config.display()))?;
            println!(
                "[{:7.1}s] start #{index} {} {} ({} cores, {} free)",
                started.elapsed().as_secs_f64(),
                entry.command,
                entry.config.display(),
                entry.cores,
                free - entry.cores
            );
            free -= entry.cores;
            active.push(Active {
                index,
                child,
                cores: entry.cores,
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut i = 0;
        while i < active.len() {
            match active[i].child.try_wait() {
                Ok(Some(status)) => {
                    let mut done = active.swap_remove(i);
                    free += done.cores;
                    if status.success() {
                        println!(
                            "[{:7.1}s] done  #{}",
                            started.elapsed().as_secs_f64(),
                            done.index
                        );
                    } else {
                        failures += 1;
                        let mut stderr = String::new();
                        if let Some(mut pipe) = done.child.stderr.take() {
                            use std::io::Read as _;
                            let _ = pipe.read_to_string(&mut stderr);
                        }
                        println!(
                            "[{:7.1}s] FAIL  #{} ({status}): {}",
                            started.elapsed().as_secs_f64(),
                            done.index,
                            stderr.trim()
                        );
                    }
                }
                Ok(None) => i += 1,
                Err(e) => return Err(format!("waiting on child: {e}")),
            }
        }
    }
    println!(
        "sweep finished in {:.1}s with {failures} failure(s)",
        started.elapsed().as_secs_f64()
    );
    if failures > 0 {
        return Err(format!("{failures} sweep run(s) failed"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// lab selfplay
// ---------------------------------------------------------------------------

/// Fully explicit configuration for `lab selfplay` (Expert Iteration,
/// Phase 3). Every field is a §23-mandated experiment axis or a fixed
/// mechanical parameter of the loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelfPlayConfig {
    seed: u64,
    model_width: usize,
    generations: u32,
    games_per_generation: u64,
    /// Optimizer steps per generation at the recipe batch size.
    steps_per_generation: u64,
    /// Exploration probability (§12.4); the label stays the expert move.
    epsilon: f64,
    /// Node budget per move during data generation.
    gen_node_budget: u64,
    /// Node budget per move for per-generation exploitability probes.
    eval_node_budget: u64,
    /// FIFO replay window in generations (1 = current generation only;
    /// §12.5 sanctions one fixed window once forgetting is measured).
    replay_generations: u32,
    /// Promotion mode (§12.6): "oracle" for exactly solved games,
    /// "match" for paired games against the champion.
    promotion: String,
    /// Pairs per promotion/progression match. Must be 0 in oracle mode
    /// and positive in match mode.
    promotion_pairs: u64,
    /// Uniform-random shared opening plies per match pair. Must be 0 in
    /// oracle mode.
    opening_plies: u32,
    threads: usize,
    /// Optional checkpoint directory to initialize the generation-0
    /// champion from, for chunked resumable campaigns (tools/fc_train.sh).
    /// Dims must match `model_width` and `game`. Progression baselines
    /// are per-chunk.
    #[serde(default)]
    init_checkpoint: Option<PathBuf>,
    /// Optional `replay.jsonl` from a previous run to pre-fill the FIFO
    /// replay window (every self-play run writes one). Without it, the
    /// first generations of a continuation chunk train on a single
    /// generation of data and regress against a replay-trained champion
    /// (observed on fc-full: two strikes and a spurious §12.6 halt).
    #[serde(default)]
    init_replay: Option<PathBuf>,
    game: GameSpec,
}

fn selfplay(config_path: &Path) -> Result<(), String> {
    let config: SelfPlayConfig = read_config(config_path)?;
    if config.generations == 0 || config.games_per_generation == 0 {
        return Err("generations and games_per_generation must be positive".into());
    }
    if config.replay_generations == 0 {
        return Err("replay_generations must be at least 1".into());
    }
    if matches!(config.game, GameSpec::Chess {}) && config.promotion == "oracle" {
        return Err("chess self-play requires match promotion (not exactly solvable)".into());
    }
    match config.promotion.as_str() {
        "oracle" => {
            if config.promotion_pairs != 0 || config.opening_plies != 0 {
                return Err(
                    "oracle promotion takes promotion_pairs = 0 and opening_plies = 0".into(),
                );
            }
        }
        "match" => {
            if config.promotion_pairs == 0 {
                return Err("match promotion requires positive promotion_pairs".into());
            }
        }
        other => return Err(format!("unknown promotion mode {other:?}")),
    }
    if !(0.0..=1.0).contains(&config.epsilon) {
        return Err("epsilon must be in [0, 1]".into());
    }
    let label = format!(
        "selfplay-{}-w{}-g{}-e{}-s{}",
        config.game.label(),
        config.model_width,
        config.games_per_generation,
        (config.epsilon * 100.0).round() as u32,
        config.seed
    );
    let (run_dir, mut manifest) = start_run(&label, &config, config.seed, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    say(
        format!(
            "thread plan: {} self-play workers; single-threaded training",
            config.threads
        ),
        &mut log,
    );
    dispatch_game!(&config.game, game, {
        let dims = ModelDims {
            feature_count: game.feature_count(),
            action_count: game.action_count(),
            width: config.model_width,
        };
        run_selfplay(&game, dims, &config, &run_dir, &mut manifest, &mut log)?;
    });
    finish_run(&run_dir, manifest, &log)
}

fn run_selfplay<G: Game>(
    game: &G,
    dims: ModelDims,
    config: &SelfPlayConfig,
    run_dir: &RunDir,
    manifest: &mut Manifest,
    log: &mut String,
) -> Result<(), String> {
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();
    let device = Default::default();

    // Oracle corpus for evaluation only (never for training labels);
    // match-mode games are not exactly solvable, so they use paired
    // matches instead (§12.6).
    // Loopy games must never touch the acyclic exact solver: its
    // path-dependent repetition memoization is unsound and its
    // optimal-move lists can come back empty (D029).
    let loopy = matches!(config.game, GameSpec::ForwardChess { .. });
    let mut fc_val_probes = None;
    let mut fc_test_probes = None;
    let dataset = if config.promotion == "oracle" {
        // Forward chess repeats, so its oracle comes from the
        // retrograde solver (evaluation buckets thinned to stay
        // bounded); acyclic games keep the exact-solver corpus. The
        // searched-decision probe states are drawn from the same
        // solution before it is dropped.
        let dataset = if loopy {
            let solution = solve_retrograde(game, 60_000_000)?;
            fc_val_probes = Some(retrograde_searched_candidates(
                game,
                &solution,
                CorpusSplit::Val,
                500,
            ));
            fc_test_probes = Some(retrograde_searched_candidates(
                game,
                &solution,
                CorpusSplit::Test,
                2000,
            ));
            build_retrograde_dataset(game, &solution, 20_000)
        } else {
            build_exact_dataset(game)
        };
        say(
            format!(
                "oracle corpus for evaluation: {} val / {} test states",
                dataset.val.len(),
                dataset.test.len()
            ),
            log,
        );
        Some(dataset)
    } else {
        say("match promotion: no oracle corpus".to_string(), log);
        None
    };

    run_dir
        .create_subdir("selfplay")
        .map_err(|e| e.to_string())?;
    run_dir
        .create_subdir("checkpoints")
        .map_err(|e| e.to_string())?;

    // Generation 0 champion: random initialization from the run seed,
    // or a prior campaign checkpoint when resuming in chunks.
    TrainBackend::seed(&device, config.seed);
    let mut champion = PolicyValueNet::<TrainBackend>::new(dims, &device);
    if let Some(init) = &config.init_checkpoint {
        let (_, init_dims, _) = load_checkpoint(init)?;
        if init_dims != dims {
            return Err(format!(
                "init_checkpoint dims {init_dims:?} do not match configured {dims:?}"
            ));
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        champion = champion
            .load_file(init.join("model"), &recorder, &device)
            .map_err(|e| format!("loading init_checkpoint: {e}"))?;
        say(format!("champion initialized from {}", init.display()), log);
    }
    manifest.model_parameter_count = champion.num_params() as u64;
    let mut champion_val = dataset
        .as_ref()
        .map(|d| evaluate_model_oracle(&champion.valid(), &d.val, dims, d.max_features));
    let gen0_compiled = CompiledNet::from_net(&champion.valid(), dims);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let mut generation_summaries: Vec<serde_json::Value> = Vec::new();
    let mut consecutive_rejections = 0u32;
    let mut max_features_seen = 1usize;
    let mut replay: std::collections::VecDeque<Vec<TrainRow>> = std::collections::VecDeque::new();
    if let Some(path) = &config.init_replay {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading init_replay {}: {e}", path.display()))?;
        let mut loaded: Vec<(u64, Vec<TrainRow>)> = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("init_replay line {}: {e}", index + 1))?;
            let generation = value["generation"]
                .as_u64()
                .ok_or("init_replay: missing generation tag")?;
            let rows: Vec<TrainRow> = serde_json::from_value(value["rows"].clone())
                .map_err(|e| format!("init_replay line {} rows: {e}", index + 1))?;
            loaded.push((generation, rows));
        }
        loaded.sort_by_key(|(generation, _)| *generation);
        let skip = loaded
            .len()
            .saturating_sub(config.replay_generations as usize);
        for (_, rows) in loaded.into_iter().skip(skip) {
            replay.push_back(rows);
        }
        say(
            format!(
                "replay window pre-filled from {}: {} generations, {} rows",
                path.display(),
                replay.len(),
                replay.iter().map(Vec::len).sum::<usize>()
            ),
            log,
        );
    }

    for generation in 0..config.generations {
        let gen_started = Instant::now();
        let compiled = CompiledNet::from_net(&champion.valid(), dims);

        // 1. Frozen-champion self-play data.
        let (records, stats) = generate_selfplay(
            game,
            &compiled,
            config.games_per_generation,
            config.gen_node_budget,
            config.epsilon,
            config.seed,
            generation,
            config.threads,
        );
        let lines: Vec<String> = records
            .iter()
            .map(|r| serde_json::to_string(r).expect("serializable record"))
            .collect();
        run_dir
            .write_text(
                &format!("selfplay/gen_{generation:03}.jsonl"),
                &(lines.join("\n") + "\n"),
            )
            .map_err(|e| e.to_string())?;

        // 2. Candidate: warm start from champion, fixed optimizer steps
        // on the FIFO replay window.
        replay.push_back(
            records
                .iter()
                .map(|r| r.to_row())
                .collect::<Vec<TrainRow>>(),
        );
        while replay.len() > config.replay_generations as usize {
            replay.pop_front();
        }
        let rows: Vec<TrainRow> = replay.iter().flatten().cloned().collect();
        let max_features = dataset
            .as_ref()
            .map(|d| d.max_features)
            .unwrap_or_else(|| rows.iter().map(|r| r.features.len()).max().unwrap_or(1));
        max_features_seen = max_features_seen.max(max_features);
        let stream_seed = config.seed ^ (u64::from(generation) << 20);
        let mut last_losses = (f32::NAN, f32::NAN);
        let candidate = train_steps(
            champion.clone(),
            dims,
            &rows,
            max_features,
            stream_seed,
            config.steps_per_generation,
            |m, _| last_losses = (m.wdl_loss, m.policy_loss),
        );

        // 3. Promotion (§12.6). Oracle mode, rule v2: candidate regret
        // may exceed the champion's by at most a fixed noise tolerance
        // (the strict rule halted runs on ~0.006 regret jitter at
        // plateau; see DECISIONS.md). Match mode: paired games against
        // the frozen champion; promote only when the candidate's 95%
        // lower confidence bound exceeds 0.5.
        const PROMOTION_REGRET_TOLERANCE: f64 = 0.005;
        let mut candidate_val = None;
        let mut promotion_match = None;
        let promoted = if let Some(dataset) = &dataset {
            let metrics =
                evaluate_model_oracle(&candidate.valid(), &dataset.val, dims, dataset.max_features);
            let promoted = metrics.mean_regret_levels
                <= champion_val
                    .as_ref()
                    .expect("oracle mode")
                    .mean_regret_levels
                    + PROMOTION_REGRET_TOLERANCE;
            if promoted {
                champion_val = Some(metrics.clone());
            }
            candidate_val = Some(metrics);
            promoted
        } else {
            let candidate_compiled = CompiledNet::from_net(&candidate.valid(), dims);
            let result = play_paired_match(
                game,
                &candidate_compiled,
                &compiled,
                config.promotion_pairs,
                config.opening_plies,
                config.gen_node_budget,
                config.gen_node_budget,
                config.seed ^ (u64::from(generation) << 32) ^ 0x9a7c,
                config.threads,
            );
            let promoted = result.score_lcb95 > 0.5;
            promotion_match = Some(result);
            promoted
        };
        // A match-mode candidate that is not provably better keeps the
        // champion, and only a PROVABLE regression (upper confidence
        // bound below 0.5) strikes toward the §12.6 halt: a sub-0.5
        // score alone is plateau noise and halted fc-full spuriously
        // (0.433/0.475 over 60 games; D030).
        let regression = match &promotion_match {
            Some(result) => result.score_ucb95 < 0.5,
            None => !promoted,
        };
        if promoted {
            champion = candidate;
            consecutive_rejections = 0;
        } else if regression {
            consecutive_rejections += 1;
        }

        // 4. Per-generation probes with the (possibly updated) champion:
        // oracle metrics when solvable, otherwise a champion-progression
        // match against the frozen generation-0 baseline.
        let probe_net = CompiledNet::from_net(&champion.valid(), dims);
        let mut exploit = None;
        let mut searched = None;
        let mut progression = None;
        if dataset.is_some() {
            // The exploitability play-out uses the acyclic exact solver
            // as the perfect opponent; loopy games skip it (D029).
            if !loopy {
                exploit = Some(exploitability_vs_perfect(
                    game,
                    &probe_net,
                    config.eval_node_budget,
                    config.threads,
                ));
            }
            searched = Some(match &fc_val_probes {
                Some(probes) => searched_decision_metrics_on(
                    game,
                    &probe_net,
                    probes,
                    config.eval_node_budget,
                    config.threads,
                ),
                None => searched_decision_metrics(
                    game,
                    &probe_net,
                    CorpusSplit::Val,
                    500,
                    config.eval_node_budget,
                    config.threads,
                ),
            });
        } else {
            progression = Some(play_paired_match(
                game,
                &probe_net,
                &gen0_compiled,
                config.promotion_pairs,
                config.opening_plies,
                config.eval_node_budget,
                config.eval_node_budget,
                config.seed ^ 0x6e40,
                config.threads,
            ));
        }

        champion
            .valid()
            .clone()
            .save_file(
                run_dir
                    .path()
                    .join(format!("checkpoints/gen_{generation:03}")),
                &recorder,
            )
            .map_err(|e| format!("saving generation checkpoint: {e}"))?;

        let distinct_positions = {
            let mut keys = std::collections::HashSet::new();
            for record in &records {
                keys.insert(record.features.clone());
            }
            keys.len()
        };
        let gen_summary = serde_json::json!({
            "generation": generation,
            "selfplay": stats,
            "distinct_positions": distinct_positions,
            "replay_rows": rows.len(),
            "train_wdl_loss": last_losses.0,
            "train_policy_loss": last_losses.1,
            "candidate_val": candidate_val,
            "promotion_match": promotion_match,
            "promoted": promoted,
            "champion_val": champion_val,
            "exploitability": exploit,
            "searched_val": searched,
            "progression_vs_gen0": progression,
            "generation_seconds": gen_started.elapsed().as_secs_f64(),
        });
        run_dir
            .append_jsonl("metrics.jsonl", &[&gen_summary])
            .map_err(|e| e.to_string())?;
        let line = if let (Some(candidate_val), Some(searched)) = (&candidate_val, &searched) {
            let exploit_part = match &exploit {
                Some(e) => format!("exploit drops {}/{}", e.avoidable_drops, e.games),
                None => "exploit n/a (loopy)".to_string(),
            };
            format!(
                "gen {generation}: {} games, {} positions ({} explore), losses {:.3}/{:.3}, \
                 val regret {:.4} ({}), searched@{} acc {:.4}, {exploit_part} ({:.1}s)",
                stats.games,
                stats.positions,
                stats.exploratory_moves,
                last_losses.0,
                last_losses.1,
                candidate_val.mean_regret_levels,
                if promoted { "promoted" } else { "REJECTED" },
                config.eval_node_budget,
                searched.action_accuracy,
                gen_started.elapsed().as_secs_f64(),
            )
        } else {
            let m = promotion_match.as_ref().expect("match mode");
            let p = progression.as_ref().expect("match mode");
            format!(
                "gen {generation}: {} games, {} positions ({} explore), losses {:.3}/{:.3}, \
                 vs champion {:.3} (lcb {:.3}, {}), vs gen0 {:.3} ({:.1}s)",
                stats.games,
                stats.positions,
                stats.exploratory_moves,
                last_losses.0,
                last_losses.1,
                m.score,
                m.score_lcb95,
                if promoted { "promoted" } else { "REJECTED" },
                p.score,
                gen_started.elapsed().as_secs_f64(),
            )
        };
        say(line, log);
        generation_summaries.push(gen_summary);

        if consecutive_rejections >= 2 {
            say(
                "halting: two consecutive candidates were worse than the champion \
                 (plan §12.6: stop and diagnose)"
                    .to_string(),
                log,
            );
            break;
        }
    }

    // Persist the FIFO replay window (also on halt) so a follow-on
    // campaign chunk can continue training seamlessly via init_replay.
    {
        let file = std::fs::File::create(run_dir.path().join("replay.jsonl"))
            .map_err(|e| format!("creating replay.jsonl: {e}"))?;
        let mut writer = std::io::BufWriter::new(file);
        for (index, generation_rows) in replay.iter().enumerate() {
            let line = serde_json::json!({"generation": index, "rows": generation_rows});
            writer
                .write_all(line.to_string().as_bytes())
                .and_then(|()| writer.write_all(b"\n"))
                .map_err(|e| format!("writing replay.jsonl: {e}"))?;
        }
        writer
            .flush()
            .map_err(|e| format!("flushing replay.jsonl: {e}"))?;
    }

    // Final champion: probe-compatible checkpoint plus full held-out
    // evaluation.
    let final_infer = champion.valid();
    run_dir
        .create_subdir("checkpoint")
        .map_err(|e| e.to_string())?;
    final_infer
        .clone()
        .save_file(run_dir.path().join("checkpoint/model"), &recorder)
        .map_err(|e| format!("saving final checkpoint: {e}"))?;
    let checkpoint_max_features = dataset
        .as_ref()
        .map(|d| d.max_features)
        .unwrap_or(max_features_seen);
    run_dir
        .write_json(
            "checkpoint/model.json",
            &serde_json::json!({
                "dims": dims,
                "max_features": checkpoint_max_features,
                "seed": config.seed,
            }),
        )
        .map_err(|e| e.to_string())?;
    let final_compiled = CompiledNet::from_net(&final_infer, dims);
    let mut final_summary = serde_json::json!({
        "config": config,
        "parameter_count": manifest.model_parameter_count,
        "generations": generation_summaries,
    });
    if let Some(dataset) = &dataset {
        let raw_test =
            evaluate_model_oracle(&final_infer, &dataset.test, dims, dataset.max_features);
        let searched_test = match &fc_test_probes {
            Some(probes) => searched_decision_metrics_on(
                game,
                &final_compiled,
                probes,
                config.eval_node_budget,
                config.threads,
            ),
            None => searched_decision_metrics(
                game,
                &final_compiled,
                CorpusSplit::Test,
                2000,
                config.eval_node_budget,
                config.threads,
            ),
        };
        let exploit_final = (!loopy).then(|| {
            exploitability_vs_perfect(
                game,
                &final_compiled,
                config.eval_node_budget,
                config.threads,
            )
        });
        let exploit_part = match &exploit_final {
            Some(e) => format!("exploit drops {}/{}", e.avoidable_drops, e.games),
            None => "exploit n/a (loopy)".to_string(),
        };
        say(
            format!(
                "final: raw test regret {:.4}, searched test acc {:.4} / regret {:.4}, \
                 {exploit_part}",
                raw_test.mean_regret_levels,
                searched_test.action_accuracy,
                searched_test.mean_regret_levels,
            ),
            log,
        );
        final_summary["final_raw_test"] = serde_json::to_value(&raw_test).expect("serializable");
        final_summary["final_searched_test"] =
            serde_json::to_value(&searched_test).expect("serializable");
        final_summary["final_exploitability"] =
            serde_json::to_value(&exploit_final).expect("serializable");
    } else {
        // Champion progression: a larger final match against the frozen
        // generation-0 baseline.
        let final_match = play_paired_match(
            game,
            &final_compiled,
            &gen0_compiled,
            config.promotion_pairs * 2,
            config.opening_plies,
            config.eval_node_budget,
            config.eval_node_budget,
            config.seed ^ 0xf17a1,
            config.threads,
        );
        say(
            format!(
                "final: vs gen0 score {:.3} (lcb {:.3}) over {} games, mean plies {:.1}",
                final_match.score,
                final_match.score_lcb95,
                final_match.games,
                final_match.mean_plies,
            ),
            log,
        );
        final_summary["final_vs_gen0"] = serde_json::to_value(&final_match).expect("serializable");
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    say(
        format!("wall {wall_seconds:.0}s cpu {cpu_seconds:.0}s"),
        log,
    );
    final_summary["wall_seconds"] = serde_json::json!(wall_seconds);
    final_summary["cpu_seconds"] = serde_json::json!(cpu_seconds);
    final_summary["peak_rss_bytes"] = serde_json::json!(peak_rss_bytes().unwrap_or(0));
    run_dir
        .write_json("summary.json", &final_summary)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// lab evaluate: oracle probe
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// lab play (interactive forward chess)
// ---------------------------------------------------------------------------

fn describe_outcome(outcome: selfplay_lab::game::Outcome) -> &'static str {
    use selfplay_lab::game::{Outcome, Player};
    match outcome {
        Outcome::Draw => "Draw.",
        Outcome::Win(Player::One) => "White wins.",
        Outcome::Win(Player::Two) => "Black wins.",
    }
}

fn fc_engine_move(
    game: &ForwardChess,
    net: Option<&CompiledNet>,
    state: &mut selfplay_lab::games::forward_chess::FcState,
    nodes: u64,
) -> Result<selfplay_lab::games::forward_chess::FcMove, String> {
    let mut searcher: Searcher<ForwardChess> = Searcher::new(
        Some(selfplay_lab::training::SELFPLAY_TT_LOG2),
        MoveOrdering::Natural,
    );
    let result = match net {
        Some(net) => {
            let mut evaluator = selfplay_lab::model::ModelEvaluator::new(net);
            searcher.search(game, state, 512, nodes, &mut evaluator)
        }
        None => searcher.search(game, state, 512, nodes, &mut ZeroEvaluator),
    };
    result
        .best_move
        .ok_or_else(|| "search returned no move".to_string())
}

fn bench(
    game_label: &str,
    checkpoint: Option<&Path>,
    budgets: &[u64],
    movetime_ms: u64,
    positions: usize,
    seed: u64,
) -> Result<(), String> {
    use selfplay_lab::games::forward_chess::Ruleset;
    let spec = match game_label {
        "fc-tiny" => GameSpec::ForwardChess {
            ruleset: Ruleset::Tiny,
        },
        "fc-small" => GameSpec::ForwardChess {
            ruleset: Ruleset::Small,
        },
        "fc-medium" => GameSpec::ForwardChess {
            ruleset: Ruleset::Medium,
        },
        "fc-full" => GameSpec::ForwardChess {
            ruleset: Ruleset::Full,
        },
        "chess" => GameSpec::Chess {},
        other => {
            return Err(format!(
                "unknown game {other}; use fc-tiny|fc-small|fc-medium|fc-full|chess"
            ))
        }
    };
    if budgets.is_empty() {
        return Err("give at least one node budget".to_string());
    }
    dispatch_game!(&spec, game, {
        let net = match checkpoint {
            Some(dir) => {
                let (net, dims, _) = load_checkpoint(dir)?;
                if dims.feature_count != game.feature_count()
                    || dims.action_count != game.action_count()
                {
                    return Err(format!(
                        "checkpoint dims {dims:?} do not fit {game_label} \
                         ({} features / {} actions)",
                        game.feature_count(),
                        game.action_count()
                    ));
                }
                let params = (dims.feature_count + 1) * dims.width
                    + 2 * (dims.width * dims.width + dims.width)
                    + (3 * dims.width + 3)
                    + (dims.action_count * dims.width + dims.action_count);
                println!(
                    "bench {game_label}: checkpoint {} (w{}, {} params, {:.2} MB f32)",
                    dir.display(),
                    dims.width,
                    params,
                    params as f64 * 4.0 / 1e6
                );
                Some(CompiledNet::from_net(&net, dims))
            }
            None => {
                println!("bench {game_label}: zero evaluator (unlearned search)");
                None
            }
        };
        bench_run(&game, net.as_ref(), budgets, movetime_ms, positions, seed)
    })
}

fn bench_run<G: Game>(
    game: &G,
    net: Option<&CompiledNet>,
    budgets: &[u64],
    movetime_ms: u64,
    positions: usize,
    seed: u64,
) -> Result<(), String> {
    use selfplay_lab::training::SELFPLAY_TT_LOG2;

    fn search_once<G: Game>(
        searcher: &mut Searcher<G>,
        game: &G,
        state: &mut G::State,
        net: Option<&CompiledNet>,
        budget: u64,
    ) -> selfplay_lab::search::SearchResult<G::Move> {
        match net {
            Some(net) => {
                let mut evaluator = selfplay_lab::model::ModelEvaluator::new(net);
                searcher.search(game, state, 512, budget, &mut evaluator)
            }
            None => searcher.search(game, state, 512, budget, &mut ZeroEvaluator),
        }
    }

    // Deterministic ε-greedy self-play with the benched checkpoint
    // supplies positions from the distribution the engine actually
    // searches in games.
    let gen_nodes = budgets[0];
    let mut states: Vec<G::State> = Vec::new();
    let mut searcher: Searcher<G> = Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
    let mut moves = Vec::new();
    let mut game_idx = 0u64;
    while states.len() < positions {
        let mut state = game.initial_state();
        let mut ply = 0u64;
        while game.outcome(&state).is_none() && ply < 240 && states.len() < positions {
            states.push(state.clone());
            game.legal_moves(&state, &mut moves);
            let roll = splitmix64(seed ^ (game_idx << 32) ^ ply);
            let mv = if roll.is_multiple_of(10) {
                moves[(splitmix64(roll) as usize) % moves.len()]
            } else {
                search_once(&mut searcher, game, &mut state, net, gen_nodes)
                    .best_move
                    .ok_or("search returned no move")?
            };
            game.make_move(&mut state, mv);
            ply += 1;
        }
        game_idx += 1;
    }
    println!(
        "{} positions from {} self-play game(s), seed {seed}; single thread\n",
        states.len(),
        game_idx
    );

    for &budget in budgets {
        let mut searcher: Searcher<G> =
            Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
        let (mut nodes_total, mut depth_total) = (0u64, 0u64);
        let (mut depth_min, mut depth_max) = (u32::MAX, 0u32);
        let start = Instant::now();
        for st in &states {
            let mut state = st.clone();
            let result = search_once(&mut searcher, game, &mut state, net, budget);
            nodes_total += result.nodes;
            depth_total += u64::from(result.completed_depth);
            depth_min = depth_min.min(result.completed_depth);
            depth_max = depth_max.max(result.completed_depth);
        }
        let secs = start.elapsed().as_secs_f64();
        let n = states.len() as f64;
        println!(
            "budget {budget:>7} nodes: {:>8.1} ms/move  {:>9.0} nodes/s  \
             depth mean {:>4.1} [{}..{}]  mean {:.0} nodes searched",
            secs * 1000.0 / n,
            nodes_total as f64 / secs,
            depth_total as f64 / n,
            depth_min,
            depth_max,
            nodes_total as f64 / n,
        );
    }

    if movetime_ms > 0 {
        let mut searcher: Searcher<G> =
            Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
        let (mut nodes_total, mut depth_total) = (0u64, 0u64);
        let (mut depth_min, mut depth_max) = (u32::MAX, 0u32);
        let start = Instant::now();
        for st in &states {
            let mut state = st.clone();
            searcher.set_deadline(Some(
                Instant::now() + std::time::Duration::from_millis(movetime_ms),
            ));
            let result = search_once(&mut searcher, game, &mut state, net, u64::MAX);
            nodes_total += result.nodes;
            depth_total += u64::from(result.completed_depth);
            depth_min = depth_min.min(result.completed_depth);
            depth_max = depth_max.max(result.completed_depth);
        }
        searcher.set_deadline(None);
        let secs = start.elapsed().as_secs_f64();
        let n = states.len() as f64;
        println!(
            "movetime {movetime_ms:>4} ms:      {:>8.1} ms/move  {:>9.0} nodes/s  \
             depth mean {:>4.1} [{}..{}]  mean {:.0} nodes searched",
            secs * 1000.0 / n,
            nodes_total as f64 / secs,
            depth_total as f64 / n,
            depth_min,
            depth_max,
            nodes_total as f64 / n,
        );
    }
    Ok(())
}

fn play(game_label: &str, checkpoint: Option<&Path>, nodes: u64, side: &str) -> Result<(), String> {
    use selfplay_lab::game::Player;
    use selfplay_lab::games::forward_chess::Ruleset;

    let ruleset = match game_label {
        "fc-tiny" => Ruleset::Tiny,
        "fc-small" => Ruleset::Small,
        "fc-medium" => Ruleset::Medium,
        "fc-full" => Ruleset::Full,
        other => {
            return Err(format!(
                "unknown game {other}; use fc-tiny|fc-small|fc-medium|fc-full \
                 (standard chess: use the `uci` binary in a UCI GUI)"
            ))
        }
    };
    let game = ForwardChess::new(ruleset);
    let human = match side {
        "white" => Some(Player::One),
        "black" => Some(Player::Two),
        "none" => None,
        other => return Err(format!("unknown side {other}; use white|black|none")),
    };
    let net = match checkpoint {
        Some(dir) => {
            let (net, dims, _) = load_checkpoint(dir)?;
            if dims.feature_count != game.feature_count()
                || dims.action_count != game.action_count()
            {
                return Err(format!(
                    "checkpoint dims {dims:?} do not fit {game_label} \
                     ({} features / {} actions)",
                    game.feature_count(),
                    game.action_count()
                ));
            }
            Some(CompiledNet::from_net(&net, dims))
        }
        None => None,
    };
    println!(
        "Forward Chess {game_label} — engine: {} at {nodes} nodes/move.",
        if net.is_some() {
            "checkpoint"
        } else {
            "unlearned search (zero evaluator)"
        }
    );
    println!("Moves: coordinates like a2a3 or a7a8=Q. Commands: ? hint undo quit.");

    let mut state = game.initial_state();
    let mut history = vec![state.clone()];
    let mut moves = Vec::new();
    let stdin = std::io::stdin();
    loop {
        println!("\n{}", game.render_ascii(&state));
        if let Some(outcome) = game.outcome(&state) {
            println!("Game over: {}", describe_outcome(outcome));
            break;
        }
        let human_turn = human == Some(game.side_to_move(&state));
        game.legal_moves(&state, &mut moves);
        if !human_turn {
            let mv = fc_engine_move(&game, net.as_ref(), &mut state, nodes)?;
            println!("engine plays {}", game.format_move(mv));
            game.make_move(&mut state, mv);
            history.push(state.clone());
            continue;
        }
        print!("> ");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if stdin
            .read_line(&mut line)
            .map_err(|e| format!("reading input: {e}"))?
            == 0
        {
            println!();
            break; // EOF
        }
        let input = line.trim();
        match input {
            "" => {}
            "quit" | "exit" | "q" => break,
            "?" => {
                let listed: Vec<String> = moves.iter().map(|&m| game.format_move(m)).collect();
                println!("legal: {}", listed.join(" "));
            }
            "hint" => {
                let mv = fc_engine_move(&game, net.as_ref(), &mut state, nodes)?;
                println!("hint: {}", game.format_move(mv));
            }
            "undo" => {
                let take_back = if history.len() > 2 {
                    2
                } else {
                    history.len() - 1
                };
                if take_back == 0 {
                    println!("nothing to undo");
                } else {
                    history.truncate(history.len() - take_back);
                    state = history.last().expect("initial state remains").clone();
                }
            }
            text => {
                let wanted = text.to_ascii_lowercase().replace('=', "");
                let found = moves
                    .iter()
                    .copied()
                    .find(|&m| game.format_move(m).to_ascii_lowercase().replace('=', "") == wanted);
                match found {
                    Some(mv) => {
                        game.make_move(&mut state, mv);
                        history.push(state.clone());
                    }
                    None => println!("not a legal move here; type ? to list legal moves"),
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// lab fit (SHSD Stage C: structured evaluators on exact data)
// ---------------------------------------------------------------------------

/// Where the exact `(state, WDL)` stream comes from.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FitSource {
    /// A checksummed retrograde tablebase written by `lab solve`
    /// (fc-small scale; D029 format).
    Tablebase { path: PathBuf },
    /// Solve the instance in-process (fc-tiny scale).
    Solve { max_positions: usize },
}

/// Model family under fit. Both families consume identical train,
/// validation, and test states, so comparisons are paired.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FitModel {
    /// Structured linear WDL model over a named feature recipe.
    Structured {
        recipe: FcRecipe,
        steps: u64,
        batch: usize,
        lr: f64,
        l2: f64,
    },
    /// The raw sparse MLP baseline (§9.3) under training recipe v1.
    RawMlp { width: usize, steps: u64 },
}

impl FitModel {
    fn label(&self) -> String {
        match self {
            FitModel::Structured { recipe, .. } => recipe.label().to_string(),
            FitModel::RawMlp { width, .. } => format!("mlp-w{width}"),
        }
    }
}

/// Fully explicit configuration for `lab fit`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FitConfig {
    game: GameSpec,
    source: FitSource,
    model: FitModel,
    /// Train-set size: the `train_positions` bucket-0..7 states with the
    /// smallest `splitmix64(key ^ FIT_TRAIN_SALT)` — subsets nest across
    /// sizes, so every data-ladder rung trains on a superset of the
    /// previous one (§48.5).
    train_positions: usize,
    /// Cap per evaluation bucket (existing D029 thinning rule).
    eval_cap: usize,
    /// Probe: searched-decision states drawn from the test bucket.
    probe_states: usize,
    probe_nodes: Vec<u64>,
    seed: u64,
    threads: usize,
}

/// Salt of the nested train-subset selection (ledger: engineering).
const FIT_TRAIN_SALT: u64 = 0x0f17_5a17_ab5e_ed01;

/// One exactly labelled state.
struct FitState {
    state: selfplay_lab::games::forward_chess::FcState,
    wdl: Wdl,
}

/// A probe state with its exact optimal-action set.
struct ProbePosition {
    state: selfplay_lab::games::forward_chess::FcState,
    optimal: Vec<u32>,
}

#[derive(Serialize)]
struct ProbeReport {
    states: usize,
    /// Argmax over one-ply child evaluations (no search).
    raw_optimal_rate: f64,
    /// Aligned with the configured probe budgets.
    searched_optimal_rate: Vec<f64>,
}

#[derive(Serialize)]
struct FitSplitMetrics {
    log_loss: f64,
    accuracy: f64,
}

fn fit(config_path: &Path) -> Result<(), String> {
    let config: FitConfig = read_config(config_path)?;
    let GameSpec::ForwardChess { ruleset } = &config.game else {
        return Err("lab fit currently supports forward_chess only".into());
    };
    if config.train_positions == 0 || config.eval_cap == 0 {
        return Err("train_positions and eval_cap must be positive".into());
    }
    let game = ForwardChess::new(*ruleset);
    let label = format!(
        "fit-{}-{}-n{}-s{}",
        config.game.label(),
        config.model.label(),
        config.train_positions,
        config.seed
    );
    let (run_dir, manifest) = start_run(&label, &config, config.seed, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();

    // -- Load the exact stream and select train/val/test states.
    let mut key_wdl: std::collections::HashMap<u64, Wdl> = std::collections::HashMap::new();
    let mut bucket_sizes = [0u64; 2];
    let visit_count = |game: &ForwardChess,
                       state: &selfplay_lab::games::forward_chess::FcState,
                       wdl: Wdl,
                       key_wdl: &mut std::collections::HashMap<u64, Wdl>,
                       bucket_sizes: &mut [u64; 2]| {
        let key = game.position_key(state);
        key_wdl.insert(key, wdl);
        if game.outcome(state).is_none() {
            match splitmix64(key) % 10 {
                8 => bucket_sizes[0] += 1,
                9 => bucket_sizes[1] += 1,
                _ => {}
            }
        }
    };

    // Selection state for the second pass.
    let mut train_heap: std::collections::BinaryHeap<(u64, u64)> =
        std::collections::BinaryHeap::new(); // (selection hash, key)
    let mut train_states: std::collections::HashMap<u64, FitState> =
        std::collections::HashMap::new();
    let mut val_states: Vec<FitState> = Vec::new();
    let mut test_states: Vec<FitState> = Vec::new();

    let select = |game: &ForwardChess,
                  state: selfplay_lab::games::forward_chess::FcState,
                  wdl: Wdl,
                  denominators: &[u64; 2],
                  train_heap: &mut std::collections::BinaryHeap<(u64, u64)>,
                  train_states: &mut std::collections::HashMap<u64, FitState>,
                  val_states: &mut Vec<FitState>,
                  test_states: &mut Vec<FitState>| {
        if game.outcome(&state).is_some() {
            return;
        }
        let key = game.position_key(&state);
        match splitmix64(key) % 10 {
            8 | 9 => {
                let slot = (splitmix64(key) % 10 - 8) as usize;
                if splitmix64(key ^ selfplay_lab::training::EVAL_THIN_SALT)
                    .is_multiple_of(denominators[slot])
                {
                    let bucket = if slot == 0 {
                        &mut *val_states
                    } else {
                        &mut *test_states
                    };
                    bucket.push(FitState { state, wdl });
                }
            }
            _ => {
                let hash = splitmix64(key ^ FIT_TRAIN_SALT);
                if train_heap.len() < config.train_positions {
                    train_heap.push((hash, key));
                    train_states.insert(key, FitState { state, wdl });
                } else if let Some(&(max_hash, max_key)) = train_heap.peek() {
                    if hash < max_hash {
                        train_heap.pop();
                        train_states.remove(&max_key);
                        train_heap.push((hash, key));
                        train_states.insert(key, FitState { state, wdl });
                    }
                }
            }
        }
    };

    match &config.source {
        FitSource::Tablebase { path } => {
            let bytes =
                std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
            say(
                format!(
                    "tablebase: {} ({} MB)",
                    path.display(),
                    bytes.len() / 1_000_000
                ),
                &mut log,
            );
            let mut cursor = std::io::Cursor::new(&bytes);
            read_tablebase_with(&game, &mut cursor, |_, state, wdl| {
                visit_count(&game, &state, wdl, &mut key_wdl, &mut bucket_sizes);
                Ok(())
            })?;
            let denominators =
                bucket_sizes.map(|n| n.div_ceil(config.eval_cap.max(1) as u64).max(1));
            let mut cursor = std::io::Cursor::new(&bytes);
            read_tablebase_with(&game, &mut cursor, |_, state, wdl| {
                select(
                    &game,
                    state,
                    wdl,
                    &denominators,
                    &mut train_heap,
                    &mut train_states,
                    &mut val_states,
                    &mut test_states,
                );
                Ok(())
            })?;
        }
        FitSource::Solve { max_positions } => {
            say(
                "solving in-process by retrograde analysis".to_string(),
                &mut log,
            );
            let solution = solve_retrograde(&game, *max_positions)?;
            for (state, &wdl) in solution.states.iter().zip(&solution.values) {
                visit_count(&game, state, wdl, &mut key_wdl, &mut bucket_sizes);
            }
            let denominators =
                bucket_sizes.map(|n| n.div_ceil(config.eval_cap.max(1) as u64).max(1));
            for (state, &wdl) in solution.states.iter().zip(&solution.values) {
                select(
                    &game,
                    state.clone(),
                    wdl,
                    &denominators,
                    &mut train_heap,
                    &mut train_states,
                    &mut val_states,
                    &mut test_states,
                );
            }
        }
    }
    let mut train: Vec<FitState> = train_states.into_values().collect();
    // Deterministic order regardless of hash-map iteration.
    train.sort_unstable_by_key(|s| game.position_key(&s.state));
    val_states.sort_unstable_by_key(|s| game.position_key(&s.state));
    test_states.sort_unstable_by_key(|s| game.position_key(&s.state));
    say(
        format!(
            "selected {} train / {} val / {} test states ({} positions mapped)",
            train.len(),
            val_states.len(),
            test_states.len(),
            key_wdl.len()
        ),
        &mut log,
    );
    if train.is_empty() || val_states.is_empty() || test_states.is_empty() {
        return Err("empty split; lower eval_cap or check the source".into());
    }

    // -- Probe positions: test-bucket states with exact optimal sets.
    let child_wdl = |state: &selfplay_lab::games::forward_chess::FcState,
                     mv: selfplay_lab::games::forward_chess::FcMove|
     -> Result<Wdl, String> {
        let mut child = state.clone();
        game.make_move(&mut child, mv);
        if let Some(outcome) = game.outcome(&child) {
            return Ok(Wdl::from_outcome(outcome, game.side_to_move(&child)));
        }
        key_wdl
            .get(&game.position_key(&child))
            .copied()
            .ok_or_else(|| "child position missing from the exact map".to_string())
    };
    let mut probes: Vec<ProbePosition> = Vec::new();
    let mut moves = Vec::new();
    for fit_state in test_states.iter().take(config.probe_states) {
        game.legal_moves(&fit_state.state, &mut moves);
        let mut optimal = Vec::new();
        for &mv in &moves {
            if child_wdl(&fit_state.state, mv)?.flip() == fit_state.wdl {
                optimal.push(game.action_id(&fit_state.state, mv));
            }
        }
        if optimal.is_empty() {
            return Err("exact optimal set is empty; map or rules inconsistency".into());
        }
        probes.push(ProbePosition {
            state: fit_state.state.clone(),
            optimal,
        });
    }
    say(format!("probe positions: {}", probes.len()), &mut log);

    // -- Fit the configured model and evaluate.
    let (val_metrics, test_metrics, prior, probe_report, extraction_ns) = match &config.model {
        FitModel::Structured {
            recipe,
            steps,
            batch,
            lr,
            l2,
        } => {
            use selfplay_lab::structured_eval::{
                class_prior_baseline, evaluate_linear_wdl, fit_linear_wdl, FitHyper,
                StructuredEvaluator, StructuredRow,
            };
            let dimension = FcExtractor::new(&game, *recipe).dimension();
            let extract_rows = |states: &[FitState]| -> (Vec<StructuredRow>, f64) {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(config.threads)
                    .build()
                    .expect("rayon pool");
                let t0 = Instant::now();
                let rows = pool.install(|| {
                    use rayon::prelude::*;
                    states
                        .par_iter()
                        .map_init(
                            || FcExtractor::new(&game, *recipe),
                            |extractor, fit_state| {
                                let mut x = Vec::new();
                                extractor.extract(&game, &fit_state.state, &mut x);
                                StructuredRow {
                                    x,
                                    wdl: fit_state.wdl,
                                }
                            },
                        )
                        .collect()
                });
                let ns = t0.elapsed().as_nanos() as f64 / states.len().max(1) as f64;
                (rows, ns)
            };
            let (train_rows, extraction_ns) = extract_rows(&train);
            let (val_rows, _) = extract_rows(&val_states);
            let (test_rows, _) = extract_rows(&test_states);
            say(
                format!(
                    "extracted {} features/position mean, {:.0} ns/position, dimension {}",
                    train_rows.iter().map(|r| r.x.len()).sum::<usize>() as f64
                        / train_rows.len() as f64,
                    extraction_ns,
                    dimension
                ),
                &mut log,
            );
            let hyper = FitHyper {
                steps: *steps,
                batch: *batch,
                lr: *lr,
                l2: *l2,
            };
            let mut metric_rows: Vec<serde_json::Value> = Vec::new();
            let model = {
                let val_rows_ref = &val_rows;
                fit_linear_wdl(
                    recipe.label(),
                    dimension,
                    &train_rows,
                    hyper,
                    config.seed,
                    |step| {
                        if step.step % 500 == 0 || step.step == hyper.steps {
                            metric_rows.push(serde_json::json!({
                                "step": step.step,
                                "train_loss": step.train_loss,
                            }));
                        }
                        let _ = val_rows_ref;
                    },
                )
            };
            run_dir
                .append_jsonl("metrics.jsonl", &metric_rows)
                .map_err(|e| e.to_string())?;
            let (val_loss, val_acc) = evaluate_linear_wdl(&model, &val_rows);
            let (test_loss, test_acc) = evaluate_linear_wdl(&model, &test_rows);
            let prior = class_prior_baseline(&train_rows, &test_rows);
            run_dir
                .create_subdir("checkpoint")
                .map_err(|e| e.to_string())?;
            run_dir
                .write_json("checkpoint/structured.json", &model)
                .map_err(|e| e.to_string())?;
            // Weight inspection for the report (§52.5): top |weight|
            // features by class margin.
            let mut inspect: Vec<(String, [f32; 3])> = Vec::new();
            let extractor = FcExtractor::new(&game, *recipe);
            let mut indices: Vec<usize> = (0..dimension).collect();
            indices.sort_by(|&a, &b| {
                let ma = (model.weights[a][2] - model.weights[a][0]).abs();
                let mb = (model.weights[b][2] - model.weights[b][0]).abs();
                mb.total_cmp(&ma)
            });
            for &i in indices.iter().take(25) {
                inspect.push((extractor.feature_name(i as u32), model.weights[i]));
            }
            run_dir
                .write_json("checkpoint/weight_inspection.json", &inspect)
                .map_err(|e| e.to_string())?;
            let probe_report =
                probe_decisions(&game, &probes, &config.probe_nodes, config.threads, || {
                    StructuredEvaluator::new(&model, FcExtractor::new(&game, *recipe))
                });
            (
                FitSplitMetrics {
                    log_loss: val_loss,
                    accuracy: val_acc,
                },
                FitSplitMetrics {
                    log_loss: test_loss,
                    accuracy: test_acc,
                },
                prior,
                probe_report,
                extraction_ns,
            )
        }
        FitModel::RawMlp { width, steps } => {
            // Recipe v1 on the identical states: raw sparse features,
            // WDL target, uniform-over-optimal policy target.
            let build_rows = |states: &[FitState]| -> Result<Vec<TrainRow>, String> {
                let mut rows = Vec::with_capacity(states.len());
                let mut features = Vec::new();
                let mut moves = Vec::new();
                for fit_state in states {
                    game.encode_features(&fit_state.state, &mut features);
                    game.legal_moves(&fit_state.state, &mut moves);
                    let mut legal = Vec::with_capacity(moves.len());
                    let mut policy_actions = Vec::new();
                    for &mv in &moves {
                        let action = game.action_id(&fit_state.state, mv);
                        legal.push(action);
                        if child_wdl(&fit_state.state, mv)?.flip() == fit_state.wdl {
                            policy_actions.push(action);
                        }
                    }
                    rows.push(TrainRow {
                        features: features.clone(),
                        legal,
                        wdl: fit_state.wdl,
                        policy_actions,
                    });
                }
                Ok(rows)
            };
            let t0 = Instant::now();
            let train_rows = build_rows(&train)?;
            let extraction_ns = t0.elapsed().as_nanos() as f64 / train.len().max(1) as f64;
            let val_rows = build_rows(&val_states)?;
            let test_rows = build_rows(&test_states)?;
            let max_features = train_rows
                .iter()
                .chain(&val_rows)
                .chain(&test_rows)
                .map(|r| r.features.len())
                .max()
                .unwrap_or(1);
            let dims = ModelDims {
                feature_count: game.feature_count(),
                action_count: game.action_count(),
                width: *width,
            };
            let mut metric_rows: Vec<serde_json::Value> = Vec::new();
            let net = train_supervised(
                dims,
                &train_rows,
                max_features,
                config.seed,
                *steps,
                |m, _| {
                    if m.step % 500 == 0 || m.step == *steps {
                        metric_rows.push(serde_json::json!({
                            "step": m.step,
                            "wdl_loss": m.wdl_loss,
                            "policy_loss": m.policy_loss,
                        }));
                    }
                },
            );
            run_dir
                .append_jsonl("metrics.jsonl", &metric_rows)
                .map_err(|e| e.to_string())?;
            let inference = net.valid();
            let compiled = CompiledNet::from_net(&inference, dims);
            let eval_rows = |rows: &[TrainRow]| -> FitSplitMetrics {
                let mut loss = 0.0;
                let mut correct = 0u64;
                let mut h2 = Vec::new();
                let mut wdl_logits = [0.0f32; 3];
                for row in rows {
                    compiled.forward_hidden(&row.features, &mut h2);
                    compiled.wdl_head(&h2, &mut wdl_logits);
                    let m = wdl_logits[0].max(wdl_logits[1]).max(wdl_logits[2]);
                    let e = [
                        f64::from(wdl_logits[0] - m).exp(),
                        f64::from(wdl_logits[1] - m).exp(),
                        f64::from(wdl_logits[2] - m).exp(),
                    ];
                    let s = e[0] + e[1] + e[2];
                    let p = [e[0] / s, e[1] / s, e[2] / s];
                    loss -= p[row.wdl as usize].max(1e-300).ln();
                    let argmax = (0..3)
                        .max_by(|&a, &b| p[a].total_cmp(&p[b]))
                        .expect("3 classes");
                    correct += u64::from(argmax == row.wdl as usize);
                }
                FitSplitMetrics {
                    log_loss: loss / rows.len() as f64,
                    accuracy: correct as f64 / rows.len() as f64,
                }
            };
            let val_metrics = eval_rows(&val_rows);
            let test_metrics = eval_rows(&test_rows);
            // Class-prior floor on the same test rows.
            let mut counts = [0u64; 3];
            for row in &train_rows {
                counts[row.wdl as usize] += 1;
            }
            let total: u64 = counts.iter().sum();
            let p: Vec<f64> = counts
                .iter()
                .map(|&c| (c as f64 / total as f64).max(1e-12))
                .collect();
            let argmax = (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).expect("3");
            let mut prior_loss = 0.0;
            let mut prior_correct = 0u64;
            for row in &test_rows {
                prior_loss -= p[row.wdl as usize].ln();
                prior_correct += u64::from(argmax == row.wdl as usize);
            }
            let prior = (
                prior_loss / test_rows.len() as f64,
                prior_correct as f64 / test_rows.len() as f64,
            );
            run_dir
                .create_subdir("checkpoint")
                .map_err(|e| e.to_string())?;
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            inference
                .clone()
                .save_file(run_dir.path().join("checkpoint/model"), &recorder)
                .map_err(|e| format!("saving checkpoint: {e}"))?;
            run_dir
                .write_json(
                    "checkpoint/model.json",
                    &serde_json::json!({
                        "dims": dims,
                        "max_features": max_features,
                        "seed": config.seed,
                        "training_steps": steps,
                        "train_positions": train.len(),
                    }),
                )
                .map_err(|e| e.to_string())?;
            let probe_report =
                probe_decisions(&game, &probes, &config.probe_nodes, config.threads, || {
                    ModelEvaluator::new(&compiled)
                });
            (
                val_metrics,
                test_metrics,
                prior,
                probe_report,
                extraction_ns,
            )
        }
    };

    // Search-only baseline row on the same probes (§9.2).
    let zero_probe = probe_decisions(&game, &probes, &config.probe_nodes, config.threads, || {
        ZeroEvaluator
    });

    say(
        format!(
            "val: logloss {:.4} acc {:.4} | test: logloss {:.4} acc {:.4} (prior: {:.4}/{:.4})",
            val_metrics.log_loss,
            val_metrics.accuracy,
            test_metrics.log_loss,
            test_metrics.accuracy,
            prior.0,
            prior.1
        ),
        &mut log,
    );
    say(
        format!(
            "probe raw optimal {:.4}; searched {:?} -> {:?}; zero-eval searched {:?}",
            probe_report.raw_optimal_rate,
            config.probe_nodes,
            probe_report.searched_optimal_rate,
            zero_probe.searched_optimal_rate
        ),
        &mut log,
    );

    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    run_dir
        .write_json(
            "summary.json",
            &serde_json::json!({
                "model": config.model.label(),
                "train_states": train.len(),
                "val_states": val_states.len(),
                "test_states": test_states.len(),
                "extraction_ns_per_position": extraction_ns,
                "class_prior": { "log_loss": prior.0, "accuracy": prior.1 },
                "val": val_metrics,
                "test": test_metrics,
                "probe": probe_report,
                "probe_zero_evaluator": zero_probe,
                "cost": {
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "utilization": cpu_seconds / (wall_seconds * config.threads as f64),
                    "peak_rss_bytes": peak_rss_bytes(),
                },
            }),
        )
        .map_err(|e| e.to_string())?;
    finish_run(&run_dir, manifest, &log)
}

/// Decision quality of an evaluator on exact probe positions: raw
/// one-ply argmax and full searches at each budget, scored against the
/// exact optimal-action sets.
fn probe_decisions<E, F>(
    game: &ForwardChess,
    probes: &[ProbePosition],
    budgets: &[u64],
    threads: usize,
    make_eval: F,
) -> ProbeReport
where
    E: Evaluator<ForwardChess>,
    F: Fn() -> E + Sync,
{
    use selfplay_lab::training::SELFPLAY_TT_LOG2;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let results: Vec<(bool, Vec<bool>)> = pool.install(|| {
        use rayon::prelude::*;
        probes
            .par_iter()
            .map(|probe| {
                let mut evaluator = make_eval();
                let mut state = probe.state.clone();
                let mut moves = Vec::new();
                game.legal_moves(&state, &mut moves);
                // Raw: pick the child whose evaluation is worst for the
                // opponent (ties -> first in stable action-ID order).
                let mut best_action = None;
                let mut best_score = i32::MIN;
                for &mv in &moves {
                    let undo = game.make_move(&mut state, mv);
                    let score = match game.outcome(&state) {
                        Some(outcome) => {
                            match Wdl::from_outcome(outcome, game.side_to_move(&state)) {
                                Wdl::Loss => selfplay_lab::search::SCORE_WIN,
                                Wdl::Draw => 0,
                                Wdl::Win => -selfplay_lab::search::SCORE_WIN,
                            }
                        }
                        None => -evaluator.leaf_value(game, &state),
                    };
                    game.unmake_move(&mut state, mv, undo);
                    if score > best_score {
                        best_score = score;
                        best_action = Some(game.action_id(&state, mv));
                    }
                }
                let raw_ok = best_action.is_some_and(|a| probe.optimal.contains(&a));
                let searched_ok = budgets
                    .iter()
                    .map(|&budget| {
                        let mut searcher: Searcher<ForwardChess> =
                            Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
                        let result = searcher.search(game, &mut state, 512, budget, &mut evaluator);
                        result
                            .best_move
                            .is_some_and(|mv| probe.optimal.contains(&game.action_id(&state, mv)))
                    })
                    .collect();
                (raw_ok, searched_ok)
            })
            .collect()
    });
    let n = probes.len().max(1) as f64;
    ProbeReport {
        states: probes.len(),
        raw_optimal_rate: results.iter().filter(|(raw, _)| *raw).count() as f64 / n,
        searched_optimal_rate: (0..budgets.len())
            .map(|i| results.iter().filter(|(_, s)| s[i]).count() as f64 / n)
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// lab relabel (SHSD Stage B instrumentation)
// ---------------------------------------------------------------------------

/// Counterfactual child labelling policy (SHSD §19.5).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ChildPolicy {
    None,
    All,
}

/// Exact oracle joined onto the records (SHSD §9.5).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum OracleSpec {
    /// No oracle join.
    None {},
    /// Solve the configured game in-process by retrograde analysis and
    /// join by position key (repetition-as-draw caveat, D029).
    Retrograde { max_positions: usize },
}

/// Fully explicit configuration for `lab relabel`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelabelConfig {
    game: GameSpec,
    /// `"zero"` or a checkpoint directory. The evaluator drives
    /// trajectory play and every labelling search, and is recorded in
    /// each record (teacher versioning, SHSD §20.5).
    evaluator: String,
    trajectories: TrajectorySpec,
    /// Keep a position iff `splitmix64(key ^ salt) % sample_one_in == 0`
    /// (1 = keep every visited position).
    sample_one_in: u64,
    /// Cap on distinct labelled positions.
    max_positions: usize,
    /// Shadow budgets compared against the deep label (may be empty).
    shallow_nodes: Vec<u64>,
    deep_nodes: u64,
    children: ChildPolicy,
    child_nodes: u64,
    oracle: OracleSpec,
    seed: u64,
    threads: usize,
}

fn relabel(config_path: &Path) -> Result<(), String> {
    let config: RelabelConfig = read_config(config_path)?;
    if config.deep_nodes == 0 || config.sample_one_in == 0 || config.max_positions == 0 {
        return Err("deep_nodes, sample_one_in, and max_positions must be positive".into());
    }
    if matches!(config.game, GameSpec::Chess {}) && !matches!(config.oracle, OracleSpec::None {}) {
        return Err("chess cannot be exactly solved; use oracle kind = \"none\"".into());
    }
    let label = format!("relabel-{}", config.game.label());
    let (run_dir, manifest) = start_run(&label, &config, config.seed, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    dispatch_game!(&config.game, game, {
        if config.evaluator == "zero" {
            say(
                "evaluator: zero (search-only baseline)".to_string(),
                &mut log,
            );
            relabel_run(&game, &config, &run_dir, &mut log, || ZeroEvaluator)?;
        } else {
            let dir = PathBuf::from(&config.evaluator);
            let (net, dims, _) = load_checkpoint(&dir)?;
            if dims.feature_count != game.feature_count()
                || dims.action_count != game.action_count()
            {
                return Err(format!(
                    "checkpoint dims {dims:?} do not fit {} ({} features / {} actions)",
                    config.game.label(),
                    game.feature_count(),
                    game.action_count()
                ));
            }
            say(
                format!("evaluator: {} (w{})", dir.display(), dims.width),
                &mut log,
            );
            let compiled = CompiledNet::from_net(&net, dims);
            relabel_run(&game, &config, &run_dir, &mut log, || {
                ModelEvaluator::new(&compiled)
            })?;
        }
    });
    finish_run(&run_dir, manifest, &log)
}

/// Cost accounting for one relabelling run (SHSD §47.8).
#[derive(Serialize)]
struct RelabelCost {
    wall_seconds: f64,
    cpu_seconds: f64,
    utilization: f64,
    positions_per_second: f64,
    total_search_nodes: u64,
    peak_rss_bytes: Option<u64>,
}

#[derive(Serialize)]
struct RelabelOutput<'a> {
    summary: &'a RelabelSummary,
    oracle_join: Option<OracleJoinStats>,
    cost: RelabelCost,
}

fn relabel_run<G, E, F>(
    game: &G,
    config: &RelabelConfig,
    run_dir: &RunDir,
    log: &mut String,
    make_eval: F,
) -> Result<(), String>
where
    G: Game,
    E: Evaluator<G>,
    F: Fn() -> E + Sync,
{
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();

    let samples = collect_positions(
        game,
        &config.trajectories,
        config.sample_one_in,
        config.max_positions,
        config.seed,
        config.threads,
        &make_eval,
    );
    let total_weight: u64 = samples.iter().map(|s| u64::from(s.weight)).sum();
    say(
        format!(
            "sampled {} distinct positions ({} visits) from {} trajectories",
            samples.len(),
            total_weight,
            config.trajectories.games()
        ),
        log,
    );
    if samples.is_empty() {
        return Err("no positions sampled; lower sample_one_in or add trajectories".into());
    }

    let mut records = label_positions(
        game,
        &samples,
        &config.evaluator,
        &config.shallow_nodes,
        config.deep_nodes,
        config.children == ChildPolicy::All,
        config.child_nodes,
        config.threads,
        &make_eval,
    );
    say(format!("labelled {} positions", records.len()), log);

    let oracle_join = match &config.oracle {
        OracleSpec::None {} => None,
        OracleSpec::Retrograde { max_positions } => {
            say(
                "solving the exact oracle by retrograde analysis".to_string(),
                log,
            );
            let solution = solve_retrograde(game, *max_positions)?;
            let stats = join_retrograde_oracle(game, &mut records, &solution);
            say(
                format!(
                    "oracle join: {} joined, {} missed of {} positions",
                    stats.joined,
                    stats.missed,
                    records.len()
                ),
                log,
            );
            Some(stats)
        }
    };

    let summary = summarize(&records, &config.shallow_nodes);
    run_dir
        .append_jsonl("records.jsonl", &records)
        .map_err(|e| format!("writing records.jsonl: {e}"))?;

    let wall_seconds = started.elapsed().as_secs_f64();
    let cpu_seconds = process_cpu_seconds().unwrap_or(0.0) - cpu_before;
    let total_search_nodes: u64 = records
        .iter()
        .map(|r| {
            r.shallow.iter().map(|l| l.nodes).sum::<u64>()
                + r.deep.nodes
                + r.children.iter().map(|c| c.nodes).sum::<u64>()
        })
        .sum();
    let cost = RelabelCost {
        wall_seconds,
        cpu_seconds,
        utilization: cpu_seconds / (wall_seconds * config.threads as f64),
        positions_per_second: records.len() as f64 / wall_seconds,
        total_search_nodes,
        peak_rss_bytes: peak_rss_bytes(),
    };
    say(
        format!(
            "cost: {:.1}s wall, {:.1}s cpu (utilization {:.2}), {:.1} positions/s, {} nodes",
            cost.wall_seconds,
            cost.cpu_seconds,
            cost.utilization,
            cost.positions_per_second,
            cost.total_search_nodes
        ),
        log,
    );
    for line in relabel_headline(&summary) {
        say(line, log);
    }
    run_dir
        .write_json(
            "summary.json",
            &RelabelOutput {
                summary: &summary,
                oracle_join,
                cost,
            },
        )
        .map_err(|e| format!("writing summary.json: {e}"))?;
    Ok(())
}

/// Human-readable headline of a relabel summary for the run log.
fn relabel_headline(summary: &RelabelSummary) -> Vec<String> {
    let mut lines = Vec::new();
    for budget in &summary.shallow {
        lines.push(format!(
            "shallow {:>6} nodes: best-move agreement with deep {:.3}, mean |value residual| {:.1} ({} heuristic pairs)",
            budget.node_budget,
            budget.best_move_agree_rate,
            budget.mean_abs_value_residual,
            budget.heuristic_pairs
        ));
    }
    lines.push(format!(
        "deep stability: last-iteration stable {:.3}, mean best-move changes {:.2}",
        summary.deep_last_iteration_stable, summary.deep_mean_best_move_changes
    ));
    lines.push(format!(
        "ordering: deep-best mean rank {:.2}, top-1 {:.3}, top-3 {:.3}",
        summary.order_rank_mean, summary.order_top1_rate, summary.order_top3_rate
    ));
    if let Some(oracle) = &summary.oracle {
        lines.push(format!(
            "oracle: {} joined (W/D/L {}/{}/{}), deep optimal-decision rate {:.4}",
            oracle.joined,
            oracle.wdl_counts[0],
            oracle.wdl_counts[1],
            oracle.wdl_counts[2],
            oracle.deep_optimal_rate
        ));
        for (i, rate) in oracle.shallow_optimal_rate.iter().enumerate() {
            lines.push(format!(
                "oracle: shallow[{i}] optimal-decision rate {rate:.4}"
            ));
        }
        if let Some(rate) = oracle.child_top_optimal_rate {
            lines.push(format!("oracle: child-argmax optimal rate {rate:.4}"));
        }
    }
    lines
}

/// Load a checkpoint directory (`model.bin` + `model.json`) into an
/// inference network.
fn load_checkpoint(dir: &Path) -> Result<(PolicyValueNet<InferBackend>, ModelDims, usize), String> {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("model.json"))
            .map_err(|e| format!("reading {}/model.json: {e}", dir.display()))?,
    )
    .map_err(|e| format!("parsing model.json: {e}"))?;
    let dims: ModelDims = serde_json::from_value(meta["dims"].clone())
        .map_err(|e| format!("model.json dims: {e}"))?;
    let max_features = meta["max_features"]
        .as_u64()
        .ok_or("model.json missing max_features")? as usize;
    let device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let net = PolicyValueNet::<InferBackend>::new(dims, &device)
        .load_file(dir.join("model"), &recorder, &device)
        .map_err(|e| format!("loading checkpoint: {e}"))?;
    Ok((net, dims, max_features))
}

fn probe(config: ProbeConfig) -> Result<(), String> {
    if matches!(config.game, GameSpec::Chess {}) {
        return Err("oracle probes need an exactly solvable game; use match_probe".into());
    }
    if matches!(config.game, GameSpec::ForwardChess { .. }) {
        return Err(
            "oracle probes use the acyclic exact solver; forward chess is loopy \
             (use selfplay oracle promotion or match_probe)"
                .into(),
        );
    }
    if config.node_budgets.is_empty() {
        return Err("node_budgets must not be empty".into());
    }
    let label = format!("probe-{}", config.game.label());
    let (run_dir, manifest) = start_run(&label, &config, 0, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    let (net, dims, max_features) = load_checkpoint(&config.checkpoint)?;
    let mut manifest = manifest;
    manifest.model_parameter_count = net.num_params() as u64;

    dispatch_game!(&config.game, game, {
        run_probe(
            &game,
            &net,
            dims,
            max_features,
            &config,
            &run_dir,
            &manifest,
            &mut log,
        )?;
    });
    finish_run(&run_dir, manifest, &log)
}

#[allow(clippy::too_many_arguments)] // probe context: checkpoint + config + run plumbing
fn run_probe<G: Game>(
    game: &G,
    net: &PolicyValueNet<InferBackend>,
    dims: ModelDims,
    max_features: usize,
    config: &ProbeConfig,
    run_dir: &RunDir,
    manifest: &Manifest,
    log: &mut String,
) -> Result<(), String> {
    {
        {
            let started = Instant::now();
            let dataset = build_exact_dataset(game);
            let raw_val = evaluate_model_oracle(net, &dataset.val, dims, max_features);
            let raw_test = evaluate_model_oracle(net, &dataset.test, dims, max_features);
            say(
                format!(
                    "raw: val regret {:.4}, test regret {:.4}, test action acc {:.4}",
                    raw_val.mean_regret_levels,
                    raw_test.mean_regret_levels,
                    raw_test.action_accuracy
                ),
                log,
            );
            let compiled = CompiledNet::from_net(net, dims);
            let mut budget_reports = Vec::new();
            for &budget in &config.node_budgets {
                let searched = searched_decision_metrics(
                    game,
                    &compiled,
                    CorpusSplit::Test,
                    config.searched_sample,
                    budget,
                    config.threads,
                );
                let exploit = exploitability_vs_perfect(game, &compiled, budget, config.threads);
                say(
                    format!(
                        "searched@{budget}: acc {:.4}, regret {:.4}, depth {:.1}; \
                         exploit drops {}/{} (mean levels {:.3})",
                        searched.action_accuracy,
                        searched.mean_regret_levels,
                        searched.mean_completed_depth,
                        exploit.avoidable_drops,
                        exploit.games,
                        exploit.mean_levels_lost,
                    ),
                    log,
                );
                budget_reports.push(serde_json::json!({
                    "node_budget": budget,
                    "searched_test": searched,
                    "exploitability": exploit,
                }));
            }
            let disagreement = search_disagreement_analysis(
                game,
                &compiled,
                CorpusSplit::Test,
                config.searched_sample.min(1000),
                &config.node_budgets,
                config.threads,
            );
            for report in &disagreement {
                say(
                    format!(
                        "disagreement@{} vs deepest: {:.3} (deeper fixes {}, breaks {}, \
                         neutral {})",
                        report.node_budget,
                        report.disagreement_with_deepest,
                        report.deeper_fixes,
                        report.deeper_breaks,
                        report.neutral,
                    ),
                    log,
                );
            }
            run_dir
                .write_json(
                    "summary.json",
                    &serde_json::json!({
                        "config": config,
                        "parameter_count": manifest.model_parameter_count,
                        "raw_val": raw_val,
                        "raw_test": raw_test,
                        "budgets": budget_reports,
                        "disagreement": disagreement,
                        "wall_seconds": started.elapsed().as_secs_f64(),
                        "peak_rss_bytes": peak_rss_bytes().unwrap_or(0),
                    }),
                )
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// lab teacher (plan §26: Stockfish diagnostic ceiling — teacher-assisted,
// never the tabula-rasa champion)
// ---------------------------------------------------------------------------

/// Fully explicit configuration for the Stockfish diagnostic.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TeacherConfig {
    seed: u64,
    model_width: usize,
    /// Number of labelled positions (90/10 train/held-out split).
    positions: usize,
    /// Fixed Stockfish search depth used for every label.
    stockfish_depth: u32,
    /// Path to the Stockfish binary.
    stockfish_path: PathBuf,
    training_steps: u64,
    threads: usize,
}

/// One Stockfish-labelled position.
struct TeacherRow {
    features: Vec<u32>,
    legal: Vec<u32>,
    wdl: Wdl,
    best_action: u32,
}

/// Drive one Stockfish process over a batch of FENs at a fixed depth.
fn stockfish_label(
    stockfish: &Path,
    depth: u32,
    fens: &[String],
) -> Result<Vec<(i32, String)>, String> {
    use std::io::{BufRead, BufReader, Write};
    let mut child = std::process::Command::new(stockfish)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("spawning {}: {e}", stockfish.display()))?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(stdout);
    let mut labels = Vec::with_capacity(fens.len());
    writeln!(stdin, "uci").map_err(|e| e.to_string())?;
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if line.starts_with("uciok") {
            break;
        }
    }
    for fen in fens {
        writeln!(stdin, "position fen {fen}").map_err(|e| e.to_string())?;
        writeln!(stdin, "go depth {depth}").map_err(|e| e.to_string())?;
        let mut score_cp = 0i32;
        let best = loop {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("stockfish exited early".into());
            }
            if line.starts_with("info") {
                let words: Vec<&str> = line.split_whitespace().collect();
                if let Some(i) = words.iter().position(|&w| w == "score") {
                    match (words.get(i + 1), words.get(i + 2)) {
                        (Some(&"cp"), Some(v)) => score_cp = v.parse().unwrap_or(0),
                        (Some(&"mate"), Some(v)) => {
                            let mate: i32 = v.parse().unwrap_or(0);
                            score_cp = if mate > 0 { 10_000 } else { -10_000 };
                        }
                        _ => {}
                    }
                }
            } else if let Some(rest) = line.strip_prefix("bestmove ") {
                break rest.split_whitespace().next().unwrap_or("").to_string();
            }
        };
        labels.push((score_cp, best));
    }
    writeln!(stdin, "quit").ok();
    child.wait().ok();
    Ok(labels)
}

fn teacher(config_path: &Path) -> Result<(), String> {
    let config: TeacherConfig = read_config(config_path)?;
    if config.positions < 100 {
        return Err("need at least 100 positions".into());
    }
    let (run_dir, mut manifest) = start_run("teacher-chess", &config, config.seed, config.threads)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    let started = Instant::now();
    let game = Chess::new();
    let dims = ModelDims {
        feature_count: game.feature_count(),
        action_count: game.action_count(),
        width: config.model_width,
    };

    // 1. Position corpus from seeded random legal trajectories, sampled
    // sparsely along each game to decorrelate.
    use rand::Rng as _;
    use rand::SeedableRng as _;
    use selfplay_lab::game::Game as _;
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(config.seed ^ 0x5f);
    let mut states = Vec::new();
    let mut fens = Vec::new();
    while states.len() < config.positions {
        let mut state = game.initial_state();
        let mut moves = Vec::new();
        loop {
            if game.outcome(&state).is_some() {
                break;
            }
            if rng.gen::<f64>() < 0.15 {
                states.push(state.clone());
                fens.push(format!("{}", game.board_of(&state)));
                if states.len() >= config.positions {
                    break;
                }
            }
            game.legal_moves(&state, &mut moves);
            let mv = moves[rng.gen_range(0..moves.len())];
            game.make_move(&mut state, mv);
        }
    }
    say(
        format!(
            "corpus: {} positions ({:.1}s)",
            states.len(),
            started.elapsed().as_secs_f64()
        ),
        &mut log,
    );

    // 2. Label with a fixed, documented Stockfish budget, in parallel.
    let label_started = Instant::now();
    let chunk = states.len().div_ceil(config.threads.max(1));
    let fen_chunks: Vec<&[String]> = fens.chunks(chunk).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .map_err(|e| e.to_string())?;
    let labels: Vec<(i32, String)> = pool.install(|| {
        use rayon::prelude::*;
        fen_chunks
            .par_iter()
            .map(|chunk| {
                stockfish_label(&config.stockfish_path, config.stockfish_depth, chunk)
                    .expect("stockfish labelling failed")
            })
            .collect::<Vec<_>>()
            .concat()
    });
    say(
        format!(
            "labelled at depth {} in {:.1}s ({:.1} pos/s)",
            config.stockfish_depth,
            label_started.elapsed().as_secs_f64(),
            labels.len() as f64 / label_started.elapsed().as_secs_f64()
        ),
        &mut log,
    );

    // 3. Rows: WDL from the documented cp rule (win > 100 cp, loss < -100),
    // policy target = Stockfish best move.
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    let mut max_features = 1usize;
    for (state, (cp, best)) in states.iter().zip(&labels) {
        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        let Some(best_move) = selfplay_lab::games::chess::parse_move_text(&game, state, best)
        else {
            skipped += 1;
            continue;
        };
        let mut features = Vec::new();
        game.encode_features(state, &mut features);
        max_features = max_features.max(features.len());
        rows.push(TeacherRow {
            features,
            legal: moves.iter().map(|&m| game.action_id(state, m)).collect(),
            wdl: if *cp > 100 {
                Wdl::Win
            } else if *cp < -100 {
                Wdl::Loss
            } else {
                Wdl::Draw
            },
            best_action: game.action_id(state, best_move),
        });
    }
    say(
        format!(
            "rows: {} usable, {skipped} skipped (unparsable bestmove)",
            rows.len()
        ),
        &mut log,
    );

    // 4. Train (recipe v1) on 90%, hold out 10% by position order hash.
    let train_rows: Vec<TrainRow> = rows
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 10 != 9)
        .map(|(_, r)| TrainRow {
            features: r.features.clone(),
            legal: r.legal.clone(),
            wdl: r.wdl,
            policy_actions: vec![r.best_action],
        })
        .collect();
    let held_out: Vec<&TeacherRow> = rows
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 10 == 9)
        .map(|(_, r)| r)
        .collect();
    let net = train_supervised(
        dims,
        &train_rows,
        max_features,
        config.seed,
        config.training_steps,
        |_, _| {},
    );
    manifest.model_parameter_count = net.num_params() as u64;

    // 5. Held-out agreement with the teacher (and chance baselines).
    let compiled = CompiledNet::from_net(&net.valid(), dims);
    let mut wdl_hits = 0usize;
    let mut move_hits = 0usize;
    let mut legal_sum = 0f64;
    let mut wdl_logits = [0.0f32; 3];
    let mut action_logits = Vec::new();
    for row in &held_out {
        compiled.forward(&row.features, &mut wdl_logits, &mut action_logits);
        let predicted = (0..3)
            .max_by(|&a, &b| wdl_logits[a].total_cmp(&wdl_logits[b]))
            .unwrap();
        wdl_hits += usize::from(predicted == row.wdl as usize);
        let best_legal = row
            .legal
            .iter()
            .max_by(|&&a, &&b| action_logits[a as usize].total_cmp(&action_logits[b as usize]))
            .copied()
            .unwrap();
        move_hits += usize::from(best_legal == row.best_action);
        legal_sum += 1.0 / row.legal.len() as f64;
    }
    let n = held_out.len().max(1) as f64;
    let wdl_accuracy = wdl_hits as f64 / n;
    let move_agreement = move_hits as f64 / n;
    let chance_move = legal_sum / n;
    say(
        format!(
            "held-out ({} positions): teacher-WDL accuracy {:.4}, best-move agreement {:.4} \
             (chance {:.4})",
            held_out.len(),
            wdl_accuracy,
            move_agreement,
            chance_move,
        ),
        &mut log,
    );
    run_dir
        .write_json(
            "summary.json",
            &serde_json::json!({
                "config": config,
                "parameter_count": manifest.model_parameter_count,
                "rows": rows.len(),
                "held_out": held_out.len(),
                "teacher_wdl_accuracy": wdl_accuracy,
                "best_move_agreement": move_agreement,
                "chance_move_agreement": chance_move,
                "wall_seconds": started.elapsed().as_secs_f64(),
                "peak_rss_bytes": peak_rss_bytes().unwrap_or(0),
            }),
        )
        .map_err(|e| e.to_string())?;
    finish_run(&run_dir, manifest, &log)
}

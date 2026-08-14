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
use selfplay_lab::evaluation::{
    evaluate_model_oracle, exploitability_vs_perfect, play_paired_match,
    retrograde_searched_candidates, search_disagreement_analysis, searched_decision_metrics,
    searched_decision_metrics_on, CorpusSplit, OracleMetrics,
};
use selfplay_lab::evaluation::{run_random_arena, ArenaSummary};
use selfplay_lab::experiment::{
    collect_manifest, peak_rss_bytes, process_cpu_seconds, unix_seconds, Manifest, RunDir,
};
use selfplay_lab::game::Game;
use selfplay_lab::games::breakthrough::Breakthrough;
use selfplay_lab::games::chess::Chess;
use selfplay_lab::games::connect_k::ConnectK;
use selfplay_lab::games::forward_chess::{read_tablebase_with, write_tablebase, ForwardChess};
use selfplay_lab::games::othello::Othello;
use selfplay_lab::games::GameSpec;
use selfplay_lab::model::{CompiledNet, InferBackend, ModelDims, PolicyValueNet, TrainBackend};
use selfplay_lab::search::{
    enumerate_solved, exhaustive_negamax, solve_retrograde, ExactSolver, MoveOrdering, Searcher,
    Wdl, ZeroEvaluator,
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
        LabCommand::Sweep { manifest } => sweep(&manifest),
        LabCommand::Play {
            game,
            checkpoint,
            nodes,
            side,
        } => play(&game, checkpoint.as_deref(), nodes, &side),
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
        if denominator > 1 && splitmix64(key) % denominator != 0 {
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
    /// `lab` subcommand to run: "train", "solve", or "evaluate".
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
            "train" | "solve" | "evaluate" | "selfplay"
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
    /// Dims must match `model_width` and `game`. The replay buffer
    /// restarts empty each chunk (documented discontinuity); progression
    /// baselines are per-chunk.
    #[serde(default)]
    init_checkpoint: Option<PathBuf>,
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
        // A match-mode candidate that is not provably better but not
        // observably worse (score >= 0.5) keeps the champion without
        // counting toward the halt: only evidence of regression strikes.
        let regression = match &promotion_match {
            Some(result) => result.score < 0.5,
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

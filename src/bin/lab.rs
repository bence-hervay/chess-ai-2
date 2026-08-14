//! `lab` — the experiment command-line surface.
//!
//! Commands (each appears in the phase that requires it):
//! - `lab evaluate <config>`: paired evaluation arena (Phase 0);
//! - `lab solve <config>`: exact solving and search-correctness
//!   experiments (Phase 1).

use clap::{Parser, Subcommand};
use selfplay_lab::evaluation::{run_random_arena, ArenaSummary};
use selfplay_lab::experiment::{
    collect_manifest, peak_rss_bytes, process_cpu_seconds, unix_seconds, Manifest, RunDir,
};
use selfplay_lab::game::Game;
use selfplay_lab::games::connect_k::ConnectK;
use selfplay_lab::games::GameSpec;
use selfplay_lab::search::{
    enumerate_solved, exhaustive_negamax, ExactSolver, MoveOrdering, Searcher, Wdl,
};
use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
}

/// Fully explicit configuration for `lab evaluate`. Every field is required.
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
}

impl SolveMethod {
    fn label(self) -> &'static str {
        match self {
            SolveMethod::Exact => "exact",
            SolveMethod::Exhaustive => "exhaustive",
            SolveMethod::AlphaBeta => "ab",
            SolveMethod::AlphaBetaTt => "abtt",
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
    optimal: Vec<u32>,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        LabCommand::Evaluate { config } => evaluate(&config),
        LabCommand::Solve { config } => solve(&config),
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
    let config: EvaluationConfig = read_config(config_path)?;
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

    let (arena, metrics) = match &config.game {
        GameSpec::ConnectK {
            width,
            height,
            k,
            gravity,
        } => {
            let game = ConnectK::new(*width, *height, *k, *gravity)?;
            run_arena(&game, &config, &run_dir)?
        }
    };

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
    let label = format!("solve-{}-{}", config.method.label(), config.game.label());
    let (run_dir, manifest) = start_run(&label, &config, 0, 1)?;
    let mut log = String::new();
    say(
        format!("run directory: {}", run_dir.path().display()),
        &mut log,
    );
    say("thread plan: 1 solver thread".to_string(), &mut log);

    match &config.game {
        GameSpec::ConnectK {
            width,
            height,
            k,
            gravity,
        } => {
            let game = ConnectK::new(*width, *height, *k, *gravity)?;
            let max_depth = u32::from(*width) * u32::from(*height);
            run_solve(&game, max_depth, &config, &run_dir, &mut log)?;
        }
    }
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
                    optimal: position
                        .optimal
                        .iter()
                        .map(|&m| game.action_id(position.state, m))
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
            let result = searcher.search(game, &mut state, max_depth, u64::MAX, &|_, _| 0);
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

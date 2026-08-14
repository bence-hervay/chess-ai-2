//! `lab` — the experiment command-line surface.
//!
//! Commands (each appears in the phase that requires it):
//! - `lab evaluate <config>`: paired evaluation arena (Phase 0);
//! - `lab solve <config>`: exact solving and search-correctness
//!   experiments (Phase 1);
//! - `lab train <config>`: supervised training on exact corpora (Phase 2);
//! - `lab sweep <manifest>`: CPU-slot scheduling of many runs (Phase 2).

use burn::module::{AutodiffModule, Module as _};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use clap::{Parser, Subcommand};
use selfplay_lab::evaluation::{evaluate_model_oracle, OracleMetrics};
use selfplay_lab::evaluation::{run_random_arena, ArenaSummary};
use selfplay_lab::experiment::{
    collect_manifest, peak_rss_bytes, process_cpu_seconds, unix_seconds, Manifest, RunDir,
};
use selfplay_lab::game::Game;
use selfplay_lab::games::connect_k::ConnectK;
use selfplay_lab::games::GameSpec;
use selfplay_lab::model::{InferBackend, ModelDims, PolicyValueNet};
use selfplay_lab::search::{
    enumerate_solved, exhaustive_negamax, ExactSolver, MoveOrdering, Searcher, Wdl,
};
use selfplay_lab::training::{
    build_exact_dataset, make_batch, train_supervised, Example, BATCH_SIZE, EVAL_EVERY,
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
    /// Train the policy/value network on an exact corpus.
    Train {
        /// Path to a training configuration file.
        config: PathBuf,
    },
    /// Run a manifest of experiment configs with CPU-slot scheduling.
    Sweep {
        /// Path to a sweep manifest (JSONL: {"command","config","cores"}).
        manifest: PathBuf,
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
    child_wdl: Vec<Wdl>,
    optimal: Vec<u32>,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        LabCommand::Evaluate { config } => evaluate(&config),
        LabCommand::Solve { config } => solve(&config),
        LabCommand::Train { config } => train(&config),
        LabCommand::Sweep { manifest } => sweep(&manifest),
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

    match &config.game {
        GameSpec::ConnectK {
            width,
            height,
            k,
            gravity,
        } => {
            let game = ConnectK::new(*width, *height, *k, *gravity)?;
            let dims = ModelDims {
                feature_count: game.feature_count(),
                action_count: game.action_count(),
                width: config.model_width,
            };
            run_train(&game, dims, &config, &run_dir, &mut manifest, &mut log)?;
        }
    }
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
    let subset: Vec<Example> = {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut order: Vec<usize> = (0..dataset.train.len()).collect();
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(0x0D47_45E7);
        order.shuffle(&mut rng);
        order
            .into_iter()
            .take(config.train_positions)
            .map(|i| dataset.train[i].clone())
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
    let probe: Vec<&Example> = dataset.val.iter().take(64).collect();
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
    let train_metrics = evaluate_model_oracle(&restored, &subset, dims, dataset.max_features);
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
        if !matches!(entry.command.as_str(), "train" | "solve" | "evaluate") {
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

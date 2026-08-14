//! `lab` — the experiment command-line surface.
//!
//! Phase 0 exposes only `lab evaluate <config>`: a random-versus-random
//! paired arena with a self-contained run directory.

use clap::{Parser, Subcommand};
use selfplay_lab::evaluation::{run_random_arena, ArenaSummary};
use selfplay_lab::experiment::{
    collect_manifest, peak_rss_bytes, process_cpu_seconds, unix_seconds, RunDir,
};
use selfplay_lab::game::Game;
use selfplay_lab::games::connect_k::ConnectK;
use selfplay_lab::games::GameSpec;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

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

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        LabCommand::Evaluate { config } => evaluate(&config),
    };
    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

fn evaluate(config_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(config_path)
        .map_err(|e| format!("reading {}: {e}", config_path.display()))?;
    let config: EvaluationConfig =
        toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", config_path.display()))?;
    if config.pairs == 0 {
        return Err("pairs must be positive".into());
    }
    if config.threads == 0 {
        return Err("threads must be positive".into());
    }

    let mut manifest = collect_manifest(config.seed, config.threads);
    let run_dir = RunDir::create(
        Path::new("runs"),
        &config.game.label(),
        &manifest.git_commit,
    )
    .map_err(|e| format!("creating run directory: {e}"))?;
    run_dir
        .write_text(
            "resolved.toml",
            &toml::to_string_pretty(&config).expect("serializable"),
        )
        .map_err(|e| e.to_string())?;
    run_dir
        .write_json("manifest.json", &manifest)
        .map_err(|e| e.to_string())?;

    let mut log = String::new();
    let say = |line: String, log: &mut String| {
        println!("{line}");
        log.push_str(&line);
        log.push('\n');
    };

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

    let summary = match &config.game {
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

    let wall = summary.wall_seconds;
    say(
        format!(
            "{} games: P1 {} / draws {} / P2 {} (agent A points {:.1}), mean plies {:.1}",
            summary.arena.games,
            summary.arena.p1_wins,
            summary.arena.draws,
            summary.arena.p2_wins,
            summary.arena.agent_a_points,
            summary.arena.total_plies as f64 / summary.arena.games.max(1) as f64,
        ),
        &mut log,
    );
    say(
        format!(
            "throughput: {:.0} games/s, {:.0} moves/s; wall {:.2}s, cpu {:.2}s, \
             utilization {:.1}%, peak RSS {:.1} MiB",
            summary.metrics.games_per_second,
            summary.metrics.moves_per_second,
            wall,
            summary.metrics.cpu_seconds,
            100.0 * summary.metrics.utilization,
            summary.metrics.peak_rss_bytes as f64 / (1024.0 * 1024.0),
        ),
        &mut log,
    );

    run_dir
        .write_text("stdout.log", &log)
        .map_err(|e| e.to_string())?;
    manifest.end_unix_seconds = Some(unix_seconds());
    manifest.exit_status = "success".to_string();
    run_dir
        .write_json("manifest.json", &manifest)
        .map_err(|e| e.to_string())?;
    Ok(())
}

struct EvaluationResult {
    arena: ArenaSummary,
    metrics: RunMetrics,
    wall_seconds: f64,
}

fn run_arena<G: Game>(
    game: &G,
    config: &EvaluationConfig,
    run_dir: &RunDir,
) -> Result<EvaluationResult, String> {
    let cpu_before = process_cpu_seconds().unwrap_or(0.0);
    let started = Instant::now();
    let mut sink_error: Option<std::io::Error> = None;
    let arena = run_random_arena(game, config.pairs, config.seed, config.threads, |batch| {
        if sink_error.is_none() {
            if let Err(e) = run_dir.append_jsonl("games/games.jsonl", batch) {
                sink_error = Some(e);
            }
        }
    });
    if let Some(e) = sink_error {
        return Err(format!("writing game records: {e}"));
    }
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
    Ok(EvaluationResult {
        arena,
        metrics,
        wall_seconds,
    })
}

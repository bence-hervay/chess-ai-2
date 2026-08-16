//! UCI chess engine binary (plan §26).
//!
//! Usage: `uci [checkpoint-dir]` — the optional argument (also settable
//! via `setoption name Checkpoint value <dir>`) points at a saved
//! policy/value checkpoint (`model.bin` + `model.json`). Without one,
//! the engine searches with a zero evaluator (legal but planless play,
//! used by protocol compliance tests).
//!
//! Supported: `uci`, `isready`, `ucinewgame`, `position startpos|fen ...
//! [moves ...]`, `go [nodes N] [depth D] [movetime T] [wtime T btime T
//! [winc T] [binc T]] [infinite]`, `quit`. Unknown or malformed commands
//! are ignored (with a note on stderr), as UCI requires.
//!
//! UCI castling moves (`e1g1`) are translated to cozy-chess's
//! king-takes-rook representation (`e1h1`) and back.

use burn::module::Module as _;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use cozy_chess::{Color, Piece, Square};
use selfplay_lab::features::chess::ChessMoveFeatures;
use selfplay_lab::game::Game;
use selfplay_lab::games::chess::{Chess, ChessMove, ChessState};
use selfplay_lab::model::{CompiledNet, InferBackend, ModelDims, ModelEvaluator, PolicyValueNet};
use selfplay_lab::search::{MoveOrdering, Searcher, ZeroEvaluator};
use selfplay_lab::structured_eval::{MlpRankedEvaluator, MoveRanker};
use std::io::BufRead as _;
use std::path::Path;
use std::time::{Duration, Instant};

/// Transposition table size for interactive play (2^20 entries).
const UCI_TT_LOG2: u32 = 20;
/// Iterative-deepening cap; wall-clock or node budgets stop earlier.
const MAX_DEPTH: u32 = 128;

fn load_compiled(dir: &Path) -> Result<CompiledNet, String> {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("model.json"))
            .map_err(|e| format!("reading {}/model.json: {e}", dir.display()))?,
    )
    .map_err(|e| format!("parsing model.json: {e}"))?;
    let dims: ModelDims = serde_json::from_value(meta["dims"].clone())
        .map_err(|e| format!("model.json dims: {e}"))?;
    let device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let net = PolicyValueNet::<InferBackend>::new(dims, &device)
        .load_file(dir.join("model"), &recorder, &device)
        .map_err(|e| format!("loading checkpoint: {e}"))?;
    Ok(CompiledNet::from_net(&net, dims))
}

use selfplay_lab::games::chess::parse_move_text as parse_uci_move;

/// Render a wrapper move as a standard UCI string (castling back to
/// king-two-files notation).
fn format_uci_move(state: &ChessState, mv: ChessMove) -> String {
    let stm_is_white = state_side(state) == Color::White;
    let castle = match (mv.0.from, mv.0.to, stm_is_white) {
        (Square::E1, Square::H1, true) => Some("e1g1"),
        (Square::E1, Square::A1, true) => Some("e1c1"),
        (Square::E8, Square::H8, false) => Some("e8g8"),
        (Square::E8, Square::A8, false) => Some("e8c8"),
        _ => None,
    };
    // Only actual castling (king on its start square moving onto an own
    // rook) uses the translation; a plain rook capture with those
    // coordinates cannot occur for the side to move's own rook.
    if let Some(text) = castle {
        if is_own_rook_target(state, mv) {
            return text.to_string();
        }
    }
    let mut text = format!("{}{}", mv.0.from, mv.0.to);
    if let Some(piece) = mv.0.promotion {
        text.push(match piece {
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            _ => '?',
        });
    }
    text
}

fn state_side(state: &ChessState) -> Color {
    Chess::new().side_to_move_color(state)
}

fn is_own_rook_target(state: &ChessState, mv: ChessMove) -> bool {
    Chess::new().is_own_rook_target(state, mv)
}

fn main() {
    let stdin = std::io::stdin();
    let game = Chess::new();
    let mut state = game.initial_state();
    let mut compiled: Option<CompiledNet> = None;
    // `--random <seed>`: uniform random legal mover (evaluation baseline
    // only, §27). Otherwise the first argument is a checkpoint directory.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut random_rng: Option<rand_chacha::ChaCha12Rng> = None;
    // `--qs=<dir>`: checkpoint plus horizon quiescence (SHSD G1/J1).
    // Single-token form so fastchess engine specs stay one word.
    let mut quiescence = false;
    if let Some(seed_text) = args.first().and_then(|a| a.strip_prefix("--random=")) {
        use rand::SeedableRng as _;
        let seed = seed_text.parse().unwrap_or(0u64);
        random_rng = Some(rand_chacha::ChaCha12Rng::seed_from_u64(seed));
    } else if args.first().map(String::as_str) == Some("--random") {
        use rand::SeedableRng as _;
        let seed = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0u64);
        random_rng = Some(rand_chacha::ChaCha12Rng::seed_from_u64(seed));
    } else if let Some(dir) = args.first().and_then(|a| a.strip_prefix("--qs=")) {
        quiescence = true;
        match load_compiled(Path::new(dir)) {
            Ok(net) => compiled = Some(net),
            Err(e) => eprintln!("info string checkpoint error: {e}"),
        }
    } else if let Some(dir) = args.first() {
        match load_compiled(Path::new(dir)) {
            Ok(net) => compiled = Some(net),
            Err(e) => eprintln!("info string checkpoint error: {e}"),
        }
    }
    let mut ranker: Option<MoveRanker> = None;
    let mut searcher: Searcher<Chess> = Searcher::new(Some(UCI_TT_LOG2), MoveOrdering::Natural);
    searcher.set_quiescence(quiescence);

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let mut words = line.split_whitespace();
        match words.next() {
            Some("uci") => {
                println!("id name selfplay-lab");
                println!("id author selfplay-lab research");
                println!("option name Checkpoint type string default <empty>");
                println!("option name Ranker type string default <empty>");
                println!("option name Quiescence type check default false");
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => {
                state = game.initial_state();
                searcher = Searcher::new(Some(UCI_TT_LOG2), MoveOrdering::Natural);
                searcher.set_quiescence(quiescence);
            }
            Some("setoption") => {
                let rest: Vec<&str> = words.collect();
                if let Some(pos) = rest.iter().position(|&w| w == "value") {
                    let value_text = rest[pos + 1..].join(" ");
                    if rest.get(..pos).is_some_and(|n| n.contains(&"Checkpoint")) {
                        match load_compiled(Path::new(&value_text)) {
                            Ok(net) => compiled = Some(net),
                            Err(e) => eprintln!("info string checkpoint error: {e}"),
                        }
                    } else if rest.get(..pos).is_some_and(|n| n.contains(&"Ranker")) {
                        match std::fs::read_to_string(&value_text)
                            .map_err(|e| e.to_string())
                            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
                        {
                            Ok(r) => ranker = Some(r),
                            Err(e) => eprintln!("info string ranker error: {e}"),
                        }
                    } else if rest.get(..pos).is_some_and(|n| n.contains(&"Quiescence")) {
                        quiescence = value_text.trim() == "true";
                        searcher.set_quiescence(quiescence);
                    }
                }
            }
            Some("position") => {
                let rest: Vec<&str> = words.collect();
                let (base, moves_at) = if rest.first() == Some(&"startpos") {
                    (Some(game.initial_state()), 1)
                } else if rest.first() == Some(&"fen") {
                    let end = rest
                        .iter()
                        .position(|&w| w == "moves")
                        .unwrap_or(rest.len());
                    let fen = rest[1..end].join(" ");
                    (game.state_from_fen(&fen).ok(), end)
                } else {
                    (None, 0)
                };
                let Some(mut next) = base else {
                    eprintln!("info string ignored malformed position command");
                    continue;
                };
                let mut ok = true;
                if rest.get(moves_at) == Some(&"moves") {
                    for text in &rest[moves_at + 1..] {
                        match parse_uci_move(&game, &next, text) {
                            Some(mv) => {
                                game.make_move(&mut next, mv);
                            }
                            None => {
                                eprintln!("info string illegal move {text} ignored");
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if ok {
                    state = next;
                }
            }
            Some("go") => {
                let mut node_budget = u64::MAX;
                let mut depth = MAX_DEPTH;
                let mut deadline: Option<Instant> = None;
                let rest: Vec<&str> = words.collect();
                let value = |key: &str| -> Option<u64> {
                    rest.iter()
                        .position(|&w| w == key)
                        .and_then(|i| rest.get(i + 1))
                        .and_then(|v| v.parse().ok())
                };
                if let Some(n) = value("nodes") {
                    node_budget = n;
                }
                if let Some(d) = value("depth") {
                    depth = d as u32;
                }
                if let Some(ms) = value("movetime") {
                    // 20% abort-lag margin so the reply lands inside the
                    // allotted time.
                    deadline = Some(Instant::now() + Duration::from_millis((ms * 8 / 10).max(5)));
                }
                let our_time = match state_side(&state) {
                    Color::White => value("wtime").zip(Some(value("winc").unwrap_or(0))),
                    Color::Black => value("btime").zip(Some(value("binc").unwrap_or(0))),
                };
                if let Some((time_ms, inc_ms)) = our_time {
                    // Simple allocation: 1/30 of remaining plus half the
                    // increment, with a 20% abort-lag margin, floor 10 ms.
                    let budget = ((time_ms / 30 + inc_ms / 2) * 8 / 10).max(10);
                    deadline = Some(Instant::now() + Duration::from_millis(budget));
                }
                if game.outcome(&state).is_some() {
                    println!("bestmove 0000");
                    continue;
                }
                if let Some(rng) = &mut random_rng {
                    use rand::Rng as _;
                    let mut legal = Vec::new();
                    game.legal_moves(&state, &mut legal);
                    let mv = legal[rng.gen_range(0..legal.len())];
                    println!("info depth 0 nodes 0 score cp 0");
                    println!("bestmove {}", format_uci_move(&state, mv));
                    continue;
                }
                // A `go` without any limit (or `go infinite`, which this
                // synchronous engine cannot serve) gets a bounded
                // default so the engine always answers.
                if node_budget == u64::MAX && deadline.is_none() && depth == MAX_DEPTH {
                    deadline = Some(Instant::now() + Duration::from_millis(1000));
                }
                searcher.set_deadline(deadline);
                let started = Instant::now();
                let result = match (&compiled, &ranker) {
                    (Some(net), Some(rk)) => {
                        let mut evaluator =
                            MlpRankedEvaluator::new(net, rk, ChessMoveFeatures::new(&game));
                        searcher.search(&game, &mut state, depth, node_budget, &mut evaluator)
                    }
                    (Some(net), None) => {
                        let mut evaluator = ModelEvaluator::new(net);
                        searcher.search(&game, &mut state, depth, node_budget, &mut evaluator)
                    }
                    _ => searcher.search(&game, &mut state, depth, node_budget, &mut ZeroEvaluator),
                };
                searcher.set_deadline(None);
                let elapsed = started.elapsed().as_millis().max(1);
                let best = result.best_move.expect("non-terminal position");
                println!(
                    "info depth {} nodes {} time {} nps {} score cp {}",
                    result.completed_depth,
                    result.nodes,
                    elapsed,
                    (result.nodes as u128 * 1000 / elapsed),
                    result.value,
                );
                println!("bestmove {}", format_uci_move(&state, best));
            }
            Some("stop") => {
                // Searches are synchronous; nothing to stop.
            }
            Some("quit") => break,
            Some(other) => {
                eprintln!("info string unknown command {other:?} ignored");
            }
            None => {}
        }
    }
}

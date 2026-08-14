//! UCI compliance smoke test (plan §26): protocol handshake, position
//! setup including castling notation, budgeted search, and graceful
//! handling of illegal commands and malformed input.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

struct Engine {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
}

impl Engine {
    fn start() -> Engine {
        let mut child = Command::new(env!("CARGO_BIN_EXE_uci"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn uci binary");
        let stdin = child.stdin.take().unwrap();
        let reader = BufReader::new(child.stdout.take().unwrap());
        Engine {
            child,
            stdin,
            reader,
        }
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").expect("engine stdin");
    }

    /// Read lines until one starts with `prefix`; return it.
    fn expect(&mut self, prefix: &str) -> String {
        for _ in 0..200 {
            let mut line = String::new();
            if self.reader.read_line(&mut line).expect("engine stdout") == 0 {
                break;
            }
            if line.starts_with(prefix) {
                return line.trim().to_string();
            }
        }
        panic!("engine never produced a line starting with {prefix:?}");
    }
}

#[test]
fn uci_handshake_search_and_illegal_input() {
    let mut engine = Engine::start();
    engine.send("uci");
    engine.expect("id name");
    engine.expect("uciok");
    engine.send("isready");
    engine.expect("readyok");

    // Malformed and unknown commands must be ignored without dying.
    engine.send("banana");
    engine.send("position");
    engine.send("position fen not-a-fen");
    engine.send("go nodes"); // missing value: treated as unbounded? must not hang: give depth
    engine.send("isready");
    // The bare `go nodes` line above starts a depth-limited-by-default
    // search only when parsed; our parser ignores the missing value and
    // uses defaults, so bound it now with a fresh quick search instead.
    let _ = engine.expect("bestmove");
    engine.expect("readyok");

    // Standard castling notation must be accepted in `position moves`.
    engine.send("position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1");
    engine.send("go nodes 5000");
    let best = engine.expect("bestmove");
    let mv = best.strip_prefix("bestmove ").unwrap();
    assert!(
        mv.len() >= 4 && mv.chars().all(|c| c.is_ascii_alphanumeric()),
        "bestmove must be a coordinate move, got {best:?}"
    );

    // Terminal position: engine must answer without a move.
    engine.send("position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    engine.send("go nodes 1000");
    let best = engine.expect("bestmove");
    assert_eq!(best, "bestmove 0000", "checkmated position has no move");

    engine.send("quit");
    let status = engine.child.wait().expect("engine exit");
    assert!(status.success());
}

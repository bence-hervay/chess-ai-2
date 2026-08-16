//! Standard chess via the `cozy-chess` rules backend (plan §26).
//!
//! Rules and legality come entirely from `cozy_chess::Board`; this
//! wrapper adds the history the `Game` trait needs for threefold
//! repetition, converts to sparse perspective features, and gives moves
//! stable perspective-relative action IDs.
//!
//! Draw rules implemented: stalemate, fifty-move (both via
//! `Board::status`), and threefold repetition via a hash history that
//! resets on irreversible moves (halfmove-clock resets).
//!
//! Known approximation, standard in chess engines: `position_key` is
//! the board hash only — repetition history is not part of the key, so
//! transposition entries may be reused across different repetition
//! contexts.
//!
//! Features (allowed facts only, §26): piece type and square for both
//! sides (perspective-mirrored ranks), castling rights, en-passant
//! file. Forbidden concepts (material, mobility, attack maps,
//! piece-square tables, pawn categories, king safety) are absent.

use crate::game::{ActionId, FeatureId, Game, Outcome, Player};
use cozy_chess::{Board, Color, GameStatus, Piece, Square};

/// Feature layout: `square(64) x piece(6) x own/opp(2)` = 768, then 4
/// castling rights (own short, own long, opp short, opp long), then 8
/// en-passant files.
const FEATURE_COUNT: usize = 64 * 12 + 4 + 8;
/// Action layout: `from(64) x to(64)` = 4096 for normal moves and queen
/// promotions, plus `3 underpromotion pieces x 8 from-files x 3
/// directions` = 72.
const ACTION_COUNT: usize = 64 * 64 + 3 * 8 * 3;

/// Standard chess rules object.
pub struct Chess;

/// A legal chess move (newtype for a total order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChessMove(pub cozy_chess::Move);

fn move_key(mv: &cozy_chess::Move) -> (u8, u8, u8) {
    (
        mv.from as u8,
        mv.to as u8,
        mv.promotion.map(|p| p as u8 + 1).unwrap_or(0),
    )
}

impl Ord for ChessMove {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        move_key(&self.0).cmp(&move_key(&other.0))
    }
}

impl PartialOrd for ChessMove {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Game state: the board plus the hashes of prior positions since the
/// last irreversible move (for threefold detection).
#[derive(Clone, PartialEq, Debug)]
pub struct ChessState {
    board: Board,
    history: Vec<u64>,
    outcome: Option<Outcome>,
}

/// Undo record: `cozy_chess::Board` is `Copy`, so undo restores the
/// full board; the history either pops one entry or restores the
/// pre-reset tail.
pub struct ChessUndo {
    board: Board,
    cleared_history: Option<Vec<u64>>,
    prev_outcome: Option<Outcome>,
}

fn player_of(color: Color) -> Player {
    match color {
        Color::White => Player::One,
        Color::Black => Player::Two,
    }
}

impl Chess {
    pub fn new() -> Chess {
        Chess
    }

    pub fn cell_count(&self) -> u32 {
        64
    }

    pub fn feature_count(&self) -> usize {
        FEATURE_COUNT
    }

    pub fn action_count(&self) -> usize {
        ACTION_COUNT
    }

    /// Build a state from a FEN string (history starts empty).
    pub fn state_from_fen(&self, fen: &str) -> Result<ChessState, String> {
        let board: Board = fen.parse().map_err(|e| format!("bad FEN {fen:?}: {e}"))?;
        Ok(self.state_from_board(board))
    }

    fn state_from_board(&self, board: Board) -> ChessState {
        let mut state = ChessState {
            board,
            history: Vec::new(),
            outcome: None,
        };
        state.outcome = self.compute_outcome(&state);
        state
    }

    fn compute_outcome(&self, state: &ChessState) -> Option<Outcome> {
        match state.board.status() {
            // `Won` means the side to move is checkmated.
            GameStatus::Won => Some(Outcome::Win(
                player_of(state.board.side_to_move()).opponent(),
            )),
            GameStatus::Drawn => Some(Outcome::Draw),
            GameStatus::Ongoing => {
                let hash = state.board.hash();
                let repeats = state.history.iter().filter(|&&h| h == hash).count();
                if repeats >= 2 {
                    Some(Outcome::Draw)
                } else {
                    None
                }
            }
        }
    }

    /// Perspective square index: ranks are flipped for Black so that
    /// "forward" is always toward higher ranks.
    fn perspective_square(square: Square, side: Color) -> usize {
        square.relative_to(side) as usize
    }

    /// The underlying board (FEN via `Display`; diagnostics).
    pub fn board_of<'a>(&self, state: &'a ChessState) -> &'a Board {
        &state.board
    }

    /// The cozy-chess colour to move (UCI plumbing).
    pub fn side_to_move_color(&self, state: &ChessState) -> Color {
        state.board.side_to_move()
    }

    /// Is `mv` the mover's king moving onto the mover's own rook — i.e.
    /// castling in cozy-chess's encoding? (UCI notation differs.)
    pub fn is_own_rook_target(&self, state: &ChessState, mv: ChessMove) -> bool {
        state.board.piece_on(mv.0.from) == Some(Piece::King)
            && state.board.color_on(mv.0.to) == Some(state.board.side_to_move())
    }
}

impl Default for Chess {
    fn default() -> Self {
        Chess
    }
}

impl Game for Chess {
    type State = ChessState;
    type Move = ChessMove;
    type Undo = ChessUndo;

    fn initial_state(&self) -> ChessState {
        self.state_from_board(Board::startpos())
    }

    fn side_to_move(&self, state: &ChessState) -> Player {
        player_of(state.board.side_to_move())
    }

    fn legal_moves(&self, state: &ChessState, moves: &mut Vec<ChessMove>) {
        moves.clear();
        state.board.generate_moves(|piece_moves| {
            for mv in piece_moves {
                moves.push(ChessMove(mv));
            }
            false
        });
        // cozy-chess yields moves grouped by piece; sort for the stable
        // action-ID order the search contract requires.
        moves.sort();
    }

    fn make_move(&self, state: &mut ChessState, mv: ChessMove) -> ChessUndo {
        let prev_board = state.board.clone();
        let prev_hash = state.board.hash();
        let undo_outcome = state.outcome;
        state.board.play(mv.0);
        let cleared_history = if state.board.halfmove_clock() == 0 {
            // Irreversible move: prior positions can never repeat.
            Some(std::mem::take(&mut state.history))
        } else {
            state.history.push(prev_hash);
            None
        };
        state.outcome = self.compute_outcome(state);
        ChessUndo {
            board: prev_board,
            cleared_history,
            prev_outcome: undo_outcome,
        }
    }

    fn unmake_move(&self, state: &mut ChessState, _mv: ChessMove, undo: ChessUndo) {
        state.board = undo.board;
        match undo.cleared_history {
            Some(history) => state.history = history,
            None => {
                state.history.pop();
            }
        }
        state.outcome = undo.prev_outcome;
    }

    fn outcome(&self, state: &ChessState) -> Option<Outcome> {
        state.outcome
    }

    fn position_key(&self, state: &ChessState) -> u64 {
        state.board.hash()
    }

    fn encode_features(&self, state: &ChessState, features: &mut Vec<FeatureId>) {
        features.clear();
        let stm = state.board.side_to_move();
        for square in Square::ALL {
            if let Some(piece) = state.board.piece_on(square) {
                let color = state.board.color_on(square).expect("occupied square");
                let own = color == stm;
                let sq = Self::perspective_square(square, stm);
                features.push((sq * 12 + piece as usize * 2 + usize::from(!own)) as FeatureId);
            }
        }
        let rights_base = 64 * 12;
        for (slot, color) in [(0, stm), (2, !stm)] {
            let rights = state.board.castle_rights(color);
            if rights.short.is_some() {
                features.push((rights_base + slot) as FeatureId);
            }
            if rights.long.is_some() {
                features.push((rights_base + slot + 1) as FeatureId);
            }
        }
        if let Some(file) = state.board.en_passant() {
            features.push((rights_base + 4 + file as usize) as FeatureId);
        }
        features.sort_unstable();
    }

    fn action_id(&self, state: &ChessState, mv: ChessMove) -> ActionId {
        let stm = state.board.side_to_move();
        let from = Self::perspective_square(mv.0.from, stm);
        let to = Self::perspective_square(mv.0.to, stm);
        match mv.0.promotion {
            None | Some(Piece::Queen) => (from * 64 + to) as ActionId,
            Some(piece) => {
                let promo_index = match piece {
                    Piece::Knight => 0usize,
                    Piece::Bishop => 1,
                    Piece::Rook => 2,
                    _ => unreachable!("pawns only promote to N/B/R/Q"),
                };
                let from_file = from % 8;
                let to_file = to % 8;
                let direction = (to_file as i32 - from_file as i32 + 1) as usize;
                (64 * 64 + promo_index * 24 + from_file * 3 + direction) as ActionId
            }
        }
    }

    fn is_tactical(&self, state: &ChessState, mv: ChessMove) -> bool {
        if mv.0.promotion.is_some() {
            return true;
        }
        // Captures: destination occupied by the enemy, or en passant
        // (a pawn changing file onto an empty square). cozy-chess
        // castling is king-takes-rook — own-piece destination, so it
        // is correctly not counted as a capture here.
        let board = &state.board;
        if let Some(color) = board.color_on(mv.0.to) {
            return color != board.side_to_move();
        }
        board.piece_on(mv.0.from) == Some(cozy_chess::Piece::Pawn)
            && mv.0.from.file() != mv.0.to.file()
    }
}

/// Parse a UCI move string against a state's legal moves, translating
/// standard castling notation (`e1g1`) to cozy-chess king-takes-rook.
pub fn parse_move_text(game: &Chess, state: &ChessState, text: &str) -> Option<ChessMove> {
    let mut legal = Vec::new();
    game.legal_moves(state, &mut legal);
    if let Ok(mv) = text.parse::<cozy_chess::Move>() {
        if let Some(&found) = legal.iter().find(|m| m.0 == mv) {
            return Some(found);
        }
        let translated = match (mv.from, mv.to) {
            (Square::E1, Square::G1) => Some((Square::E1, Square::H1)),
            (Square::E1, Square::C1) => Some((Square::E1, Square::A1)),
            (Square::E8, Square::G8) => Some((Square::E8, Square::H8)),
            (Square::E8, Square::C8) => Some((Square::E8, Square::A8)),
            _ => None,
        };
        if let Some((from, to)) = translated {
            return legal
                .iter()
                .find(|m| m.0.from == from && m.0.to == to && m.0.promotion.is_none())
                .copied();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    fn perft(game: &Chess, state: &mut ChessState, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        let mut nodes = 0;
        for &mv in &moves {
            let undo = game.make_move(state, mv);
            nodes += perft(game, state, depth - 1);
            game.unmake_move(state, mv, undo);
        }
        nodes
    }

    #[test]
    fn established_perft_positions() {
        let game = Chess::new();
        // (FEN, [d1, d2, d3, d4]) — standard published perft values.
        let cases: [(&str, &[u64]); 5] = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                &[20, 400, 8_902, 197_281],
            ),
            (
                // Kiwipete: castling, pins, en passant, promotions.
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                &[48, 2_039, 97_862],
            ),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                &[14, 191, 2_812, 43_238],
            ),
            (
                // Promotion-heavy position 4.
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                &[6, 264, 9_467],
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                &[44, 1_486, 62_379],
            ),
        ];
        for (fen, expected) in cases {
            let mut state = game.state_from_fen(fen).unwrap();
            for (i, &nodes) in expected.iter().enumerate() {
                assert_eq!(
                    perft(&game, &mut state, i as u32 + 1),
                    nodes,
                    "perft({}) of {fen}",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn randomized_differential_against_cozy_chess() {
        let game = Chess::new();
        let mut rng = ChaCha12Rng::seed_from_u64(0xd1ff);
        for _ in 0..60 {
            let mut state = game.initial_state();
            let mut moves = Vec::new();
            for _ply in 0..120 {
                if game.outcome(&state).is_some() {
                    break;
                }
                game.legal_moves(&state, &mut moves);
                // Differential: our move list equals cozy's, sorted.
                let mut reference = Vec::new();
                state.board.generate_moves(|pm| {
                    for m in pm {
                        reference.push(ChessMove(m));
                    }
                    false
                });
                reference.sort();
                assert_eq!(moves, reference);
                // Status agreement while ongoing (threefold aside, which
                // cozy does not track).
                if state.board.status() != GameStatus::Ongoing {
                    panic!("outcome() should have been terminal");
                }
                let mv = moves[rng.gen_range(0..moves.len())];
                let before = state.clone();
                let undo = game.make_move(&mut state, mv);
                game.unmake_move(&mut state, mv, undo);
                assert_eq!(state, before, "make/unmake must restore exactly");
                assert_eq!(game.position_key(&state), game.position_key(&before));
                game.make_move(&mut state, mv);
            }
        }
    }

    #[test]
    fn check_mate_and_stalemate() {
        let game = Chess::new();
        // Fool's mate: checkmate, Black (P2) wins.
        let state = game
            .state_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .unwrap();
        assert_eq!(game.outcome(&state), Some(Outcome::Win(Player::Two)));
        // Classic stalemate: draw.
        let state = game
            .state_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
            .unwrap();
        assert_eq!(game.outcome(&state), Some(Outcome::Draw));
        // Check but not mate: ongoing.
        let state = game
            .state_from_fen("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
            .unwrap();
        assert!(game.outcome(&state).is_none());
    }

    #[test]
    fn castling_en_passant_and_all_promotions() {
        let game = Chess::new();
        // Both castling moves legal and playable.
        let mut state = game
            .state_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        // cozy-chess encodes castling as king-takes-rook.
        let short: ChessMove = ChessMove("e1h1".parse().unwrap());
        let long: ChessMove = ChessMove("e1a1".parse().unwrap());
        assert!(moves.contains(&short) && moves.contains(&long));
        game.make_move(&mut state, short);
        assert!(state.board.king(Color::White) == Square::G1);

        // En passant is generated and captures the pawn.
        let mut state = game
            .state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
            .unwrap();
        game.legal_moves(&state, &mut moves);
        let ep: ChessMove = ChessMove("e5d6".parse().unwrap());
        assert!(moves.contains(&ep));
        game.make_move(&mut state, ep);
        assert_eq!(state.board.piece_on(Square::D5), None, "captured pawn gone");

        // All four promotions, and their action IDs are distinct.
        let state = game
            .state_from_fen("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
            .unwrap();
        game.legal_moves(&state, &mut moves);
        let promos: Vec<ChessMove> = moves
            .iter()
            .filter(|m| m.0.promotion.is_some())
            .copied()
            .collect();
        assert_eq!(promos.len(), 4);
        let mut ids: Vec<ActionId> = promos.iter().map(|&m| game.action_id(&state, m)).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 4, "promotion action IDs must be distinct");
        for &mv in &promos {
            let mut s = state.clone();
            game.make_move(&mut s, mv);
            assert_eq!(s.board.piece_on(Square::B8), mv.0.promotion);
        }
    }

    #[test]
    fn threefold_repetition_is_a_draw() {
        let game = Chess::new();
        let mut state = game
            .state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
            .unwrap();
        let seq = [
            "a1b1", "e8d8", "b1a1", "d8e8", "a1b1", "e8d8", "b1a1", "d8e8",
        ];
        for (i, uci) in seq.iter().enumerate() {
            assert!(
                game.outcome(&state).is_none(),
                "premature termination before move {i}"
            );
            game.make_move(&mut state, ChessMove(uci.parse().unwrap()));
        }
        // The start position has now occurred three times.
        assert_eq!(game.outcome(&state), Some(Outcome::Draw));
    }

    #[test]
    fn fifty_move_rule_is_a_draw() {
        let game = Chess::new();
        let mut state = game
            .state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 70")
            .unwrap();
        assert!(game.outcome(&state).is_none());
        game.make_move(&mut state, ChessMove("a1a2".parse().unwrap()));
        assert_eq!(game.outcome(&state), Some(Outcome::Draw));
    }

    #[test]
    fn features_and_action_ids_are_perspective_consistent() {
        let game = Chess::new();
        // Colour-mirrored position: white-to-move vs black-to-move with
        // ranks flipped must encode identically.
        let white = game
            .state_from_fen("4k3/2p5/8/8/8/8/2P5/4K2R w K - 0 1")
            .unwrap();
        let black = game
            .state_from_fen("4k2r/2p5/8/8/8/8/2P5/4K3 b k - 0 1")
            .unwrap();
        let (mut fw, mut fb) = (Vec::new(), Vec::new());
        game.encode_features(&white, &mut fw);
        game.encode_features(&black, &mut fb);
        assert_eq!(fw, fb, "mirrored positions must encode identically");
        let mut mw = Vec::new();
        let mut mb = Vec::new();
        game.legal_moves(&white, &mut mw);
        game.legal_moves(&black, &mut mb);
        let mut iw: Vec<ActionId> = mw.iter().map(|&m| game.action_id(&white, m)).collect();
        let mut ib: Vec<ActionId> = mb.iter().map(|&m| game.action_id(&black, m)).collect();
        iw.sort_unstable();
        ib.sort_unstable();
        assert_eq!(iw, ib, "mirrored action IDs must match");
        // Action IDs are unique per state and in range.
        let mut dedup = iw.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), iw.len());
        assert!(iw.iter().all(|&id| (id as usize) < game.action_count()));
    }
}

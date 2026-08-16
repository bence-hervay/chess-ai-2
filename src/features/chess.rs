//! State-action measurements for standard chess (SHSD §35.2, J2 card).
//!
//! The same rule-level vocabulary as the Forward Chess move features,
//! minus orientation (chess pieces have none): moving/captured piece
//! type, promotion choice, check after the move, destination and
//! source attack facts, relative destination rank, castling, en
//! passant. No piece values, no strategic signs — the ranker learns
//! them (learned MVV-LVA rather than a hardcoded one).

use crate::features::{FeatureEntry, MoveFeatures};
use crate::games::chess::{Chess, ChessMove, ChessState};
use cozy_chess::{Board, Color, Piece, Square};

/// Move-feature index layout (dimension 25).
const MF_MOVING: usize = 0; // ..6: moving piece type
const MF_CAPTURED: usize = 6; // ..12: captured piece type
const MF_IS_CAPTURE: usize = 12;
const MF_PROMO_CHOICE: usize = 13; // ..17: N,B,R,Q
const MF_IS_PROMO: usize = 17;
const MF_GIVES_CHECK: usize = 18;
const MF_DEST_ATTACKED: usize = 19;
const MF_DEST_DEFENDED: usize = 20;
const MF_SOURCE_ATTACKED: usize = 21;
const MF_DEST_REL_RANK: usize = 22;
const MF_IS_CASTLE: usize = 23;
const MF_IS_EP: usize = 24;
pub const CHESS_MOVE_FEATURE_DIMENSION: usize = 25;

/// Is `square` attacked by any piece of `by` on `board`?
fn square_attacked(board: &Board, square: Square, by: Color) -> bool {
    let them = board.colors(by);
    let occupied = board.occupied();
    if !(cozy_chess::get_knight_moves(square) & them & board.pieces(Piece::Knight)).is_empty() {
        return true;
    }
    if !(cozy_chess::get_king_moves(square) & them & board.pieces(Piece::King)).is_empty() {
        return true;
    }
    // Pawns attacking `square` sit on the squares a pawn of the OTHER
    // colour on `square` would attack.
    if !(cozy_chess::get_pawn_attacks(square, !by) & them & board.pieces(Piece::Pawn)).is_empty() {
        return true;
    }
    let diagonal = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
    if !(cozy_chess::get_bishop_moves(square, occupied) & them & diagonal).is_empty() {
        return true;
    }
    let straight = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    !(cozy_chess::get_rook_moves(square, occupied) & them & straight).is_empty()
}

/// Chess move-feature extractor.
pub struct ChessMoveFeatures;

impl ChessMoveFeatures {
    pub fn new(_game: &Chess) -> ChessMoveFeatures {
        ChessMoveFeatures
    }
}

impl MoveFeatures<Chess> for ChessMoveFeatures {
    fn dimension(&self) -> usize {
        CHESS_MOVE_FEATURE_DIMENSION
    }

    fn extract(
        &mut self,
        game: &Chess,
        state: &ChessState,
        mv: ChessMove,
        out: &mut Vec<FeatureEntry>,
    ) {
        out.clear();
        let board = game.board_of(state);
        let stm = board.side_to_move();
        let piece = board
            .piece_on(mv.0.from)
            .expect("move from occupied square");
        out.push(((MF_MOVING + piece as usize) as u32, 1.0));

        let castle = game.is_own_rook_target(state, mv);
        let is_ep = piece == Piece::Pawn
            && mv.0.from.file() != mv.0.to.file()
            && board.piece_on(mv.0.to).is_none();
        if !castle {
            if let Some(victim) = board.piece_on(mv.0.to) {
                out.push(((MF_CAPTURED + victim as usize) as u32, 1.0));
                out.push((MF_IS_CAPTURE as u32, 1.0));
            } else if is_ep {
                out.push(((MF_CAPTURED + Piece::Pawn as usize) as u32, 1.0));
                out.push((MF_IS_CAPTURE as u32, 1.0));
                out.push((MF_IS_EP as u32, 1.0));
            }
        }
        if let Some(promoted) = mv.0.promotion {
            let slot = match promoted {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                Piece::Queen => 3,
                _ => unreachable!("promotion choices are N/B/R/Q"),
            };
            out.push(((MF_PROMO_CHOICE + slot) as u32, 1.0));
            out.push((MF_IS_PROMO as u32, 1.0));
        }
        // Direct + discovered check: cozy tracks checkers after play.
        let mut after = board.clone();
        after.play_unchecked(mv.0);
        if !after.checkers().is_empty() {
            out.push((MF_GIVES_CHECK as u32, 1.0));
        }
        if square_attacked(board, mv.0.to, !stm) {
            out.push((MF_DEST_ATTACKED as u32, 1.0));
        }
        if square_attacked(board, mv.0.to, stm) {
            out.push((MF_DEST_DEFENDED as u32, 1.0));
        }
        if square_attacked(board, mv.0.from, !stm) {
            out.push((MF_SOURCE_ATTACKED as u32, 1.0));
        }
        let rel_rank = mv.0.to.relative_to(stm).rank() as i32;
        if rel_rank != 0 {
            out.push((MF_DEST_REL_RANK as u32, rel_rank as f32));
        }
        if castle {
            out.push((MF_IS_CASTLE as u32, 1.0));
        }
    }

    fn feature_name(&self, index: u32) -> String {
        const PIECES: [&str; 6] = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
        let index = index as usize;
        match index {
            i if i < MF_CAPTURED => format!("moving/{}", PIECES[i - MF_MOVING]),
            i if i < MF_IS_CAPTURE => format!("captured/{}", PIECES[i - MF_CAPTURED]),
            MF_IS_CAPTURE => "is_capture".into(),
            i if i < MF_IS_PROMO => {
                format!("promo_choice/{}", ["N", "B", "R", "Q"][i - MF_PROMO_CHOICE])
            }
            MF_IS_PROMO => "is_promotion".into(),
            MF_GIVES_CHECK => "gives_check".into(),
            MF_DEST_ATTACKED => "dest_attacked".into(),
            MF_DEST_DEFENDED => "dest_defended".into(),
            MF_SOURCE_ATTACKED => "source_attacked".into(),
            MF_DEST_REL_RANK => "dest_relative_rank".into(),
            MF_IS_CASTLE => "is_castle".into(),
            MF_IS_EP => "is_en_passant".into(),
            _ => format!("unknown/{index}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;

    #[test]
    fn hand_computed_chess_move_features() {
        let game = Chess::new();
        // Italian: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 — White to move.
        let state = game
            .state_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            .unwrap();
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        let mut extractor = ChessMoveFeatures::new(&game);
        let mut x = Vec::new();
        let find = |from: &str, to: &str, moves: &[ChessMove]| {
            moves
                .iter()
                .copied()
                .find(|m| format!("{}", m.0.from) == from && format!("{}", m.0.to) == to)
                .expect("move exists")
        };
        let get = |x: &[FeatureEntry], i: usize| {
            x.iter()
                .find(|&&(f, _)| f as usize == i)
                .map(|&(_, v)| v)
                .unwrap_or(0.0)
        };
        // Nxe5: knight takes the defended e5 pawn.
        let nxe5 = find("f3", "e5", &moves);
        extractor.extract(&game, &state, nxe5, &mut x);
        assert_eq!(get(&x, MF_MOVING + Piece::Knight as usize), 1.0);
        assert_eq!(get(&x, MF_CAPTURED + Piece::Pawn as usize), 1.0);
        assert_eq!(get(&x, MF_IS_CAPTURE), 1.0);
        assert_eq!(get(&x, MF_DEST_ATTACKED), 1.0, "e5 is defended by Nc6");
        assert_eq!(get(&x, MF_GIVES_CHECK), 0.0);
        // Bxf7+: bishop takes f7 with check.
        let bxf7 = find("c4", "f7", &moves);
        extractor.extract(&game, &state, bxf7, &mut x);
        assert_eq!(get(&x, MF_GIVES_CHECK), 1.0, "Bxf7 is check");
        assert_eq!(get(&x, MF_CAPTURED + Piece::Pawn as usize), 1.0);
        assert_eq!(get(&x, MF_DEST_ATTACKED), 1.0, "f7 defended by the king");
        // Castling: e1h1 in cozy encoding (king takes own rook).
        let castle = find("e1", "h1", &moves);
        extractor.extract(&game, &state, castle, &mut x);
        assert_eq!(get(&x, MF_IS_CASTLE), 1.0);
        assert_eq!(get(&x, MF_IS_CAPTURE), 0.0, "castling is not a capture");
        // Names resolve.
        for i in 0..CHESS_MOVE_FEATURE_DIMENSION {
            assert!(!extractor.feature_name(i as u32).is_empty());
        }
    }
}

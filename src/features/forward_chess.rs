//! Structured measurements for Forward Chess (SHSD §56 initial families).
//!
//! Portability levels (§16): counts and mobility are U (universal),
//! promotion distance is D (directional), attack/defence and check are
//! C (chess-family), orientation split is F (Forward-Chess-specific).
//!
//! Encoding conventions:
//! - Everything is a *difference* (mover − opponent) or a mover-side
//!   measurement, so one weight vector serves both sides.
//! - Square-indexed features use each piece **owner's** perspective
//!   cell (the board rotated 180° for Black), which makes extraction
//!   exactly invariant under the rotation colour swap — the property
//!   test below is the §69.2 symmetry check (a D031-class guard).
//! - No feature asserts a strategic sign; fitting decides (§6.2).

use crate::features::{FeatureEntry, FeatureExtractor};
use crate::game::Game;
use crate::games::forward_chess::{ForwardChess, Piece};

/// Named, immutable feature recipes (§12.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FcRecipe {
    /// Level 0 (§17.1): piece-type/orientation count differences plus
    /// total material. 10 dimensions.
    FcCountsV0,
    /// Level 1 (§9.4/§56): counts + attacked/defended + mobility +
    /// check + promotion distance + total material + piece-square
    /// tables per orientation.
    FcStructuredLinearV1,
}

impl FcRecipe {
    pub fn label(self) -> &'static str {
        match self {
            FcRecipe::FcCountsV0 => "fc_counts_v0",
            FcRecipe::FcStructuredLinearV1 => "fc_structured_linear_v1",
        }
    }
}

/// `(piece, reversed)` combinations that can occur in legal play:
/// all six natural pieces, and the four reversed promotion choices.
const COMBOS: [(Piece, bool); 10] = [
    (Piece::Pawn, false),
    (Piece::Knight, false),
    (Piece::Bishop, false),
    (Piece::Rook, false),
    (Piece::Queen, false),
    (Piece::King, false),
    (Piece::Knight, true),
    (Piece::Bishop, true),
    (Piece::Rook, true),
    (Piece::Queen, true),
];

fn combo_index(piece: Piece, reversed: bool) -> usize {
    if !reversed {
        piece as usize
    } else {
        match piece {
            Piece::Knight => 6,
            Piece::Bishop => 7,
            Piece::Rook => 8,
            Piece::Queen => 9,
            _ => panic!("reversed {piece:?} cannot occur in legal play"),
        }
    }
}

fn combo_name(index: usize) -> String {
    let (piece, reversed) = COMBOS[index];
    format!("{:?}{}", piece, if reversed { "-rev" } else { "-nat" })
}

/// Count-difference slots: every combo except the natural king (whose
/// difference is identically zero while both kings live).
const COUNT_SLOTS: usize = 9;

fn count_slot(combo: usize) -> Option<usize> {
    match combo {
        5 => None, // natural king
        c if c < 5 => Some(c),
        c => Some(c - 1), // reversed combos shift down by one
    }
}

/// Structured feature extractor for Forward Chess.
pub struct FcExtractor {
    recipe: FcRecipe,
    /// Promotion-distance slots: d = 1 ..= height-2.
    promo_slots: usize,
    cells: usize,
    moves_buf: Vec<<ForwardChess as Game>::Move>,
    scratch: Vec<FeatureEntry>,
}

impl FcExtractor {
    pub fn new(game: &ForwardChess, recipe: FcRecipe) -> FcExtractor {
        FcExtractor {
            recipe,
            promo_slots: usize::from(game.height()).saturating_sub(2),
            cells: game.cell_count() as usize,
            moves_buf: Vec::new(),
            scratch: Vec::new(),
        }
    }

    pub fn recipe(&self) -> FcRecipe {
        self.recipe
    }

    // Fixed index layout (v1); v0 is a prefix-plus-material subset.
    fn idx_counts(&self) -> usize {
        0
    }
    fn idx_attacked(&self) -> usize {
        COUNT_SLOTS
    }
    fn idx_defended(&self) -> usize {
        self.idx_attacked() + 6
    }
    fn idx_mobility(&self) -> usize {
        self.idx_defended() + 1
    }
    fn idx_check(&self) -> usize {
        self.idx_mobility() + 1
    }
    fn idx_promo(&self) -> usize {
        self.idx_check() + 1
    }
    fn idx_material(&self) -> usize {
        match self.recipe {
            FcRecipe::FcCountsV0 => COUNT_SLOTS,
            FcRecipe::FcStructuredLinearV1 => self.idx_promo() + self.promo_slots,
        }
    }
    fn idx_psq(&self) -> usize {
        self.idx_material() + 1
    }
}

impl FeatureExtractor<ForwardChess> for FcExtractor {
    fn dimension(&self) -> usize {
        match self.recipe {
            FcRecipe::FcCountsV0 => COUNT_SLOTS + 1,
            FcRecipe::FcStructuredLinearV1 => self.idx_psq() + COMBOS.len() * self.cells,
        }
    }

    fn extract(
        &mut self,
        game: &ForwardChess,
        state: &<ForwardChess as Game>::State,
        out: &mut Vec<FeatureEntry>,
    ) {
        out.clear();
        let full = self.recipe == FcRecipe::FcStructuredLinearV1;
        let mover = game.side_to_move(state);
        let cells = game.state_cells(state);

        let mut counts = [0i32; COUNT_SLOTS];
        let mut attacked = [0i32; 6];
        let mut defended = 0i32;
        let mut promo = [0i32; 8]; // height <= 10 in practice; slots used: promo_slots
        let mut material = 0i32;
        self.scratch.clear();

        for (cell, &code) in cells.iter().enumerate() {
            if code == 0 {
                continue;
            }
            let (owner, piece, reversed) = ForwardChess::unpack_code(code);
            let combo = combo_index(piece, reversed);
            let sign = if owner == mover { 1i32 } else { -1i32 };
            if let Some(slot) = count_slot(combo) {
                counts[slot] += sign;
            }
            if piece != Piece::King {
                material += 1;
            }
            if full {
                let cell = cell as u16;
                if game.cell_attacked_by(cells, cell, owner.opponent()) {
                    // Mover's piece attacked by the opponent counts -1;
                    // the opponent's piece attacked by the mover counts +1.
                    attacked[piece as usize] -= sign;
                }
                if piece != Piece::King && game.cell_attacked_by(cells, cell, owner) {
                    defended += sign;
                }
                if piece == Piece::Pawn && !reversed {
                    let d = usize::from(game.pawn_promotion_distance(owner, cell));
                    debug_assert!(d >= 1, "an unpromoted pawn is short of the last rank");
                    if d - 1 < self.promo_slots {
                        promo[d - 1] += sign;
                    }
                }
                let rel = usize::from(game.perspective_cell(cell, owner));
                self.scratch.push((
                    (self.idx_psq() + combo * self.cells + rel) as u32,
                    sign as f32,
                ));
            }
        }

        for (slot, &value) in counts.iter().enumerate() {
            if value != 0 {
                out.push(((self.idx_counts() + slot) as u32, value as f32));
            }
        }
        if full {
            for (slot, &value) in attacked.iter().enumerate() {
                if value != 0 {
                    out.push(((self.idx_attacked() + slot) as u32, value as f32));
                }
            }
            if defended != 0 {
                out.push((self.idx_defended() as u32, defended as f32));
            }
            game.legal_moves(state, &mut self.moves_buf);
            out.push((self.idx_mobility() as u32, self.moves_buf.len() as f32));
            if game.in_check(state) {
                out.push((self.idx_check() as u32, 1.0));
            }
            for (slot, &value) in promo.iter().take(self.promo_slots).enumerate() {
                if value != 0 {
                    out.push(((self.idx_promo() + slot) as u32, value as f32));
                }
            }
        }
        if material != 0 {
            out.push((self.idx_material() as u32, material as f32));
        }
        if full {
            // PSQ entries: a mover and an opponent piece can map to the
            // same (combo, perspective cell); merge and drop zeros.
            self.scratch.sort_unstable_by_key(|&(index, _)| index);
            let mut iter = self.scratch.iter().copied().peekable();
            while let Some((index, mut value)) = iter.next() {
                while iter.peek().is_some_and(|&(next, _)| next == index) {
                    value += iter.next().expect("peeked").1;
                }
                if value != 0.0 {
                    out.push((index, value));
                }
            }
        }
        debug_assert!(out.windows(2).all(|w| w[0].0 != w[1].0 || w[0].0 < w[1].0));
    }

    fn feature_name(&self, index: u32) -> String {
        let index = index as usize;
        let full = self.recipe == FcRecipe::FcStructuredLinearV1;
        if index < COUNT_SLOTS {
            let combo = (0..COMBOS.len())
                .find(|&c| count_slot(c) == Some(index))
                .expect("count slot maps back");
            return format!("count_diff/{}", combo_name(combo));
        }
        if !full {
            return "material_total".to_string();
        }
        if index < self.idx_defended() {
            let piece = index - self.idx_attacked();
            return format!("attacked_diff/{}", combo_name(piece));
        }
        if index == self.idx_defended() {
            return "defended_diff".to_string();
        }
        if index == self.idx_mobility() {
            return "mover_mobility".to_string();
        }
        if index == self.idx_check() {
            return "mover_in_check".to_string();
        }
        if index < self.idx_material() {
            return format!("promo_dist_diff/d{}", index - self.idx_promo() + 1);
        }
        if index == self.idx_material() {
            return "material_total".to_string();
        }
        let rest = index - self.idx_psq();
        let (combo, rel) = (rest / self.cells, rest % self.cells);
        format!("psq_diff/{}/r{}", combo_name(combo), rel)
    }
}

/// State-action (move) measurements for ordering (§35.2 rule-level and
/// tactical inputs). Search-history inputs (TT move, killers) are not
/// features: the TT-move-first rule already applies on top of any
/// evaluator ordering.
pub struct FcMoveFeatures {
    width: u16,
    height: u16,
    cells_scratch: Vec<u8>,
}

/// Move-feature index layout (dimension 35).
const MF_MOVING: usize = 0; // ..10: moving piece combo
const MF_CAPTURED: usize = 10; // ..20: captured piece combo
const MF_IS_CAPTURE: usize = 20;
const MF_PROMO_CHOICE: usize = 21; // ..25: N,B,R,Q
const MF_IS_PROMO: usize = 25;
const MF_GIVES_CHECK: usize = 26;
const MF_FORWARD_DISP: usize = 27;
const MF_HORIZONTAL: usize = 28;
const MF_DEST_ATTACKED: usize = 29;
const MF_DEST_DEFENDED: usize = 30;
const MF_SOURCE_ATTACKED: usize = 31;
const MF_DEST_REL_RANK: usize = 32;
const MF_IS_CASTLE: usize = 33;
const MF_IS_EP: usize = 34;
pub const MOVE_FEATURE_DIMENSION: usize = 35;

impl FcMoveFeatures {
    pub fn new(game: &ForwardChess) -> FcMoveFeatures {
        FcMoveFeatures {
            width: game.width(),
            height: game.height(),
            cells_scratch: Vec::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        MOVE_FEATURE_DIMENSION
    }

    /// Clear `out` and append the move's features, mover perspective.
    pub fn extract(
        &mut self,
        game: &ForwardChess,
        state: &<ForwardChess as Game>::State,
        mv: <ForwardChess as Game>::Move,
        out: &mut Vec<FeatureEntry>,
    ) {
        out.clear();
        let cells = game.state_cells(state);
        let mover = game.side_to_move(state);
        let opponent = mover.opponent();
        let (_, piece, reversed) = ForwardChess::unpack_code(cells[usize::from(mv.from)]);
        out.push(((MF_MOVING + combo_index(piece, reversed)) as u32, 1.0));

        let ep = game.state_ep(state);
        let is_ep = piece == Piece::Pawn && Some(mv.to) == ep && cells[usize::from(mv.to)] == 0;
        let captured_code = cells[usize::from(mv.to)];
        if captured_code != 0 || is_ep {
            let (victim_piece, victim_rev) = if is_ep {
                (Piece::Pawn, false)
            } else {
                let (_, p, r) = ForwardChess::unpack_code(captured_code);
                (p, r)
            };
            out.push((
                (MF_CAPTURED + combo_index(victim_piece, victim_rev)) as u32,
                1.0,
            ));
            out.push((MF_IS_CAPTURE as u32, 1.0));
        }
        if let Some(promoted) = mv.promotion {
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

        // Direct + discovered check: apply the move to a scratch board
        // and test whether the opponent's king is attacked.
        self.cells_scratch.clear();
        self.cells_scratch.extend_from_slice(cells);
        game.apply_move_to_cells(&mut self.cells_scratch, ep, mv);
        let enemy_king = game.king_cell(&self.cells_scratch, opponent);
        if game.cell_attacked_by(&self.cells_scratch, enemy_king, mover) {
            out.push((MF_GIVES_CHECK as u32, 1.0));
        }

        let from_rank = i32::from(mv.from / self.width);
        let to_rank = i32::from(mv.to / self.width);
        let forward = if mover == crate::game::Player::One {
            1
        } else {
            -1
        };
        let disp = (to_rank - from_rank) * forward;
        if disp != 0 {
            out.push((MF_FORWARD_DISP as u32, disp as f32));
        } else {
            out.push((MF_HORIZONTAL as u32, 1.0));
        }
        if game.cell_attacked_by(cells, mv.to, opponent) {
            out.push((MF_DEST_ATTACKED as u32, 1.0));
        }
        if game.cell_attacked_by(cells, mv.to, mover) {
            out.push((MF_DEST_DEFENDED as u32, 1.0));
        }
        if game.cell_attacked_by(cells, mv.from, opponent) {
            out.push((MF_SOURCE_ATTACKED as u32, 1.0));
        }
        let rel_rank = if mover == crate::game::Player::One {
            to_rank
        } else {
            i32::from(self.height) - 1 - to_rank
        };
        if rel_rank != 0 {
            out.push((MF_DEST_REL_RANK as u32, rel_rank as f32));
        }
        if piece == Piece::King
            && (i32::from(mv.to % self.width) - i32::from(mv.from % self.width)).abs() == 2
        {
            out.push((MF_IS_CASTLE as u32, 1.0));
        }
        if is_ep {
            out.push((MF_IS_EP as u32, 1.0));
        }
    }

    pub fn feature_name(index: u32) -> String {
        let index = index as usize;
        match index {
            i if i < MF_CAPTURED => format!("moving/{}", combo_name(i - MF_MOVING)),
            i if i < MF_IS_CAPTURE => format!("captured/{}", combo_name(i - MF_CAPTURED)),
            MF_IS_CAPTURE => "is_capture".into(),
            i if i < MF_IS_PROMO => {
                format!("promo_choice/{}", ["N", "B", "R", "Q"][i - MF_PROMO_CHOICE])
            }
            MF_IS_PROMO => "is_promotion".into(),
            MF_GIVES_CHECK => "gives_check".into(),
            MF_FORWARD_DISP => "forward_displacement".into(),
            MF_HORIZONTAL => "is_horizontal".into(),
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
    use crate::game::Player;
    use crate::games::forward_chess::Ruleset;
    use rand::Rng as _;
    use rand::SeedableRng as _;

    fn extract_all(
        game: &ForwardChess,
        recipe: FcRecipe,
        state: &<ForwardChess as Game>::State,
    ) -> Vec<FeatureEntry> {
        let mut extractor = FcExtractor::new(game, recipe);
        let mut out = Vec::new();
        extractor.extract(game, state, &mut out);
        assert!(out
            .iter()
            .all(|&(i, _)| (i as usize) < extractor.dimension()));
        out
    }

    /// 180°-rotation colour swap of a state, built through the public
    /// custom-state constructor (the exact symmetry of every ruleset's
    /// initial layout).
    fn rotate_swap(
        game: &ForwardChess,
        state: &<ForwardChess as Game>::State,
    ) -> <ForwardChess as Game>::State {
        let cells = game.state_cells(state);
        let total = game.cell_count() as u16;
        let name = |cell: u16| {
            format!(
                "{}{}",
                (b'a' + (cell % game.width()) as u8) as char,
                cell / game.width() + 1
            )
        };
        let mut pieces: Vec<(String, Player, Piece, bool)> = Vec::new();
        for (cell, &code) in cells.iter().enumerate() {
            if code == 0 {
                continue;
            }
            let (owner, piece, reversed) = ForwardChess::unpack_code(code);
            pieces.push((
                name(total - 1 - cell as u16),
                owner.opponent(),
                piece,
                reversed,
            ));
        }
        let refs: Vec<(&str, Player, Piece, bool)> = pieces
            .iter()
            .map(|(n, o, p, r)| (n.as_str(), *o, *p, *r))
            .collect();
        game.custom_state(
            &refs,
            game.side_to_move(state).opponent(),
            [false, false, false, false],
            None,
        )
    }

    #[test]
    fn rotation_swap_invariance_on_random_positions() {
        // Features must be identical for a position and its rotated
        // colour swap (mover perspective). Random castle-free positions
        // on the small board; both recipes.
        let game = ForwardChess::new(Ruleset::Small);
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(41);
        let mut moves = Vec::new();
        let mut tested = 0;
        for _ in 0..40 {
            let mut state = game.initial_state();
            loop {
                if game.outcome(&state).is_some() {
                    break;
                }
                for recipe in [FcRecipe::FcCountsV0, FcRecipe::FcStructuredLinearV1] {
                    let a = extract_all(&game, recipe, &state);
                    let b = extract_all(&game, recipe, &rotate_swap(&game, &state));
                    assert_eq!(a, b, "recipe {recipe:?} must be rotation-swap invariant");
                }
                tested += 1;
                game.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                game.make_move(&mut state, mv);
            }
        }
        assert!(tested > 100);
    }

    #[test]
    fn hand_computed_position() {
        // Small board (4x4): White Kb1, Rc1, Pa2; Black Kd4, reversed
        // rook b3 (a promoted piece; its "ahead" is Black's backward,
        // i.e. toward rank 4). White to move; nobody is in check.
        let game = ForwardChess::new(Ruleset::Small);
        let state = game.custom_state(
            &[
                ("b1", Player::One, Piece::King, false),
                ("c1", Player::One, Piece::Rook, false),
                ("a2", Player::One, Piece::Pawn, false),
                ("d4", Player::Two, Piece::King, false),
                ("b3", Player::Two, Piece::Rook, true),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let game_ref = &game;
        let mut extractor = FcExtractor::new(game_ref, FcRecipe::FcStructuredLinearV1);
        let mut out = Vec::new();
        extractor.extract(game_ref, &state, &mut out);
        let get = |index: usize| {
            out.iter()
                .find(|&&(i, _)| i as usize == index)
                .map(|&(_, v)| v)
                .unwrap_or(0.0)
        };
        // Count diffs: mover has P-nat +1, R-nat +1; opponent has R-rev -1.
        assert_eq!(get(0), 1.0, "pawn-nat diff");
        assert_eq!(get(3), 1.0, "rook-nat diff");
        assert_eq!(
            get(7),
            -1.0,
            "rook-rev diff (slot 7: combos shift past the skipped king)"
        );
        // Material total: 3 non-king pieces.
        assert_eq!(get(extractor.idx_material()), 3.0);
        // White pawn on a2: two ranks from promotion (a4).
        assert_eq!(get(extractor.idx_promo() + 1), 1.0, "pawn at distance 2");
        // The reversed rook's ahead is rank 4; nothing attacks b1.
        assert_eq!(get(extractor.idx_check()), 0.0, "white is not in check");
        // The only piece under attack is Black's reversed rook on b3,
        // hit by the a2 pawn's diagonal-ahead capture: +1 for the mover
        // on the rook victim row.
        assert_eq!(get(extractor.idx_attacked() + Piece::Rook as usize), 1.0);
        // Defended: Kb1 covers a2 and c1; Black's b3 rook is undefended.
        assert_eq!(get(extractor.idx_defended()), 2.0);
        // Mobility equals the legal move count (hand count: 3 king,
        // 4 rook, pawn a3 + axb3 = 9; the a2-a4 double step is illegal
        // because it would land on the promotion rank).
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        assert_eq!(moves.len(), 9);
        assert_eq!(get(extractor.idx_mobility()), moves.len() as f32);
        // PSQ: White rook (combo 3) on c1 = cell 2 from White's view.
        assert_eq!(get(extractor.idx_psq() + 3 * 16 + 2), 1.0);
        // Black reversed rook (combo 8) on b3 = cell 9; Black's view is
        // rotated: 15 - 9 = 6.
        assert_eq!(get(extractor.idx_psq() + 8 * 16 + 6), -1.0);
        // Feature names resolve for every emitted index.
        for &(index, _) in &out {
            assert!(!extractor.feature_name(index).is_empty());
        }
    }

    #[test]
    fn move_features_are_rotation_swap_invariant() {
        // A move and its rotated-colour-swapped counterpart must have
        // identical features (mover perspective).
        let game = ForwardChess::new(Ruleset::Small);
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(43);
        let mut extractor = FcMoveFeatures::new(&game);
        let mut moves = Vec::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let total = game.cell_count() as u16;
        let mut tested = 0;
        for _ in 0..25 {
            let mut state = game.initial_state();
            loop {
                if game.outcome(&state).is_some() {
                    break;
                }
                let rotated = rotate_swap(&game, &state);
                game.legal_moves(&state, &mut moves);
                let move_list = moves.clone();
                for &mv in &move_list {
                    let rotated_mv = crate::games::forward_chess::FcMove {
                        from: total - 1 - mv.from,
                        to: total - 1 - mv.to,
                        promotion: mv.promotion,
                    };
                    extractor.extract(&game, &state, mv, &mut a);
                    extractor.extract(&game, &rotated, rotated_mv, &mut b);
                    assert_eq!(a, b, "move features must be rotation-swap invariant");
                    tested += 1;
                }
                let mv = move_list[rng.gen_range(0..move_list.len())];
                game.make_move(&mut state, mv);
            }
        }
        assert!(tested > 300);
    }

    #[test]
    fn hand_computed_move_features() {
        // Same position as `hand_computed_position`; White to move.
        let game = ForwardChess::new(Ruleset::Small);
        let state = game.custom_state(
            &[
                ("b1", Player::One, Piece::King, false),
                ("c1", Player::One, Piece::Rook, false),
                ("a2", Player::One, Piece::Pawn, false),
                ("d4", Player::Two, Piece::King, false),
                ("b3", Player::Two, Piece::Rook, true),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let mut extractor = FcMoveFeatures::new(&game);
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        let axb3 = moves
            .iter()
            .copied()
            .find(|m| m.from == game.square("a2") && m.to == game.square("b3"))
            .expect("axb3 is legal");
        let mut x = Vec::new();
        extractor.extract(&game, &state, axb3, &mut x);
        let get = |i: usize| {
            x.iter()
                .find(|&&(f, _)| f as usize == i)
                .map(|&(_, v)| v)
                .unwrap_or(0.0)
        };
        assert_eq!(get(MF_MOVING), 1.0, "moving pawn-nat");
        assert_eq!(get(MF_CAPTURED + 8), 1.0, "captures rook-rev (combo 8)");
        assert_eq!(get(MF_IS_CAPTURE), 1.0);
        assert_eq!(get(MF_FORWARD_DISP), 1.0, "one rank forward");
        assert_eq!(get(MF_DEST_REL_RANK), 2.0, "b3 is rank 3 = rel rank 2");
        assert_eq!(get(MF_DEST_ATTACKED), 0.0, "b3 undefended by black");
        assert_eq!(get(MF_SOURCE_ATTACKED), 0.0);
        // After axb3 the pawn on b3 attacks a4 and c4 - not the king
        // on d4; no check.
        assert_eq!(get(MF_GIVES_CHECK), 0.0);
        // Rook lift c1-c4 gives check? White rook on c4 attacks
        // horizontally d4 = Black king: yes.
        let rook_c4 = moves
            .iter()
            .copied()
            .find(|m| m.from == game.square("c1") && m.to == game.square("c4"))
            .expect("Rc4 is legal");
        extractor.extract(&game, &state, rook_c4, &mut x);
        let get = |i: usize| {
            x.iter()
                .find(|&&(f, _)| f as usize == i)
                .map(|&(_, v)| v)
                .unwrap_or(0.0)
        };
        assert_eq!(get(MF_GIVES_CHECK), 1.0, "Rc4+ through the empty file");
        assert_eq!(get(MF_MOVING + 3), 1.0, "moving rook-nat");
        assert_eq!(get(MF_IS_CAPTURE), 0.0);
        // Every emitted index has a name.
        for &(index, _) in &x {
            assert!(!FcMoveFeatures::feature_name(index).is_empty());
        }
    }

    #[test]
    fn v0_is_the_documented_subset() {
        let game = ForwardChess::new(Ruleset::Tiny);
        let state = game.initial_state();
        let v0 = extract_all(&game, FcRecipe::FcCountsV0, &state);
        let extractor = FcExtractor::new(&game, FcRecipe::FcCountsV0);
        assert_eq!(extractor.dimension(), 10);
        // Initial position is symmetric: all count diffs zero, only
        // total material remains.
        assert_eq!(v0, vec![(9, 2.0)]);
    }
}

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

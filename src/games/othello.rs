//! Parameterized Othello (Reversi) on a `width` x `height` board, both
//! even and at least 4.
//!
//! A placement must bracket at least one contiguous line of opponent
//! discs (in any of the 8 directions) between the new disc and an own
//! disc; every bracketed line flips. A player with no legal placement
//! must pass — the pass is an explicit move, so turns still alternate
//! (the search's negamax perspective flip stays valid). The game ends
//! when neither player has a placement (including a full board); the
//! side with more discs wins, equal discs draw.
//!
//! Passes never repeat positions: two consecutive passes would mean the
//! game is over, so the pass move is only legal while the opponent
//! still has a placement, and discs are otherwise only added.
//!
//! Features encode board occupancy from the side to move's perspective
//! only — no corner, mobility, stability, frontier, or parity concepts
//! (plan §25 restrictions).
//!
//! Cells are indexed `cell = y * width + x`.

use crate::game::{ActionId, FeatureId, Game, Outcome, Player};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

/// Fixed seed for Zobrist key generation so position keys are stable
/// across processes and runs.
const ZOBRIST_SEED: u64 = 0x07e1_1006_0a11_5eed;

const EMPTY: u8 = 0;
/// Upper bound on discs flipped by one move (8 rays of at most 6
/// interior discs on boards up to 8x8).
const MAX_FLIPS: usize = 48;

fn cell_code(player: Player) -> u8 {
    match player {
        Player::One => 1,
        Player::Two => 2,
    }
}

/// Rules object for one Othello parameterization.
pub struct Othello {
    width: u16,
    height: u16,
    /// `zobrist[cell][player_index]`
    zobrist: Vec<[u64; 2]>,
    zobrist_side: u64,
}

/// A move: place a disc on `cell`, or pass (only legal when no
/// placement exists).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum OthelloMove {
    Place(u16),
    Pass,
}

/// Board state. `cells[cell]` is `0` (empty), `1` (Player One) or
/// `2` (Player Two).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OthelloState {
    cells: Vec<u8>,
    to_move: Player,
    key: u64,
    discs: [u16; 2],
    outcome: Option<Outcome>,
}

/// Undo record: the flipped cells (restored to the opponent) plus the
/// previous cached outcome.
pub struct OthelloUndo {
    flipped: [u16; MAX_FLIPS],
    flip_count: u8,
    prev_outcome: Option<Outcome>,
}

const DIRECTIONS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

impl Othello {
    pub fn new(width: u16, height: u16) -> Result<Othello, String> {
        if width < 4 || height < 4 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(format!(
                "othello needs even dimensions of at least 4, got {width}x{height}"
            ));
        }
        let cells = usize::from(width) * usize::from(height);
        let mut rng = ChaCha12Rng::seed_from_u64(ZOBRIST_SEED);
        let zobrist = (0..cells)
            .map(|_| [rng.gen::<u64>(), rng.gen::<u64>()])
            .collect();
        let zobrist_side = rng.gen::<u64>();
        Ok(Othello {
            width,
            height,
            zobrist,
            zobrist_side,
        })
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn cell_count(&self) -> u32 {
        u32::from(self.width) * u32::from(self.height)
    }

    /// 2 features per cell: own disc and opponent disc.
    pub fn feature_count(&self) -> usize {
        2 * self.cell_count() as usize
    }

    /// One action per cell plus the pass action (ID `cell_count`).
    pub fn action_count(&self) -> usize {
        self.cell_count() as usize + 1
    }

    fn compute_key(&self, cells: &[u8], to_move: Player) -> u64 {
        let mut key = 0u64;
        for (cell, &code) in cells.iter().enumerate() {
            if code != EMPTY {
                key ^= self.zobrist[cell][usize::from(code) - 1];
            }
        }
        if to_move == Player::Two {
            key ^= self.zobrist_side;
        }
        key
    }

    /// Walk the ray from `cell` in direction `(dx, dy)`: if it holds one
    /// or more `opponent` discs immediately followed by a `player` disc,
    /// call `flip` for each bracketed cell and return true.
    fn bracket_ray(
        &self,
        cells: &[u8],
        cell: u16,
        (dx, dy): (i32, i32),
        player: Player,
        mut flip: impl FnMut(u16),
    ) -> bool {
        let w = i32::from(self.width);
        let h = i32::from(self.height);
        let own = cell_code(player);
        let opp = cell_code(player.opponent());
        let mut x = i32::from(cell % self.width) + dx;
        let mut y = i32::from(cell / self.width) + dy;
        let mut seen = 0u32;
        while x >= 0 && x < w && y >= 0 && y < h {
            let code = cells[(y * w + x) as usize];
            if code == opp {
                seen += 1;
            } else if code == own {
                if seen > 0 {
                    let mut fx = i32::from(cell % self.width) + dx;
                    let mut fy = i32::from(cell / self.width) + dy;
                    for _ in 0..seen {
                        flip((fy * w + fx) as u16);
                        fx += dx;
                        fy += dy;
                    }
                    return true;
                }
                return false;
            } else {
                return false;
            }
            x += dx;
            y += dy;
        }
        false
    }

    /// Would placing on `cell` (which must be empty) flip anything?
    fn placement_is_legal(&self, cells: &[u8], cell: u16, player: Player) -> bool {
        DIRECTIONS
            .iter()
            .any(|&dir| self.bracket_ray(cells, cell, dir, player, |_| {}))
    }

    fn has_placement(&self, cells: &[u8], player: Player) -> bool {
        (0..self.cell_count() as u16).any(|cell| {
            cells[usize::from(cell)] == EMPTY && self.placement_is_legal(cells, cell, player)
        })
    }

    fn count_outcome(&self, discs: [u16; 2]) -> Outcome {
        match discs[0].cmp(&discs[1]) {
            std::cmp::Ordering::Greater => Outcome::Win(Player::One),
            std::cmp::Ordering::Less => Outcome::Win(Player::Two),
            std::cmp::Ordering::Equal => Outcome::Draw,
        }
    }

    /// Build an arbitrary position (tests and diagnostics).
    pub fn custom_state(
        &self,
        discs: &[(u16, Player)],
        to_move: Player,
    ) -> Result<OthelloState, String> {
        let mut cells = vec![EMPTY; self.cell_count() as usize];
        let mut counts = [0u16; 2];
        for &(cell, player) in discs {
            if cell >= self.cell_count() as u16 {
                return Err(format!("cell {cell} out of bounds"));
            }
            if cells[usize::from(cell)] != EMPTY {
                return Err(format!("cell {cell} occupied twice"));
            }
            cells[usize::from(cell)] = cell_code(player);
            counts[usize::from(cell_code(player)) - 1] += 1;
        }
        let outcome = if !self.has_placement(&cells, Player::One)
            && !self.has_placement(&cells, Player::Two)
        {
            Some(self.count_outcome(counts))
        } else {
            None
        };
        Ok(OthelloState {
            key: self.compute_key(&cells, to_move),
            cells,
            to_move,
            discs: counts,
            outcome,
        })
    }
}

impl Game for Othello {
    type State = OthelloState;
    type Move = OthelloMove;
    type Undo = OthelloUndo;

    fn initial_state(&self) -> OthelloState {
        let (cx, cy) = (self.width / 2, self.height / 2);
        let at = |x: u16, y: u16| y * self.width + x;
        // Standard central pattern: the first player's discs on the
        // anti-diagonal.
        self.custom_state(
            &[
                (at(cx - 1, cy), Player::One),
                (at(cx, cy - 1), Player::One),
                (at(cx - 1, cy - 1), Player::Two),
                (at(cx, cy), Player::Two),
            ],
            Player::One,
        )
        .expect("initial position is valid")
    }

    fn side_to_move(&self, state: &OthelloState) -> Player {
        state.to_move
    }

    fn legal_moves(&self, state: &OthelloState, moves: &mut Vec<OthelloMove>) {
        moves.clear();
        for cell in 0..self.cell_count() as u16 {
            if state.cells[usize::from(cell)] == EMPTY
                && self.placement_is_legal(&state.cells, cell, state.to_move)
            {
                moves.push(OthelloMove::Place(cell));
            }
        }
        if moves.is_empty() {
            // Only reached in non-terminal states, i.e. the opponent
            // still has a placement: the mover must pass.
            moves.push(OthelloMove::Pass);
        }
    }

    fn make_move(&self, state: &mut OthelloState, mv: OthelloMove) -> OthelloUndo {
        let mover = state.to_move;
        let mover_code = cell_code(mover);
        let mut undo = OthelloUndo {
            flipped: [0; MAX_FLIPS],
            flip_count: 0,
            prev_outcome: state.outcome,
        };
        if let OthelloMove::Place(cell) = mv {
            debug_assert_eq!(state.cells[usize::from(cell)], EMPTY);
            let mut count = 0usize;
            for &dir in &DIRECTIONS {
                let flipped = &mut undo.flipped;
                self.bracket_ray(&state.cells, cell, dir, mover, |flip| {
                    flipped[count] = flip;
                    count += 1;
                });
            }
            debug_assert!(count > 0, "placement must flip");
            debug_assert!(count <= MAX_FLIPS);
            state.cells[usize::from(cell)] = mover_code;
            state.key ^= self.zobrist[usize::from(cell)][usize::from(mover_code) - 1];
            for &flip in &undo.flipped[..count] {
                let old = state.cells[usize::from(flip)];
                state.key ^= self.zobrist[usize::from(flip)][usize::from(old) - 1];
                state.cells[usize::from(flip)] = mover_code;
                state.key ^= self.zobrist[usize::from(flip)][usize::from(mover_code) - 1];
            }
            state.discs[usize::from(mover_code) - 1] += 1 + count as u16;
            state.discs[2 - usize::from(mover_code)] -= count as u16;
            undo.flip_count = count as u8;
        }
        state.to_move = mover.opponent();
        state.key ^= self.zobrist_side;
        state.outcome = if !self.has_placement(&state.cells, Player::One)
            && !self.has_placement(&state.cells, Player::Two)
        {
            Some(self.count_outcome(state.discs))
        } else {
            None
        };
        undo
    }

    fn unmake_move(&self, state: &mut OthelloState, mv: OthelloMove, undo: OthelloUndo) {
        let mover = state.to_move.opponent();
        let mover_code = cell_code(mover);
        let opp_code = cell_code(mover.opponent());
        state.key ^= self.zobrist_side;
        state.to_move = mover;
        if let OthelloMove::Place(cell) = mv {
            for &flip in &undo.flipped[..usize::from(undo.flip_count)] {
                state.key ^= self.zobrist[usize::from(flip)][usize::from(mover_code) - 1];
                state.cells[usize::from(flip)] = opp_code;
                state.key ^= self.zobrist[usize::from(flip)][usize::from(opp_code) - 1];
            }
            state.cells[usize::from(cell)] = EMPTY;
            state.key ^= self.zobrist[usize::from(cell)][usize::from(mover_code) - 1];
            let flips = u16::from(undo.flip_count);
            state.discs[usize::from(mover_code) - 1] -= flips + 1;
            state.discs[usize::from(opp_code) - 1] += flips;
        }
        state.outcome = undo.prev_outcome;
    }

    fn outcome(&self, state: &OthelloState) -> Option<Outcome> {
        state.outcome
    }

    fn position_key(&self, state: &OthelloState) -> u64 {
        state.key
    }

    fn encode_features(&self, state: &OthelloState, features: &mut Vec<FeatureId>) {
        features.clear();
        let own = cell_code(state.to_move);
        for (cell, &code) in state.cells.iter().enumerate() {
            if code != EMPTY {
                let relative = u32::from(code != own);
                features.push(2 * cell as FeatureId + relative);
            }
        }
    }

    fn action_id(&self, _state: &OthelloState, mv: OthelloMove) -> ActionId {
        match mv {
            OthelloMove::Place(cell) => ActionId::from(cell),
            OthelloMove::Pass => self.cell_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(w: u16, h: u16) -> Othello {
        Othello::new(w, h).unwrap()
    }

    /// Independent slow reference: a placement on an empty cell is legal
    /// iff scanning outward in some direction meets 1+ opponent discs
    /// then an own disc; passes are legal exactly when no placement is.
    fn reference_moves(g: &Othello, state: &OthelloState) -> Vec<OthelloMove> {
        let w = i32::from(g.width);
        let h = i32::from(g.height);
        let own = cell_code(state.to_move);
        let opp = cell_code(state.to_move.opponent());
        let mut moves = Vec::new();
        for cell in 0..g.cell_count() as u16 {
            if state.cells[usize::from(cell)] != EMPTY {
                continue;
            }
            let (cx, cy) = (i32::from(cell % g.width), i32::from(cell / g.width));
            let mut legal = false;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let mut steps = 1;
                    let mut bracketed = 0;
                    loop {
                        let (x, y) = (cx + dx * steps, cy + dy * steps);
                        if x < 0 || x >= w || y < 0 || y >= h {
                            break;
                        }
                        let code = state.cells[(y * w + x) as usize];
                        if code == opp {
                            bracketed += 1;
                        } else {
                            if code == own && bracketed > 0 {
                                legal = true;
                            }
                            break;
                        }
                        steps += 1;
                    }
                }
            }
            if legal {
                moves.push(OthelloMove::Place(cell));
            }
        }
        if moves.is_empty() {
            moves.push(OthelloMove::Pass);
        }
        moves
    }

    fn random_playout_states(g: &Othello, seed: u64, max_states: usize) -> Vec<OthelloState> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut states = Vec::new();
        'outer: while states.len() < max_states {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            loop {
                if g.outcome(&state).is_some() {
                    continue 'outer;
                }
                states.push(state.clone());
                if states.len() >= max_states {
                    break 'outer;
                }
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                g.make_move(&mut state, mv);
            }
        }
        states
    }

    #[test]
    fn fast_movegen_matches_slow_reference() {
        for (w, h) in [(4, 4), (6, 4), (6, 6)] {
            let g = game(w, h);
            for state in random_playout_states(&g, 0x07e1 + u64::from(w), 300) {
                let mut fast = Vec::new();
                g.legal_moves(&state, &mut fast);
                assert_eq!(fast, reference_moves(&g, &state));
            }
        }
    }

    #[test]
    fn make_unmake_restores_state_exactly() {
        let g = game(6, 6);
        let mut rng = ChaCha12Rng::seed_from_u64(0x0f11);
        for _ in 0..100 {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            while g.outcome(&state).is_none() {
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                let before = state.clone();
                let undo = g.make_move(&mut state, mv);
                g.unmake_move(&mut state, mv, undo);
                assert_eq!(state, before);
                g.make_move(&mut state, mv);
            }
        }
    }

    #[test]
    fn recomputed_key_and_disc_counts_match() {
        let g = game(4, 4);
        for state in random_playout_states(&g, 0x2b, 400) {
            assert_eq!(state.key, g.compute_key(&state.cells, state.to_move));
            let ones = state.cells.iter().filter(|&&c| c == 1).count() as u16;
            let twos = state.cells.iter().filter(|&&c| c == 2).count() as u16;
            assert_eq!(state.discs, [ones, twos]);
        }
    }

    #[test]
    fn colour_swap_encodes_identically() {
        let g = game(6, 6);
        for state in random_playout_states(&g, 0x51a9, 200) {
            let swapped_discs: Vec<(u16, Player)> = state
                .cells
                .iter()
                .enumerate()
                .filter(|(_, &c)| c != EMPTY)
                .map(|(cell, &c)| (cell as u16, if c == 1 { Player::Two } else { Player::One }))
                .collect();
            let swapped = g
                .custom_state(&swapped_discs, state.to_move.opponent())
                .unwrap();
            let (mut fa, mut fb) = (Vec::new(), Vec::new());
            g.encode_features(&state, &mut fa);
            g.encode_features(&swapped, &mut fb);
            assert_eq!(fa, fb, "colour-swapped features must match");
            let mut ma = Vec::new();
            let mut mb = Vec::new();
            g.legal_moves(&state, &mut ma);
            g.legal_moves(&swapped, &mut mb);
            assert_eq!(ma, mb, "colour-swapped legal moves must match");
        }
    }

    #[test]
    fn flips_capture_multiple_directions_at_once() {
        let g = game(4, 4);
        // P1 discs at 0, 2, 8; P2 at 1, 5; placing P1 at 4? Construct:
        // place at cell 5's neighbourhood. Simpler canonical: P1 to move
        // places at cell 0 bracketing east (1) toward 2 and south (4)
        // toward 8.
        let state = g
            .custom_state(
                &[
                    (1, Player::Two),
                    (2, Player::One),
                    (4, Player::Two),
                    (8, Player::One),
                    (5, Player::Two),
                    (10, Player::One),
                ],
                Player::One,
            )
            .unwrap();
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        assert!(moves.contains(&OthelloMove::Place(0)));
        let mut s = state.clone();
        g.make_move(&mut s, OthelloMove::Place(0));
        // East (1), south (4), and diagonal (5) lines all flip.
        assert_eq!(s.cells[1], 1);
        assert_eq!(s.cells[4], 1);
        assert_eq!(s.cells[5], 1);
        assert_eq!(s.discs, [7, 0]);
    }

    #[test]
    fn pass_mechanics_and_disc_count_outcome() {
        let g = game(4, 4);
        // Neither side can place (isolated corner discs with no
        // bracketable runs): terminal immediately, decided by discs.
        let state = g
            .custom_state(&[(0, Player::Two), (15, Player::One)], Player::Two)
            .unwrap();
        assert_eq!(g.outcome(&state), Some(Outcome::Draw));
        let state = g
            .custom_state(
                &[(0, Player::Two), (3, Player::Two), (15, Player::One)],
                Player::One,
            )
            .unwrap();
        assert_eq!(g.outcome(&state), Some(Outcome::Win(Player::Two)));

        // Find forced passes in real play and verify their mechanics:
        // the only legal move is Pass, the board is unchanged, the side
        // flips, the opponent then has a placement, and unmake restores.
        let mut rng = ChaCha12Rng::seed_from_u64(0x9a55);
        let mut passes_seen = 0;
        for _ in 0..400 {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            while g.outcome(&state).is_none() {
                g.legal_moves(&state, &mut moves);
                if moves == vec![OthelloMove::Pass] {
                    passes_seen += 1;
                    let before = state.clone();
                    let undo = g.make_move(&mut state, OthelloMove::Pass);
                    assert_eq!(state.cells, before.cells, "pass must not move discs");
                    assert_eq!(state.to_move, before.to_move.opponent());
                    assert!(g.outcome(&state).is_none());
                    let mut opponent_moves = Vec::new();
                    g.legal_moves(&state, &mut opponent_moves);
                    assert_ne!(
                        opponent_moves,
                        vec![OthelloMove::Pass],
                        "a pass is only legal while the opponent can place"
                    );
                    g.unmake_move(&mut state, OthelloMove::Pass, undo);
                    assert_eq!(state, before);
                }
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                g.make_move(&mut state, mv);
            }
        }
        assert!(passes_seen > 10, "random 4x4 play should hit forced passes");
    }

    #[test]
    fn games_terminate_with_correct_material_outcome() {
        let g = game(4, 4);
        let mut rng = ChaCha12Rng::seed_from_u64(0x600d);
        for _ in 0..300 {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            let mut plies = 0;
            let outcome = loop {
                if let Some(outcome) = g.outcome(&state) {
                    break outcome;
                }
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                g.make_move(&mut state, mv);
                plies += 1;
                assert!(plies < 100, "othello must terminate");
            };
            let expected = match state.discs[0].cmp(&state.discs[1]) {
                std::cmp::Ordering::Greater => Outcome::Win(Player::One),
                std::cmp::Ordering::Less => Outcome::Win(Player::Two),
                std::cmp::Ordering::Equal => Outcome::Draw,
            };
            assert_eq!(outcome, expected);
        }
    }
}

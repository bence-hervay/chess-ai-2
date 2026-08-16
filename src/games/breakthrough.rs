//! Parameterized Breakthrough on a `width` x `height` board.
//!
//! Each player starts with `rows` full ranks of pawns (Player One at the
//! bottom moving up, Player Two at the top moving down). A pawn moves one
//! square straight or diagonally forward to an empty square; it may move
//! diagonally forward onto an enemy pawn, capturing it. Straight moves
//! never capture.
//!
//! Terminal rule (selected ruleset, recorded here): a player wins by
//! moving a pawn onto the opponent's home rank; a player with no legal
//! move (including having no pawns) loses. Pawns only advance, so no
//! position can repeat and every game terminates; there are no draws.
//!
//! Cells are indexed `cell = y * width + x`, with row `y = 0` at the
//! bottom (Player One's home rank).

use crate::game::{ActionId, FeatureId, Game, Outcome, Player};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

/// Fixed seed for Zobrist key generation so position keys are stable
/// across processes and runs.
const ZOBRIST_SEED: u64 = 0xb4ea_c704_9a11_5eed;

const EMPTY: u8 = 0;

fn cell_code(player: Player) -> u8 {
    match player {
        Player::One => 1,
        Player::Two => 2,
    }
}

fn player_index(player: Player) -> usize {
    match player {
        Player::One => 0,
        Player::Two => 1,
    }
}

/// Rules object for one Breakthrough parameterization.
pub struct Breakthrough {
    width: u16,
    height: u16,
    rows: u16,
    /// `zobrist[cell][player_index]`
    zobrist: Vec<[u64; 2]>,
    zobrist_side: u64,
}

/// A move: source and destination cell indices.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BreakthroughMove {
    pub from: u16,
    pub to: u16,
}

/// Board state. `cells[cell]` is `0` (empty), `1` (Player One) or
/// `2` (Player Two).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BreakthroughState {
    cells: Vec<u8>,
    to_move: Player,
    key: u64,
    /// Live pawn counts per player index.
    pawns: [u16; 2],
    outcome: Option<Outcome>,
}

/// Undo record for `unmake_move`.
pub struct BreakthroughUndo {
    captured: u8,
    prev_outcome: Option<Outcome>,
}

impl Breakthrough {
    pub fn new(width: u16, height: u16, rows: u16) -> Result<Breakthrough, String> {
        if width < 2 {
            return Err(format!("width {width} < 2 makes diagonal play degenerate"));
        }
        if rows == 0 {
            return Err("rows must be at least 1".into());
        }
        if 2 * rows >= height {
            return Err(format!(
                "{rows} pawn rows per side need at least {} ranks, board has {height}",
                2 * rows + 1
            ));
        }
        let cells = usize::from(width) * usize::from(height);
        let mut rng = ChaCha12Rng::seed_from_u64(ZOBRIST_SEED);
        let zobrist = (0..cells)
            .map(|_| [rng.gen::<u64>(), rng.gen::<u64>()])
            .collect();
        let zobrist_side = rng.gen::<u64>();
        Ok(Breakthrough {
            width,
            height,
            rows,
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

    /// Total distinct feature IDs (2 per cell: own-pawn and opponent-pawn,
    /// in the side-to-move's mirrored frame).
    pub fn feature_count(&self) -> usize {
        2 * self.cell_count() as usize
    }

    /// Total distinct action IDs (3 forward directions per source cell,
    /// in the side-to-move's mirrored frame).
    pub fn action_count(&self) -> usize {
        3 * self.cell_count() as usize
    }

    /// Build an arbitrary position (tests and diagnostics). Validates
    /// pawn placement (no pawn on the opponent's home rank, since such a
    /// position would already be terminal) and computes the derived
    /// fields, including the no-move loss rule.
    pub fn custom_state(
        &self,
        pawns: &[(u16, Player)],
        to_move: Player,
    ) -> Result<BreakthroughState, String> {
        let mut cells = vec![EMPTY; self.cell_count() as usize];
        let mut counts = [0u16; 2];
        for &(cell, player) in pawns {
            if cell >= self.cell_count() as u16 {
                return Err(format!("cell {cell} out of bounds"));
            }
            let y = cell / self.width;
            if (player == Player::One && y == self.height - 1) || (player == Player::Two && y == 0)
            {
                return Err(format!("pawn on opponent home rank at cell {cell}"));
            }
            if cells[usize::from(cell)] != EMPTY {
                return Err(format!("cell {cell} occupied twice"));
            }
            cells[usize::from(cell)] = cell_code(player);
            counts[player_index(player)] += 1;
        }
        let mut state = BreakthroughState {
            key: self.compute_key(&cells, to_move),
            cells,
            to_move,
            pawns: counts,
            outcome: None,
        };
        state.outcome = if !self.has_any_move(&state) {
            Some(Outcome::Win(to_move.opponent()))
        } else {
            None
        };
        Ok(state)
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

    fn forward_dy(player: Player) -> i32 {
        match player {
            Player::One => 1,
            Player::Two => -1,
        }
    }

    /// Iterate the (up to three) forward destinations of `from` for
    /// `player`, in dx order -1, 0, +1.
    fn destinations(&self, from: u16, player: Player) -> impl Iterator<Item = (u16, bool)> + '_ {
        let w = i32::from(self.width);
        let h = i32::from(self.height);
        let x = i32::from(from % self.width);
        let y = i32::from(from / self.width);
        let dy = Self::forward_dy(player);
        [-1i32, 0, 1].into_iter().filter_map(move |dx| {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || nx >= w || ny < 0 || ny >= h {
                None
            } else {
                Some(((ny * w + nx) as u16, dx == 0))
            }
        })
    }

    fn move_is_open(&self, cells: &[u8], player: Player, to: u16, straight: bool) -> bool {
        let target = cells[usize::from(to)];
        if straight {
            target == EMPTY
        } else {
            target != cell_code(player)
        }
    }

    /// Does `state.to_move` have at least one legal move?
    fn has_any_move(&self, state: &BreakthroughState) -> bool {
        let own = cell_code(state.to_move);
        for (cell, &code) in state.cells.iter().enumerate() {
            if code == own {
                for (to, straight) in self.destinations(cell as u16, state.to_move) {
                    if self.move_is_open(&state.cells, state.to_move, to, straight) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Cell index in the side-to-move frame: Player Two sees the board
    /// mirrored vertically so "forward" is always +y.
    fn perspective_cell(&self, cell: u16, player: Player) -> u16 {
        match player {
            Player::One => cell,
            Player::Two => {
                let x = cell % self.width;
                let y = cell / self.width;
                (self.height - 1 - y) * self.width + x
            }
        }
    }
}

impl Game for Breakthrough {
    type State = BreakthroughState;
    type Move = BreakthroughMove;
    type Undo = BreakthroughUndo;

    fn initial_state(&self) -> BreakthroughState {
        let mut pawns = Vec::new();
        for y in 0..self.rows {
            for x in 0..self.width {
                pawns.push((y * self.width + x, Player::One));
                pawns.push(((self.height - 1 - y) * self.width + x, Player::Two));
            }
        }
        self.custom_state(&pawns, Player::One)
            .expect("initial position is valid")
    }

    fn side_to_move(&self, state: &BreakthroughState) -> Player {
        state.to_move
    }

    fn legal_moves(&self, state: &BreakthroughState, moves: &mut Vec<BreakthroughMove>) {
        moves.clear();
        let own = cell_code(state.to_move);
        for (cell, &code) in state.cells.iter().enumerate() {
            if code == own {
                let from = cell as u16;
                for (to, straight) in self.destinations(from, state.to_move) {
                    if self.move_is_open(&state.cells, state.to_move, to, straight) {
                        moves.push(BreakthroughMove { from, to });
                    }
                }
            }
        }
    }

    fn make_move(&self, state: &mut BreakthroughState, mv: BreakthroughMove) -> BreakthroughUndo {
        let mover = state.to_move;
        let mover_code = cell_code(mover);
        debug_assert_eq!(state.cells[usize::from(mv.from)], mover_code);
        let captured = state.cells[usize::from(mv.to)];
        debug_assert_ne!(captured, mover_code);
        let undo = BreakthroughUndo {
            captured,
            prev_outcome: state.outcome,
        };

        state.cells[usize::from(mv.from)] = EMPTY;
        state.key ^= self.zobrist[usize::from(mv.from)][usize::from(mover_code) - 1];
        if captured != EMPTY {
            state.key ^= self.zobrist[usize::from(mv.to)][usize::from(captured) - 1];
            state.pawns[player_index(mover.opponent())] -= 1;
        }
        state.cells[usize::from(mv.to)] = mover_code;
        state.key ^= self.zobrist[usize::from(mv.to)][usize::from(mover_code) - 1];
        state.to_move = mover.opponent();
        state.key ^= self.zobrist_side;

        let to_row = mv.to / self.width;
        let reached_home = match mover {
            Player::One => to_row == self.height - 1,
            Player::Two => to_row == 0,
        };
        state.outcome = if reached_home || !self.has_any_move(state) {
            // Reaching the opponent's home rank wins; otherwise the new
            // side to move loses if it has no legal move (this subsumes
            // elimination of all its pawns).
            Some(Outcome::Win(mover))
        } else {
            None
        };
        undo
    }

    fn unmake_move(
        &self,
        state: &mut BreakthroughState,
        mv: BreakthroughMove,
        undo: BreakthroughUndo,
    ) {
        let mover = state.to_move.opponent();
        let mover_code = cell_code(mover);
        state.key ^= self.zobrist_side;
        state.to_move = mover;
        state.key ^= self.zobrist[usize::from(mv.to)][usize::from(mover_code) - 1];
        state.cells[usize::from(mv.to)] = undo.captured;
        if undo.captured != EMPTY {
            state.key ^= self.zobrist[usize::from(mv.to)][usize::from(undo.captured) - 1];
            state.pawns[player_index(mover.opponent())] += 1;
        }
        state.cells[usize::from(mv.from)] = mover_code;
        state.key ^= self.zobrist[usize::from(mv.from)][usize::from(mover_code) - 1];
        state.outcome = undo.prev_outcome;
    }

    fn outcome(&self, state: &BreakthroughState) -> Option<Outcome> {
        state.outcome
    }

    fn position_key(&self, state: &BreakthroughState) -> u64 {
        state.key
    }

    fn encode_features(&self, state: &BreakthroughState, features: &mut Vec<FeatureId>) {
        features.clear();
        let own = cell_code(state.to_move);
        for (cell, &code) in state.cells.iter().enumerate() {
            if code != EMPTY {
                let relative = u32::from(code != own);
                let pcell = self.perspective_cell(cell as u16, state.to_move);
                features.push(2 * FeatureId::from(pcell) + relative);
            }
        }
        features.sort_unstable();
    }

    fn action_id(&self, state: &BreakthroughState, mv: BreakthroughMove) -> ActionId {
        let from = self.perspective_cell(mv.from, state.to_move);
        let to = self.perspective_cell(mv.to, state.to_move);
        let dx = i32::from(to % self.width) - i32::from(from % self.width);
        debug_assert_eq!(to / self.width, from / self.width + 1);
        let direction = (dx + 1) as u32; // 0 = forward-left, 1 = straight, 2 = forward-right
        3 * ActionId::from(from) + direction
    }

    fn is_tactical(&self, state: &BreakthroughState, mv: BreakthroughMove) -> bool {
        // Only diagonal moves may capture; the destination is enemy-
        // occupied exactly when this move captures.
        state.cells[usize::from(mv.to)] != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(w: u16, h: u16, rows: u16) -> Breakthrough {
        Breakthrough::new(w, h, rows).unwrap()
    }

    /// Slow reference move generator: test every (from, to) pair against
    /// a rule predicate written independently of the fast generator.
    fn reference_moves(g: &Breakthrough, state: &BreakthroughState) -> Vec<BreakthroughMove> {
        let mut moves = Vec::new();
        let own = cell_code(state.to_move);
        let w = i32::from(g.width);
        for from in 0..g.cell_count() as u16 {
            if state.cells[usize::from(from)] != own {
                continue;
            }
            for to in 0..g.cell_count() as u16 {
                let (fx, fy) = (i32::from(from) % w, i32::from(from) / w);
                let (tx, ty) = (i32::from(to) % w, i32::from(to) / w);
                let dy = ty - fy;
                let dx = tx - fx;
                if dy != Breakthrough::forward_dy(state.to_move) || dx.abs() > 1 {
                    continue;
                }
                let target = state.cells[usize::from(to)];
                let legal = if dx == 0 {
                    target == EMPTY
                } else {
                    target != own
                };
                if legal {
                    moves.push(BreakthroughMove { from, to });
                }
            }
        }
        moves.sort();
        moves
    }

    fn random_playout_states(
        g: &Breakthrough,
        seed: u64,
        max_states: usize,
    ) -> Vec<BreakthroughState> {
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
        for (w, h, rows) in [(2, 4, 1), (3, 5, 1), (4, 6, 2), (5, 7, 2)] {
            let g = game(w, h, rows);
            for state in random_playout_states(&g, 0xb42e + u64::from(w), 300) {
                let mut fast = Vec::new();
                g.legal_moves(&state, &mut fast);
                let mut fast_sorted = fast.clone();
                fast_sorted.sort();
                assert_eq!(fast_sorted, reference_moves(&g, &state));
                assert!(!fast.is_empty(), "non-terminal state must have moves");
            }
        }
    }

    #[test]
    fn make_unmake_restores_state_exactly() {
        let g = game(4, 6, 2);
        let mut rng = ChaCha12Rng::seed_from_u64(0x77);
        for _ in 0..200 {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            while g.outcome(&state).is_none() {
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                let before = state.clone();
                let undo = g.make_move(&mut state, mv);
                g.unmake_move(&mut state, mv, undo);
                assert_eq!(state, before);
                assert_eq!(g.position_key(&state), g.position_key(&before));
                let undo = g.make_move(&mut state, mv);
                let _ = undo;
            }
        }
    }

    #[test]
    fn recomputed_key_matches_incremental() {
        let g = game(3, 5, 1);
        for state in random_playout_states(&g, 0x1c3, 400) {
            assert_eq!(state.key, g.compute_key(&state.cells, state.to_move));
        }
    }

    /// Colour reflection: mirroring the board vertically and swapping
    /// colours and side to move yields a rules-equivalent position with
    /// identical feature encoding and action IDs.
    #[test]
    fn colour_reflection_encodes_identically() {
        let g = game(4, 6, 2);
        for state in random_playout_states(&g, 0x5ca1e, 200) {
            let reflected_pawns: Vec<(u16, Player)> = state
                .cells
                .iter()
                .enumerate()
                .filter(|(_, &c)| c != EMPTY)
                .map(|(cell, &c)| {
                    let mirrored = g.perspective_cell(cell as u16, Player::Two);
                    let owner = if c == 1 { Player::Two } else { Player::One };
                    (mirrored, owner)
                })
                .collect();
            let reflected = g
                .custom_state(&reflected_pawns, state.to_move.opponent())
                .unwrap();

            let (mut fa, mut fb) = (Vec::new(), Vec::new());
            g.encode_features(&state, &mut fa);
            g.encode_features(&reflected, &mut fb);
            assert_eq!(fa, fb, "reflected features must match");

            let mut ma = Vec::new();
            let mut mb = Vec::new();
            g.legal_moves(&state, &mut ma);
            g.legal_moves(&reflected, &mut mb);
            let mut ids_a: Vec<ActionId> = ma.iter().map(|&m| g.action_id(&state, m)).collect();
            let mut ids_b: Vec<ActionId> = mb.iter().map(|&m| g.action_id(&reflected, m)).collect();
            ids_a.sort_unstable();
            ids_b.sort_unstable();
            assert_eq!(ids_a, ids_b, "reflected action IDs must match");
            assert_eq!(g.outcome(&state).is_some(), g.outcome(&reflected).is_some());
        }
    }

    #[test]
    fn terminal_races_and_captures() {
        let g = game(3, 5, 1);
        // P1 pawn one step from the home rank wins by advancing.
        let mut state = g
            .custom_state(&[(3 * 3, Player::One), (7, Player::Two)], Player::One)
            .unwrap();
        assert!(g.outcome(&state).is_none());
        g.make_move(
            &mut state,
            BreakthroughMove {
                from: 9,
                to: 12, // straight to the top rank
            },
        );
        assert_eq!(g.outcome(&state), Some(Outcome::Win(Player::One)));

        // Capturing the last enemy pawn wins (opponent cannot move).
        let mut state = g
            .custom_state(&[(4, Player::One), (8, Player::Two)], Player::One)
            .unwrap();
        g.make_move(&mut state, BreakthroughMove { from: 4, to: 8 });
        assert_eq!(g.outcome(&state), Some(Outcome::Win(Player::One)));

        // Straight moves never capture: the straight destination onto an
        // enemy pawn is not generated.
        let state = g
            .custom_state(&[(4, Player::One), (7, Player::Two)], Player::One)
            .unwrap();
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        assert!(!moves.contains(&BreakthroughMove { from: 4, to: 7 }));
        assert!(moves.contains(&BreakthroughMove { from: 4, to: 6 }));
        assert!(moves.contains(&BreakthroughMove { from: 4, to: 8 }));

        // P2 wins symmetrically by reaching rank 0.
        let mut state = g
            .custom_state(&[(10, Player::One), (3, Player::Two)], Player::Two)
            .unwrap();
        g.make_move(&mut state, BreakthroughMove { from: 3, to: 0 });
        assert_eq!(g.outcome(&state), Some(Outcome::Win(Player::Two)));
    }

    #[test]
    fn blocked_pawns_and_the_no_move_rule() {
        // Individual pawns can be fully blocked: enemy straight ahead
        // (straight moves never capture) and own pawns / board edge on
        // the diagonals.
        let g = game(3, 5, 1);
        let state = g
            .custom_state(
                &[
                    (0, Player::One), // a1: straight a2 enemy, diag b2 own
                    (4, Player::One), // b2 keeps a1's diagonal closed
                    (3, Player::Two), // a2 blocks a1's straight
                    (7, Player::Two), // spare enemy pawn
                ],
                Player::One,
            )
            .unwrap();
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        assert!(
            !moves.iter().any(|m| m.from == 0),
            "corner pawn must be immobile"
        );
        assert!(g.outcome(&state).is_none(), "other pawns can still move");

        // A side with no pawns has no moves and has already lost through
        // the capture that removed its last pawn (covered in
        // terminal_races_and_captures). Because diagonal moves may always
        // capture, a side that still owns pawns almost always has a move;
        // the no-move rule is the terminal backstop. Sanity: every
        // reachable non-terminal state in random play has a move.
        let g = game(2, 6, 1);
        for s in random_playout_states(&g, 0xdead, 200) {
            assert!(g.has_any_move(&s));
        }

        // Direct unit check of the rule itself: a custom position where
        // the mover is fully boxed in is scored as a loss at build time.
        // (P1 pawns a1,b1; P2 pawns a2,b2 with their diagonals covered by
        // P1's own pawns - not constructible without giving P1 a capture,
        // so exercise the rule through the builder on an empty-mover
        // board instead.)
        let g = game(3, 5, 1);
        let state = g.custom_state(&[(7, Player::Two)], Player::One).unwrap();
        assert_eq!(
            g.outcome(&state),
            Some(Outcome::Win(Player::Two)),
            "a mover with no pawns (hence no moves) has lost"
        );
    }

    #[test]
    fn no_draws_and_games_terminate() {
        let g = game(4, 6, 2);
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
                assert!(plies < 1000, "game must terminate");
            };
            assert!(
                matches!(outcome, Outcome::Win(_)),
                "Breakthrough has no draws"
            );
        }
    }

    #[test]
    fn action_ids_are_stable_and_distinct_per_state() {
        let g = game(4, 6, 2);
        for state in random_playout_states(&g, 0xac71, 200) {
            let mut moves = Vec::new();
            g.legal_moves(&state, &mut moves);
            let ids: Vec<ActionId> = moves.iter().map(|&m| g.action_id(&state, m)).collect();
            let mut dedup = ids.clone();
            dedup.sort_unstable();
            dedup.dedup();
            assert_eq!(
                dedup.len(),
                ids.len(),
                "action IDs must be unique per state"
            );
            for &id in &ids {
                assert!((id as usize) < g.action_count());
            }
        }
    }
}

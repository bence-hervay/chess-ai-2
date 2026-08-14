//! Parameterized Connect-k on a `width` x `height` board.
//!
//! Players alternately place stones. The first player to own `k` contiguous
//! stones in a row, column, or diagonal wins; a full board is a draw.
//! With `gravity` on, a stone falls to the lowest empty cell of its column
//! (Connect Four). With `gravity` off, any empty cell may be played
//! (k-in-a-row / gomoku style).
//!
//! Cells are indexed `cell = y * width + x`, with row `y = 0` at the bottom.

use crate::game::{ActionId, FeatureId, Game, Outcome, Player};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

/// Fixed seed for Zobrist key generation so position keys are stable
/// across processes and runs.
const ZOBRIST_SEED: u64 = 0x00c0_44ec_7a11_5eed;

const EMPTY: u8 = 0;

fn cell_code(player: Player) -> u8 {
    match player {
        Player::One => 1,
        Player::Two => 2,
    }
}

/// Rules object for one Connect-k parameterization.
pub struct ConnectK {
    width: u16,
    height: u16,
    k: u16,
    gravity: bool,
    /// `zobrist[cell][player_index]`
    zobrist: Vec<[u64; 2]>,
    zobrist_side: u64,
}

/// A move: the destination cell index.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ConnectKMove(pub u16);

/// Board state. `cells[cell]` is `0` (empty), `1` (Player One) or
/// `2` (Player Two).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ConnectKState {
    cells: Vec<u8>,
    to_move: Player,
    key: u64,
    stones: u32,
    outcome: Option<Outcome>,
}

/// Undo record for `unmake_move`.
pub struct ConnectKUndo {
    prev_outcome: Option<Outcome>,
}

impl ConnectK {
    pub fn new(width: u16, height: u16, k: u16, gravity: bool) -> Result<ConnectK, String> {
        if width == 0 || height == 0 {
            return Err(format!("board {width}x{height} has no cells"));
        }
        if k < 2 {
            return Err(format!("k = {k} is trivial; require k >= 2"));
        }
        if k > width && k > height {
            return Err(format!(
                "k = {k} exceeds both board dimensions {width}x{height}"
            ));
        }
        let cells = usize::from(width) * usize::from(height);
        let mut rng = ChaCha12Rng::seed_from_u64(ZOBRIST_SEED);
        let zobrist = (0..cells)
            .map(|_| [rng.gen::<u64>(), rng.gen::<u64>()])
            .collect();
        let zobrist_side = rng.gen::<u64>();
        Ok(ConnectK {
            width,
            height,
            k,
            gravity,
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

    /// Number of contiguous same-owner stones through `cell` in the best
    /// of the four line directions.
    fn longest_line_through(&self, cells: &[u8], cell: u16) -> u16 {
        let owner = cells[usize::from(cell)];
        debug_assert_ne!(owner, EMPTY);
        let w = i32::from(self.width);
        let h = i32::from(self.height);
        let x0 = i32::from(cell % self.width);
        let y0 = i32::from(cell / self.width);
        let mut best = 0u16;
        for (dx, dy) in [(1i32, 0i32), (0, 1), (1, 1), (1, -1)] {
            let mut run = 1u16;
            for sign in [1i32, -1] {
                let (mut x, mut y) = (x0 + sign * dx, y0 + sign * dy);
                while x >= 0 && x < w && y >= 0 && y < h {
                    let c = usize::try_from(y * w + x).expect("in-bounds cell");
                    if cells[c] != owner {
                        break;
                    }
                    run += 1;
                    x += sign * dx;
                    y += sign * dy;
                }
            }
            best = best.max(run);
        }
        best
    }

    /// ASCII rendering for logs and debugging.
    pub fn render(&self, state: &ConnectKState) -> String {
        let mut out = String::new();
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                let c = match state.cells[usize::from(y * self.width + x)] {
                    1 => 'X',
                    2 => 'O',
                    _ => '.',
                };
                out.push(c);
            }
            out.push('\n');
        }
        out
    }
}

impl Game for ConnectK {
    type State = ConnectKState;
    type Move = ConnectKMove;
    type Undo = ConnectKUndo;

    fn initial_state(&self) -> ConnectKState {
        ConnectKState {
            cells: vec![EMPTY; self.cell_count() as usize],
            to_move: Player::One,
            key: 0,
            stones: 0,
            outcome: None,
        }
    }

    fn side_to_move(&self, state: &ConnectKState) -> Player {
        state.to_move
    }

    fn legal_moves(&self, state: &ConnectKState, moves: &mut Vec<ConnectKMove>) {
        moves.clear();
        if state.outcome.is_some() {
            return;
        }
        if self.gravity {
            for x in 0..self.width {
                for y in 0..self.height {
                    let cell = y * self.width + x;
                    if state.cells[usize::from(cell)] == EMPTY {
                        moves.push(ConnectKMove(cell));
                        break;
                    }
                }
            }
        } else {
            for cell in 0..self.cell_count() as u16 {
                if state.cells[usize::from(cell)] == EMPTY {
                    moves.push(ConnectKMove(cell));
                }
            }
        }
    }

    fn make_move(&self, state: &mut ConnectKState, mv: ConnectKMove) -> ConnectKUndo {
        let cell = mv.0;
        debug_assert!(state.outcome.is_none(), "move in terminal state");
        debug_assert_eq!(state.cells[usize::from(cell)], EMPTY, "cell occupied");
        debug_assert!(
            !self.gravity
                || cell < self.width
                || state.cells[usize::from(cell - self.width)] != EMPTY,
            "gravity violated: cell below is empty"
        );
        let undo = ConnectKUndo {
            prev_outcome: state.outcome,
        };
        let mover = state.to_move;
        state.cells[usize::from(cell)] = cell_code(mover);
        state.stones += 1;
        state.key ^= self.zobrist[usize::from(cell)][usize::from(cell_code(mover) - 1)];
        state.key ^= self.zobrist_side;
        state.to_move = mover.opponent();
        state.outcome = if self.longest_line_through(&state.cells, cell) >= self.k {
            Some(Outcome::Win(mover))
        } else if state.stones == self.cell_count() {
            Some(Outcome::Draw)
        } else {
            None
        };
        undo
    }

    fn unmake_move(&self, state: &mut ConnectKState, mv: ConnectKMove, undo: ConnectKUndo) {
        let cell = mv.0;
        let mover = state.to_move.opponent();
        debug_assert_eq!(state.cells[usize::from(cell)], cell_code(mover));
        state.cells[usize::from(cell)] = EMPTY;
        state.stones -= 1;
        state.key ^= self.zobrist[usize::from(cell)][usize::from(cell_code(mover) - 1)];
        state.key ^= self.zobrist_side;
        state.to_move = mover;
        state.outcome = undo.prev_outcome;
    }

    fn outcome(&self, state: &ConnectKState) -> Option<Outcome> {
        state.outcome
    }

    fn position_key(&self, state: &ConnectKState) -> u64 {
        state.key
    }

    fn encode_features(&self, state: &ConnectKState, features: &mut Vec<FeatureId>) {
        features.clear();
        let own = cell_code(state.to_move);
        for (cell, &code) in state.cells.iter().enumerate() {
            if code != EMPTY {
                let relative = u32::from(code != own);
                features.push(2 * cell as FeatureId + relative);
            }
        }
    }

    fn action_id(&self, _state: &ConnectKState, mv: ConnectKMove) -> ActionId {
        if self.gravity {
            ActionId::from(mv.0 % self.width)
        } else {
            ActionId::from(mv.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slow, obviously correct terminal scan used as a differential oracle.
    fn reference_outcome(game: &ConnectK, state: &ConnectKState) -> Option<Outcome> {
        for cell in 0..game.cell_count() as u16 {
            let code = state.cells[usize::from(cell)];
            if code != EMPTY && game.longest_line_through(&state.cells, cell) >= game.k {
                let winner = if code == 1 { Player::One } else { Player::Two };
                return Some(Outcome::Win(winner));
            }
        }
        if state.stones == game.cell_count() {
            return Some(Outcome::Draw);
        }
        None
    }

    /// Play a sequence of raw cell indices (no-gravity boards).
    fn play_cells(game: &ConnectK, cells: &[u16]) -> ConnectKState {
        let mut state = game.initial_state();
        for &c in cells {
            game.make_move(&mut state, ConnectKMove(c));
        }
        state
    }

    /// Play a sequence of columns, resolving each to its lowest empty cell
    /// (gravity boards).
    fn play_cols(game: &ConnectK, cols: &[u16]) -> ConnectKState {
        let mut state = game.initial_state();
        for &col in cols {
            let cell = (0..game.height)
                .map(|y| y * game.width + col)
                .find(|&c| state.cells[usize::from(c)] == EMPTY)
                .expect("column full");
            game.make_move(&mut state, ConnectKMove(cell));
        }
        state
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(ConnectK::new(0, 5, 3, true).is_err());
        assert!(ConnectK::new(5, 0, 3, true).is_err());
        assert!(ConnectK::new(5, 5, 1, true).is_err());
        assert!(ConnectK::new(3, 3, 4, true).is_err());
        assert!(ConnectK::new(7, 6, 4, true).is_ok());
    }

    #[test]
    fn gravity_moves_are_column_bottoms() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        let state = game.initial_state();
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        let bottom_row: Vec<ConnectKMove> = (0..7).map(ConnectKMove).collect();
        assert_eq!(moves, bottom_row);

        // After playing column 3, its next move is the cell one row up.
        let state = play_cols(&game, &[3]);
        game.legal_moves(&state, &mut moves);
        assert!(moves.contains(&ConnectKMove(7 + 3)));
        assert!(!moves.contains(&ConnectKMove(3)));
        assert_eq!(moves.len(), 7);
    }

    #[test]
    fn no_gravity_moves_are_all_empty_cells() {
        let game = ConnectK::new(4, 4, 3, false).unwrap();
        let state = play_cells(&game, &[5]);
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        assert_eq!(moves.len(), 15);
        assert!(!moves.contains(&ConnectKMove(5)));
    }

    #[test]
    fn horizontal_vertical_and_diagonal_wins() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        // Horizontal: P1 fills bottom row columns 0..=3; P2 stacks column 6.
        let state = play_cols(&game, &[0, 6, 1, 6, 2, 6, 3]);
        assert_eq!(game.outcome(&state), Some(Outcome::Win(Player::One)));

        // Vertical: P1 stacks column 0.
        let state = play_cols(&game, &[0, 1, 0, 2, 0, 3, 0]);
        assert_eq!(game.outcome(&state), Some(Outcome::Win(Player::One)));

        // Diagonal: P1 builds (0,0),(1,1),(2,2),(3,3).
        let state = play_cols(&game, &[0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]);
        assert_eq!(game.outcome(&state), Some(Outcome::Win(Player::One)));
    }

    #[test]
    fn draw_on_full_board() {
        let game = ConnectK::new(3, 3, 3, false).unwrap();
        // Final board (top to bottom): O X X / X O O / X O X — no 3-line.
        let state = play_cells(&game, &[0, 1, 2, 4, 3, 5, 7, 6, 8]);
        assert_eq!(game.outcome(&state), Some(Outcome::Draw));
    }

    #[test]
    fn terminal_positions_generate_no_moves() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        let state = play_cols(&game, &[0, 6, 1, 6, 2, 6, 3]);
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        assert!(moves.is_empty());
    }

    #[test]
    fn make_unmake_restores_state_and_key() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        let mut state = play_cols(&game, &[3, 3, 4]);
        let before = state.clone();
        let key_before = game.position_key(&state);
        let mv = ConnectKMove(5);
        let undo = game.make_move(&mut state, mv);
        assert_ne!(game.position_key(&state), key_before);
        game.unmake_move(&mut state, mv, undo);
        assert_eq!(state, before);
        assert_eq!(game.position_key(&state), key_before);
    }

    #[test]
    fn incremental_outcome_matches_reference_scan() {
        let game = ConnectK::new(5, 4, 3, true).unwrap();
        let mut state = game.initial_state();
        let mut moves = Vec::new();
        // Deterministic pseudo-random playout without an RNG dependency.
        let mut x = 0x1234_5678_u64;
        loop {
            assert_eq!(game.outcome(&state), reference_outcome(&game, &state));
            game.legal_moves(&state, &mut moves);
            if moves.is_empty() {
                break;
            }
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mv = moves[(x >> 33) as usize % moves.len()];
            game.make_move(&mut state, mv);
        }
        assert!(game.outcome(&state).is_some());
    }

    #[test]
    fn features_are_side_to_move_relative() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        // A: P1 on cell 0, P2 on cell 1, P1 to move.
        let a = play_cells(&game, &[0, 1]);
        // B: P2 on cell 0, P1 on cell 1, P1 to move (played 1 then 0).
        let b = play_cells(&game, &[1, 0]);
        let (mut fa, mut fb) = (Vec::new(), Vec::new());
        game.encode_features(&a, &mut fa);
        game.encode_features(&b, &mut fb);
        assert_eq!(fa, vec![0, 3]); // own stone on cell 0, opponent on cell 1
        assert_eq!(fb, vec![1, 2]); // opponent on cell 0, own stone on cell 1

        // Colour-swapped mirror positions with swapped side to move encode
        // identically: (P1 on 0, P2 to move) vs (P2 on 0, P1 to move).
        let c = play_cells(&game, &[0]); // P1 on 0, P2 to move
        let d = play_cells(&game, &[6, 0]); // P2 on 0 (and P1 on 6), P1 to move
        let (mut fc, mut fd) = (Vec::new(), Vec::new());
        game.encode_features(&c, &mut fc);
        game.encode_features(&d, &mut fd);
        assert_eq!(fc, vec![1]); // opponent stone on cell 0
        assert_eq!(fd, vec![1, 12]); // opponent on cell 0, own on cell 6

        // Keys of distinct positions differ and include the side to move.
        assert_ne!(game.position_key(&a), game.position_key(&b));
        assert_ne!(
            game.position_key(&c),
            game.position_key(&game.initial_state())
        );
    }

    #[test]
    fn colour_swap_flips_outcome() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        // P1 wins on the bottom row.
        let p1_win = play_cols(&game, &[0, 6, 1, 6, 2, 6, 3]);
        // Mirror game: P1 wastes moves in column 5/6 while P2 executes the
        // same bottom-row plan.
        let p2_win = play_cols(&game, &[6, 0, 5, 1, 5, 2, 5, 3]);
        assert_eq!(game.outcome(&p1_win), Some(Outcome::Win(Player::One)));
        assert_eq!(game.outcome(&p2_win), Some(Outcome::Win(Player::Two)));
    }

    #[test]
    fn action_ids_are_stable() {
        let gravity = ConnectK::new(7, 6, 4, true).unwrap();
        let state = play_cols(&gravity, &[3]);
        // Column 3's move is now cell 10, but its action ID stays 3.
        assert_eq!(gravity.action_id(&state, ConnectKMove(10)), 3);
        let free = ConnectK::new(5, 5, 4, false).unwrap();
        let state = free.initial_state();
        assert_eq!(free.action_id(&state, ConnectKMove(17)), 17);
    }

    #[test]
    fn legal_moves_do_not_mutate_state() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        let state = play_cols(&game, &[3, 2, 1]);
        let before = state.clone();
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        assert_eq!(state, before);
    }
}

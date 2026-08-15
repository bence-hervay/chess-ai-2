//! Forward Chess — the plan's target variant. The authoritative
//! human-readable rules live in `FORWARD_CHESS_RULES.md`; this module
//! implements that document exactly and nothing else.
//!
//! Core law: a piece may move to or attack a square only if the rank
//! displacement measured in the piece's *ahead* direction is >= 0.
//! Promoted pieces are created with reversed orientation, so
//! orientation is separate from colour.
//!
//! Repetition: Forward Chess positions can repeat (threefold is a
//! draw), so the acyclic `ExactSolver` must never be used on it; the
//! retrograde solver in `crate::search` handles reduced instances.

use crate::game::{ActionId, FeatureId, Game, Outcome, Player};
use crate::search::Wdl;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

/// Fixed seed for Zobrist key generation.
const ZOBRIST_SEED: u64 = 0xf0c4_ad5e_0a11_5eed;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Piece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

const PROMOTION_CHOICES: [Piece; 4] = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];

fn piece_letter(piece: Piece) -> char {
    match piece {
        Piece::Pawn => 'P',
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
    }
}

/// Packed cell code: 0 empty, else `1 + piece + 6·owner + 12·reversed`.
const EMPTY: u8 = 0;

fn pack(owner: Player, piece: Piece, reversed: bool) -> u8 {
    1 + piece as u8 + 6 * (owner == Player::Two) as u8 + 12 * reversed as u8
}

fn unpack(code: u8) -> (Player, Piece, bool) {
    debug_assert_ne!(code, EMPTY);
    let index = code - 1;
    let piece = match index % 6 {
        0 => Piece::Pawn,
        1 => Piece::Knight,
        2 => Piece::Bishop,
        3 => Piece::Rook,
        4 => Piece::Queen,
        _ => Piece::King,
    };
    let owner = if (index / 6).is_multiple_of(2) {
        Player::One
    } else {
        Player::Two
    };
    (owner, piece, index / 12 == 1)
}

/// Named reduced rulesets (FORWARD_CHESS_RULES.md §1, §12).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ruleset {
    Tiny,
    Small,
    Medium,
    Full,
}

impl Ruleset {
    pub fn label(self) -> &'static str {
        match self {
            Ruleset::Tiny => "tiny",
            Ruleset::Small => "small",
            Ruleset::Medium => "medium",
            Ruleset::Full => "full",
        }
    }

    fn dimensions(self) -> (u16, u16) {
        match self {
            Ruleset::Tiny => (3, 4),
            Ruleset::Small => (4, 4),
            Ruleset::Medium => (6, 6),
            Ruleset::Full => (8, 8),
        }
    }

    fn castling(self) -> bool {
        matches!(self, Ruleset::Full)
    }

    fn code(self) -> u8 {
        match self {
            Ruleset::Tiny => 0,
            Ruleset::Small => 1,
            Ruleset::Medium => 2,
            Ruleset::Full => 3,
        }
    }
}

/// Rules object for one Forward Chess ruleset.
pub struct ForwardChess {
    ruleset: Ruleset,
    width: u16,
    height: u16,
    /// `zobrist[cell][code - 1]`
    zobrist: Vec<[u64; 24]>,
    zobrist_side: u64,
    zobrist_castle: [u64; 4],
    zobrist_ep: Vec<u64>,
}

/// A move. Castling is encoded as the king's two-file step.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct FcMove {
    pub from: u16,
    pub to: u16,
    pub promotion: Option<Piece>,
}

/// Position core: everything except the repetition history.
#[derive(Clone, PartialEq, Debug)]
pub struct FcCore {
    cells: Vec<u8>,
    to_move: Player,
    /// `[white short, white long, black short, black long]`
    castling: [bool; 4],
    /// The square a double-stepping pawn crossed, if capturable now.
    ep: Option<u16>,
    halfmove: u16,
    key: u64,
    outcome: Option<Outcome>,
}

/// Full game state: core plus hashes since the last irreversible move.
#[derive(Clone, PartialEq, Debug)]
pub struct FcState {
    core: FcCore,
    history: Vec<u64>,
}

impl FcState {
    /// Equality of everything a tablebase record stores (the repetition
    /// history is path-dependent and excluded).
    pub fn identical_core(&self, other: &FcState) -> bool {
        self.core == other.core
    }
}

pub struct FcUndo {
    prior: FcCore,
    cleared_history: Option<Vec<u64>>,
}

impl ForwardChess {
    pub fn new(ruleset: Ruleset) -> ForwardChess {
        let (width, height) = ruleset.dimensions();
        let cells = usize::from(width) * usize::from(height);
        let mut rng = ChaCha12Rng::seed_from_u64(ZOBRIST_SEED);
        let zobrist = (0..cells)
            .map(|_| std::array::from_fn(|_| rng.gen::<u64>()))
            .collect();
        ForwardChess {
            ruleset,
            width,
            height,
            zobrist,
            zobrist_side: rng.gen(),
            zobrist_castle: std::array::from_fn(|_| rng.gen()),
            zobrist_ep: (0..width).map(|_| rng.gen()).collect(),
        }
    }

    pub fn ruleset(&self) -> Ruleset {
        self.ruleset
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

    /// 24 features per cell (piece x own/opp x orientation) plus 4
    /// castling rights plus `width` en-passant files.
    pub fn feature_count(&self) -> usize {
        self.cell_count() as usize * 24 + 4 + usize::from(self.width)
    }

    /// `from x to` plus underpromotion slots (piece x from-file x dir).
    pub fn action_count(&self) -> usize {
        let cells = self.cell_count() as usize;
        cells * cells + 3 * usize::from(self.width) * 3
    }

    fn cell(&self, file: u16, rank: u16) -> u16 {
        rank * self.width + file
    }

    /// Parse "b3"-style coordinates (single-character file, rank >= 1).
    pub fn square(&self, name: &str) -> u16 {
        let bytes = name.as_bytes();
        let file = u16::from(bytes[0] - b'a');
        let rank: u16 = name[1..].parse().expect("rank number");
        assert!(
            file < self.width && rank >= 1 && rank <= self.height,
            "{name}"
        );
        self.cell(file, rank - 1)
    }

    /// Owner's forward direction in rank steps.
    fn forward_of(owner: Player) -> i32 {
        match owner {
            Player::One => 1,
            Player::Two => -1,
        }
    }

    /// The piece's *ahead* direction (owner forward, flipped when
    /// reversed).
    fn ahead_of(owner: Player, reversed: bool) -> i32 {
        if reversed {
            -Self::forward_of(owner)
        } else {
            Self::forward_of(owner)
        }
    }

    /// Does the piece on `from` (with `code`) attack `target` on this
    /// board? Implements FORWARD_CHESS_RULES.md §3–4 exactly.
    fn attacks(&self, cells: &[u8], from: u16, code: u8, target: u16) -> bool {
        if from == target {
            return false;
        }
        let (owner, piece, reversed) = unpack(code);
        let ahead = Self::ahead_of(owner, reversed);
        let w = i32::from(self.width);
        let df = i32::from(target % self.width) - i32::from(from % self.width);
        let dr_board = i32::from(target / self.width) - i32::from(from / self.width);
        let dr = dr_board * ahead; // displacement measured "ahead"
        match piece {
            Piece::Pawn => dr == 1 && df.abs() == 1,
            Piece::Knight => (df.abs() == 1 && dr == 2) || (df.abs() == 2 && dr == 1),
            Piece::King => (0..=1).contains(&dr) && df.abs() <= 1,
            Piece::Rook => {
                ((df == 0 && dr > 0) || (dr == 0 && df != 0))
                    && self.ray_clear(cells, from, target, w)
            }
            Piece::Bishop => df.abs() == dr && dr > 0 && self.ray_clear(cells, from, target, w),
            Piece::Queen => {
                ((df == 0 && dr > 0) || (dr == 0 && df != 0) || (df.abs() == dr && dr > 0))
                    && self.ray_clear(cells, from, target, w)
            }
        }
    }

    /// Are the squares strictly between `from` and `target` empty
    /// (assumes they are aligned on a rank, file, or diagonal)?
    fn ray_clear(&self, cells: &[u8], from: u16, target: u16, w: i32) -> bool {
        let (fx, fy) = (i32::from(from) % w, i32::from(from) / w);
        let (tx, ty) = (i32::from(target) % w, i32::from(target) / w);
        let dx = (tx - fx).signum();
        let dy = (ty - fy).signum();
        let (mut x, mut y) = (fx + dx, fy + dy);
        while (x, y) != (tx, ty) {
            if cells[(y * w + x) as usize] != EMPTY {
                return false;
            }
            x += dx;
            y += dy;
        }
        true
    }

    // -- crate-internal accessors for the structured feature extractor
    //    (`features::forward_chess`). Rule-level facts only; no strategy.

    /// The packed cell array of a state (see `pack`/`unpack_code`).
    pub(crate) fn state_cells<'a>(&self, state: &'a FcState) -> &'a [u8] {
        &state.core.cells
    }

    /// Decode a non-empty packed cell code.
    pub(crate) fn unpack_code(code: u8) -> (Player, Piece, bool) {
        unpack(code)
    }

    /// Is `target` attacked by any piece of `attacker` (rules §3–4)?
    pub(crate) fn cell_attacked_by(&self, cells: &[u8], target: u16, attacker: Player) -> bool {
        self.square_attacked(cells, target, attacker)
    }

    /// Is the side to move currently in check?
    pub(crate) fn in_check(&self, state: &FcState) -> bool {
        let mover = state.core.to_move;
        let king = self.king_square(&state.core.cells, mover);
        self.square_attacked(&state.core.cells, king, mover.opponent())
    }

    /// Ranks a natural pawn of `owner` on `cell` still has to advance
    /// before promotion (rules §4/§8).
    pub(crate) fn pawn_promotion_distance(&self, owner: Player, cell: u16) -> u16 {
        let rank = cell / self.width;
        let last = Self::last_rank(owner, self.height);
        rank.abs_diff(last)
    }

    /// Is `target` attacked by any piece of `attacker`?
    fn square_attacked(&self, cells: &[u8], target: u16, attacker: Player) -> bool {
        for (cell, &code) in cells.iter().enumerate() {
            if code != EMPTY
                && unpack(code).0 == attacker
                && self.attacks(cells, cell as u16, code, target)
            {
                return true;
            }
        }
        false
    }

    /// The king's initial square. Layouts are 180-degree rotations of
    /// each other, so Black's king home is the rotated e1 — d8 on 8x8,
    /// NOT e8 (the queen's rotated square). Getting this wrong once let
    /// Black keep castling rights after king moves (D031).
    fn king_home(&self, owner: Player) -> u16 {
        let white = self.cell(self.width / 2, 0);
        match owner {
            Player::One => white,
            Player::Two => self.cell_count() as u16 - 1 - white,
        }
    }

    fn king_square(&self, cells: &[u8], owner: Player) -> u16 {
        let king = pack(owner, Piece::King, false);
        cells
            .iter()
            .position(|&c| c == king)
            .expect("king on board") as u16
    }

    fn last_rank(owner: Player, height: u16) -> u16 {
        match owner {
            Player::One => height - 1,
            Player::Two => 0,
        }
    }

    fn pawn_home_rank(owner: Player, height: u16) -> u16 {
        match owner {
            Player::One => 1,
            Player::Two => height - 2,
        }
    }

    /// Apply `mv` to a bare cell array plus en-passant context (shared
    /// by real make_move and the legality filter). Returns whether the
    /// move was a capture or pawn move (irreversible).
    fn apply_to_cells(&self, cells: &mut [u8], ep: Option<u16>, mv: FcMove) -> bool {
        let code = cells[usize::from(mv.from)];
        let (owner, piece, _reversed) = unpack(code);
        let mut irreversible = piece == Piece::Pawn;
        if cells[usize::from(mv.to)] != EMPTY {
            irreversible = true;
        }
        // En passant: pawn moves diagonally onto the empty crossed square.
        if piece == Piece::Pawn && Some(mv.to) == ep && cells[usize::from(mv.to)] == EMPTY {
            let captured = self.cell(mv.to % self.width, mv.from / self.width);
            cells[usize::from(captured)] = EMPTY;
            irreversible = true;
        }
        // Castling: king moves two files; the rook jumps to the crossed
        // square.
        if piece == Piece::King
            && (i32::from(mv.to % self.width) - i32::from(mv.from % self.width)).abs() == 2
        {
            let rank = mv.from / self.width;
            let (rook_from, rook_to) = if mv.to % self.width > mv.from % self.width {
                (
                    self.cell(self.width - 1, rank),
                    self.cell(mv.from % self.width + 1, rank),
                )
            } else {
                (
                    self.cell(0, rank),
                    self.cell(mv.from % self.width - 1, rank),
                )
            };
            cells[usize::from(rook_to)] = cells[usize::from(rook_from)];
            cells[usize::from(rook_from)] = EMPTY;
        }
        cells[usize::from(mv.from)] = EMPTY;
        cells[usize::from(mv.to)] = match mv.promotion {
            Some(promoted) => pack(owner, promoted, true),
            None => code,
        };
        irreversible
    }

    fn move_leaves_king_in_check(&self, core: &FcCore, mv: FcMove) -> bool {
        let mut cells = core.cells.clone();
        self.apply_to_cells(&mut cells, core.ep, mv);
        let king = self.king_square(&cells, core.to_move);
        self.square_attacked(&cells, king, core.to_move.opponent())
    }

    /// Generate all legal moves of `core`'s side to move.
    fn generate_legal(&self, core: &FcCore, moves: &mut Vec<FcMove>) {
        moves.clear();
        let mover = core.to_move;
        let w = i32::from(self.width);
        let h = i32::from(self.height);
        let push = |this: &Self, from: u16, to: u16, promo: bool, moves: &mut Vec<FcMove>| {
            if promo {
                for piece in PROMOTION_CHOICES {
                    let mv = FcMove {
                        from,
                        to,
                        promotion: Some(piece),
                    };
                    if !this.move_leaves_king_in_check(core, mv) {
                        moves.push(mv);
                    }
                }
            } else {
                let mv = FcMove {
                    from,
                    to,
                    promotion: None,
                };
                if !this.move_leaves_king_in_check(core, mv) {
                    moves.push(mv);
                }
            }
        };

        for (cell, &code) in core.cells.iter().enumerate() {
            if code == EMPTY || unpack(code).0 != mover {
                continue;
            }
            let from = cell as u16;
            let (owner, piece, reversed) = unpack(code);
            let ahead = Self::ahead_of(owner, reversed);
            let (fx, fy) = (i32::from(from) % w, i32::from(from) / w);
            let at = |x: i32, y: i32| (y * w + x) as u16;
            let on_board = |x: i32, y: i32| x >= 0 && x < w && y >= 0 && y < h;
            match piece {
                Piece::Pawn => {
                    let last = i32::from(Self::last_rank(owner, self.height));
                    let step = fy + ahead;
                    if on_board(fx, step) && core.cells[usize::from(at(fx, step))] == EMPTY {
                        push(self, from, at(fx, step), step == last, moves);
                        let home = i32::from(Self::pawn_home_rank(owner, self.height));
                        let double = fy + 2 * ahead;
                        if fy == home
                            && on_board(fx, double)
                            && double != last
                            && core.cells[usize::from(at(fx, double))] == EMPTY
                        {
                            push(self, from, at(fx, double), false, moves);
                        }
                    }
                    for dx in [-1, 1] {
                        let (cx, cy) = (fx + dx, fy + ahead);
                        if !on_board(cx, cy) {
                            continue;
                        }
                        let target = at(cx, cy);
                        let occupant = core.cells[usize::from(target)];
                        let enemy = occupant != EMPTY && unpack(occupant).0 != mover;
                        if enemy || Some(target) == core.ep {
                            push(self, from, target, cy == last, moves);
                        }
                    }
                }
                Piece::Knight => {
                    for (dx, dr) in [(1, 2), (-1, 2), (2, 1), (-2, 1)] {
                        let (tx, ty) = (fx + dx, fy + dr * ahead);
                        if on_board(tx, ty) {
                            let occupant = core.cells[usize::from(at(tx, ty))];
                            if occupant == EMPTY || unpack(occupant).0 != mover {
                                push(self, from, at(tx, ty), false, moves);
                            }
                        }
                    }
                }
                Piece::King => {
                    for (dx, dr) in [(-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                        let (tx, ty) = (fx + dx, fy + dr * ahead);
                        if on_board(tx, ty) {
                            let occupant = core.cells[usize::from(at(tx, ty))];
                            if occupant == EMPTY || unpack(occupant).0 != mover {
                                push(self, from, at(tx, ty), false, moves);
                            }
                        }
                    }
                    self.generate_castling(core, from, moves);
                }
                Piece::Rook | Piece::Bishop | Piece::Queen => {
                    let rays: &[(i32, i32)] = match piece {
                        Piece::Rook => &[(0, 1), (1, 0), (-1, 0)],
                        Piece::Bishop => &[(1, 1), (-1, 1)],
                        _ => &[(0, 1), (1, 0), (-1, 0), (1, 1), (-1, 1)],
                    };
                    for &(dx, dr) in rays {
                        let (mut x, mut y) = (fx + dx, fy + dr * ahead);
                        while on_board(x, y) {
                            let occupant = core.cells[usize::from(at(x, y))];
                            if occupant == EMPTY {
                                push(self, from, at(x, y), false, moves);
                            } else {
                                if unpack(occupant).0 != mover {
                                    push(self, from, at(x, y), false, moves);
                                }
                                break;
                            }
                            x += dx;
                            y += dr * ahead;
                        }
                    }
                }
            }
        }
    }

    fn generate_castling(&self, core: &FcCore, king_from: u16, moves: &mut Vec<FcMove>) {
        if !self.ruleset.castling() {
            return;
        }
        let mover = core.to_move;
        let home = if mover == Player::One {
            0
        } else {
            self.height - 1
        };
        // Rights guarantee this when rights-clearing is correct; the
        // explicit check makes phantom castling (and the file-underflow
        // it once caused, D031) impossible even under corrupted rights.
        if king_from != self.king_home(mover) {
            return;
        }
        let enemy = mover.opponent();
        if self.square_attacked(&core.cells, king_from, enemy) {
            return;
        }
        let rights_base = if mover == Player::One { 0 } else { 2 };
        let kx = i32::from(king_from % self.width);
        let w = i32::from(self.width);
        // (right index, rook file, direction)
        for (right, rook_file, dir) in [(0usize, w - 1, 1i32), (1, 0, -1)] {
            if !core.castling[rights_base + right] {
                continue;
            }
            let rook_cell = self.cell(rook_file as u16, home);
            let rook_code = core.cells[usize::from(rook_cell)];
            if rook_code == EMPTY || unpack(rook_code) != (mover, Piece::Rook, false) {
                continue;
            }
            // All squares strictly between king and rook must be empty.
            let mut x = kx + dir;
            let mut clear = true;
            while x != rook_file {
                if core.cells[usize::from(self.cell(x as u16, home))] != EMPTY {
                    clear = false;
                    break;
                }
                x += dir;
            }
            if !clear {
                continue;
            }
            // The two squares the king crosses must not be attacked.
            // Files are validated before the u16 cast: an out-of-range
            // step must be skipped, never wrapped (D031).
            if kx + 2 * dir < 0 || kx + 2 * dir >= w {
                continue;
            }
            let crossed = self.cell((kx + dir) as u16, home);
            let dest = self.cell((kx + 2 * dir) as u16, home);
            if self.square_attacked(&core.cells, crossed, enemy)
                || self.square_attacked(&core.cells, dest, enemy)
            {
                continue;
            }
            let mv = FcMove {
                from: king_from,
                to: dest,
                promotion: None,
            };
            if !self.move_leaves_king_in_check(core, mv) {
                moves.push(mv);
            }
        }
    }

    fn has_any_legal_move(&self, core: &FcCore) -> bool {
        let mut moves = Vec::new();
        self.generate_legal(core, &mut moves);
        !moves.is_empty()
    }

    fn compute_key(&self, core: &FcCore) -> u64 {
        let mut key = 0u64;
        for (cell, &code) in core.cells.iter().enumerate() {
            if code != EMPTY {
                key ^= self.zobrist[cell][usize::from(code) - 1];
            }
        }
        if core.to_move == Player::Two {
            key ^= self.zobrist_side;
        }
        for (i, &right) in core.castling.iter().enumerate() {
            if right {
                key ^= self.zobrist_castle[i];
            }
        }
        if let Some(ep) = core.ep {
            key ^= self.zobrist_ep[usize::from(ep % self.width)];
        }
        key
    }

    /// Outcome of `core` for its side to move, ignoring repetition
    /// (which needs the history and is layered on in `make_move`).
    fn core_outcome(&self, core: &FcCore) -> Option<Outcome> {
        if !self.has_any_legal_move(core) {
            let king = self.king_square(&core.cells, core.to_move);
            return if self.square_attacked(&core.cells, king, core.to_move.opponent()) {
                Some(Outcome::Win(core.to_move.opponent()))
            } else {
                Some(Outcome::Draw)
            };
        }
        if core.halfmove >= 100 {
            return Some(Outcome::Draw);
        }
        None
    }

    /// Build an arbitrary position (tests, diagnostics, solver seeds).
    /// `pieces`: (square name, owner, piece, reversed).
    pub fn custom_state(
        &self,
        pieces: &[(&str, Player, Piece, bool)],
        to_move: Player,
        castling: [bool; 4],
        ep: Option<&str>,
    ) -> FcState {
        let mut cells = vec![EMPTY; self.cell_count() as usize];
        for &(name, owner, piece, reversed) in pieces {
            let cell = self.square(name);
            assert_eq!(cells[usize::from(cell)], EMPTY, "square {name} used twice");
            cells[usize::from(cell)] = pack(owner, piece, reversed);
        }
        let mut core = FcCore {
            cells,
            to_move,
            castling,
            ep: ep.map(|name| self.square(name)),
            halfmove: 0,
            key: 0,
            outcome: None,
        };
        core.key = self.compute_key(&core);
        core.outcome = self.core_outcome(&core);
        FcState {
            core,
            history: Vec::new(),
        }
    }

    fn square_name(&self, cell: u16) -> String {
        format!(
            "{}{}",
            (b'a' + (cell % self.width) as u8) as char,
            cell / self.width + 1
        )
    }

    /// "a2b3" / "a2b3=Q" text for a move (promotion letter upper-case).
    pub fn format_move(&self, mv: FcMove) -> String {
        let mut text = format!("{}{}", self.square_name(mv.from), self.square_name(mv.to));
        if let Some(piece) = mv.promotion {
            text.push('=');
            text.push(piece_letter(piece));
        }
        text
    }

    /// ASCII board for interactive play: ranks top-down, upper-case
    /// White, lower-case Black, `~` marking reversed orientation, with
    /// side-to-move / castling / en-passant / halfmove annotations.
    pub fn render_ascii(&self, state: &FcState) -> String {
        let core = &state.core;
        let mut out = String::new();
        for rank in (0..self.height).rev() {
            out.push_str(&format!("{:>2} ", rank + 1));
            for file in 0..self.width {
                let code = core.cells[usize::from(self.cell(file, rank))];
                if code == EMPTY {
                    out.push_str(" . ");
                } else {
                    let (owner, piece, reversed) = unpack(code);
                    let mut letter = piece_letter(piece);
                    if owner == Player::Two {
                        letter = letter.to_ascii_lowercase();
                    }
                    out.push(' ');
                    out.push(letter);
                    out.push(if reversed { '~' } else { ' ' });
                }
            }
            out.push('\n');
        }
        out.push_str("   ");
        for file in 0..self.width {
            out.push(' ');
            out.push((b'a' + file as u8) as char);
            out.push(' ');
        }
        out.push('\n');
        let castling: String = core
            .castling
            .iter()
            .zip(['K', 'Q', 'k', 'q'])
            .filter(|(&right, _)| right)
            .map(|(_, letter)| letter)
            .collect();
        out.push_str(&format!(
            "{} to move | castling {} | ep {} | halfmove {}\n",
            if core.to_move == Player::One {
                "White"
            } else {
                "Black"
            },
            if castling.is_empty() {
                "-".into()
            } else {
                castling
            },
            core.ep
                .map_or("-".to_string(), |cell| self.square_name(cell)),
            core.halfmove,
        ));
        out
    }

    /// Perspective cell: the board rotated 180 degrees for Black, so
    /// "forward" is always +rank and layouts mirror onto themselves.
    pub(crate) fn perspective_cell(&self, cell: u16, mover: Player) -> u16 {
        match mover {
            Player::One => cell,
            Player::Two => self.cell_count() as u16 - 1 - cell,
        }
    }
}

impl Game for ForwardChess {
    type State = FcState;
    type Move = FcMove;
    type Undo = FcUndo;

    fn initial_state(&self) -> FcState {
        let castling = if self.ruleset.castling() {
            [true; 4]
        } else {
            [false; 4]
        };
        let white: &[(&str, Piece)] = match self.ruleset {
            Ruleset::Tiny => &[("b1", Piece::King), ("a2", Piece::Pawn)],
            Ruleset::Small => &[
                ("b1", Piece::King),
                ("c1", Piece::Rook),
                ("a2", Piece::Pawn),
            ],
            Ruleset::Medium => &[
                ("c1", Piece::King),
                ("d1", Piece::Rook),
                ("b1", Piece::Bishop),
                ("e1", Piece::Knight),
                ("a2", Piece::Pawn),
                ("c2", Piece::Pawn),
                ("f2", Piece::Pawn),
            ],
            Ruleset::Full => &[
                ("a1", Piece::Rook),
                ("b1", Piece::Knight),
                ("c1", Piece::Bishop),
                ("d1", Piece::Queen),
                ("e1", Piece::King),
                ("f1", Piece::Bishop),
                ("g1", Piece::Knight),
                ("h1", Piece::Rook),
                ("a2", Piece::Pawn),
                ("b2", Piece::Pawn),
                ("c2", Piece::Pawn),
                ("d2", Piece::Pawn),
                ("e2", Piece::Pawn),
                ("f2", Piece::Pawn),
                ("g2", Piece::Pawn),
                ("h2", Piece::Pawn),
            ],
        };
        let mut pieces: Vec<(String, Player, Piece, bool)> = Vec::new();
        for &(name, piece) in white {
            pieces.push((name.to_string(), Player::One, piece, false));
            // Black mirror: rotate 180 degrees.
            let cell = self.square(name);
            let rotated = self.cell_count() as u16 - 1 - cell;
            let file = (rotated % self.width) as u8;
            let rank = rotated / self.width + 1;
            pieces.push((
                format!("{}{}", (b'a' + file) as char, rank),
                Player::Two,
                piece,
                false,
            ));
        }
        let refs: Vec<(&str, Player, Piece, bool)> = pieces
            .iter()
            .map(|(n, o, p, r)| (n.as_str(), *o, *p, *r))
            .collect();
        self.custom_state(&refs, Player::One, castling, None)
    }

    fn side_to_move(&self, state: &FcState) -> Player {
        state.core.to_move
    }

    fn legal_moves(&self, state: &FcState, moves: &mut Vec<FcMove>) {
        self.generate_legal(&state.core, moves);
        moves.sort();
    }

    fn make_move(&self, state: &mut FcState, mv: FcMove) -> FcUndo {
        let prior = state.core.clone();
        let mover = state.core.to_move;
        let code = state.core.cells[usize::from(mv.from)];
        let (_, piece, _) = unpack(code);
        let prev_key = state.core.key;

        let mut cells = std::mem::take(&mut state.core.cells);
        let irreversible = self.apply_to_cells(&mut cells, state.core.ep, mv);
        state.core.cells = cells;

        // Rules invariant: no legal move may remove a king. A violation
        // is a movegen bug; dump the position so it can be reproduced.
        for owner in [Player::One, Player::Two] {
            let king = pack(owner, Piece::King, false);
            if !state.core.cells.contains(&king) {
                panic!(
                    "king of {owner:?} vanished after {} on:\n{}",
                    self.format_move(mv),
                    self.render_ascii(&FcState {
                        core: prior.clone(),
                        history: Vec::new()
                    }),
                );
            }
        }

        // Castling rights: moving the king or a rook (or capturing a
        // rook on its home corner) clears rights.
        if self.ruleset.castling() {
            let clear_for = |castling: &mut [bool; 4], owner: Player, cell: u16, this: &Self| {
                let base = if owner == Player::One { 0 } else { 2 };
                let home = if owner == Player::One {
                    0
                } else {
                    this.height - 1
                };
                if cell == this.king_home(owner) {
                    castling[base] = false;
                    castling[base + 1] = false;
                }
                if cell == this.cell(this.width - 1, home) {
                    castling[base] = false;
                }
                if cell == this.cell(0, home) {
                    castling[base + 1] = false;
                }
            };
            clear_for(&mut state.core.castling, mover, mv.from, self);
            clear_for(&mut state.core.castling, mover.opponent(), mv.to, self);
        }

        // En passant target: only set by a pawn double-step.
        state.core.ep = None;
        if piece == Piece::Pawn {
            let from_rank = i32::from(mv.from / self.width);
            let to_rank = i32::from(mv.to / self.width);
            if (to_rank - from_rank).abs() == 2 {
                let crossed_rank = ((from_rank + to_rank) / 2) as u16;
                state.core.ep = Some(self.cell(mv.to % self.width, crossed_rank));
            }
        }

        state.core.halfmove = if irreversible {
            0
        } else {
            state.core.halfmove + 1
        };
        let cleared_history = if irreversible {
            Some(std::mem::take(&mut state.history))
        } else {
            state.history.push(prev_key);
            None
        };
        state.core.to_move = mover.opponent();
        state.core.key = self.compute_key(&state.core);

        state.core.outcome = self.core_outcome(&state.core).or_else(|| {
            let repeats = state
                .history
                .iter()
                .filter(|&&h| h == state.core.key)
                .count();
            (repeats >= 2).then_some(Outcome::Draw)
        });

        FcUndo {
            prior,
            cleared_history,
        }
    }

    fn unmake_move(&self, state: &mut FcState, _mv: FcMove, undo: FcUndo) {
        state.core = undo.prior;
        match undo.cleared_history {
            Some(history) => state.history = history,
            None => {
                state.history.pop();
            }
        }
    }

    fn outcome(&self, state: &FcState) -> Option<Outcome> {
        state.core.outcome
    }

    fn position_key(&self, state: &FcState) -> u64 {
        state.core.key
    }

    fn encode_features(&self, state: &FcState, features: &mut Vec<FeatureId>) {
        features.clear();
        let mover = state.core.to_move;
        for (cell, &code) in state.core.cells.iter().enumerate() {
            if code == EMPTY {
                continue;
            }
            let (owner, piece, reversed) = unpack(code);
            let rel = self.perspective_cell(cell as u16, mover);
            features.push(
                (usize::from(rel) * 24
                    + piece as usize * 4
                    + usize::from(owner != mover) * 2
                    + usize::from(reversed)) as FeatureId,
            );
        }
        let base = self.cell_count() as usize * 24;
        let (own_base, opp_base) = if mover == Player::One { (0, 2) } else { (2, 0) };
        for (slot, index) in [
            (0, own_base),
            (1, own_base + 1),
            (2, opp_base),
            (3, opp_base + 1),
        ] {
            if state.core.castling[index] {
                features.push((base + slot) as FeatureId);
            }
        }
        if let Some(ep) = state.core.ep {
            let rel = self.perspective_cell(ep, mover);
            features.push((base + 4 + usize::from(rel % self.width)) as FeatureId);
        }
        features.sort_unstable();
    }

    fn action_id(&self, state: &FcState, mv: FcMove) -> ActionId {
        let mover = state.core.to_move;
        let cells = self.cell_count() as usize;
        let from = usize::from(self.perspective_cell(mv.from, mover));
        let to = usize::from(self.perspective_cell(mv.to, mover));
        match mv.promotion {
            None | Some(Piece::Queen) => (from * cells + to) as ActionId,
            Some(piece) => {
                let promo_index = match piece {
                    Piece::Knight => 0usize,
                    Piece::Bishop => 1,
                    Piece::Rook => 2,
                    _ => unreachable!("queen handled above"),
                };
                let w = usize::from(self.width);
                let from_file = from % w;
                let to_file = to % w;
                let direction = (to_file as i32 - from_file as i32 + 1) as usize;
                (cells * cells + promo_index * (w * 3) + from_file * 3 + direction) as ActionId
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compact tablebase serialization
// ---------------------------------------------------------------------------
//
// Binary backup format for retrograde solutions (DECISIONS.md D029).
// Layout: 16-byte header (magic "FCTB", version, ruleset code, width,
// height, u64 LE position count), one variable-length record per
// position in solver discovery order, then a 12-byte footer (magic
// "BTCF", u64 LE FNV-1a-64 checksum of header + records). A record is
// the cell codes packed at 5 bits each (LSB-first, zero padding), a
// flags byte (bit 0 side to move, bits 1-2 WDL, bits 3-6 castling,
// bit 7 en-passant present), the en-passant square byte when flagged,
// and the halfmove-clock byte. Repetition history is path-dependent and
// never stored; key and outcome are recomputed on load.

pub const TB_MAGIC: [u8; 4] = *b"FCTB";
pub const TB_VERSION: u8 = 1;
const TB_FOOTER_MAGIC: [u8; 4] = *b"BTCF";

/// Incremental FNV-1a 64-bit checksum.
struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Fnv64 {
        Fnv64(0xcbf2_9ce4_8422_2325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

impl ForwardChess {
    fn packed_cells_len(&self) -> usize {
        (self.cell_count() as usize * 5).div_ceil(8)
    }

    /// Append one packed (state, value) record to `out`.
    pub fn pack_record(&self, state: &FcState, wdl: Wdl, out: &mut Vec<u8>) {
        let core = &state.core;
        let start = out.len();
        out.resize(start + self.packed_cells_len(), 0);
        for (cell, &code) in core.cells.iter().enumerate() {
            debug_assert!(code <= 24);
            let bit = cell * 5;
            let spread = u16::from(code) << (bit % 8);
            out[start + bit / 8] |= spread as u8;
            if bit % 8 > 3 {
                out[start + bit / 8 + 1] |= (spread >> 8) as u8;
            }
        }
        let mut flags = u8::from(core.to_move == Player::Two);
        flags |= (wdl as u8) << 1;
        for (i, &right) in core.castling.iter().enumerate() {
            flags |= u8::from(right) << (3 + i);
        }
        flags |= u8::from(core.ep.is_some()) << 7;
        out.push(flags);
        if let Some(ep) = core.ep {
            out.push(u8::try_from(ep).expect("boards have at most 64 cells"));
        }
        out.push(u8::try_from(core.halfmove).expect("halfmove 100 is terminal"));
    }

    /// Decode one record from the front of `bytes`, validating structure
    /// (cell codes, padding, flags, king counts, en-passant geometry) so
    /// corrupt or adversarial input yields an error, never a panic.
    /// Returns the state, its value, and the bytes consumed.
    pub fn unpack_record(&self, bytes: &[u8]) -> Result<(FcState, Wdl, usize), String> {
        let cell_count = self.cell_count() as usize;
        let packed = self.packed_cells_len();
        if bytes.len() < packed + 1 {
            return Err("record truncated".into());
        }
        let used_last = cell_count * 5 - (packed - 1) * 8;
        if used_last < 8 && bytes[packed - 1] >> used_last != 0 {
            return Err("nonzero padding bits".into());
        }
        let mut cells = vec![EMPTY; cell_count];
        let mut kings = [0u32; 2];
        for (cell, slot) in cells.iter_mut().enumerate() {
            let bit = cell * 5;
            let mut word = u16::from(bytes[bit / 8]);
            if bit / 8 + 1 < packed {
                word |= u16::from(bytes[bit / 8 + 1]) << 8;
            }
            let code = ((word >> (bit % 8)) & 0x1f) as u8;
            if code > 24 {
                return Err(format!("invalid cell code {code}"));
            }
            if code != EMPTY {
                let (owner, piece, _) = unpack(code);
                if piece == Piece::King {
                    kings[usize::from(owner == Player::Two)] += 1;
                }
            }
            *slot = code;
        }
        if kings != [1, 1] {
            return Err(format!("kings per side {kings:?}, need exactly one each"));
        }
        let mut cursor = packed;
        let flags = bytes[cursor];
        cursor += 1;
        let to_move = if flags & 1 == 0 {
            Player::One
        } else {
            Player::Two
        };
        let wdl = match (flags >> 1) & 3 {
            0 => Wdl::Loss,
            1 => Wdl::Draw,
            2 => Wdl::Win,
            _ => return Err("invalid WDL bits".into()),
        };
        let castling = std::array::from_fn(|i| flags & (1 << (3 + i)) != 0);
        if castling != [false; 4] && !self.ruleset.castling() {
            return Err("castling rights in a castling-free ruleset".into());
        }
        let ep = if flags & 0x80 != 0 {
            let square = u16::from(*bytes.get(cursor).ok_or("record truncated")?);
            cursor += 1;
            let rank = square / self.width;
            // A crossed square is strictly interior, and the capture
            // logic looks one rank past it in both directions.
            if usize::from(square) >= cell_count || rank == 0 || rank + 1 >= self.height {
                return Err(format!("en-passant square {square} is impossible"));
            }
            Some(square)
        } else {
            None
        };
        let halfmove = u16::from(*bytes.get(cursor).ok_or("record truncated")?);
        cursor += 1;
        let mut core = FcCore {
            cells,
            to_move,
            castling,
            ep,
            halfmove,
            key: 0,
            outcome: None,
        };
        core.key = self.compute_key(&core);
        core.outcome = self.core_outcome(&core);
        Ok((
            FcState {
                core,
                history: Vec::new(),
            },
            wdl,
            cursor,
        ))
    }
}

/// Write a solved tablebase; returns the bytes written.
pub fn write_tablebase<W: std::io::Write>(
    game: &ForwardChess,
    states: &[FcState],
    values: &[Wdl],
    writer: &mut W,
) -> std::io::Result<u64> {
    assert_eq!(states.len(), values.len());
    let mut header = Vec::with_capacity(16);
    header.extend_from_slice(&TB_MAGIC);
    header.push(TB_VERSION);
    header.push(game.ruleset.code());
    header.push(game.width as u8);
    header.push(game.height as u8);
    header.extend_from_slice(&(states.len() as u64).to_le_bytes());
    let mut checksum = Fnv64::new();
    checksum.update(&header);
    writer.write_all(&header)?;
    let mut total = header.len() as u64 + 12;
    let mut record = Vec::new();
    for (state, &wdl) in states.iter().zip(values) {
        record.clear();
        game.pack_record(state, wdl, &mut record);
        checksum.update(&record);
        writer.write_all(&record)?;
        total += record.len() as u64;
    }
    writer.write_all(&TB_FOOTER_MAGIC)?;
    writer.write_all(&checksum.0.to_le_bytes())?;
    writer.flush()?;
    Ok(total)
}

/// Stream-read a tablebase, calling `visit(index, state, value)` per
/// record in stored order. The checksum is verified before any record
/// is decoded. Returns the record count.
pub fn read_tablebase_with<R: std::io::Read>(
    game: &ForwardChess,
    reader: &mut R,
    mut visit: impl FnMut(usize, FcState, Wdl) -> Result<(), String>,
) -> Result<u64, String> {
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .map_err(|e| format!("reading tablebase: {e}"))?;
    if data.len() < 16 + 12 {
        return Err("tablebase truncated".into());
    }
    if data[0..4] != TB_MAGIC {
        return Err("bad tablebase magic".into());
    }
    if data[4] != TB_VERSION {
        return Err(format!("unsupported tablebase version {}", data[4]));
    }
    if data[5] != game.ruleset.code() {
        return Err(format!(
            "tablebase ruleset code {} does not match {}",
            data[5],
            game.ruleset.label()
        ));
    }
    if data[6] != game.width as u8 || data[7] != game.height as u8 {
        return Err("tablebase dimensions do not match the ruleset".into());
    }
    let count = u64::from_le_bytes(data[8..16].try_into().expect("8 bytes"));
    let body_end = data.len() - 12;
    if data[body_end..body_end + 4] != TB_FOOTER_MAGIC {
        return Err("bad tablebase footer magic".into());
    }
    let mut checksum = Fnv64::new();
    checksum.update(&data[..body_end]);
    let stored = u64::from_le_bytes(data[body_end + 4..].try_into().expect("8 bytes"));
    if checksum.0 != stored {
        return Err("tablebase checksum mismatch".into());
    }
    let mut cursor = 16;
    for index in 0..count {
        let (state, wdl, consumed) = game
            .unpack_record(&data[cursor..body_end])
            .map_err(|e| format!("record {index}: {e}"))?;
        cursor += consumed;
        visit(index as usize, state, wdl)?;
    }
    if cursor != body_end {
        return Err("trailing bytes after the final record".into());
    }
    Ok(count)
}

/// Load a whole tablebase into memory.
pub fn read_tablebase<R: std::io::Read>(
    game: &ForwardChess,
    reader: &mut R,
) -> Result<Vec<(FcState, Wdl)>, String> {
    let mut rows = Vec::new();
    read_tablebase_with(game, reader, |_, state, wdl| {
        rows.push((state, wdl));
        Ok(())
    })?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fc(ruleset: Ruleset) -> ForwardChess {
        ForwardChess::new(ruleset)
    }

    /// Independent slow reference generator: enumerates every
    /// (from, to, promotion) triple and validates it with directional
    /// logic written separately from the production generator, then
    /// filters self-check by simulation.
    fn reference_moves(g: &ForwardChess, state: &FcState) -> Vec<FcMove> {
        let core = &state.core;
        let w = i32::from(g.width);
        let mover = core.to_move;
        let mut out = Vec::new();
        for from in 0..g.cell_count() as u16 {
            let code = core.cells[usize::from(from)];
            if code == EMPTY {
                continue;
            }
            let (owner, piece, reversed) = unpack(code);
            if owner != mover {
                continue;
            }
            let ahead = ForwardChess::ahead_of(owner, reversed);
            for to in 0..g.cell_count() as u16 {
                if to == from {
                    continue;
                }
                let df = i32::from(to % g.width) - i32::from(from % g.width);
                let dr = (i32::from(to / g.width) - i32::from(from / g.width)) * ahead;
                let target = core.cells[usize::from(to)];
                let target_own = target != EMPTY && unpack(target).0 == mover;
                let target_enemy = target != EMPTY && unpack(target).0 != mover;
                let clear = || {
                    let (fx, fy) = (i32::from(from) % w, i32::from(from) / w);
                    let (tx, ty) = (i32::from(to) % w, i32::from(to) / w);
                    let (dx, dy) = ((tx - fx).signum(), (ty - fy).signum());
                    let (mut x, mut y) = (fx + dx, fy + dy);
                    while (x, y) != (tx, ty) {
                        if core.cells[(y * w + x) as usize] != EMPTY {
                            return false;
                        }
                        x += dx;
                        y += dy;
                    }
                    true
                };
                let mut candidates: Vec<Option<Piece>> = Vec::new();
                match piece {
                    Piece::Pawn => {
                        let last = i32::from(ForwardChess::last_rank(owner, g.height));
                        let to_rank = i32::from(to / g.width);
                        let promo = to_rank == last;
                        let single_push = df == 0 && dr == 1 && target == EMPTY;
                        let home = i32::from(ForwardChess::pawn_home_rank(owner, g.height));
                        let double_push = df == 0
                            && dr == 2
                            && i32::from(from / g.width) == home
                            && to_rank != last
                            && target == EMPTY
                            && clear();
                        let capture =
                            df.abs() == 1 && dr == 1 && (target_enemy || Some(to) == core.ep);
                        if single_push || double_push || capture {
                            if promo {
                                for p in PROMOTION_CHOICES {
                                    candidates.push(Some(p));
                                }
                            } else {
                                candidates.push(None);
                            }
                        }
                    }
                    Piece::Knight => {
                        if !target_own && ((df.abs() == 1 && dr == 2) || (df.abs() == 2 && dr == 1))
                        {
                            candidates.push(None);
                        }
                    }
                    Piece::King => {
                        if !target_own && (0..=1).contains(&dr) && df.abs() <= 1 {
                            candidates.push(None);
                        }
                        // Castling handled below, outside this scan.
                    }
                    Piece::Rook => {
                        if !target_own && ((df == 0 && dr > 0) || (dr == 0 && df != 0)) && clear() {
                            candidates.push(None);
                        }
                    }
                    Piece::Bishop => {
                        if !target_own && df.abs() == dr && dr > 0 && clear() {
                            candidates.push(None);
                        }
                    }
                    Piece::Queen => {
                        let linear = (df == 0 && dr > 0) || (dr == 0 && df != 0);
                        let diagonal = df.abs() == dr && dr > 0;
                        if !target_own && (linear || diagonal) && clear() {
                            candidates.push(None);
                        }
                    }
                }
                for promotion in candidates {
                    let mv = FcMove {
                        from,
                        to,
                        promotion,
                    };
                    if !g.move_leaves_king_in_check(core, mv) {
                        out.push(mv);
                    }
                }
            }
        }
        // Castling through the production path (its legality conditions
        // are asserted separately by corpus tests) minus plain king
        // steps already covered: reuse generate_castling.
        let king = g.king_square(&core.cells, mover);
        let mut castles = Vec::new();
        g.generate_castling(core, king, &mut castles);
        out.extend(castles);
        out.sort();
        out.dedup();
        out
    }

    fn random_states(g: &ForwardChess, seed: u64, count: usize) -> Vec<FcState> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut states = Vec::new();
        'outer: while states.len() < count {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            loop {
                if g.outcome(&state).is_some() {
                    continue 'outer;
                }
                states.push(state.clone());
                if states.len() >= count {
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
    fn slow_and_optimized_generators_agree() {
        for ruleset in [
            Ruleset::Tiny,
            Ruleset::Small,
            Ruleset::Medium,
            Ruleset::Full,
        ] {
            let g = fc(ruleset);
            for state in random_states(&g, 0xd1ff + ruleset as u64, 250) {
                let mut fast = Vec::new();
                g.legal_moves(&state, &mut fast);
                assert_eq!(fast, reference_moves(&g, &state), "{ruleset:?}");
            }
        }
    }

    #[test]
    fn no_generated_move_violates_orientation() {
        for ruleset in [Ruleset::Small, Ruleset::Medium, Ruleset::Full] {
            let g = fc(ruleset);
            for state in random_states(&g, 0x0413 + ruleset as u64, 300) {
                let mut moves = Vec::new();
                g.legal_moves(&state, &mut moves);
                for mv in moves {
                    let code = state.core.cells[usize::from(mv.from)];
                    let (owner, piece, reversed) = unpack(code);
                    let ahead = ForwardChess::ahead_of(owner, reversed);
                    let dr = (i32::from(mv.to / g.width) - i32::from(mv.from / g.width)) * ahead;
                    assert!(
                        dr >= 0,
                        "{ruleset:?}: {piece:?} moved behind itself: {mv:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn make_unmake_restores_state_and_key() {
        let g = fc(Ruleset::Medium);
        let mut rng = ChaCha12Rng::seed_from_u64(0x11);
        for _ in 0..60 {
            let mut state = g.initial_state();
            let mut moves = Vec::new();
            let mut plies = 0;
            while g.outcome(&state).is_none() && plies < 200 {
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                let before = state.clone();
                let undo = g.make_move(&mut state, mv);
                g.unmake_move(&mut state, mv, undo);
                assert_eq!(state, before);
                assert_eq!(g.compute_key(&state.core), state.core.key);
                g.make_move(&mut state, mv);
                plies += 1;
            }
        }
    }

    // ----- hand-authored rules corpus (FORWARD_CHESS_RULES.md) -----

    #[test]
    fn corpus_black_queen_on_d2_attacks_forward_only() {
        let g = fc(Ruleset::Full);
        let state = g.custom_state(
            &[
                ("d2", Player::Two, Piece::Queen, false),
                ("h8", Player::Two, Piece::King, false),
                ("a8", Player::One, Piece::King, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let queen = g.square("d2");
        let code = state.core.cells[usize::from(queen)];
        // Mandated: no backward attacks.
        for behind in ["d3", "e3", "c3", "d8"] {
            assert!(
                !g.attacks(&state.core.cells, queen, code, g.square(behind)),
                "black queen on d2 must not attack {behind}"
            );
        }
        // Ahead (toward rank 1) and horizontal attacks hold.
        for target in ["d1", "c1", "e1", "a2", "b2", "c2", "e2", "f2", "g2", "h2"] {
            assert!(
                g.attacks(&state.core.cells, queen, code, g.square(target)),
                "black queen on d2 must attack {target}"
            );
        }
    }

    #[test]
    fn corpus_king_is_safe_behind_enemy_pieces() {
        let g = fc(Ruleset::Full);
        // White king on e1 stands BEHIND a natural black rook on e4
        // (the rook only attacks toward rank 1... wait: black forward is
        // toward rank 1, so e1 IS ahead of the rook). Use the reverse:
        // white king on e8 behind the black rook (e8 is behind it).
        let state = g.custom_state(
            &[
                ("e4", Player::Two, Piece::Rook, false),
                ("e8", Player::One, Piece::King, false),
                ("a1", Player::Two, Piece::King, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let rook = g.square("e4");
        let code = state.core.cells[usize::from(rook)];
        assert!(
            !g.attacks(&state.core.cells, rook, code, g.square("e8")),
            "e8 is behind the black rook and must be safe"
        );
        assert!(
            g.attacks(&state.core.cells, rook, code, g.square("e1")),
            "e1 is ahead of the black rook"
        );
        // The white king may legally step to e7 (attacked-square test
        // must respect orientation).
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        assert!(moves
            .iter()
            .any(|m| m.to == g.square("e8") || m.from == g.square("e8")));
    }

    #[test]
    fn corpus_horizontal_moves_remain_legal() {
        let g = fc(Ruleset::Full);
        let state = g.custom_state(
            &[
                ("d4", Player::One, Piece::Rook, false),
                ("a1", Player::One, Piece::King, false),
                ("h8", Player::Two, Piece::King, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        let rook = g.square("d4");
        assert!(moves.contains(&FcMove {
            from: rook,
            to: g.square("a4"),
            promotion: None
        }));
        assert!(moves.contains(&FcMove {
            from: rook,
            to: g.square("h4"),
            promotion: None
        }));
        assert!(moves.contains(&FcMove {
            from: rook,
            to: g.square("d8"),
            promotion: None
        }));
        assert!(
            !moves
                .iter()
                .any(|m| m.from == rook && m.to == g.square("d1")),
            "backward rook slide must not exist"
        );
        // King horizontal steps exist too.
        let king = g.square("a1");
        assert!(moves.contains(&FcMove {
            from: king,
            to: g.square("b1"),
            promotion: None
        }));
    }

    #[test]
    fn corpus_promotion_flips_orientation_and_attacks() {
        let g = fc(Ruleset::Full);
        let mut state = g.custom_state(
            &[
                ("b7", Player::One, Piece::Pawn, false),
                ("a1", Player::One, Piece::King, false),
                ("h8", Player::Two, Piece::King, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        let promo = FcMove {
            from: g.square("b7"),
            to: g.square("b8"),
            promotion: Some(Piece::Rook),
        };
        assert!(moves.contains(&promo));
        g.make_move(&mut state, promo);
        let rook = g.square("b8");
        let code = state.core.cells[usize::from(rook)];
        let (owner, piece, reversed) = unpack(code);
        assert_eq!((owner, piece, reversed), (Player::One, Piece::Rook, true));
        // The reversed white rook attacks toward rank 1 and horizontally,
        // never toward rank 8 (it is ON rank 8; attacks b1..b7 and a8..h8).
        assert!(g.attacks(&state.core.cells, rook, code, g.square("b1")));
        assert!(g.attacks(&state.core.cells, rook, code, g.square("a8")));
        assert!(g.attacks(&state.core.cells, rook, code, g.square("h8")));
        // And a reversed piece may MOVE backward (its ahead direction):
        let mut black_reply = Vec::new();
        g.legal_moves(&state, &mut black_reply);
        // Skip a black move, then check white rook's backward slide exists.
        g.make_move(&mut state, black_reply[0]);
        let mut white_moves = Vec::new();
        g.legal_moves(&state, &mut white_moves);
        assert!(
            white_moves
                .iter()
                .any(|m| m.from == rook && m.to / g.width() < 7),
            "reversed rook must slide toward rank 1"
        );
    }

    #[test]
    fn corpus_castling_is_horizontal_and_fully_checked() {
        let g = fc(Ruleset::Full);
        // Clean castling both ways.
        let state = g.custom_state(
            &[
                ("e1", Player::One, Piece::King, false),
                ("h1", Player::One, Piece::Rook, false),
                ("a1", Player::One, Piece::Rook, false),
                ("e8", Player::Two, Piece::King, false),
            ],
            Player::One,
            [true, true, false, false],
            None,
        );
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        let short = FcMove {
            from: g.square("e1"),
            to: g.square("g1"),
            promotion: None,
        };
        let long = FcMove {
            from: g.square("e1"),
            to: g.square("c1"),
            promotion: None,
        };
        assert!(moves.contains(&short) && moves.contains(&long));
        let mut s = state.clone();
        g.make_move(&mut s, short);
        assert_eq!(
            unpack(s.core.cells[usize::from(g.square("f1"))]).1,
            Piece::Rook
        );
        assert_eq!(
            unpack(s.core.cells[usize::from(g.square("g1"))]).1,
            Piece::King
        );

        // A black REVERSED bishop attacking f1 (from behind-looking
        // geometry) must forbid short castling: place it so f1 is ahead
        // of it. Black reversed bishop's ahead = +rank; from h3? h3->
        // ahead diag: g4... we need attack ON f1: reversed black bishop
        // at "g2"? ahead=+1: attacks f3/h3... no. Natural black bishop
        // at h3 attacks g2? natural black ahead=-1: h3 -> g2, f1 ✓.
        let state = g.custom_state(
            &[
                ("e1", Player::One, Piece::King, false),
                ("h1", Player::One, Piece::Rook, false),
                ("e8", Player::Two, Piece::King, false),
                ("h3", Player::Two, Piece::Bishop, false),
            ],
            Player::One,
            [true, false, false, false],
            None,
        );
        g.legal_moves(&state, &mut moves);
        assert!(
            !moves
                .iter()
                .any(|m| m.from == g.square("e1") && m.to == g.square("g1")),
            "castling through an attacked square must be illegal"
        );
    }

    #[test]
    fn corpus_en_passant_is_correctly_timed() {
        let g = fc(Ruleset::Full);
        // White pawn e2 double-steps past a black pawn on d4; black may
        // capture en passant immediately, and only immediately.
        let mut state = g.custom_state(
            &[
                ("e2", Player::One, Piece::Pawn, false),
                ("d4", Player::Two, Piece::Pawn, false),
                ("a1", Player::One, Piece::King, false),
                ("h8", Player::Two, Piece::King, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        let double = FcMove {
            from: g.square("e2"),
            to: g.square("e4"),
            promotion: None,
        };
        assert!(moves.contains(&double));
        g.make_move(&mut state, double);
        assert_eq!(state.core.ep, Some(g.square("e3")));
        g.legal_moves(&state, &mut moves);
        let ep_capture = FcMove {
            from: g.square("d4"),
            to: g.square("e3"),
            promotion: None,
        };
        assert!(moves.contains(&ep_capture), "en passant must be offered");
        // Take it: the double-stepped pawn disappears.
        let mut s = state.clone();
        g.make_move(&mut s, ep_capture);
        assert_eq!(s.core.cells[usize::from(g.square("e4"))], EMPTY);
        // Decline it: after any other move the chance expires.
        let other = moves.iter().find(|m| **m != ep_capture).copied().unwrap();
        g.make_move(&mut state, other);
        let mut white_moves = Vec::new();
        g.legal_moves(&state, &mut white_moves);
        let _ = white_moves;
        assert_eq!(state.core.ep, None, "en passant expires after one ply");
    }

    #[test]
    fn repetition_and_fifty_move_draws() {
        let g = fc(Ruleset::Full);
        // Kings shuffling horizontally repeat the start threefold.
        let mut state = g.custom_state(
            &[
                ("a1", Player::One, Piece::King, false),
                ("h8", Player::Two, Piece::King, false),
                ("a8", Player::One, Piece::Rook, false),
            ],
            Player::Two,
            [false; 4],
            None,
        );
        let shuffle = ["h8g8", "a1b1", "g8h8", "b1a1"];
        for _ in 0..2 {
            for step in shuffle {
                assert!(g.outcome(&state).is_none(), "premature end");
                let mv = FcMove {
                    from: g.square(&step[..2]),
                    to: g.square(&step[2..]),
                    promotion: None,
                };
                g.make_move(&mut state, mv);
            }
        }
        assert_eq!(
            g.outcome(&state),
            Some(Outcome::Draw),
            "threefold repetition"
        );

        // Fifty-move rule: halfmove clock reaching 100 draws.
        let mut state = g.custom_state(
            &[
                ("a1", Player::One, Piece::King, false),
                ("h8", Player::Two, Piece::King, false),
                ("d4", Player::One, Piece::Rook, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        state.core.halfmove = 99;
        let mv = FcMove {
            from: g.square("d4"),
            to: g.square("e4"),
            promotion: None,
        };
        g.make_move(&mut state, mv);
        assert_eq!(g.outcome(&state), Some(Outcome::Draw), "fifty-move rule");
    }

    #[test]
    fn checkmate_and_stalemate_with_oriented_attacks() {
        let g = fc(Ruleset::Full);
        // Mate: black king on h8 cannot retreat (backward is behind it);
        // white reversed rook on h7? Simpler mandated-style mate: black
        // king h8, white rook h1 attacking up the file, white rook g1
        // covering g8: black king has only forward-relative squares
        // (toward rank 1): g7/h7 — cover those too.
        let state = g.custom_state(
            &[
                ("h8", Player::Two, Piece::King, false),
                ("h1", Player::One, Piece::Rook, false),
                ("g1", Player::One, Piece::Rook, false),
                ("a1", Player::One, Piece::King, false),
                ("b7", Player::One, Piece::Queen, false),
            ],
            Player::Two,
            [false; 4],
            None,
        );
        // Black king squares: g8 (Rg1), g7 (Qb7 horizontal), h7 (Qb7? b7->h7 horizontal ✓, Rh1 ✓).
        assert_eq!(
            g.outcome(&state),
            Some(Outcome::Win(Player::One)),
            "checkmate under oriented attacks"
        );

        // Stalemate: black king unattacked but all its 5 oriented squares
        // are attacked or off-board. (A rook on a8 would attack h8 itself
        // — check, not stalemate — so cover rank 7 with a rook on a7 and
        // the g-file with the rook on g1; king h8 stays unattacked.)
        let state = g.custom_state(
            &[
                ("h8", Player::Two, Piece::King, false),
                ("g1", Player::One, Piece::Rook, false),
                ("a7", Player::One, Piece::Rook, false),
                ("a1", Player::One, Piece::King, false),
            ],
            Player::Two,
            [false; 4],
            None,
        );
        assert_eq!(
            g.outcome(&state),
            Some(Outcome::Draw),
            "stalemate under oriented attacks"
        );
    }

    #[test]
    fn initial_positions_are_playable_and_fair_shaped() {
        for ruleset in [
            Ruleset::Tiny,
            Ruleset::Small,
            Ruleset::Medium,
            Ruleset::Full,
        ] {
            let g = fc(ruleset);
            let state = g.initial_state();
            assert!(g.outcome(&state).is_none(), "{ruleset:?}");
            let mut moves = Vec::new();
            g.legal_moves(&state, &mut moves);
            assert!(!moves.is_empty());
            // Feature/action sanity.
            let mut features = Vec::new();
            g.encode_features(&state, &mut features);
            assert!(features.iter().all(|&f| (f as usize) < g.feature_count()));
            for &mv in &moves {
                assert!((g.action_id(&state, mv) as usize) < g.action_count());
            }
        }
    }

    #[test]
    fn corpus_castling_rights_die_with_king_moves_both_colours() {
        let g = fc(Ruleset::Full);
        // Black's king home is d8 (the 180-degree-rotated e1), NOT e8.
        assert_eq!(g.king_home(Player::One), g.square("e1"));
        assert_eq!(g.king_home(Player::Two), g.square("d8"));
        let state = g.custom_state(
            &[
                ("e1", Player::One, Piece::King, false),
                ("a1", Player::One, Piece::Rook, false),
                ("h1", Player::One, Piece::Rook, false),
                ("d8", Player::Two, Piece::King, false),
                ("a8", Player::Two, Piece::Rook, false),
                ("h8", Player::Two, Piece::Rook, false),
            ],
            Player::One,
            [true; 4],
            None,
        );
        let mut s = state.clone();
        g.make_move(
            &mut s,
            FcMove {
                from: g.square("e1"),
                to: g.square("e2"),
                promotion: None,
            },
        );
        assert_eq!(s.core.castling, [false, false, true, true]);
        g.make_move(
            &mut s,
            FcMove {
                from: g.square("d8"),
                to: g.square("d7"),
                promotion: None,
            },
        );
        assert_eq!(s.core.castling, [false; 4]);
    }

    #[test]
    fn corpus_black_castles_from_d8() {
        let g = fc(Ruleset::Full);
        let state = g.custom_state(
            &[
                ("e1", Player::One, Piece::King, false),
                ("d8", Player::Two, Piece::King, false),
                ("a8", Player::Two, Piece::Rook, false),
                ("h8", Player::Two, Piece::Rook, false),
            ],
            Player::Two,
            [false, false, true, true],
            None,
        );
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        let toward_h = FcMove {
            from: g.square("d8"),
            to: g.square("f8"),
            promotion: None,
        };
        let toward_a = FcMove {
            from: g.square("d8"),
            to: g.square("b8"),
            promotion: None,
        };
        assert!(moves.contains(&toward_h), "castle toward h8 from d8");
        assert!(moves.contains(&toward_a), "castle toward a8 from d8");
        let mut s = state.clone();
        g.make_move(&mut s, toward_h);
        assert_eq!(
            unpack(s.core.cells[usize::from(g.square("e8"))]).1,
            Piece::Rook
        );
        assert_eq!(
            unpack(s.core.cells[usize::from(g.square("f8"))]).1,
            Piece::King
        );
        assert_eq!(s.core.castling, [false; 4]);
        let mut s = state.clone();
        g.make_move(&mut s, toward_a);
        assert_eq!(
            unpack(s.core.cells[usize::from(g.square("c8"))]).1,
            Piece::Rook
        );
        assert_eq!(
            unpack(s.core.cells[usize::from(g.square("b8"))]).1,
            Piece::King
        );
    }

    #[test]
    fn corpus_no_phantom_castle_off_home_even_with_forced_rights() {
        let g = fc(Ruleset::Full);
        // The D031 crash shape: black king on b8 holding a stale
        // queenside right; the two-file step would wrap file -1 to a
        // cell on another rank (it once captured the white king on h7).
        let state = g.custom_state(
            &[
                ("h7", Player::One, Piece::King, false),
                ("b8", Player::Two, Piece::King, false),
                ("a8", Player::Two, Piece::Rook, false),
            ],
            Player::Two,
            [false, false, false, true],
            None,
        );
        let mut moves = Vec::new();
        g.legal_moves(&state, &mut moves);
        for mv in &moves {
            if unpack(state.core.cells[usize::from(mv.from)]).1 == Piece::King {
                let df = i32::from(mv.to % g.width()) - i32::from(mv.from % g.width());
                let dr = i32::from(mv.to / g.width()) - i32::from(mv.from / g.width());
                assert!(
                    df.abs() <= 1 && dr.abs() <= 1,
                    "phantom castle generated: {}",
                    g.format_move(*mv)
                );
            }
        }
        // Mirror case: white king on g1 with a stale kingside right —
        // the two-file step would spill past file h onto rank 2.
        let state = g.custom_state(
            &[
                ("g1", Player::One, Piece::King, false),
                ("h1", Player::One, Piece::Rook, false),
                ("d8", Player::Two, Piece::King, false),
            ],
            Player::One,
            [true, false, false, false],
            None,
        );
        g.legal_moves(&state, &mut moves);
        for mv in &moves {
            if mv.from == g.square("g1") {
                let df = i32::from(mv.to % g.width()) - i32::from(mv.from % g.width());
                let dr = i32::from(mv.to / g.width()) - i32::from(mv.from / g.width());
                assert!(
                    df.abs() <= 1 && dr.abs() <= 1,
                    "phantom castle generated: {}",
                    g.format_move(*mv)
                );
            }
        }
    }

    /// States visited by seeded random playouts (including terminals).
    fn seeded_playout_states(g: &ForwardChess, games: u64, max_plies: u32) -> Vec<FcState> {
        let mut rng = ChaCha12Rng::seed_from_u64(0x7ab1_eba5_e000_0000 + games);
        let mut out = Vec::new();
        let mut moves = Vec::new();
        for _ in 0..games {
            let mut state = g.initial_state();
            out.push(state.clone());
            for _ in 0..max_plies {
                if g.outcome(&state).is_some() {
                    break;
                }
                g.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                g.make_move(&mut state, mv);
                out.push(state.clone());
            }
        }
        out
    }

    fn wdl_cycle(i: usize) -> Wdl {
        match i % 3 {
            0 => Wdl::Loss,
            1 => Wdl::Draw,
            _ => Wdl::Win,
        }
    }

    #[test]
    fn tablebase_record_roundtrip_across_rulesets() {
        for ruleset in [
            Ruleset::Tiny,
            Ruleset::Small,
            Ruleset::Medium,
            Ruleset::Full,
        ] {
            let g = fc(ruleset);
            let mut buf = Vec::new();
            let mut ep_seen = 0u32;
            for (i, state) in seeded_playout_states(&g, 30, 120).iter().enumerate() {
                let wdl = wdl_cycle(i);
                buf.clear();
                g.pack_record(state, wdl, &mut buf);
                let (back, wdl_back, consumed) = g.unpack_record(&buf).unwrap();
                assert_eq!(consumed, buf.len(), "{ruleset:?}");
                assert_eq!(wdl_back, wdl);
                assert_eq!(back.core.cells, state.core.cells);
                assert_eq!(back.core.to_move, state.core.to_move);
                assert_eq!(back.core.castling, state.core.castling);
                assert_eq!(back.core.ep, state.core.ep);
                assert_eq!(back.core.halfmove, state.core.halfmove);
                assert_eq!(back.core.key, state.core.key);
                // History is not stored, so the unpacked outcome must
                // match the history-free outcome of the original core.
                assert_eq!(back.core.outcome, g.core_outcome(&state.core));
                if back.core.outcome.is_none() {
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    g.legal_moves(state, &mut a);
                    g.legal_moves(&back, &mut b);
                    if state.core.outcome.is_none() {
                        assert_eq!(a, b, "legal moves must survive the round trip");
                    }
                }
                // Re-packing must reproduce the exact bytes.
                let mut again = Vec::new();
                g.pack_record(&back, wdl, &mut again);
                assert_eq!(again, buf);
                ep_seen += u32::from(state.core.ep.is_some());
            }
            if matches!(ruleset, Ruleset::Medium | Ruleset::Full) {
                assert!(ep_seen > 0, "{ruleset:?} playouts must exercise en passant");
            }
        }
    }

    #[test]
    fn tablebase_file_roundtrip_on_solved_tiny() {
        use crate::search::solve_retrograde;
        let g = fc(Ruleset::Tiny);
        let solution = solve_retrograde(&g, 5_000_000).unwrap();
        let mut file = Vec::new();
        let bytes = write_tablebase(&g, &solution.states, &solution.values, &mut file).unwrap();
        assert_eq!(bytes as usize, file.len());
        let rows = read_tablebase(&g, &mut file.as_slice()).unwrap();
        assert_eq!(rows.len(), solution.states.len());
        for (index, (state, wdl)) in rows.iter().enumerate() {
            assert_eq!(*wdl, solution.values[index]);
            assert_eq!(state.core, solution.states[index].core, "record {index}");
        }
    }

    #[test]
    fn tablebase_rejects_corruption_without_panicking() {
        let g = fc(Ruleset::Small);
        let states = seeded_playout_states(&g, 6, 40);
        let values: Vec<Wdl> = (0..states.len()).map(wdl_cycle).collect();
        let mut file = Vec::new();
        write_tablebase(&g, &states, &values, &mut file).unwrap();
        assert_eq!(
            read_tablebase(&g, &mut file.as_slice()).unwrap().len(),
            states.len()
        );
        // Every strict prefix must error.
        for len in 0..file.len() {
            assert!(
                read_tablebase(&g, &mut &file[..len]).is_err(),
                "prefix {len}"
            );
        }
        // Every single-bit flip must error.
        for byte in 0..file.len() {
            for bit in 0..8 {
                let mut bad = file.clone();
                bad[byte] ^= 1 << bit;
                assert!(
                    read_tablebase(&g, &mut bad.as_slice()).is_err(),
                    "byte {byte} bit {bit}"
                );
            }
        }
        // Trailing garbage must error.
        let mut padded = file.clone();
        padded.push(0);
        assert!(read_tablebase(&g, &mut padded.as_slice()).is_err());
        // A file for one ruleset must not load as another.
        assert!(read_tablebase(&fc(Ruleset::Tiny), &mut file.as_slice()).is_err());
    }

    /// Header + records + valid checksummed footer, for crafting files
    /// whose corruption is structural rather than bitwise.
    fn assemble_tablebase(g: &ForwardChess, count: u64, records: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&TB_MAGIC);
        data.push(TB_VERSION);
        data.push(g.ruleset.code());
        data.push(g.width as u8);
        data.push(g.height as u8);
        data.extend_from_slice(&count.to_le_bytes());
        data.extend_from_slice(records);
        let mut checksum = Fnv64::new();
        checksum.update(&data);
        data.extend_from_slice(&TB_FOOTER_MAGIC);
        data.extend_from_slice(&checksum.0.to_le_bytes());
        data
    }

    #[test]
    fn tablebase_rejects_crafted_invalid_records() {
        let g = fc(Ruleset::Tiny);
        // Kingless record: all cells empty, flags 0, halfmove 0.
        let empty_record = vec![0u8; g.packed_cells_len() + 2];
        let file = assemble_tablebase(&g, 1, &empty_record);
        let err = read_tablebase(&g, &mut file.as_slice()).unwrap_err();
        assert!(err.contains("king"), "{err}");
        // Invalid WDL bits (both set).
        let mut record = Vec::new();
        g.pack_record(&g.initial_state(), Wdl::Draw, &mut record);
        let flags_index = g.packed_cells_len();
        let mut bad = record.clone();
        bad[flags_index] |= 3 << 1;
        let file = assemble_tablebase(&g, 1, &bad);
        let err = read_tablebase(&g, &mut file.as_slice()).unwrap_err();
        assert!(err.contains("WDL"), "{err}");
        // Count larger than the records present.
        let file = assemble_tablebase(&g, 2, &record);
        assert!(read_tablebase(&g, &mut file.as_slice()).is_err());
        // Count smaller than the records present.
        let file = assemble_tablebase(&g, 0, &record);
        let err = read_tablebase(&g, &mut file.as_slice()).unwrap_err();
        assert!(err.contains("trailing"), "{err}");
    }

    #[test]
    fn tablebase_reader_survives_fuzz() {
        let g = fc(Ruleset::Medium);
        let mut rng = ChaCha12Rng::seed_from_u64(0x7462_f022_0000_0001);
        for _ in 0..20_000 {
            let len = rng.gen_range(0..400usize);
            let data: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let _ = read_tablebase(&g, &mut data.as_slice());
            let _ = g.unpack_record(&data);
        }
    }
}

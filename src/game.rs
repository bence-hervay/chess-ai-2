//! The core game interface.
//!
//! Every supported game implements [`Game`]. The trait exposes only
//! rule-level facts: legal moves, state transitions, terminal outcomes,
//! hashing, raw feature encoding, and stable action identifiers. It must
//! never expose strategic conclusions (material, mobility, king safety, ...).

/// One of the two players in an alternating-turn zero-sum game.
///
/// `One` is the player who moves first from the initial state.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Player {
    One,
    Two,
}

impl Player {
    pub fn opponent(self) -> Player {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

/// Terminal result of a finished game.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Outcome {
    Win(Player),
    Draw,
}

/// A sparse feature identifier. Feature IDs encode raw rule-state facts
/// from the perspective of the player to move.
pub type FeatureId = u32;

/// A stable integer identifier for a move, independent of move ordering.
pub type ActionId = u32;

/// A deterministic, two-player, alternating-turn, perfect-information,
/// zero-sum game.
///
/// Contract:
/// - `legal_moves` must not mutate the state and must be deterministic;
/// - `make_move` may only be called with a move returned by `legal_moves`
///   on a non-terminal state;
/// - `unmake_move` must restore the state to exact equality with the state
///   before the corresponding `make_move`, including `position_key`;
/// - `position_key` must be deterministic across processes and runs;
/// - `encode_features` must emit raw facts in a deterministic order, from
///   the perspective of the player to move where practical.
pub trait Game: Send + Sync + 'static {
    type State: Clone + PartialEq + std::fmt::Debug + Send + Sync;
    type Move: Copy + Eq + Ord + std::fmt::Debug + Send + Sync;
    type Undo;

    fn initial_state(&self) -> Self::State;

    fn side_to_move(&self, state: &Self::State) -> Player;

    /// Append all legal moves to `moves`. `moves` is cleared first.
    fn legal_moves(&self, state: &Self::State, moves: &mut Vec<Self::Move>);

    fn make_move(&self, state: &mut Self::State, mv: Self::Move) -> Self::Undo;

    fn unmake_move(&self, state: &mut Self::State, mv: Self::Move, undo: Self::Undo);

    /// `Some(outcome)` iff the state is terminal.
    fn outcome(&self, state: &Self::State) -> Option<Outcome>;

    /// 64-bit hash of the position, including the side to move.
    fn position_key(&self, state: &Self::State) -> u64;

    /// Append the active sparse feature IDs to `features`. `features` is
    /// cleared first.
    fn encode_features(&self, state: &Self::State, features: &mut Vec<FeatureId>);

    /// Stable action ID of a legal move in this state.
    fn action_id(&self, state: &Self::State, mv: Self::Move) -> ActionId;

    /// Rule-level fact: does this move capture material or promote?
    /// Quiescence search extends only these moves at the horizon. The
    /// default (no tactical moves) makes quiescence a no-op for games
    /// without a quiet/tactical distinction.
    fn is_tactical(&self, state: &Self::State, mv: Self::Move) -> bool {
        let _ = (state, mv);
        false
    }
}

//! Structured position measurements (SHSD program §6.2, §16, §21).
//!
//! Extractors compute *measurements* of a position — counts, geometry,
//! attack relations — from the side to move's perspective. They never
//! assert strategy: every sign and magnitude is learned downstream
//! (§6.2). Extractors are game-specific implementations of one generic
//! trait, exactly like `Game` implementations.
//!
//! Perspective contract: features must be invariant under any exact
//! symmetry of the game that swaps the players (for Forward Chess, the
//! 180°-rotation colour swap). This is the §69.2 symmetry test for
//! structured evaluation, and it is what makes one weight vector serve
//! both sides in negamax search.

pub mod chess;
pub mod forward_chess;

use crate::game::Game;

/// A sparse structured feature vector entry: `(feature index, value)`.
pub type FeatureEntry = (u32, f32);

/// Computes structured measurements of a position, side-to-move
/// perspective. `&mut self` allows internal scratch buffers; extraction
/// must remain a pure function of `(game, state)`.
pub trait FeatureExtractor<G: Game> {
    /// Total dimension of the feature space.
    fn dimension(&self) -> usize;

    /// Clear `out` and append the active features. Indices must be
    /// unique and `< dimension()`.
    fn extract(&mut self, game: &G, state: &G::State, out: &mut Vec<FeatureEntry>);

    /// Stable, human-readable name of a feature index (diagnostics,
    /// weight inspection, the parameter ledger).
    fn feature_name(&self, index: u32) -> String;
}

/// Computes state-action (move) measurements for ordering models
/// (§35.2). Same purity contract as [`FeatureExtractor`].
pub trait MoveFeatures<G: Game> {
    fn dimension(&self) -> usize;

    /// Clear `out` and append the move's features, mover perspective.
    fn extract(&mut self, game: &G, state: &G::State, mv: G::Move, out: &mut Vec<FeatureEntry>);

    /// Stable, human-readable name of a feature index.
    fn feature_name(&self, index: u32) -> String;
}

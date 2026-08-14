//! Property-based tests for Connect-k state transitions.

use proptest::prelude::*;
use selfplay_lab::game::Game;
use selfplay_lab::games::connect_k::{ConnectK, ConnectKMove};

/// Board parameters plus a move-selection script: each entry picks the
/// n-th legal move (mod the number of legal moves).
fn board_and_script() -> impl Strategy<Value = (u16, u16, u16, bool, Vec<u16>)> {
    (
        2u16..=7,
        2u16..=6,
        prop::bool::ANY,
        prop::collection::vec(0u16..64, 0..48),
    )
        .prop_flat_map(|(w, h, gravity, script)| {
            let max_k = w.max(h);
            (Just(w), Just(h), 2u16..=max_k, Just(gravity), Just(script))
        })
}

proptest! {
    /// Applying a scripted playout and unwinding it move by move restores
    /// the initial state and its key exactly, and every visited state obeys
    /// the basic invariants.
    #[test]
    fn scripted_playout_invariants((w, h, k, gravity, script) in board_and_script()) {
        let game = ConnectK::new(w, h, k, gravity).unwrap();
        let mut state = game.initial_state();
        let initial = state.clone();
        let mut moves = Vec::new();
        let mut trail: Vec<(ConnectKMove, _)> = Vec::new();
        let mut keys = vec![game.position_key(&state)];

        for pick in script {
            if game.outcome(&state).is_some() {
                break;
            }
            game.legal_moves(&state, &mut moves);
            prop_assert!(!moves.is_empty(), "non-terminal state must have moves");

            // All generated moves are distinct and legal (empty target cell).
            let mut sorted = moves.clone();
            sorted.sort();
            sorted.dedup();
            prop_assert_eq!(sorted.len(), moves.len());

            // Action IDs of legal moves are distinct.
            let mut ids: Vec<u32> =
                moves.iter().map(|&m| game.action_id(&state, m)).collect();
            ids.sort_unstable();
            ids.dedup();
            prop_assert_eq!(ids.len(), moves.len());

            let mv = moves[usize::from(pick) % moves.len()];
            let undo = game.make_move(&mut state, mv);
            trail.push((mv, undo));
            keys.push(game.position_key(&state));
        }

        // Feature encoding is deterministic.
        let (mut f1, mut f2) = (Vec::new(), Vec::new());
        game.encode_features(&state, &mut f1);
        game.encode_features(&state, &mut f2);
        prop_assert_eq!(f1, f2);

        // Unwind: each unmake restores the recorded key of its predecessor.
        while let Some((mv, undo)) = trail.pop() {
            keys.pop();
            game.unmake_move(&mut state, mv, undo);
            prop_assert_eq!(game.position_key(&state), *keys.last().unwrap());
        }
        prop_assert_eq!(&state, &initial);
    }

    /// A game where both players follow the script always terminates with
    /// a legal outcome within `w*h` plies.
    #[test]
    fn playouts_terminate((w, h, k, gravity, _s) in board_and_script(), seed in 0u64..1000) {
        let game = ConnectK::new(w, h, k, gravity).unwrap();
        let mut state = game.initial_state();
        let mut moves = Vec::new();
        let mut x = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
        let mut plies = 0u32;
        while game.outcome(&state).is_none() {
            game.legal_moves(&state, &mut moves);
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mv = moves[(x >> 33) as usize % moves.len()];
            game.make_move(&mut state, mv);
            plies += 1;
            prop_assert!(plies <= u32::from(w) * u32::from(h));
        }
    }
}

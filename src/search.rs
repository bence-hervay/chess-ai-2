//! Search: the exact solver (research/test oracle) and the production
//! alpha–beta search.
//!
//! Score convention (all scores are from the side to move's perspective):
//! - a forced win at ply `p` from the search root scores `SCORE_WIN - p`,
//!   so shorter wins score higher;
//! - a draw scores 0;
//! - non-terminal leaves score the evaluator's value, clamped to
//!   `[-SCORE_EVAL_MAX, SCORE_EVAL_MAX]`, far below any terminal score.

use crate::game::{Game, Outcome, Player};
use serde::{Deserialize, Serialize};

pub const SCORE_WIN: i32 = 32_000;
/// Scores with magnitude above this bound are proven win/loss scores.
pub const SCORE_TERMINAL_BOUND: i32 = 30_000;
/// Maximum magnitude a leaf evaluator may return.
pub const SCORE_EVAL_MAX: i32 = 1_000;

const SCORE_INF: i32 = i32::MAX - 1;

/// Game-theoretic value from the side to move's perspective.
/// Ordered so that `Loss < Draw < Win`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Serialize, Deserialize)]
pub enum Wdl {
    Loss,
    Draw,
    Win,
}

impl Wdl {
    pub fn flip(self) -> Wdl {
        match self {
            Wdl::Loss => Wdl::Win,
            Wdl::Draw => Wdl::Draw,
            Wdl::Win => Wdl::Loss,
        }
    }

    pub fn from_outcome(outcome: Outcome, side_to_move: Player) -> Wdl {
        match outcome {
            Outcome::Draw => Wdl::Draw,
            Outcome::Win(winner) if winner == side_to_move => Wdl::Win,
            Outcome::Win(_) => Wdl::Loss,
        }
    }

    /// Interpret the score of a search that solved the position (searched
    /// to terminal everywhere or with an exact leaf oracle).
    pub fn from_solved_score(score: i32) -> Wdl {
        if score > SCORE_TERMINAL_BOUND {
            Wdl::Win
        } else if score < -SCORE_TERMINAL_BOUND {
            Wdl::Loss
        } else {
            Wdl::Draw
        }
    }
}

/// Exact memoized negamax solver over the reachable state DAG.
///
/// No pruning: every reachable state below the query is solved exactly
/// once, so node counts equal DAG edges plus memo hits. This is the
/// research/test oracle, not the production search engine.
pub struct ExactSolver {
    memo: std::collections::HashMap<u64, Wdl>,
    pub nodes: u64,
    pub memo_hits: u64,
}

impl Default for ExactSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactSolver {
    pub fn new() -> ExactSolver {
        ExactSolver {
            memo: std::collections::HashMap::new(),
            nodes: 0,
            memo_hits: 0,
        }
    }

    /// Number of distinct solved (non-terminal) states.
    pub fn solved_states(&self) -> usize {
        self.memo.len()
    }

    pub fn solve<G: Game>(&mut self, game: &G, state: &mut G::State) -> Wdl {
        self.nodes += 1;
        if let Some(outcome) = game.outcome(state) {
            return Wdl::from_outcome(outcome, game.side_to_move(state));
        }
        let key = game.position_key(state);
        if let Some(&wdl) = self.memo.get(&key) {
            self.memo_hits += 1;
            return wdl;
        }
        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        debug_assert!(!moves.is_empty(), "non-terminal state without moves");
        let mut best = Wdl::Loss;
        for &mv in &moves {
            let undo = game.make_move(state, mv);
            let value = self.solve(game, state).flip();
            game.unmake_move(state, mv, undo);
            best = best.max(value);
        }
        self.memo.insert(key, best);
        best
    }

    /// The set of game-theoretically optimal moves (category-optimal:
    /// every move preserving the position's exact WDL value).
    pub fn optimal_moves<G: Game>(
        &mut self,
        game: &G,
        state: &mut G::State,
        optimal: &mut Vec<G::Move>,
    ) -> Wdl {
        optimal.clear();
        let value = self.solve(game, state);
        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        for &mv in &moves {
            let undo = game.make_move(state, mv);
            let child = self.solve(game, state).flip();
            game.unmake_move(state, mv, undo);
            if child == value {
                optimal.push(mv);
            }
        }
        value
    }
}

/// A solved non-terminal position reported by [`enumerate_solved`].
pub struct SolvedPosition<'a, G: Game> {
    pub state: &'a G::State,
    pub ply: u32,
    pub value: Wdl,
    pub legal: &'a [G::Move],
    pub optimal: &'a [G::Move],
}

/// Depth-first enumeration of every reachable state from the initial
/// position, deduplicated by position key. `visit` is called exactly once
/// per non-terminal state, in deterministic DFS preorder.
pub fn enumerate_solved<G: Game>(
    game: &G,
    solver: &mut ExactSolver,
    mut visit: impl FnMut(SolvedPosition<'_, G>),
) {
    fn recurse<G: Game>(
        game: &G,
        solver: &mut ExactSolver,
        state: &mut G::State,
        ply: u32,
        seen: &mut std::collections::HashSet<u64>,
        visit: &mut impl FnMut(SolvedPosition<'_, G>),
    ) {
        if game.outcome(state).is_some() {
            return;
        }
        if !seen.insert(game.position_key(state)) {
            return;
        }
        let mut optimal = Vec::new();
        let value = solver.optimal_moves(game, state, &mut optimal);
        let mut legal = Vec::new();
        game.legal_moves(state, &mut legal);
        visit(SolvedPosition {
            state,
            ply,
            value,
            legal: &legal,
            optimal: &optimal,
        });
        for &mv in &legal {
            let undo = game.make_move(state, mv);
            recurse(game, solver, state, ply + 1, seen, visit);
            game.unmake_move(state, mv, undo);
        }
    }
    let mut state = game.initial_state();
    let mut seen = std::collections::HashSet::new();
    recurse(game, solver, &mut state, 0, &mut seen, &mut visit);
}

/// Plain full-depth negamax without memoization or pruning. Test/research
/// reference only; node counts grow with the full game tree.
pub fn exhaustive_negamax<G: Game>(
    game: &G,
    state: &mut G::State,
    ply: u32,
    nodes: &mut u64,
) -> i32 {
    *nodes += 1;
    if let Some(outcome) = game.outcome(state) {
        return terminal_score(outcome, game.side_to_move(state), ply);
    }
    let mut moves = Vec::new();
    game.legal_moves(state, &mut moves);
    let mut best = -SCORE_INF;
    for &mv in &moves {
        let undo = game.make_move(state, mv);
        let value = -exhaustive_negamax(game, state, ply + 1, nodes);
        game.unmake_move(state, mv, undo);
        best = best.max(value);
    }
    best
}

fn terminal_score(outcome: Outcome, side_to_move: Player, ply: u32) -> i32 {
    match outcome {
        Outcome::Draw => 0,
        Outcome::Win(winner) if winner == side_to_move => SCORE_WIN - ply as i32,
        Outcome::Win(_) => -(SCORE_WIN - ply as i32),
    }
}

/// Move iteration order. `Natural` follows the game's generation order
/// (stable action-ID order); `Reversed` deliberately degrades ordering for
/// controlled experiments.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveOrdering {
    Natural,
    Reversed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry<M> {
    key: u64,
    score: i32,
    depth: u32,
    bound: Bound,
    best: Option<M>,
}

struct TranspositionTable<M> {
    entries: Vec<Option<TtEntry<M>>>,
    mask: usize,
    probes: u64,
    hits: u64,
}

impl<M: Copy> TranspositionTable<M> {
    fn new(log2_entries: u32) -> TranspositionTable<M> {
        let len = 1usize << log2_entries;
        TranspositionTable {
            entries: vec![None; len],
            mask: len - 1,
            probes: 0,
            hits: 0,
        }
    }

    fn probe(&mut self, key: u64) -> Option<TtEntry<M>> {
        self.probes += 1;
        let entry = self.entries[(key as usize) & self.mask];
        match entry {
            Some(e) if e.key == key => {
                self.hits += 1;
                Some(e)
            }
            _ => None,
        }
    }

    fn store(&mut self, entry: TtEntry<M>) {
        let index = (entry.key as usize) & self.mask;
        self.entries[index] = Some(entry);
    }
}

/// Convert a search-perspective score to its transposition-table form
/// (terminal distances stored relative to the node, not the root).
fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score > SCORE_TERMINAL_BOUND {
        score + ply as i32
    } else if score < -SCORE_TERMINAL_BOUND {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score > SCORE_TERMINAL_BOUND {
        score - ply as i32
    } else if score < -SCORE_TERMINAL_BOUND {
        score + ply as i32
    } else {
        score
    }
}

/// Result of one iterative-deepening search.
#[derive(Clone, Copy, Debug)]
pub struct SearchResult<M> {
    pub best_move: Option<M>,
    pub value: i32,
    pub completed_depth: u32,
    /// Total nodes including work on the final, possibly aborted depth.
    pub nodes: u64,
    /// Nodes spent up to and including the last fully completed depth.
    pub nodes_at_completed_depth: u64,
}

/// The production search: negamax with alpha–beta pruning, iterative
/// deepening, an optional transposition table, and a deterministic node
/// budget. Search is single-threaded; parallelism comes from independent
/// games.
pub struct Searcher<G: Game> {
    tt: Option<TranspositionTable<G::Move>>,
    ordering: MoveOrdering,
    nodes: u64,
    node_limit: u64,
    aborted: bool,
}

impl<G: Game> Searcher<G> {
    /// `tt_log2_entries: None` disables the transposition table (used only
    /// by controlled experiments; production keeps it on).
    pub fn new(tt_log2_entries: Option<u32>, ordering: MoveOrdering) -> Searcher<G> {
        Searcher {
            tt: tt_log2_entries.map(TranspositionTable::new),
            ordering,
            nodes: 0,
            node_limit: u64::MAX,
            aborted: false,
        }
    }

    pub fn tt_stats(&self) -> (u64, u64) {
        self.tt
            .as_ref()
            .map(|t| (t.probes, t.hits))
            .unwrap_or((0, 0))
    }

    /// Iterative-deepening search. Returns the result of the last fully
    /// completed depth; work on an aborted deeper iteration is counted in
    /// `nodes` but not in `nodes_at_completed_depth`.
    ///
    /// `eval` scores non-terminal depth-0 leaves from the side to move's
    /// perspective and must stay within `[-SCORE_EVAL_MAX, SCORE_EVAL_MAX]`.
    pub fn search(
        &mut self,
        game: &G,
        state: &mut G::State,
        max_depth: u32,
        max_nodes: u64,
        eval: &impl Fn(&G, &G::State) -> i32,
    ) -> SearchResult<G::Move> {
        self.nodes = 0;
        self.node_limit = max_nodes;
        self.aborted = false;

        let mut result = SearchResult {
            best_move: None,
            value: 0,
            completed_depth: 0,
            nodes: 0,
            nodes_at_completed_depth: 0,
        };
        if game.outcome(state).is_some() {
            return result;
        }

        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        self.order_moves(game.position_key(state), &mut moves);

        for depth in 1..=max_depth {
            let mut best_move = moves[0];
            let mut best_score = -SCORE_INF;
            let mut alpha = -SCORE_INF;
            for &mv in &moves {
                let undo = game.make_move(state, mv);
                let score = -self.negamax(game, state, depth - 1, 1, -SCORE_INF, -alpha, eval);
                game.unmake_move(state, mv, undo);
                if self.aborted {
                    break;
                }
                if score > best_score {
                    best_score = score;
                    best_move = mv;
                    alpha = alpha.max(score);
                }
            }
            if self.aborted {
                break;
            }
            result.best_move = Some(best_move);
            result.value = best_score;
            result.completed_depth = depth;
            result.nodes_at_completed_depth = self.nodes;
            // A proven win/loss with distance within the searched horizon
            // cannot change at deeper depths; stop deterministically.
            if best_score.abs() > SCORE_TERMINAL_BOUND
                && (SCORE_WIN - best_score.abs()) as u32 <= depth
            {
                break;
            }
            // Put the current best move first for the next iteration.
            if let Some(pos) = moves.iter().position(|&m| m == best_move) {
                moves[..=pos].rotate_right(1);
            }
        }
        result.nodes = self.nodes;
        result
    }

    fn order_moves(&mut self, key: u64, moves: &mut [G::Move]) {
        if self.ordering == MoveOrdering::Reversed {
            moves.reverse();
        }
        if let Some(tt) = &mut self.tt {
            if let Some(entry) = tt.probe(key) {
                if let Some(best) = entry.best {
                    if let Some(pos) = moves.iter().position(|&m| m == best) {
                        moves[..=pos].rotate_right(1);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // search kernel: (depth, ply, alpha, beta, eval) are irreducible
    fn negamax(
        &mut self,
        game: &G,
        state: &mut G::State,
        depth: u32,
        ply: u32,
        mut alpha: i32,
        mut beta: i32,
        eval: &impl Fn(&G, &G::State) -> i32,
    ) -> i32 {
        self.nodes += 1;
        if self.nodes >= self.node_limit {
            self.aborted = true;
            return 0;
        }
        if let Some(outcome) = game.outcome(state) {
            return terminal_score(outcome, game.side_to_move(state), ply);
        }
        if depth == 0 {
            let value = eval(game, state);
            debug_assert!(value.abs() <= SCORE_EVAL_MAX, "leaf eval out of range");
            return value;
        }

        let key = game.position_key(state);
        let alpha_original = alpha;
        let mut tt_move: Option<G::Move> = None;
        if let Some(tt) = &mut self.tt {
            if let Some(entry) = tt.probe(key) {
                tt_move = entry.best;
                if entry.depth >= depth {
                    let score = score_from_tt(entry.score, ply);
                    match entry.bound {
                        Bound::Exact => return score,
                        Bound::Lower => alpha = alpha.max(score),
                        Bound::Upper => beta = beta.min(score),
                    }
                    if alpha >= beta {
                        return score;
                    }
                }
            }
        }

        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        debug_assert!(!moves.is_empty(), "non-terminal state without moves");
        if self.ordering == MoveOrdering::Reversed {
            moves.reverse();
        }
        if let Some(best) = tt_move {
            if let Some(pos) = moves.iter().position(|&m| m == best) {
                moves[..=pos].rotate_right(1);
            }
        }

        let mut best_score = -SCORE_INF;
        let mut best_move = None;
        for &mv in &moves {
            let undo = game.make_move(state, mv);
            let score = -self.negamax(game, state, depth - 1, ply + 1, -beta, -alpha, eval);
            game.unmake_move(state, mv, undo);
            if self.aborted {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        if let Some(tt) = &mut self.tt {
            let bound = if best_score <= alpha_original {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            tt.store(TtEntry {
                key,
                score: score_to_tt(best_score, ply),
                depth,
                bound,
                best: best_move,
            });
        }
        best_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::connect_k::{ConnectK, ConnectKMove};

    fn zero_eval(_: &ConnectK, _: &<ConnectK as Game>::State) -> i32 {
        0
    }

    fn full_depth(game: &ConnectK) -> u32 {
        game.cell_count()
    }

    /// Deterministic sample of reachable states: LCG playouts truncated at
    /// random plies.
    fn sample_states(
        game: &ConnectK,
        count: usize,
        mut lcg: u64,
    ) -> Vec<<ConnectK as Game>::State> {
        let mut states = Vec::new();
        let mut moves = Vec::new();
        while states.len() < count {
            let mut state = game.initial_state();
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let stop_ply = (lcg >> 33) as u32 % game.cell_count();
            for _ in 0..stop_ply {
                if game.outcome(&state).is_some() {
                    break;
                }
                game.legal_moves(&state, &mut moves);
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mv = moves[(lcg >> 33) as usize % moves.len()];
                game.make_move(&mut state, mv);
            }
            if game.outcome(&state).is_none() {
                states.push(state);
            }
        }
        states
    }

    fn small_games() -> Vec<ConnectK> {
        vec![
            ConnectK::new(3, 3, 3, true).unwrap(),
            ConnectK::new(3, 3, 3, false).unwrap(),
            ConnectK::new(4, 3, 3, true).unwrap(),
        ]
    }

    #[test]
    fn tic_tac_toe_is_a_draw_for_every_method() {
        let game = ConnectK::new(3, 3, 3, false).unwrap();
        let mut state = game.initial_state();

        let mut solver = ExactSolver::new();
        assert_eq!(solver.solve(&game, &mut state), Wdl::Draw);

        let mut nodes = 0u64;
        let value = exhaustive_negamax(&game, &mut state, 0, &mut nodes);
        assert_eq!(Wdl::from_solved_score(value), Wdl::Draw);

        for tt in [None, Some(16)] {
            let mut searcher = Searcher::new(tt, MoveOrdering::Natural);
            let result =
                searcher.search(&game, &mut state, full_depth(&game), u64::MAX, &zero_eval);
            assert_eq!(result.completed_depth, full_depth(&game));
            assert_eq!(Wdl::from_solved_score(result.value), Wdl::Draw);
        }
    }

    #[test]
    fn exhaustive_and_alpha_beta_scores_agree_on_sampled_states() {
        for game in small_games() {
            for mut state in sample_states(&game, 30, 0xabcdef) {
                let mut nodes = 0u64;
                let reference = exhaustive_negamax(&game, &mut state, 0, &mut nodes);
                for (tt, ordering) in [
                    (None, MoveOrdering::Natural),
                    (None, MoveOrdering::Reversed),
                    (Some(14), MoveOrdering::Natural),
                    (Some(14), MoveOrdering::Reversed),
                ] {
                    let mut searcher = Searcher::new(tt, ordering);
                    let result =
                        searcher.search(&game, &mut state, full_depth(&game), u64::MAX, &zero_eval);
                    assert_eq!(
                        result.value, reference,
                        "tt={tt:?} ordering={ordering:?} disagrees with exhaustive negamax"
                    );
                }
            }
        }
    }

    #[test]
    fn search_best_moves_are_exactly_optimal() {
        for game in small_games() {
            let mut solver = ExactSolver::new();
            for mut state in sample_states(&game, 30, 0x1357) {
                let mut optimal = Vec::new();
                let wdl = solver.optimal_moves(&game, &mut state, &mut optimal);
                for ordering in [MoveOrdering::Natural, MoveOrdering::Reversed] {
                    let mut searcher: Searcher<ConnectK> = Searcher::new(Some(14), ordering);
                    let result =
                        searcher.search(&game, &mut state, full_depth(&game), u64::MAX, &zero_eval);
                    assert_eq!(Wdl::from_solved_score(result.value), wdl);
                    let best = result
                        .best_move
                        .expect("non-terminal search returns a move");
                    assert!(
                        optimal.contains(&best),
                        "search chose a suboptimal move under {ordering:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn oracle_leaves_produce_optimal_root_actions_at_depth_one() {
        let game = ConnectK::new(4, 3, 3, true).unwrap();
        let solver = std::cell::RefCell::new(ExactSolver::new());
        let oracle_eval = |g: &ConnectK, s: &<ConnectK as Game>::State| -> i32 {
            let mut s = s.clone();
            match solver.borrow_mut().solve(g, &mut s) {
                Wdl::Win => 500,
                Wdl::Draw => 0,
                Wdl::Loss => -500,
            }
        };
        let mut check_solver = ExactSolver::new();
        for mut state in sample_states(&game, 40, 0x2468) {
            let mut optimal = Vec::new();
            check_solver.optimal_moves(&game, &mut state, &mut optimal);
            let mut searcher: Searcher<ConnectK> = Searcher::new(None, MoveOrdering::Natural);
            let result = searcher.search(&game, &mut state, 1, u64::MAX, &oracle_eval);
            let best = result.best_move.unwrap();
            assert!(
                optimal.contains(&best),
                "depth-1 search with oracle leaves chose a suboptimal move"
            );
        }
    }

    #[test]
    fn node_budget_returns_last_completed_iteration() {
        let game = ConnectK::new(4, 4, 3, true).unwrap();
        let mut state = game.initial_state();
        let mut full: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
        let unlimited = full.search(&game, &mut state, 10, u64::MAX, &zero_eval);
        assert!(unlimited.nodes > 2_000);

        // A budget that lands mid-iteration must reproduce the result of an
        // unlimited search truncated at the completed depth.
        for budget in [500u64, 2_000, 10_000] {
            let mut limited: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
            let capped = limited.search(&game, &mut state, 10, budget, &zero_eval);
            assert!(capped.nodes <= budget);
            let mut reference: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
            let truncated = reference.search(
                &game,
                &mut state,
                capped.completed_depth,
                u64::MAX,
                &zero_eval,
            );
            assert_eq!(capped.value, truncated.value);
            assert_eq!(capped.best_move, truncated.best_move);
            assert_eq!(
                capped.nodes_at_completed_depth,
                truncated.nodes_at_completed_depth
            );
        }
    }

    #[test]
    fn search_is_deterministic() {
        let game = ConnectK::new(5, 4, 4, true).unwrap();
        let mut state = game.initial_state();
        let run = || {
            let mut searcher: Searcher<ConnectK> = Searcher::new(Some(16), MoveOrdering::Natural);
            let mut state2 = state.clone();
            let r = searcher.search(&game, &mut state2, 12, 200_000, &zero_eval);
            (
                r.value,
                r.best_move,
                r.completed_depth,
                r.nodes,
                r.nodes_at_completed_depth,
            )
        };
        assert_eq!(run(), run());
        let _ = &mut state;
    }

    #[test]
    fn immediate_win_is_found_with_mate_distance_score() {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        let mut state = game.initial_state();
        // P1 threatens on cells 0..=2 bottom row; P2 stacks column 6.
        for cell in [0u16, 6, 1, 13, 2, 20] {
            game.make_move(&mut state, ConnectKMove(cell));
        }
        let mut searcher: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
        let result = searcher.search(&game, &mut state, 6, u64::MAX, &zero_eval);
        assert_eq!(result.value, SCORE_WIN - 1, "win in one ply");
        assert_eq!(result.best_move, Some(ConnectKMove(3)));
    }

    #[test]
    fn ordering_changes_node_counts_but_never_values_on_asymmetric_states() {
        // From the (symmetric) initial position, natural and reversed
        // orderings explore mirror-isomorphic trees and cost identical
        // nodes; asymmetric sampled positions expose real ordering effects.
        let game = ConnectK::new(5, 4, 4, true).unwrap();
        let (mut nodes_natural, mut nodes_reversed) = (0u64, 0u64);
        let mut some_position_differs = false;
        for mut state in sample_states(&game, 8, 0x9d2c) {
            let mut results = Vec::new();
            for ordering in [MoveOrdering::Natural, MoveOrdering::Reversed] {
                let mut searcher: Searcher<ConnectK> = Searcher::new(None, ordering);
                let r = searcher.search(&game, &mut state, full_depth(&game), u64::MAX, &zero_eval);
                results.push(r);
            }
            assert_eq!(results[0].value, results[1].value);
            nodes_natural += results[0].nodes;
            nodes_reversed += results[1].nodes;
            if results[0].nodes != results[1].nodes {
                some_position_differs = true;
            }
        }
        assert!(
            some_position_differs,
            "orderings should differ in node cost on some asymmetric position"
        );
        println!(
            "ordering node totals over 8 sampled 5x4k4g positions: \
             natural={nodes_natural} reversed={nodes_reversed}"
        );
    }

    #[test]
    fn enumeration_matches_state_based_reference_on_tic_tac_toe() {
        let game = ConnectK::new(3, 3, 3, false).unwrap();

        // Reference: BFS with full-state dedup (no hashing involved).
        fn reference_count(
            game: &ConnectK,
            state: &mut <ConnectK as Game>::State,
            seen: &mut std::collections::HashSet<(Vec<u32>, Player)>,
        ) -> u64 {
            if game.outcome(state).is_some() {
                return 0;
            }
            let mut features = Vec::new();
            game.encode_features(state, &mut features);
            // features are side-relative; pair them with the absolute side
            // to move to identify the absolute position.
            if !seen.insert((features, game.side_to_move(state))) {
                return 0;
            }
            let mut count = 1;
            let mut moves = Vec::new();
            game.legal_moves(state, &mut moves);
            for &mv in &moves {
                let undo = game.make_move(state, mv);
                count += reference_count(game, state, seen);
                game.unmake_move(state, mv, undo);
            }
            count
        }
        let mut state = game.initial_state();
        let mut seen = std::collections::HashSet::new();
        let reference = reference_count(&game, &mut state, &mut seen);

        let mut solver = ExactSolver::new();
        let mut visited = 0u64;
        let mut initial_value = None;
        enumerate_solved(&game, &mut solver, |position| {
            visited += 1;
            if position.ply == 0 {
                initial_value = Some(position.value);
            }
            assert!(
                !position.optimal.is_empty(),
                "solved state must have an optimal move"
            );
            assert!(position.optimal.len() <= position.legal.len());
        });
        assert_eq!(
            visited, reference,
            "Zobrist-keyed enumeration disagrees with state dedup"
        );
        // External anchor: tic-tac-toe has 4,520 reachable non-terminal
        // positions (5,478 total minus 958 terminal).
        assert_eq!(visited, 4_520);
        assert_eq!(initial_value, Some(Wdl::Draw));
    }
}

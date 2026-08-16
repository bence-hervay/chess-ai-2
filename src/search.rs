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

/// Safety bound on quiescence recursion (engineering; captures and
/// promotions strictly consume material so real chains are far
/// shorter — see parameter_ledger.json).
const QS_MAX_DEPTH: u32 = 32;

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
///
/// `child_values[i]` is the exact value of playing `legal[i]`, from the
/// current mover's perspective; a move is optimal iff its child value
/// equals `value`.
pub struct SolvedPosition<'a, G: Game> {
    pub state: &'a G::State,
    pub ply: u32,
    pub value: Wdl,
    pub legal: &'a [G::Move],
    pub child_values: &'a [Wdl],
}

impl<G: Game> SolvedPosition<'_, G> {
    pub fn optimal_moves(&self) -> impl Iterator<Item = G::Move> + '_ {
        self.legal
            .iter()
            .zip(self.child_values)
            .filter(|(_, &v)| v == self.value)
            .map(|(&m, _)| m)
    }
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
        let mut legal = Vec::new();
        game.legal_moves(state, &mut legal);
        let mut child_values = Vec::with_capacity(legal.len());
        for &mv in &legal {
            let undo = game.make_move(state, mv);
            child_values.push(solver.solve(game, state).flip());
            game.unmake_move(state, mv, undo);
        }
        let value = child_values
            .iter()
            .copied()
            .max()
            .expect("non-terminal has moves");
        visit(SolvedPosition {
            state,
            ply,
            value,
            legal: &legal,
            child_values: &child_values,
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

/// Solution of a repetition-capable ("loopy") game by reachable-graph
/// retrograde analysis. Positions are states deduplicated by
/// `position_key`; values are game-theoretic WDL from each position's
/// mover under the convention that unresolvable cycles are draws
/// (repetition-as-draw) and the fifty-move rule is ignored — the
/// standard tablebase caveat, documented in DECISIONS.md.
pub struct RetrogradeSolution<G: Game> {
    /// Reachable positions in discovery (BFS) order; index 0 is the
    /// initial position.
    pub states: Vec<G::State>,
    /// Value of each position for its side to move.
    pub values: Vec<Wdl>,
    /// `position_key` → index.
    pub index_of: std::collections::HashMap<u64, u32>,
    /// Successor indices, aligned with `legal_moves` order; empty for
    /// terminal positions.
    pub edges: Vec<Vec<u32>>,
}

impl<G: Game> RetrogradeSolution<G> {
    /// Exact value of each legal move of position `index` (mover's
    /// perspective), aligned with `legal_moves` order.
    pub fn child_values(&self, index: usize) -> Vec<Wdl> {
        self.edges[index]
            .iter()
            .map(|&child| self.values[child as usize].flip())
            .collect()
    }
}

/// Solve a game whose positions may repeat, by forward reachability plus
/// backward induction. Fails if more than `max_positions` positions are
/// reachable.
pub fn solve_retrograde<G: Game>(
    game: &G,
    max_positions: usize,
) -> Result<RetrogradeSolution<G>, String> {
    use std::collections::HashMap;
    let mut index_of: HashMap<u64, u32> = HashMap::new();
    let mut states: Vec<G::State> = Vec::new();
    let mut edges: Vec<Vec<u32>> = Vec::new();

    let initial = game.initial_state();
    index_of.insert(game.position_key(&initial), 0);
    states.push(initial);
    edges.push(Vec::new());

    // Forward BFS. Expanding always starts from a stored state (whose
    // repetition history is at most one entry), so spurious threefold
    // draws cannot fire during enumeration.
    let mut moves = Vec::new();
    let mut cursor = 0usize;
    while cursor < states.len() {
        if game.outcome(&states[cursor]).is_some() {
            cursor += 1;
            continue;
        }
        let mut state = states[cursor].clone();
        game.legal_moves(&state, &mut moves);
        let move_list = moves.clone();
        let mut successors = Vec::with_capacity(move_list.len());
        for &mv in &move_list {
            let undo = game.make_move(&mut state, mv);
            let key = game.position_key(&state);
            let index = match index_of.get(&key) {
                Some(&index) => index,
                None => {
                    let index = states.len() as u32;
                    if states.len() >= max_positions {
                        return Err(format!("more than {max_positions} reachable positions"));
                    }
                    index_of.insert(key, index);
                    states.push(state.clone());
                    edges.push(Vec::new());
                    index
                }
            };
            successors.push(index);
            game.unmake_move(&mut state, mv, undo);
        }
        edges[cursor] = successors;
        cursor += 1;
    }

    // Backward induction. `Win` propagates immediately; `Loss` needs all
    // successors resolved; leftovers are draw cycles.
    let n = states.len();
    let mut reverse: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut pending: Vec<u32> = vec![0; n];
    for (parent, successors) in edges.iter().enumerate() {
        pending[parent] = successors.len() as u32;
        for &child in successors {
            reverse[child as usize].push(parent as u32);
        }
    }
    let mut values: Vec<Option<Wdl>> = vec![None; n];
    let mut queue = std::collections::VecDeque::new();
    for (index, state) in states.iter().enumerate() {
        if let Some(outcome) = game.outcome(state) {
            values[index] = Some(Wdl::from_outcome(outcome, game.side_to_move(state)));
            queue.push_back(index as u32);
        }
    }
    let mut best: Vec<Wdl> = vec![Wdl::Loss; n];
    while let Some(child) = queue.pop_front() {
        let child_value = values[child as usize].expect("resolved");
        let gain = child_value.flip();
        for &parent in &reverse[child as usize] {
            let p = parent as usize;
            if values[p].is_some() {
                continue;
            }
            if gain > best[p] {
                best[p] = gain;
            }
            pending[p] -= 1;
            if gain == Wdl::Win {
                values[p] = Some(Wdl::Win);
                queue.push_back(parent);
            } else if pending[p] == 0 {
                values[p] = Some(best[p]);
                queue.push_back(parent);
            }
        }
    }
    let values: Vec<Wdl> = values
        .into_iter()
        .map(|value| value.unwrap_or(Wdl::Draw))
        .collect();

    Ok(RetrogradeSolution {
        states,
        values,
        index_of,
        edges,
    })
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

/// Leaf evaluation and policy move ordering supplied to the search
/// (plan §11.2).
pub trait Evaluator<G: Game> {
    /// Score a non-terminal depth-0 leaf from the side to move's
    /// perspective, within `[-SCORE_EVAL_MAX, SCORE_EVAL_MAX]`.
    fn leaf_value(&mut self, game: &G, state: &G::State) -> i32;

    /// Policy scores for `moves` (higher = search first), aligned by
    /// index. Return `false` to keep pure stable action-ID order.
    fn policy_scores(
        &mut self,
        game: &G,
        state: &G::State,
        moves: &[G::Move],
        out: &mut Vec<f32>,
    ) -> bool {
        let _ = (game, state, moves, out);
        false
    }
}

/// Zero leaf value, no policy ordering: the model-free baseline.
pub struct ZeroEvaluator;

impl<G: Game> Evaluator<G> for ZeroEvaluator {
    fn leaf_value(&mut self, _game: &G, _state: &G::State) -> i32 {
        0
    }
}

/// Move iteration order. `Natural` follows the game's generation order
/// (stable action-ID order); `Reversed` deliberately degrades ordering for
/// controlled experiments. Policy scores (when the evaluator provides
/// them) are applied before this flag, and the transposition-table move
/// always comes first.
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
#[derive(Clone, Debug)]
pub struct SearchResult<M> {
    pub best_move: Option<M>,
    pub value: i32,
    pub completed_depth: u32,
    /// Total nodes including work on the final, possibly aborted depth.
    pub nodes: u64,
    /// Nodes spent up to and including the last fully completed depth.
    pub nodes_at_completed_depth: u64,
    /// `(best move, value)` after each completed iteration, indexed by
    /// `depth - 1`. Best-move stability across depths is teacher-record
    /// evidence (SHSD program §18.2).
    pub per_depth: Vec<(M, i32)>,
    /// Nodes spent inside quiescence (subset of `nodes`; §68.1).
    pub quiescence_nodes: u64,
}

/// The production search: negamax with alpha–beta pruning, iterative
/// deepening, an optional transposition table, and a deterministic node
/// budget. Search is single-threaded; parallelism comes from independent
/// games.
pub struct Searcher<G: Game> {
    tt: Option<TranspositionTable<G::Move>>,
    ordering: MoveOrdering,
    /// Resolve captures/promotions at the horizon (SHSD §37.3 rung 2).
    /// Off by default; off is the correctness-preserving reference
    /// mode (§14.3) and is bit-identical to the pre-quiescence search.
    quiescence: bool,
    nodes: u64,
    qnodes: u64,
    node_limit: u64,
    /// Optional wall-clock deadline (§11.4), checked every 128 nodes
    /// (fine-grained because model evaluators cost ~0.2 ms per node).
    /// Node-budget runs leave this `None` and stay fully deterministic.
    deadline: Option<std::time::Instant>,
    aborted: bool,
    scores: Vec<f32>,
}

impl<G: Game> Searcher<G> {
    /// `tt_log2_entries: None` disables the transposition table (used only
    /// by controlled experiments; production keeps it on).
    pub fn new(tt_log2_entries: Option<u32>, ordering: MoveOrdering) -> Searcher<G> {
        Searcher {
            tt: tt_log2_entries.map(TranspositionTable::new),
            ordering,
            quiescence: false,
            nodes: 0,
            qnodes: 0,
            node_limit: u64::MAX,
            deadline: None,
            aborted: false,
            scores: Vec::new(),
        }
    }

    /// Set or clear the wall-clock deadline for subsequent searches.
    pub fn set_deadline(&mut self, deadline: Option<std::time::Instant>) {
        self.deadline = deadline;
    }

    /// Enable or disable horizon quiescence (captures and promotions
    /// only; games without a tactical classification are unaffected).
    pub fn set_quiescence(&mut self, on: bool) {
        self.quiescence = on;
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
        eval: &mut impl Evaluator<G>,
    ) -> SearchResult<G::Move> {
        self.nodes = 0;
        self.qnodes = 0;
        self.node_limit = max_nodes;
        self.aborted = false;

        let mut result = SearchResult {
            best_move: None,
            value: 0,
            completed_depth: 0,
            nodes: 0,
            nodes_at_completed_depth: 0,
            per_depth: Vec::new(),
            quiescence_nodes: 0,
        };
        if game.outcome(state).is_some() {
            return result;
        }

        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        self.order_moves(game, state, game.position_key(state), &mut moves, eval);
        // If the budget is too small for even one completed iteration,
        // fall back to the first ordered move (the policy argmax when the
        // evaluator provides scores).
        result.best_move = Some(moves[0]);

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
            result.per_depth.push((best_move, best_score));
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
        result.quiescence_nodes = self.qnodes;
        result
    }

    /// Order: transposition-table move, then descending policy score,
    /// then stable action-ID order (§11.2). `Reversed` degrades the
    /// non-TT portion for controlled experiments.
    fn order_moves(
        &mut self,
        game: &G,
        state: &G::State,
        key: u64,
        moves: &mut [G::Move],
        eval: &mut impl Evaluator<G>,
    ) {
        let mut scores = std::mem::take(&mut self.scores);
        if eval.policy_scores(game, state, moves, &mut scores) {
            debug_assert_eq!(scores.len(), moves.len());
            // Stable sort keeps action-ID order among equal scores.
            let mut indexed: Vec<(usize, G::Move)> = moves.iter().copied().enumerate().collect();
            indexed.sort_by(|(i, _), (j, _)| scores[*j].total_cmp(&scores[*i]));
            for (slot, (_, mv)) in indexed.into_iter().enumerate() {
                moves[slot] = mv;
            }
        }
        self.scores = scores;
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
        eval: &mut impl Evaluator<G>,
    ) -> i32 {
        self.nodes += 1;
        if self.nodes >= self.node_limit {
            self.aborted = true;
            return 0;
        }
        if self.nodes.is_multiple_of(128) {
            if let Some(deadline) = self.deadline {
                if std::time::Instant::now() >= deadline {
                    self.aborted = true;
                    return 0;
                }
            }
        }
        if let Some(outcome) = game.outcome(state) {
            return terminal_score(outcome, game.side_to_move(state), ply);
        }
        if depth == 0 {
            if self.quiescence {
                // The horizon node was already counted at this
                // function's entry; qsearch counts it again, so undo
                // one increment to keep node accounting exact.
                self.nodes -= 1;
                return self.qsearch(game, state, ply, alpha, beta, eval, 0);
            }
            let value = eval.leaf_value(game, state);
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
        let mut scores = std::mem::take(&mut self.scores);
        if eval.policy_scores(game, state, &moves, &mut scores) {
            debug_assert_eq!(scores.len(), moves.len());
            let mut indexed: Vec<(usize, G::Move)> = moves.iter().copied().enumerate().collect();
            indexed.sort_by(|(i, _), (j, _)| scores[*j].total_cmp(&scores[*i]));
            for (slot, (_, mv)) in indexed.into_iter().enumerate() {
                moves[slot] = mv;
            }
        }
        self.scores = scores;
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

    /// Horizon quiescence (SHSD §37.3 rung 2): stand pat on the static
    /// evaluation, then resolve captures and promotions only. No
    /// transposition-table interaction; nodes count against the budget
    /// and are reported separately. Checks receive no special
    /// treatment yet (identical to the pre-quiescence leaf behavior in
    /// check); evasions are rung 3.
    #[allow(clippy::too_many_arguments)] // same kernel shape as negamax
    fn qsearch(
        &mut self,
        game: &G,
        state: &mut G::State,
        ply: u32,
        mut alpha: i32,
        beta: i32,
        eval: &mut impl Evaluator<G>,
        qdepth: u32,
    ) -> i32 {
        self.nodes += 1;
        self.qnodes += 1;
        if self.nodes >= self.node_limit {
            self.aborted = true;
            return 0;
        }
        if self.nodes.is_multiple_of(128) {
            if let Some(deadline) = self.deadline {
                if std::time::Instant::now() >= deadline {
                    self.aborted = true;
                    return 0;
                }
            }
        }
        if let Some(outcome) = game.outcome(state) {
            return terminal_score(outcome, game.side_to_move(state), ply);
        }
        let stand = eval.leaf_value(game, state);
        debug_assert!(stand.abs() <= SCORE_EVAL_MAX, "leaf eval out of range");
        if qdepth >= QS_MAX_DEPTH || stand >= beta {
            return stand;
        }
        alpha = alpha.max(stand);

        let mut moves = Vec::new();
        game.legal_moves(state, &mut moves);
        moves.retain(|&mv| game.is_tactical(state, mv));
        if moves.is_empty() {
            return stand;
        }
        let mut scores = std::mem::take(&mut self.scores);
        if eval.policy_scores(game, state, &moves, &mut scores) {
            debug_assert_eq!(scores.len(), moves.len());
            let mut indexed: Vec<(usize, G::Move)> = moves.iter().copied().enumerate().collect();
            indexed.sort_by(|(i, _), (j, _)| scores[*j].total_cmp(&scores[*i]));
            for (slot, (_, mv)) in indexed.into_iter().enumerate() {
                moves[slot] = mv;
            }
        }
        self.scores = scores;

        let mut best = stand;
        for &mv in &moves {
            let undo = game.make_move(state, mv);
            let score = -self.qsearch(game, state, ply + 1, -beta, -alpha, eval, qdepth + 1);
            game.unmake_move(state, mv, undo);
            if self.aborted {
                return 0;
            }
            best = best.max(score);
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::connect_k::{ConnectK, ConnectKMove};

    /// Exact-oracle leaf evaluator: Win=+1000, Draw=0, Loss=-1000.
    struct OracleLeafEval(ExactSolver);

    impl Evaluator<ConnectK> for OracleLeafEval {
        fn leaf_value(&mut self, game: &ConnectK, state: &<ConnectK as Game>::State) -> i32 {
            let mut s = state.clone();
            match self.0.solve(game, &mut s) {
                Wdl::Win => SCORE_EVAL_MAX,
                Wdl::Draw => 0,
                Wdl::Loss => -SCORE_EVAL_MAX,
            }
        }
    }

    /// Deterministic non-trivial policy: prefers high action IDs, so
    /// ordering differs from both natural and reversed baselines.
    struct HighIdPolicy;

    impl Evaluator<ConnectK> for HighIdPolicy {
        fn leaf_value(&mut self, _: &ConnectK, _: &<ConnectK as Game>::State) -> i32 {
            0
        }
        fn policy_scores(
            &mut self,
            game: &ConnectK,
            state: &<ConnectK as Game>::State,
            moves: &[ConnectKMove],
            out: &mut Vec<f32>,
        ) -> bool {
            out.clear();
            out.extend(moves.iter().map(|&m| game.action_id(state, m) as f32));
            true
        }
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
            let result = searcher.search(
                &game,
                &mut state,
                full_depth(&game),
                u64::MAX,
                &mut ZeroEvaluator,
            );
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
                    let result = searcher.search(
                        &game,
                        &mut state,
                        full_depth(&game),
                        u64::MAX,
                        &mut ZeroEvaluator,
                    );
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
                    let result = searcher.search(
                        &game,
                        &mut state,
                        full_depth(&game),
                        u64::MAX,
                        &mut ZeroEvaluator,
                    );
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
        let mut oracle_eval = OracleLeafEval(ExactSolver::new());
        let mut check_solver = ExactSolver::new();
        for mut state in sample_states(&game, 40, 0x2468) {
            let mut optimal = Vec::new();
            check_solver.optimal_moves(&game, &mut state, &mut optimal);
            let mut searcher: Searcher<ConnectK> = Searcher::new(None, MoveOrdering::Natural);
            let result = searcher.search(&game, &mut state, 1, u64::MAX, &mut oracle_eval);
            let best = result.best_move.unwrap();
            assert!(
                optimal.contains(&best),
                "depth-1 search with oracle leaves chose a suboptimal move"
            );
        }
    }

    #[test]
    fn policy_ordering_changes_nodes_but_never_values() {
        let game = ConnectK::new(4, 4, 4, true).unwrap();
        let (mut nodes_plain, mut nodes_policy) = (0u64, 0u64);
        for mut state in sample_states(&game, 15, 0x77aa) {
            let mut plain: Searcher<ConnectK> = Searcher::new(Some(16), MoveOrdering::Natural);
            let a = plain.search(
                &game,
                &mut state,
                full_depth(&game),
                u64::MAX,
                &mut ZeroEvaluator,
            );
            let mut ordered: Searcher<ConnectK> = Searcher::new(Some(16), MoveOrdering::Natural);
            let b = ordered.search(
                &game,
                &mut state,
                full_depth(&game),
                u64::MAX,
                &mut HighIdPolicy,
            );
            assert_eq!(a.value, b.value, "policy ordering changed the search value");
            nodes_plain += a.nodes;
            nodes_policy += b.nodes;
        }
        assert_ne!(
            nodes_plain, nodes_policy,
            "high-id policy should change node counts on some position"
        );
    }

    #[test]
    fn node_budget_returns_last_completed_iteration() {
        let game = ConnectK::new(4, 4, 3, true).unwrap();
        let mut state = game.initial_state();
        let mut full: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
        let unlimited = full.search(&game, &mut state, 10, u64::MAX, &mut ZeroEvaluator);
        assert!(unlimited.nodes > 2_000);

        // A budget that lands mid-iteration must reproduce the result of an
        // unlimited search truncated at the completed depth.
        for budget in [500u64, 2_000, 10_000] {
            let mut limited: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
            let capped = limited.search(&game, &mut state, 10, budget, &mut ZeroEvaluator);
            assert!(capped.nodes <= budget);
            let mut reference: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
            let truncated = reference.search(
                &game,
                &mut state,
                capped.completed_depth,
                u64::MAX,
                &mut ZeroEvaluator,
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
            let r = searcher.search(&game, &mut state2, 12, 200_000, &mut ZeroEvaluator);
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
    fn quiescence_is_neutral_without_tactical_moves() {
        // Connect-k defines no tactical moves, so quiescence ON must
        // reproduce OFF exactly — values, moves, and node counts.
        let game = ConnectK::new(5, 4, 4, true).unwrap();
        let run = |quiescence: bool| {
            let mut searcher: Searcher<ConnectK> = Searcher::new(Some(14), MoveOrdering::Natural);
            searcher.set_quiescence(quiescence);
            let mut state = game.initial_state();
            let r = searcher.search(&game, &mut state, 6, 50_000, &mut ZeroEvaluator);
            (r.value, r.best_move, r.nodes, r.quiescence_nodes)
        };
        let (v_off, m_off, n_off, q_off) = run(false);
        let (v_on, m_on, n_on, q_on) = run(true);
        assert_eq!(v_off, v_on);
        assert_eq!(m_off, m_on);
        assert_eq!(n_off, n_on, "no tactical moves: node counts identical");
        assert_eq!(q_off, 0);
        assert!(
            q_on > 0 && q_on < n_on,
            "horizon leaves pass through qsearch; interior nodes do not"
        );
    }

    #[test]
    fn quiescence_resolves_a_hanging_capture() {
        // Forward Chess small: White Kd1, Rb2; Black Kd4, Pb3, Pa4.
        // Rxb3 wins a pawn at the horizon but loses the rook to axb3.
        // A material evaluator + depth-1 search: without quiescence the
        // root value believes the pawn grab (+400); with quiescence the
        // recapture is resolved (-100).
        use crate::games::forward_chess::{ForwardChess, Piece, Ruleset};
        struct Material;
        impl Evaluator<ForwardChess> for Material {
            fn leaf_value(
                &mut self,
                game: &ForwardChess,
                state: &<ForwardChess as Game>::State,
            ) -> i32 {
                let mover = game.side_to_move(state);
                let mut score = 0;
                for &code in game.state_cells(state) {
                    if code == 0 {
                        continue;
                    }
                    let (owner, piece, _) = ForwardChess::unpack_code(code);
                    let value = match piece {
                        Piece::Pawn => 100,
                        Piece::Rook => 500,
                        Piece::King => 0,
                        _ => 300,
                    };
                    score += if owner == mover { value } else { -value };
                }
                score
            }
        }
        let game = ForwardChess::new(Ruleset::Small);
        let state = game.custom_state(
            &[
                ("d1", Player::One, Piece::King, false),
                ("b2", Player::One, Piece::Rook, false),
                ("d4", Player::Two, Piece::King, false),
                ("b3", Player::Two, Piece::Pawn, false),
                ("a4", Player::Two, Piece::Pawn, false),
            ],
            Player::One,
            [false; 4],
            None,
        );
        let run = |quiescence: bool| {
            let mut searcher: Searcher<ForwardChess> =
                Searcher::new(Some(12), MoveOrdering::Natural);
            searcher.set_quiescence(quiescence);
            let mut s = state.clone();
            searcher.search(&game, &mut s, 1, u64::MAX, &mut Material)
        };
        let off = run(false);
        let on = run(true);
        assert_eq!(off.value, 400, "horizon effect: the pawn grab looks won");
        let grab = off.best_move.expect("some move");
        assert_eq!(
            (grab.from, grab.to),
            (game.square("b2"), game.square("b3")),
            "without quiescence the search grabs the poisoned pawn"
        );
        assert_eq!(
            on.value, 300,
            "quiescence sees the recapture and keeps the rook instead"
        );
        assert_ne!(
            on.best_move.expect("some move").to,
            game.square("b3"),
            "with quiescence the poisoned pawn is declined"
        );
        assert!(on.quiescence_nodes > 0);
        // Budget abort stays clean with quiescence on.
        let mut searcher: Searcher<ForwardChess> = Searcher::new(Some(12), MoveOrdering::Natural);
        searcher.set_quiescence(true);
        let mut s = state.clone();
        let tiny = searcher.search(&game, &mut s, 8, 40, &mut Material);
        assert!(tiny.nodes <= 41);
        assert!(tiny.best_move.is_some());
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
        let result = searcher.search(&game, &mut state, 6, u64::MAX, &mut ZeroEvaluator);
        assert_eq!(result.value, SCORE_WIN - 1, "win in one ply");
        assert_eq!(result.best_move, Some(ConnectKMove(3)));
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "measurement test; run in release")]
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
                let r = searcher.search(
                    &game,
                    &mut state,
                    full_depth(&game),
                    u64::MAX,
                    &mut ZeroEvaluator,
                );
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
    fn breakthrough_small_boards_solve_consistently() {
        use crate::games::breakthrough::Breakthrough;
        for (w, h, rows) in [(2, 4, 1), (3, 4, 1), (2, 5, 1)] {
            let game = Breakthrough::new(w, h, rows).unwrap();
            let mut state = game.initial_state();
            let mut solver = ExactSolver::new();
            let exact = solver.solve(&game, &mut state);
            let mut nodes = 0u64;
            let exhaustive =
                Wdl::from_solved_score(exhaustive_negamax(&game, &mut state, 0, &mut nodes));
            assert_eq!(exact, exhaustive, "{w}x{h}r{rows} root value");
            let mut searcher: Searcher<Breakthrough> =
                Searcher::new(Some(14), MoveOrdering::Natural);
            let depth = game.cell_count() * 2; // captures shorten games; bound generously
            let result = searcher.search(&game, &mut state, depth, u64::MAX, &mut ZeroEvaluator);
            assert_eq!(
                Wdl::from_solved_score(result.value),
                exact,
                "{w}x{h}r{rows} alpha-beta agreement"
            );
            // Breakthrough has no draws: the root is decisive.
            assert_ne!(exact, Wdl::Draw, "{w}x{h}r{rows} must be decisive");
        }
    }

    #[test]
    fn othello_4x4_solves_consistently() {
        use crate::games::othello::Othello;
        let game = Othello::new(4, 4).unwrap();
        let mut state = game.initial_state();
        let mut solver = ExactSolver::new();
        let exact = solver.solve(&game, &mut state);
        let mut nodes = 0u64;
        let exhaustive =
            Wdl::from_solved_score(exhaustive_negamax(&game, &mut state, 0, &mut nodes));
        assert_eq!(exact, exhaustive, "4x4 othello root value");
        let mut searcher: Searcher<Othello> = Searcher::new(Some(14), MoveOrdering::Natural);
        let result = searcher.search(&game, &mut state, 64, u64::MAX, &mut ZeroEvaluator);
        assert_eq!(Wdl::from_solved_score(result.value), exact);
        println!(
            "4x4 othello root: {exact:?} ({} solved states)",
            solver.solved_states()
        );
    }

    #[test]
    fn chess_mates_in_n_with_deterministic_evaluator() {
        use crate::games::chess::{Chess, ChessMove};
        let game = Chess::new();
        // (FEN, mate distance in plies, forced best first move or None)
        let cases: [(&str, i32, Option<&str>); 3] = [
            // Back-rank mate in one.
            ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 1, Some("a1a8")),
            // Ladder mate in two moves (three plies): Rb7 then Ra8#.
            ("6k1/8/8/8/8/8/1R6/R5K1 w - - 0 1", 3, None),
            // Mated side: best defence still loses in one ply after any move?
            // Use a mate-in-one against the mover's opponent instead:
            // Black to move, White mates next: value is a loss at 2 plies.
            ("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", 0, None),
        ];
        for (fen, plies, best) in cases {
            let mut state = game.state_from_fen(fen).unwrap();
            for tt in [None, Some(16)] {
                let mut searcher: Searcher<Chess> = Searcher::new(tt, MoveOrdering::Natural);
                let result = searcher.search(&game, &mut state, 6, u64::MAX, &mut ZeroEvaluator);
                if plies > 0 {
                    assert_eq!(
                        result.value,
                        SCORE_WIN - plies,
                        "{fen}: expected mate in {plies} plies (tt={tt:?})"
                    );
                    if let Some(best) = best {
                        let expected = ChessMove(best.parse().unwrap());
                        assert_eq!(result.best_move, Some(expected), "{fen}");
                    }
                } else {
                    // Checkmated already: search on terminal states is not
                    // called; this row asserts the outcome directly.
                    assert!(game.outcome(&state).is_some(), "{fen}");
                }
            }
        }
    }

    #[test]
    fn chess_repetition_draw_is_seen_by_search() {
        use crate::games::chess::{Chess, ChessMove};
        let game = Chess::new();
        // KQ vs KR: White to move is winning, but if the position is one
        // repetition away from a threefold draw, the draw is available
        // and search must still prefer the win over shuffling.
        let mut state = game
            .state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
            .unwrap();
        // Manufacture near-threefold history: play the shuffle twice.
        for uci in ["a1b1", "e8d8", "b1a1", "d8e8", "a1b1", "e8d8", "b1a1"] {
            game.make_move(&mut state, ChessMove(uci.parse().unwrap()));
        }
        // Black to move; d8e8 would complete the third repetition and
        // draw. A deterministic material-counting test evaluator makes
        // every non-drawing continuation score badly for Black, so the
        // search must claim the draw.
        struct RookCounter;
        impl Evaluator<Chess> for RookCounter {
            fn leaf_value(&mut self, game: &Chess, state: &<Chess as Game>::State) -> i32 {
                // Score rook possession through the raw feature encoding
                // (piece index 3 = rook, even relative index = own).
                let mut features = Vec::new();
                game.encode_features(state, &mut features);
                let mut score = 0i32;
                for f in features {
                    let piece = (f as usize % 12) / 2;
                    let own = (f as usize).is_multiple_of(2);
                    if piece == 3 {
                        score += if own { 500 } else { -500 };
                    }
                }
                score
            }
        }
        let mut searcher: Searcher<Chess> = Searcher::new(Some(16), MoveOrdering::Natural);
        let result = searcher.search(&game, &mut state, 4, 200_000, &mut RookCounter);
        let draw_move = ChessMove("d8e8".parse().unwrap());
        assert_eq!(
            result.best_move,
            Some(draw_move),
            "the losing side must claim the repetition draw"
        );
        assert_eq!(result.value, 0, "repetition draw scores zero");
    }

    #[test]
    fn retrograde_matches_exact_solver_on_acyclic_games() {
        // Cross-validation on games the memoized negamax solver already
        // handles: values must agree position by position.
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let solution = solve_retrograde(&game, 100_000).unwrap();
        let mut solver = ExactSolver::new();
        for (index, state) in solution.states.iter().enumerate() {
            if game.outcome(state).is_none() {
                let mut s = state.clone();
                assert_eq!(
                    solver.solve(&game, &mut s),
                    solution.values[index],
                    "connect-k position {index}"
                );
            }
        }
        use crate::games::breakthrough::Breakthrough;
        let game = Breakthrough::new(3, 4, 1).unwrap();
        let solution = solve_retrograde(&game, 1_000_000).unwrap();
        let mut solver = ExactSolver::new();
        let mut root = game.initial_state();
        assert_eq!(solver.solve(&game, &mut root), solution.values[0]);
    }

    #[test]
    fn retrograde_solves_tiny_forward_chess_stably() {
        use crate::games::forward_chess::{ForwardChess, Ruleset};
        let game = ForwardChess::new(Ruleset::Tiny);
        let a = solve_retrograde(&game, 5_000_000).unwrap();
        let b = solve_retrograde(&game, 5_000_000).unwrap();
        assert_eq!(a.values, b.values, "solver must be deterministic");
        assert_eq!(a.states.len(), b.states.len());
        // Child values must be consistent: every position's value equals
        // the max over its children's flipped values (or its terminal
        // outcome), with draws allowed for cycles.
        for index in 0..a.states.len() {
            if game.outcome(&a.states[index]).is_some() {
                continue;
            }
            let child_values = a.child_values(index);
            let best = child_values.iter().copied().max().unwrap();
            assert_eq!(
                a.values[index], best,
                "position {index}: value must equal best child value"
            );
        }
        println!(
            "tiny forward chess: {} positions, root {:?}",
            a.states.len(),
            a.values[0]
        );
    }

    #[test]
    fn retrograde_optimal_play_realizes_root_value_under_real_rules() {
        use crate::games::forward_chess::{ForwardChess, Ruleset};
        let game = ForwardChess::new(Ruleset::Tiny);
        let solution = solve_retrograde(&game, 5_000_000).unwrap();
        let root_value = solution.values[0];
        // Play optimal-vs-optimal with the REAL state (history tracked,
        // threefold and fifty-move live): the realized outcome category
        // must match the solved root value; cycles terminate via the
        // actual repetition rule.
        let mut state = game.initial_state();
        let mut moves = Vec::new();
        let mut plies = 0;
        let outcome = loop {
            if let Some(outcome) = game.outcome(&state) {
                break outcome;
            }
            assert!(plies < 500, "optimal play must terminate under real rules");
            let index = solution.index_of[&game.position_key(&state)] as usize;
            let value = solution.values[index];
            game.legal_moves(&state, &mut moves);
            let child_values = solution.child_values(index);
            let choice = moves
                .iter()
                .zip(&child_values)
                .find(|(_, &v)| v == value)
                .map(|(&m, _)| m)
                .expect("an optimal move exists");
            game.make_move(&mut state, choice);
            plies += 1;
        };
        let mover_perspective_root = root_value;
        let realized = Wdl::from_outcome(outcome, crate::game::Player::One);
        assert_eq!(
            realized, mover_perspective_root,
            "optimal play realized {realized:?} but the root is {mover_perspective_root:?}"
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
            let optimal: Vec<_> = position.optimal_moves().collect();
            assert!(
                !optimal.is_empty(),
                "solved state must have an optimal move"
            );
            assert!(optimal.len() <= position.legal.len());
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

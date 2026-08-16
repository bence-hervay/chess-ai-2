//! Typed teacher records with target provenance, and the deep-search
//! relabelling instrumentation (SHSD program §18.2, §19.5, §19.9, §55).
//!
//! A position is stored as the exact action path from the game's initial
//! state. Replaying the path reconstructs the full state, including
//! repetition history and move counters that `position_key` may not
//! cover. Records are game-agnostic; the game and all search settings
//! live in the run manifest and resolved configuration.

use crate::game::{ActionId, Game, Outcome};
use crate::search::{
    Evaluator, MoveOrdering, RetrogradeSolution, SearchResult, Searcher, Wdl, SCORE_EVAL_MAX,
    SCORE_TERMINAL_BOUND, SCORE_WIN,
};
use crate::training::{splitmix64, SELFPLAY_TT_LOG2};
use rand::Rng as _;
use rand::SeedableRng as _;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Where a record's target came from (§19.9). Every stored target
/// carries exactly one provenance value; mixing sources in one dataset
/// without this tag is a defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    Exact,
    Terminal,
    InternalSearch,
    StockfishTeacher,
    HistoricalMatch,
    CounterfactualSearch,
    ShadowSearch,
}

/// One search's outcome at a fixed node budget, from the side to move's
/// perspective. `per_depth[d-1]` is `(best action, value)` after the
/// completed iteration at depth `d` (best-move stability, §18.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchLabel {
    pub node_budget: u64,
    pub nodes: u64,
    pub completed_depth: u32,
    pub value: i32,
    pub best_action: ActionId,
    pub per_depth: Vec<(ActionId, i32)>,
}

impl SearchLabel {
    fn from_result<G: Game>(
        game: &G,
        state: &G::State,
        node_budget: u64,
        result: &SearchResult<G::Move>,
    ) -> SearchLabel {
        SearchLabel {
            node_budget,
            nodes: result.nodes,
            completed_depth: result.completed_depth,
            value: result.value,
            best_action: game.action_id(
                state,
                result.best_move.expect("search on non-terminal state"),
            ),
            per_depth: result
                .per_depth
                .iter()
                .map(|&(mv, value)| (game.action_id(state, mv), value))
                .collect(),
        }
    }
}

/// One searched child of the record's position (§19.5). `value` is from
/// the *parent* mover's perspective (the child search value negated);
/// mate distances are relative to the child root, so they are one ply
/// short of the parent's convention. Terminal children get exact
/// terminal values and zero nodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildLabel {
    pub action: ActionId,
    pub node_budget: u64,
    pub nodes: u64,
    pub completed_depth: u32,
    pub value: i32,
}

/// One teacher-labelled position.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherRecord {
    /// Action path from the initial state (exact reconstruction).
    pub path: Vec<ActionId>,
    /// `position_key` at the position (dedup and oracle join).
    pub key: u64,
    pub ply: u32,
    /// Times this position was visited across sampled trajectories
    /// (§20.4: duplicate frequency is data, not noise).
    pub weight: u32,
    pub provenance: Provenance,
    /// Identifier of the evaluator that produced every label in this
    /// record ("zero" or a checkpoint path).
    pub evaluator: String,
    pub shallow: Vec<SearchLabel>,
    pub deep: SearchLabel,
    /// Rank (0 = first) of `deep.best_action` in the evaluator-induced
    /// root move ordering (policy scores if provided, else stable
    /// action-ID order). Move-ordering error evidence (§55).
    pub deep_best_order_rank: u32,
    /// Counterfactual child labels in legal-move generation order; each
    /// carries its own `action` id, so consumers should match by action
    /// rather than by index. Empty when child search is disabled.
    pub children: Vec<ChildLabel>,
    /// Exact WDL for the side to move, when an oracle was joined.
    pub exact_wdl: Option<Wdl>,
    /// Exact optimal actions, when an oracle was joined.
    pub exact_optimal: Option<Vec<ActionId>>,
}

/// Replay an action path from the initial state. Errors on illegal or
/// early-terminal paths instead of panicking: paths cross process and
/// code-revision boundaries.
pub fn replay_path<G: Game>(game: &G, path: &[ActionId]) -> Result<G::State, String> {
    let mut state = game.initial_state();
    let mut moves = Vec::new();
    for (step, &action) in path.iter().enumerate() {
        if game.outcome(&state).is_some() {
            return Err(format!("path step {step}: state is already terminal"));
        }
        game.legal_moves(&state, &mut moves);
        let mv = moves
            .iter()
            .copied()
            .find(|&m| game.action_id(&state, m) == action)
            .ok_or_else(|| format!("path step {step}: action {action} is not legal"))?;
        game.make_move(&mut state, mv);
    }
    Ok(state)
}

/// How positions are visited before sampling (§19.2/§19.7). Exploration
/// in `selfplay` trajectories samples a uniform random legal move (not
/// the apprentice policy, unlike training self-play): trajectory
/// diversity is the goal here and uniformity keeps the generator
/// evaluator-agnostic.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TrajectorySpec {
    /// The evaluator's own search plays both sides with ε-uniform
    /// exploration.
    Selfplay {
        games: u64,
        nodes: u64,
        epsilon: f64,
    },
    /// Every move uniform random (maximal breadth, no evaluator bias).
    Random { games: u64 },
}

impl TrajectorySpec {
    pub fn games(&self) -> u64 {
        match self {
            TrajectorySpec::Selfplay { games, .. } | TrajectorySpec::Random { games } => *games,
        }
    }
}

/// One sampled position awaiting labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionSample {
    pub path: Vec<ActionId>,
    pub key: u64,
    pub ply: u32,
    pub weight: u32,
}

/// Salt separating position-sampling hashes from split/thinning hashes
/// (same convention as `training::EVAL_THIN_SALT`).
const SAMPLE_SALT: u64 = 0x7eac_4e2d_a7a5_1e11;

fn trajectory_rng(seed: u64, game_index: u64) -> ChaCha12Rng {
    ChaCha12Rng::seed_from_u64(splitmix64(
        splitmix64(seed) ^ splitmix64(0x7261_6a00 | game_index),
    ))
}

/// Generate trajectories and collect deduplicated position samples.
///
/// Deterministic for a fixed `(spec, sample_one_in, max_positions,
/// seed)` regardless of `threads`: per-game RNG streams depend only on
/// `(seed, game_index)`, and the merge processes games in index order.
/// A position is sampled iff `splitmix64(key ^ SAMPLE_SALT) %
/// sample_one_in == 0`, so a given position is always in or always out
/// (§20.1); repeat visits increase `weight`. Once `max_positions`
/// distinct positions are collected, new positions are refused but
/// weights keep counting.
#[allow(clippy::too_many_arguments)] // §55 sampling axes are irreducible
pub fn collect_positions<G, E, F>(
    game: &G,
    spec: &TrajectorySpec,
    sample_one_in: u64,
    max_positions: usize,
    seed: u64,
    threads: usize,
    quiescence: bool,
    make_eval: F,
) -> Vec<PositionSample>
where
    G: Game,
    E: Evaluator<G>,
    F: Fn() -> E + Sync,
{
    assert!(sample_one_in >= 1, "sample_one_in must be at least 1");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    let per_game: Vec<Vec<(Vec<ActionId>, u64, u32)>> = pool.install(|| {
        use rayon::prelude::*;
        (0..spec.games())
            .into_par_iter()
            .map(|game_index| {
                let mut rng = trajectory_rng(seed, game_index);
                let mut evaluator = make_eval();
                let mut searcher: Searcher<G> =
                    Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
                searcher.set_quiescence(quiescence);
                let mut state = game.initial_state();
                let mut path: Vec<ActionId> = Vec::new();
                let mut moves = Vec::new();
                let mut sampled = Vec::new();
                loop {
                    if game.outcome(&state).is_some() {
                        break;
                    }
                    let key = game.position_key(&state);
                    if splitmix64(key ^ SAMPLE_SALT).is_multiple_of(sample_one_in) {
                        sampled.push((path.clone(), key, path.len() as u32));
                    }
                    game.legal_moves(&state, &mut moves);
                    let mv = match spec {
                        TrajectorySpec::Random { .. } => moves[rng.gen_range(0..moves.len())],
                        TrajectorySpec::Selfplay { nodes, epsilon, .. } => {
                            let result =
                                searcher.search(game, &mut state, 512, *nodes, &mut evaluator);
                            let expert = result
                                .best_move
                                .expect("non-terminal search returns a move");
                            if rng.gen::<f64>() < *epsilon {
                                moves[rng.gen_range(0..moves.len())]
                            } else {
                                expert
                            }
                        }
                    };
                    path.push(game.action_id(&state, mv));
                    game.make_move(&mut state, mv);
                }
                sampled
            })
            .collect()
    });

    let mut by_key: HashMap<u64, usize> = HashMap::new();
    let mut samples: Vec<PositionSample> = Vec::new();
    for game_samples in per_game {
        for (path, key, ply) in game_samples {
            match by_key.get(&key) {
                Some(&index) => samples[index].weight += 1,
                None => {
                    if samples.len() >= max_positions {
                        continue;
                    }
                    by_key.insert(key, samples.len());
                    samples.push(PositionSample {
                        path,
                        key,
                        ply,
                        weight: 1,
                    });
                }
            }
        }
    }
    samples
}

/// Rank of `action` in the evaluator-induced root ordering: descending
/// policy score with stable action-ID order on ties (mirroring the
/// search's non-TT ordering), or plain stable action-ID order when the
/// evaluator provides no policy.
fn order_rank<G: Game, E: Evaluator<G>>(
    game: &G,
    state: &G::State,
    moves: &[G::Move],
    action: ActionId,
    eval: &mut E,
) -> u32 {
    let mut scores = Vec::new();
    let mut order: Vec<usize> = (0..moves.len()).collect();
    if eval.policy_scores(game, state, moves, &mut scores) {
        order.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]));
    }
    order
        .iter()
        .position(|&i| game.action_id(state, moves[i]) == action)
        .expect("deep best action is a legal move") as u32
}

/// Exact value of a terminal child from the parent mover's perspective,
/// consistent with the search's shortest-win convention one ply down.
fn terminal_child_value(outcome: Outcome, parent_wins: impl Fn(Outcome) -> Option<bool>) -> i32 {
    match parent_wins(outcome) {
        None => 0,
        Some(true) => SCORE_WIN - 1,
        Some(false) => -(SCORE_WIN - 1),
    }
}

/// Label every sample with shallow, deep, and optional counterfactual
/// child searches. Deterministic and thread-count-independent: every
/// search is node-limited with a fresh transposition table.
#[allow(clippy::too_many_arguments)] // the §55 labelling axes are irreducible
pub fn label_positions<G, E, F>(
    game: &G,
    samples: &[PositionSample],
    evaluator_id: &str,
    shallow_nodes: &[u64],
    deep_nodes: u64,
    label_children: bool,
    child_nodes: u64,
    threads: usize,
    quiescence: bool,
    make_eval: F,
) -> Vec<TeacherRecord>
where
    G: Game,
    E: Evaluator<G>,
    F: Fn() -> E + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    pool.install(|| {
        use rayon::prelude::*;
        samples
            .par_iter()
            .map(|sample| {
                let mut state = replay_path(game, &sample.path)
                    .expect("sampled paths replay on the same code revision");
                let mut evaluator = make_eval();
                let search_at = |state: &mut G::State, budget: u64, eval: &mut E| {
                    let mut searcher: Searcher<G> =
                        Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
                    searcher.set_quiescence(quiescence);
                    searcher.search(game, state, 512, budget, eval)
                };

                let shallow: Vec<SearchLabel> = shallow_nodes
                    .iter()
                    .map(|&budget| {
                        let result = search_at(&mut state, budget, &mut evaluator);
                        SearchLabel::from_result(game, &state, budget, &result)
                    })
                    .collect();
                let deep_result = search_at(&mut state, deep_nodes, &mut evaluator);
                let deep = SearchLabel::from_result(game, &state, deep_nodes, &deep_result);

                let mut moves = Vec::new();
                game.legal_moves(&state, &mut moves);
                let deep_best_order_rank =
                    order_rank(game, &state, &moves, deep.best_action, &mut evaluator);

                let children = if label_children {
                    let side = game.side_to_move(&state);
                    moves
                        .iter()
                        .map(|&mv| {
                            let action = game.action_id(&state, mv);
                            let undo = game.make_move(&mut state, mv);
                            let label = match game.outcome(&state) {
                                Some(outcome) => ChildLabel {
                                    action,
                                    node_budget: child_nodes,
                                    nodes: 0,
                                    completed_depth: 0,
                                    value: terminal_child_value(outcome, |o| match o {
                                        Outcome::Draw => None,
                                        Outcome::Win(winner) => Some(winner == side),
                                    }),
                                },
                                None => {
                                    let result = search_at(&mut state, child_nodes, &mut evaluator);
                                    ChildLabel {
                                        action,
                                        node_budget: child_nodes,
                                        nodes: result.nodes,
                                        completed_depth: result.completed_depth,
                                        value: -result.value,
                                    }
                                }
                            };
                            game.unmake_move(&mut state, mv, undo);
                            label
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                TeacherRecord {
                    path: sample.path.clone(),
                    key: sample.key,
                    ply: sample.ply,
                    weight: sample.weight,
                    provenance: Provenance::InternalSearch,
                    evaluator: evaluator_id.to_string(),
                    shallow,
                    deep,
                    deep_best_order_rank,
                    children,
                    exact_wdl: None,
                    exact_optimal: None,
                }
            })
            .collect()
    })
}

/// Result of joining records against an exact retrograde solution.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct OracleJoinStats {
    pub joined: u64,
    pub missed: u64,
}

/// Attach exact WDL and the exact optimal-action set to every record
/// whose `key` appears in the solution. The join is by `position_key`,
/// so it inherits the retrograde caveats (repetition-as-draw, no
/// fifty-move rule; D028/D029).
pub fn join_retrograde_oracle<G: Game>(
    game: &G,
    records: &mut [TeacherRecord],
    solution: &RetrogradeSolution<G>,
) -> OracleJoinStats {
    let mut stats = OracleJoinStats::default();
    let mut moves = Vec::new();
    for record in records.iter_mut() {
        let Some(&index) = solution.index_of.get(&record.key) else {
            stats.missed += 1;
            continue;
        };
        let index = index as usize;
        let state = &solution.states[index];
        let value = solution.values[index];
        game.legal_moves(state, &mut moves);
        let child_values = solution.child_values(index);
        record.exact_wdl = Some(value);
        record.exact_optimal = Some(
            moves
                .iter()
                .zip(&child_values)
                .filter(|(_, &child)| child == value)
                .map(|(&mv, _)| game.action_id(state, mv))
                .collect(),
        );
        stats.joined += 1;
    }
    stats
}

/// Aggregate quality metrics of one relabelling run: enough to
/// attribute error to the model, the search budget, or the move
/// ordering (§55 gate), without reading individual records.
#[derive(Clone, Debug, Serialize)]
pub struct RelabelSummary {
    pub positions: u64,
    pub total_weight: u64,
    /// Per shallow budget, aligned with the configured budgets.
    pub shallow: Vec<BudgetAgreement>,
    /// Deep-search self-stability: fraction of records whose best move
    /// did not change over the last completed iteration, and the mean
    /// number of best-move changes across all iterations.
    pub deep_last_iteration_stable: f64,
    pub deep_mean_best_move_changes: f64,
    /// Move-ordering quality: rank of the deep best action in the
    /// evaluator-induced root ordering.
    pub order_rank_mean: f64,
    pub order_top1_rate: f64,
    pub order_top3_rate: f64,
    /// Oracle metrics; `None` when no oracle was joined.
    pub oracle: Option<OracleSummary>,
}

/// Agreement of one shallow budget with the deep label.
#[derive(Clone, Debug, Serialize)]
pub struct BudgetAgreement {
    pub node_budget: u64,
    pub best_move_agree_rate: f64,
    /// Mean |shallow − deep| over records where both values are
    /// heuristic (inside the eval clamp), in score units.
    pub mean_abs_value_residual: f64,
    pub heuristic_pairs: u64,
}

/// Exact-oracle agreement per label source.
#[derive(Clone, Debug, Serialize)]
pub struct OracleSummary {
    pub joined: u64,
    pub missed: u64,
    /// WDL counts of joined positions (win/draw/loss for side to move).
    pub wdl_counts: [u64; 3],
    /// Optimal-decision rate per shallow budget (aligned), then deep.
    pub shallow_optimal_rate: Vec<f64>,
    pub deep_optimal_rate: f64,
    /// Fraction of joined records with child labels whose best-valued
    /// child is an exact optimal action.
    pub child_top_optimal_rate: Option<f64>,
}

pub fn summarize(records: &[TeacherRecord], shallow_budgets: &[u64]) -> RelabelSummary {
    let n = records.len() as f64;
    let positions = records.len() as u64;
    let total_weight = records.iter().map(|r| u64::from(r.weight)).sum();

    let shallow = shallow_budgets
        .iter()
        .enumerate()
        .map(|(i, &budget)| {
            let mut agree = 0u64;
            let mut residual_sum = 0f64;
            let mut heuristic_pairs = 0u64;
            for record in records {
                let label = &record.shallow[i];
                agree += u64::from(label.best_action == record.deep.best_action);
                if label.value.abs() <= SCORE_EVAL_MAX && record.deep.value.abs() <= SCORE_EVAL_MAX
                {
                    residual_sum += f64::from((label.value - record.deep.value).abs());
                    heuristic_pairs += 1;
                }
            }
            BudgetAgreement {
                node_budget: budget,
                best_move_agree_rate: agree as f64 / n,
                mean_abs_value_residual: if heuristic_pairs == 0 {
                    0.0
                } else {
                    residual_sum / heuristic_pairs as f64
                },
                heuristic_pairs,
            }
        })
        .collect();

    let mut last_stable = 0u64;
    let mut change_sum = 0u64;
    for record in records {
        let pd = &record.deep.per_depth;
        if pd.len() < 2 || pd[pd.len() - 1].0 == pd[pd.len() - 2].0 {
            last_stable += 1;
        }
        change_sum += pd.windows(2).filter(|w| w[0].0 != w[1].0).count() as u64;
    }

    let order_rank_mean = records
        .iter()
        .map(|r| f64::from(r.deep_best_order_rank))
        .sum::<f64>()
        / n;
    let order_top1_rate = records
        .iter()
        .filter(|r| r.deep_best_order_rank == 0)
        .count() as f64
        / n;
    let order_top3_rate = records
        .iter()
        .filter(|r| r.deep_best_order_rank < 3)
        .count() as f64
        / n;

    let joined: Vec<&TeacherRecord> = records.iter().filter(|r| r.exact_wdl.is_some()).collect();
    let oracle = if joined.is_empty() {
        None
    } else {
        let jn = joined.len() as f64;
        let mut wdl_counts = [0u64; 3];
        for record in &joined {
            wdl_counts[record.exact_wdl.expect("joined") as usize] += 1;
        }
        let optimal_rate = |pick: &dyn Fn(&TeacherRecord) -> ActionId| {
            joined
                .iter()
                .filter(|r| r.exact_optimal.as_ref().expect("joined").contains(&pick(r)))
                .count() as f64
                / jn
        };
        let shallow_optimal_rate = (0..shallow_budgets.len())
            .map(|i| optimal_rate(&move |r: &TeacherRecord| r.shallow[i].best_action))
            .collect();
        let deep_optimal_rate = optimal_rate(&|r: &TeacherRecord| r.deep.best_action);
        let with_children: Vec<&&TeacherRecord> =
            joined.iter().filter(|r| !r.children.is_empty()).collect();
        let child_top_optimal_rate = if with_children.is_empty() {
            None
        } else {
            Some(
                with_children
                    .iter()
                    .filter(|r| {
                        let best = r
                            .children
                            .iter()
                            .max_by_key(|c| c.value)
                            .expect("non-empty children");
                        r.exact_optimal
                            .as_ref()
                            .expect("joined")
                            .contains(&best.action)
                    })
                    .count() as f64
                    / with_children.len() as f64,
            )
        };
        Some(OracleSummary {
            joined: joined.len() as u64,
            missed: positions - joined.len() as u64,
            wdl_counts,
            shallow_optimal_rate,
            deep_optimal_rate,
            child_top_optimal_rate,
        })
    };

    RelabelSummary {
        positions,
        total_weight,
        shallow,
        deep_last_iteration_stable: last_stable as f64 / n,
        deep_mean_best_move_changes: change_sum as f64 / n,
        order_rank_mean,
        order_top1_rate,
        order_top3_rate,
        oracle,
    }
}

/// Sanity guard used by tests and callers: a deep label whose value
/// proves a terminal result must agree in sign with any exact WDL.
pub fn value_wdl_sign_consistent(value: i32, wdl: Wdl) -> bool {
    if value > SCORE_TERMINAL_BOUND {
        wdl == Wdl::Win
    } else if value < -SCORE_TERMINAL_BOUND {
        wdl == Wdl::Loss
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::connect_k::ConnectK;
    use crate::games::forward_chess::{ForwardChess, Ruleset};
    use crate::search::{solve_retrograde, ZeroEvaluator};

    fn random_spec(games: u64) -> TrajectorySpec {
        TrajectorySpec::Random { games }
    }

    #[test]
    fn replay_path_reconstructs_positions_exactly() {
        // Walk random trajectories on two rule families and check that
        // replaying every prefix reproduces the visited position key.
        let fc = ForwardChess::new(Ruleset::Tiny);
        let ck = ConnectK::new(3, 3, 3, true).unwrap();
        fn check<G: Game>(game: &G, seed: u64) {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            let mut state = game.initial_state();
            let mut moves = Vec::new();
            let mut path = Vec::new();
            while game.outcome(&state).is_none() && path.len() < 60 {
                let replayed = replay_path(game, &path).expect("prefix replays");
                assert_eq!(
                    game.position_key(&replayed),
                    game.position_key(&state),
                    "replaccording prefix must match the live state"
                );
                game.legal_moves(&state, &mut moves);
                let mv = moves[rng.gen_range(0..moves.len())];
                path.push(game.action_id(&state, mv));
                game.make_move(&mut state, mv);
            }
            // An illegal continuation must error, not panic.
            let mut bogus = path.clone();
            bogus.push(u32::MAX);
            assert!(replay_path(game, &bogus).is_err());
        }
        check(&fc, 7);
        check(&ck, 8);
    }

    #[test]
    fn record_serde_round_trip() {
        let record = TeacherRecord {
            path: vec![3, 1, 4],
            key: 0xdead_beef,
            ply: 3,
            weight: 2,
            provenance: Provenance::InternalSearch,
            evaluator: "zero".to_string(),
            shallow: vec![SearchLabel {
                node_budget: 50,
                nodes: 50,
                completed_depth: 2,
                value: -12,
                best_action: 4,
                per_depth: vec![(1, 3), (4, -12)],
            }],
            deep: SearchLabel {
                node_budget: 400,
                nodes: 398,
                completed_depth: 4,
                value: 25,
                best_action: 1,
                per_depth: vec![(1, 3), (4, -12), (1, 20), (1, 25)],
            },
            deep_best_order_rank: 1,
            children: vec![ChildLabel {
                action: 1,
                node_budget: 100,
                nodes: 100,
                completed_depth: 2,
                value: 25,
            }],
            exact_wdl: Some(Wdl::Draw),
            exact_optimal: Some(vec![1, 4]),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: TeacherRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record, back);
    }

    #[test]
    fn collect_positions_is_deterministic_and_deduplicated() {
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let run = |threads: usize| {
            collect_positions(&game, &random_spec(40), 2, 500, 11, threads, false, || {
                ZeroEvaluator
            })
        };
        let a = run(1);
        let b = run(4);
        assert_eq!(a, b, "sampling must not depend on thread count");
        assert!(!a.is_empty());
        let mut seen = std::collections::HashSet::new();
        for sample in &a {
            assert!(seen.insert(sample.key), "positions must be distinct");
            assert!(splitmix64(sample.key ^ SAMPLE_SALT).is_multiple_of(2));
            assert!(sample.weight >= 1);
            assert_eq!(sample.path.len() as u32, sample.ply);
        }
        // The initial position hashes into every trajectory: if sampled,
        // its weight must equal the number of games.
        if let Some(root) = a.iter().find(|s| s.path.is_empty()) {
            assert_eq!(root.weight, 40);
        }
        // The cap on distinct positions is respected.
        let capped = collect_positions(&game, &random_spec(40), 1, 5, 11, 2, false, || {
            ZeroEvaluator
        });
        assert_eq!(capped.len(), 5);
    }

    #[test]
    fn labelling_is_thread_independent_and_well_formed() {
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let samples = collect_positions(&game, &random_spec(20), 1, 60, 3, 2, false, || {
            ZeroEvaluator
        });
        let label = |threads: usize| {
            label_positions(
                &game,
                &samples,
                "zero",
                &[20, 100],
                2_000,
                true,
                200,
                threads,
                false,
                || ZeroEvaluator,
            )
        };
        let a = label(1);
        let b = label(4);
        assert_eq!(a, b, "labels must not depend on thread count");
        let mut state_moves = Vec::new();
        for record in &a {
            assert_eq!(record.shallow.len(), 2);
            assert_eq!(record.provenance, Provenance::InternalSearch);
            for l in record.shallow.iter().chain(std::iter::once(&record.deep)) {
                assert_eq!(
                    l.per_depth.len(),
                    l.completed_depth as usize,
                    "one per-depth entry per completed iteration"
                );
                if let Some(&(action, value)) = l.per_depth.last() {
                    assert_eq!(action, l.best_action);
                    assert_eq!(value, l.value);
                }
                assert!(l.nodes <= l.node_budget || l.completed_depth == 0);
            }
            // Children align with the legal-move list of the replayed state.
            let state = replay_path(&game, &record.path).unwrap();
            game.legal_moves(&state, &mut state_moves);
            assert_eq!(record.children.len(), state_moves.len());
            assert!(record.deep_best_order_rank < state_moves.len() as u32);
        }
    }

    #[test]
    fn oracle_join_and_deep_search_agree_on_solved_game() {
        // Connect-k 3x3 gravity has 505 states; a 50k-node budget solves
        // every sampled position outright, so the deep label's decision
        // must always be exactly optimal and mate-proven values must
        // agree with the oracle's WDL — an end-to-end cross-check of
        // trajectory sampling, replay, search, and the oracle join.
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let samples = collect_positions(&game, &random_spec(30), 1, 80, 5, 2, false, || {
            ZeroEvaluator
        });
        let mut records = label_positions(
            &game,
            &samples,
            "zero",
            &[30],
            50_000,
            true,
            50_000,
            2,
            false,
            || ZeroEvaluator,
        );
        let solution = solve_retrograde(&game, 10_000).unwrap();
        let stats = join_retrograde_oracle(&game, &mut records, &solution);
        assert_eq!(stats.missed, 0, "every reachable position must join");
        assert_eq!(stats.joined as usize, records.len());
        for record in &records {
            let optimal = record.exact_optimal.as_ref().unwrap();
            assert!(!optimal.is_empty());
            assert!(
                optimal.contains(&record.deep.best_action),
                "a full-width solve must pick an optimal move"
            );
            assert!(value_wdl_sign_consistent(
                record.deep.value,
                record.exact_wdl.unwrap()
            ));
            // The child argmax must also be optimal at this budget.
            let best_child = record.children.iter().max_by_key(|c| c.value).unwrap();
            assert!(optimal.contains(&best_child.action));
        }
        let summary = summarize(&records, &[30]);
        let oracle = summary.oracle.as_ref().unwrap();
        assert_eq!(oracle.deep_optimal_rate, 1.0);
        assert_eq!(oracle.child_top_optimal_rate, Some(1.0));
    }
}

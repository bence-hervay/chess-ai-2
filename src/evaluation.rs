//! Match evaluation: the paired arena.
//!
//! The arena plays independent games in parallel, one single-threaded game
//! per rayon task. Every game's RNG stream is derived only from
//! `(run_seed, pair, slot)`, so results are byte-identical regardless of
//! thread count or scheduling.

use crate::game::{ActionId, Game, Outcome, Player};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

/// Pairs are processed in batches so that game records can be streamed to
/// disk with bounded memory while preserving deterministic order.
const PAIR_BATCH: u64 = 8192;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Deterministic per-game RNG derived from the run seed, the pair index,
/// and the slot within the pair (0 or 1).
pub fn game_rng(run_seed: u64, pair: u64, slot: u64) -> ChaCha12Rng {
    let mixed =
        splitmix64(splitmix64(run_seed) ^ splitmix64(pair.wrapping_mul(2).wrapping_add(slot)));
    ChaCha12Rng::seed_from_u64(mixed)
}

/// One completed game inside a pair.
///
/// In slot 0 agent A plays as Player One; in slot 1 the colours are swapped.
#[derive(Clone, Debug, Serialize)]
pub struct GameRecord {
    pub pair: u64,
    pub slot: u8,
    pub plies: u32,
    /// `"p1_win"`, `"p2_win"`, or `"draw"`.
    pub outcome: &'static str,
    pub actions: Vec<ActionId>,
}

impl GameRecord {
    /// Score of agent A in this game: 1.0 win, 0.5 draw, 0.0 loss.
    pub fn agent_a_score(&self) -> f64 {
        let a_is_p1 = self.slot == 0;
        match (self.outcome, a_is_p1) {
            ("draw", _) => 0.5,
            ("p1_win", true) | ("p2_win", false) => 1.0,
            _ => 0.0,
        }
    }
}

fn outcome_str(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Win(Player::One) => "p1_win",
        Outcome::Win(Player::Two) => "p2_win",
        Outcome::Draw => "draw",
    }
}

/// Work counters accumulated across all games of a run.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ArenaCounters {
    pub games: u64,
    /// Moves actually played (equivalently: non-initial states generated).
    pub moves_played: u64,
    /// Total legal moves enumerated across all move generations.
    pub legal_moves_generated: u64,
}

impl ArenaCounters {
    fn add(&mut self, other: &ArenaCounters) {
        self.games += other.games;
        self.moves_played += other.moves_played;
        self.legal_moves_generated += other.legal_moves_generated;
    }
}

/// Play one game where every move is chosen uniformly at random among the
/// legal moves, using `rng` as the only source of randomness.
fn play_random_game<G: Game>(
    game: &G,
    rng: &mut ChaCha12Rng,
    counters: &mut ArenaCounters,
) -> (Vec<ActionId>, Outcome) {
    let mut state = game.initial_state();
    let mut moves = Vec::new();
    let mut actions = Vec::new();
    loop {
        if let Some(outcome) = game.outcome(&state) {
            counters.games += 1;
            return (actions, outcome);
        }
        game.legal_moves(&state, &mut moves);
        counters.legal_moves_generated += moves.len() as u64;
        let mv = moves[rng.gen_range(0..moves.len())];
        actions.push(game.action_id(&state, mv));
        game.make_move(&mut state, mv);
        counters.moves_played += 1;
    }
}

/// Aggregate result of a random-versus-random paired arena run.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ArenaSummary {
    pub pairs: u64,
    pub games: u64,
    pub p1_wins: u64,
    pub p2_wins: u64,
    pub draws: u64,
    pub agent_a_points: f64,
    pub total_plies: u64,
    pub counters: ArenaCounters,
}

impl ArenaSummary {
    fn absorb(&mut self, record: &GameRecord) {
        self.games += 1;
        match record.outcome {
            "p1_win" => self.p1_wins += 1,
            "p2_win" => self.p2_wins += 1,
            _ => self.draws += 1,
        }
        self.agent_a_points += record.agent_a_score();
        self.total_plies += u64::from(record.plies);
    }

    fn add(&mut self, other: &ArenaSummary) {
        self.games += other.games;
        self.p1_wins += other.p1_wins;
        self.p2_wins += other.p2_wins;
        self.draws += other.draws;
        self.agent_a_points += other.agent_a_points;
        self.total_plies += other.total_plies;
        self.counters.add(&other.counters);
    }
}

/// Run `pairs` paired random-versus-random games on `threads` workers.
///
/// `sink` receives the JSONL-serialized game records of each batch in
/// deterministic pair order. Serialization happens inside the workers so
/// the serial section is only the file write.
pub fn run_random_arena<G: Game>(
    game: &G,
    pairs: u64,
    run_seed: u64,
    threads: usize,
    mut sink: impl FnMut(&str),
) -> ArenaSummary {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    let mut summary = ArenaSummary {
        pairs,
        ..ArenaSummary::default()
    };

    let mut batch_start = 0u64;
    while batch_start < pairs {
        let batch_end = (batch_start + PAIR_BATCH).min(pairs);
        let results: Vec<(ArenaSummary, String)> = pool.install(|| {
            (batch_start..batch_end)
                .into_par_iter()
                .map(|pair| {
                    let mut local = ArenaSummary::default();
                    let mut lines = String::new();
                    for slot in 0..2u64 {
                        let mut rng = game_rng(run_seed, pair, slot);
                        let (actions, outcome) =
                            play_random_game(game, &mut rng, &mut local.counters);
                        let record = GameRecord {
                            pair,
                            slot: slot as u8,
                            plies: actions.len() as u32,
                            outcome: outcome_str(outcome),
                            actions,
                        };
                        local.absorb(&record);
                        lines.push_str(
                            &serde_json::to_string(&record).expect("serializable record"),
                        );
                        lines.push('\n');
                    }
                    (local, lines)
                })
                .collect()
        });
        for (local, lines) in &results {
            summary.add(local);
            sink(lines);
        }
        batch_start = batch_end;
    }
    summary
}

/// Raw-model quality against exact oracle labels (plan §17.3, §32).
#[derive(Clone, Debug, Serialize)]
pub struct OracleMetrics {
    pub states: usize,
    pub wdl_accuracy: f64,
    /// Mean natural-log loss of the WDL head.
    pub wdl_log_loss: f64,
    /// Mean Brier score over the three WDL classes.
    pub brier: f64,
    /// Fraction of states where the argmax legal action is optimal.
    pub action_accuracy: f64,
    /// Mean policy probability mass on the optimal-action set.
    pub optimal_mass: f64,
    /// Mean decision regret in WDL levels lost (win->draw or draw->loss
    /// = 1, win->loss = 2).
    pub mean_regret_levels: f64,
    /// Accuracy of the argmax action stratified by state value
    /// `[loss, draw, win]`.
    pub action_accuracy_by_wdl: [f64; 3],
}

/// Evaluate a raw model (no search) against exact labels.
pub fn evaluate_model_oracle(
    net: &crate::model::PolicyValueNet<crate::model::InferBackend>,
    examples: &[crate::training::Example],
    dims: crate::model::ModelDims,
    max_features: usize,
) -> OracleMetrics {
    use crate::training::make_batch;
    let device = Default::default();
    let mut wdl_correct = 0usize;
    let mut log_loss = 0.0f64;
    let mut brier = 0.0f64;
    let mut action_correct = 0usize;
    let mut optimal_mass = 0.0f64;
    let mut regret_levels = 0u64;
    let mut by_wdl_correct = [0usize; 3];
    let mut by_wdl_total = [0usize; 3];

    for chunk in examples.chunks(1024) {
        let rows: Vec<crate::training::TrainRow> =
            chunk.iter().map(crate::training::TrainRow::from).collect();
        let refs: Vec<&crate::training::TrainRow> = rows.iter().collect();
        let batch = make_batch::<crate::model::InferBackend>(&refs, dims, max_features, &device);
        let (wdl_logits, action_logits) =
            net.forward(batch.feature_ids.clone(), batch.feature_mask.clone());
        let wdl_probs = burn::tensor::activation::softmax(wdl_logits, 1)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let masked = action_logits + (batch.legal_mask.clone() - 1.0) * 1e9;
        let action_probs = burn::tensor::activation::softmax(masked, 1)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        for (row, example) in chunk.iter().enumerate() {
            let target = example.wdl as usize;
            let probs = &wdl_probs[row * 3..row * 3 + 3];
            let predicted = (0..3)
                .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
                .unwrap();
            if predicted == target {
                wdl_correct += 1;
            }
            log_loss += -f64::from(probs[target].max(1e-12)).ln();
            for (class, &p) in probs.iter().enumerate() {
                let t = if class == target { 1.0 } else { 0.0 };
                brier += (f64::from(p) - t).powi(2);
            }

            let action_count = dims.action_count;
            let row_probs = &action_probs[row * action_count..(row + 1) * action_count];
            let chosen_index = example
                .legal
                .iter()
                .enumerate()
                .max_by(|(_, &a), (_, &b)| row_probs[a as usize].total_cmp(&row_probs[b as usize]))
                .map(|(i, _)| i)
                .expect("legal actions");
            let optimal = example.optimal_indices();
            by_wdl_total[target] += 1;
            if optimal.contains(&chosen_index) {
                action_correct += 1;
                by_wdl_correct[target] += 1;
            }
            optimal_mass += optimal
                .iter()
                .map(|&i| f64::from(row_probs[example.legal[i] as usize]))
                .sum::<f64>();
            let child = example.child_wdl[chosen_index];
            regret_levels += match (example.wdl, child) {
                (a, b) if a == b => 0,
                (crate::search::Wdl::Win, crate::search::Wdl::Draw) => 1,
                (crate::search::Wdl::Win, crate::search::Wdl::Loss) => 2,
                (crate::search::Wdl::Draw, crate::search::Wdl::Loss) => 1,
                _ => 0,
            };
        }
    }

    let n = examples.len().max(1) as f64;
    let ratio = |c: usize, t: usize| {
        if t == 0 {
            f64::NAN
        } else {
            c as f64 / t as f64
        }
    };
    OracleMetrics {
        states: examples.len(),
        wdl_accuracy: wdl_correct as f64 / n,
        wdl_log_loss: log_loss / n,
        brier: brier / n,
        action_accuracy: action_correct as f64 / n,
        optimal_mass: optimal_mass / n,
        mean_regret_levels: regret_levels as f64 / n,
        action_accuracy_by_wdl: [
            ratio(by_wdl_correct[0], by_wdl_total[0]),
            ratio(by_wdl_correct[1], by_wdl_total[1]),
            ratio(by_wdl_correct[2], by_wdl_total[2]),
        ],
    }
}

/// Decision quality when the model searches before acting (plan §23
/// diagnostic matrix, "learned model plus search" column). States are a
/// deterministic hash-selected sample of one split of the exact corpus.
#[derive(Clone, Debug, Serialize)]
pub struct SearchedMetrics {
    pub states: usize,
    pub node_budget: u64,
    /// Fraction of sampled states where the searched action is optimal.
    pub action_accuracy: f64,
    pub mean_regret_levels: f64,
    pub mean_nodes: f64,
    pub mean_completed_depth: f64,
}

/// Corpus split selector matching the training split hash buckets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CorpusSplit {
    Val,
    Test,
}

/// A candidate state for searched-decision evaluation: selection hash,
/// state, exact value, legal action IDs, exact child values.
type SolvedCandidate<G> = (
    u64,
    <G as Game>::State,
    crate::search::Wdl,
    Vec<u32>,
    Vec<crate::search::Wdl>,
);

fn splitmix64_local(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Evaluate search-guided decisions on up to `cap` states of `split`,
/// selected deterministically by position hash (independent of `cap`
/// ordering effects: a state is in the sample iff its selection hash is
/// among the smallest). Runs games' searches in parallel.
pub fn searched_decision_metrics<G: Game>(
    game: &G,
    net: &crate::model::CompiledNet,
    split: CorpusSplit,
    cap: usize,
    node_budget: u64,
    threads: usize,
) -> SearchedMetrics {
    use crate::search::{enumerate_solved, ExactSolver, MoveOrdering, Searcher, Wdl};

    // Collect the split's states with their exact child values.
    let mut solver = ExactSolver::new();
    let mut candidates: Vec<SolvedCandidate<G>> = Vec::new();
    enumerate_solved(game, &mut solver, |position| {
        let bucket = splitmix64_local(game.position_key(position.state)) % 10;
        let wanted = match split {
            CorpusSplit::Val => 8,
            CorpusSplit::Test => 9,
        };
        if bucket == wanted {
            let selection = splitmix64_local(game.position_key(position.state) ^ 0x5EA3C4);
            candidates.push((
                selection,
                position.state.clone(),
                position.value,
                position
                    .legal
                    .iter()
                    .map(|&m| game.action_id(position.state, m))
                    .collect(),
                position.child_values.to_vec(),
            ));
        }
    });
    candidates.sort_by_key(|(selection, ..)| *selection);
    candidates.truncate(cap);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    let results: Vec<(bool, u64, u64, u32)> = pool.install(|| {
        candidates
            .par_iter()
            .map(|(_, state, value, legal, child_values)| {
                let mut evaluator = crate::model::ModelEvaluator::new(net);
                let mut searcher: Searcher<G> = Searcher::new(
                    Some(crate::training::SELFPLAY_TT_LOG2),
                    MoveOrdering::Natural,
                );
                let mut s = state.clone();
                let result = searcher.search(game, &mut s, 512, node_budget, &mut evaluator);
                let chosen = result.best_move.expect("non-terminal state");
                let chosen_id = game.action_id(state, chosen);
                let index = legal
                    .iter()
                    .position(|&a| a == chosen_id)
                    .expect("chosen move is legal");
                let child = child_values[index];
                let regret = match (*value, child) {
                    (a, b) if a == b => 0u64,
                    (Wdl::Win, Wdl::Draw) | (Wdl::Draw, Wdl::Loss) => 1,
                    (Wdl::Win, Wdl::Loss) => 2,
                    _ => 0,
                };
                (
                    child == *value,
                    regret,
                    result.nodes,
                    result.completed_depth,
                )
            })
            .collect()
    });

    let n = results.len().max(1) as f64;
    SearchedMetrics {
        states: results.len(),
        node_budget,
        action_accuracy: results.iter().filter(|(ok, ..)| *ok).count() as f64 / n,
        mean_regret_levels: results.iter().map(|(_, r, ..)| *r as f64).sum::<f64>() / n,
        mean_nodes: results
            .iter()
            .map(|(_, _, nodes, _)| *nodes as f64)
            .sum::<f64>()
            / n,
        mean_completed_depth: results.iter().map(|(.., d)| f64::from(*d)).sum::<f64>() / n,
    }
}

/// Search-depth disagreement analysis (§24): on one shared state
/// sample, compare the action chosen at each budget with the deepest
/// budget's choice and with exact optimality.
#[derive(Clone, Debug, Serialize)]
pub struct DisagreementReport {
    pub node_budget: u64,
    pub states: usize,
    /// Fraction of states where this budget picks a different action
    /// than the deepest probed budget.
    pub disagreement_with_deepest: f64,
    /// Of the disagreeing states: deepest is optimal, this budget is not.
    pub deeper_fixes: u64,
    /// Of the disagreeing states: this budget is optimal, deepest is not.
    pub deeper_breaks: u64,
    /// Both optimal (equally good alternatives) or both suboptimal.
    pub neutral: u64,
    pub action_accuracy: f64,
}

/// Run every probed budget on the same hash-selected sample of `split`.
pub fn search_disagreement_analysis<G: Game>(
    game: &G,
    net: &crate::model::CompiledNet,
    split: CorpusSplit,
    cap: usize,
    budgets: &[u64],
    threads: usize,
) -> Vec<DisagreementReport> {
    use crate::search::{enumerate_solved, ExactSolver, MoveOrdering, Searcher};
    let mut solver = ExactSolver::new();
    let mut candidates: Vec<SolvedCandidate<G>> = Vec::new();
    enumerate_solved(game, &mut solver, |position| {
        let bucket = splitmix64_local(game.position_key(position.state)) % 10;
        let wanted = match split {
            CorpusSplit::Val => 8,
            CorpusSplit::Test => 9,
        };
        if bucket == wanted {
            let selection = splitmix64_local(game.position_key(position.state) ^ 0x5EA3C4);
            candidates.push((
                selection,
                position.state.clone(),
                position.value,
                position
                    .legal
                    .iter()
                    .map(|&m| game.action_id(position.state, m))
                    .collect(),
                position.child_values.to_vec(),
            ));
        }
    });
    candidates.sort_by_key(|(selection, ..)| *selection);
    candidates.truncate(cap);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    // Per state: chosen action index and optimality per budget.
    let per_state: Vec<Vec<(usize, bool)>> = pool.install(|| {
        candidates
            .par_iter()
            .map(|(_, state, value, legal, child_values)| {
                let mut evaluator = crate::model::ModelEvaluator::new(net);
                budgets
                    .iter()
                    .map(|&budget| {
                        let mut searcher: Searcher<G> = Searcher::new(
                            Some(crate::training::SELFPLAY_TT_LOG2),
                            MoveOrdering::Natural,
                        );
                        let mut s = state.clone();
                        let result = searcher.search(game, &mut s, 512, budget, &mut evaluator);
                        let chosen = result.best_move.expect("non-terminal state");
                        let chosen_id = game.action_id(state, chosen);
                        let index = legal
                            .iter()
                            .position(|&a| a == chosen_id)
                            .expect("chosen move is legal");
                        (index, child_values[index] == *value)
                    })
                    .collect()
            })
            .collect()
    });

    let deepest = budgets.len() - 1;
    let n = per_state.len().max(1) as f64;
    budgets
        .iter()
        .enumerate()
        .map(|(bi, &budget)| {
            let mut disagree = 0u64;
            let mut fixes = 0u64;
            let mut breaks = 0u64;
            let mut neutral = 0u64;
            let mut correct = 0u64;
            for row in &per_state {
                let (index, optimal) = row[bi];
                let (deep_index, deep_optimal) = row[deepest];
                correct += u64::from(optimal);
                if index != deep_index {
                    disagree += 1;
                    match (deep_optimal, optimal) {
                        (true, false) => fixes += 1,
                        (false, true) => breaks += 1,
                        _ => neutral += 1,
                    }
                }
            }
            DisagreementReport {
                node_budget: budget,
                states: per_state.len(),
                disagreement_with_deepest: disagree as f64 / n,
                deeper_fixes: fixes,
                deeper_breaks: breaks,
                neutral,
                action_accuracy: correct as f64 / n,
            }
        })
        .collect()
}

/// Exploitability against perfect opposition (plan §32.3): the agent
/// (search + model) plays both colours from the initial state and every
/// distinct state at ply 1 and 2; the opponent always plays the first
/// optimal move. A game "drops levels" when its result is worse for the
/// agent than the start state's exact value.
#[derive(Clone, Debug, Serialize)]
pub struct ExploitabilityReport {
    pub games: u64,
    /// Sum over games of WDL levels lost versus the start value.
    pub levels_lost: u64,
    pub mean_levels_lost: f64,
    /// Games where a win or draw was avoidably given away.
    pub avoidable_drops: u64,
    pub agent_wins: u64,
    pub draws: u64,
    pub agent_losses: u64,
}

pub fn exploitability_vs_perfect<G: Game>(
    game: &G,
    net: &crate::model::CompiledNet,
    node_budget: u64,
    threads: usize,
) -> ExploitabilityReport {
    use crate::game::Player;
    use crate::search::{ExactSolver, MoveOrdering, Searcher, Wdl};

    // Start set: initial state plus all distinct states at plies 1-2.
    let mut starts: Vec<G::State> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let initial = game.initial_state();
    let mut frontier = vec![(initial.clone(), 0u32)];
    while let Some((state, ply)) = frontier.pop() {
        if game.outcome(&state).is_some() {
            continue;
        }
        if seen.insert(game.position_key(&state)) {
            starts.push(state.clone());
        }
        if ply < 2 {
            let mut moves = Vec::new();
            game.legal_moves(&state, &mut moves);
            for &mv in &moves {
                let mut child = state.clone();
                game.make_move(&mut child, mv);
                frontier.push((child, ply + 1));
            }
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    let games: Vec<(u64, bool, i8)> = pool.install(|| {
        starts
            .par_iter()
            .flat_map(|start| {
                [Player::One, Player::Two]
                    .into_par_iter()
                    .map(|agent_side| {
                        let mut solver = ExactSolver::new();
                        let mut evaluator = crate::model::ModelEvaluator::new(net);
                        let mut searcher: Searcher<G> = Searcher::new(
                            Some(crate::training::SELFPLAY_TT_LOG2),
                            MoveOrdering::Natural,
                        );
                        let mut state = start.clone();
                        // Exact value of the start from the agent's perspective.
                        let mover_value = solver.solve(game, &mut state);
                        let agent_to_move = game.side_to_move(&state) == agent_side;
                        let start_value = if agent_to_move {
                            mover_value
                        } else {
                            mover_value.flip()
                        };
                        let outcome = loop {
                            if let Some(outcome) = game.outcome(&state) {
                                break outcome;
                            }
                            let mv = if game.side_to_move(&state) == agent_side {
                                let result = searcher.search(
                                    game,
                                    &mut state,
                                    512,
                                    node_budget,
                                    &mut evaluator,
                                );
                                result
                                    .best_move
                                    .expect("non-terminal search returns a move")
                            } else {
                                let mut optimal = Vec::new();
                                solver.optimal_moves(game, &mut state, &mut optimal);
                                optimal[0]
                            };
                            game.make_move(&mut state, mv);
                        };
                        let result_value = match outcome {
                            crate::game::Outcome::Draw => Wdl::Draw,
                            crate::game::Outcome::Win(winner) if winner == agent_side => Wdl::Win,
                            crate::game::Outcome::Win(_) => Wdl::Loss,
                        };
                        let levels = (start_value as i64 - result_value as i64).max(0) as u64;
                        let score = match result_value {
                            Wdl::Win => 1i8,
                            Wdl::Draw => 0,
                            Wdl::Loss => -1,
                        };
                        (levels, levels > 0, score)
                    })
            })
            .collect()
    });

    let total = games.len() as u64;
    let levels_lost: u64 = games.iter().map(|(l, ..)| *l).sum();
    ExploitabilityReport {
        games: total,
        levels_lost,
        mean_levels_lost: levels_lost as f64 / total.max(1) as f64,
        avoidable_drops: games.iter().filter(|(_, dropped, _)| *dropped).count() as u64,
        agent_wins: games.iter().filter(|(.., s)| *s > 0).count() as u64,
        draws: games.iter().filter(|(.., s)| *s == 0).count() as u64,
        agent_losses: games.iter().filter(|(.., s)| *s < 0).count() as u64,
    }
}

/// Result of a paired model-versus-model match (§12.6 non-exact
/// promotion). The candidate's score is 1 per win, 0.5 per draw.
#[derive(Clone, Debug, Serialize)]
pub struct MatchResult {
    pub pairs: u64,
    pub games: u64,
    pub candidate_points: f64,
    /// candidate_points / games, in [0, 1].
    pub score: f64,
    /// 95% lower confidence bound on the score, treating pairs as the
    /// independent unit (normal approximation).
    pub score_lcb95: f64,
    pub candidate_wins: u64,
    pub draws: u64,
    pub candidate_losses: u64,
    pub mean_plies: f64,
}

/// Play `pairs` paired games between two model+search agents. Each pair
/// shares one opening (uniform random legal moves for `opening_plies`
/// plies, seeded by `(run_seed, pair)`) and swaps colours between its
/// two games; both agents search deterministically with `node_budget`.
#[allow(clippy::too_many_arguments)] // match protocol parameters are irreducible
pub fn play_paired_match<G: Game>(
    game: &G,
    candidate: &crate::model::CompiledNet,
    champion: &crate::model::CompiledNet,
    pairs: u64,
    opening_plies: u32,
    candidate_nodes: u64,
    champion_nodes: u64,
    run_seed: u64,
    threads: usize,
) -> MatchResult {
    use crate::model::ModelEvaluator;
    use crate::search::{MoveOrdering, Searcher};

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    // (candidate score per pair over 2 games, plies)
    let per_pair: Vec<(f64, u64, [u64; 3])> = pool.install(|| {
        (0..pairs)
            .into_par_iter()
            .map(|pair| {
                let mut pair_score = 0.0f64;
                let mut plies_total = 0u64;
                let mut tally = [0u64; 3]; // win/draw/loss for candidate
                for slot in 0..2u64 {
                    // Shared opening: derived from the pair only, so both
                    // slots start from the same position.
                    let mut opening_rng = game_rng(run_seed, pair, 2);
                    let mut state = game.initial_state();
                    let mut moves = Vec::new();
                    for _ in 0..opening_plies {
                        if game.outcome(&state).is_some() {
                            break;
                        }
                        game.legal_moves(&state, &mut moves);
                        let mv = moves[opening_rng.gen_range(0..moves.len())];
                        game.make_move(&mut state, mv);
                    }
                    let candidate_is_p1 = slot == 0;
                    let mut cand_eval = ModelEvaluator::new(candidate);
                    let mut champ_eval = ModelEvaluator::new(champion);
                    let mut cand_search: Searcher<G> = Searcher::new(
                        Some(crate::training::SELFPLAY_TT_LOG2),
                        MoveOrdering::Natural,
                    );
                    let mut champ_search: Searcher<G> = Searcher::new(
                        Some(crate::training::SELFPLAY_TT_LOG2),
                        MoveOrdering::Natural,
                    );
                    let outcome = loop {
                        if let Some(outcome) = game.outcome(&state) {
                            break outcome;
                        }
                        let mover_is_candidate =
                            (game.side_to_move(&state) == Player::One) == candidate_is_p1;
                        let result = if mover_is_candidate {
                            cand_search.search(
                                game,
                                &mut state,
                                512,
                                candidate_nodes,
                                &mut cand_eval,
                            )
                        } else {
                            champ_search.search(
                                game,
                                &mut state,
                                512,
                                champion_nodes,
                                &mut champ_eval,
                            )
                        };
                        game.make_move(
                            &mut state,
                            result
                                .best_move
                                .expect("non-terminal search returns a move"),
                        );
                        plies_total += 1;
                    };
                    let score = match outcome {
                        Outcome::Draw => 0.5,
                        Outcome::Win(winner) => {
                            if (winner == Player::One) == candidate_is_p1 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    pair_score += score;
                    if score > 0.75 {
                        tally[0] += 1;
                    } else if score > 0.25 {
                        tally[1] += 1;
                    } else {
                        tally[2] += 1;
                    }
                }
                (pair_score, plies_total, tally)
            })
            .collect()
    });

    let games = pairs * 2;
    let points: f64 = per_pair.iter().map(|(s, ..)| s).sum();
    let score = points / games as f64;
    // Variance over per-pair scores (each pair contributes score/2 in [0,1]).
    let pair_scores: Vec<f64> = per_pair.iter().map(|(s, ..)| s / 2.0).collect();
    let n = pairs.max(1) as f64;
    let mean = pair_scores.iter().sum::<f64>() / n;
    let var = pair_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n.max(1.0);
    let lcb = mean - 1.96 * (var / n).sqrt();
    let total_plies: u64 = per_pair.iter().map(|(_, p, _)| p).sum();
    MatchResult {
        pairs,
        games,
        candidate_points: points,
        score,
        score_lcb95: lcb,
        candidate_wins: per_pair.iter().map(|(.., t)| t[0]).sum(),
        draws: per_pair.iter().map(|(.., t)| t[1]).sum(),
        candidate_losses: per_pair.iter().map(|(.., t)| t[2]).sum(),
        mean_plies: total_plies as f64 / games.max(1) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::connect_k::ConnectK;

    fn run(threads: usize, seed: u64) -> (ArenaSummary, String) {
        let game = ConnectK::new(7, 6, 4, true).unwrap();
        let mut jsonl = String::new();
        let summary = run_random_arena(&game, 50, seed, threads, |lines| jsonl.push_str(lines));
        (summary, jsonl)
    }

    #[test]
    fn identical_records_across_thread_counts() {
        let (s1, r1) = run(1, 7);
        let (s4, r4) = run(4, 7);
        assert_eq!(s1.games, 100);
        assert_eq!(s1.p1_wins, s4.p1_wins);
        assert_eq!(s1.p2_wins, s4.p2_wins);
        assert_eq!(s1.draws, s4.draws);
        assert_eq!(
            r1, r4,
            "records must be byte-identical across thread counts"
        );
    }

    #[test]
    fn different_seeds_differ() {
        let (_, r1) = run(2, 1);
        let (_, r2) = run(2, 2);
        assert_ne!(r1, r2, "different run seeds should produce different games");
    }

    #[test]
    fn recorded_games_are_terminal_and_legal_length() {
        let (summary, jsonl) = run(2, 3);
        let records: Vec<serde_json::Value> = jsonl
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(summary.games, records.len() as u64);
        for r in &records {
            let plies = r["plies"].as_u64().unwrap();
            assert!(plies >= 7, "a 7x6 k=4 game needs at least 7 plies");
            assert!(plies <= 42);
            assert_eq!(r["actions"].as_array().unwrap().len() as u64, plies);
        }
    }
}

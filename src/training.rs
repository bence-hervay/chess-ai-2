//! Supervised training on exact corpora (Phase 2) and the fixed training
//! recipe.
//!
//! Recipe constants below are part of the versioned recipe (plan §8):
//! changing one requires a named experiment and a DECISIONS.md entry.

use crate::game::Game;
use crate::model::{ModelDims, PolicyValueNet, TrainBackend};
use crate::search::{enumerate_solved, solve_retrograde, ExactSolver, Wdl};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::tensor::activation::log_softmax;
use burn::tensor::ElementConversion;
use burn::tensor::{Int, Tensor, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::Serialize;

/// Fixed training recipe v1 (Adam, constant learning rate, summed
/// cross-entropy losses `L = L_wdl + L_policy`).
pub const BATCH_SIZE: usize = 256;
pub const LEARNING_RATE: f64 = 1e-3;
/// Validation metrics are computed every this many optimizer steps.
pub const EVAL_EVERY: u64 = 1000;

/// One labelled position. Everything is in side-to-move perspective and
/// uses stable action IDs.
#[derive(Clone, Debug)]
pub struct Example {
    pub features: Vec<u32>,
    pub legal: Vec<u32>,
    /// Exact value of each legal action (aligned with `legal`).
    pub child_wdl: Vec<Wdl>,
    pub wdl: Wdl,
    pub ply: u32,
}

impl Example {
    /// Indices into `legal` of the game-theoretically optimal actions.
    pub fn optimal_indices(&self) -> Vec<usize> {
        self.child_wdl
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == self.wdl)
            .map(|(i, _)| i)
            .collect()
    }
}

/// One training row: the policy target is uniform over
/// `policy_actions` and the value target is `wdl` (both side-to-move).
/// Oracle rows target the optimal-action set; self-play rows target the
/// single expert-search action (plan §12.2).
#[derive(Clone, Debug)]
pub struct TrainRow {
    pub features: Vec<u32>,
    pub legal: Vec<u32>,
    pub wdl: Wdl,
    pub policy_actions: Vec<u32>,
}

impl From<&Example> for TrainRow {
    fn from(example: &Example) -> TrainRow {
        TrainRow {
            features: example.features.clone(),
            legal: example.legal.clone(),
            wdl: example.wdl,
            policy_actions: example
                .optimal_indices()
                .into_iter()
                .map(|i| example.legal[i])
                .collect(),
        }
    }
}

/// Exact-state dataset stratified by position hash: 80% train, 10%
/// validation, 10% test. States are unique, so splits cannot leak.
pub struct ExactDataset {
    pub train: Vec<Example>,
    pub val: Vec<Example>,
    pub test: Vec<Example>,
    pub max_features: usize,
}

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Enumerate and solve every reachable state of `game`, splitting by
/// position-key hash.
pub fn build_exact_dataset<G: Game>(game: &G) -> ExactDataset {
    let mut solver = ExactSolver::new();
    let mut dataset = ExactDataset {
        train: Vec::new(),
        val: Vec::new(),
        test: Vec::new(),
        max_features: 0,
    };
    let mut features = Vec::new();
    enumerate_solved(game, &mut solver, |position| {
        game.encode_features(position.state, &mut features);
        let example = Example {
            features: features.clone(),
            legal: position
                .legal
                .iter()
                .map(|&m| game.action_id(position.state, m))
                .collect(),
            child_wdl: position.child_values.to_vec(),
            wdl: position.value,
            ply: position.ply,
        };
        dataset.max_features = dataset.max_features.max(example.features.len());
        let bucket = splitmix64(game.position_key(position.state)) % 10;
        match bucket {
            0..=7 => dataset.train.push(example),
            8 => dataset.val.push(example),
            _ => dataset.test.push(example),
        }
    });
    dataset
}

/// Second hash stream for thinning evaluation buckets, so thinning is
/// independent of the split assignment.
const EVAL_THIN_SALT: u64 = 0x5eed_ab1e_0f0f_0f0f;

/// Oracle dataset for repetition-capable games via the retrograde
/// solver. Evaluation-only: `train` stays empty; `val`/`test` are the
/// usual position-key buckets, deterministically thinned to at most
/// about `eval_cap` states each so evaluation cost stays bounded on
/// large instances.
pub fn build_retrograde_dataset<G: Game>(
    game: &G,
    max_positions: usize,
    eval_cap: usize,
) -> Result<ExactDataset, String> {
    let solution = solve_retrograde(game, max_positions)?;
    let mut dataset = ExactDataset {
        train: Vec::new(),
        val: Vec::new(),
        test: Vec::new(),
        max_features: 0,
    };
    let mut bucket_sizes = [0u64; 2];
    for state in &solution.states {
        if game.outcome(state).is_some() {
            continue;
        }
        match splitmix64(game.position_key(state)) % 10 {
            8 => bucket_sizes[0] += 1,
            9 => bucket_sizes[1] += 1,
            _ => {}
        }
    }
    let denominators =
        bucket_sizes.map(|n| n.div_ceil(eval_cap.max(1) as u64).max(1));
    let mut features = Vec::new();
    let mut legal = Vec::new();
    for (index, state) in solution.states.iter().enumerate() {
        if game.outcome(state).is_some() {
            continue;
        }
        let key = game.position_key(state);
        let slot = match splitmix64(key) % 10 {
            8 => 0usize,
            9 => 1,
            _ => continue,
        };
        if splitmix64(key ^ EVAL_THIN_SALT) % denominators[slot] != 0 {
            continue;
        }
        game.encode_features(state, &mut features);
        game.legal_moves(state, &mut legal);
        let example = Example {
            features: features.clone(),
            legal: legal.iter().map(|&m| game.action_id(state, m)).collect(),
            child_wdl: solution.child_values(index),
            wdl: solution.values[index],
            ply: 0,
        };
        dataset.max_features = dataset.max_features.max(example.features.len());
        if slot == 0 {
            dataset.val.push(example);
        } else {
            dataset.test.push(example);
        }
    }
    Ok(dataset)
}

/// Tensor batch for a slice of examples.
pub struct Batch<B: Backend> {
    pub feature_ids: Tensor<B, 2, Int>,
    pub feature_mask: Tensor<B, 2>,
    /// One-hot WDL targets `[batch, 3]` in `Wdl` discriminant order.
    pub wdl_target: Tensor<B, 2>,
    /// Uniform-over-optimal policy target `[batch, action_count]`.
    pub policy_target: Tensor<B, 2>,
    /// 1.0 for legal actions, 0.0 otherwise `[batch, action_count]`.
    pub legal_mask: Tensor<B, 2>,
}

pub fn make_batch<B: Backend>(
    rows: &[&TrainRow],
    dims: ModelDims,
    max_features: usize,
    device: &B::Device,
) -> Batch<B> {
    let n = rows.len();
    let pad = dims.feature_count as i64;
    let mut ids = vec![pad; n * max_features];
    let mut mask = vec![0.0f32; n * max_features];
    let mut wdl = vec![0.0f32; n * 3];
    let mut policy = vec![0.0f32; n * dims.action_count];
    let mut legal = vec![0.0f32; n * dims.action_count];
    for (row, example) in rows.iter().enumerate() {
        for (i, &f) in example.features.iter().enumerate() {
            ids[row * max_features + i] = i64::from(f);
            mask[row * max_features + i] = 1.0;
        }
        wdl[row * 3 + example.wdl as usize] = 1.0;
        for &a in &example.legal {
            legal[row * dims.action_count + a as usize] = 1.0;
        }
        let share = 1.0 / example.policy_actions.len() as f32;
        for &a in &example.policy_actions {
            policy[row * dims.action_count + a as usize] = share;
        }
    }
    Batch {
        feature_ids: Tensor::from_data(TensorData::new(ids, [n, max_features]), device),
        feature_mask: Tensor::from_data(TensorData::new(mask, [n, max_features]), device),
        wdl_target: Tensor::from_data(TensorData::new(wdl, [n, 3]), device),
        policy_target: Tensor::from_data(TensorData::new(policy, [n, dims.action_count]), device),
        legal_mask: Tensor::from_data(TensorData::new(legal, [n, dims.action_count]), device),
    }
}

/// Combined loss `L = L_wdl + L_policy` (both cross-entropies) plus the
/// two components as scalars.
pub fn batch_loss<B: Backend>(
    net: &PolicyValueNet<B>,
    batch: &Batch<B>,
) -> (Tensor<B, 1>, f32, f32) {
    let (wdl_logits, action_logits) =
        net.forward(batch.feature_ids.clone(), batch.feature_mask.clone());
    let wdl_logp = log_softmax(wdl_logits, 1);
    let wdl_loss = -(wdl_logp * batch.wdl_target.clone()).sum_dim(1).mean();
    let masked = action_logits + (batch.legal_mask.clone() - 1.0) * 1e9;
    let policy_logp = log_softmax(masked, 1);
    let policy_loss = -(policy_logp * batch.policy_target.clone())
        .sum_dim(1)
        .mean();
    let wdl_value: f32 = wdl_loss.clone().into_scalar().elem();
    let policy_value: f32 = policy_loss.clone().into_scalar().elem();
    (wdl_loss + policy_loss, wdl_value, policy_value)
}

/// One line of `metrics.jsonl` during training.
#[derive(Serialize)]
pub struct TrainStepMetrics {
    pub step: u64,
    pub wdl_loss: f32,
    pub policy_loss: f32,
    pub examples_seen: u64,
    pub wall_seconds: f64,
}

/// Deterministic supervised training. Returns the trained network (on the
/// autodiff backend; call `.valid()` for inference).
///
/// `on_step` observes each optimizer step's losses (for logging and
/// periodic validation).
pub fn train_supervised(
    dims: ModelDims,
    train: &[TrainRow],
    max_features: usize,
    seed: u64,
    training_steps: u64,
    on_step: impl FnMut(&TrainStepMetrics, &PolicyValueNet<TrainBackend>),
) -> PolicyValueNet<TrainBackend> {
    let device = Default::default();
    TrainBackend::seed(&device, seed);
    let net = PolicyValueNet::<TrainBackend>::new(dims, &device);
    train_steps(
        net,
        dims,
        train,
        max_features,
        seed,
        training_steps,
        on_step,
    )
}

/// The core optimizer loop, warm-starting from `net` with a fresh Adam
/// state. The batch stream is keyed by `stream_seed` only.
pub fn train_steps(
    mut net: PolicyValueNet<TrainBackend>,
    dims: ModelDims,
    train: &[TrainRow],
    max_features: usize,
    stream_seed: u64,
    training_steps: u64,
    mut on_step: impl FnMut(&TrainStepMetrics, &PolicyValueNet<TrainBackend>),
) -> PolicyValueNet<TrainBackend> {
    let device = Default::default();
    let mut optim = AdamConfig::new().init::<TrainBackend, PolicyValueNet<TrainBackend>>();
    let mut rng = ChaCha12Rng::seed_from_u64(splitmix64(stream_seed) ^ 0x7261_696e);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut cursor = train.len(); // force initial shuffle
    let started = std::time::Instant::now();

    for step in 1..=training_steps {
        let mut chosen: Vec<&TrainRow> = Vec::with_capacity(BATCH_SIZE);
        while chosen.len() < BATCH_SIZE {
            if cursor >= order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            chosen.push(&train[order[cursor]]);
            cursor += 1;
        }
        let batch = make_batch::<TrainBackend>(&chosen, dims, max_features, &device);
        let (loss, wdl_loss, policy_loss) = batch_loss(&net, &batch);
        let grads = GradientsParams::from_grads(loss.backward(), &net);
        net = optim.step(LEARNING_RATE, net, grads);
        on_step(
            &TrainStepMetrics {
                step,
                wdl_loss,
                policy_loss,
                examples_seen: step * BATCH_SIZE as u64,
                wall_seconds: started.elapsed().as_secs_f64(),
            },
            &net,
        );
    }
    net
}

/// Transposition-table size for self-play and probe searches (2^16
/// entries per game; tables are per-game so results are independent of
/// scheduling).
pub const SELFPLAY_TT_LOG2: u32 = 16;

/// One recorded self-play position (plan §12.2), written per generation.
#[derive(Clone, Debug, Serialize)]
pub struct SelfPlayRecord {
    pub generation: u32,
    pub game_index: u64,
    pub ply: u32,
    pub features: Vec<u32>,
    pub legal: Vec<u32>,
    /// Search-selected expert action (the policy target).
    pub expert_action: u32,
    /// Action actually played (differs when exploring).
    pub played_action: u32,
    pub exploratory: bool,
    /// Final game outcome from this mover's perspective (the value target).
    pub outcome_wdl: Wdl,
    pub search_nodes: u64,
    pub completed_depth: u32,
}

impl SelfPlayRecord {
    pub fn to_row(&self) -> TrainRow {
        TrainRow {
            features: self.features.clone(),
            legal: self.legal.clone(),
            wdl: self.outcome_wdl,
            policy_actions: vec![self.expert_action],
        }
    }
}

/// Aggregate statistics of one self-play generation.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SelfPlayStats {
    pub games: u64,
    pub positions: u64,
    pub exploratory_moves: u64,
    pub total_search_nodes: u64,
    pub total_plies: u64,
    pub p1_wins: u64,
    pub draws: u64,
    pub p2_wins: u64,
}

fn selfplay_rng(run_seed: u64, generation: u32, game_index: u64) -> ChaCha12Rng {
    let mixed =
        splitmix64(splitmix64(run_seed) ^ splitmix64((u64::from(generation) << 40) | game_index));
    ChaCha12Rng::seed_from_u64(mixed)
}

/// Play `games` self-play games with the frozen champion (plan §12.1/12.4):
/// the expert search plays its best move except with probability
/// `epsilon`, when a legal move is sampled from the apprentice policy;
/// the recorded label is always the expert move. Deterministic per
/// `(run_seed, generation, game_index)` regardless of thread count.
#[allow(clippy::too_many_arguments)] // §12.1 loop parameters are irreducible
pub fn generate_selfplay<G: Game>(
    game: &G,
    net: &crate::model::CompiledNet,
    games: u64,
    node_budget: u64,
    epsilon: f64,
    run_seed: u64,
    generation: u32,
    threads: usize,
) -> (Vec<SelfPlayRecord>, SelfPlayStats) {
    use crate::model::ModelEvaluator;
    use crate::search::{MoveOrdering, Searcher};
    use rand::Rng as _;

    let max_depth = 512;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    let per_game: Vec<(Vec<SelfPlayRecord>, SelfPlayStats)> = pool.install(|| {
        use rayon::prelude::*;
        (0..games)
            .into_par_iter()
            .map(|game_index| {
                let mut rng = selfplay_rng(run_seed, generation, game_index);
                let mut evaluator = ModelEvaluator::new(net);
                let mut searcher: Searcher<G> =
                    Searcher::new(Some(SELFPLAY_TT_LOG2), MoveOrdering::Natural);
                let mut state = game.initial_state();
                let mut records = Vec::new();
                let mut stats = SelfPlayStats {
                    games: 1,
                    ..SelfPlayStats::default()
                };
                let mut moves = Vec::new();
                let mut ply = 0u32;
                let outcome = loop {
                    if let Some(outcome) = game.outcome(&state) {
                        break outcome;
                    }
                    let result =
                        searcher.search(game, &mut state, max_depth, node_budget, &mut evaluator);
                    let expert = result
                        .best_move
                        .expect("non-terminal search returns a move");
                    game.legal_moves(&state, &mut moves);
                    let exploratory = rng.gen::<f64>() < epsilon;
                    let played = if exploratory {
                        sample_policy_move(game, &state, &moves, &mut evaluator, &mut rng)
                    } else {
                        expert
                    };
                    let mut features = Vec::new();
                    game.encode_features(&state, &mut features);
                    records.push(SelfPlayRecord {
                        generation,
                        game_index,
                        ply,
                        features,
                        legal: moves.iter().map(|&m| game.action_id(&state, m)).collect(),
                        expert_action: game.action_id(&state, expert),
                        played_action: game.action_id(&state, played),
                        exploratory,
                        outcome_wdl: Wdl::Draw, // back-filled below
                        search_nodes: result.nodes,
                        completed_depth: result.completed_depth,
                    });
                    stats.positions += 1;
                    stats.exploratory_moves += u64::from(exploratory);
                    stats.total_search_nodes += result.nodes;
                    game.make_move(&mut state, played);
                    ply += 1;
                };
                stats.total_plies = u64::from(ply);
                match outcome {
                    crate::game::Outcome::Win(crate::game::Player::One) => stats.p1_wins = 1,
                    crate::game::Outcome::Win(crate::game::Player::Two) => stats.p2_wins = 1,
                    crate::game::Outcome::Draw => stats.draws = 1,
                }
                // Value target: final outcome from each mover's perspective
                // (§12.3). The mover at ply p is Player One iff p is even
                // only if the game alternates strictly - derive instead
                // from parity of plies remaining.
                for record in &mut records {
                    record.outcome_wdl = match outcome {
                        crate::game::Outcome::Draw => Wdl::Draw,
                        crate::game::Outcome::Win(winner) => {
                            let mover_is_p1 = record.ply % 2 == 0;
                            let winner_is_p1 = winner == crate::game::Player::One;
                            if mover_is_p1 == winner_is_p1 {
                                Wdl::Win
                            } else {
                                Wdl::Loss
                            }
                        }
                    };
                }
                (records, stats)
            })
            .collect()
    });

    let mut records = Vec::new();
    let mut stats = SelfPlayStats::default();
    for (game_records, game_stats) in per_game {
        records.extend(game_records);
        stats.games += game_stats.games;
        stats.positions += game_stats.positions;
        stats.exploratory_moves += game_stats.exploratory_moves;
        stats.total_search_nodes += game_stats.total_search_nodes;
        stats.total_plies += game_stats.total_plies;
        stats.p1_wins += game_stats.p1_wins;
        stats.draws += game_stats.draws;
        stats.p2_wins += game_stats.p2_wins;
    }
    (records, stats)
}

/// Sample a legal move from the apprentice policy (masked softmax over
/// the champion's action logits).
fn sample_policy_move<G: Game>(
    game: &G,
    state: &G::State,
    moves: &[G::Move],
    evaluator: &mut crate::model::ModelEvaluator<'_>,
    rng: &mut ChaCha12Rng,
) -> G::Move {
    use rand::Rng as _;
    let logits = evaluator.action_logits(game, state);
    let scores: Vec<f64> = moves
        .iter()
        .map(|&m| f64::from(logits[game.action_id(state, m) as usize]))
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut draw = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        draw -= w;
        if draw <= 0.0 {
            return moves[i];
        }
    }
    moves[moves.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::connect_k::ConnectK;

    #[test]
    fn dataset_split_is_disjoint_and_complete() {
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let ds = build_exact_dataset(&game);
        let total = ds.train.len() + ds.val.len() + ds.test.len();
        assert_eq!(total, 505); // Phase 1 measured state count
        assert!(ds.max_features <= 9);
        // Every example's optimal set is non-empty and within legal.
        for e in ds.train.iter().chain(&ds.val).chain(&ds.test) {
            let optimal = e.optimal_indices();
            assert!(!optimal.is_empty());
            assert_eq!(e.legal.len(), e.child_wdl.len());
        }
        // Splits are roughly 80/10/10.
        assert!(ds.train.len() > 350 && ds.train.len() < 450);
    }

    #[test]
    fn memorization_of_a_tiny_batch() {
        let _guard = crate::model::BACKEND_RNG_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let ds = build_exact_dataset(&game);
        let tiny: Vec<TrainRow> = ds.train.iter().take(32).map(TrainRow::from).collect();
        let dims = ModelDims {
            feature_count: game.feature_count(),
            action_count: game.action_count(),
            width: 32,
        };
        let mut final_losses = (f32::NAN, f32::NAN);
        train_supervised(dims, &tiny, ds.max_features, 1, 300, |m, _| {
            final_losses = (m.wdl_loss, m.policy_loss);
        });
        assert!(
            final_losses.0 < 0.05,
            "tiny batch WDL loss should be memorized, got {}",
            final_losses.0
        );
    }

    #[test]
    fn selfplay_generation_is_deterministic_and_well_formed() {
        let _guard = crate::model::BACKEND_RNG_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let dims = crate::model::ModelDims {
            feature_count: game.feature_count(),
            action_count: game.action_count(),
            width: 16,
        };
        let device = Default::default();
        use burn::prelude::Backend as _;
        crate::model::InferBackend::seed(&device, 5);
        let net = crate::model::PolicyValueNet::<crate::model::InferBackend>::new(dims, &device);
        let compiled = crate::model::CompiledNet::from_net(&net, dims);
        let run = |threads: usize| {
            let (records, stats) = generate_selfplay(&game, &compiled, 6, 100, 0.2, 42, 0, threads);
            (
                records
                    .iter()
                    .map(|r| {
                        (
                            r.game_index,
                            r.ply,
                            r.expert_action,
                            r.played_action,
                            r.outcome_wdl,
                        )
                    })
                    .collect::<Vec<_>>(),
                stats.positions,
            )
        };
        let (a, positions) = run(1);
        let (b, _) = run(4);
        assert_eq!(a, b, "self-play must not depend on thread count");
        assert!(positions > 0);
        // Expert and played actions are always legal; the label is the
        // expert action even on exploratory moves.
        let (records, _) = generate_selfplay(&game, &compiled, 6, 100, 1.0, 42, 0, 2);
        for r in &records {
            assert!(r.legal.contains(&r.expert_action));
            assert!(r.legal.contains(&r.played_action));
            assert!(r.exploratory);
            assert_eq!(r.to_row().policy_actions, vec![r.expert_action]);
        }
    }

    #[test]
    fn training_is_deterministic() {
        let _guard = crate::model::BACKEND_RNG_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let game = ConnectK::new(3, 3, 3, true).unwrap();
        let ds = build_exact_dataset(&game);
        let dims = ModelDims {
            feature_count: game.feature_count(),
            action_count: game.action_count(),
            width: 16,
        };
        let run = || {
            let mut losses = Vec::new();
            let rows: Vec<TrainRow> = ds.train.iter().map(TrainRow::from).collect();
            train_supervised(dims, &rows, ds.max_features, 9, 30, |m, _| {
                losses.push((m.wdl_loss, m.policy_loss));
            });
            losses
        };
        assert_eq!(
            run(),
            run(),
            "identical seeds must produce identical loss traces"
        );
    }
}

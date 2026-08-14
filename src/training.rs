//! Supervised training on exact corpora (Phase 2) and the fixed training
//! recipe.
//!
//! Recipe constants below are part of the versioned recipe (plan §8):
//! changing one requires a named experiment and a DECISIONS.md entry.

use crate::game::Game;
use crate::model::{ModelDims, PolicyValueNet, TrainBackend};
use crate::search::{enumerate_solved, ExactSolver, Wdl};
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

/// Exact-state dataset stratified by position hash: 80% train, 10%
/// validation, 10% test. States are unique, so splits cannot leak.
pub struct ExactDataset {
    pub train: Vec<Example>,
    pub val: Vec<Example>,
    pub test: Vec<Example>,
    pub max_features: usize,
}

fn splitmix64(mut x: u64) -> u64 {
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
    examples: &[&Example],
    dims: ModelDims,
    max_features: usize,
    device: &B::Device,
) -> Batch<B> {
    let n = examples.len();
    let pad = dims.feature_count as i64;
    let mut ids = vec![pad; n * max_features];
    let mut mask = vec![0.0f32; n * max_features];
    let mut wdl = vec![0.0f32; n * 3];
    let mut policy = vec![0.0f32; n * dims.action_count];
    let mut legal = vec![0.0f32; n * dims.action_count];
    for (row, example) in examples.iter().enumerate() {
        for (i, &f) in example.features.iter().enumerate() {
            ids[row * max_features + i] = i64::from(f);
            mask[row * max_features + i] = 1.0;
        }
        wdl[row * 3 + example.wdl as usize] = 1.0;
        for &a in &example.legal {
            legal[row * dims.action_count + a as usize] = 1.0;
        }
        let optimal = example.optimal_indices();
        let share = 1.0 / optimal.len() as f32;
        for i in optimal {
            policy[row * dims.action_count + example.legal[i] as usize] = share;
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
    train: &[Example],
    max_features: usize,
    seed: u64,
    training_steps: u64,
    mut on_step: impl FnMut(&TrainStepMetrics, &PolicyValueNet<TrainBackend>),
) -> PolicyValueNet<TrainBackend> {
    let device = Default::default();
    TrainBackend::seed(&device, seed);
    let mut net = PolicyValueNet::<TrainBackend>::new(dims, &device);
    let mut optim = AdamConfig::new().init::<TrainBackend, PolicyValueNet<TrainBackend>>();
    let mut rng = ChaCha12Rng::seed_from_u64(splitmix64(seed) ^ 0x7261_696e);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut cursor = train.len(); // force initial shuffle
    let started = std::time::Instant::now();

    for step in 1..=training_steps {
        let mut chosen: Vec<&Example> = Vec::with_capacity(BATCH_SIZE);
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
        let tiny: Vec<Example> = ds.train.iter().take(32).cloned().collect();
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
            train_supervised(dims, &ds.train, ds.max_features, 9, 30, |m, _| {
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

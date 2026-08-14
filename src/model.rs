//! The compact sparse policy/value network (plan §10).
//!
//! Architecture (single capacity knob `model_width` = W):
//! - embedding table: `feature_count x W`; active sparse features are
//!   summed and scaled by `1/sqrt(n)`;
//! - two fixed ReLU layers `W -> W -> W`;
//! - WDL head: `W -> 3` logits (loss/draw/win order matches [`Wdl`]);
//! - policy head: per-action embedding `A_a` and bias `c_a`; the logit of
//!   a legal action is `A_a . h2 + c_a`. Only legal actions are scored.
//!
//! Inputs are raw rule-state facts encoded from the side to move's
//! perspective; no handcrafted concepts.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::relu;
use burn::tensor::{Int, Tensor};

/// CPU training backend (pure-Rust NdArray with autodiff).
pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray>;
/// CPU inference backend.
pub type InferBackend = burn::backend::NdArray;

#[derive(Module, Debug)]
pub struct PolicyValueNet<B: Backend> {
    features: Embedding<B>,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    wdl: Linear<B>,
    actions: Linear<B>,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelDims {
    pub feature_count: usize,
    pub action_count: usize,
    pub width: usize,
}

impl<B: Backend> PolicyValueNet<B> {
    pub fn new(dims: ModelDims, device: &B::Device) -> Self {
        PolicyValueNet {
            features: EmbeddingConfig::new(dims.feature_count + 1, dims.width).init(device),
            hidden1: LinearConfig::new(dims.width, dims.width).init(device),
            hidden2: LinearConfig::new(dims.width, dims.width).init(device),
            wdl: LinearConfig::new(dims.width, 3).init(device),
            actions: LinearConfig::new(dims.width, dims.action_count).init(device),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.num_params()
    }

    /// Forward pass.
    ///
    /// `feature_ids`: `[batch, max_features]`, padded with the reserved
    /// padding index (`feature_count`, the last embedding row).
    /// `feature_mask`: `[batch, max_features]`, 1.0 for real features.
    ///
    /// Returns `(wdl_logits [batch, 3], action_logits [batch, action_count])`.
    /// Action logits are raw; the caller masks illegal actions.
    pub fn forward(
        &self,
        feature_ids: Tensor<B, 2, Int>,
        feature_mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let embedded = self.features.forward(feature_ids); // [b, f, w]
        let masked = embedded * feature_mask.clone().unsqueeze_dim::<3>(2);
        let counts = feature_mask.sum_dim(1); // [b, 1]
        let scale = counts.clamp_min(1.0).sqrt().recip();
        let x = masked.sum_dim(1).squeeze_dims::<2>(&[1]) * scale; // [b, w]
        let h1 = relu(self.hidden1.forward(x));
        let h2 = relu(self.hidden2.forward(h1));
        let wdl = self.wdl.forward(h2.clone());
        let actions = self.actions.forward(h2);
        (wdl, actions)
    }
}

/// Weights extracted to plain arrays for fast batch-1 inference during
/// search and self-play (plan §10.6: framework inference profiled as the
/// bottleneck; this path is validated against the framework to
/// near-exact tolerance by a mandatory test).
///
/// Layout: all matrices row-major `[out][in]` for cache-friendly dot
/// products. `Send + Sync`; share one instance across worker threads.
pub struct CompiledNet {
    pub dims: ModelDims,
    embed: Vec<f32>, // (feature_count + 1) x W
    w1: Vec<f32>,    // W x W
    b1: Vec<f32>,
    w2: Vec<f32>, // W x W
    b2: Vec<f32>,
    wv: Vec<f32>, // 3 x W
    bv: Vec<f32>,
    wa: Vec<f32>, // action_count x W
    ba: Vec<f32>,
}

/// Extract `[in, out]`-shaped Burn Linear params as row-major `[out][in]`.
fn linear_parts(layer: &Linear<InferBackend>) -> (Vec<f32>, Vec<f32>) {
    let data = layer.weight.val().to_data();
    let [d_in, d_out] = [data.shape[0], data.shape[1]];
    let flat = data.to_vec::<f32>().unwrap();
    let mut transposed = vec![0.0f32; flat.len()];
    for i in 0..d_in {
        for o in 0..d_out {
            transposed[o * d_in + i] = flat[i * d_out + o];
        }
    }
    let bias = layer
        .bias
        .as_ref()
        .expect("linear layers use biases")
        .val()
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    (transposed, bias)
}

fn matvec(weights: &[f32], bias: &[f32], input: &[f32], out: &mut [f32]) {
    let n = input.len();
    for (o, out_slot) in out.iter_mut().enumerate() {
        let row = &weights[o * n..(o + 1) * n];
        // Eight independent accumulators so the loop vectorizes instead
        // of serializing on one add chain.
        let mut acc = [0.0f32; 8];
        let mut row_chunks = row.chunks_exact(8);
        let mut in_chunks = input.chunks_exact(8);
        for (rc, ic) in (&mut row_chunks).zip(&mut in_chunks) {
            for lane in 0..8 {
                acc[lane] += rc[lane] * ic[lane];
            }
        }
        let mut sum = bias[o] + acc.iter().sum::<f32>();
        for (w, x) in row_chunks.remainder().iter().zip(in_chunks.remainder()) {
            sum += w * x;
        }
        *out_slot = sum;
    }
}

impl CompiledNet {
    pub fn from_net(net: &PolicyValueNet<InferBackend>, dims: ModelDims) -> CompiledNet {
        let (w1, b1) = linear_parts(&net.hidden1);
        let (w2, b2) = linear_parts(&net.hidden2);
        let (wv, bv) = linear_parts(&net.wdl);
        let (wa, ba) = linear_parts(&net.actions);
        CompiledNet {
            dims,
            embed: net.features.weight.val().to_data().to_vec::<f32>().unwrap(),
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            wa,
            ba,
        }
    }

    /// Forward one state's sparse features. Writes raw WDL logits
    /// (`[loss, draw, win]`) and raw action logits (indexed by action ID).
    pub fn forward(
        &self,
        features: &[u32],
        wdl_logits: &mut [f32; 3],
        action_logits: &mut Vec<f32>,
    ) {
        let w = self.dims.width;
        let mut x = vec![0.0f32; w];
        for &f in features {
            let row = &self.embed[f as usize * w..(f as usize + 1) * w];
            for (slot, v) in x.iter_mut().zip(row) {
                *slot += v;
            }
        }
        let scale = 1.0 / (features.len().max(1) as f32).sqrt();
        for slot in &mut x {
            *slot *= scale;
        }
        let mut h1 = vec![0.0f32; w];
        matvec(&self.w1, &self.b1, &x, &mut h1);
        for v in &mut h1 {
            *v = v.max(0.0);
        }
        let mut h2 = vec![0.0f32; w];
        matvec(&self.w2, &self.b2, &h1, &mut h2);
        for v in &mut h2 {
            *v = v.max(0.0);
        }
        matvec(&self.wv, &self.bv, &h2, wdl_logits);
        action_logits.resize(self.dims.action_count, 0.0);
        matvec(&self.wa, &self.ba, &h2, action_logits);
    }
}

/// The model as a search evaluator (plan §11.2): leaf values from the
/// WDL expectation, move ordering from the policy head. Uses the
/// compiled inference path.
pub struct ModelEvaluator<'a> {
    net: &'a CompiledNet,
    features: Vec<u32>,
    /// Cached forward output for the last state (keyed by position hash),
    /// since `leaf_value` and `policy_scores` may hit the same state.
    cache_key: Option<u64>,
    cache_wdl: [f32; 3],
    cache_logits: Vec<f32>,
}

impl<'a> ModelEvaluator<'a> {
    pub fn new(net: &'a CompiledNet) -> ModelEvaluator<'a> {
        ModelEvaluator {
            net,
            features: Vec::new(),
            cache_key: None,
            cache_wdl: [0.0; 3],
            cache_logits: Vec::new(),
        }
    }

    fn forward_state<G: crate::game::Game>(&mut self, game: &G, state: &G::State) {
        let key = game.position_key(state);
        if self.cache_key == Some(key) {
            return;
        }
        game.encode_features(state, &mut self.features);
        let mut wdl_logits = [0.0f32; 3];
        let mut logits = std::mem::take(&mut self.cache_logits);
        self.net
            .forward(&self.features, &mut wdl_logits, &mut logits);
        // Stable softmax over the three WDL logits.
        let max = wdl_logits[0].max(wdl_logits[1]).max(wdl_logits[2]);
        let exp = [
            (wdl_logits[0] - max).exp(),
            (wdl_logits[1] - max).exp(),
            (wdl_logits[2] - max).exp(),
        ];
        let sum = exp[0] + exp[1] + exp[2];
        self.cache_wdl = [exp[0] / sum, exp[1] / sum, exp[2] / sum];
        self.cache_logits = logits;
        self.cache_key = Some(key);
    }

    /// WDL probabilities `[loss, draw, win]` for `state` (side to move).
    pub fn wdl_probs<G: crate::game::Game>(&mut self, game: &G, state: &G::State) -> [f32; 3] {
        self.forward_state(game, state);
        self.cache_wdl
    }

    /// Raw action logits indexed by action ID.
    pub fn action_logits<G: crate::game::Game>(&mut self, game: &G, state: &G::State) -> &[f32] {
        self.forward_state(game, state);
        &self.cache_logits
    }
}

impl<G: crate::game::Game> crate::search::Evaluator<G> for ModelEvaluator<'_> {
    /// Leaf value = `1000 * (P(win) - P(loss))`, the WDL expectation
    /// scaled to the search's evaluation range.
    fn leaf_value(&mut self, game: &G, state: &G::State) -> i32 {
        let [loss, _draw, win] = self.wdl_probs(game, state);
        let value = (f64::from(win) - f64::from(loss)) * f64::from(crate::search::SCORE_EVAL_MAX);
        (value.round() as i32).clamp(
            -crate::search::SCORE_EVAL_MAX,
            crate::search::SCORE_EVAL_MAX,
        )
    }

    fn policy_scores(
        &mut self,
        game: &G,
        state: &G::State,
        moves: &[G::Move],
        out: &mut Vec<f32>,
    ) -> bool {
        self.forward_state(game, state);
        out.clear();
        for &mv in moves {
            out.push(self.cache_logits[game.action_id(state, mv) as usize]);
        }
        true
    }
}

/// Burn's backend RNG is process-global: tests that seed it or draw from
/// it (model init) must hold this lock so concurrent tests stay
/// deterministic. Production is unaffected (one process, one training run).
#[cfg(test)]
pub(crate) static BACKEND_RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
fn rng_guard() -> std::sync::MutexGuard<'static, ()> {
    BACKEND_RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    use burn::tensor::TensorData;

    fn dims() -> ModelDims {
        ModelDims {
            feature_count: 32,
            action_count: 7,
            width: 16,
        }
    }

    fn batch<B: Backend>(device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        // Two examples: 3 features and 2 features (padded with index 32).
        let ids =
            Tensor::<B, 2, Int>::from_data(TensorData::from([[0i64, 5, 9], [3, 12, 32]]), device);
        let mask = Tensor::<B, 2>::from_data(
            TensorData::from([[1.0f32, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            device,
        );
        (ids, mask)
    }

    #[test]
    fn forward_shapes_and_determinism() {
        let _guard = super::rng_guard();
        let device = Default::default();
        InferBackend::seed(&device, 7);
        let net = PolicyValueNet::<InferBackend>::new(dims(), &device);
        let (ids, mask) = batch::<InferBackend>(&device);
        let (wdl, actions) = net.forward(ids.clone(), mask.clone());
        assert_eq!(wdl.dims(), [2, 3]);
        assert_eq!(actions.dims(), [2, 7]);
        let (wdl2, _) = net.forward(ids, mask);
        assert_eq!(
            wdl.to_data().to_vec::<f32>().unwrap(),
            wdl2.to_data().to_vec::<f32>().unwrap()
        );
        assert!(net.parameter_count() > 0);
    }

    #[test]
    fn adam_step_reduces_simple_loss() {
        let _guard = super::rng_guard();
        let device = Default::default();
        TrainBackend::seed(&device, 3);
        let mut net = PolicyValueNet::<TrainBackend>::new(dims(), &device);
        let mut optim = AdamConfig::new().init::<TrainBackend, PolicyValueNet<TrainBackend>>();
        let (ids, mask) = batch::<TrainBackend>(&device);
        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..50 {
            let (wdl, _) = net.forward(ids.clone(), mask.clone());
            // Drive the first WDL logit of each example toward zero.
            let loss = (wdl.clone() * wdl).mean();
            let value = loss.clone().into_scalar();
            if step == 0 {
                first = value;
            }
            last = value;
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = optim.step(1e-2, net, grads);
        }
        assert!(
            last < first * 0.5,
            "Adam failed to reduce loss: {first} -> {last}"
        );
    }

    #[test]
    fn compiled_inference_matches_framework_and_is_fast() {
        use crate::game::Game as _;
        use crate::games::connect_k::ConnectK;
        use crate::search::Evaluator as _;
        let _guard = super::rng_guard();
        let device = Default::default();
        InferBackend::seed(&device, 21);
        let game = ConnectK::new(4, 4, 4, true).unwrap();
        let dims = ModelDims {
            feature_count: game.feature_count(),
            action_count: game.action_count(),
            width: 128,
        };
        let net = PolicyValueNet::<InferBackend>::new(dims, &device);
        let compiled = CompiledNet::from_net(&net, dims);

        // Mandated validation (plan §10.6): compiled path must match the
        // framework to near-exact tolerance on many states.
        let mut rng_state = 0x1234_5678u64;
        let mut states_checked = 0;
        let mut stack = vec![game.initial_state()];
        while let Some(state) = stack.pop() {
            if states_checked >= 60 || game.outcome(&state).is_some() {
                if stack.is_empty() {
                    break;
                }
                continue;
            }
            states_checked += 1;
            let mut features = Vec::new();
            game.encode_features(&state, &mut features);
            // Framework forward.
            let n = features.len();
            let ids: Vec<i64> = features.iter().map(|&f| i64::from(f)).collect();
            let id_tensor = Tensor::<InferBackend, 2, Int>::from_data(
                burn::tensor::TensorData::new(ids, [1, n]),
                &device,
            );
            let mask = Tensor::<InferBackend, 2>::ones([1, n], &device);
            let (wdl_t, act_t) = net.forward(id_tensor, mask);
            let wdl_ref = wdl_t.to_data().to_vec::<f32>().unwrap();
            let act_ref = act_t.to_data().to_vec::<f32>().unwrap();
            // Compiled forward.
            let mut wdl = [0.0f32; 3];
            let mut act = Vec::new();
            compiled.forward(&features, &mut wdl, &mut act);
            for (a, b) in wdl.iter().zip(&wdl_ref) {
                assert!((a - b).abs() < 1e-4, "wdl mismatch: {a} vs {b}");
            }
            for (a, b) in act.iter().zip(&act_ref) {
                assert!((a - b).abs() < 1e-4, "action mismatch: {a} vs {b}");
            }
            // Descend to a pseudo-random child to diversify states.
            let mut moves = Vec::new();
            game.legal_moves(&state, &mut moves);
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mv = moves[(rng_state >> 33) as usize % moves.len()];
            let mut child = state.clone();
            game.make_move(&mut child, mv);
            stack.push(child);
            if stack.len() < 3 {
                stack.push(game.initial_state());
            }
        }
        assert!(states_checked >= 30, "too few states validated");

        // Evaluator interface sanity + latency.
        let mut eval = ModelEvaluator::new(&compiled);
        let state = game.initial_state();
        let v1 = eval.leaf_value(&game, &state);
        assert_eq!(v1, eval.leaf_value(&game, &state));
        assert!(v1.abs() <= crate::search::SCORE_EVAL_MAX);
        let mut moves = Vec::new();
        game.legal_moves(&state, &mut moves);
        let mut scores = Vec::new();
        assert!(eval.policy_scores(&game, &state, &moves, &mut scores));
        assert_eq!(scores.len(), moves.len());

        let mut features = Vec::new();
        game.encode_features(&state, &mut features);
        let mut wdl = [0.0f32; 3];
        let mut act = Vec::new();
        let started = std::time::Instant::now();
        for _ in 0..2000 {
            compiled.forward(&features, &mut wdl, &mut act);
        }
        let per_eval = started.elapsed().as_secs_f64() / 2000.0;
        println!("compiled batch-1 forward latency: {:.2} us", per_eval * 1e6);
    }

    #[test]
    fn save_load_reproduces_outputs() {
        let _guard = super::rng_guard();
        let device = Default::default();
        InferBackend::seed(&device, 11);
        let net = PolicyValueNet::<InferBackend>::new(dims(), &device);
        let (ids, mask) = batch::<InferBackend>(&device);
        let (wdl_before, act_before) = net.forward(ids.clone(), mask.clone());

        let dir = std::env::temp_dir().join(format!("burn-model-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model");
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        net.clone().save_file(&path, &recorder).unwrap();

        let restored = PolicyValueNet::<InferBackend>::new(dims(), &device)
            .load_file(&path, &recorder, &device)
            .unwrap();
        let (wdl_after, act_after) = restored.forward(ids, mask);
        assert_eq!(
            wdl_before.to_data().to_vec::<f32>().unwrap(),
            wdl_after.to_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            act_before.to_data().to_vec::<f32>().unwrap(),
            act_after.to_data().to_vec::<f32>().unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

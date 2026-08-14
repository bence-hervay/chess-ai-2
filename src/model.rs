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

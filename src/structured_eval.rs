//! Compact structured evaluation models (SHSD §17, §46.1).
//!
//! Level 1 of the evaluation-model ladder: a multinomial logistic WDL
//! model over structured measurements, fitted by deterministic
//! minibatch Adam with L2 regularization — a direct Rust
//! implementation, no tensor framework (§11.4). The model plugs into
//! the production search through `search::Evaluator` using the same
//! leaf-value convention as the raw MLP (D019:
//! `round(1000 · (P(win) − P(loss)))`).

use crate::features::{FeatureEntry, FeatureExtractor};
use crate::game::Game;
use crate::search::{Evaluator, Wdl, SCORE_EVAL_MAX};
use crate::training::splitmix64;
use rand::seq::SliceRandom;
use rand::SeedableRng as _;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

/// Linear three-class WDL model. Class order follows the `Wdl`
/// discriminants: 0 = Loss, 1 = Draw, 2 = Win (side to move).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearWdl {
    /// Named feature recipe this model was fitted for (§12.3).
    pub recipe: String,
    pub dimension: usize,
    /// `weights[feature][class]`.
    pub weights: Vec<[f32; 3]>,
    pub bias: [f32; 3],
}

impl LinearWdl {
    pub fn zeros(recipe: &str, dimension: usize) -> LinearWdl {
        LinearWdl {
            recipe: recipe.to_string(),
            dimension,
            weights: vec![[0.0; 3]; dimension],
            bias: [0.0; 3],
        }
    }

    pub fn parameter_count(&self) -> usize {
        3 * self.dimension + 3
    }

    pub fn logits(&self, x: &[FeatureEntry]) -> [f64; 3] {
        let mut z = [
            f64::from(self.bias[0]),
            f64::from(self.bias[1]),
            f64::from(self.bias[2]),
        ];
        for &(index, value) in x {
            let w = &self.weights[index as usize];
            let v = f64::from(value);
            z[0] += f64::from(w[0]) * v;
            z[1] += f64::from(w[1]) * v;
            z[2] += f64::from(w[2]) * v;
        }
        z
    }

    /// `[P(loss), P(draw), P(win)]` for the side to move.
    pub fn probs(&self, x: &[FeatureEntry]) -> [f64; 3] {
        softmax(self.logits(x))
    }

    /// Search leaf value (D019 convention), clamped to the eval range.
    pub fn leaf_value(&self, x: &[FeatureEntry]) -> i32 {
        let p = self.probs(x);
        let value = (1000.0 * (p[2] - p[0])).round() as i32;
        value.clamp(-SCORE_EVAL_MAX, SCORE_EVAL_MAX)
    }
}

fn softmax(z: [f64; 3]) -> [f64; 3] {
    let m = z[0].max(z[1]).max(z[2]);
    let e = [(z[0] - m).exp(), (z[1] - m).exp(), (z[2] - m).exp()];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

/// One training example: sparse structured features and the exact or
/// teacher WDL for the side to move.
#[derive(Clone, Debug)]
pub struct StructuredRow {
    pub x: Vec<FeatureEntry>,
    pub wdl: Wdl,
}

/// Fit hyperparameters. All four are experimentally selected values
/// (§6.3): the selection experiment is recorded in the run that chose
/// them, and they arrive here through required config fields.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FitHyper {
    pub steps: u64,
    pub batch: usize,
    pub lr: f64,
    pub l2: f64,
}

/// Metrics reported during fitting.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FitStep {
    pub step: u64,
    pub train_loss: f64,
}

/// Deterministic minibatch Adam on the multinomial logistic loss with
/// L2 on weights (not bias). The batch stream depends only on `seed`.
pub fn fit_linear_wdl(
    recipe: &str,
    dimension: usize,
    rows: &[StructuredRow],
    hyper: FitHyper,
    seed: u64,
    mut on_step: impl FnMut(&FitStep),
) -> LinearWdl {
    assert!(!rows.is_empty(), "cannot fit on an empty training set");
    let mut model = LinearWdl::zeros(recipe, dimension);
    let params = 3 * dimension + 3;
    let mut m = vec![0.0f64; params];
    let mut v = vec![0.0f64; params];
    let mut grad = vec![0.0f64; params];
    let (beta1, beta2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);

    let mut rng = ChaCha12Rng::seed_from_u64(splitmix64(seed) ^ 0x5f17_11ea);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let mut cursor = rows.len(); // force initial shuffle

    for step in 1..=hyper.steps {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let mut batch_loss = 0.0;
        for _ in 0..hyper.batch {
            if cursor >= order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            let row = &rows[order[cursor]];
            cursor += 1;
            let p = model.probs(&row.x);
            let target = row.wdl as usize;
            batch_loss -= p[target].max(1e-300).ln();
            for class in 0..3 {
                let delta = p[class] - f64::from(u8::from(class == target));
                grad[3 * dimension + class] += delta;
                for &(index, value) in &row.x {
                    grad[3 * index as usize + class] += delta * f64::from(value);
                }
            }
        }
        let scale = 1.0 / hyper.batch as f64;
        // L2 on weights only.
        for feature in 0..dimension {
            for class in 0..3 {
                let i = 3 * feature + class;
                grad[i] = grad[i] * scale + hyper.l2 * f64::from(model.weights[feature][class]);
            }
        }
        for class in 0..3 {
            let i = 3 * dimension + class;
            grad[i] *= scale;
        }
        // Adam update.
        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);
        for i in 0..params {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let update = hyper.lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
            if i < 3 * dimension {
                model.weights[i / 3][i % 3] -= update as f32;
            } else {
                model.bias[i - 3 * dimension] -= update as f32;
            }
        }
        on_step(&FitStep {
            step,
            train_loss: batch_loss * scale,
        });
    }
    model
}

/// Mean log-loss and accuracy of the model on labelled rows.
pub fn evaluate_linear_wdl(model: &LinearWdl, rows: &[StructuredRow]) -> (f64, f64) {
    if rows.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut loss = 0.0;
    let mut correct = 0u64;
    for row in rows {
        let p = model.probs(&row.x);
        loss -= p[row.wdl as usize].max(1e-300).ln();
        let argmax = (0..3)
            .max_by(|&a, &b| p[a].total_cmp(&p[b]))
            .expect("3 classes");
        correct += u64::from(argmax == row.wdl as usize);
    }
    (loss / rows.len() as f64, correct as f64 / rows.len() as f64)
}

/// Log-loss and accuracy of always predicting the training class
/// frequencies (the trivial calibration floor).
pub fn class_prior_baseline(train: &[StructuredRow], eval: &[StructuredRow]) -> (f64, f64) {
    let mut counts = [0u64; 3];
    for row in train {
        counts[row.wdl as usize] += 1;
    }
    let total: u64 = counts.iter().sum();
    let p: Vec<f64> = counts
        .iter()
        .map(|&c| (c as f64 / total as f64).max(1e-12))
        .collect();
    let argmax = (0..3)
        .max_by(|&a, &b| p[a].total_cmp(&p[b]))
        .expect("3 classes");
    let mut loss = 0.0;
    let mut correct = 0u64;
    for row in eval {
        loss -= p[row.wdl as usize].ln();
        correct += u64::from(argmax == row.wdl as usize);
    }
    (loss / eval.len() as f64, correct as f64 / eval.len() as f64)
}

/// A structured linear model plus its extractor as a search evaluator.
pub struct StructuredEvaluator<'a, G: Game, X: FeatureExtractor<G>> {
    model: &'a LinearWdl,
    extractor: X,
    buf: Vec<FeatureEntry>,
    _game: std::marker::PhantomData<G>,
}

impl<'a, G: Game, X: FeatureExtractor<G>> StructuredEvaluator<'a, G, X> {
    pub fn new(model: &'a LinearWdl, extractor: X) -> Self {
        assert_eq!(
            model.dimension,
            extractor.dimension(),
            "model and extractor dimensions must agree"
        );
        StructuredEvaluator {
            model,
            extractor,
            buf: Vec::new(),
            _game: std::marker::PhantomData,
        }
    }
}

impl<'a, G: Game, X: FeatureExtractor<G>> Evaluator<G> for StructuredEvaluator<'a, G, X> {
    fn leaf_value(&mut self, game: &G, state: &G::State) -> i32 {
        self.extractor.extract(game, state, &mut self.buf);
        self.model.leaf_value(&self.buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_rows() -> Vec<StructuredRow> {
        // Feature 0 positive => Win, negative => Loss, near zero => Draw.
        let mut rows = Vec::new();
        for i in -6i32..=6 {
            let wdl = if i > 1 {
                Wdl::Win
            } else if i < -1 {
                Wdl::Loss
            } else {
                Wdl::Draw
            };
            rows.push(StructuredRow {
                x: vec![(0, i as f32), (1, 1.0)],
                wdl,
            });
        }
        rows
    }

    const HYPER: FitHyper = FitHyper {
        steps: 800,
        batch: 13,
        lr: 0.05,
        l2: 1e-6,
    };

    #[test]
    fn gradient_matches_finite_differences() {
        // Analytic single-example gradient vs central differences on a
        // handful of parameters (§69.5 gradient check).
        let rows = toy_rows();
        let mut model = LinearWdl::zeros("toy", 2);
        model.weights[0] = [0.3, -0.2, 0.1];
        model.weights[1] = [-0.05, 0.02, 0.4];
        model.bias = [0.01, -0.03, 0.02];
        let loss = |m: &LinearWdl| -> f64 {
            rows.iter()
                .map(|r| -m.probs(&r.x)[r.wdl as usize].ln())
                .sum::<f64>()
                / rows.len() as f64
        };
        // Analytic full-batch gradient.
        let mut grad_w = [[0.0f64; 3]; 2];
        let mut grad_b = [0.0f64; 3];
        for r in &rows {
            let p = model.probs(&r.x);
            for class in 0..3 {
                let delta = p[class] - f64::from(u8::from(class == r.wdl as usize));
                grad_b[class] += delta;
                for &(i, v) in &r.x {
                    grad_w[i as usize][class] += delta * f64::from(v);
                }
            }
        }
        let n = rows.len() as f64;
        let h = 1e-4;
        #[allow(clippy::needless_range_loop)] // parameter indices, not iteration
        for feature in 0..2 {
            for class in 0..3 {
                let mut plus = model.clone();
                plus.weights[feature][class] += h as f32;
                let mut minus = model.clone();
                minus.weights[feature][class] -= h as f32;
                let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
                let analytic = grad_w[feature][class] / n;
                assert!(
                    (numeric - analytic).abs() < 1e-2 * analytic.abs().max(1e-3),
                    "w[{feature}][{class}]: numeric {numeric} vs analytic {analytic}"
                );
            }
        }
        #[allow(clippy::needless_range_loop)] // parameter indices, not iteration
        for class in 0..3 {
            let mut plus = model.clone();
            plus.bias[class] += h as f32;
            let mut minus = model.clone();
            minus.bias[class] -= h as f32;
            let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
            let analytic = grad_b[class] / n;
            assert!((numeric - analytic).abs() < 1e-2 * analytic.abs().max(1e-3));
        }
    }

    #[test]
    fn fit_learns_the_toy_rule_and_is_deterministic() {
        let rows = toy_rows();
        let run = || {
            let mut losses = Vec::new();
            let model = fit_linear_wdl("toy", 2, &rows, HYPER, 7, |s| {
                losses.push(s.train_loss);
            });
            (model, losses)
        };
        let (model, losses) = run();
        let (model2, losses2) = run();
        assert_eq!(losses, losses2, "fitting must be deterministic");
        assert_eq!(model.weights, model2.weights);
        let (loss, acc) = evaluate_linear_wdl(&model, &rows);
        assert!(acc == 1.0, "toy rule must be learned exactly, got {acc}");
        assert!(loss < 0.35, "toy log-loss should be small, got {loss}");
        // The learned decision structure: feature 0 separates win from
        // loss (learned, not asserted a priori).
        assert!(model.weights[0][2] > model.weights[0][0]);
        // Leaf values follow the sign of the advantage and stay in range.
        let strong = model.leaf_value(&[(0, 6.0), (1, 1.0)]);
        let weak = model.leaf_value(&[(0, -6.0), (1, 1.0)]);
        assert!(strong > 0 && weak < 0 && strong <= SCORE_EVAL_MAX);
    }

    #[test]
    fn serde_round_trip() {
        let rows = toy_rows();
        let model = fit_linear_wdl("toy", 2, &rows, HYPER, 7, |_| {});
        let json = serde_json::to_string(&model).unwrap();
        let back: LinearWdl = serde_json::from_str(&json).unwrap();
        assert_eq!(model.weights, back.weights);
        assert_eq!(model.bias, back.bias);
        assert_eq!(model.recipe, back.recipe);
    }

    #[test]
    fn class_prior_is_a_floor() {
        let rows = toy_rows();
        let model = fit_linear_wdl("toy", 2, &rows, HYPER, 7, |_| {});
        let (fit_loss, _) = evaluate_linear_wdl(&model, &rows);
        let (prior_loss, _) = class_prior_baseline(&rows, &rows);
        assert!(fit_loss < prior_loss);
    }
}

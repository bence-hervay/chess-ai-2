//! Compact structured evaluation models (SHSD §17, §46.1).
//!
//! Level 1 of the evaluation-model ladder: a multinomial logistic WDL
//! model over structured measurements, fitted by deterministic
//! minibatch Adam with L2 regularization — a direct Rust
//! implementation, no tensor framework (§11.4). The model plugs into
//! the production search through `search::Evaluator` using the same
//! leaf-value convention as the raw MLP (D019:
//! `round(1000 · (P(win) − P(loss)))`).

use crate::features::{FeatureEntry, FeatureExtractor, MoveFeatures};
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

/// One soft-target training example: the target is a probability
/// distribution over WDL classes (e.g. calibrated deep-search values,
/// §18.2). Hard labels are the one-hot special case.
#[derive(Clone, Debug)]
pub struct SoftRow {
    pub x: Vec<FeatureEntry>,
    pub target: [f64; 3],
}

/// Deterministic minibatch Adam on the (soft) multinomial logistic
/// loss with L2 on weights (not bias). The batch stream depends only
/// on `seed`. Hard-label fitting wraps this with one-hot targets.
pub fn fit_linear_wdl_soft(
    recipe: &str,
    dimension: usize,
    rows: &[SoftRow],
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
            for class in 0..3 {
                if row.target[class] > 0.0 {
                    batch_loss -= row.target[class] * p[class].max(1e-300).ln();
                }
                let delta = p[class] - row.target[class];
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

/// Hard-label fitting: the one-hot special case of
/// [`fit_linear_wdl_soft`].
pub fn fit_linear_wdl(
    recipe: &str,
    dimension: usize,
    rows: &[StructuredRow],
    hyper: FitHyper,
    seed: u64,
    on_step: impl FnMut(&FitStep),
) -> LinearWdl {
    let soft: Vec<SoftRow> = rows
        .iter()
        .map(|row| {
            let mut target = [0.0; 3];
            target[row.wdl as usize] = 1.0;
            SoftRow {
                x: row.x.clone(),
                target,
            }
        })
        .collect();
    fit_linear_wdl_soft(recipe, dimension, &soft, hyper, seed, on_step)
}

/// Mean soft cross-entropy and argmax agreement on soft-target rows.
pub fn evaluate_linear_wdl_soft(model: &LinearWdl, rows: &[SoftRow]) -> (f64, f64) {
    if rows.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut loss = 0.0;
    let mut agree = 0u64;
    for row in rows {
        let p = model.probs(&row.x);
        for (class, &prob) in p.iter().enumerate() {
            if row.target[class] > 0.0 {
                loss -= row.target[class] * prob.max(1e-300).ln();
            }
        }
        let pa = (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).expect("3");
        let ta = (0..3)
            .max_by(|&a, &b| row.target[a].total_cmp(&row.target[b]))
            .expect("3");
        agree += u64::from(pa == ta);
    }
    (loss / rows.len() as f64, agree as f64 / rows.len() as f64)
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

/// Linear pairwise move-ranking model (§35.4 starting point). Scores
/// state-action features; higher = search first. Trained on
/// better/worse move pairs with the logistic ranking loss (§18.5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoveRanker {
    /// Named move-feature recipe this ranker was fitted for.
    pub recipe: String,
    pub dimension: usize,
    pub weights: Vec<f32>,
}

impl MoveRanker {
    pub fn zeros(recipe: &str, dimension: usize) -> MoveRanker {
        MoveRanker {
            recipe: recipe.to_string(),
            dimension,
            weights: vec![0.0; dimension],
        }
    }

    pub fn score(&self, x: &[FeatureEntry]) -> f64 {
        x.iter()
            .map(|&(i, v)| f64::from(self.weights[i as usize]) * f64::from(v))
            .sum()
    }
}

/// One ranking pair: `better` should score above `worse`.
#[derive(Clone, Debug)]
pub struct RankPair {
    pub better: Vec<FeatureEntry>,
    pub worse: Vec<FeatureEntry>,
}

/// Deterministic minibatch Adam on the pairwise logistic ranking loss
/// `-ln sigma(s(better) - s(worse))` with L2.
pub fn fit_move_ranker(
    recipe: &str,
    dimension: usize,
    pairs: &[RankPair],
    hyper: FitHyper,
    seed: u64,
    mut on_step: impl FnMut(&FitStep),
) -> MoveRanker {
    assert!(!pairs.is_empty(), "cannot fit on an empty pair set");
    let mut model = MoveRanker::zeros(recipe, dimension);
    let mut m = vec![0.0f64; dimension];
    let mut v = vec![0.0f64; dimension];
    let mut grad = vec![0.0f64; dimension];
    let (beta1, beta2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
    let mut rng = ChaCha12Rng::seed_from_u64(splitmix64(seed) ^ 0x4a4e_4b52);
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    let mut cursor = pairs.len();

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
            let pair = &pairs[order[cursor]];
            cursor += 1;
            let margin = model.score(&pair.better) - model.score(&pair.worse);
            let p = 1.0 / (1.0 + (-margin).exp());
            batch_loss -= p.max(1e-300).ln();
            let delta = p - 1.0; // d(loss)/d(margin)
            for &(i, x) in &pair.better {
                grad[i as usize] += delta * f64::from(x);
            }
            for &(i, x) in &pair.worse {
                grad[i as usize] -= delta * f64::from(x);
            }
        }
        let scale = 1.0 / hyper.batch as f64;
        for (i, g) in grad.iter_mut().enumerate() {
            *g = *g * scale + hyper.l2 * f64::from(model.weights[i]);
        }
        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);
        for i in 0..dimension {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            model.weights[i] -= (hyper.lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps)) as f32;
        }
        on_step(&FitStep {
            step,
            train_loss: batch_loss * scale,
        });
    }
    model
}

/// Mean pairwise loss and accuracy (better ranked above worse).
pub fn evaluate_move_ranker(model: &MoveRanker, pairs: &[RankPair]) -> (f64, f64) {
    if pairs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut loss = 0.0;
    let mut correct = 0u64;
    for pair in pairs {
        let margin = model.score(&pair.better) - model.score(&pair.worse);
        let p = 1.0 / (1.0 + (-margin).exp());
        loss -= p.max(1e-300).ln();
        correct += u64::from(margin > 0.0);
    }
    (
        loss / pairs.len() as f64,
        correct as f64 / pairs.len() as f64,
    )
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

/// Forward Chess evaluator combining the structured value model with an
/// optional learned move-ordering model (§35). The search's
/// TT-move-first rule applies on top of whatever this orders.
pub struct FcOrderedEvaluator<'a> {
    model: &'a LinearWdl,
    ranker: Option<&'a MoveRanker>,
    extractor: crate::features::forward_chess::FcExtractor,
    move_features: crate::features::forward_chess::FcMoveFeatures,
    buf: Vec<FeatureEntry>,
}

impl<'a> FcOrderedEvaluator<'a> {
    pub fn new(
        game: &crate::games::forward_chess::ForwardChess,
        model: &'a LinearWdl,
        recipe: crate::features::forward_chess::FcRecipe,
        ranker: Option<&'a MoveRanker>,
    ) -> Self {
        let extractor = crate::features::forward_chess::FcExtractor::new(game, recipe);
        assert_eq!(model.dimension, extractor.dimension());
        if let Some(ranker) = ranker {
            assert_eq!(
                ranker.dimension,
                crate::features::forward_chess::MOVE_FEATURE_DIMENSION
            );
        }
        FcOrderedEvaluator {
            model,
            ranker,
            extractor,
            move_features: crate::features::forward_chess::FcMoveFeatures::new(game),
            buf: Vec::new(),
        }
    }
}

impl Evaluator<crate::games::forward_chess::ForwardChess> for FcOrderedEvaluator<'_> {
    fn leaf_value(
        &mut self,
        game: &crate::games::forward_chess::ForwardChess,
        state: &crate::games::forward_chess::FcState,
    ) -> i32 {
        self.extractor.extract(game, state, &mut self.buf);
        self.model.leaf_value(&self.buf)
    }

    fn policy_scores(
        &mut self,
        game: &crate::games::forward_chess::ForwardChess,
        state: &crate::games::forward_chess::FcState,
        moves: &[crate::games::forward_chess::FcMove],
        out: &mut Vec<f32>,
    ) -> bool {
        let Some(ranker) = self.ranker else {
            return false;
        };
        out.clear();
        for &mv in moves {
            self.move_features.extract(game, state, mv, &mut self.buf);
            out.push(ranker.score(&self.buf) as f32);
        }
        true
    }
}

/// MLP value with learned move ordering (F2/J2 composite): the
/// raw-model baseline's WDL head at leaves, a `MoveRanker` over the
/// game's move features for ordering. Lets ordering models be
/// evaluated against the policy head with the value model held fixed.
pub struct MlpRankedEvaluator<'a, MF> {
    inner: crate::model::ModelEvaluator<'a>,
    ranker: &'a MoveRanker,
    move_features: MF,
    buf: Vec<FeatureEntry>,
}

impl<'a, MF> MlpRankedEvaluator<'a, MF> {
    pub fn new<G: Game>(
        net: &'a crate::model::CompiledNet,
        ranker: &'a MoveRanker,
        move_features: MF,
    ) -> Self
    where
        MF: crate::features::MoveFeatures<G>,
    {
        assert_eq!(ranker.dimension, move_features.dimension());
        MlpRankedEvaluator {
            inner: crate::model::ModelEvaluator::new(net),
            ranker,
            move_features,
            buf: Vec::new(),
        }
    }
}

impl<G: Game, MF: crate::features::MoveFeatures<G>> Evaluator<G> for MlpRankedEvaluator<'_, MF> {
    fn leaf_value(&mut self, game: &G, state: &G::State) -> i32 {
        self.inner.leaf_value(game, state)
    }

    fn policy_scores(
        &mut self,
        game: &G,
        state: &G::State,
        moves: &[G::Move],
        out: &mut Vec<f32>,
    ) -> bool {
        out.clear();
        for &mv in moves {
            self.move_features.extract(game, state, mv, &mut self.buf);
            out.push(self.ranker.score(&self.buf) as f32);
        }
        true
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
    fn ranker_learns_toy_rule_deterministically_and_round_trips() {
        // Feature 0 marks the better move.
        let pairs: Vec<RankPair> = (0..40)
            .map(|i| RankPair {
                better: vec![(0, 1.0), (1, (i % 3) as f32)],
                worse: vec![(1, (i % 5) as f32)],
            })
            .collect();
        let hyper = FitHyper {
            steps: 400,
            batch: 8,
            lr: 0.05,
            l2: 1e-6,
        };
        let run = || {
            let mut losses = Vec::new();
            let model = fit_move_ranker("toy_moves", 2, &pairs, hyper, 3, |s| {
                losses.push(s.train_loss)
            });
            (model, losses)
        };
        let (model, losses) = run();
        let (model2, losses2) = run();
        assert_eq!(losses, losses2);
        assert_eq!(model.weights, model2.weights);
        let (loss, acc) = evaluate_move_ranker(&model, &pairs);
        assert_eq!(acc, 1.0);
        assert!(loss < 0.2);
        assert!(model.weights[0] > 0.5, "the better-move marker is learned");
        let back: MoveRanker =
            serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
        assert_eq!(back.weights, model.weights);
    }

    #[test]
    fn ordering_changes_nodes_not_values() {
        // §69.4: with and without the ranker, search values at full
        // depth must be identical; only node counts may differ.
        use crate::features::forward_chess::{FcRecipe, MOVE_FEATURE_DIMENSION};
        use crate::games::forward_chess::{ForwardChess, Ruleset};
        use crate::search::{MoveOrdering, Searcher};
        let game = ForwardChess::new(Ruleset::Tiny);
        let model = {
            // A tiny fitted model: random-ish deterministic weights.
            let mut m = LinearWdl::zeros(
                FcRecipe::FcStructuredLinearV1.label(),
                crate::features::FeatureExtractor::<ForwardChess>::dimension(
                    &crate::features::forward_chess::FcExtractor::new(
                        &game,
                        FcRecipe::FcStructuredLinearV1,
                    ),
                ),
            );
            for (i, w) in m.weights.iter_mut().enumerate() {
                let h = crate::training::splitmix64(i as u64);
                w[2] = ((h % 1000) as f32 - 500.0) / 2000.0;
                w[0] = -w[2];
            }
            m
        };
        let mut ranker = MoveRanker::zeros("fc_move_v1", MOVE_FEATURE_DIMENSION);
        for (i, w) in ranker.weights.iter_mut().enumerate() {
            *w = ((crate::training::splitmix64(i as u64 ^ 7) % 100) as f32 - 50.0) / 100.0;
        }
        let mut rng = ChaCha12Rng::seed_from_u64(11);
        use rand::Rng as _;
        let mut moves = Vec::new();
        let mut state = game.initial_state();
        for _ in 0..30 {
            if crate::game::Game::outcome(&game, &state).is_some() {
                break;
            }
            let depth = 5;
            let mut plain =
                FcOrderedEvaluator::new(&game, &model, FcRecipe::FcStructuredLinearV1, None);
            let mut ordered = FcOrderedEvaluator::new(
                &game,
                &model,
                FcRecipe::FcStructuredLinearV1,
                Some(&ranker),
            );
            let mut s1: Searcher<ForwardChess> = Searcher::new(Some(12), MoveOrdering::Natural);
            let mut s2: Searcher<ForwardChess> = Searcher::new(Some(12), MoveOrdering::Natural);
            let r1 = s1.search(&game, &mut state, depth, u64::MAX, &mut plain);
            let r2 = s2.search(&game, &mut state, depth, u64::MAX, &mut ordered);
            assert_eq!(r1.value, r2.value, "ordering must never change values");
            assert_eq!(r1.completed_depth, r2.completed_depth);
            crate::game::Game::legal_moves(&game, &state, &mut moves);
            let mv = moves[rng.gen_range(0..moves.len())];
            crate::game::Game::make_move(&game, &mut state, mv);
        }
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

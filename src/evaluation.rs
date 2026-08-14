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
        let refs: Vec<&crate::training::Example> = chunk.iter().collect();
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

# Compute ↔ strength profile — Forward Chess and standard chess (2026-08-15)

What this answers: how much compute and memory the engines need to
play at a given strength — nodes/second and search depth on a single
GCP n2 core, nodes reachable in a 2 s move, model sizes and
architecture, the measured Elo value of search and of training
compute, what it costs to train wider models, and a grounded estimate
of what an NNUE-style evaluator would buy. Chess numbers are included
throughout for comparison.

Honesty rules (plan §29): Forward Chess Elo is **relative pool Elo**
under our exact protocol (anchor gen0 = 0), never human/FIDE
comparable, and never comparable to chess Elo. Chess numbers are
**protocol-relative logistic Elo** vs the named opponents. Every
number below is measured unless marked *estimate*.

## 1. Hardware baseline

All single-core numbers: one core of a GCP **n2** VM (Intel Xeon
Cascade Lake, 2.80 GHz), the same core type the training campaigns
ran on. Measurement tool: `lab bench` (single-threaded, deterministic
positions drawn from self-play with the benched checkpoint; see §9).

## 2. Model architecture and size

One architecture everywhere (plan §10), one capacity knob: width W.

- **Input**: sparse rule-state features (piece × square × side-to-move
  perspective facts; ≤ 37 active for Forward Chess full, of 1,548
  possible; chess: 780 possible), no handcrafted concepts.
- **Trunk**: embedding table `feature_count × W`, active rows summed
  and scaled by 1/√n, then two ReLU layers `W → W → W`.
- **Heads**: WDL `W → 3`; policy = per-action embedding, logit of
  action a is `A_a · h2 + c_a` (4,168 actions in both games; only
  legal actions are scored).
- **Inference**: custom compiled f32 path (`CompiledNet`), lazy heads
  (search asks for WDL at leaves without paying the 4,168-wide action
  head; that split alone was a 3.1× self-play speedup in Phase 7).

Parameter counts (Forward Chess full, 1,548 features / 4,168 actions):

| width | parameters | model.bin | trunk share | action-head share |
|---|---:|---:|---:|---:|
| w32 | 189,323 | 0.76 MB | 27% | 73% |
| w64 | 378,571 | 1.51 MB | 28% | 72% |
| w128 | 769,355 | 3.08 MB | 30% | 70% |

The action head owns ~70% of the parameters but almost none of the
search cost (lazy heads); the *search-time* cost per leaf is the
trunk: ~`37·W + 2·W² + 3·W` multiply-adds ≈ 10.8k at w64.

## 3. Single-core search speed, depth, and the 2-second move

`lab bench`, 60 positions from self-play, one thread, TT 2^16 entries.
"depth" is the fully completed iterative-deepening depth (full-width
alpha-beta, no quiescence/extensions — Forward Chess full has ~30–40
legal moves, so each depth step is expensive).

### Forward Chess (full 8×8) — strongest engine (c03, w64)

| budget | ms/move | nodes/s | completed ID depth |
|---:|---:|---:|---|
| 400 | 9.8 | 39,600 | 2.0 [1..3] |
| 800 | 17.3 | 44,000 | 2.1 [1..4] |
| 1,600 | 35.3 | 42,400 | 2.7 [1..4] |
| 6,400 | 161.5 | 36,700 | 3.4 [1..5] |
| 25,600 | 635.6 | 37,000 | 4.1 [1..6] |
| **2 s/move** | 1,836 | 37,300 | **4.8 [1..7], 68,500 nodes** |

Width comparison at 2 s/move on one core:

| evaluator | nodes/s | nodes in 2 s | depth | peak RSS |
|---|---:|---:|---:|---:|
| w32 | 49,900 | 91,600 | 4.7 | 13.2 MB |
| w64 | 37,300 | 68,500 | 4.8 | 13.6 MB |
| w128 | 27,300 | 54,600 | 4.3 | 17.0 MB |
| none (zero evaluator) | 108,800 | 210,700 | 6.8 | 11.4 MB |

The zero-evaluator row is the search/movegen overhead floor: the
model eval owns **66%** of node cost at w64 (54% at w32, 75% at
w128). Effective branching factor ≈ 10 (68,500^(1/4.8)) against
~30–40 legal moves — full-width alpha-beta + TT, no quiescence or
pruning extensions.

### Standard chess

| evaluator | 400 nodes | nodes/s | nodes in 2 s | depth in 2 s | peak RSS |
|---|---:|---:|---:|---:|---:|
| w32 (164,747 params) | 3.0 ms | 117–166k | 234,800 | 5.6 [5..7] | 13.0 MB |
| w64 (329,419 params, the −375 anchor engine) | 5.3 ms | 76–95k | 155,000 | 5.2 [4..6] | 15.6 MB |

Chess searches 2–3× faster per node than Forward Chess (cozy-chess
bitboard movegen vs our portable directional generator; 780 vs 1,548
features). At the Phase 7 anchor time control (0.3 s/move) the anchor
engine was searching **~23,000 nodes/move when it scored −375 ± 43 vs
Stockfish-17.1-at-20-nodes**. In-campaign throughput is lower than
these solo numbers (7 workers sharing 8 vCPUs plus feature encoding
and record writing: ~11k/25k nodes/s/thread at w64/w32).

### Memory while playing

Playing memory is tiny: **11–17 MB peak RSS** end to end — the model
(0.76 / 1.51 / 3.08 MB f32 at w32/w64/w128), the 2^16-entry
transposition table (~2 MB), and the binary. A Raspberry-Pi-class
device could host it. Training needs a few hundred MB per worker
(Burn autodiff + replay window ≈ 65k positions). The one genuinely
memory-hungry activity is exact solving: the 4×4 Forward Chess
retrograde solve held 46.5M positions at a 15 GB peak (D029
tablebase artifact: 559 MB).

## 4. Elo vs search compute (fixed model, more nodes)

### Forward Chess — strongest engine (c03, +476 pool Elo at 400 nodes)

Paired self-matches vs its own 400-node self
(`configs/phase_09/elo_vs_nodes_c03.toml`, 40 games per point; Elo =
logistic transform of the score, 95% CI from the score interval):

| nodes | vs 400-node self | Elo equivalent | 95% CI |
|---:|---:|---:|---|
| 50 (⅛×) | 0.088 | −407 | [−717, −289] |
| 100 (¼×) | 0.075 | −436 | [−925, −308] |
| 200 (½×) | 0.200 | −241 | [−396, −141] |
| 400 (1×) | 0.500 | 0 | control |
| 800 (2×) | 0.600 | +70 | [−19, +170] |
| 1,600 (4×) | 0.838 | +285 | [+180, +469] |
| 6,400 (16×) | 0.950 | +512 | [+353, —] |
| 25,600 (64×) | 0.975 | +636 | [+442, —] |

Monotone across a 256× budget range: **~134 Elo per doubling of
search on average** (steeper in the middle, ~90/doubling at the
top). A 2-second single-core move (68,500 nodes, §3) sits 2.7×
beyond the last measured point — ≥ +636 and ≈ +700 extrapolated
over the engine's own 400-node self, i.e. the 2 s/move engine is
around **+1,100–1,200 relative pool Elo** if pool ratings transfer
additively (*estimate*; the pool itself was rated at 400 nodes).

### Forward Chess — 400-node campaign champion (clean rules, 40 games/point)

| nodes | vs 400-node self | Elo equivalent |
|---:|---:|---:|
| 1,600 (4×) | 0.775 | +215 |
| 6,400 (16×) | 0.838 | +285 |
| 25,600 (64×) | 0.988 | +759 |

### Standard chess — Phase 7 champion (100 pairs/point)

| nodes | vs 400-node self | Elo equivalent |
|---:|---:|---:|
| 100 (¼×) | 0.070 | −449 |
| 400 (1×) | 0.500 (control) | 0 |
| 1,600 (4×) | 0.740 | +182 |
| 6,400 (16×) | 0.935 | +463 |

Rule of thumb from both games at this strength: **one doubling of
search nodes ≈ +90–175 Elo (≈ +134 mean)**, measured monotone over a
256× range on Forward Chess and 64× on chess. Iterative deepening
and the transposition table are already inside these numbers; depth
is bought only by nodes.

## 5. Elo vs training compute (fixed search, more generations)

One generation = 200 self-play games at the expert node budget +
2,000 Adam steps (batch 256) + 120 gated match games. Measured
generation wall time (7 worker threads on 8 n2 vCPUs):

| width | expert nodes | s/generation | vCPU·h/generation |
|---|---:|---:|---:|
| w32 | 400 | 117 | 0.26 |
| w64 | 400 | 140 | 0.31 |
| w128 | 400 | 209 | 0.46 |
| w32 | 800 | 139 | 0.31 |
| w64 | 800 | 177 | 0.39 |

The pool-Elo trajectory of the best line (w64, tabula rasa; vCPU·h
= wall × 8, i.e. what the VM bills):

| pool Elo | generations | wall time | vCPU·h |
|---:|---:|---:|---:|
| +201 | 8 | 19 min | 2.5 |
| +303 | 16 | 38 min | 5.1 |
| +364 | 24 | 57 min | 7.6 |
| +476 | 56 (last 24 at n800) | 2 h 14 min | 17.9 |

So at this scale: **the first ~+300 pool Elo costs ~5 vCPU-hours;
each further ~+100 costs roughly a doubling of total training
compute**, and the 400→800-node expert fork is what kept the curve
climbing past +360. The parallel w32 line reached statistical parity
with c03 (0.550 over 80 games) for ~15 vCPU·h — cheaper per
generation but it hit its capacity ceiling first under deep search.

Chess (Phase 7, same recipe): the w64 16-generation pilot that
scored the −375 ± 43 anchor trained in **39 min wall ≈ 5.2 vCPU·h**;
the strongest-per-compute chess engine (w32, 8 generations) trained
in **15 min ≈ 2.0 vCPU·h**. Within the pilot, generations 3 → 7 were
worth ≈ +200 protocol Elo; after that the curve plateaued at this
game-per-generation budget — in both games, more training compute
buys Elo fast early and then needs the expert-budget lever (or more
games per generation) to keep converting compute into strength.

## 6. How fast could larger models train? *(measured + extrapolation)*

Generation time grows sub-linearly in W (117 s → 140 s → 209 s for
4× the parameters) because a generation is dominated by the 2,000
Adam steps and by fixed search/movegen overhead, not by trunk math;
extrapolating both parts, **an estimated w256 generation is
~6 min** (≈ 3× w32). The real cost of width is not time but **data**: w128 was
already data-starved at 200 games/generation in both games (only 4/8
candidates promoted in chess; slower early Elo in Forward Chess), so
honest scaling to w256+ requires proportionally more games per
generation — i.e., the compute bill scales with data, not
parameters. At this VM's scale, w32–w64 is the efficient band, which
both games confirmed independently.

## 7. What would NNUE buy? *(analysis grounded in measurements)*

Our evaluator is already NNUE-shaped in structure (sparse features →
accumulator-style embedding sum → tiny dense layers) but **not** in
implementation: full recompute per leaf, f32 scalar math. NNUE's two
tricks and what they are worth here:

1. **Incremental accumulator updates** (make/unmake patches the
   feature sum instead of recomputing): removes the `37·W` gather-sum
   ≈ 20–25% of leaf FLOPs. Worth ~1.2–1.3× nodes/s.
2. **int8/int16 quantization + SIMD** on the `W×W` layers: 4–8× on
   the remaining ~75% of eval cost.

The measured bound (§3): eval is 66% of node cost at w64
(17.6 µs of 26.8 µs/node), so even a *free* evaluator caps the gain
at 2.9×. Realistic NNUE (incremental + int8 cutting eval ~5×) lands
at **≈ 2.0–2.2× nodes/s at w64** (37k → ~78k), a bit less at w32
where eval is only half the cost. By the measured search-scaling
value (§4, +70–175 Elo per doubling around the operating point),
that 2× is worth roughly **+70–175 Elo at fixed move time** — real
but not transformative.

The larger prizes NNUE points at, in measured order:

1. **Search overhead itself**: our no-model search runs 109k nodes/s
   where engine-grade movegen runs millions; an optimized search
   core (staged movegen, no per-node allocation) is worth more than
   quantized eval — plausibly 5–10× nodes on top, i.e. another
   +250–500 Elo at fixed time by §4's curve *(estimate)*.
2. **Spend the speed on capacity, not depth**: NNUE's real-world
   strength lives in its huge first layer (1024–3072 wide vs our
   32–128) at unchanged speed. Our width data says capacity is
   currently data-limited, not speed-limited (w32 ≈ w64 at parity;
   w128 data-starved) — so an NNUE-size accumulator only pays off
   together with proportionally more training games.

Both are inference/engineering changes, orthogonal to the learning
results, which is why the plan's simplicity contract keeps them out
of the phase gates.

## 8. Where this sits vs Stockfish (standard chess)

Measured anchors for the Phase 7 champion at 0.3 s/move
(protocol-relative, `timemargin=100`, D027):

| opponent | games | score | protocol Elo |
|---|---:|---:|---|
| random legal mover | 100 | 99.0% | ≈ +800 |
| Stockfish 17.1, UCI_Elo 1320 | 200 | 3.5% | −576 ± 141 |
| Stockfish 17.1 @ 20 nodes/move | 300 | 10.33% | **−375 ± 43** |

Context for "how far is Stockfish": full Stockfish runs ~1–2 M
nodes/s/core with NNUE and selective search reaching depth 20+ in
2 s; we measured 77k–235k nodes/s at full-width depth ~5–6. The
instructive anchor is SF **limited to 20 nodes/move** still being
+375 ahead: with 1,000× fewer nodes than our engine, its evaluation
and move ordering are worth more than our entire search — strength
at this scale is eval quality first, nodes second. By §4's measured
~134 Elo per doubling, buying +375 with search alone would take
~8× more nodes (NNUE speed and/or time); training gains (§5) and
eval quality are the cheaper currency — consistent with the plan's
goal of demonstrating learning dynamics, not competitive chess
strength.

## 9. Reproduce

```bash
# single-core speed/depth/movetime benchmark (new this session)
lab bench --game fc-full --checkpoint campaigns/fc_full_w64_n800/champion \
          --nodes 400,800,1600,6400,25600 --movetime-ms 2000
lab bench --game chess --nodes 400,1600 --movetime-ms 2000 \
          --checkpoint runs/20260814-131835-selfplay-chess-w64-g200-e10-s1-c03fefc/checkpoint

# Elo-vs-nodes ladder for the best engine
lab evaluate configs/phase_09/elo_vs_nodes_c03.toml

# Elo curves / campaign costs
tools/fc_rating.py --campaign campaigns/fc_full_w64_n800 --skip-baseline \
    --add gen0=campaigns/fc_full_w64/baseline_gen0 ...
```

Raw bench transcripts: `reports/compute_strength_profile/` in this
directory. Elo conversion used throughout: Elo = 400·log₁₀(s/(1−s)).

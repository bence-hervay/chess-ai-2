# SHSD F3 note — ranker iteration on the quiescent teacher (null result)

A cheap loop-closure test following G1, preserved per §6.7.

## Question

Does refitting the F2 move ranker on the calmer quiescent-teacher
records (G1's stability dataset: 7,079 fc-full positions, deep-label
stability 0.684 vs 0.366) produce a stronger ordering model?

## Result

No. Match, new-ranker vs F2-ranker, both with quiescence, same value
model, 20 ms/move, 300 pairs: **0.517 [0.480, 0.553]** — a tie.

Held-out pairwise accuracy actually *fell* on the calmer labels
(0.555 vs 0.611): the noisy 400-node child labels of F2 were partially
*predictably* noisy (hanging piece ↔ bad, capture ↔ good — exactly
what rule-level move features express), while the quiescent labels
resolve those tactics internally and expose distinctions the
35-feature representation cannot make.

## Interpretation and decision

Ordering at fc-full is now **representation-limited, not
label-limited** (the Case-6/7 pattern of §73 applied to the ordering
model). Teacher iteration for the ranker is closed until the move
representation grows — candidate features are search-history inputs
and structural state-action deltas (§35.2), to be admitted only
through the §14.2 protocol after value work. The F2 ranker remains the
production ordering.

## Reproducibility

Fit: `runs/20260816-005342-fit-fc-full-move-ranker-records-*`
(config `configs/shsd/stage_g/ranker_qsrecords_s1.toml`); match:
`runs/*ordermatch*` from
`configs/shsd/stage_g/newranker_vs_oldranker_20ms.toml`.

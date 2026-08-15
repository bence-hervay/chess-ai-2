#!/usr/bin/env python3
"""Generate the C1 sample-efficiency ladder configs + sweep manifest.

Usage: gen_ladder.py <lr> <l2>   (values selected by the grid, by val loss)

Arms (card: reports/shsd/cards/stage_c1_structured_linear.md):
  structured fc_structured_linear_v1  x N{1k,10k,100k,1M} x seeds{1,2,3}
  structured fc_counts_v0             x N                 x seeds{1,2,3}
  raw MLP w32 (recipe v1)             x N                 x seeds{1,2}

MLP steps scale with N (engineering choice, convergence monitored via
metrics.jsonl); structured fits use a fixed 20k-step budget (cheap,
converged under L2). N=1M entries request 2 sweep cores purely as a
memory throttle (~3 GB peak per 1M fit on the 16 GB VM).
"""
import os
import sys

LR, L2 = sys.argv[1], sys.argv[2]
TB = "runs/20260814-190733-solve-retro-fc-small-4deabb1/tablebase.bin"
HERE = os.path.dirname(os.path.abspath(__file__))
NS = [(1000, "1k"), (10000, "10k"), (100000, "100k"), (1000000, "1m")]
MLP_STEPS = {1000: 4000, 10000: 8000, 100000: 20000, 1000000: 40000}

COMMON = """train_positions = {n}
eval_cap = 20000
probe_states = 1000
probe_nodes = [400, 6400]
seed = {seed}
threads = 1

[game]
kind = "forward_chess"
ruleset = "small"

[source]
kind = "tablebase"
path = "{tb}"

"""

manifest = []
def emit(name, body, cores):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        f.write(body)
    manifest.append(
        '{"command":"fit","config":"configs/shsd/stage_c/%s","cores":%d}' % (name, cores)
    )

for n, tag in NS:
    cores = 2 if n == 1000000 else 1
    for seed in (1, 2, 3):
        for recipe in ("fc_structured_linear_v1", "fc_counts_v0"):
            short = "v1" if recipe.endswith("v1") else "v0"
            body = COMMON.format(n=n, seed=seed, tb=TB) + (
                '[model]\nkind = "structured"\nrecipe = "%s"\n'
                "steps = 20000\nbatch = 256\nlr = %s\nl2 = %s\n" % (recipe, LR, L2)
            )
            emit("ladder_%s_%s_s%d.toml" % (short, tag, seed), body, cores)
    for seed in (1, 2):
        body = COMMON.format(n=n, seed=seed, tb=TB) + (
            '[model]\nkind = "raw_mlp"\nwidth = 32\nsteps = %d\n' % MLP_STEPS[n]
        )
        emit("ladder_mlp32_%s_s%d.toml" % (tag, seed), body, cores)

with open(os.path.join(HERE, "ladder_manifest.jsonl"), "w") as f:
    f.write("\n".join(manifest) + "\n")
print("wrote %d configs + manifest" % len(manifest))

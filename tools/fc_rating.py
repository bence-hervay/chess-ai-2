#!/usr/bin/env python3
"""fc_rating.py — approximate internal Elo curve for a training campaign.

Plays deterministic paired matches (via `lab evaluate` match_probe)
between the snapshots of a tools/fc_train.sh campaign — the random-init
baseline, each chunk champion, and the final champion — then fits
relative Elo by maximum likelihood (logistic model, draws counted as
half a win) with bootstrap confidence intervals.

The pool and schedule follow plan §29: gauntlet of every snapshot vs
the latest champion, plus the adjacent-snapshot chain for connectivity
(use --round-robin for all pairs). Ratings are RELATIVE, engine-pool
Elo under this exact protocol — never comparable to human/FIDE Elo, and
Forward Chess Elo is never comparable to chess Elo (plan §29).

Match results are cached under campaigns/<name>/rating/matches/ — rerun
the script after more chunks and only the new pairings are played.

The first pool member is the 0-Elo anchor: normally the campaign's
random-init baseline_gen0 (for a fork, the fork point). --add members
are inserted first, so adding a parent champion makes IT the anchor.

Typical use:
    tools/fc_rating.py --campaign campaigns/fc_full_w64
    tools/fc_rating.py --campaign campaigns/fc_full_w64 --round-robin
"""

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

LAB = Path("target/release/lab")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--campaign", required=True, help="campaign directory (tools/fc_train.sh)")
    p.add_argument("--pairs", type=int, default=40, help="game pairs per pairing [40]")
    p.add_argument("--nodes", type=int, default=None,
                   help="node budget per move [campaign EVAL_NODES]")
    p.add_argument("--opening-plies", type=int, default=2,
                   help="shared random opening plies per pair [2]")
    p.add_argument("--seed", type=int, default=777, help="base seed for matches/bootstrap [777]")
    p.add_argument("--threads", type=int, default=7, help="worker threads per match [7]")
    p.add_argument("--round-robin", action="store_true",
                   help="play all snapshot pairs (default: gauntlet vs latest + adjacent chain)")
    p.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                   help="add an external checkpoint to the pool (repeatable), e.g. a "
                        "parent campaign's champion when rating a fork: "
                        "--add parent=campaigns/base/champion")
    p.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples [200]")
    p.add_argument("--skip-matches", action="store_true",
                   help="fit from cached results only (no new matches)")
    return p.parse_args()


def campaign_env(campaign: Path) -> dict:
    env = {}
    for line in (campaign / "campaign.env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def pool_members(campaign: Path):
    members = []
    gen0 = campaign / "baseline_gen0"
    if (gen0 / "model.bin").exists():
        members.append(("gen0", gen0))
    for chunk in sorted(campaign.glob("chunk_*")):
        if (chunk / "DONE").exists() and (chunk / "checkpoint" / "model.bin").exists():
            members.append((chunk.name.replace("chunk_0", "c").replace("chunk_", "c"), chunk / "checkpoint"))
    return members


def schedule(members, round_robin):
    pairs = set()
    last = len(members) - 1
    if round_robin:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((i, j))
    else:
        for i in range(last):
            pairs.add((i, last))       # gauntlet vs latest
        for i in range(last):
            pairs.add((i, i + 1))      # adjacent chain
    return sorted(pairs)


def stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:12], 16)


def play_pairing(args, env, cache: Path, name_a, ckpt_a, name_b, ckpt_b):
    out = cache / f"{name_a}_vs_{name_b}.json"
    if out.exists():
        return json.loads(out.read_text())
    if args.skip_matches:
        return None
    nodes = args.nodes or int(env["EVAL_NODES"])
    config = cache.parent / "configs" / f"{name_a}_vs_{name_b}.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        'kind = "match_probe"\n'
        f'checkpoint = "{ckpt_a}"\n'
        f'opponent_checkpoint = "{ckpt_b}"\n'
        f"node_budgets = [{nodes}]\n"
        f"baseline_nodes = {nodes}\n"
        f"pairs = {args.pairs}\n"
        f"opening_plies = {args.opening_plies}\n"
        f"seed = {stable_seed(args.seed, name_a, name_b, args.pairs, nodes)}\n"
        f"threads = {args.threads}\n\n"
        "[game]\n"
        'kind = "forward_chess"\n'
        f'ruleset = "{env["RULESET"]}"\n'
    )
    print(f"  playing {name_a} vs {name_b} ({args.pairs} pairs at {nodes} nodes)...", flush=True)
    proc = subprocess.run([str(LAB), "evaluate", str(config)],
                          capture_output=True, text=True, check=True)
    run_dir = re.search(r"^run directory: (.+)$", proc.stdout, re.M).group(1)
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    match = summary["budgets"][0]["match"]
    record = {
        "a": name_a, "b": name_b,
        "wins": match["candidate_wins"], "draws": match["draws"],
        "losses": match["candidate_losses"], "games": match["games"],
        "score": match["score"], "run_dir": run_dir,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return record


def fit_elo(names, results, anchor=0):
    """Logistic MLE by gradient ascent; draws count half. Anchor fixed at 0."""
    rating = {n: 0.0 for n in names}
    scale = math.log(10) / 400
    for _ in range(4000):
        grad = {n: 0.0 for n in names}
        for r in results:
            wins, draws, losses = r["wins"], r["draws"], r["losses"]
            d = rating[r["a"]] - rating[r["b"]]
            p = 1 / (1 + math.exp(-scale * d))
            g = scale * ((wins + 0.5 * draws) - (wins + draws + losses) * p)
            grad[r["a"]] += g
            grad[r["b"]] -= g
        step = 4000.0
        for n in names:
            rating[n] += step * grad[n] / max(1, sum(x["games"] for x in results))
        rating_anchor = rating[names[anchor]]
        for n in names:
            rating[n] -= rating_anchor
    return rating


def bootstrap_ci(names, results, resamples, seed):
    rng = random.Random(seed)
    samples = {n: [] for n in names}
    for _ in range(resamples):
        fake = []
        for r in results:
            n = r["games"]
            outcomes = ["w"] * r["wins"] + ["d"] * r["draws"] + ["l"] * r["losses"]
            draw = [outcomes[rng.randrange(n)] for _ in range(n)]
            fake.append({"a": r["a"], "b": r["b"],
                         "wins": draw.count("w"), "draws": draw.count("d"),
                         "losses": draw.count("l"), "games": n})
        fit = fit_elo(names, fake)
        for n in names:
            samples[n].append(fit[n])
    ci = {}
    for n in names:
        xs = sorted(samples[n])
        lo = xs[int(0.025 * len(xs))]
        hi = xs[min(len(xs) - 1, int(0.975 * len(xs)))]
        ci[n] = (lo, hi)
    return ci


def ascii_curve(names, rating, width=48):
    values = [rating[n] for n in names]
    lo, hi = min(values), max(values)
    span = max(1.0, hi - lo)
    lines = []
    for n in names:
        bar = int((rating[n] - lo) / span * width)
        lines.append(f"{n:>6} {'#' * bar}{' ' * (width - bar)} {rating[n]:+7.1f}")
    return "\n".join(lines)


def main():
    args = parse_args()
    campaign = Path(args.campaign)
    env = campaign_env(campaign)
    members = pool_members(campaign)
    for spec in args.add:
        if "=" not in spec:
            sys.exit(f"--add wants NAME=PATH, got {spec}")
        name, path = spec.split("=", 1)
        path = Path(path)
        if not (path / "model.bin").exists():
            sys.exit(f"--add {spec}: no model.bin under {path}")
        members.insert(0, (name, path))
    if len(members) < 2:
        sys.exit("need at least two snapshots (run tools/fc_train.sh first)")
    names = [n for n, _ in members]
    print(f"pool: {' '.join(names)} ({env['GAME']} w{env['WIDTH']})")

    cache = campaign / "rating" / "matches"
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for i, j in schedule(members, args.round_robin):
        record = play_pairing(args, env, cache,
                              members[i][0], members[i][1],
                              members[j][0], members[j][1])
        if record:
            results.append(record)
    if not results:
        sys.exit("no match results (remove --skip-matches?)")

    rating = fit_elo(names, results)
    ci = bootstrap_ci(names, results, args.bootstrap, args.seed)

    lines = ["| snapshot | Elo (rel. gen0) | 95% CI | pairings |", "|---|---:|---|---:|"]
    for n in names:
        involved = sum(1 for r in results if n in (r["a"], r["b"]))
        lines.append(f"| {n} | {rating[n]:+.1f} | [{ci[n][0]:+.1f}, {ci[n][1]:+.1f}] | {involved} |")
    table = "\n".join(lines)
    curve = ascii_curve(names, rating)
    print("\n" + table + "\n\n" + curve)

    out = campaign / "rating"
    (out / "ratings.md").write_text(
        f"# Internal pool Elo — {campaign.name}\n\n"
        "Relative engine-pool Elo under this exact protocol (never\n"
        "human/FIDE-comparable; Forward Chess and chess Elo are not\n"
        "comparable to each other). Anchor: gen0 = 0.\n\n"
        + table + "\n\n```\n" + curve + "\n```\n")
    with (out / "ratings.csv").open("w") as f:
        f.write("snapshot,elo,ci_lo,ci_hi\n")
        for n in names:
            f.write(f"{n},{rating[n]:.2f},{ci[n][0]:.2f},{ci[n][1]:.2f}\n")
    print(f"\nwritten: {out}/ratings.md, {out}/ratings.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate `lab fit` summaries into a markdown table (SHSD C1).

Usage: tools/shsd_fit_table.py [run-dir-glob ...]
Defaults to runs/*fit-fc-small*. Groups by (model, train size), shows
mean +- spread over seeds for test log-loss/accuracy and probe rates.
"""
import glob
import json
import sys
from collections import defaultdict


def main():
    patterns = sys.argv[1:] or ["runs/*fit-fc-small*"]
    rows = []
    for pattern in patterns:
        for run in sorted(glob.glob(pattern)):
            try:
                summary = json.load(open(f"{run}/summary.json"))
                resolved = open(f"{run}/resolved.toml").read()
            except (OSError, json.JSONDecodeError):
                continue
            seed = next(
                (
                    line.split("=")[1].strip()
                    for line in resolved.splitlines()
                    if line.startswith("seed")
                ),
                "?",
            )
            rows.append(
                {
                    "model": summary["model"],
                    "n": summary["train_states"],
                    "seed": seed,
                    "test_ll": summary["test"]["log_loss"],
                    "test_acc": summary["test"]["accuracy"],
                    "prior_ll": summary["class_prior"]["log_loss"],
                    "raw": summary["probe"]["raw_optimal_rate"],
                    "searched": summary["probe"]["searched_optimal_rate"],
                    "zero": summary["probe_zero_evaluator"]["searched_optimal_rate"],
                    "ns": summary["extraction_ns_per_position"],
                    "wall": summary["cost"]["wall_seconds"],
                    "run": run,
                }
            )
    if not rows:
        print("no fit summaries found")
        return

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["n"])].append(row)

    def agg(values):
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / 2 if len(values) > 1 else 0.0
        return mean, spread

    print(
        "| model | N | seeds | test log-loss | test acc | probe raw | "
        "probe @400 | probe @6400 | wall s |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (model, n) in sorted(groups, key=lambda k: (k[0], k[1])):
        group = groups[(model, n)]
        ll, ll_s = agg([r["test_ll"] for r in group])
        acc, acc_s = agg([r["test_acc"] for r in group])
        raw, _ = agg([r["raw"] for r in group])
        s400, _ = agg([r["searched"][0] for r in group])
        s6400, _ = agg([r["searched"][1] for r in group]) if len(group[0]["searched"]) > 1 else (float("nan"), 0)
        wall, _ = agg([r["wall"] for r in group])
        print(
            f"| {model} | {n} | {len(group)} | {ll:.4f} ±{ll_s:.4f} | "
            f"{acc:.4f} ±{acc_s:.4f} | {raw:.4f} | {s400:.4f} | {s6400:.4f} | {wall:.0f} |"
        )
    zero = rows[0]["zero"]
    print(f"\nzero-evaluator searched baseline (same probes): {zero}")
    print(f"class-prior test log-loss: {rows[0]['prior_ll']:.4f}")


if __name__ == "__main__":
    main()

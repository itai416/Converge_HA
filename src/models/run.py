"""Run a set of experiments under repeated nested CV, in parallel, and summarise them.

An experiment is name -> (model class, fixed params, tuning grid, feature frame). Every experiment is run for
each loss x weighting x repeat (fold assignment), and models with a `seed` parameter also for each seed.

Outputs in out_dir:
  runs.csv              one row per (model, loss, weighting, repeat, seed) with the three metrics
  summary.csv           mean, std, min, max of each metric over all runs of a configuration
  metrics_rep0.csv      repeat 0 (= folds.csv), seed-averaged predictions, 95% bootstrap CI over complexes
  oof_rep0.parquet      those repeat-0 out-of-fold predictions
  chosen_params.json    hyperparameters picked by the inner CV in every outer fold of every run
"""
import inspect
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.eval.metrics import bootstrap, metrics
from src.models.cv import nested_cv

LOSSES = ["huber", "mse"]
WEIGHTINGS = ["none", "sqrt"]


def _job(cls, fixed, grid, X, y, groups, outer, loss, w, seed):
    extra = {"seed": seed} if seed is not None else {}
    make = lambda **p: cls(**fixed, loss=loss, **extra, **p)  # noqa: E731
    return nested_cv(make, grid, X, y, groups, outer, w_scheme=w)


def run_experiments(experiments: dict, df: pd.DataFrame, rep_folds: pd.DataFrame, out_dir, n_seeds=3, n_jobs=15,
                    losses=LOSSES, weightings=WEIGHTINGS):
    """df: one row per mutation with complex, ddG; rep_folds: complex -> rep0..repN fold columns."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = [c for c in rep_folds.columns if c.startswith("rep")]
    folds = df[["complex"]].merge(rep_folds, on="complex", how="left", validate="m:1")
    assert folds[reps].notna().all().all()
    y, groups = df["ddG"].values, df["complex"]

    tasks = []
    for name, (cls, fixed, grid, X) in experiments.items():
        seeds = range(n_seeds) if "seed" in inspect.signature(cls.__init__).parameters else [None]
        for loss in losses:
            if cls.__name__ == "MeanBaseline" and loss != losses[0]:
                continue  # the mean does not depend on the loss
            for w in weightings:
                for r in reps:
                    for s in seeds:
                        tasks.append(((name, loss, w, r, s), (cls, fixed, grid, X, y, groups, folds[r].values, loss, w, s)))
    print(f"{len(tasks)} runs ({len(reps)} repeats, {n_seeds} seeds for seeded models) on {n_jobs} workers", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_job)(*a) for _, a in tasks)

    runs, chosen, rep0 = [], {}, {}
    base = df[["complex", "ddG"]]
    for (key, _), (pred, ch) in zip(tasks, results):
        name, loss, w, r, s = key
        runs.append({"model": name, "loss": loss, "weighting": w, "repeat": r, "seed": s,
                     **metrics(base.assign(pred=pred))})
        chosen[" | ".join(map(str, key))] = ch
        if r == reps[0]:
            rep0.setdefault((name, loss, w), []).append(pred)
    runs = pd.DataFrame(runs)
    runs.to_csv(out_dir / "runs.csv", index=False)

    cfg = ["model", "loss", "weighting"]
    summary = runs.groupby(cfg, sort=False)[["per_complex_spearman", "pooled_pearson", "rmse"]].agg(
        ["mean", "std", "min", "max"])
    summary["n_runs"] = runs.groupby(cfg, sort=False).size()
    summary.round(4).to_csv(out_dir / "summary.csv")

    ci_rows, oof = [], df[["complex", "Mutation(s)_cleaned", "ddG"]].copy()
    for (name, loss, w), preds in rep0.items():
        p = np.mean(preds, axis=0)  # average over seeds
        oof[f"{name} | {loss} | w={w}"] = p
        for m, r in bootstrap(base.assign(pred=p)).iterrows():
            ci_rows.append({"model": name, "loss": loss, "weighting": w, "metric": m, **r.to_dict()})
    pd.DataFrame(ci_rows).to_csv(out_dir / "metrics_rep0.csv", index=False)
    oof.to_parquet(out_dir / "oof_rep0.parquet", index=False)
    (out_dir / "chosen_params.json").write_text(json.dumps(chosen, indent=1, default=str))
    edges = grid_edge_report(chosen, experiments)
    edges.to_csv(out_dir / "grid_edges.csv", index=False)
    flagged = edges[edges["edge_share"] >= 0.5]
    if len(flagged):
        print("WARNING: hyperparameters chosen at a grid edge in >= 50% of folds (optimum may lie outside the grid):\n"
              + flagged.to_string(index=False), flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)
    return summary


def grid_edge_report(chosen: dict, experiments: dict) -> pd.DataFrame:
    """For every tuned numeric hyperparameter: how often inner CV picked the lowest / highest grid value.

    Only grids with >= 3 numeric values are judged: with two values every choice is an edge."""
    rows = []
    for name, (_, _, grid, _) in experiments.items():
        picks = [p for key, folds in chosen.items() if key.split(" | ")[0] == name for p in folds.values()]
        for h, values in grid.items():
            nums = sorted(v for v in values if isinstance(v, (int, float)))
            if len(nums) < 3 or not picks:
                continue
            got = [p[h] for p in picks]
            lo, hi = sum(g == nums[0] for g in got), sum(g == nums[-1] for g in got)
            rows.append({"model": name, "param": h, "grid": nums, "n": len(got), "at_low": lo, "at_high": hi,
                         "edge_share": round((lo + hi) / len(got), 2)})
    return pd.DataFrame(rows)

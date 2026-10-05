"""Stage 7 stratified reporting and Stage 8 error analysis, from the saved out-of-fold predictions (primary split, repeat 0).

Models: C augmented ridge (best per-complex Spearman) and D LightGBM (best tree model). Writes tables and figures to
results/error_analysis/. Usage: python scripts/10_error_analysis.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import RESULTS, load_oof_frame  # noqa: E402
from src.eval.metrics import MIN_ROWS, bootstrap, metrics  # noqa: E402

OUT = RESULTS / "error_analysis"
MODELS = {"C: ridge": "C: augmented ridge | huber | w=none", "D: LightGBM": "D: LightGBM | huber | w=none"}
STRATA = ["antigen", "side", "loc", "to_ala", "anti_idiotype"]


def rmse(resid):
    return float(np.sqrt((resid ** 2).mean()))


def load():
    oof, base = load_oof_frame(["iMutation_Location(s)", "antigen_group", "wt_aa", "mut_aa", "ddG_std", "n_repeats"],
                               geom_cols=["on_antibody"], fold_cols=["fold_complex", "antigen_seen"])
    base["loc"] = base["iMutation_Location(s)"]
    base["to_ala"] = np.where(base["mut_aa"] == "A", "to Ala", "other")
    base["anti_idiotype"] = np.where(base["antigen_group"] == "anti_idiotype", "anti-idiotype", "other")
    base["antigen"] = np.where(base["antigen_seen"], "seen antigen", "unseen antigen")
    base["ddG_bin"] = pd.cut(base["ddG"], [-9, -0.5, 0.5, 1, 2, 9], labels=["< -0.5 (improving)", "-0.5..0.5", "0.5..1", "1..2", "> 2"])
    return oof, base


def stratified(oof, base):
    """Metrics of every model within each level of each stratum, plus the overall row with a bootstrap CI printed."""
    def row(name, stratum, level, g, m):
        return {"model": name, "stratum": stratum, "level": level, "rows": len(g), "complexes": g["complex"].nunique(),
                "complexes>=10": int((g.groupby("complex").size() >= MIN_ROWS).sum()),
                "per_complex_spearman": m["per_complex_spearman"],
                "within_stratum_spearman": spearmanr(g["pred"], g["ddG"])[0] if len(g) > 2 else np.nan, "rmse": m["rmse"]}

    rows = []
    for name, col in MODELS.items():
        d = base.assign(pred=oof[col].values)
        rows += [row(name, s, lvl, g, metrics(g)) for s in STRATA for lvl, g in d.groupby(s)]
        ci = bootstrap(d[["complex", "ddG", "pred"]], n=500)
        rows.append(row(name, "all", "all", d, ci["value"]))
        print(name, "bootstrap CI over complexes:\n", ci.round(3))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oof, base = load()
    tab = stratified(oof, base)
    tab.to_csv(OUT / "stratified_metrics.csv", index=False)
    print(tab.round(3).to_string(index=False))

    # residuals by true ddG magnitude (regression toward the mean) and calibration, for the best model
    d = base.assign(pred=oof[MODELS["C: ridge"]].values)
    d["resid"] = d["pred"] - d["ddG"]
    res = d.groupby("ddG_bin", observed=True).agg(rows=("resid", "size"), mean_true=("ddG", "mean"), mean_pred=("pred", "mean"),
                                                  mean_resid=("resid", "mean"), rmse=("resid", rmse))
    res.to_csv(OUT / "residual_by_true_ddG.csv")
    print(res.round(2))
    for col in ["loc", "side", "wt_aa", "mut_aa"]:
        t = d.groupby(col).agg(rows=("resid", "size"), mean_resid=("resid", "mean"), rmse=("resid", rmse))
        t[t.rows >= 15].to_csv(OUT / f"residual_by_{col}.csv")

    # per-complex performance vs support
    pc = pd.DataFrame([{"complex": c, "antigen": g["antigen_group"].iloc[0], "seen": bool(g["antigen_seen"].iloc[0]), "rows": len(g),
                        "spearman": spearmanr(g["pred"], g["ddG"])[0] if len(g) >= MIN_ROWS else np.nan,
                        "rmse": rmse(g["resid"]), "ddG_std": g["ddG"].std()}
                       for c, g in d.groupby("complex")]).sort_values("rows", ascending=False)
    pc.to_csv(OUT / "per_complex.csv", index=False)
    print(pc.round(2).to_string(index=False))

    # worst errors for case studies
    d["abs_err"] = d["resid"].abs()
    worst = d.sort_values("abs_err", ascending=False).head(12)[
        ["complex", "Mutation(s)_cleaned", "antigen_group", "loc", "side", "ddG", "pred", "ddG_std", "n_repeats"]]
    worst.to_csv(OUT / "worst_errors.csv", index=False)
    print(worst.round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].scatter(d["ddG"], d["pred"], s=8, alpha=.5)
    ax[0].plot([-3, 8], [-3, 8], "k--", lw=1)
    ax[0].set(xlabel="observed ddG", ylabel="predicted ddG (out-of-fold)", title="C: ridge, predicted vs observed")
    ax[1].bar(range(len(res)), res["mean_resid"])
    ax[1].set_xticks(range(len(res)), res.index.astype(str), rotation=20)
    ax[1].set(ylabel="mean (pred - obs)", title="Regression toward the mean: bias by true ddG")
    t = tab[(tab.model == "C: ridge") & tab.stratum.isin(["antigen", "side", "loc"])]
    ax[2].barh(t["stratum"] + ": " + t["level"].astype(str), t["within_stratum_spearman"])
    ax[2].set(xlabel="Spearman within stratum", title="Where the model ranks well")
    fig.tight_layout()
    fig.savefig(OUT / "error_analysis.png", dpi=130)


if __name__ == "__main__":
    main()

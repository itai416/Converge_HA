"""EDA of the features we get for free (no structure needed) on the deduplicated v1 set (668 rows).
Figures go to results/eda/ (06-09)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import RESULTS, is_v1, load_dedup  # noqa: E402
# charge at pH 7, Kyte-Doolittle hydropathy, Zamyatnin residue volumes (A^3)
from src.features.geometry import CHARGE as CHG, KD as HYD, VOLUME as VOL  # noqa: E402

OUT = RESULTS / "eda"
sns.set_theme(style="whitegrid", context="talk")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def load():
    d = load_dedup()
    v = d[is_v1(d)].copy()
    v["loc"] = v["iMutation_Location(s)"]
    v["to_ala"] = v["mut_aa"] == "A"
    v["d_vol"] = v["mut_aa"].map(VOL) - v["wt_aa"].map(VOL)
    v["d_hyd"] = v["mut_aa"].map(HYD) - v["wt_aa"].map(HYD)
    v["d_chg"] = v["mut_aa"].map(CHG).fillna(0) - v["wt_aa"].map(CHG).fillna(0)
    return v


def adj_r2(v, col):
    """Share of ddG variance explained by group means of `col`, corrected for the number of groups."""
    y = v["ddG"]
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - v.groupby(col)["ddG"].transform("mean")) ** 2).sum()
    n, k = len(v), v[col].nunique()
    return 1 - (ss_res / (n - k)) / (ss_tot / (n - 1))


def noise_ceiling(v):
    """Pooled variance of repeated measurements vs total variance -> best achievable R^2 / Pearson."""
    r = v[v["n_repeats"] > 1]
    noise = ((r["n_repeats"] - 1) * r["ddG_std"] ** 2).sum() / (r["n_repeats"] - 1).sum()
    r2 = 1 - noise / v["ddG"].var()
    return noise, r2, np.sqrt(r2)


def plot_explained(v):
    cols = {"complex": "complex", "antigen_group": "antigen group", "loc": "location class",
            "wt_aa": "wild-type aa", "mut_aa": "mutant aa", "to_ala": "to-Ala flag"}
    r2 = pd.Series({lab: adj_r2(v, c) for c, lab in cols.items()}).sort_values()
    _, ceil, _ = noise_ceiling(v)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(r2.index, r2.values, color="C0")
    for i, x in enumerate(r2.values):
        ax.text(x + 0.005, i, f"{x:.2f}", va="center")
    ax.axvline(ceil, color="C3", ls="--", label=f"noise ceiling {ceil:.2f}")
    ax.set(xlabel="share of ΔΔG variance explained (adjusted R²)", xlim=(0, 1),
           title=f"How much does knowing each feature tell us? (n={len(v)})")
    ax.legend(loc="lower right")
    save(fig, "06_variance_explained.png")
    print("adjusted R2 per feature:\n", r2.round(3).to_string())


def plot_location(v):
    order = ["COR", "SUP", "RIM", "INT", "SUR"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=v, x="loc", y="ddG", order=order, color="0.85", fliersize=0, ax=ax)
    sns.stripplot(data=v, x="loc", y="ddG", order=order, size=3, alpha=0.5, ax=ax)
    ax.axhline(0, color="k", lw=1)
    n = v["loc"].value_counts()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c}\n(n={n[c]})" for c in order])
    ax.set(title="ΔΔG by interface location class", xlabel="", ylabel="ΔΔG (kcal/mol)")
    save(fig, "07_location_class.png")
    t = v.groupby("loc")["ddG"].agg(["size", "mean", "median", "std"]).loc[order].round(2)
    t["pct_>1"] = (v.groupby("loc")["ddG"].apply(lambda s: (s > 1).mean()) * 100).loc[order].round(0)
    print(t.to_string())


def plot_substitution(v):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 2.2]})
    sns.boxplot(data=v, x="to_ala", y="ddG", color="0.85", fliersize=0, ax=axes[0])
    sns.stripplot(data=v, x="to_ala", y="ddG", size=3, alpha=0.5, ax=axes[0])
    n = v["to_ala"].value_counts()
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([f"other\n(n={n[False]})", f"to-Ala\n(n={n[True]})"])
    axes[0].set(xlabel="", ylabel="ΔΔG (kcal/mol)", title="to-Ala vs other")
    g = v.groupby("wt_aa")["ddG"].agg(["mean", "sem", "size"]).sort_values("mean")
    axes[1].bar(g.index, g["mean"], yerr=g["sem"], color="C0", capsize=3)
    for i, (a, row) in enumerate(g.iterrows()):
        axes[1].text(i, -0.15, f"{int(row['size'])}", ha="center", va="top", fontsize=10)
    axes[1].set(title="Mean ΔΔG by wild-type residue (n below bars, ± sem)", ylabel="mean ΔΔG")
    save(fig, "08_substitution.png")
    print(v.groupby("to_ala")["ddG"].agg(["size", "mean", "median", "std"]).round(2).to_string())
    print(v[v["to_ala"]].groupby("loc")["ddG"].agg(["size", "mean"]).round(2).T.to_string())


def plot_descriptors(v):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (c, lab) in zip(axes, [("d_vol", "Δ volume (Å³)"), ("d_hyd", "Δ hydropathy"), ("d_chg", "Δ charge")]):
        rho = spearmanr(v[c], v["ddG"])[0]
        sns.regplot(data=v, x=c, y="ddG", ax=ax, scatter_kws=dict(s=10, alpha=0.4), line_kws=dict(color="C3"),
                    x_jitter=0.05 if c == "d_chg" else 0)
        ax.set(title=f"{lab}: Spearman {rho:.2f}", xlabel=lab, ylabel="ΔΔG (kcal/mol)")
        print(f"{c}: Spearman {rho:.2f}")
    save(fig, "09_descriptors.png")


def main():
    v = load()
    print(f"v1 dedup rows: {len(v)}")
    noise, r2, rho = noise_ceiling(v)
    print(f"noise: pooled repeat std {np.sqrt(noise):.2f} vs total std {v['ddG'].std():.2f} -> "
          f"ceiling R2 {r2:.2f}, Pearson {rho:.2f}")
    plot_explained(v)
    plot_location(v)
    plot_substitution(v)
    plot_descriptors(v)


if __name__ == "__main__":
    main()

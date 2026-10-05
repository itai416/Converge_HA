"""Basic EDA on the SKEMPI 2.0 AB/AG subset: label distribution, rows per complex,
ΔΔG per complex, and mutations per row. Figures go to results/eda/."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.clean import load_clean  # noqa: E402
from src.data.dedup import dedup  # noqa: E402
from src.data.io import is_v1  # noqa: E402

OUT = ROOT / "results" / "eda"

sns.set_theme(style="whitegrid", context="talk")


def load():
    df = load_clean()
    df["kind"] = np.where(df["n_mut"] == 1, "single-point", "multi-point")
    df["valid"] = ~df["censored"]
    df["v1"] = is_v1(df)
    return df


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def plot_ddg_hist(df):
    ok = df[df["valid"]]
    v1 = df[df["v1"]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(v1["ddG"], bins=40, kde=True, ax=axes[0], color="C0")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].axvline(v1["ddG"].median(), color="C3", ls="--", label=f"median {v1['ddG'].median():.2f}")
    axes[0].set(title=f"ΔΔG, v1 set (n={len(v1)}, skew={v1['ddG'].skew():.2f})", xlabel="ΔΔG (kcal/mol)")
    axes[0].legend()
    sns.histplot(ok, x="ddG", hue="kind", multiple="stack", bins=40, ax=axes[1])
    axes[1].set(title=f"ΔΔG, all uncensored valid rows (n={len(ok)})", xlabel="ΔΔG (kcal/mol)")
    save(fig, "01_ddg_hist.png")
    print("v1 ddG quantiles:\n", v1["ddG"].describe(percentiles=[.05, .25, .5, .75, .95]).round(2))
    print(f"share improving (<-0.5): {(v1.ddG < -0.5).mean():.2%}, neutral: "
          f"{v1.ddG.between(-0.5, 1).mean():.2%}, destabilizing (>1): {(v1.ddG > 1).mean():.2%}")


def plot_rows_per_complex(df):
    cat = np.select([df["censored"], df["n_mut"] > 1], ["censored", "multi-point"], "single-point")
    df = df.assign(category=cat)
    order = df.groupby("complex").size().sort_values(ascending=False).index
    tab = (df.groupby(["complex", "category"]).size().unstack(fill_value=0)
           .reindex(order)[["single-point", "multi-point", "censored"]])
    fig, ax = plt.subplots(figsize=(18, 7))
    tab.plot.bar(stacked=True, ax=ax, color=["C0", "C1", "C7"], width=0.85)
    ax.set(title=f"Rows per complex ({len(tab)} complexes, {len(df)} rows)", xlabel="", ylabel="rows")
    ax.tick_params(axis="x", labelsize=9)
    save(fig, "02_rows_per_complex.png")
    print(tab.sum(axis=1).describe().round(1))


def plot_ddg_per_complex(df):
    v1 = df[df["v1"]]
    order = v1.groupby("complex")["ddG"].median().sort_values().index
    counts = v1.groupby("complex").size()
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.boxplot(data=v1, x="complex", y="ddG", order=order, color="0.85", fliersize=0, ax=ax)
    sns.stripplot(data=v1, x="complex", y="ddG", order=order, size=3, alpha=0.7, ax=ax)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c} (n={counts[c]})" for c in order], rotation=90, fontsize=8)
    ax.set(title="ΔΔG per complex, v1 set (sorted by median)", xlabel="", ylabel="ΔΔG (kcal/mol)")
    save(fig, "03_ddg_per_complex.png")


def plot_mutations_per_row(df):
    tab = df.groupby(["n_mut", "censored"]).size().unstack(fill_value=0)
    tab.columns = ["uncensored" if not c else "censored" for c in tab.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    tab.plot.bar(stacked=True, ax=ax, color=["C0", "C7"][: tab.shape[1]], rot=0)
    for i, tot in enumerate(tab.sum(axis=1)):
        ax.text(i, tot, str(tot), ha="center", va="bottom", fontsize=11)
    ax.set(title="Mutations per row", xlabel="# point mutations in the row", ylabel="rows")
    save(fig, "04_mutations_per_row.png")
    print(tab)


def plot_repeats(df):
    """Mutations measured more than once: per-mutation mean +- std of the repeats, grouped by complex."""
    rep = dedup(df)
    rep = rep[~rep["censored"] & (rep["n_repeats"] > 1)]
    order = rep.groupby("complex").size().sort_values(ascending=False).index
    pos = {c: i for i, c in enumerate(order)}
    n = rep.groupby("complex")["ddG"].transform("size")
    rep = rep.assign(x=rep["complex"].map(pos) + (rep.groupby("complex").cumcount() - n / 2) * 0.8 / n)  # spread within a complex
    per_c = rep.groupby("complex").agg(n=("ddG", "size"), std=("ddG_std", "mean")).reindex(order)

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax = axes[0]
    ax.errorbar(rep["x"], rep["ddG"], yerr=rep["ddG_std"], fmt="o", ms=4, lw=1, alpha=0.8, capsize=2)
    ax.axhline(0, color="k", lw=1)
    ax.set(title=f"Repeated mutations: mean ΔΔG ± std of the repeats ({len(rep)} mutations, {len(order)} complexes)",
           ylabel="ΔΔG (kcal/mol)")
    axes[1].bar(range(len(order)), per_c["std"], color="C1")
    axes[1].set(ylabel="mean std of repeats", xlabel="")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([f"{c} (n={per_c.loc[c, 'n']})" for c in order], rotation=90, fontsize=9)
    save(fig, "05_repeats_mean_std.png")
    print(f"repeated mutations: {len(rep)}; mean std {rep['ddG_std'].mean():.2f}, "
          f"median {rep['ddG_std'].median():.2f}, max {rep['ddG_std'].max():.2f}")


def main():
    df = load()
    print(f"AB/AG rows: {len(df)}, complexes: {df.complex.nunique()}, valid: {df.valid.sum()}, "
          f"censored: {df.censored.sum()}, v1: {df.v1.sum()} rows in {df[df.v1].complex.nunique()} complexes")
    plot_ddg_hist(df)
    plot_rows_per_complex(df)
    plot_ddg_per_complex(df)
    plot_mutations_per_row(df)
    plot_repeats(df)
    print(f"figures saved to {OUT}")


if __name__ == "__main__":
    main()

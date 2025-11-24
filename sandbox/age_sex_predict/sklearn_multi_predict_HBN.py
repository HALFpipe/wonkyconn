import os

###load arrow/21.0.0!!
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("/lustre07/scratch/pbergere/project_predict_sex/usefull_code")

from scikit_pipeline import *

path_pheno = "/lustre07/scratch/pbergere/study-HBN_desc-participants.tsv"

path_project = Path("/lustre07/scratch/pbergere/project_predict_sex/connectomes")

path_full_df = "/lustre07/scratch/pbergere/timeseries_halfpipe/halfpipe_HBN/fmriprep-25.0.0/atlas-schaefercombined"


# --- helpers de chargement ---
def _load_features(path_project, strategy: str, nroi: int):
    df = pd.read_parquet(path_project / f"features_{strategy}_nroi{nroi}_hbn.parquet", engine="fastparquet")
    # garder M/F pour la tâche "sex"
    df = df[df["sex"].isin(["Male", "Female"])].copy()

    corr_cols = [c for c in df.columns if c.startswith("corr")]
    if not corr_cols:
        raise ValueError("Aucune colonne 'corr*' dans le parquet.")

    X = df[corr_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y_age = df["age"].to_numpy()
    y_sex = df["sex"].to_numpy()
    return X, y_age, y_sex


# --- benchmark ---
def run_benchmark_strategies(list_strats, path_project, *, nroi=431, n_splits=10, n_pca=100, n_jobs=4, save_csv=True):
    """
    Boucle sur les stratégies, lance training_pipeline pour âge (ridge) et sexe (logreg),
    et construit deux DataFrames :
      - per_split_df : une ligne par split / métrique / stratégie / cible
      - summary_df   : moyennes et écarts-types par stratégie / métrique / cible
    """
    rows_split = []
    rows_summary = []

    for strat in list_strats:
        # 1) load
        print("loading of strat:", strat)
        X, y_age, y_sex = _load_features(path_project, strat, nroi)

        # 2) age (régression)
        df_scores_age, _summary_age = training_pipeline(X, y_age, n_splits=n_splits, n_pca=n_pca, n_jobs=n_jobs)
        # convertir les noms/valeurs (MAE/RMSE positifs)
        # per-split
        if "neg_root_mean_squared_error" in df_scores_age.columns:
            vals = -df_scores_age["neg_root_mean_squared_error"].to_numpy()
            for i, v in enumerate(vals):
                rows_split.append(dict(strategy=strat, target="age", metric="RMSE", split=i, value=float(v)))
            rows_summary.append(
                dict(
                    strategy=strat,
                    target="age",
                    metric="RMSE",
                    mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)),
                    n_splits=len(vals),
                )
            )
        if "neg_mean_absolute_error" in df_scores_age.columns:
            vals = -df_scores_age["neg_mean_absolute_error"].to_numpy()
            for i, v in enumerate(vals):
                rows_split.append(dict(strategy=strat, target="age", metric="MAE", split=i, value=float(v)))
            rows_summary.append(
                dict(
                    strategy=strat,
                    target="age",
                    metric="MAE",
                    mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)),
                    n_splits=len(vals),
                )
            )
        if "r2" in df_scores_age.columns:
            vals = df_scores_age["r2"].to_numpy()
            for i, v in enumerate(vals):
                rows_split.append(dict(strategy=strat, target="age", metric="R2", split=i, value=float(v)))
            rows_summary.append(
                dict(
                    strategy=strat,
                    target="age",
                    metric="R2",
                    mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)),
                    n_splits=len(vals),
                )
            )

        # 3) SEX (classification)
        df_scores_sex, _summary_sex = training_pipeline(X, y_sex, n_splits=n_splits, n_pca=n_pca, n_jobs=n_jobs)

        sex_metrics = [
            c
            for c in df_scores_sex.columns
            if c in {"accuracy", "roc_auc", "roc_auc_ovr", "f1", "f1_macro", "f1_weighted", "balanced_accuracy"}
        ]
        for m in sex_metrics:
            vals = df_scores_sex[m].to_numpy()
            # nom canonique pour AUC
            m_name = "AUC" if m in {"roc_auc", "roc_auc_ovr"} else m.upper()
            for i, v in enumerate(vals):
                rows_split.append(dict(strategy=strat, target="sex", metric=m_name, split=i, value=float(v)))
            rows_summary.append(
                dict(
                    strategy=strat,
                    target="sex",
                    metric=m_name,
                    mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)),
                    n_splits=len(vals),
                )
            )

    per_split_df = pd.DataFrame(rows_split).sort_values(["strategy", "target", "metric", "split"]).reset_index(drop=True)
    summary_df = pd.DataFrame(rows_summary).sort_values(["strategy", "target", "metric"]).reset_index(drop=True)

    if save_csv:
        per_split_df.to_csv("benchmark_per_split.csv", index=False)
        summary_df.to_csv("benchmark_summary.csv", index=False)

    return per_split_df, summary_df


def plot_one_metric(sub_df, title, ylabel, fname=None, color_map=None, horizontal=False):
    """Trace une seule figure (une métrique pour une cible)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, row in enumerate(sub_df.itertuples()):
        strat = row.strategy
        mean, std = row.mean, row.std
        c = color_map[strat] if color_map is not None else "C0"
        hatch = "//" if "gsr" in strat.lower() else None

        if horizontal:
            ax.barh(strat, mean, xerr=std, capsize=4, color=c, hatch=hatch, edgecolor="black")
        else:
            ax.bar(strat, mean, yerr=std, capsize=4, color=c, hatch=hatch, edgecolor="black")

    if horizontal:
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()  # remet l'ordre du bas vers le haut
    else:
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    plt.tight_layout()

    # show in notebook
    plt.show()

    # save
    if fname is not None:
        plt.savefig(fname, dpi=150)

    plt.close(fig)


def plot_benchmark_bars(summary_df, *, save_dir="fig_benchmark", horizontal=False):
    """Boucle sur toutes les métriques et appelle plot_one_metric."""
    os.makedirs(save_dir, exist_ok=True)

    strategies = summary_df["strategy"].unique()
    colors = plt.colormaps["tab10"].resampled(len(strategies))
    color_map = {s: colors(i) for i, s in enumerate(strategies)}

    for tgt in summary_df["target"].unique():
        for metric in summary_df[summary_df["target"] == tgt]["metric"].unique():
            sub = summary_df[(summary_df["target"] == tgt) & (summary_df["metric"] == metric)]
            if sub.empty:
                continue
            title = f"{tgt.capitalize()} — {metric} (mean ± std)"
            ylabel = metric
            fname = os.path.join(save_dir, f"{tgt}_{metric.lower()}.png")
            plot_one_metric(sub, title, ylabel, fname, color_map, horizontal)


list_strats = [
    "baseline",
    "simple",
    "simple_no_wm_csf",
    "simple+gsr",
    "simple+gsr_no_wm_csf",
    "scrubbing.5",
    "scrubbing.5_no_wm_csf",
    "scrubbing.5+gsr",
    "scrubbing.5+gsr_no_wm_csf",
    "compcor",
    "compcor+gsr",
    "compcor_only",
]


NROI = 431
n_splits = 20
n_pca = 100
n_jobs = 4

per_split_df, summary_df = run_benchmark_strategies(
    list_strats, path_project, nroi=NROI, n_splits=n_splits, n_pca=n_pca, n_jobs=n_jobs
)

# Figures
plot_benchmark_bars(summary_df, save_dir="fig_benchmark", horizontal=True)

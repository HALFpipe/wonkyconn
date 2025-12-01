from functools import partial
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

sns.set_palette("colorblind")

palette = sns.color_palette(n_colors=13)

matplotlib.rcParams["font.family"] = "DejaVu Sans"


# seann: added type for series
def _make_group_label(group_by: list[str], values: str | Sequence[str]) -> str:
    if isinstance(values, str):
        values = [values]

    label: str = ""
    for a, b in zip(group_by, values, strict=True):
        if label:
            label += "\n"
        label += f"{a}-{b}"
    return label


def plot(result_frame: pd.DataFrame, group_by: list[str], output_dir: Path) -> None:
    """
    Plot summary metrics based on the given result data frame.

    Args:
        result_frame (pd.DataFrame): Must contain columns:
            - median_absolute_qcfc
            - percentage_significant_qcfc
            - distance_dependence
            - gcor
            - dmn_similarity
            - dmn_vis_distance_vs_dmn_fpn
            - gradients_similarity
            - confound_regression_percentage
            - motion_scrubbing_percentage
            - nonsteady_states_detector_percentage
            - sex_auc
            - age_mae

            plus the columns listed in group_by (used as index).
        group_by (list[str]): The list of columns that the results are grouped by.
        output_dir (Path): The directory to save the plot image into as "metrics.png".
    """
    # seann: added type for series
    group_labels: "pd.Series[str]" = pd.Series(result_frame.index.map(partial(_make_group_label, group_by)))
    data_frame = result_frame.reset_index()

    figure, axes_array = plt.subplots(
        nrows=1,
        ncols=11,
        figsize=(36, 4),
        constrained_layout=True,
        sharey=True,
        dpi=300,
    )

    (
        median_absolute_qcfc_axes,
        percentage_significant_qcfc_axes,
        distance_dependence_axes,
        gcor_axes,
        dmn_axes,
        modular_dist_axes,
        gradients_axes,
        sex_auc_axes,
        age_mae_axes,
        degrees_of_freedom_loss_axes,
        legend_axes,
    ) = axes_array

    sns.barplot(y=group_labels, x=data_frame.median_absolute_qcfc, color=palette[0], ax=median_absolute_qcfc_axes)
    median_absolute_qcfc_axes.set_title("Median absolute value of QC-FC correlations")
    median_absolute_qcfc_axes.set_xlabel("Median absolute value")
    median_absolute_qcfc_axes.set_ylabel("Group")

    sns.barplot(
        y=group_labels, x=data_frame.percentage_significant_qcfc, color=palette[1], ax=percentage_significant_qcfc_axes
    )
    percentage_significant_qcfc_axes.set_title("% significant QC–FC edges")
    percentage_significant_qcfc_axes.set_xlabel("Percentage %")

    sns.barplot(y=group_labels, x=data_frame.distance_dependence, color=palette[2], ax=distance_dependence_axes)
    distance_dependence_axes.set_title("Distance dependence of QC-FC")
    distance_dependence_axes.set_xlabel("Absolute value of Spearman's $\\rho$")

    # seann: GCOR visualization with horizontal bars and SEM whiskers
    sns.barplot(y=group_labels, x=data_frame.gcor, color=palette[3], ax=gcor_axes)
    gcor_axes.set_title("Global correlation (GCOR)")
    gcor_axes.set_xlabel("Mean correlation")

    sns.barplot(y=group_labels, x=data_frame.dmn_similarity, color=palette[4], ax=dmn_axes)
    dmn_axes.set_title("Similarity with DMN")
    dmn_axes.set_xlabel("Mean correlation")

    sns.barplot(y=group_labels, x=data_frame.dmn_vis_distance_vs_dmn_fpn, color=palette[5], ax=modular_dist_axes)
    modular_dist_axes.set_title("Differences between\nDMN-FPN vs DMN-visual")
    modular_dist_axes.set_xlabel("Mean t-vlaue")

    sns.barplot(y=group_labels, x=data_frame.gradients_similarity, color=palette[6], ax=gradients_axes)
    gradients_axes.set_title("Gradient similarity")
    gradients_axes.set_xlabel("Mean similarity (Spearman's $\\rho$)")

    # --- Sex prediction (AUC) with errorbar (std)
    if "sex_auc" in data_frame.columns:
        sex_auc_axes.barh(
            y=group_labels,
            width=data_frame.sex_auc,
            xerr=data_frame.sex_auc_std,
            color=palette[8],
            ecolor="black",
            capsize=3,
        )
        sex_auc_axes.set_title("Sex prediction (AUC)")
        sex_auc_axes.set_xlabel("AUC (ROC)")
    else:
        sex_auc_axes.set_visible(False)

    # --- Age prediction (MAE) with errorbar (std)
    if "age_mae" in data_frame.columns:
        age_mae_axes.barh(
            y=group_labels,
            width=data_frame.age_mae,
            xerr=data_frame.age_mae_std,
            color=palette[9],
            ecolor="black",
            capsize=3,
        )
        age_mae_axes.set_title("Age prediction (MAE)")
        age_mae_axes.set_xlabel("MAE (years)")
    else:
        age_mae_axes.set_visible(False)

    plot_degrees_of_freedom_loss(
        data_frame,
        group_labels,
        degrees_of_freedom_loss_axes,
        legend_axes,
        [palette[10], palette[11], palette[12]],
    )

    figure.savefig(output_dir / "metrics.png")


def plot_degrees_of_freedom_loss(
    result_frame: pd.DataFrame,
    group_labels: "pd.Series[str]",
    degrees_of_freedom_loss_axes: Axes,
    legend_axes: Axes,
    colors: list[str],
) -> None:
    sns.barplot(
        y=group_labels,
        x=result_frame.confound_regression_percentage,
        color=colors[0],
        ax=degrees_of_freedom_loss_axes,
    )
    sns.barplot(
        y=group_labels,
        x=result_frame.motion_scrubbing_percentage,
        color=colors[1],
        ax=degrees_of_freedom_loss_axes,
    )
    sns.barplot(
        y=group_labels,
        x=result_frame.nonsteady_states_detector_percentage,
        color=colors[2],
        ax=degrees_of_freedom_loss_axes,
    )

    degrees_of_freedom_loss_axes.set_title("Percentage of DoF lost")
    degrees_of_freedom_loss_axes.set_xlabel("Percentage %")
    labels = ["Confounds regression", "Motion scrubbing", "Non-steady states detector"]
    handles = [mpatches.Patch(color=c, label=label) for c, label in zip(colors, labels, strict=False)]
    legend_axes.legend(handles=handles)
    legend_axes.axis("off")

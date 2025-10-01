from __future__ import annotations  # seann: added future import for annotations to allow type hints in function signatures
from functools import partial
from pathlib import Path
import matplotlib
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_palette("colorblind")
palette = sns.color_palette(n_colors=7)

matplotlib.rcParams["font.family"] = "DejaVu Sans"


# seann: added type for series
def _make_group_label(group_by: list[str], values: "pd.Series[str]") -> str:
    label: str = ""
    for a, b in zip(group_by, values, strict=True):
        if label:
            label += "\n"
        label += f"{a}-{b}"
    return label


def plot(result_frame: pd.DataFrame, group_by: list[str], output_dir: Path) -> None:
    """
    Plot all three metrics based on the given result data frame.

    Args:
        result_frame (pd.DataFrame): The DataFrame containing the the columns "median_absolute_qcfc",
            "percentage_significant_qcfc", "distance_dependence", "confound_regression_percentage",
            "motion_scrubbing_percentage", and "nonsteady_states_detector_percentage", and the
            columns in the `group_by` variable.
        group_by (list[str]): The list of columns that the results are grouped by.
        output_dir (Path): The directory to save the plot image into as "metrics.png".

    Returns:
        None
    """
    # seann: added type for series
    group_labels: "pd.Series[str]" = pd.Series(result_frame.index.map(partial(_make_group_label, group_by)))
    data_frame = result_frame.reset_index()

    figure, axes_array = plt.subplots(
        nrows=1,
        ncols=6,
        figsize=(26, 4),
        constrained_layout=True,
        sharey=True,
    )

    (
        median_absolute_qcfc_axes,
        percentage_significant_qcfc_axes,
        distance_dependence_axes,
        gcor_axes,
        degrees_of_freedom_loss_axes,
        legend_axes,
    ) = axes_array

    sns.barplot(
        y=group_labels,
        x=data_frame.median_absolute_qcfc,
        color=palette[0],
        ax=median_absolute_qcfc_axes,
    )
    median_absolute_qcfc_axes.set_title("Median absolute value of QC-FC correlations")
    median_absolute_qcfc_axes.set_xlabel("Median absolute value")
    median_absolute_qcfc_axes.set_ylabel("Group")

    sns.barplot(
        y=group_labels,
        x=data_frame.percentage_significant_qcfc,
        color=palette[1],
        ax=percentage_significant_qcfc_axes,
    )
    percentage_significant_qcfc_axes.set_title("Percentage of significant QC-FC correlations")
    percentage_significant_qcfc_axes.set_xlabel("Percentage %")

    sns.barplot(
        y=group_labels,
        x=data_frame.distance_dependence,
        color=palette[2],
        ax=distance_dependence_axes,
    )
    distance_dependence_axes.set_title("Distance dependence of QC-FC")
    distance_dependence_axes.set_xlabel("Absolute value of Spearman's $\\rho$")

    # seann: GCOR visualization with horizontal bars and SEM whiskers
    gcor_axes.barh(group_labels, data_frame.gcor_mean, color=palette[3])
    gcor_axes.set_title("Global correlation (GCOR)")
    gcor_axes.set_xlabel("Mean correlation")
    gcor_axes.errorbar(
        data_frame.gcor_mean,
        group_labels,
        xerr=data_frame.gcor_sem,
        fmt="none",
        ecolor="black",
        capsize=3,
    )

    plot_degrees_of_freedom_loss(
        data_frame,
        group_labels,
        degrees_of_freedom_loss_axes,
        legend_axes,
        [palette[4], palette[5], palette[6]],
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
    degrees_of_freedom_loss_axes.set_title("Percentage of degrees of freedom lost")
    degrees_of_freedom_loss_axes.set_xlabel("Percentage %")
    labels = [
        "Confounds regression",
        "Motion scrubbing",
        "Non-steady states detector",
    ]
    handles = [mpatches.Patch(color=c, label=label) for c, label in zip(colors, labels)]
    legend_axes.legend(handles=handles)
    legend_axes.axis("off")

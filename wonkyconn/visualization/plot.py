from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

sns.set_palette("colorblind")

palette = sns.color_palette(n_colors=13)

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def plot(records: list[dict], group_by: list[str], output_dir: Path) -> None:
    """
    Plot summary metrics based on the given result data frame.

    Args:
        records (list[dict]): Must contain keys:
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
    # separate dmn similarity from the rest of the metrics
    dmn_sim_array = []
    for record in records:
        df_dmn_similarity = record.pop("dmn_similarity")
        df_dmn_similarity["atlas"] = record["atlas"]
        df_dmn_similarity["feature"] = record["feature"]
        dmn_sim_array.append(df_dmn_similarity[["corr_with_dmn", "atlas", "feature"]])

    df_dmn_sim_array = pd.concat(dmn_sim_array, ignore_index=True)
    df_dmn_sim_array["group_labels"] = df_dmn_sim_array["atlas"] + "-" + df_dmn_sim_array["feature"]

    # summarize the info
    for record, dmn_sim in zip(records, dmn_sim_array, strict=True):
        record["dmn_similarity"] = dmn_sim["corr_with_dmn"].mean()
    result_frame = pd.DataFrame.from_records(records, index=group_by)
    data_frame = result_frame.reset_index()
    data_frame["group_labels"] = data_frame["atlas"] + "-" + data_frame["feature"]

    figure, axes_array = plt.subplots(
        nrows=2,
        ncols=6,
        figsize=(27, 9),
        constrained_layout=True,
        sharey=True,
        dpi=300,
    )

    motion_axes, insight_axes = axes_array
    (
        median_absolute_qcfc_axes,
        percentage_significant_qcfc_axes,
        distance_dependence_axes,
        gcor_axes,
        _,
        legend_axes,
    ) = motion_axes
    (
        dmn_mean_axes,
        modular_dist_axes,
        gradients_axes,
        sex_auc_axes,
        age_mae_axes,
        degrees_of_freedom_loss_axes,
    ) = insight_axes

    sns.barplot(data=data_frame, y="group_labels", x="median_absolute_qcfc", color=palette[0], ax=median_absolute_qcfc_axes)
    median_absolute_qcfc_axes.set_title("Median absolute value of QC-FC correlations")
    median_absolute_qcfc_axes.set_xlabel("Median absolute value")
    median_absolute_qcfc_axes.set_ylabel("Group")

    sns.barplot(
        data=data_frame,
        y="group_labels",
        x="percentage_significant_qcfc",
        color=palette[1],
        ax=percentage_significant_qcfc_axes,
    )
    percentage_significant_qcfc_axes.set_title("% significant QC–FC edges")
    percentage_significant_qcfc_axes.set_xlabel("Percentage %")

    sns.barplot(data=data_frame, y="group_labels", x="distance_dependence", color=palette[2], ax=distance_dependence_axes)
    distance_dependence_axes.set_title("Distance dependence of QC-FC")
    distance_dependence_axes.set_xlabel("Absolute value of Spearman's $\\rho$")

    # seann: GCOR visualization with horizontal bars and SEM whiskers
    sns.barplot(data=data_frame, y="group_labels", x="gcor", color=palette[3], ax=gcor_axes)
    gcor_axes.set_title("Global correlation (GCOR)")
    gcor_axes.set_xlabel("Mean correlation")

    sns.barplot(data=df_dmn_sim_array, y="group_labels", x="corr_with_dmn", color=palette[4], ax=dmn_mean_axes, errorbar="sd")
    dmn_mean_axes.set_title("Similarity with DMN")
    dmn_mean_axes.set_xlabel("Mean correlation")

    sns.barplot(data=data_frame, y="group_labels", x="dmn_vis_distance_vs_dmn_fpn", color=palette[5], ax=modular_dist_axes)
    modular_dist_axes.set_title("Differences between\nDMN-FPN vs DMN-visual")
    modular_dist_axes.set_xlabel("Mean t-value")

    sns.barplot(data=data_frame, y="group_labels", x="gradients_similarity", color=palette[7], ax=gradients_axes)
    gradients_axes.set_title("Gradient similarity")
    gradients_axes.set_xlabel("Mean similarity (Spearman's $\\rho$)")

    # --- Sex prediction (AUC) with errorbar (std)
    if "sex_auc" in data_frame.columns:
        sex_auc_axes.barh(
            y=data_frame.group_labels,
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
            y=data_frame.group_labels,
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
        degrees_of_freedom_loss_axes,
        legend_axes,
        [palette[10], palette[11], palette[12]],
    )

    figure.savefig(output_dir / "metrics.png")


def plot_degrees_of_freedom_loss(
    result_frame: pd.DataFrame,
    degrees_of_freedom_loss_axes: Axes,
    legend_axes: Axes,
    colors: list[str],
) -> None:
    sns.barplot(
        data=result_frame,
        y="group_labels",
        x="confound_regression_percentage",
        color=colors[0],
        ax=degrees_of_freedom_loss_axes,
    )
    sns.barplot(
        data=result_frame,
        y="group_labels",
        x="motion_scrubbing_percentage",
        color=colors[1],
        ax=degrees_of_freedom_loss_axes,
    )
    sns.barplot(
        data=result_frame,
        y="group_labels",
        x="nonsteady_states_detector_percentage",
        color=colors[2],
        ax=degrees_of_freedom_loss_axes,
    )

    degrees_of_freedom_loss_axes.set_title("Percentage of DoF lost")
    degrees_of_freedom_loss_axes.set_xlabel("Percentage %")
    labels = ["Confounds regression", "Motion scrubbing", "Non-steady states detector"]
    handles = [mpatches.Patch(color=c, label=label) for c, label in zip(colors, labels, strict=False)]
    legend_axes.legend(handles=handles)
    legend_axes.axis("off")

"""Network similarity measures."""

from __future__ import (
    annotations,
)

from typing import Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy import stats

from ..base import ConnectivityMatrix


def single_subject_within_network_connectivity(
    connectivity_matrix: ConnectivityMatrix,
    region_membership: pd.DataFrame,
    yeo_network_index: int = 7,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64]:
    """_summary_

    Args:
        connectivity_matrix (ConnectivityMatrix): Functional connectivity matrix.
        region_membership (pd.DataFrame): Atlas parcel correspondence to Yeo 7 network. Generate throguh class Atlas.
        yeo_network_index (int, optional): Yeo network for similarity calculation. Defaults to 7.

    Returns:
        npt.NDArray[np.float64]: Mean of functional connectivity value within each network.
        npt.NDArray[np.float64]: Standard deviation of functional connectivity value within each network.
        np.float64: Average pearson's correlation of the connectivity pattern with the selected network.
    """

    yeo_network_labels = range(region_membership.shape[1])
    # if it's a high res atlas, get roi that only belong to one given network
    roi_index_single_network_membership = region_membership.index[region_membership.T.sum() == 1]
    roi_index_single_and_dmn = roi_index_single_network_membership[
        region_membership.loc[roi_index_single_network_membership, f"yeo7-{yeo_network_index}"] == 1
    ]
    roi_index = roi_index_single_and_dmn - 1  # convert to zero based index
    if roi_index.empty:  # low res atlas will give you empty list
        # use the original membership instead
        roi_index = region_membership.index[region_membership[f"yeo7-{yeo_network_index}"] == 1]
        roi_index -= 1

    # calculate similarity of the thresholded individual level seed based connectivity with the binary mask
    subj_average_connectivity_within_network = []
    subj_variance_connectivity_within_network = []
    subj_corr_wtih_network = []
    for idx_roi in roi_index:
        seed_based_map = connectivity_matrix.load()[int(idx_roi), :]

        # isolate the parcal per network
        isolate_parcel = [
            region_membership[f"yeo7-{int(n + 1)}"].to_numpy(dtype=np.float32) * seed_based_map for n in yeo_network_labels
        ]
        networks = np.asarray(isolate_parcel)
        networks[networks == 0] = np.nan
        mean_within_network_connection = np.nanmean(networks, axis=1)
        std_within_network_connection = np.nanstd(networks, axis=1)

        # Remove rows with NaN values
        mask = np.array(
            [
                np.all(np.isfinite(row))
                for row in np.column_stack((seed_based_map, region_membership[f"yeo7-{yeo_network_index}"].to_numpy()))
            ]
        )

        correlation_with_given_network = np.corrcoef(
            seed_based_map[mask], region_membership[f"yeo7-{yeo_network_index}"].to_numpy()[mask]
        )

        subj_average_connectivity_within_network.append(mean_within_network_connection)
        subj_variance_connectivity_within_network.append(std_within_network_connection)
        subj_corr_wtih_network.append(correlation_with_given_network[1, 0])

    # summarise of the given subject
    subj_average_connectivity_within_network = np.asarray(subj_average_connectivity_within_network).mean(axis=0)
    subj_variance_connectivity_within_network = np.asarray(subj_variance_connectivity_within_network).mean(axis=0)
    return (
        subj_average_connectivity_within_network,
        subj_variance_connectivity_within_network,
        np.nanmean(subj_corr_wtih_network),
    )


def network_similarity(
    connectivity_matrices: list[ConnectivityMatrix], region_membership: pd.DataFrame
) -> Tuple[np.float64, np.float64]:
    """Calculate network similarity of of default mode network recovered through functional connectivity and Yeo's template.

    Args:
        connectivity_matrices (list[ConnectivityMatrix]): A collection of functional connectivity matrices of the same
            atlas and denoising method.
        region_membership (pd.DataFrame): Atlas parcel correspondence to Yeo 7 network. Generate throguh the Atlas class.

    Returns:
        Tuple[np.float64, np.float64]: Group level statistics of average correlation with the default mode network and
            t-statistics of DMN-FPN distance vs DMN-VIS distance.
    """
    average_connectivity_within_network, std_connectivity_within_network, corr_wtih_dmn = [], [], []
    for cm in connectivity_matrices:
        mean, std, corr = single_subject_within_network_connectivity(cm, region_membership, yeo_network_index=7)
        average_connectivity_within_network.append(mean)
        std_connectivity_within_network.append(std)
        corr_wtih_dmn.append(corr)

    df_average = pd.DataFrame(
        np.asarray(average_connectivity_within_network), columns=[f"mean_{r}" for r in region_membership.columns]
    )
    df_std = pd.DataFrame(np.asarray(std_connectivity_within_network), columns=[f"sd_{r}" for r in region_membership.columns])
    corr_dmn = pd.DataFrame(np.asarray(corr_wtih_dmn), columns=["corr_with_dmn"])
    summary = pd.concat([df_average, df_std, corr_dmn], axis=1)

    # calculate distance
    summary["mean-diff_dmn_visual"] = summary["mean_yeo7-7"] - summary["mean_yeo7-1"]
    summary["mean-diff_dmn_fpn"] = summary["mean_yeo7-7"] - summary["mean_yeo7-6"]

    mean_corr_with_dmn = summary["corr_with_dmn"].mean()
    t_stats_dmn_vis_fpn, _ = stats.ttest_rel(summary["mean-diff_dmn_visual"], summary["mean-diff_dmn_fpn"])
    return np.float64(mean_corr_with_dmn), np.float64(t_stats_dmn_vis_fpn)

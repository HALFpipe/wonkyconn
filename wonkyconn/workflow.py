"""
Process fMRIPrep outputs to timeseries based on denoising strategy.
"""

import argparse
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm

from .atlas import Atlas
from .base import ConnectivityMatrix
from .features.age_sex_prediction import age_sex_scores
from .features.calculate_degrees_of_freedom import (
    calculate_degrees_of_freedom_loss,
)
from .features.calculate_gradients_correlation import calculate_gradients_similarity, extract_gradients
from .features.distance_dependence import calculate_distance_dependence
from .features.gcor import calculate_gcor
from .features.network import network_similarity
from .features.quality_control_connectivity import (
    calculate_median_absolute,
    calculate_qcfc,
    calculate_qcfc_percentage,
)
from .file_index.bids import BIDSIndex
from .logger import gc_log, set_verbosity
from .visualization.plot import plot


def is_halfpipe(index: BIDSIndex) -> bool:
    for path in index.tags_by_paths.keys():
        try:
            derivatives_index = path.parts.index("derivatives")
        except ValueError:
            continue
        subdirectory = path.parts[derivatives_index + 1]
        if subdirectory == "halfpipe":
            return True
    return False


def workflow(args: argparse.Namespace) -> None:
    if "pytest" not in sys.modules:
        set_verbosity(args.verbosity)
    gc_log.debug(vars(args))

    # Check BIDS path
    bids_dir = args.bids_dir
    index = BIDSIndex()
    index.put(bids_dir)

    # BEP017 by default
    seg_key = "seg"
    group_by: list[str] = [seg_key]
    metric_key = "MeanFramewiseDisplacement"
    relmat_base_query = dict(suffix="relmat")
    has_header = True
    if is_halfpipe(index):
        seg_key = "atlas"
        group_by = ["feature", "atlas"]
        metric_key = "FDMean"
        relmat_base_query = dict(desc="correlation", suffix="matrix")
        has_header = False

    # Check output path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data frame (participants: age, gender, etc.)
    data_frame = load_data_frame(args)

    # Load atlases
    atlases: dict[str, Atlas] = {name: Atlas.create(name, Path(atlas_path_str)) for name, atlas_path_str in args.atlas}
    # seann: Add debugging to see what the atlas dictionary contains
    gc_log.debug(f"Atlas dictionary contains: {list(atlases.keys())}")

    Group = namedtuple("Group", group_by)  # type: ignore[misc]

    grouped_connectivity_matrix: defaultdict[tuple[str, ...], list[ConnectivityMatrix]] = defaultdict(list)

    segs: set[str] = set()
    for timeseries_path in index.get(suffix="timeseries", extension=".tsv"):
        query = dict(**index.get_tags(timeseries_path))
        del query["suffix"]

        metadata = index.get_metadata(timeseries_path)
        if not metadata:
            gc_log.warning(f"Skipping {timeseries_path} due to missing metadata")
            continue

        if "NumberOfVolumes" not in metadata:
            with timeseries_path.open("r") as file_handle:
                line_count = sum(1 for _ in file_handle)
            if has_header:
                line_count -= 1
            metadata["NumberOfVolumes"] = line_count

        relmat_query = query | relmat_base_query | dict(extension=".tsv")
        for relmat_path in index.get(**relmat_query):
            group = Group(*(index.get_tag_value(relmat_path, key) for key in group_by))
            gc_log.debug(f"Processing group {group} with file {relmat_path}")
            connectivity_matrix = ConnectivityMatrix(relmat_path, metadata, has_header=has_header)
            grouped_connectivity_matrix[group].append(connectivity_matrix)
            seg = index.get_tag_value(relmat_path, seg_key)
            if seg is None:
                raise ValueError(f'Connectivity matrix "{relmat_path}" does not have key "{seg_key}"')
            segs.add(seg)

    if not grouped_connectivity_matrix:
        raise ValueError("No groups found")

    distance_matrices: dict[str, npt.NDArray[np.float64]] = {seg: atlases[seg].get_distance_matrix() for seg in segs}
    region_memberships: dict[str, pd.DataFrame] = {seg: atlases[seg].get_yeo7_membership() for seg in segs}

    records: list[dict[str, Any]] = list()
    for key, connectivity_matrices in tqdm(grouped_connectivity_matrix.items(), unit="groups"):
        record = make_record(
            index,
            data_frame,
            connectivity_matrices,
            distance_matrices,
            region_memberships,
            metric_key,
            seg_key,
            atlases,
        )
        record.update(dict(zip(group_by, key, strict=False)))
        records.append(record)

    result_frame = pd.DataFrame.from_records(records, index=group_by)
    result_frame.to_csv(output_dir / "metrics.tsv", sep="\t")

    plot(result_frame, group_by, output_dir)


def make_record(
    index: BIDSIndex,
    data_frame: pd.DataFrame,
    connectivity_matrices: list[ConnectivityMatrix],
    distance_matrices: dict[str, npt.NDArray[np.float64]],
    region_memberships: dict[str, pd.DataFrame],
    metric_key: str,
    seg_key: str,
    atlases: dict[str, Atlas],
) -> dict[str, Any]:
    # seann: added sub- tag when looking up subjects only if sub- is not already present
    seg_subjects: list[str] = list()
    for c in connectivity_matrices:
        sub = index.get_tag_value(c.path, "sub")

        if sub is None:
            raise ValueError(f'Connectivity matrix "{c.path}" does not have a subject tag')

        if sub in data_frame.index:
            seg_subjects.append(sub)
            continue

        sub = f"sub-{sub}"
        if sub in data_frame.index:
            seg_subjects.append(sub)
            continue

        raise ValueError(f"Subject {sub} not found in participants file")

    # Slice phenotypes (age, gender, etc.) for just this group
    seg_data_frame = data_frame.loc[seg_subjects]
    qcfc = calculate_qcfc(seg_data_frame, connectivity_matrices, metric_key)

    (seg,) = index.get_tag_values(seg_key, {c.path for c in connectivity_matrices})
    distance_matrix = distance_matrices[seg]

    # seann: compute group-level GCOR statistics (mean and SEM)
    gcor = calculate_gcor(connectivity_matrices)

    dmn_similarity, t_stats_dmn_vis_fpn = network_similarity(connectivity_matrices, region_memberships[seg])
    atlas = atlases[seg].image
    gradients, gradients_group = extract_gradients(connectivity_matrices, atlas)

    record = dict(
        median_absolute_qcfc=calculate_median_absolute(qcfc.correlation),
        percentage_significant_qcfc=calculate_qcfc_percentage(qcfc),
        distance_dependence=calculate_distance_dependence(qcfc, distance_matrix),
        gcor=gcor,
        dmn_similarity=dmn_similarity,
        dmn_vis_distance_vs_dmn_fpn=t_stats_dmn_vis_fpn,
        gradients_similarity=calculate_gradients_similarity(gradients, gradients_group),
        **calculate_degrees_of_freedom_loss(connectivity_matrices)._asdict(),
    )

    # age / sex predictability metrics
    try:
        ages = seg_data_frame["age"].to_numpy()
        genders = seg_data_frame["gender"].to_numpy()

        scores = age_sex_scores(
            connectivity_matrices,
            ages=ages,
            genders=genders,
            n_splits=100,
            random_state=42,
            n_pca=100,
            n_jobs=4,
            clf_model="logreg",  # logistic regression for sex
            reg_model="ridge",  # ridge regression for age
        )

        # scores is:
        # {
        #   "sex_auc": float,
        #   "sex_auc_std": float,
        #   "sex_accuracy": float,
        #   "age_mae": float,
        #   "age_mae_std": float,
        #   "age_r2": float,
        # }
        record.update(scores)

    except Exception as exc:
        gc_log.warning(f"[age_sex_prediction] Skipping age/sex prediction for this group due to error: {exc!r}")
        # If it fails, we still want consistent columns in the output.
        record.update(
            dict(
                sex_auc=np.nan,
                sex_auc_std=np.nan,
                sex_accuracy=np.nan,
                age_mae=np.nan,
                age_mae_std=np.nan,
                age_r2=np.nan,
            )
        )

    return record


def load_data_frame(args: argparse.Namespace) -> pd.DataFrame:
    data_frame = pd.read_csv(
        args.phenotypes,
        sep="\t",
        index_col="participant_id",
        dtype={"participant_id": str},
    )
    if "gender" not in data_frame.columns:
        raise ValueError('Phenotypes file is missing the "gender" column')
    if "age" not in data_frame.columns:
        raise ValueError('Phenotypes file is missing the "age" column')
    return data_frame

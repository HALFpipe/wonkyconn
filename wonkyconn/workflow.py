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
from .logger import logger, set_verbosity
from .visualization.plot import plot

_DEFAULT_N_SPLITS = 20
_DEFAULT_N_PCA = 100
_DEFAULT_N_JOBS = 4


def is_halfpipe(index: BIDSIndex) -> bool:
    """Check whether the indexed dataset was produced by HALFpipe."""
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
    """Run the group-level connectivity quality-control pipeline."""
    if "pytest" not in sys.modules:
        set_verbosity(args.verbosity)
    logger.debug(vars(args))

    # check if light mode is enabled - if so, it will not run the age and sex prediction and gradient similarity
    disable_prediction_gradient = getattr(args, "light_mode", False)

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
    logger.debug(f"Atlas dictionary contains: {list(atlases.keys())}")

    Group = namedtuple("Group", group_by)  # type: ignore[misc]

    grouped_connectivity_matrix: defaultdict[tuple[str, ...], list[ConnectivityMatrix]] = defaultdict(list)

    segs: set[str] = set()
    for timeseries_path in index.get(suffix="timeseries", extension=".tsv"):
        query = dict(**index.get_tags(timeseries_path))
        del query["suffix"]

        metadata = index.get_metadata(timeseries_path)
        if not metadata:
            logger.warning(f"Skipping {timeseries_path} due to missing metadata")
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
            logger.debug(f"Processing group {group} with file {relmat_path}")
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
            disable_prediction_gradient,
        )
        record.update(dict(zip(group_by, key, strict=False)))
        if len(group_by) == 2:
            record["dmn_similarity"].to_csv(output_dir / f"dmn_similarity_{'-'.join(group_by)}.tsv", sep="\t")
        else:
            record["dmn_similarity"].to_csv(output_dir / f"dmn_similarity_{group_by[0]}.tsv", sep="\t")

        dmn_similarity_std = record["dmn_similarity"].loc[:, "corr_with_dmn"].std()
        dmn_similarity_avg = record["dmn_similarity"].loc[:, "corr_with_dmn"].mean()
        record["dmn_similarity_std"] = dmn_similarity_std
        record["dmn_similarity_mean"] = dmn_similarity_avg

        records.append(record)

    plot(records, group_by, output_dir)

    for record in records:
        record.pop("dmn_similarity")

    result_frame = pd.DataFrame.from_records(records, index=group_by)
    result_frame.to_csv(output_dir / "metrics.tsv", sep="\t")


def make_record(
    index: BIDSIndex,
    data_frame: pd.DataFrame,
    connectivity_matrices: list[ConnectivityMatrix],
    distance_matrices: dict[str, npt.NDArray[np.float64]],
    region_memberships: dict[str, pd.DataFrame],
    metric_key: str,
    seg_key: str,
    atlases: dict[str, Atlas],
    disable_prediction_gradient: bool,
) -> dict[str, Any]:
    """Compute all QC metrics for a single group of connectivity matrices."""
    seg_subjects: list[str] = list()
    filtered: list[ConnectivityMatrix] = list()

    for c in connectivity_matrices:
        sub = index.get_tag_value(c.path, "sub")

        if sub is None:
            raise ValueError(f'Connectivity matrix "{c.path}" does not have a subject tag')

        # Try both formats
        candidates = [sub, f"sub-{sub}"]
        found = next((s for s in candidates if s in data_frame.index), None)

        if found:
            seg_subjects.append(found)
            filtered.append(c)
        else:
            logger.info(f"Skipping subject {sub}: not found in phenotype file.")

    #  Renaming for consistency
    connectivity_matrices[:] = filtered

    # Slice phenotypes (age, gender, etc.) for just this group
    seg_data_frame = data_frame.loc[seg_subjects]
    qcfc = calculate_qcfc(seg_data_frame, connectivity_matrices, metric_key)

    (seg,) = index.get_tag_values(seg_key, {c.path for c in connectivity_matrices})
    distance_matrix = distance_matrices[seg]

    gcor = calculate_gcor(connectivity_matrices)

    dmn_similarity_summary, t_stats_dmn_vis_fpn = network_similarity(connectivity_matrices, region_memberships[seg])
    atlas = atlases[seg].image

    record = dict(
        median_absolute_qcfc=calculate_median_absolute(qcfc.correlation),
        percentage_significant_qcfc=calculate_qcfc_percentage(qcfc),
        distance_dependence=calculate_distance_dependence(qcfc, distance_matrix),
        gcor=gcor,
        dmn_similarity=dmn_similarity_summary,
        dmn_vis_distance_vs_dmn_fpn=t_stats_dmn_vis_fpn,
        **calculate_degrees_of_freedom_loss(connectivity_matrices)._asdict(),
    )

    if disable_prediction_gradient:
        logger.info("Light mode enabled - skipping age and sex prediction, gradient similarity.")
        record.update(
            dict(
                sex_auc=np.nan,
                sex_auc_ci_lower=np.nan,
                sex_auc_ci_upper=np.nan,
                sex_accuracy=np.nan,
                age_mae=np.nan,
                age_mae_ci_lower=np.nan,
                age_mae_ci_upper=np.nan,
                age_r2=np.nan,
                gradients_similarity=np.nan,
            )
        )  # place holders
        return record
    # Gradient similarity
    gradients, gradients_group = extract_gradients(connectivity_matrices, atlas)
    record["gradients_similarity"] = calculate_gradients_similarity(gradients, gradients_group)

    # age / sex predictability metrics
    try:
        ages = seg_data_frame["age"].to_numpy()
        genders = seg_data_frame["gender"].to_numpy()

        scores = age_sex_scores(
            connectivity_matrices,
            ages=ages,
            genders=genders,
            n_splits=_DEFAULT_N_SPLITS,
            random_state=42,
            n_pca=_DEFAULT_N_PCA,
            n_jobs=_DEFAULT_N_JOBS,
        )

        # scores is:
        # {
        #   "sex_auc": float,
        #   "sex_auc_ci_lower": float,
        #   "sex_auc_ci_upper": float,
        #   "sex_accuracy": float,
        #   "age_mae": float,
        #   "age_mae_ci_lower": float,
        #   "age_mae_ci_upper": float,
        #   "age_r2": float,
        # }
        record.update(scores)

    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning(f"[age_sex_prediction] Skipping age/sex prediction for this group due to error: {exc!r}")
        # If it fails, we still want consistent columns in the output.
        record.update(
            dict(
                sex_auc=np.nan,
                sex_auc_ci_lower=np.nan,
                sex_auc_ci_upper=np.nan,
                sex_accuracy=np.nan,
                age_mae=np.nan,
                age_mae_ci_lower=np.nan,
                age_mae_ci_upper=np.nan,
                age_r2=np.nan,
            )
        )

    return record


def load_data_frame(args: argparse.Namespace) -> pd.DataFrame:
    """Load a phenotype TSV with ``participant_id``, ``gender``, and ``age`` columns."""
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

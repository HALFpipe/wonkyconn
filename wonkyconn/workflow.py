"""
Process fMRIPrep outputs to timeseries based on denoising strategy.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .atlas import Atlas
from .base import ConnectivityMatrix
from .features.calculate_degrees_of_freedom import (
    calculate_degrees_of_freedom_loss,
)
from .features.distance_dependence import calculate_distance_dependence
from .features.gcor import calculate_gcor
from .features.quality_control_connectivity import (
    calculate_median_absolute,
    calculate_qcfc,
    calculate_qcfc_percentage,
)
from .file_index.bids import BIDSIndex
from .logger import gc_log, set_verbosity
from .visualization.plot import plot


def workflow(args: argparse.Namespace) -> None:
    set_verbosity(args.verbosity)
    gc_log.info(vars(args))

    # Check BIDS path
    bids_dir = args.bids_dir
    index = BIDSIndex()
    index.put(bids_dir)

    # Check output path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data frame
    data_frame = load_data_frame(args)

    # Fix --atlas: Load atlas (only 1 atlas is supported for now)
    label, atlas_path_str = args.atlas
    seg_to_atlas: dict[str, Atlas] = {label: Atlas.create(label, Path(atlas_path_str))}
    specified_atlas = label
    gc_log.info(f"Will only process matrices for atlas: {specified_atlas}")

    # Fix: changed from using namedtuple to a dict to avoid type error
    group_by = args.group_by

    grouped_connectivity_matrix: defaultdict[tuple[str, ...], list[ConnectivityMatrix]] = defaultdict(list)

    for timeseries_path in index.get(suffix="timeseries", extension=".tsv"):
        query = dict(**index.get_tags(timeseries_path))
        del query["suffix"]

        metadata = index.get_metadata(timeseries_path)
        if not metadata:
            gc_log.warning(f"Skipping {timeseries_path} due to missing metadata")
            continue

        # Seann: Filter connectivity matrices by atlas type
        for relmat_path in index.get(suffix="relmat", **query):
            # Filter matrices by atlas type
            matrix_seg = index.get_tag_value(relmat_path, "seg")
            if matrix_seg != specified_atlas:
                gc_log.debug(f"Skipping matrix with atlas {matrix_seg}")
                continue

            # changed from Group to tuple to avoid type error
            group = tuple(index.get_tag_value(relmat_path, key) or "NA" for key in group_by)

            connectivity_matrix = ConnectivityMatrix(relmat_path, metadata)
            grouped_connectivity_matrix[group].append(connectivity_matrix)

    if not grouped_connectivity_matrix:
        raise ValueError("No groups found")

    records: list[dict[str, Any]] = []
    for group, connectivity_matrices in tqdm(grouped_connectivity_matrix.items(), unit="groups"):
        record = make_record(index, data_frame, seg_to_atlas, connectivity_matrices)
        record.update(dict(zip(group_by, group)))
        records.append(record)

    result_frame = pd.DataFrame.from_records(records, index=group_by)
    result_frame.to_csv(output_dir / "metrics.tsv", sep="\t")

    plot(result_frame, group_by, output_dir)


def make_record(
    index: BIDSIndex,
    data_frame: pd.DataFrame,
    seg_to_atlas: dict[str, Atlas],
    connectivity_matrices: list[ConnectivityMatrix],
) -> dict[str, Any]:

    # seann: Add debugging to see what the atlas dictionary contains
    gc_log.info(f"Atlas dictionary contains: {list(seg_to_atlas.keys())}")

    # seann: added sub- tag when looking up subjects only if sub- is not already present
    seg_subjects = []
    for c in connectivity_matrices:
        sub = index.get_tag_value(c.path, "sub")  # returns either "2" or "sub-2"
        if not str(sub).startswith("sub-"):
            sub = f"sub-{sub}"
        seg_subjects.append(sub)

    seg_data_frame = data_frame.loc[seg_subjects]
    qcfc = calculate_qcfc(seg_data_frame, connectivity_matrices)

    (seg,) = index.get_tag_values("seg", {c.path for c in connectivity_matrices})
    atlas = seg_to_atlas[seg]

    # seann: compute group-level GCOR statistics (mean and SEM)
    gcor = calculate_gcor(connectivity_matrices)

    record = dict(
        median_absolute_qcfc=calculate_median_absolute(qcfc.correlation),
        percentage_significant_qcfc=calculate_qcfc_percentage(qcfc),
        distance_dependence=calculate_distance_dependence(qcfc, atlas),
        gcor_mean=gcor.mean,
        gcor_sem=gcor.sem,
        **calculate_degrees_of_freedom_loss(connectivity_matrices)._asdict(),
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

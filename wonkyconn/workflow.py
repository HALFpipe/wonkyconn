"""
Process fMRIPrep outputs to timeseries based on denoising strategy.
"""

import argparse
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm

from .atlas import Atlas
from .base import ConnectivityMatrix
from .features.calculate_degrees_of_freedom import (
    calculate_degrees_of_freedom_loss,
)
from .features.distance_dependence import calculate_distance_dependence
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
    gc_log.debug(vars(args))

    # Check BIDS path
    bids_dir = args.bids_dir
    index = BIDSIndex()
    index.put(bids_dir)

    # Check output path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data frame
    data_frame = load_data_frame(args)

    # Load atlases
    seg_to_atlas: dict[str, Atlas] = {
        seg: Atlas.create(seg, Path(atlas_path_str))
        for seg, atlas_path_str in args.seg_to_atlas
    }
    # seann: Add debugging to see what the atlas dictionary contains
    gc_log.debug(f"Atlas dictionary contains: {list(seg_to_atlas.keys())}")

    group_by: list[str] = args.group_by
    Group = namedtuple("Group", group_by)  # type: ignore[misc]

    grouped_connectivity_matrix: defaultdict[
        tuple[str, ...], list[ConnectivityMatrix]
    ] = defaultdict(list)

    segs: set[str] = set()
    for timeseries_path in index.get(suffix="timeseries", extension=".tsv"):
        query = dict(**index.get_tags(timeseries_path))
        del query["suffix"]

        metadata = index.get_metadata(timeseries_path)
        if not metadata:
            gc_log.warning(f"Skipping {timeseries_path} due to missing metadata")
            continue

        relmat_query = query | dict(desc="correlation", suffix="matrix")
        for relmat_path in index.get(**relmat_query):
            group = Group(*(index.get_tag_value(relmat_path, key) for key in group_by))
            connectivity_matrix = ConnectivityMatrix(relmat_path, metadata)
            grouped_connectivity_matrix[group].append(connectivity_matrix)
            seg = index.get_tag_value(relmat_path, args.seg_key)
            if seg is None:
                raise ValueError(
                    f'Connectivity matrix "{relmat_path}" does not have key "{args.seg_key}"'
                )
            segs.add(seg)

    if not grouped_connectivity_matrix:
        raise ValueError("No groups found")

    distance_matrices: dict[str, npt.NDArray[np.float64]] = {
        seg: seg_to_atlas[seg].get_distance_matrix() for seg in segs
    }

    records: list[dict[str, Any]] = list()
    for key, connectivity_matrices in tqdm(
        grouped_connectivity_matrix.items(), unit="groups"
    ):
        record = make_record(
            index, data_frame, connectivity_matrices, distance_matrices, args
        )
        record.update(dict(zip(group_by, key)))
        records.append(record)

    result_frame = pd.DataFrame.from_records(records, index=group_by)
    result_frame.to_csv(output_dir / "metrics.tsv", sep="\t")

    plot(result_frame, group_by, output_dir)


def make_record(
    index: BIDSIndex,
    data_frame: pd.DataFrame,
    connectivity_matrices: list[ConnectivityMatrix],
    distance_matrices: dict[str, npt.NDArray[np.float64]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    # seann: added sub- tag when looking up subjects only if sub- is not already present
    seg_subjects: list[str] = list()
    for c in connectivity_matrices:
        sub = index.get_tag_value(c.path, "sub")

        if sub is None:
            raise ValueError(
                f'Connectivity matrix "{c.path}" does not have a subject tag'
            )

        if sub in data_frame.index:
            seg_subjects.append(sub)
            continue

        sub = f"sub-{sub}"
        if sub in data_frame.index:
            seg_subjects.append(sub)
            continue

        raise ValueError(f"Subject {sub} not found in participants file")

    seg_data_frame = data_frame.loc[seg_subjects]  # type: ignore[index]
    qcfc = calculate_qcfc(seg_data_frame, connectivity_matrices, args.metric_key)

    (seg,) = index.get_tag_values(args.seg_key, {c.path for c in connectivity_matrices})
    distance_matrix = distance_matrices[seg]

    record = dict(
        median_absolute_qcfc=calculate_median_absolute(qcfc.correlation),
        percentage_significant_qcfc=calculate_qcfc_percentage(qcfc),
        distance_dependence=calculate_distance_dependence(qcfc, distance_matrix),
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

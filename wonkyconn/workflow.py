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
from .features.calculate_degrees_of_freedom import calculate_degrees_of_freedom_loss
from .features.distance_dependence import calculate_distance_dependence
from .features.quality_control_connectivity import (
    calculate_median_absolute,
    calculate_qcfc,
    calculate_qcfc_percentage,
)
from .file_index.bids import BIDSIndex
from .logger import gc_log, set_verbosity


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

    # Load atlases
    seg_to_atlas: dict[str, Atlas] = {
        seg: Atlas.create(seg, Path(atlas_path_str))
        for seg, atlas_path_str in args.seg_to_atlas
    }

    seg_to_connectivity_matrices: defaultdict[str, list[ConnectivityMatrix]] = (
        defaultdict(list)
    )
    for timeseries_path in index.get(suffix="timeseries", extension=".tsv"):
        query = dict(**index.get_tags(timeseries_path))
        del query["suffix"]

        metadata = index.get_metadata(timeseries_path)
        if not metadata:
            gc_log.warning(f"Skipping {timeseries_path} due to missing metadata")
            continue

        for relmat_path in index.get(suffix="relmat", **query):
            seg = index.get_tag_value(relmat_path, "seg")
            if seg not in seg_to_atlas:
                gc_log.warning(f"Skipping {relmat_path} due to missing atlas for {seg}")
                continue
            connectivity_matrix = ConnectivityMatrix(relmat_path, metadata)
            seg_to_connectivity_matrices[seg].append(connectivity_matrix)

    if not seg_to_connectivity_matrices:
        raise ValueError("No connectivity matrices found")

    records: list[dict[str, Any]] = []
    for seg, connectivity_matrices in tqdm(
        seg_to_connectivity_matrices.items(), unit="seg"
    ):
        seg_subjects = [
            index.get_tag_value(c.path, "sub") for c in connectivity_matrices
        ]
        seg_data_frame = data_frame.loc[seg_subjects]

        qcfc = calculate_qcfc(seg_data_frame, connectivity_matrices)

        atlas = seg_to_atlas[seg]
        record = dict(
            seg=seg,
            median_absolute_qcfc=calculate_median_absolute(qcfc.correlation),
            percentage_significant_qcfc=calculate_qcfc_percentage(qcfc),
            distance_dependence=calculate_distance_dependence(qcfc, atlas),
            degrees_of_freedom_loss=calculate_degrees_of_freedom_loss(
                connectivity_matrices
            ),
        )
        records.append(record)

    result_frame = pd.DataFrame.from_records(records)
    result_frame.to_csv(output_dir / "metrics.tsv", sep="\t", index=False)

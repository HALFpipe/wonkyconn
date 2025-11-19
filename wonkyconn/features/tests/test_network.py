import json
from pathlib import Path

import datalad.api as dl

from wonkyconn.atlas import Atlas
from wonkyconn.base import ConnectivityMatrix
from wonkyconn.features.network import single_subject_within_network_connectivity
from wonkyconn.tests.conftest import data_path as data_path


def test_single_subject_within_network_connectivity(data_path: Path) -> None:  # noqa: F811
    dseg_path = data_path / "atlases" / "atlas-Schaefer2018Combined_dseg.nii.gz"
    relmat_path = (
        data_path
        / "halfpipe/derivatives/halfpipe/"
        / "sub-10171/func/task-rest/"
        / "sub-10171_task-rest_feature-cCompCor_atlas-Schaefer2018Combined_desc-correlation_matrix.tsv"
    )
    metadata_path = (
        data_path
        / "halfpipe/derivatives/halfpipe/"
        / "sub-10171/func/task-rest/sub-10171_task-rest_feature-cCompCor_atlas-Schaefer2018Combined_timeseries.json"
    )
    dl.get(str(dseg_path))
    dl.get(str(relmat_path))
    dl.get(str(metadata_path))
    atlas = Atlas.create("Schaefer2018Combined", dseg_path)
    with metadata_path.open("r") as file:
        metadata = json.load(file)

    connectivity_matrix = ConnectivityMatrix(relmat_path, metadata, has_header=True)
    region_membership = atlas.get_yeo7_membership()

    mean, _, _ = single_subject_within_network_connectivity(connectivity_matrix, region_membership, yeo_network_index=7)
    assert len(mean) == 7

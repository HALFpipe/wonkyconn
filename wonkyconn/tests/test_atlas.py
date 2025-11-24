from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from nilearn.plotting import find_probabilistic_atlas_cut_coords
from templateflow.api import get as get_template

from wonkyconn.atlas import Atlas


def test_dseg_atlas(data_path: Path) -> None:
    # atlas_path = data_path / YEO_NETWORK_MAP
    # dl.get(str(atlas_path))

    url = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/MNI/Centroid_coordinates/"
        "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    )
    _centroids = pd.read_csv(url).loc[:, ["R", "A", "S"]].values
    _distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(_centroids))

    path = get_template(
        template="MNI152NLin6Asym",
        atlas="Schaefer2018",
        desc="400Parcels7Networks",
        resolution=2,
        suffix="dseg",
        extension=".nii.gz",
    )
    assert isinstance(path, Path)

    atlas = Atlas.create("Schaefer2018400Parcels7Networks", path)
    centroids = atlas.get_centroids()

    distance = np.sqrt(np.square(_centroids - centroids).sum(axis=1))
    assert distance.mean() < 2  # mm

    distance_matrix = atlas.get_distance_matrix()
    assert np.abs(_distance_matrix - distance_matrix).mean() < 1  # mm

    region_membership = atlas.get_yeo7_membership()
    assert region_membership.shape == (400, 7)


def _get_centroids(path: Path):
    """
    Compute centroids.

    Parameters
    ----------

    d : int
        Atlas dimension.

    """
    centroids = find_probabilistic_atlas_cut_coords(path)
    return centroids


def test_probseg_atlas(data_path: Path) -> None:
    # "TODO: @haoting wants to revisit this test, to check if the assertion values make sense"
    # atlas_path = data_path / YEO_NETWORK_MAP
    # dl.get(str(atlas_path))

    path = get_template(
        template="MNI152NLin2009cAsym",
        atlas="DiFuMo",
        desc="64dimensions",  # or 64, 1024, …
        suffix="probseg",
        resolution=3,  # matches “res-03”
        extension=".nii.gz",
    )
    assert isinstance(path, Path)

    _centroids = _get_centroids(path)
    _distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(_centroids))

    atlas = Atlas.create("DiFuMo256dimensions", path)
    centroids = atlas.get_centroids()

    distance = np.sqrt(np.square(_centroids - centroids).sum(axis=1))
    assert distance.mean() < 50  # mm

    distance_matrix = atlas.get_distance_matrix()
    assert np.abs(_distance_matrix - distance_matrix).mean() < 50  # mm

    region_membership = atlas.get_yeo7_membership()
    assert region_membership.shape == (64, 7)
    assert region_membership.values.sum() == 64

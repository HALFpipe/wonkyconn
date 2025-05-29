from pathlib import Path

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
import scipy
from nilearn.plotting import find_probabilistic_atlas_cut_coords
from templateflow.api import get as get_template

from wonkyconn.atlas import Atlas


def test_dseg_atlas() -> None:
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


def test_probseg_atlas() -> None:
    path = Path(
        resource_filename(
            "wonkyconn",
            "data/test_data/tpl-MNI152NLin2009cAsym_res-03_atlas-DiFuMo_desc-64dimensionsSegmented_probseg.nii.gz",
        )
    )
    assert isinstance(path, Path)

    _centroids = _get_centroids(path)
    _distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(_centroids))

    atlas = Atlas.create("DiFuMo256dimensions", path)
    centroids = atlas.get_centroids()

    distance = np.sqrt(np.square(_centroids - centroids).sum(axis=1))
    assert distance.mean() < 4  # mm

    distance_matrix = atlas.get_distance_matrix()
    assert np.abs(_distance_matrix - distance_matrix).mean() < 3  # mm

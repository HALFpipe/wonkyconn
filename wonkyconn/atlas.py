from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nilearn.image import iter_img, load_img, math_img, resample_to_img  # type: ignore[import-not-found]
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker  # type: ignore[import-not-found]
from numpy import typing as npt

from .logger import gc_log

YEO_NETWORK_MAP = "atlases/atlas-Yeo7NetworksMNI152FreeSurferConformed1mmLiberal_dseg.nii.gz"


@dataclass
class Atlas(ABC):
    """
    Abstract base class representing a brain atlas.

    Attributes:
        seg (str): The "seg" value that the atlas corresponds to. A "seg" uniquely
            identifies an atlas in a given space and resolution.
        image (nib.nifti1.Nifti1Image): The Nifti1Image object for the atlas file.

    """

    seg: str
    image: nib.nifti1.Nifti1Image

    structure: npt.NDArray[np.bool_] = field(default_factory=lambda: np.ones((3, 3, 3), dtype=bool))

    @abstractmethod
    def get_centroid_points(self) -> npt.NDArray[np.float64]:
        """
        Returns the centroid points of the atlas regions.

        Returns:
            npt.NDArray[np.float64]: An array of centroid indices.
        """
        raise NotImplementedError

    def get_centroids(self) -> npt.NDArray[np.float64]:
        """
        Returns the centroid coordinates of the atlas regions.

        Returns:
            npt.NDArray[np.float64]: An array of centroid coordinates.
        """
        centroid_points = self.get_centroid_points()
        centroid_coordinates = nib.affines.apply_affine(self.image.affine, centroid_points)
        return centroid_coordinates

    def get_distance_matrix(self) -> npt.NDArray[np.float64]:
        """
        Calculates the pairwise distance matrix between the centroids
        of the atlas regions.

        Returns:
            npt.NDArray[np.float64]: The distance matrix.
        """
        centroids = self.get_centroids()
        return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(centroids))

    def load_yeo7_network(self) -> nib.nifti1.Nifti1Image:
        """
        Load and resample the yeo 7 networks to the atlas's space.

        Returns:
            nib.nifti1.Nifti1Image: resampled yeo 7 networks image.
        """
        # yeo7_nii = load_img(files("wonkyconn").parent / "data" / YEO_NETWORK_MAP)  # datalad managed directory at root
        yeo7_nii = load_img(files("wonkyconn") / "data" / YEO_NETWORK_MAP)  # in wonkyconn/data
        yeo7_nii = list(iter_img(yeo7_nii))[0]  # for some reason there's a fourth dimension
        yeo7_nii = resample_to_img(yeo7_nii, self.image, interpolation="nearest", copy_header=True, force_resample=True)
        return yeo7_nii

    @abstractmethod
    def get_yeo7_membership(self) -> pd.DataFrame:
        """Get the yeo7 network membership of the atlas.

        Returns:
            pd.DataFrame: membership of each parcel in shape (number of parcels, 7)
        """
        raise NotImplementedError

    @staticmethod
    def create(seg: str, path: Path) -> "Atlas":
        """
        Create an Atlas object based based on it's "seg" value and path.

        Parameters:
            seg (str): The "seg" value.
            path (Path): The path to the image.

        Returns:
            Atlas: An instance of the Atlas class.

        Raises:
            None

        """
        image = nib.nifti1.load(path)

        if image.ndim <= 3 or image.shape[3] == 1:
            return DsegAtlas(seg, nib.funcs.squeeze_image(image))
        else:
            return ProbsegAtlas(seg, image)


@dataclass
class DsegAtlas(Atlas):
    def get_array(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.image.dataobj, dtype=np.int64)

    def _check_single_connected_component(self, array: npt.NDArray[np.int64]) -> None:
        for i in range(1, array.max() + 1):
            mask = array == i
            _, num_features = scipy.ndimage.label(mask, structure=self.structure)
            if num_features > 1:
                gc_log.warning(f'Atlas "{self.seg}" region {i} has more than a single connected component')

    def get_centroid_points(self) -> npt.NDArray[np.float64]:
        array = self.get_array()
        self._check_single_connected_component(array)
        return np.asarray(
            scipy.ndimage.center_of_mass(
                input=array > 0,
                labels=array,
                index=np.arange(1, array.max() + 1),
            )
        )

    def get_yeo7_membership(self) -> pd.DataFrame:
        yeo7_nii = self.load_yeo7_network()
        network_labels = np.unique(yeo7_nii.dataobj)[1:]  # first value (0) is background
        region_labels = np.unique(self.image.dataobj)[1:]
        region_membership = pd.DataFrame(0, index=region_labels, columns=[f"yeo7-{int(n)}" for n in network_labels])

        for n in network_labels:
            cur_region = math_img(f"img=={n}", img=yeo7_nii)
            masker = NiftiMasker(cur_region)
            atlas_parcel_in_network = masker.fit_transform(self.image)
            atlas_parcel_in_network = np.unique(atlas_parcel_in_network)[1:]
            region_membership.loc[atlas_parcel_in_network, f"yeo7-{int(n)}"] = 1
        return region_membership


@dataclass
class ProbsegAtlas(Atlas):
    epsilon: float = 1e-6

    def _get_centroid_point(self, i: int, array: npt.NDArray[np.float64]) -> tuple[float, ...]:
        mask = array > self.epsilon
        _, num_features = scipy.ndimage.label(mask, structure=self.structure)
        if num_features > 1:
            gc_log.warning(f'Atlas "{self.seg}" region {i} has more than a single connected component')
        return scipy.ndimage.center_of_mass(array)

    def get_centroid_points(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [self._get_centroid_point(i, image.get_fdata()) for i, image in enumerate(nib.funcs.four_to_three(self.image))]
        )

    def get_yeo7_membership(self) -> pd.DataFrame:
        yeo7_nii = self.load_yeo7_network()
        network_labels = np.unique(yeo7_nii.dataobj)[1:]  # first value (0) is background
        region_labels = (np.arange(self.image.shape[-1]) + 1).tolist()  # time is the last dimension in probseg atlas
        region_membership = pd.DataFrame(0, index=region_labels, columns=[f"yeo7-{int(n)}" for n in network_labels])

        yeo7_masker = NiftiLabelsMasker(yeo7_nii)

        # shape: (number of parcels, 7)
        # for each parcel, if there's more overlap of one network with the given parcel, we mark the parcel belong to
        # the network
        summary_per_network = yeo7_masker.fit_transform(self.image)
        parcel_membership = (np.argmax(summary_per_network, axis=1) + 1).tolist()

        for region, yeo_network in zip(region_labels, parcel_membership, strict=False):
            region_membership.loc[region, f"yeo7-{int(yeo_network)}"] = 1
        return region_membership

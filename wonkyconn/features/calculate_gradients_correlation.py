import glob
from pathlib import Path
from typing import Iterable, List, Tuple

import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps  # type: ignore[import-not-found]
from joblib import Parallel, delayed  # type: ignore[import-not-found]
from nilearn import image  # type: ignore[import-not-found]
from nilearn.maskers import NiftiLabelsMasker  # type: ignore[import-not-found]
from scipy import stats

from ..base import ConnectivityMatrix


def remove_nan_from_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove rows/columns from a connectivity matrix that contain NaN values.

    Parameters
    ----------
    matrix : np.ndarray
        Square connectivity matrix.

    Returns
    -------
    np.ndarray
        Cleaned connectivity matrix with NaN rows/columns removed.
    np.ndarray
        Indices of the kept rows/columns.
    """

    col_mask = ~np.all(np.isnan(matrix), axis=0)

    kept_idx = np.where(col_mask)[0]
    conn_clean = matrix[np.ix_(kept_idx, kept_idx)]

    return conn_clean, kept_idx


def remove_nan_roi_atlas(atlas: nib.Nifti1Image, kept_idx: np.ndarray) -> nib.Nifti1Image:
    """
    Remove ROIs from an atlas that are not present in a connectivity matrix.

    Parameters
    ----------
    atlas_path : str
        Path to the original NIfTI atlas.
    conn_matrix : np.ndarray
        Square connectivity matrix corresponding to the atlas.
    save_path : str, optional
        Path to save the masked atlas. If None, the image is returned but not saved.

    Returns
    -------
    nib.Nifti1Image
        New atlas image with only ROIs present in conn_matrix.
    """

    # Load atlas
    atlas_data = atlas.get_fdata()

    roi_labels = sorted([int(x) for x in np.unique(atlas_data) if x != 0])

    # Now remove labels that were removed from the conn matrix
    kept_labels = [roi_labels[i] for i in kept_idx]

    # Make a mask image that keeps only kept_labels
    keep_mask = np.isin(atlas_data, kept_labels)
    kept_atlas_data = atlas_data.copy()
    kept_atlas_data[~keep_mask] = 0
    return nib.Nifti1Image(kept_atlas_data, atlas.affine, atlas.header), kept_labels


def overlapping_atlas_with_mask(subject_atlas: nib.Nifti1Image, group_mask: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Create a new atlas that only contains the regions that overlap with the group gradient mask.

    Parameters
    ----------
    subject_atlas : nib.Nifti1Image
        The subject's atlas in NIfTI format.
    group_mask : nib.Nifti1Image
        The group gradient mask in NIfTI format.

    Returns
    -------
    nib.Nifti1Image
        A new atlas that only contains the regions that overlap with the group gradient mask.
    """

    mask_gradient_resampled = image.resample_to_img(
        group_mask, subject_atlas, interpolation="nearest", copy_header=True, force_resample=True
    )

    # Get arrays
    atlas_data = subject_atlas.get_fdata()
    mask_data = mask_gradient_resampled.get_fdata() > 0  # binarized mask

    masked_atlas_data = np.where(mask_data, atlas_data, 0)

    return nib.Nifti1Image(masked_atlas_data, affine=subject_atlas.affine, header=subject_atlas.header)


def clean_matrix_from_atlas(matrix: np.ndarray, atlas: nib.Nifti1Image) -> np.ndarray:
    """
    Remove rows/columns from a connectivity matrix for regions not present in the atlas.

    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix
    atlas : nib.Nifti1Image
        Masked atlas

    Returns
    -------
    np.ndarray
        Connectivity matrix limited to regions present in the atlas.
    """

    atlas_data = atlas.get_fdata()
    roi_labels = sorted(np.unique(atlas_data[atlas_data > 0]))

    # Get indices corresponding to these labels (assuming atlas labels are 1-based)
    indices_to_keep = [int(lab - 1) for lab in roi_labels]

    if max(indices_to_keep) >= matrix.shape[0]:
        return matrix
    else:
        return matrix[np.ix_(indices_to_keep, indices_to_keep)]


def process_single_matrix(
    connectivity_matrix: ConnectivityMatrix,
    atlas: nib.Nifti1Image,
    gradient_mask: nib.Nifti1Image,
    gradient_imgs: List[nib.Nifti1Image],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single connectivity matrix to compute its gradients aligned to group gradients.

    Parameters
    ----------
    connectivity_matrix : ConnectivityMatrix
        The connectivity matrix to process.
    atlas : nib.Nifti1Image
        The atlas NIfTI image.
    gradient_mask : nib.Nifti1Image
        The group gradient mask NIfTI image.
    gradient_imgs : List[nib.Nifti1Image]
        List of group gradient NIfTI images.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The individual gradients aligned to group gradients.
        - The group gradients as a NumPy array.
    """

    matrix = np.asarray(connectivity_matrix.load(), dtype=np.float64)
    conn_clean, kept_idx = remove_nan_from_matrix(matrix)
    atlas_mask_without_nan, _ = remove_nan_roi_atlas(atlas, kept_idx)
    masked_atlas = overlapping_atlas_with_mask(atlas_mask_without_nan, gradient_mask)
    masked_matrix = clean_matrix_from_atlas(conn_clean, masked_atlas)
    masker = NiftiLabelsMasker(labels_img=masked_atlas, mask_img=gradient_mask)

    # Transform pre-loaded group gradients
    group_gradients = []
    for grad_img in gradient_imgs:
        grad_vals = masker.fit_transform(grad_img)  # shape (1, n_regions)
        group_gradients.append(grad_vals.squeeze())
    group_gradients_np = np.vstack(group_gradients).T  # shape (n_regions, n_components)

    # Compute individual gradients
    gm = GradientMaps(n_components=5, alignment="procrustes", kernel="normalized_angle")
    ind_gradient = gm.fit(masked_matrix, reference=group_gradients_np)

    return ind_gradient.aligned_, group_gradients_np


def extract_gradients(
    connectivity_matrices: Iterable[ConnectivityMatrix],
    atlas: nib.Nifti1Image,
    n_jobs: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Calculate the gradients for each individual and load group-level gradients
    for alignment.

    Parameters
    ----------
    connectivity_matrices : Iterable[ConnectivityMatrix]
        List of connectivity matrices to process.
    atlas : nib.Nifti1Image
        The atlas NIfTI image.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is 4.

    Returns
    -------
    gradients : list[np.ndarray]
        List of individual gradients
    group_gradients_list : list[np.ndarray]
        List of group-level gradients, one per individual
    """

    repo_root = Path(__file__).resolve().parent.parent

    path_gradients = repo_root / "data" / "gradients"
    gradient_mask = nib.load(path_gradients / "gradientmask_cortical_subcortical.nii.gz")

    # Load all group gradient templates
    gradient_files = sorted(glob.glob(str(path_gradients / "templates" / "gradient*_cortical_subcortical.nii.gz")))
    gradient_imgs = [nib.load(fname) for fname in gradient_files]

    # Parallel processing
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_matrix)(cm, atlas, gradient_mask, gradient_imgs) for cm in connectivity_matrices
    )

    # Each result is (gradients_i, group_gradients_i)
    gradients = [r[0] for r in results]
    group_gradients_list = [r[1] for r in results]

    return gradients, group_gradients_list


def calculate_group_gradients_similarity(correlations: List[float]) -> float:
    """
    Calculate the mean of a all participant's correlation values.

    Parameters:
    - correlations (list): A list of correlation values.

    Returns:
    - float: The mean of the correlation values.
    """
    return float(np.mean(correlations))


def calculate_gradients_similarity(
    gradients: List[np.ndarray],
    group_gradients: List[np.ndarray],
) -> float:
    """
    Calculate similarity between individual and group gradients via Spearman
    correlations + Fisher z-transform.

    Parameters
    ----------
    gradients : list[np.ndarray]
        Individual gradients, shape (n_vertices, n_components)
    group_gradients : list[np.ndarray]
        Matched group-level gradients for each subject, same shape

    Returns
    -------
    similarities : float
       Averaged similarity value across subject (mean Fisher-z across components)
    """
    similarities = []

    for subj_idx, (ind_grad, grp_grad) in enumerate(zip(gradients, group_gradients, strict=True)):
        if np.array(ind_grad).shape != grp_grad.shape:
            raise ValueError(
                f"Shape mismatch for subject {subj_idx}: individual {np.array(ind_grad).shape} vs group {grp_grad.shape}"
            )

        n_components = ind_grad.shape[1]

        # Spearman correlation over components
        rho_list = []
        for comp in range(n_components):
            r, _ = stats.spearmanr(ind_grad[:, comp], grp_grad[:, comp])
            rho_list.append(r)

        # Fisher r-to-z transform, mean over components
        z_mean = np.mean(np.arctanh(rho_list))
        similarities.append(z_mean)

    return calculate_group_gradients_similarity(similarities)

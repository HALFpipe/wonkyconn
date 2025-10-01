These results were generated with
giga_connectome (version 0.5.1.dev13+gd722c8a.d20240502, https://giga-connectome.readthedocs.io/en/latest/)
using Nilearn (version 0.10.3, RRID:SCR_001362)
and TemplateFlow (version 0.8.1).

The following steps were followed.

1. Retrieve subject specific grey matter mask in MNI space (MNI152NLin2009cAsym).

1. Sampled the Schaefer20187Networks atlas to the space of subject specific grey matter mask in MNI space.

1. Calculated the conjunction of the subject specific grey matter mask and resampled atlas to find valid parcels.

1. Used the subject specific grey matter mask and atlas to extract time series and connectomes for each subject.
   The time series data were denoised as follow:

    - Time series extractions through label or map maskers are performed on the denoised nifti file.
    - Denoising steps were performed on the voxel level:
        - spatial smoothing (FWHM: 5.0 mm)
        - detrending, only if high pass filter was not applied through confounds
        - regressing out confounds (using a simple strategy)
        - standardized (using zscore)
    - Extracted time series from atlas
    - Computed correlation matrix (Pearson's correlation with LedoitWolf covariance estimator)

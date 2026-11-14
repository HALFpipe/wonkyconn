[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SIMEXP/wonkyconn/main.svg)](https://results.pre-commit.ci/latest/github/SIMEXP/wonkyconn/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![https://github.com/psf/black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Documentation Status](https://readthedocs.org/projects/wonkyconn/badge/?version=latest)](https://wonkyconn.readthedocs.io/en/latest/?badge=latest)

# wonkyconn

Evaluating the residual motion in fMRI connectome and visualise reports.

We are currently working towards 0.1.0 release!!!! This is not a stable project yet.

## CLI

- Allow specifying the variables for QC-FC

  Needs to have gender and age

  ```bash
      --phenotypes file.tsv
  ```

- Calculate distance dependence based on given BEP17 naming for atlas

  ```bash
    --atlas Schaefer20187Networks200Parcels file.nii.gz
  ```

## Tasks

- Migrate the code from <https://github.com/SIMEXP/fmriprep-denoise-benchmark>
- Add test data
- Figure out how many confound regressors were used (should this be part of the BIDS metadata)

# wonkyconn

Evaluating the residual motion in fMRI connectome and visualise reports

## CLI

- Allow specifying the variables for QC-FC

  Needs to have gender and age

  ```bash
      --phenotypes file.tsv
  ```

- Calculate distance dependence based on given BEP17 naming for atlas

  ```bash
    --seg-to-atlas Schaefer20187Networks200Parcels file.nii.gz
  ```

## Tasks

- Migrate the code from <https://github.com/SIMEXP/fmriprep-denoise-benchmark>
- Add test data
- Figure out how many confound regressors were used (should this be part of the BIDS metadata)

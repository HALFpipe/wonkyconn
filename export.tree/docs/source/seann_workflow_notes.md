# Workflow Documentation for development

This document outlines how to run the WONKYCONN codebase for analyzing fMRI connectivity matrices.

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Core Workflow](#core-workflow)
5. [Troubleshooting](#troubleshooting)
6. [References](#references)

---

## Overview

WONKYCONN takes outputs from giga_connectome (or outputs from HALFPIPE aligned to BEP017 specification) and analyzes connectivity matrices to evaluate residual motion in fMRI connectomes.

## Environment Setup

1. Install the package and its dependencies:
    ```bash
    pip install -r requirements.txt
    pip install e .
    ```

## Data Preparation

Required input files:

1. **BIDS-formatted fMRIPrep derivatives**, containing:
    - Timeseries (.tsv files)
    - Connectivity matrices

2. **Phenotypes file** (participants.tsv) with required columns:
    - participant_id
    - gender (coded as 'M'=0 'F'=1)
    - age (in years)

3. **Atlas files** in NIfTI format (.nii.gz)

## Core Workflow

WONKYCONN uses a single command-line interface. There are no separate scripts to run manually.

1. **Basic Command Structure:**
    ```bash
    wonkyconn <bids_dir> <output_dir> group \
        --phenotypes <phenotypes.tsv> \
        --atlas <seg_label> <atlas_path>
    ```

2. **Full Example:**
    ```bash
    wonkyconn /path/to/fmriprep/output /path/to/output group \
        --phenotypes participants.tsv \
        --atlas schaefer400 /path/to/atlas/schaefer400.nii.gz \
        --group-by seg desc \
        --verbosity 2
    ```

3. **Command Arguments:**
    - `bids_dir`: Input BIDS directory containing fMRIPrep outputs
    - `output_dir`: Directory for results
    - `--phenotypes`: Path to participants.tsv file
    - `--atlas`: Pairs of segmentation label and atlas path
    - `--group-by`: Tags to group results by (default: seg)
    - `--verbosity`: Logging level (0-3)

4. **Outputs:**
    - metrics.tsv: Contains calculated connectivity measures
    - Visualization plots

## Troubleshooting

Common issues:

1. **Missing Phenotypes Columns:**
    - Ensure participants.tsv has required columns: participant_id, gender, age
    - gender must be coded as 'M' = 0 or 'F' = 0, not as strings
    - age must be in years

2. **BIDS Format Issues:**
    - Verify input directory follows BIDS derivatives format
    - Check file naming conventions match BEP017 specification

## References

- [BIDS Specification](https://bids-specification.readthedocs.io/)
- [fMRIPrep Documentation](https://fmriprep.org/)
- [BEP017 Specification](https://bids.neuroimaging.io/extensions/beps/bep_017.html)


## BUGs during running

### statsmodels not found
Traceback (most recent call last):
  File "/lustre07/scratch/seann/wonkyconn/wonkyconnenv/bin/wonkyconn", line 5, in <module>
    from wonkyconn.run import main
  File "/lustre07/scratch/seann/wonkyconn/wonkyconn/run.py", line 8, in <module>
    from .workflow import workflow, gc_log
  File "/lustre07/scratch/seann/wonkyconn/wonkyconn/workflow.py", line 17, in <module>
    from .features.quality_control_connectivity import (
  File "/lustre07/scratch/seann/wonkyconn/wonkyconn/features/quality_control_connectivity.py", line 8, in <module>
    from statsmodels.stats.multitest import multipletests
ModuleNotFoundError: No module named 'statsmodels'
- Fix: added 'statsmodels' to pyproject.toml

### patricipants.tsv needs to be properly formatted
- see prepare_participantstsv.py for quick conversion script.
- note: participants.tsv files can vary widely and prepare_participantstsv.py cannot handle all cases. Please visually inspect the participants.tsv and change it if needed to the formatting described above in Data Preparation.

### code does not grab subjects from participants tsv with 'sub-' tag
- Fix: see changes to make_record in workflow.py

### code does not grab matrices corresponding to the specific atlas passed to the CLI
- Fix: see modifications to workflow.py, filtering connectivity matrix loaded from input path for only the specified atlas. all other relmat files from other atlas are not loaded.


### Pre-commit version issue with compute canada
- Currently we use pre-commit version: 4.2.0+computecanada
- This breaks because of the +computecanada tag. ValueError: invalid literal for int() with base 10: '0+computecanada'

- Fix: we force reinstall directly from the source using this command:
    pip uninstall -y pre-commit
    pip install git+https://github.com/pre-commit/pre-commit.git@v4.2.0

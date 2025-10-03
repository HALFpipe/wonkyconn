# Installation

## Quick start (container)

Pull the latest image from docker hub, available for version > `0.5.0`(Recommended)

Apptainer (formerly known as Singularity; recommended):

```bash
apptainer build wonkyconn.simg docker://haotingwang/wonkyconn:latest
```

Docker:
```bash
docker pull haotingwang/wonkyconn:latest
```

## Install as a python package

Install the project in a Python environment:

```bash
pip install git+https://github.com/SIMEXP/wonkyconn.git
```

This method is available for all versions.
Change the tag based on version you would like to use.

# Installation

## Quick start (container)

Pull the latest image from docker hub (Recommended).

Apptainer (formerly known as Singularity; recommended):

```bash
apptainer build wonkyconn-edge.simg docker://halfpipe/wonkyconn:edge
```

Docker:
```bash
docker pull halfpipe/wonkyconn:edge
```

## Install as a python package

Install the project in a Python environment:

```bash
pip install git+https://github.com/halfpipe/wonkyconn.git
```

This method is available for all versions.
Change the tag based on version you would like to use.

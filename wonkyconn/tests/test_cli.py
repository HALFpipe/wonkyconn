"""
Simple code to smoke test the functionality.
"""

import json
import re
import shutil
from pathlib import Path
from shutil import copyfile

import datalad.api as dl
import numpy as np
import pandas as pd
import pytest
import scipy
from tqdm.auto import tqdm

from wonkyconn import __version__
from wonkyconn.file_index.bids import BIDSIndex
from wonkyconn.run import global_parser, main
from wonkyconn.workflow import workflow


def test_version(capsys):
    try:
        main(["-v"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert __version__ == captured.out.split()[0]


def test_help(capsys):
    try:
        main(["-h"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "Evaluating the residual motion in fMRI connectome and visualize reports" in captured.out


def _copy_file(path: Path, new_path: Path, sub: str) -> None:
    new_path = Path(re.sub(r"sub-\d+", sub, str(new_path)))
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if "relmat" in path.name and path.suffix == ".tsv":
        relmat = pd.read_csv(path, sep="\t")
        (n,) = set(relmat.shape)

        array = scipy.spatial.distance.squareform(relmat.to_numpy() - np.eye(n))
        np.random.shuffle(array)

        new_array = scipy.spatial.distance.squareform(array) + np.eye(n)

        new_relmat = pd.DataFrame(new_array, columns=relmat.columns)
        new_relmat.to_csv(new_path, sep="\t", index=False)
    elif "timeseries" in path.name and path.suffix == ".json":
        with open(path, "r") as f:
            content = json.load(f)
            content["MeanFramewiseDisplacement"] += np.random.uniform(0, 1)
        with open(new_path, "w") as f:
            json.dump(content, f)
    else:
        copyfile(path, new_path)


@pytest.mark.smoke
def test_giga_connectome(data_path: Path, tmp_path: Path):
    data_path = data_path / "giga_connectome" / "connectome_Schaefer20187Networks_dev"
    dl.get(str(data_path))

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    subjects = [f"sub-{i}" for i in ["2", "3", "4", "5", "6", "7"]]

    paths = list(data_path.glob("**/*"))
    for path in tqdm(paths, desc="Generating test data"):
        if not path.is_file():
            continue
        for sub in subjects:
            _copy_file(path, bids_dir / path.relative_to(data_path), str(sub))

    phenotypes = pd.DataFrame(
        dict(
            participant_id=subjects,
            age=np.random.uniform(18, 80, len(subjects)),
            gender=np.random.choice(["m", "f"], len(subjects)),
        )
    )
    phenotypes_path = bids_dir / "participants.tsv"
    phenotypes.to_csv(phenotypes_path, sep="\t", index=False)

    atlas_args: list[str] = []
    for n in [100, 200, 300, 400, 500, 600, 800]:
        atlas_args.append("--atlas")
        atlas_args.append(f"Schaefer20187Networks{n}Parcels")
        dseg_path = data_path / "atlases" / "sub-1" / "func" / f"sub-1_seg-Schaefer20187Networks{n}Parcels_dseg.nii.gz"
        atlas_args.append(str(dseg_path))

    parser = global_parser()
    argv = [
        "--phenotypes",
        str(phenotypes_path),
        *atlas_args,
        "--verbosity",
        "3",
        str(bids_dir),
        str(output_dir),
        "group",
    ]
    args = parser.parse_args(argv)

    workflow(args)

    assert (output_dir / "metrics.tsv").is_file()
    assert (output_dir / "metrics.png").is_file()


@pytest.mark.smoke
def test_halfpipe(data_path: Path, tmp_path: Path):
    bids_dir = data_path / "halfpipe"
    dl.get(str(bids_dir))

    index = BIDSIndex()
    index.put(bids_dir)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    phenotypes_path = bids_dir / "participants.tsv"

    atlas_path = data_path / "atlases"
    dl.get(str(atlas_path))

    atlas_args: list[str] = list()
    atlas_args.append("--atlas")
    atlas_args.append("Schaefer2018Combined")
    atlas_args.append(str(atlas_path / "atlas-Schaefer2018Combined_dseg.nii.gz"))

    parser = global_parser()
    # Fix --atlas: changed to use new --atlas argument
    argv = [
        "--phenotypes",
        str(phenotypes_path),
        *atlas_args,
        str(bids_dir),
        str(output_dir),
        "group",
    ]

    args = parser.parse_args(argv)
    workflow(args)

    # Add persistent storage to extract figure as artifact
    persistent_dir = Path("figures_artifacts")
    persistent_dir.mkdir(exist_ok=True)
    fig_file = output_dir / "metrics.png"
    shutil.copy(fig_file, persistent_dir / fig_file.name)

    assert (output_dir / "metrics.tsv").is_file()
    assert (output_dir / "metrics.png").is_file()

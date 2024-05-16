"""
Simple code to smoke test the functionality.
"""

from pathlib import Path

import json
import re
from shutil import copyfile
import numpy as np
import pytest
from pkg_resources import resource_filename

import pandas as pd
import scipy
from tqdm.auto import tqdm

from wonkyconn import __version__
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
    assert "show program's version number and exit" in captured.out


def _copy_file(path: Path, new_path: Path, sub: str) -> None:
    new_path = Path(re.sub(r"sub-\d+", f"sub-{sub}", str(new_path)))
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
def test_smoke(tmp_path: Path):
    data_path = Path(
        resource_filename(
            "wonkyconn", "data/test_data/connectome_Schaefer20187Networks_dev"
        )
    )

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    subjects = ["2", "3", "4", "5", "6", "7"]

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

    seg_to_atlas_args: list[str] = []
    for n in [100, 200, 300, 400, 500, 600, 800]:
        seg_to_atlas_args.append("--seg-to-atlas")
        seg_to_atlas_args.append(f"Schaefer20187Networks{n}Parcels")
        dseg_path = (
            data_path
            / "atlases"
            / "sub-1"
            / "func"
            / f"sub-1_seg-Schaefer20187Networks{n}Parcels_dseg.nii.gz"
        )
        seg_to_atlas_args.append(str(dseg_path))

    parser = global_parser()
    argv = [
        "--phenotypes",
        str(phenotypes_path),
        *seg_to_atlas_args,
        str(bids_dir),
        str(output_dir),
        "group",
    ]
    args = parser.parse_args(argv)

    workflow(args)

    # output_folder = output_dir / "sub-1" / "ses-timepoint1" / "func"

    # base = (
    #     "sub-1_ses-timepoint1_task-probabilisticclassification"
    #     "_run-01_space-MNI152NLin2009cAsym_res-2"
    #     "_atlas-Schaefer20187Networks"
    # )

    # relmat_file = output_folder / (
    #     base + "_meas-PearsonCorrelation" + "_desc-100Parcels7NetworksSimple_relmat.tsv"
    # )
    # assert relmat_file.exists()
    # relmat = pd.read_csv(relmat_file, sep="\t")
    # assert len(relmat) == 100

    # json_file = relmat_file = output_folder / (base + "_timeseries.json")
    # assert json_file.exists()
    # with open(json_file, "r") as f:
    #     content = json.load(f)
    #     assert content.get("SamplingFrequency") == 0.5

    # timeseries_file = relmat_file = output_folder / (
    #     base + "_desc-100Parcels7NetworksSimple_timeseries.tsv"
    # )
    # assert timeseries_file.exists()
    # timeseries = pd.read_csv(timeseries_file, sep="\t")
    # assert len(timeseries.columns) == 100

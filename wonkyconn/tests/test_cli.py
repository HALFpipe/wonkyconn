"""
Simple code to smoke test the functionality.
"""

import argparse
import json
import re
from math import isclose
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
import scipy
from pkg_resources import resource_filename
from tqdm.auto import tqdm

from wonkyconn import __version__
from wonkyconn.file_index.bids import BIDSIndex
from wonkyconn.run import global_parser, main
from wonkyconn.wizard import WizardConfig
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


def test_wizard_config_roundtrip(tmp_path: Path):
    config = WizardConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "output",
        analysis_level="group",
        phenotypes=tmp_path / "participants.tsv",
        atlas_label="TestAtlas",
        atlas_path=tmp_path / "atlas.nii.gz",
        group_by=["seg"],
        verbosity=3,
        debug=True,
        wizard_theme="dark",
    )

    config_path = tmp_path / "config.json"
    config.to_file(config_path)

    loaded = WizardConfig.from_file(config_path)
    assert loaded == config


def test_main_uses_config(monkeypatch, tmp_path: Path):
    from wonkyconn import run

    base_config = WizardConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "output",
        analysis_level="group",
        phenotypes=tmp_path / "participants.tsv",
        atlas_label="BaseAtlas",
        atlas_path=tmp_path / "atlas.nii.gz",
        group_by=["seg"],
        verbosity=2,
        debug=False,
        wizard_theme="light",
    )

    config_path = tmp_path / "base_config.json"
    base_config.to_file(config_path)

    captured: dict[str, argparse.Namespace] = {}

    def fake_run(args: argparse.Namespace) -> None:
        captured["args"] = args

    monkeypatch.setattr(run, "_run_with_args", fake_run)

    run.main([
        "--config",
        str(config_path),
        "--atlas",
        "OverrideAtlas",
        str(tmp_path / "override_atlas.nii.gz"),
    ])

    assert "args" in captured
    run_args = captured["args"]
    assert run_args.atlas == ["OverrideAtlas", str(tmp_path / "override_atlas.nii.gz")]
    assert run_args.bids_dir == base_config.bids_dir
    assert run_args.output_dir == base_config.output_dir
    assert run_args.verbosity == [base_config.verbosity]


def test_main_saves_config(monkeypatch, tmp_path: Path):
    from wonkyconn import run

    config_path = tmp_path / "saved_config.json"

    def fake_run(args: argparse.Namespace) -> None:
        return None

    monkeypatch.setattr(run, "_run_with_args", fake_run)

    run.main(
        [
            str(tmp_path / "bids"),
            str(tmp_path / "output"),
            "group",
            "--phenotypes",
            str(tmp_path / "participants.tsv"),
            "--atlas",
            "AtlasLabel",
            str(tmp_path / "atlas.nii.gz"),
            "--save-config",
            str(config_path),
        ]
    )

    saved = WizardConfig.from_file(config_path)
    assert saved.bids_dir == tmp_path / "bids"
    assert saved.atlas_label == "AtlasLabel"
    assert saved.wizard_theme == "light"


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
def test_giga_connectome(tmp_path: Path):
    data_path = Path(resource_filename("wonkyconn", "data/giga_connectome/connectome_Schaefer20187Networks_dev"))

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
        str(bids_dir),
        str(output_dir),
        "group",
    ]
    args = parser.parse_args(argv)

    workflow(args)

    assert (output_dir / "metrics.tsv").is_file()
    assert (output_dir / "metrics.png").is_file()


@pytest.mark.smoke
def test_halfpipe(tmp_path: Path):
    data_path = Path(resource_filename("wonkyconn", "data"))

    bids_dir = data_path / "halfpipe"

    index = BIDSIndex()
    index.put(bids_dir)
    for timeseries_path in index.get(suffix="timeseries", extension=".json"):
        timeseries_path = timeseries_path.with_suffix(".tsv")
        if not timeseries_path.is_file():
            with timeseries_path.open("w"):
                pass

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    phenotypes_path = bids_dir / "participants.tsv"

    atlas_args: list[str] = list()
    atlas_args.append("--atlas")
    atlas_args.append("Schaefer2018Combined")
    atlas_args.append(str(data_path / "atlases/atlas-Schaefer2018Combined_dseg.nii.gz"))

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

    assert (output_dir / "metrics.tsv").is_file()
    assert (output_dir / "metrics.png").is_file()


# parametrize to test different h2bids outputs instead of creating multiple test functions
@pytest.mark.parametrize("flag", ["denoise_metadata", "impute_and_metadata", "impute_nan"])
def test_smoke_h2bids(tmp_path: Path, flag):
    data_path = Path(resource_filename("wonkyconn", f"data/test_data/test_data_h2bids_{flag}"))
    atlas_label = "schaefer400"
    dseg_path = data_path / "atlas" / "atlas-Schaefer2018Combined_dseg.nii.gz"

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    subjects = [f"sub-{i}" for i in ["10159", "10171", "10189", "10206", "10217", "10225", "10227", "10228", "10235", "10249"]]

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

    parser = global_parser()
    # Fix --atlas: changed to use new --atlas argument
    argv = [
        "--phenotypes",
        str(phenotypes_path),
        "--group-by",
        "seg",
        "desc",
        "--atlas",
        atlas_label,
        str(dseg_path),
        str(bids_dir),
        str(output_dir),
        "group",
    ]

    args = parser.parse_args(argv)
    workflow(args)

    assert (output_dir / "metrics.tsv").is_file()
    assert (output_dir / "metrics.png").is_file()

    data_frame = pd.read_csv(output_dir / "metrics.tsv", sep="\t")
    data_frame = data_frame.set_index("feature")

    assert isclose(data_frame.loc["cCompCor"]["motion_scrubbing_percentage"], 0.0)
    assert data_frame.loc["cCompCor"]["confound_regression_percentage"] > 0.0

    assert data_frame.loc["motionParametersScrubbing"]["motion_scrubbing_percentage"] > 0.0
    assert data_frame.loc["motionParametersScrubbing"]["confound_regression_percentage"] > 0.0

    assert (
        data_frame.loc["motionParametersScrubbing"]["distance_dependence"] < data_frame.loc["cCompCor"]["distance_dependence"]
    )

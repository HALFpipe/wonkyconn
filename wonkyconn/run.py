from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .workflow import workflow, gc_log


def global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=("Evaluating the residual motion in fMRI connectome and visualize reports"),
    )

    # BIDS app required arguments
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help="The directory with the input dataset (e.g. fMRIPrep derivative)" "formatted according to the BIDS standard.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only group" "level is available.",
        choices=["group"],
    )

    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        default=["seg"],
        help="Select which tags to group the connectivity matrices by. Default is `seg`.",
    )
    parser.add_argument(
        "--phenotypes",
        type=str,
        help="Path to the phenotype file that has the columns `participant_id`, `gender` coded as `M` and `F` and `age` in years.",
        required=True,
    )
    #Fix: --atlas argument only accepts one atlas at a time, replaces --seg-by-atlas
    parser.add_argument(
        "--atlas",
        type=str,
        nargs=2,
        metavar=("LABEL", "ATLAS_PATH"),
        required=True,
        help="Specify the atlas label and the path to the atlas file (e.g. --atlas Schaefer2018 /path/to/atlas.nii.gz)",
    )

    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--verbosity",
        help="""
        Verbosity level.
        """,
        required=False,
        choices=[0, 1, 2, 3],
        default=2,
        type=int,
        nargs=1,
    )
    return parser


def main(argv: None | Sequence[str] = None) -> None:
    parser = global_parser()
    args = parser.parse_args(argv)

    try:
        workflow(args)
    except Exception as e:
        gc_log.exception("Exception: %s", e, exc_info=True)
        if args.debug:
            import pdb

            pdb.post_mortem()


if __name__ == "__main__":
    raise RuntimeError("run.py should not be run directly;\n" "Please `pip install` and use the `giga_connectome` command")

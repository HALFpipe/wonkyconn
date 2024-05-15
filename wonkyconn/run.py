from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .workflow import workflow


def global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Evaluating the residual motion in fMRI connectome and visualize reports"
        ),
    )

    # BIDS app required arguments
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help="The directory with the input dataset (e.g. fMRIPrep derivative)"
        "formatted according to the BIDS standard.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only group"
        "level is available.",
        choices=["group"],
    )

    parser.add_argument("-v", "--version", action="version", version=__version__)

    parser.add_argument(
        "--phenotypes",
        type=str,
        help="Path to the phenotype file that has the columns `participant_id`, `gender` coded as `M` and `F` and `age` in years.",
    )
    parser.add_argument(
        "--seg-to-atlas",
        type=str,
        nargs=2,
        action="append",
        metavar=("SEG", "ATLAS"),
        default=list(),
        help="Specify the atlas file to use for a segmentation label in the data",
    )

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
    """Entry point."""
    parser = global_parser()
    args = parser.parse_args(argv)

    workflow(args)


if __name__ == "__main__":
    raise RuntimeError(
        "run.py should not be run directly;\n"
        "Please `pip install` and use the `giga_connectome` command"
    )

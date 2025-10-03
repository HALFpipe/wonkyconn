from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .workflow import gc_log, workflow


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
        nargs="?",
        default=None,
        help="The directory with the input dataset (e.g. fMRIPrep derivative)" "formatted according to the BIDS standard.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        nargs="?",
        default=None,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only group level is available",
        choices=["group"],
        nargs="?",
        default=None,
    )

    parser.add_argument(
        "--phenotypes",
        type=str,
        help="Path to the phenotype file that has the columns `participant_id`, `gender` coded as `M` and `F` and `age` in years.",
        required=False,
    )
    parser.add_argument(
        "--atlas",
        type=str,
        nargs=2,
        action="append",
        metavar=("LABEL", "ATLAS_PATH"),
        required=False,
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
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Launch an interactive wizard to collect configuration options instead of passing them as CLI arguments.",
    )
    parser.add_argument(
        "--wizard-theme",
        choices=["light", "dark"],
        help="Preferred color theme for the interactive wizard (default: light).",
    )
    return parser


def _run_with_args(args: argparse.Namespace) -> None:
    workflow(args)


def main(argv: None | Sequence[str] = None) -> None:
    parser = global_parser()
    args = parser.parse_args(argv)

    if args.wizard:
        from .wizard import WizardAbort, collect_configuration

        try:
            wizard_args = collect_configuration(args)
        except WizardAbort as exception:
            gc_log.info("Wizard aborted: %s", exception)
            return

        wizard_args.wizard = False
        _run_with_args(wizard_args)
        return

    missing_positionals = [
        name
        for name, value in (
            ("bids_dir", args.bids_dir),
            ("output_dir", args.output_dir),
            ("analysis_level", args.analysis_level),
        )
        if value is None
    ]
    missing_optionals = [
        name
        for name, value in (
            ("phenotypes", args.phenotypes),
            ("atlas", args.atlas),
        )
        if value is None
    ]

    if missing_positionals or missing_optionals:
        missing = missing_positionals + missing_optionals
        parser.error(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Provide them explicitly or run with --wizard."
        )

    try:
        _run_with_args(args)
    except Exception as e:
        gc_log.exception("Exception: %s", e, exc_info=True)
        if args.debug:
            import pdb

            pdb.post_mortem()


if __name__ == "__main__":
    main(sys.argv[1:])

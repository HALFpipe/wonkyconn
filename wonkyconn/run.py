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
        help=(
            "The directory with the input dataset (e.g. fMRIPrep derivative) "
            "formatted according to the BIDS standard"
        ),
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        nargs="?",
        default=None,
        help="The directory where the output files should be stored",
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
        help=(
            "Path to the phenotype file that has the columns `participant_id`, "
            "`gender` coded as `M` and `F` and `age` in years"
        ),
        required=False,
    )
    # Fix: --atlas argument only accepts one atlas at a time, replaces --seg-by-atlas
    parser.add_argument(
        "--atlas",
        type=str,
        nargs=2,
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
    parser.add_argument(
        "--config",
        type=Path,
        help="Load configuration options from a JSON file produced by the wizard.",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        help="Write the final configuration to the specified JSON file.",
    )
    return parser


def _run_with_args(args: argparse.Namespace) -> None:
    workflow(args)


def main(argv: None | Sequence[str] = None) -> None:
    parser = global_parser()
    args = parser.parse_args(argv)

    config_object = None
    if getattr(args, "config", None) is not None:
        from .wizard import WizardConfig

        try:
            config_object = WizardConfig.from_file(args.config)
        except FileNotFoundError as exc:
            parser.error(f"Unable to open configuration file '{args.config}': {exc.strerror}")
        except ValueError as exc:
            parser.error(f"Configuration file '{args.config}' is invalid: {exc}")

    if args.wizard:
        from .wizard import WizardAbort, WizardConfig, collect_configuration

        try:
            wizard_config = collect_configuration(args, config_object)
        except WizardAbort as exception:
            gc_log.info("Wizard aborted: %s", exception)
            return

        wizard_args = wizard_config.to_namespace()
        wizard_args.wizard = False

        if getattr(args, "save_config", None) is not None:
            wizard_config.to_file(args.save_config)

        _run_with_args(wizard_args)
        return

    if config_object is not None:
        from .wizard import WizardConfig

        run_args = config_object.to_namespace()

        if getattr(args, "bids_dir", None) is not None:
            run_args.bids_dir = args.bids_dir
        if getattr(args, "output_dir", None) is not None:
            run_args.output_dir = args.output_dir
        if getattr(args, "phenotypes", None) is not None:
            run_args.phenotypes = args.phenotypes
        if getattr(args, "atlas", None):
            run_args.atlas = args.atlas
        verbosity_override = getattr(args, "verbosity", None)
        if isinstance(verbosity_override, list):
            run_args.verbosity = verbosity_override
        if getattr(args, "debug", False):
            run_args.debug = True
        if getattr(args, "wizard_theme", None):
            run_args.wizard_theme = args.wizard_theme

        run_args.wizard = False

        final_config = WizardConfig.from_namespace(run_args)

        if getattr(args, "save_config", None) is not None:
            final_config.to_file(args.save_config)

        _run_with_args(run_args)
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
        if getattr(args, "save_config", None) is not None:
            from .wizard import WizardConfig

            config_to_save = WizardConfig.from_namespace(args)
            config_to_save.to_file(args.save_config)
    except Exception as e:
        gc_log.exception("Exception: %s", e, exc_info=True)
        if args.debug:
            import pdb

            pdb.post_mortem()


if __name__ == "__main__":
    main(sys.argv[1:])

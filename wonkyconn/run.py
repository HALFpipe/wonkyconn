from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import WonkyConnConfig
from .workflow import gc_log, workflow


def global_parser(exit_on_error: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=("Evaluating the residual motion in fMRI connectome and visualize reports"),
        exit_on_error=exit_on_error,
    )

    # BIDS app required arguments
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help="The directory with the input dataset (e.g. fMRIPrep derivative)formatted according to the BIDS standard",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only group level is available",
        choices=["group"],
    )

    parser.add_argument(
        "--phenotypes",
        type=str,
        help=(
            "Path to the phenotype file that has the columns `participant_id`, "
            "`gender` coded as `M` and `F` and `age` in years"
        ),
        required=True,
    )
    parser.add_argument(
        "--atlas",
        type=str,
        nargs=2,
        action="append",
        metavar=("LABEL", "ATLAS_PATH"),
        required=True,
        help="Specify the atlas label and the path to the atlas file (for example --atlas Schaefer2018 /path/to/atlas.nii.gz)",
    )

    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--light-mode",
        required=False,
        action="store_true",
        default=False,
        help="Disable sex and age prediction to reduce runtime.",
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
    parser.add_argument(
        "--textual",
        action="store_true",
        help="Launch the Textual UI to configure and run wonkyconn.",
    )
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress Python RuntimeWarnings during processing.",
    )
    return parser


def _loosen_parser_for_gui(parser: argparse.ArgumentParser) -> None:
    """Allow partial parsing when launching the GUI so we can prefill fields."""
    for action in parser._actions:
        # Positional arguments have empty option_strings
        if not action.option_strings:
            action.nargs = "?"
            action.default = None
        else:
            action.required = False


def _build_initial_config(args: argparse.Namespace | None) -> WonkyConnConfig:
    return WonkyConnConfig.from_cli_args(args)


def _run_textual_ui(config: WonkyConnConfig) -> WonkyConnConfig | None:
    if not sys.stdout.isatty():
        print("The Textual UI requires an interactive terminal; run without --textual instead.", file=sys.stderr)
        sys.exit(2)

    try:
        from .textual_app import WonkyConnApp
    except ModuleNotFoundError as exc:  # textual is optional
        missing_mod = getattr(exc, "name", "") or ""
        if missing_mod.startswith("textual"):
            print(
                'The Textual UI requires the optional dependency "textual". Install it with `pip install "wonkyconn[textual]"`.',
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    app = WonkyConnApp(config)
    return app.run()


def main(argv: None | Sequence[str] = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    pre_parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    pre_parser.add_argument("--textual", action="store_true")
    pre_parsed, _ = pre_parser.parse_known_args(raw_args)
    use_gui = pre_parsed.textual

    parser = global_parser(exit_on_error=not use_gui)
    if use_gui:
        _loosen_parser_for_gui(parser)

    parsed_args: argparse.Namespace | None
    try:
        parsed_args = parser.parse_args(raw_args)
    except SystemExit as exc:
        if use_gui and exc.code != 0:
            parsed_args = None
        else:
            raise
    except argparse.ArgumentError:
        if not use_gui:
            raise
        parsed_args = None

    if use_gui:
        initial_config = _build_initial_config(parsed_args)
        result_config = _run_textual_ui(initial_config)
        if result_config is None:
            return
        args_for_workflow = result_config.to_namespace()
        debug_enabled = result_config.debug
        suppress_warnings = result_config.suppress_warnings
    else:
        config = _build_initial_config(parsed_args)
        args_for_workflow = config.to_namespace()
        debug_enabled = config.debug
        suppress_warnings = config.suppress_warnings

    if suppress_warnings:
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        workflow(args_for_workflow)
    except Exception as e:
        gc_log.exception("Exception: %s", e, exc_info=True)
        if debug_enabled:
            import pdb

            pdb.post_mortem()


if __name__ == "__main__":
    main(sys.argv[1:])

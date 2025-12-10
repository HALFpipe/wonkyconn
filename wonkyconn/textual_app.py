from __future__ import annotations

from pathlib import Path
from typing import Iterable

from textual.app import App, ComposeResult
from textual.containers import Center, Container, Horizontal, Vertical
from textual.widgets import Button, Checkbox, DirectoryTree, Footer, Header, Input, Label, Select, Static

from .config import WonkyConnConfig


class WonkyConnApp(App[WonkyConnConfig | None]):
    """Textual UI for configuring wonkyconn."""

    CSS = """
    Screen {
        layout: vertical;
        overflow: auto;
    }

    #intro {
        height: 1fr;
        padding: 2;
        align: center middle;
    }

    #intro-card {
        border: solid $accent 20%;
        padding: 2;
        width: 80%;
        height: auto;
        text-align: center;
        align: center middle;
    }

    #intro-title {
        text-style: bold;
    }

    #intro-text {
        padding-top: 1;
        padding-bottom: 1;
        text-align: center;
    }

    #main {
        layout: vertical;
        height: 1fr;
        padding: 0;
        overflow: auto;
    }

    .full-width {
        width: 100%;
    }

    .path-display {
        text-wrap: wrap;
        content-align: left middle;
    }

    #tree-container {
        min-height: 12;
        border: solid $accent 10%;
        padding: 0;
        overflow: auto;
    }

    #selection-controls {
        layout: horizontal;
        align: center middle;
    }

    #run-controls {
        layout: horizontal;
    }

    #status {
        height: 2;
        border: solid $accent 10%;
        padding: 0 1;
    }

    .status-error {
        color: $error;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel"), ("ctrl+s", "run", "Run")]

    def __init__(self, initial_config: WonkyConnConfig):
        super().__init__()
        self.initial_config = initial_config
        self.selected_path: Path | None = None
        self._path_values: dict[str, str] = dict()
        self._active_target: str = "bids_dir"
        # Default theme flag to avoid attribute errors before on_mount runs.
        self.dark = False

    def _tree_root(self) -> Path:
        candidates: Iterable[Path | None] = (
            self.initial_config.bids_dir,
            Path.cwd(),
            Path.home(),
        )
        for candidate in candidates:
            if candidate and candidate.exists() and candidate.is_dir():
                return candidate
        for fallback in (Path.cwd(), Path.home()):
            if fallback.exists() and fallback.is_dir():
                return fallback
        return Path.home()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="intro"):
            with Center():
                with Vertical(id="intro-card"):
                    with Center():
                        yield Label("Wonkyconn", id="intro-title")
                    with Center():
                        yield Static(
                            "Evaluating residual motion in fMRI connectomes.\n"
                            "Contributors: Hao-Ting Wang, Lea Waller, Clara El Khantour, Pierre Bergeret, Seann Wang.\n"
                            "Acknowledgements: SIMEXP and upstream tools (fMRIPrep, Halfpipe, Nilearn, etc.).",
                            id="intro-text",
                        )
                    with Center():
                        yield Button("Start", id="start-app", variant="primary")
        with Container(id="main"):
            # Tree at the top
            with Vertical(id="tree-container"):
                yield Label("Browse file system")
                yield DirectoryTree(self._tree_root(), id="file-tree")

            # Target selection and apply
            with Horizontal(id="selection-controls"):
                yield Select(
                    options=[
                        ("BIDS directory", "bids_dir"),
                        ("Output directory", "output_dir"),
                        ("Phenotypes TSV", "phenotypes"),
                        ("Atlas path", "atlas_path"),
                    ],
                    id="target-select",
                )
                yield Button("Use selection", id="use-focused", variant="primary")

            # Required paths
            with Vertical(classes="section"):
                yield Label("Required paths (click a field to target selection)")
                yield Button("BIDS directory: (not set)", id="bids_dir_display", variant="default", classes="full-width path-display")
                yield Button("Output directory: (not set)", id="output_dir_display", variant="default", classes="full-width path-display")
                yield Button("Phenotypes TSV: (not set)", id="phenotypes_display", variant="default", classes="full-width path-display")

            # Atlas
            with Vertical(classes="section"):
                yield Label("Atlas")
                yield Input(placeholder="Atlas label (e.g., Schaefer20187Networks400Parcels)", id="atlas_name")
                yield Button("Atlas path: (not set)", id="atlas_path_display", variant="default", classes="full-width path-display")

            # Options
            with Vertical(classes="section"):
                yield Label("Options")
                with Horizontal():
                    yield Select(
                        options=[
                            ("Errors only (0)", "0"),
                            ("Warnings (1)", "1"),
                            ("Info (2)", "2"),
                            ("Debug (3)", "3"),
                        ],
                        id="verbosity",
                    )
                    yield Checkbox("Debug", id="debug")
                    yield Checkbox("Suppress warnings", id="suppress_warnings")

            # Run/cancel
            with Horizontal(id="run-controls"):
                yield Button("Run", id="run", variant="success")
                yield Button("Cancel", id="cancel", variant="error")

            yield Static(
                "Pick a target (or click a required field), select a path in the tree, then press Use selection.",
                id="tree-help",
            )
            yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._load_initial_values()
        if self.initial_config.theme:
            self.dark = self.initial_config.theme == "dark"
        intro = self.query_one("#intro", Container)
        main = self.query_one("#main", Container)
        intro.display = True
        main.display = False

    def _apply_selection_to_path_display(self, field: str) -> None:
        if self.selected_path is None:
            self._set_status("Select a file or directory in the tree first.", error=True)
            return

        expects_dir = field in {"bids_dir", "output_dir"}
        if expects_dir and not self.selected_path.is_dir():
            self._set_status("Please select a directory for this field.", error=True)
            return
        if not expects_dir and not self.selected_path.is_file():
            self._set_status("Please select a file for this field.", error=True)
            return

        value = str(self.selected_path)
        self._path_values[field] = value
        if field == "atlas_path":
            self.query_one("#atlas_path_display", Button).label = f"Atlas path: {value}"
        else:
            display = self.query_one(f"#{field}_display", Button)
            display.label = f"{field.replace('_', ' ').title()}: {value}"
        self._set_status("Applied selected path.", error=False)

        # If this is an atlas path target (dynamic), also update the input value
        if field.startswith("atlas-") and field.endswith("-path"):
            try:
                target_input = self.query_one(f"#{field}", Input)
                target_input.value = value
            except Exception:
                pass

    def _apply_selection_to_target(self) -> None:
        """Apply selection strictly based on the target dropdown."""
        target_select = self.query_one("#target-select", Select)
        target_value = target_select.value or ""

        if target_value in {"bids_dir", "output_dir", "phenotypes", "atlas_path"}:
            self._apply_selection_to_path_display(target_value)
            return

        # No valid target selected
        self._set_status("Select a target in the dropdown before applying.", error=True)

    def _load_initial_values(self) -> None:
        if self.initial_config.bids_dir:
            value = str(self.initial_config.bids_dir)
            self._path_values["bids_dir"] = value
            self.query_one("#bids_dir_display", Button).label = f"Bids Dir: {value}"
        if self.initial_config.output_dir:
            value = str(self.initial_config.output_dir)
            self._path_values["output_dir"] = value
            self.query_one("#output_dir_display", Button).label = f"Output Dir: {value}"
        if self.initial_config.phenotypes:
            value = str(self.initial_config.phenotypes)
            self._path_values["phenotypes"] = value
            self.query_one("#phenotypes_display", Button).label = f"Phenotypes Tsv: {value}"
        if self.initial_config.atlas:
            label, path = self.initial_config.atlas[0]
            self.query_one("#atlas_name", Input).value = label
            self._path_values["atlas_path"] = str(path)
            self.query_one("#atlas_path_display", Button).label = f"Atlas Path: {path}"

        self.query_one("#verbosity", Select).value = str(self.initial_config.verbosity)
        self.query_one("#debug", Checkbox).value = bool(self.initial_config.debug)
        self.query_one("#suppress_warnings", Checkbox).value = bool(self.initial_config.suppress_warnings)

    def _set_status(self, message: str, error: bool = False) -> None:
        status = self.query_one("#status", Static)
        status.update(message)
        status.remove_class("status-error")
        if error:
            status.add_class("status-error")

    def _validate_and_build_config(self) -> WonkyConnConfig | None:
        errors: list[str] = list()

        bids_str = self._path_values.get("bids_dir", "").strip()
        output_str = self._path_values.get("output_dir", "").strip()
        phenotypes_str = self._path_values.get("phenotypes", "").strip()
        atlas_label = self.query_one("#atlas_name", Input).value.strip()
        atlas_path_str = self._path_values.get("atlas_path", "").strip()

        bids_dir = Path(bids_str).expanduser().resolve() if bids_str else None
        output_dir = Path(output_str).expanduser().resolve() if output_str else None
        phenotypes = Path(phenotypes_str).expanduser().resolve() if phenotypes_str else None
        atlas_path = Path(atlas_path_str).expanduser().resolve() if atlas_path_str else None

        if bids_dir is None:
            errors.append("BIDS directory is required.")
        elif not bids_dir.exists() or not bids_dir.is_dir():
            errors.append(f"BIDS directory must exist and be a directory: {bids_dir}")

        if output_dir is None:
            errors.append("Output directory is required.")
        elif output_dir.exists() and not output_dir.is_dir():
            errors.append(f"Output directory must be a directory: {output_dir}")

        if phenotypes is None:
            errors.append("Phenotypes TSV is required.")
        elif not (phenotypes.exists() or phenotypes.is_symlink()) or (phenotypes.exists() and not phenotypes.is_file()):
            errors.append(f"Phenotypes file must exist: {phenotypes}")

        if not atlas_label:
            errors.append("Atlas label is required.")
        if atlas_path is None:
            errors.append("Atlas path is required.")
        elif not (atlas_path.exists() or atlas_path.is_symlink()) or (atlas_path.exists() and not atlas_path.is_file()):
            errors.append(f"Atlas file must exist: {atlas_path}")

        verbosity_str = self.query_one("#verbosity", Select).value or "2"
        verbosity = int(verbosity_str)
        debug = self.query_one("#debug", Checkbox).value
        suppress_warnings = self.query_one("#suppress_warnings", Checkbox).value

        if errors:
            self._set_status("\n".join(errors), error=True)
            return None

        config = WonkyConnConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis_level="group",
            phenotypes=phenotypes,
            atlas=[(atlas_label, atlas_path)],
            verbosity=verbosity,
            debug=debug,
            theme="dark" if self.dark else "light",
            suppress_warnings=suppress_warnings,
        )
        self._set_status("Configuration validated. Running...", error=False)
        return config

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id in {"bids_dir_display", "output_dir_display", "phenotypes_display"}:
            self._active_target = button_id.replace("_display", "")
            target_select = self.query_one("#target-select", Select)
            target_select.value = self._active_target
            self._set_status(f"Target set to {self._active_target.replace('_', ' ')}", error=False)
        elif button_id == "atlas_path_display":
            target_select = self.query_one("#target-select", Select)
            target_select.value = "atlas_path"
            self._set_status("Target set to atlas path", error=False)
        elif button_id == "use-focused":
            self._apply_selection_to_target()
        elif button_id == "start-app":
            self._set_status("Welcome to wonkyconn. Configure paths below.", error=False)
            intro = self.query_one("#intro", Container)
            main = self.query_one("#main", Container)
            intro.display = False
            main.display = True
        elif button_id == "run":
            config = self._validate_and_build_config()
            if config:
                self.exit(config)
        elif button_id == "cancel":
            self.exit(None)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.selected_path = event.path
        self._set_status(f"Selected file: {event.path}", error=False)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.selected_path = event.path
        self._set_status(f"Selected directory: {event.path}", error=False)

    def on_directory_tree_node_highlighted(self, event: DirectoryTree.NodeHighlighted) -> None:
        # Capture single-click highlight so a file selection is remembered before pressing the button
        self.selected_path = event.path
        kind = "directory" if event.path.is_dir() else "file"
        self._set_status(f"Highlighted {kind}: {event.path}", error=False)

    def action_run(self) -> None:
        config = self._validate_and_build_config()
        if config:
            self.exit(config)

    def action_cancel(self) -> None:
        self.exit(None)

    def on_focus(self, event) -> None:  # type: ignore[override]
        """Track the last focused input, regardless of widget type."""
        from textual.events import Focus

        if not isinstance(event, Focus):
            return
        if isinstance(event.sender, Input):
            self._last_input = event.sender

    def on_input_clicked(self, event: Input.Clicked) -> None:
        self._last_input = event.input

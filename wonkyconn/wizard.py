from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any, Optional

from .logger import gc_log
from .file_index.bids import BIDSIndex

try:  # Optional richer TTY experience
    from prompt_toolkit.application import Application
    from prompt_toolkit.completion import PathCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window, FloatContainer, Float
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.shortcuts import PromptSession
    from prompt_toolkit.shortcuts.prompt import CompleteStyle
    from prompt_toolkit.shortcuts.dialogs import input_dialog, yes_no_dialog
    from prompt_toolkit.styles import Style

    _PROMPT_TOOLKIT_AVAILABLE = True
except Exception:  # pragma: no cover - prompt_toolkit not installed
    Application = None  # type: ignore[assignment]
    PathCompleter = None  # type: ignore[assignment]
    PromptSession = None  # type: ignore[assignment]
    CompleteStyle = None  # type: ignore[assignment]
    FormattedText = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]
    Layout = None  # type: ignore[assignment]
    HSplit = None  # type: ignore[assignment]
    Window = None  # type: ignore[assignment]
    FormattedTextControl = None  # type: ignore[assignment]
    input_dialog = None  # type: ignore[assignment]
    yes_no_dialog = None  # type: ignore[assignment]
    Style = None  # type: ignore[assignment]

    _PROMPT_TOOLKIT_AVAILABLE = False


def _supports_rich_ui() -> bool:
    return _PROMPT_TOOLKIT_AVAILABLE and PromptSession is not None


class WizardAbort(RuntimeError):
    """Raised when the user aborts the interactive wizard."""


@dataclass
class WizardConfig:
    bids_dir: Path
    output_dir: Path
    analysis_level: str
    phenotypes: Path
    atlas_label: str
    atlas_path: Path
    group_by: list[str]
    verbosity: int
    debug: bool
    wizard_theme: str

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            bids_dir=self.bids_dir,
            output_dir=self.output_dir,
            analysis_level=self.analysis_level,
            phenotypes=str(self.phenotypes),
            atlas=[self.atlas_label, str(self.atlas_path)],
            group_by=self.group_by,
            verbosity=[self.verbosity],
            debug=self.debug,
            wizard=False,
            wizard_theme=self.wizard_theme,
        )

    def to_defaults_namespace(self) -> argparse.Namespace:
        namespace = self.to_namespace()
        if hasattr(namespace, "wizard"):
            delattr(namespace, "wizard")
        return namespace

    def to_dict(self) -> dict[str, Any]:
        return {
            "bids_dir": str(self.bids_dir),
            "output_dir": str(self.output_dir),
            "analysis_level": self.analysis_level,
            "phenotypes": str(self.phenotypes),
            "atlas_label": self.atlas_label,
            "atlas_path": str(self.atlas_path),
            "group_by": list(self.group_by),
            "verbosity": int(self.verbosity),
            "debug": bool(self.debug),
            "wizard_theme": self.wizard_theme,
        }

    def to_file(self, path: Path) -> None:
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WizardConfig":
        try:
            atlas_label = payload["atlas_label"]
            atlas_path = payload["atlas_path"]
            bids_dir = payload["bids_dir"]
            output_dir = payload["output_dir"]
            phenotypes = payload["phenotypes"]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Configuration is missing required field: {exc.args[0]}") from exc

        analysis_level = payload.get("analysis_level", "group")
        group_by_raw = payload.get("group_by", ["seg"])
        if isinstance(group_by_raw, (str, Path)):
            group_by = [str(group_by_raw)]
        else:
            group_by = [str(item) for item in group_by_raw]

        verbosity_raw = payload.get("verbosity", 2)
        verbosity = int(verbosity_raw)

        return cls(
            bids_dir=Path(bids_dir),
            output_dir=Path(output_dir),
            analysis_level=str(analysis_level),
            phenotypes=Path(phenotypes),
            atlas_label=str(atlas_label),
            atlas_path=Path(atlas_path),
            group_by=group_by,
            verbosity=verbosity,
            debug=bool(payload.get("debug", False)),
            wizard_theme=str(payload.get("wizard_theme", "light")),
        )

    @classmethod
    def from_file(cls, path: Path) -> "WizardConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "WizardConfig":
        atlas = getattr(args, "atlas", None)
        if not isinstance(atlas, (list, tuple)) or len(atlas) != 2:
            raise ValueError("Namespace is missing atlas label/path pair")

        verbosity_raw = getattr(args, "verbosity", 2)
        if isinstance(verbosity_raw, list):
            verbosity = int(verbosity_raw[0]) if verbosity_raw else 2
        else:
            verbosity = int(verbosity_raw)

        theme = getattr(args, "wizard_theme", None)
        wizard_theme = str(theme) if theme else "light"

        group_by_raw = getattr(args, "group_by", None)
        if isinstance(group_by_raw, (list, tuple)):
            group_by = [str(item) for item in group_by_raw]
        elif group_by_raw is None:
            group_by = ["seg"]
        else:
            group_by = [str(group_by_raw)]

        analysis_level = getattr(args, "analysis_level", "group") or "group"

        return cls(
            bids_dir=Path(getattr(args, "bids_dir")),
            output_dir=Path(getattr(args, "output_dir")),
            analysis_level=str(analysis_level),
            phenotypes=Path(getattr(args, "phenotypes")),
            atlas_label=str(atlas[0]),
            atlas_path=Path(atlas[1]),
            group_by=group_by,
            verbosity=verbosity,
            debug=bool(getattr(args, "debug", False)),
            wizard_theme=wizard_theme,
        )


_EXIT_CHOICES = {"q", "quit", "exit"}

_BROWSER_ENTRY_LIMIT = 40

if _PROMPT_TOOLKIT_AVAILABLE and Style is not None:  # pragma: no branch
    _THEME_STYLES = {
        "light": Style.from_dict(
            {
                "browser.background": "bg:#bed8ff",
                "browser.body": "bg:#f8fbff #18345b",
                "browser.highlight": "bg:#3c7ee6 #ffffff",
                "browser.title": "bold fg:#1c3669",
                "browser.path": "fg:#31456b",
                "browser.hint": "fg:#56627d",
                "browser.error": "fg:#b3392f",
                "browser.frame": "fg:#1c3669",
            }
        ),
        "dark": Style.from_dict(
            {
                "browser.background": "bg:#050505",
                "browser.body": "bg:#000000 #ffffff",
                "browser.highlight": "bg:#ffffff #000000",
                "browser.title": "bold fg:#00ffcc",
                "browser.path": "fg:#bbbbbb",
                "browser.hint": "fg:#777777",
                "browser.error": "fg:#ff6666",
                "browser.frame": "fg:#999999",
            }
        ),
    }
    _DIALOG_STYLES = {
        "light": Style.from_dict(
            {
                "dialog": "bg:#f8fbff #18345b",
                "dialog frame.label": "bg:#1c3669 #ffffff",
                "dialog.body": "bg:#f8fbff #18345b",
                "dialog.shadow": "bg:#93b9e6",
                "button": "bg:#e0eaff #18345b",
                "button.focused": "bg:#3c7ee6 #ffffff",
                "text-area": "bg:#ffffff #18345b",
            }
        ),
        "dark": Style.from_dict(
            {
                "dialog": "bg:#101010 #ffffff",
                "dialog frame.label": "bg:#00ffcc #000000",
                "dialog.body": "bg:#101010 #ffffff",
                "dialog.shadow": "bg:#000000",
                "button": "bg:#303030 #cccccc",
                "button.focused": "bg:#ffffff #000000",
                "text-area": "bg:#1a1a1a #ffffff",
            }
        ),
    }
else:
    _THEME_STYLES = {}
    _DIALOG_STYLES = {}

_ACTIVE_MENU_STYLE: Optional[Style] = None
_ACTIVE_THEME_NAME: str = "light"
_ACTIVE_DIALOG_STYLE: Optional[Style] = None


def _normalize_theme(value: Any) -> Optional[str]:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("light", "dark"):
            return lowered
    return None


def _set_active_menu_style(theme: str) -> str:
    global _ACTIVE_MENU_STYLE, _ACTIVE_THEME_NAME, _ACTIVE_DIALOG_STYLE
    normalized = _normalize_theme(theme) or "light"
    _ACTIVE_MENU_STYLE = _THEME_STYLES.get(normalized)
    _ACTIVE_DIALOG_STYLE = _DIALOG_STYLES.get(normalized)
    _ACTIVE_THEME_NAME = normalized
    return normalized


@dataclass
class _MenuItem:
    action: str
    payload: Optional[Path]
    label: str


def collect_configuration(
    cli_args: argparse.Namespace | None = None,
    saved_config: WizardConfig | None = None,
) -> WizardConfig:
    defaults = argparse.Namespace()
    if saved_config is not None:
        for key, value in vars(saved_config.to_defaults_namespace()).items():
            setattr(defaults, key, value)

    if cli_args is not None:
        if getattr(cli_args, "bids_dir", None) is not None:
            defaults.bids_dir = getattr(cli_args, "bids_dir")
        if getattr(cli_args, "output_dir", None) is not None:
            defaults.output_dir = getattr(cli_args, "output_dir")
        if getattr(cli_args, "phenotypes", None) is not None:
            defaults.phenotypes = getattr(cli_args, "phenotypes")
        atlas_value = getattr(cli_args, "atlas", None)
        if atlas_value:
            defaults.atlas = atlas_value
        verbosity_value = getattr(cli_args, "verbosity", None)
        if isinstance(verbosity_value, list):
            defaults.verbosity = verbosity_value
        if getattr(cli_args, "debug", False):
            defaults.debug = True
        theme_value = getattr(cli_args, "wizard_theme", None)
        if theme_value:
            defaults.wizard_theme = theme_value
    gc_log.info("Launching wonkyconn wizard mode")

    theme_pref = _normalize_theme(getattr(defaults, "wizard_theme", None))
    _set_active_menu_style(theme_pref or "light")
    if theme_pref is None:
        use_dark = _prompt_bool("Use dark theme?", default=False)
        theme_pref = "dark" if use_dark else "light"
    theme_pref = _set_active_menu_style(theme_pref)

    bids_dir = _prompt_path(
        "Enter the path to the fMRIPrep-derived BIDS directory",
        default=getattr(defaults, "bids_dir", None),
        must_exist=True,
        path_type="dir",
    )
    bids_index = _build_bids_index(bids_dir)
    available_seg_labels = sorted(bids_index.get_tag_values("seg"))
    output_dir = _prompt_path(
        "Enter the output directory for reports",
        default=getattr(defaults, "output_dir", None),
        must_exist=False,
        path_type="dir",
    )

    advanced = _prompt_bool("Show advanced options (group-by tags, custom verbosity)?", default=False)

    phenotypes = _prompt_path_with_candidates(
        "Select participants.tsv file",
        base_dir=bids_dir,
        patterns=["**/participants.tsv"],
        default=getattr(defaults, "phenotypes", None),
        must_exist=True,
        path_type="file",
    )

    atlas_default_label, atlas_default_path = _unpack_atlas_default(getattr(defaults, "atlas", None))
    atlas_path = _prompt_path_with_candidates(
        "Select atlas file",
        base_dir=bids_dir,
        patterns=[
            "**/*_dseg.nii.gz",
            "**/*atlas*.nii.gz",
        ],
        default=atlas_default_path,
        must_exist=True,
        path_type="file",
    )
    atlas_label_default = _infer_atlas_label(atlas_path, atlas_default_label)
    print(f"Detected atlas label: {atlas_label_default}")
    if advanced:
        atlas_label = _prompt_atlas_label(atlas_label_default, available_seg_labels)
    else:
        atlas_label = atlas_label_default

    if advanced:
        group_by = _prompt_group_by(getattr(defaults, "group_by", ["seg"]))
        verbosity_default = _extract_verbosity_default(getattr(defaults, "verbosity", 2))
        verbosity = _prompt_choice(
            "Choose verbosity level",
            choices=[0, 1, 2, 3],
            default=verbosity_default,
        )
    else:
        existing_group = getattr(defaults, "group_by", None)
        if isinstance(existing_group, (list, tuple)):
            default_group = list(existing_group)
        elif isinstance(existing_group, str):
            default_group = [existing_group]
        elif existing_group is None:
            default_group = ["seg"]
        else:
            default_group = [str(existing_group)]
        group_by = default_group or ["seg"]
        verbosity = _extract_verbosity_default(getattr(defaults, "verbosity", 2))

    debug_default = bool(getattr(defaults, "debug", False))
    debug = _prompt_bool("Enable post-mortem debugger on crash?", default=debug_default)

    config = WizardConfig(
        bids_dir=bids_dir,
        output_dir=output_dir,
        analysis_level="group",
        phenotypes=phenotypes,
        atlas_label=atlas_label,
        atlas_path=atlas_path,
        group_by=group_by,
        verbosity=verbosity,
        debug=debug,
        wizard_theme=theme_pref,
    )

    _display_summary(config)

    if not _prompt_bool("Proceed with these settings?", default=True):
        raise WizardAbort("user cancelled at confirmation")

    gc_log.info("Wizard configuration confirmed")
    return config


def _prompt_value(prompt: str, default: str | Path | None = None) -> str:
    value = _prompt_text(prompt, default=default, allow_empty=True)
    if value == "" and default not in (None, ""):
        return str(default)
    return value


def _prompt_required_value(prompt: str, default: str | Path | None = None) -> str:
    return _prompt_text(prompt, default=default, allow_empty=False)


def _prompt_text(
    prompt: str,
    default: str | Path | None = None,
    allow_empty: bool = True,
) -> str:
    default_str = "" if default in (None, "") else str(default)
    if _supports_rich_ui() and input_dialog is not None and _ACTIVE_DIALOG_STYLE is not None:
        message = prompt
        while True:
            result = input_dialog(
                title="wonkyconn",
                text=message,
                default=default_str,
                style=_ACTIVE_DIALOG_STYLE,
            ).run()
            if result is None:
                raise WizardAbort("prompt cancelled")
            if result or allow_empty:
                return result
            message = f"{prompt}\n\nValue cannot be empty."
    else:
        suffix = f" [{default}]" if default not in (None, "") else ""
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise WizardAbort("prompt cancelled") from exc
        if raw.lower() in _EXIT_CHOICES:
            raise WizardAbort("user requested exit")
        if not raw and default_str and allow_empty:
            return default_str
        if raw or allow_empty:
            return raw
        print("This value cannot be empty. Enter 'quit' to exit the wizard.")
        return _prompt_text(prompt, default=default, allow_empty=allow_empty)


def _prompt_path(
    prompt: str,
    default: str | Path | None,
    must_exist: bool,
    path_type: str | None,
) -> Path:
    default_path: Path | None = None
    if isinstance(default, (str, Path)) and default:
        default_path = Path(default).expanduser()

    interactive_attempted = False
    while True:
        if not interactive_attempted and _supports_rich_ui():
            start_dir = _determine_start_dir(default_path, fallback=None)
            try:
                selection = _interactive_path_browser(
                    prompt,
                    start_dir=start_dir,
                    path_type=path_type,
                    patterns=None,
                    allow_manual=True,
                    must_exist=must_exist,
                )
            except WizardAbort:
                raise
            interactive_attempted = True
            if selection is not None:
                default_path = selection
                default = str(selection)
                if path_type == "dir" and not must_exist:
                    selection, accepted = _maybe_prompt_new_directory(selection)
                    if not accepted:
                        continue
                validated = _validate_path(selection, must_exist, path_type)
                if validated is not None:
                    return validated
                continue

        value = _prompt_value(prompt, default=default)
        if not value:
            print("A value is required. Enter 'quit' to exit the wizard.")
            continue

        path = Path(value).expanduser()
        if path_type == "dir" and not must_exist:
            path, accepted = _maybe_prompt_new_directory(path)
            if not accepted:
                continue
        validated = _validate_path(path, must_exist, path_type)
        if validated is not None:
            return validated
        # validation failed; loop to prompt again


def _maybe_prompt_new_directory(path: Path) -> tuple[Path, bool]:
    if path.exists() and path.is_file():
        print(f"Selected path {path} is a file. Please choose a directory.")
        return path, False

    if path.exists() and path.is_dir():
        name = _prompt_value(
            "Enter name for new directory inside this location (leave blank to use selected)",
            default="",
        ).strip()
        if name:
            return (path / name).expanduser(), True
        return path, True

    parent = path.parent
    if not parent.exists():
        print(f"Parent directory {parent} does not exist. Please choose another location or create it first.")
        return path, False

    if _prompt_bool(f"Directory {path} does not exist. Create it?", default=True):
        return path, True

    print("Directory not created. Please choose a different location.")
    return path, False


def _prompt_atlas_label(default_label: str, available_labels: list[str]) -> str:
    display_sample = ", ".join(available_labels[:10]) if available_labels else ""
    while True:
        label = _prompt_required_value("Enter an atlas label", default=default_label)
        if not available_labels or label in available_labels:
            return label
        print("Atlas label not found in dataset.")
        if display_sample:
            print(f"Available labels include: {display_sample}{' ...' if len(available_labels) > 10 else ''}")
        print("Press Enter to accept the suggested label or choose one from the dataset.")


def _prompt_group_by(default: Iterable[str]) -> list[str]:
    default_list = list(default)
    default_display = ",".join(default_list)
    response = _prompt_value(
        "Enter group-by tags (comma separated)",
        default=default_display,
    )
    items = [item.strip() for item in response.split(",") if item.strip()]
    return items or default_list or ["seg"]


def _prompt_choice(prompt: str, choices: list[int], default: int) -> int:
    choice_text = "/".join(str(choice) for choice in choices)
    while True:
        raw = _prompt_value(f"{prompt} ({choice_text})", default=str(default))
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a numeric choice.")
            continue

        if value not in choices:
            print(f"Invalid choice: {value}. Valid options: {choice_text}")
            continue

        return value


def _prompt_bool(prompt: str, default: bool) -> bool:
    if _supports_rich_ui() and yes_no_dialog is not None and _ACTIVE_DIALOG_STYLE is not None:
        result = yes_no_dialog(
            title="wonkyconn",
            text=prompt,
            yes_text="Yes",
            no_text="No",
            style=_ACTIVE_DIALOG_STYLE,
        ).run()
        return bool(result)

    default_text = "y" if default else "n"
    while True:
        raw = _prompt_text(f"{prompt} [y/n]", default=default_text, allow_empty=False)
        normalized = raw.lower()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def _display_summary(config: WizardConfig) -> None:
    print("\nCollected configuration:")
    print(f"  BIDS directory: {config.bids_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Analysis level: {config.analysis_level}")
    print(f"  Phenotypes file: {config.phenotypes}")
    print(f"  Atlas label: {config.atlas_label}")
    print(f"  Atlas path: {config.atlas_path}")
    print(f"  Group-by tags: {', '.join(config.group_by)}")
    print(f"  Verbosity: {config.verbosity}")
    print(f"  Debug on crash: {'yes' if config.debug else 'no'}")
    print(f"  Wizard theme: {config.wizard_theme}")


def _extract_verbosity_default(raw: int | list[int]) -> int:
    if isinstance(raw, list):
        return raw[0] if raw else 2
    return int(raw)


def _unpack_atlas_default(raw: list[str] | tuple[str, str] | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    try:
        label, path = raw
    except (TypeError, ValueError):
        return None, None
    return label, path


def _infer_atlas_label(path: Path, fallback: str | None) -> str | None:
    if fallback:
        return fallback
    name = path.stem
    if "_seg-" in name:
        return name.split("_seg-", 1)[1].split("_")[0]
    if "atlas-" in name:
        return name.split("atlas-", 1)[1].split("_")[0]
    return name


def _prompt_path_with_candidates(
    prompt: str,
    base_dir: Path,
    patterns: list[str],
    default: str | Path | None,
    must_exist: bool,
    path_type: str | None,
    limit: int = 15,
) -> Path:
    candidates = _find_candidates(base_dir, patterns, limit=limit)
    default_path: Path | None = None
    if isinstance(default, (str, Path)) and default:
        default_path = Path(default).expanduser()

    if _supports_rich_ui():
        start_dir = _determine_start_dir(default_path, fallback=base_dir if base_dir.exists() else None)
        try:
            selection = _interactive_path_browser(
                prompt,
                start_dir=start_dir,
                path_type=path_type,
                patterns=patterns,
                allow_manual=True,
                must_exist=must_exist,
            )
        except WizardAbort:
            raise

        if selection is not None:
            default_path = selection
            default = str(selection)
            if path_type == "dir" and not must_exist:
                selection, accepted = _maybe_prompt_new_directory(selection)
                if not accepted:
                    # allow user to re-enter selection loop
                    return _prompt_path_with_candidates(
                        prompt,
                        base_dir,
                        patterns,
                        default,
                        must_exist,
                        path_type,
                        limit,
                    )
            validated = _validate_path(selection, must_exist, path_type)
            if validated is not None:
                return validated

    default_display = str(default_path) if default_path else (str(candidates[0]) if candidates else None)

    while True:
        if candidates and not _supports_rich_ui():
            print("\nDetected files inside the BIDS directory:")
            for index, candidate in enumerate(candidates, start=1):
                display = _format_candidate(candidate, base_dir)
                print(f"  [{index}] {display}")
            print("  [0] Enter a different path")
            response = _prompt_value(
                f"{prompt} (choose number or enter path)",
                default=default_display,
            )
            if response.isdigit():
                selection = int(response)
                if selection == 0:
                    manual = _prompt_path(prompt, default=None, must_exist=must_exist, path_type=path_type)
                    return manual
                if 1 <= selection <= len(candidates):
                    chosen = candidates[selection - 1]
                    if path_type == "dir" and not must_exist:
                        chosen, accepted = _maybe_prompt_new_directory(chosen)
                        if not accepted:
                            continue
                    validated = _validate_path(chosen, True, path_type)
                    if validated is not None:
                        return validated
                    continue
                print("Invalid selection; please choose a listed number.")
                continue
            manual_value = response
        else:
            if not candidates and base_dir.exists():
                print("No matching files detected; please enter a path manually.")
            manual_value = _prompt_value(prompt, default=default_display)

        manual_path = Path(manual_value).expanduser()
        if path_type == "dir" and not must_exist:
            manual_path, accepted = _maybe_prompt_new_directory(manual_path)
            if not accepted:
                continue
        validated = _validate_path(manual_path, must_exist, path_type)
        if validated is not None:
            return validated


def _build_bids_index(bids_dir: Path) -> BIDSIndex:
    index = BIDSIndex()
    try:
        index.put(bids_dir)
    except Exception as exc:  # pragma: no cover - defensive logging
        gc_log.warning("Failed to index BIDS directory %s: %s", bids_dir, exc)
    return index


def _interactive_path_browser(
    prompt: str,
    start_dir: Path,
    path_type: str | None,
    patterns: list[str] | None,
    allow_manual: bool,
    must_exist: bool,
) -> Path | None:
    if not _supports_rich_ui():
        return None

    browser = _PathBrowser(prompt, start_dir, path_type, patterns, allow_manual, must_exist)
    return browser.run()


class _PathBrowser:
    def __init__(
        self,
        prompt: str,
        start_dir: Path,
        path_type: str | None,
        patterns: list[str] | None,
        allow_manual: bool,
        must_exist: bool,
    ) -> None:
        self.prompt = prompt
        self.current = start_dir if start_dir.exists() else _determine_start_dir(None, fallback=None)
        self.path_type = path_type
        self.patterns = patterns
        self.allow_manual = allow_manual
        self.must_exist = must_exist
        self.history: list[Path] = []

    def run(self) -> Path | None:
        while True:
            items = self._build_items()
            if not items:
                print("No selectable entries here; please enter a path manually.")
                return None

            action, payload = self._present_menu(items)

            if action == "cancel":
                raise WizardAbort("user cancelled selection")
            if action == "manual":
                return None
            if action == "backspace":
                self._go_back()
                continue
            if action == "select_dir":
                return self.current
            if action == "go_up":
                self._push_history()
                self.current = payload or self.current.parent
                continue
            if action == "enter_dir":
                if payload is not None:
                    self._push_history()
                    self.current = payload
                continue
            if action == "history_back":
                self._go_back()
                continue
            if action == "choose_file":
                return payload

    def _push_history(self) -> None:
        if not self.history or self.history[-1] != self.current:
            self.history.append(self.current)

    def _go_back(self) -> None:
        if self.history:
            self.current = self.history.pop()
            return
        parent = self.current.parent
        if parent != self.current:
            self.current = parent

    def _build_items(self) -> list[_MenuItem]:
        try:
            entries = list(self.current.iterdir())
        except OSError as exc:
            print(f"Unable to access {self.current}: {exc}")
            self._go_back()
            return []

        directories = sorted((p for p in entries if p.is_dir()), key=lambda p: p.name.lower())
        files: list[Path] = []
        if self.path_type == "file":
            files = [p for p in entries if p.is_file() and _matches_patterns(p, self.patterns)]
            files.sort(key=lambda p: p.name.lower())

        items: list[_MenuItem] = []
        if self.path_type == "dir" and (not self.must_exist or self.current.exists()):
            items.append(_MenuItem("select_dir", self.current, "✔ Use this directory"))

        if self.history:
            items.append(_MenuItem("history_back", None, "⟵ Back to previous"))

        parent = self.current.parent
        if parent != self.current:
            items.append(_MenuItem("go_up", parent, ".. (parent directory)"))

        for directory in directories[: _BROWSER_ENTRY_LIMIT]:
            items.append(_MenuItem("enter_dir", directory, f"[DIR] {directory.name}/"))

        if self.path_type == "file":
            for file_path in files[: _BROWSER_ENTRY_LIMIT]:
                items.append(_MenuItem("choose_file", file_path, file_path.name))

        if self.allow_manual:
            items.append(_MenuItem("manual", None, "Type path manually"))

        return items

    def _present_menu(self, items: list[_MenuItem]) -> tuple[str, Optional[Path]]:
        if not _supports_rich_ui() or Application is None:
            return ("manual", None)

        selected = 0

        kb = KeyBindings()

        @kb.add("up")
        def _(event) -> None:
            nonlocal selected
            selected = (selected - 1) % len(items)
            event.app.invalidate()

        @kb.add("down")
        def _(event) -> None:
            nonlocal selected
            selected = (selected + 1) % len(items)
            event.app.invalidate()

        @kb.add("enter")
        def _(event) -> None:
            event.app.exit(result=("select", selected))

        @kb.add("backspace")
        @kb.add("left")
        def _(event) -> None:
            event.app.exit(result=("backspace", None))

        @kb.add("right")
        def _(event) -> None:
            event.app.exit(result=("select", selected))

        @kb.add("c-c")
        @kb.add("escape")
        def _(event) -> None:
            event.app.exit(result=("cancel", None))

        @kb.add("tab")
        def _(event) -> None:
            nonlocal selected
            selected = (selected + 1) % len(items)
            event.app.invalidate()

        body_control = FormattedTextControl(lambda: self._render_menu(items, selected))
        body_window = Window(
            content=body_control,
            style="class:browser.body",
            always_hide_cursor=True,
            dont_extend_width=True,
        )

        root_container = FloatContainer(
            content=Window(style="class:browser.background", char=" "),
            floats=[Float(content=body_window, left=4, right=4, top=1)],
        )

        layout = Layout(root_container)
        app = Application(
            layout=layout,
            key_bindings=kb,
            mouse_support=False,
            full_screen=True,
            style=_ACTIVE_MENU_STYLE or None,
        )

        result = app.run()
        if result is None:
            return ("cancel", None)

        status, value = result
        if status == "select":
            item = items[value]
            return (item.action, item.payload)
        return (status, value)

    def _render_menu(self, items: list[_MenuItem], selected: int) -> FormattedText:
        content: list[tuple[str, str]] = []
        content.append(("class:browser.title", self.prompt))
        content.append(("class:browser.path", f"Current directory: {self.current}"))
        content.append(("class:browser.body", ""))

        for index, item in enumerate(items):
            style = "class:browser.highlight" if index == selected else "class:browser.body"
            pointer = "➤ " if index == selected else "  "
            content.append((style, f"{pointer}{item.label}"))

        hint = "↑/↓ move • Enter select • Backspace go back • Ctrl+C cancel"
        content.append(("class:browser.hint", ""))
        content.append(("class:browser.hint", hint))

        inner_width = max(len(text) for _, text in content) if content else 0
        inner_width = max(inner_width, 40)
        border_line = "┌" + "─" * (inner_width + 2) + "┐"
        bottom_line = "└" + "─" * (inner_width + 2) + "┘"

        rendered: list[tuple[str, str]] = []
        rendered.append(("class:browser.background", "\n"))
        rendered.append(("class:browser.frame", border_line + "\n"))
        for style, text in content:
            padded = text.ljust(inner_width)
            rendered.append(("class:browser.frame", "│ "))
            rendered.append((style, padded))
            rendered.append(("class:browser.frame", " │\n"))
        rendered.append(("class:browser.frame", bottom_line + "\n"))
        rendered.append(("class:browser.background", ""))
        return FormattedText(rendered)

def _determine_start_dir(default_path: Path | None, fallback: Path | None) -> Path:
    if default_path is not None:
        try:
            if default_path.exists():
                return default_path if default_path.is_dir() else default_path.parent
        except OSError:
            pass
        parent = default_path.parent if default_path.parent != default_path else None
        if parent is not None:
            try:
                if parent.exists():
                    return parent
            except OSError:
                pass

    if fallback is not None:
        try:
            if fallback.exists():
                return fallback
        except OSError:
            pass

    try:
        cwd = Path.cwd()
        if cwd.exists():
            return cwd
    except OSError:
        pass

    try:
        home = Path.home()
        if home.exists():
            return home
    except OSError:
        pass

    return Path("/")


def _matches_patterns(path: Path, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(path.match(pattern) for pattern in patterns)


def _find_candidates(base_dir: Path, patterns: Iterable[str], limit: int) -> list[Path]:
    if not base_dir or not base_dir.exists():
        return []

    results: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in base_dir.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append(candidate)
            if len(results) >= limit:
                return results
    return results


def _format_candidate(candidate: Path, base_dir: Path) -> str:
    try:
        return str(candidate.relative_to(base_dir))
    except ValueError:
        return str(candidate)


def _validate_path(path: Path, must_exist: bool, path_type: str | None) -> Path | None:
    if must_exist and not path.exists():
        print(f"Path does not exist: {path}")
        return None
    if path_type == "dir":
        if path.exists() and not path.is_dir():
            print(f"Expected a directory: {path}")
            return None
        if not path.exists() and not must_exist:
            gc_log.info("Directory will be created if it does not exist: %s", path)
    if path_type == "file" and path.exists() and not path.is_file():
        print(f"Expected a file: {path}")
        return None
    if path_type == "file" and must_exist and not path.exists():
        print(f"File not found: {path}")
        return None
    return path

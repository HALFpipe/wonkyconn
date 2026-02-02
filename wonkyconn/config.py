from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def _coerce_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


@dataclass
class WonkyConnConfig:
    """Shared configuration for CLI and GUI."""

    bids_dir: Path | None = None
    output_dir: Path | None = None
    analysis_level: str = "group"
    phenotypes: Path | None = None
    atlas: list[tuple[str, Path]] = field(default_factory=list)
    verbosity: int = 2
    debug: bool = False
    light_mode: bool = False
    theme: str | None = None  # GUI-only
    suppress_warnings: bool = False

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace | None) -> "WonkyConnConfig":
        """Create a config from argparse args (may be partial when GUI is requested)."""
        if args is None:
            return cls()

        verbosity = args.verbosity
        if isinstance(verbosity, Sequence) and not isinstance(verbosity, (str, bytes)):
            verbosity = verbosity[0]

        atlas_entries: list[tuple[str, Path]] = list()
        for label, atlas_path in getattr(args, "atlas", []) or []:
            atlas_entries.append((label, Path(atlas_path).expanduser().resolve()))

        return cls(
            bids_dir=_coerce_path(getattr(args, "bids_dir", None)),
            output_dir=_coerce_path(getattr(args, "output_dir", None)),
            analysis_level=getattr(args, "analysis_level", "group"),
            phenotypes=_coerce_path(getattr(args, "phenotypes", None)),
            atlas=atlas_entries,
            verbosity=int(verbosity) if verbosity is not None else 2,
            debug=bool(getattr(args, "debug", False)),
            light_mode=bool(getattr(args, "light_mode", False)),
            suppress_warnings=bool(getattr(args, "suppress_warnings", False)),
        )

    def to_namespace(self) -> argparse.Namespace:
        """Convert to argparse.Namespace expected by workflow."""
        if self.bids_dir is None:
            raise ValueError("bids_dir is required")
        if self.output_dir is None:
            raise ValueError("output_dir is required")
        if self.phenotypes is None:
            raise ValueError("phenotypes is required")
        if not self.atlas:
            raise ValueError("At least one atlas entry is required")

        atlas_as_str: Iterable[tuple[str, str]] = ((label, str(path)) for label, path in self.atlas)
        return argparse.Namespace(
            bids_dir=self.bids_dir,
            output_dir=self.output_dir,
            analysis_level=self.analysis_level,
            phenotypes=str(self.phenotypes),
            atlas=list(atlas_as_str),
            verbosity=self.verbosity,
            debug=self.debug,
            light_mode=self.light_mode,
        )

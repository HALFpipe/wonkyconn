# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import json
from pathlib import Path
from typing import Any, MutableSequence

from .base import FileIndex


def split_ext(path: str | Path) -> tuple[str, str]:
    """Splits filename and extension (.gz safe)
    >>> split_ext('some/file.nii.gz')
    ('file', '.nii.gz')
    >>> split_ext('some/other/file.nii')
    ('file', '.nii')
    >>> split_ext('otherext.tar.gz')
    ('otherext', '.tar.gz')
    >>> split_ext('text.txt')
    ('text', '.txt')

    Adapted from niworkflows
    """
    from pathlib import Path

    if isinstance(path, str):
        path = Path(path)

    name = str(path.name)

    safe_name = name
    for compound_extension in [".gz", ".xz"]:
        safe_name = safe_name.removesuffix(compound_extension)

    stem = Path(safe_name).stem
    return stem, name[len(stem) :]


def parse(path: Path) -> dict[str, str] | None:
    """
    Parses a BIDS-formatted filename and returns a dictionary of its tags.

    Args:
        path (Path): The path to the file to parse.

    Returns:
        dict[str, str] | None: A dictionary of the file's BIDS tags, or None if the
            file is not a valid BIDS-formatted file.
    """
    if path.is_dir():
        return None  # Skip directories

    stem, extension = split_ext(path)
    if stem.startswith("."):
        return None  # Skip hidden files

    tokens = stem.split("_")

    # Parse tokens
    keys: MutableSequence[str | None] = list()
    values: MutableSequence[str] = list()
    for token in tokens:
        if "-" in token:  # A bids tag
            key: str | None = token.split("-")[0]
            if key is None:
                continue
            keys.append(key)
            values.append(token[len(key) + 1 :])

        else:  # A suffix
            keys.append(None)
            values.append(token)

    # Extract bids suffixes
    suffixes: list[str] = list()
    while keys and keys[-1] is None:
        keys.pop(-1)
        suffixes.insert(0, values.pop(-1))

    # Merge other suffixes with their preceding tag value
    for i, (key, value) in enumerate(zip(keys, values, strict=False)):
        if i < 1:
            continue
        if key is None:
            values[i - 1] += f"_{value}"

    # Build tags
    tags = dict(
        suffix="_".join(suffixes),
    )
    if extension:
        tags["extension"] = extension
    parent_name = Path(str(path.parent)).name
    if parent_name in ("anat", "func", "fmap"):
        tags["datatype"] = parent_name
    for key, value in zip(keys, values, strict=False):
        if key is not None:
            tags[key] = value
    return tags


class BIDSIndex(FileIndex):
    def put(self, root: Path) -> None:
        for path in root.glob("**/*"):
            tags = parse(path)

            if tags is None:
                continue  # not a valid path

            for key, value in tags.items():
                self.paths_by_tags[key][value].add(path)

            self.tags_by_paths[path] = tags

    def get_metadata(self, path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = dict()

        for metadata_path in self.get_associated_paths(
            path, extension=".json"
        ):
            with metadata_path.open("r") as file:
                metadata.update(json.load(file))

        return metadata

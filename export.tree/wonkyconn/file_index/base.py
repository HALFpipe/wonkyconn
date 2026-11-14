# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Mapping


def create_defaultdict_of_set() -> defaultdict[str, set[Path]]:
    return defaultdict(set)


class FileIndex:
    def __init__(self) -> None:
        self.paths_by_tags: dict[str, dict[str, set[Path]]] = defaultdict(create_defaultdict_of_set)
        self.tags_by_paths: dict[Path, dict[str, str]] = defaultdict(dict)

    @property
    def hexdigest(self) -> str:
        """
        A forty character hash code of the paths in the index, obtained using the `sha1` algorithm.
        """
        hash_algorithm = sha1()

        for path in sorted(self.tags_by_paths.keys(), key=str):
            path_bytes = str(path).encode()
            hash_algorithm.update(path_bytes)

        return hash_algorithm.hexdigest()

    def get(self, **tags: str | None) -> set[Path]:
        """
        Find all paths that match the query tags.

        Args:
            **tags: A dictionary of tags to match against. The keys are the tag names
                    and the values are the tag values. Pass a value of `None` to
                    select paths without that tag.

        Returns:
            A set of `Path` objects that match all the specified tags.
        """

        matches: set[Path] | None = None
        for key, value in tags.items():
            if key not in self.paths_by_tags:
                return set()

            values = self.paths_by_tags[key]
            if value is not None:
                if value not in values:
                    return set()
                paths: set[Path] = values[value]
            else:
                paths_in_index = set(self.tags_by_paths.keys())
                paths = paths_in_index.difference(*values.values())

            if matches is not None:
                matches &= paths
            else:
                matches = paths.copy()

        if not matches:
            return set()
        else:
            return matches

    def get_tags(self, path: Path) -> Mapping[str, str | None]:
        if path in self.tags_by_paths:
            return self.tags_by_paths[path]
        else:
            return dict()

    def get_tag_value(self, path: Path, key: str) -> str | None:
        return self.get_tags(path).get(key)

    def set_tag_value(self, path: Path, key: str, value: str) -> None:
        # remove previous value
        if self.get_tag_value(path, key) is not None:
            previous_value = self.tags_by_paths[path].pop(key)
            self.paths_by_tags[key][previous_value].remove(path)
        if value is not None:
            self.tags_by_paths[path][key] = value
            self.paths_by_tags[key][value].add(path)

    def get_tag_mapping(self, key: str) -> Mapping[str, set[Path]]:
        return self.paths_by_tags[key]

    def get_tag_values(self, key: str, paths: set[Path] | None = None) -> set[str]:
        if key not in self.paths_by_tags:
            return set()

        if paths is None:
            return set(self.paths_by_tags[key].keys())

        return set(k for k, v in self.paths_by_tags[key].items() if not paths.isdisjoint(v))

    def get_associated_paths(self, path: Path, **tags: str) -> set[Path]:
        matches = self.get(**tags)
        for key, value in self.get_tags(path).items():
            if key == "extension":
                continue
            valid = self.get(**{key: value}) | self.get(**{key: None})
            matches &= valid
        return matches

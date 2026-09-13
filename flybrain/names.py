"""Turn a human name for a group of neurons into model indices.

Accepted forms (case-sensitive, as in the FlyWire annotations):

    sugar | bitter | water | ir94e | MN9      presets from circuits.py
    LC4  ORN_DM4  DNp09  PAM01                exact cell_type or hemibrain_type
    re:^PAM                                   regex on cell_type / hemibrain_type
    class:motor  class:descending             a whole super_class
    id:720575940660219265,7205759406...       FlyWire root IDs
    LC4+LPLC2                                 union of several specs
    LC4@left   sugar@right                    restrict to one side
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from loguru import logger

from . import circuits
from .anatomy import Anatomy
from .connectome import Connectome


class Names:
    def __init__(self, connectome: Connectome, anatomy: Anatomy) -> None:
        """
        Cache the annotation columns used for lookups.

        :param connectome: Loaded connectome.
        :param anatomy: Loaded anatomy.
        """
        self.connectome = connectome
        self.anatomy = anatomy
        self._cell_type = np.asarray(anatomy.cell_type, dtype=object)
        self._hemibrain_type = np.asarray(anatomy.hemibrain_type, dtype=object)
        self._super_class = np.asarray(anatomy.super_class, dtype=object)
        self._side = np.asarray(anatomy.side, dtype=object)

    def resolve(self, spec: str) -> np.ndarray:
        """
        Turn a spec string into model indices.

        :param spec: Name in any accepted form (see module docstring).
        :return: Sorted array of unique model indices.
        :raises KeyError: if nothing matches.
        """
        spec = spec.strip()
        side: str | None = None
        if "@" in spec:
            spec, side = spec.rsplit("@", 1)
        parts = [part for part in spec.split("+") if part]
        indices = np.unique(np.concatenate([self._resolve_one(part) for part in parts]))
        if side:
            indices = indices[self._side[indices] == side]
        if len(indices) == 0:
            raise KeyError(f"'{spec}' matched no neurons. Try re:<pattern>, or "
                           f"Names.search('{spec}') to see what is available.")
        logger.debug("resolved {!r} -> {} neurons", spec + (f"@{side}" if side else ""), len(indices))
        return indices

    def _resolve_one(self, part: str) -> np.ndarray:
        """
        Resolve a single spec part (no '+' or '@').

        :param part: One preset, cell type, or prefixed pattern.
        :return: Model indices (possibly empty).
        """
        if part in circuits.SETS:
            return self.connectome.index(circuits.SETS[part])
        if part == "MN9":
            return np.array([self.connectome.index(circuits.MN9)])
        if part.startswith("id:"):
            return self.connectome.index([int(token) for token in part[3:].split(",")])
        if part.startswith("class:"):
            return np.flatnonzero(self._super_class == part[6:])
        if part.startswith("re:"):
            pattern = re.compile(part[3:])
            matches = np.array([bool(pattern.search(cell_type) or pattern.search(hemibrain_type))
                                for cell_type, hemibrain_type in zip(self._cell_type, self._hemibrain_type)])
            return np.flatnonzero(matches)
        return np.flatnonzero((self._cell_type == part) | (self._hemibrain_type == part))

    def search(self, text: str, limit: int = 30) -> list[tuple[str, int]]:
        """
        Find cell types whose name contains a substring.

        :param text: Substring to look for.
        :param limit: Maximum number of results.
        :return: (cell type, neuron count) pairs, most numerous first.
        """
        counts: Counter[str] = Counter()
        for cell_type, hemibrain_type in zip(self._cell_type, self._hemibrain_type):
            if text in cell_type:
                counts[cell_type] += 1
            elif text in hemibrain_type:
                counts[hemibrain_type] += 1
        return counts.most_common(limit)

    def label(self, indices: np.ndarray) -> str:
        """
        Short label for a group of neurons.

        :param indices: Model indices.
        :return: The group's most common cell type.
        """
        most_common = Counter(self._cell_type[indices]).most_common(1)[0][0]
        return most_common or "?"

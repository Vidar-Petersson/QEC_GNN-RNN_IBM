# Copied 2025-07-24 from https://github.com/qiskit-community/qopt-best-practices
"""Subset finders. Currently contains reference implementation to find lines."""

from __future__ import annotations
import numpy as np
import rustworkx as rx

from qiskit.transpiler import CouplingMap
from qiskit.providers import Backend
from tqdm import tqdm


def find_lines(length: int, backend: Backend) -> list[int]:
    """Finds all possible lines of length `length` for a specific backend topology.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        The found paths.
    """

    if backend.version == 2:
        coupling_map = CouplingMap(backend.coupling_map)
    else:
        coupling_map = CouplingMap(backend.configuration().coupling_map)

    if not coupling_map.is_symmetric:
        coupling_map.make_symmetric()

    all_paths = rx.all_pairs_all_simple_paths(
        coupling_map.graph,
        min_depth=length,
        cutoff=length,
    ).values()


    seen = set()
    unique_paths = []

    for a in tqdm(all_paths, desc="Searching paths"):
        for b in a:
            for path in a[b]:
                path_list = list(path)
                path_sorted = tuple(sorted(path_list))  # hashable for set
                if path_sorted not in seen:
                    seen.add(path_sorted)
                    unique_paths.append(path_list)

    return unique_paths

    # paths = np.asarray(
    #     [
    #         (list(c), list(sorted(list(c))))
    #         for a in iter(all_paths)
    #         for b in iter(a)
    #         for c in iter(a[b])
    #     ]
    # )

    # # filter out duplicated paths
    # _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)
    # paths = paths[:, 0][unique_indices].tolist()

    # return paths
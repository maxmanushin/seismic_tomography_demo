from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class AcquisitionGeometry:
    """
    Simple acquisition description with sources and receivers on a 2D plane.

    All coordinates are expressed as (x, z) pairs in metres.  The Streamlit UI
    keeps everything in a vertical 2D slice (x horizontal, z positive downward).
    """

    sources: np.ndarray
    receivers: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", np.asarray(self.sources, dtype=float))
        object.__setattr__(self, "receivers", np.asarray(self.receivers, dtype=float))
        if self.sources.ndim != 2 or self.sources.shape[1] != 2:
            raise ValueError("sources must be shaped (N, 2)")
        if self.receivers.ndim != 2 or self.receivers.shape[1] != 2:
            raise ValueError("receivers must be shaped (M, 2)")

    @property
    def n_sources(self) -> int:
        return int(self.sources.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])

    def offsets(self) -> np.ndarray:
        """Return array of horizontal offsets for every source/receiver pair."""
        src_x = self.sources[:, 0][:, np.newaxis]
        rec_x = self.receivers[:, 0][np.newaxis, :]
        return rec_x - src_x

    def pair_indices(self) -> Iterable[Tuple[int, int]]:
        """Iterate over all source/receiver index pairs."""
        return product(range(self.n_sources), range(self.n_receivers))

    def ray_pairs(self) -> Iterator[Tuple[Point, Point]]:
        """Yield coordinate pairs for every source/receiver combination."""
        for s, r in self.pair_indices():
            yield tuple(self.sources[s]), tuple(self.receivers[r])

    @classmethod
    def surface_spread(
        cls,
        n_sources: int,
        n_receivers: int,
        spread_length: float,
        surface_depth: float = 0.0,
    ) -> "AcquisitionGeometry":
        """
        Build a symmetric surface spread with equidistant sources and receivers.
        """

        x_sources = np.linspace(0.0, spread_length, n_sources)
        x_receivers = np.linspace(0.0, spread_length, n_receivers)
        sources = np.column_stack([x_sources, np.full_like(x_sources, surface_depth)])
        receivers = np.column_stack(
            [x_receivers, np.full_like(x_receivers, surface_depth)]
        )
        return cls(sources=sources, receivers=receivers)

    @classmethod
    def crosshole(
        cls,
        n_sources: int,
        n_receivers: int,
        x_left: float,
        x_right: float,
        z_top: float,
        z_bottom: float,
    ) -> "AcquisitionGeometry":
        """
        Build a crosshole geometry: sources along the left borehole, receivers along the right.
        """

        z_sources = np.linspace(z_top, z_bottom, n_sources)
        z_receivers = np.linspace(z_top, z_bottom, n_receivers)
        sources = np.column_stack([np.full_like(z_sources, x_left), z_sources])
        receivers = np.column_stack([np.full_like(z_receivers, x_right), z_receivers])
        return cls(sources=sources, receivers=receivers)

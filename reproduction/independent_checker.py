"""Numerical quadrature checks that do not call the closed-form routines."""

from __future__ import annotations

import numpy as np

from reproduction.core import Model, unnormalized_energy_density


def midpoint_integral(model: Model, grid_limit: float = 11.0, grid_size: int = 401) -> float:
    edges = np.linspace(-grid_limit, grid_limit, grid_size + 1)
    points = 0.5 * (edges[:-1] + edges[1:])
    spacing = float(edges[1] - edges[0])
    total = 0.0
    for first in points:
        for second in points:
            total += unnormalized_energy_density(model, np.array([first, second]))
    return float(total * spacing * spacing)


def normalized_energy_density(
    model: Model, y: np.ndarray, grid_limit: float = 11.0, grid_size: int = 401
) -> float:
    return unnormalized_energy_density(model, y) / midpoint_integral(
        model, grid_limit=grid_limit, grid_size=grid_size
    )


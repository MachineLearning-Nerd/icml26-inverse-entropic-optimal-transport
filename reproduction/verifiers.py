"""Fail-closed claim verifiers and their required negative controls."""

from __future__ import annotations

import numpy as np

from reproduction.core import (
    Model,
    closed_form_z,
    component_weights,
    conditional_density,
    deterministic_model,
    unnormalized_energy_density,
    unnormalized_mixture_density,
)
from reproduction.independent_checker import midpoint_integral, normalized_energy_density


def _relative_error(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-300)


def verify_claim_2() -> dict:
    model = deterministic_model(seed=20261031)
    closed = closed_form_z(model)
    quadrature = midpoint_integral(model)
    relative_error = _relative_error(closed, quadrature)

    # Negative control: omit the quadratic completion term b_m^T A_n b_m.
    wrong_weights = np.empty_like(component_weights(model))
    for m in range(model.m_components):
        for n in range(model.n_components):
            exponent = float(model.a[n] @ model.b[m]) / model.epsilon
            wrong_weights[m, n] = model.v[m] * model.w[n] * np.exp(exponent)
    negative_value = float(wrong_weights.sum())
    negative_relative_error = _relative_error(negative_value, quadrature)

    threshold = 2.0e-5
    negative_threshold = 1.0e-3
    checker_passed = relative_error < threshold
    control_failed_as_intended = negative_relative_error > negative_threshold
    passed = checker_passed and control_failed_as_intended
    return {
        "claim": 2,
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "seed": 20261031,
        "closed_form_z": closed,
        "independent_midpoint_quadrature_z": quadrature,
        "relative_error": relative_error,
        "acceptance_threshold": threshold,
        "checker_passed": checker_passed,
        "negative_control": "omit b_m^T A_n b_m from z_mn exponent",
        "negative_control_z": negative_value,
        "negative_control_relative_error": negative_relative_error,
        "negative_control_failure_threshold": negative_threshold,
        "control_failed_as_intended": control_failed_as_intended,
        "passed": passed,
    }


def _wrong_covariance_mixture(model: Model, y: np.ndarray) -> float:
    from reproduction.core import component_means, gaussian_logpdf

    weights = component_weights(model)
    means = component_means(model)
    total = 0.0
    for m in range(model.m_components):
        for n in range(model.n_components):
            total += weights[m, n] * np.exp(
                gaussian_logpdf(y, means[m, n], model.A[n])
            )
    return float(total)


def verify_claim_3() -> dict:
    model = deterministic_model(seed=20261101)
    points = [
        np.array([0.0, 0.0]),
        np.array([0.25, -0.4]),
        np.array([1.1, 0.7]),
        np.array([-0.9, 1.4]),
    ]
    identity_errors = [
        _relative_error(
            unnormalized_mixture_density(model, point),
            unnormalized_energy_density(model, point),
        )
        for point in points
    ]
    probe = np.array([0.31, -0.27])
    closed_density = conditional_density(model, probe)
    numerical_density = normalized_energy_density(model, probe)
    normalized_relative_error = _relative_error(closed_density, numerical_density)

    wrong_errors = [
        _relative_error(
            _wrong_covariance_mixture(model, point),
            unnormalized_energy_density(model, point),
        )
        for point in points
    ]

    identity_threshold = 5.0e-13
    normalized_threshold = 2.0e-5
    negative_threshold = 1.0e-2
    checker_passed = (
        max(identity_errors) < identity_threshold
        and normalized_relative_error < normalized_threshold
    )
    control_failed_as_intended = max(wrong_errors) > negative_threshold
    passed = checker_passed and control_failed_as_intended
    return {
        "claim": 3,
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "seed": 20261101,
        "pointwise_identity_relative_errors": identity_errors,
        "max_pointwise_identity_relative_error": max(identity_errors),
        "identity_acceptance_threshold": identity_threshold,
        "closed_form_density_at_probe": closed_density,
        "independent_quadrature_density_at_probe": numerical_density,
        "normalized_relative_error": normalized_relative_error,
        "normalized_acceptance_threshold": normalized_threshold,
        "checker_passed": checker_passed,
        "negative_control": "use A_n instead of epsilon*A_n as component covariance",
        "negative_control_pointwise_relative_errors": wrong_errors,
        "negative_control_failure_threshold": negative_threshold,
        "control_failed_as_intended": control_failed_as_intended,
        "passed": passed,
    }


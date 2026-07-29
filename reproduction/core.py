"""Independent NumPy implementation of Propositions 3.1 and 3.2.

This module intentionally does not import the authors' implementation.  It uses
the notation of arXiv:2410.02628v5: cost parameters ``v_m, b_m`` and potential
parameters ``w_n, a_n, A_n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    v: np.ndarray
    b: np.ndarray
    w: np.ndarray
    a: np.ndarray
    A: np.ndarray
    epsilon: float

    @property
    def m_components(self) -> int:
        return int(self.v.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.w.shape[0])


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def gaussian_logpdf(y: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    delta = y - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    dimension = y.size
    quadratic = float(delta @ np.linalg.solve(covariance, delta))
    return float(-0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic))


def cost(model: Model, y: np.ndarray) -> float:
    logits = np.log(model.v) + (model.b @ y) / model.epsilon
    return -model.epsilon * logsumexp(logits)


def potential(model: Model, y: np.ndarray) -> float:
    terms = np.array(
        [
            np.log(model.w[n])
            + gaussian_logpdf(y, model.a[n], model.epsilon * model.A[n])
            for n in range(model.n_components)
        ]
    )
    return model.epsilon * logsumexp(terms)


def unnormalized_energy_density(model: Model, y: np.ndarray) -> float:
    return float(np.exp(-(cost(model, y) - potential(model, y)) / model.epsilon))


def component_weights(model: Model) -> np.ndarray:
    result = np.empty((model.m_components, model.n_components))
    for m in range(model.m_components):
        for n in range(model.n_components):
            b_m = model.b[m]
            exponent = (
                float(b_m @ model.A[n] @ b_m) + 2.0 * float(model.a[n] @ b_m)
            ) / (2.0 * model.epsilon)
            result[m, n] = model.v[m] * model.w[n] * np.exp(exponent)
    return result


def component_means(model: Model) -> np.ndarray:
    result = np.empty((model.m_components, model.n_components, model.a.shape[1]))
    for m in range(model.m_components):
        for n in range(model.n_components):
            result[m, n] = model.a[n] + model.A[n] @ model.b[m]
    return result


def closed_form_z(model: Model) -> float:
    return float(component_weights(model).sum())


def unnormalized_mixture_density(model: Model, y: np.ndarray) -> float:
    weights = component_weights(model)
    means = component_means(model)
    total = 0.0
    for m in range(model.m_components):
        for n in range(model.n_components):
            total += weights[m, n] * np.exp(
                gaussian_logpdf(y, means[m, n], model.epsilon * model.A[n])
            )
    return float(total)


def conditional_density(model: Model, y: np.ndarray) -> float:
    return unnormalized_mixture_density(model, y) / closed_form_z(model)


def deterministic_model(seed: int) -> Model:
    rng = np.random.default_rng(seed)
    dimension = 2
    m_components = 3
    n_components = 2
    v = rng.lognormal(mean=0.0, sigma=0.35, size=m_components)
    b = rng.normal(0.0, 0.7, size=(m_components, dimension))
    w = rng.lognormal(mean=0.0, sigma=0.35, size=n_components)
    a = rng.normal(0.0, 0.7, size=(n_components, dimension))
    A = np.empty((n_components, dimension, dimension))
    for n in range(n_components):
        factor = rng.normal(size=(dimension, dimension))
        A[n] = factor @ factor.T + 0.6 * np.eye(dimension)
    return Model(v=v, b=b, w=w, a=a, A=A, epsilon=0.4)


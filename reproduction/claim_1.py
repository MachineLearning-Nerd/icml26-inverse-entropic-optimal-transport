"""Claim 1: exact loss/inverse-EOT equivalence and full-setting data audit."""

from __future__ import annotations

import math

import sympy as sp
import torch

from src.ebieot.costs.lse import MLPLSECost
from src.ebieot.ebieot_gmm import EbieotGmm
from src.utils.samplers.discrete_ot import OTPlanSampler
from src.utils.samplers.synthetic import swiss_roll_transform


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = torch.zeros((), dtype=loss.dtype)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + (gradient.detach() ** 2).sum()
    return float(torch.sqrt(squared))


def _paper_scale_data(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    dtype = torch.float64
    x_unpaired = torch.randn(1024, 2, dtype=dtype)
    generator = torch.Generator().manual_seed(seed + 1)
    t = 1.5 * torch.pi + 3.0 * torch.pi * torch.rand(
        1024, generator=generator, dtype=dtype
    )
    y_unpaired = swiss_roll_transform(t, generator=generator, noise=0.8) / 7.5

    x_candidates = torch.randn(128, 2, dtype=dtype)
    pair_generator = torch.Generator().manual_seed(seed + 2)
    pair_t = 1.5 * torch.pi + 3.0 * torch.pi * torch.rand(
        128, generator=pair_generator, dtype=dtype
    )
    y_candidates = (
        swiss_roll_transform(pair_t, generator=pair_generator, noise=0.8) / 7.5
    )
    plan = OTPlanSampler(
        method="sinkhorn",
        reg=0.05,
        cost_function="rotation-v2",
        normalize_cost=False,
    )
    x_paired, y_paired = plan.sample_plan(x_candidates, y_candidates)
    return x_paired, y_paired, x_unpaired, y_unpaired


def _paper_scale_model(seed: int, y_for_init: torch.Tensor) -> EbieotGmm:
    torch.manual_seed(seed)
    cost = MLPLSECost(
        log_v_m_hidden_channels=[],
        b_m_hidden_channels=[],
        x_dim=2,
        y_dim=2,
        m_potentials=25,
        epsilon=1.0,
    )
    model = EbieotGmm(
        y_dim=2,
        n_potentials=50,
        cost=cost,
        epsilon=1.0,
        sampling_batch_size=128,
        A_diagonal_init=0.1,
    ).double()
    model.init_a_by_samples(y_for_init[:50])
    return model


def verify_claim_1() -> dict:
    # Route A: an exact symbolic certificate for the final step of Sec. 3.2.
    c_bar, f_bar, log_z_bar, epsilon = sp.symbols(
        "Cbar Fbar logZbar epsilon", nonzero=True
    )
    likelihood = c_bar / epsilon - f_bar / epsilon + log_z_bar
    f_c_bar = -epsilon * log_z_bar
    inverse_eot = c_bar - f_c_bar - f_bar
    certificate = sp.simplify(likelihood - inverse_eot / epsilon)
    wrong_sign = sp.simplify(
        likelihood - (c_bar + f_c_bar - f_bar) / epsilon
    )
    symbolic_passed = certificate == 0
    symbolic_negative_failed = wrong_sign != 0

    # Route B: independent direct GMM log_prob versus the three loss terms at
    # the exact Figure-2 architecture and data counts.
    seed = 20261102
    x_paired, y_paired, x_unpaired, y_unpaired = _paper_scale_data(seed)
    model = _paper_scale_model(seed + 10, y_unpaired)
    epsilon_value = model.epsilon
    log_w_n = model.log_w_n()
    a_n = model.a_n()
    A_n = model.A_n()

    direct_distribution = model.get_conditional_distribution(
        x_paired, log_w_n, a_n, A_n
    )
    direct_nll = -direct_distribution.log_prob(y_paired).mean()
    paired_term = model.cost(x_paired, y_paired).mean() / epsilon_value
    paired_log_v = model.cost.log_v_m(x_paired)
    paired_b = model.cost.b_m(x_paired)
    paired_f_c = model.f_c(log_w_n, a_n, A_n, paired_log_v, paired_b)
    paired_f = model.f(y_paired, log_w_n, a_n, A_n)
    decomposed_nll = paired_term - paired_f_c.mean() / epsilon_value - paired_f.mean() / epsilon_value
    parity_absolute_error = float(torch.abs(direct_nll - decomposed_nll))

    # Route C: the empirical semi-supervised estimator at P=128, Q=R=1024.
    x_log_v = model.cost.log_v_m(x_unpaired)
    x_b = model.cost.b_m(x_unpaired)
    x_f_c = model.f_c(log_w_n, a_n, A_n, x_log_v, x_b)
    y_f = model.f(y_unpaired, log_w_n, a_n, A_n)
    x_marginal_term = -x_f_c.mean() / epsilon_value
    y_marginal_term = -y_f.mean() / epsilon_value
    semi_supervised_loss = paired_term + x_marginal_term + y_marginal_term

    parameters = list(model.parameters())
    gradient_norms = {
        "paired_joint_P128": _gradient_norm(paired_term, parameters),
        "x_marginal_Q1024": _gradient_norm(x_marginal_term, parameters),
        "y_marginal_R1024": _gradient_norm(y_marginal_term, parameters),
        "combined": _gradient_norm(semi_supervised_loss, parameters),
    }
    combined_gradients = torch.autograd.grad(
        semi_supervised_loss, parameters, retain_graph=True, allow_unused=True
    )
    paired_gradients = torch.autograd.grad(
        paired_term, parameters, retain_graph=True, allow_unused=True
    )
    squared_difference = 0.0
    for combined, paired in zip(combined_gradients, paired_gradients):
        if combined is None and paired is None:
            continue
        if combined is None:
            combined = torch.zeros_like(paired)
        if paired is None:
            paired = torch.zeros_like(combined)
        squared_difference += float(((combined - paired) ** 2).sum())
    removed_marginals_gradient_difference = math.sqrt(squared_difference)

    wrong_decomposition = paired_term + paired_f_c.mean() / epsilon_value - paired_f.mean() / epsilon_value
    wrong_sign_absolute_error = float(torch.abs(direct_nll - wrong_decomposition))

    parity_threshold = 2.0e-12
    gradient_threshold = 1.0e-8
    negative_threshold = 1.0e-2
    full_scale_passed = (
        parity_absolute_error < parity_threshold
        and all(value > gradient_threshold for value in gradient_norms.values())
        and removed_marginals_gradient_difference > gradient_threshold
    )
    controls_failed_as_intended = (
        symbolic_negative_failed and wrong_sign_absolute_error > negative_threshold
    )
    passed = symbolic_passed and full_scale_passed and controls_failed_as_intended
    return {
        "claim": 1,
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "source_version": "arXiv:2410.02628v5",
        "source_anchors": ["Section 3.1 Eq. 13", "Section 3.2 Eq. 14"],
        "symbolic_certificate_simplifies_to": str(certificate),
        "symbolic_passed": symbolic_passed,
        "symbolic_negative_control_residual": str(wrong_sign),
        "symbolic_negative_failed_as_intended": symbolic_negative_failed,
        "seed": seed,
        "paper_scale": {"P_paired": 128, "Q_unpaired_x": 1024, "R_unpaired_y": 1024, "N": 50, "M": 25},
        "direct_conditional_nll": float(direct_nll),
        "decomposed_conditional_nll": float(decomposed_nll),
        "direct_parity_absolute_error": parity_absolute_error,
        "direct_parity_threshold": parity_threshold,
        "semi_supervised_loss": float(semi_supervised_loss),
        "gradient_norms": gradient_norms,
        "removed_marginals_gradient_difference": removed_marginals_gradient_difference,
        "gradient_threshold": gradient_threshold,
        "wrong_sign_absolute_error": wrong_sign_absolute_error,
        "negative_control_failure_threshold": negative_threshold,
        "controls_failed_as_intended": controls_failed_as_intended,
        "passed": passed,
    }


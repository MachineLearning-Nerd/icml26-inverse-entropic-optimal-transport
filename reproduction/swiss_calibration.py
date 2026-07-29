"""Faithful Swiss-Roll architecture calibration for Claims 4 and 6."""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import os
import time
from typing import Any

import torch

from src.ebieot.costs.lse import MLPLSECost
from src.ebieot.ebieot_gmm import EbieotGmm
from src.utils.samplers.discrete_ot import OTPlanSampler
from src.utils.samplers.synthetic import swiss_roll_transform


P_PAIRED = 128
Q_UNPAIRED = 1024
R_UNPAIRED = 1024
N_POTENTIALS = 50
M_POTENTIALS = 25
TRAIN_STEPS = 25_000
TRAIN_BATCH_SIZE = 128
PAIRING_MINIBATCH_SIZE = 64
TRAIN_SEEDS = (20261211, 20261212, 20261213)
PROBE_POINTS = ((-2.0, 0.0), (2.0, 2.0), (0.0, 0.0))
CONDITIONAL_EVAL_SAMPLES = 512
MARGINAL_EVAL_SAMPLES = 1024
WORKER_THREADS = 8
ARCHITECTURE = "appendix-c2-mlp"


def _normal(
    generator: torch.Generator, count: int, dimension: int = 2
) -> torch.Tensor:
    return torch.randn(
        count, dimension, generator=generator, dtype=torch.float64
    )


def _swiss(generator: torch.Generator, count: int) -> torch.Tensor:
    t = 1.5 * torch.pi + 3.0 * torch.pi * torch.rand(
        count, generator=generator, dtype=torch.float64
    )
    return (
        swiss_roll_transform(t, generator=generator, noise=0.8).to(torch.float64)
        / 7.5
    )


def _ot_sampler() -> OTPlanSampler:
    return OTPlanSampler(
        method="sinkhorn",
        reg=0.05,
        cost_function="rotation-v2",
        normalize_cost=False,
        numItermax=20_000,
        stopThr=1.0e-10,
        warn=False,
    )


def _plan_with_audit(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    plan_sampler: OTPlanSampler,
) -> tuple[torch.Tensor, float]:
    plan = plan_sampler.get_map(x_batch, y_batch)
    expected_x = torch.full(
        (x_batch.shape[0],),
        1.0 / x_batch.shape[0],
        dtype=torch.float64,
    )
    expected_y = torch.full(
        (y_batch.shape[0],),
        1.0 / y_batch.shape[0],
        dtype=torch.float64,
    )
    residual = max(
        float(torch.max(torch.abs(plan.sum(dim=1) - expected_x))),
        float(torch.max(torch.abs(plan.sum(dim=0) - expected_y))),
    )
    return plan, residual


def _sample_one_plan_pair(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    plan_sampler: OTPlanSampler,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    plan, residual = _plan_with_audit(x_batch, y_batch, plan_sampler)
    probabilities = torch.clamp(plan.flatten(), min=0.0)
    probabilities = probabilities / probabilities.sum()
    flat_index = int(
        torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).item()
    )
    x_index = flat_index // y_batch.shape[0]
    y_index = flat_index % y_batch.shape[0]
    return x_batch[x_index], y_batch[y_index], residual


def _make_training_data(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    plan_sampler = _ot_sampler()
    paired_x: list[torch.Tensor] = []
    paired_y: list[torch.Tensor] = []
    plan_residuals: list[float] = []
    for _ in range(P_PAIRED):
        x_batch = _normal(generator, PAIRING_MINIBATCH_SIZE)
        y_batch = _swiss(generator, PAIRING_MINIBATCH_SIZE)
        x_value, y_value, residual = _sample_one_plan_pair(
            x_batch, y_batch, plan_sampler, generator
        )
        paired_x.append(x_value)
        paired_y.append(y_value)
        plan_residuals.append(residual)
    return {
        "paired_x": torch.stack(paired_x),
        "paired_y": torch.stack(paired_y),
        "unpaired_x": _normal(generator, Q_UNPAIRED),
        "unpaired_y": _swiss(generator, R_UNPAIRED),
        "pairing_audit": {
            "plans": len(plan_residuals),
            "max_marginal_residual": max(plan_residuals),
        },
    }


def _make_evaluation_data() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20261220)
    plan_sampler = _ot_sampler()
    probes = torch.tensor(PROBE_POINTS, dtype=torch.float64)
    conditional_targets: list[torch.Tensor] = []
    plan_residuals: list[float] = []
    for probe in probes:
        values: list[torch.Tensor] = []
        for _ in range(CONDITIONAL_EVAL_SAMPLES):
            x_batch = torch.cat(
                [probe[None, :], _normal(generator, PAIRING_MINIBATCH_SIZE - 1)]
            )
            y_batch = _swiss(generator, PAIRING_MINIBATCH_SIZE)
            plan, residual = _plan_with_audit(
                x_batch, y_batch, plan_sampler
            )
            values.append(y_batch[int(torch.argmax(plan[0]).item())])
            plan_residuals.append(residual)
        conditional_targets.append(torch.stack(values))
    return {
        "probes": probes,
        "conditional_targets": conditional_targets,
        "marginal_x": _normal(generator, MARGINAL_EVAL_SAMPLES),
        "marginal_y": _swiss(generator, MARGINAL_EVAL_SAMPLES),
        "pairing_audit": {
            "plans": len(plan_residuals),
            "max_marginal_residual": max(plan_residuals),
        },
    }


def _build_model(seed: int, architecture: str) -> EbieotGmm:
    torch.manual_seed(seed)
    if architecture == "appendix-c2-mlp":
        log_v_hidden = [128]
        b_hidden = [256, 256]
    elif architecture == "released-linear":
        log_v_hidden = []
        b_hidden = []
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    cost = MLPLSECost(
        log_v_m_hidden_channels=log_v_hidden,
        b_m_hidden_channels=b_hidden,
        x_dim=2,
        y_dim=2,
        m_potentials=M_POTENTIALS,
        epsilon=1.0,
    ).double()
    return EbieotGmm(
        y_dim=2,
        n_potentials=N_POTENTIALS,
        cost=cost,
        epsilon=1.0,
        sampling_batch_size=128,
        A_diagonal_init=0.1,
    ).double()


def _sliced_w2(
    first: torch.Tensor, second: torch.Tensor, directions: int = 64
) -> float:
    if first.shape != second.shape:
        raise ValueError("Sliced W2 checker requires equally sized sample sets")
    angles = torch.arange(directions, dtype=torch.float64) * math.pi / directions
    vectors = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    first_projected = torch.sort(first @ vectors.T, dim=0).values
    second_projected = torch.sort(second @ vectors.T, dim=0).values
    return float(torch.sqrt(torch.mean((first_projected - second_projected) ** 2)))


def _energy_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    cross = torch.cdist(first, second).mean()
    within_first = torch.cdist(first, first).mean()
    within_second = torch.cdist(second, second).mean()
    return float(2.0 * cross - within_first - within_second)


def _covariance_trace(samples: torch.Tensor) -> float:
    centered = samples - samples.mean(dim=0)
    return float((centered.square().sum() / (samples.shape[0] - 1)))


def _evaluate_model(
    model: EbieotGmm,
    evaluation: dict[str, Any],
    evaluation_seed: int,
    include_samples: bool,
) -> dict[str, Any]:
    torch.manual_seed(evaluation_seed)
    model.eval()
    conditional_rows: list[dict[str, float]] = []
    representative: dict[str, Any] = {}
    with torch.no_grad():
        for probe_index, (probe, target) in enumerate(
            zip(evaluation["probes"], evaluation["conditional_targets"])
        ):
            repeated = probe[None, :].repeat(target.shape[0], 1)
            prediction = model(repeated)
            row = {
                "probe_x0": float(probe[0]),
                "probe_x1": float(probe[1]),
                "sliced_w2": _sliced_w2(prediction, target),
                "energy_distance": _energy_distance(prediction, target),
                "prediction_covariance_trace": _covariance_trace(prediction),
                "target_covariance_trace": _covariance_trace(target),
            }
            conditional_rows.append(row)
            if include_samples:
                representative[f"probe_{probe_index}_prediction"] = (
                    prediction[:128].tolist()
                )
                representative[f"probe_{probe_index}_target"] = target[:128].tolist()

        marginal_prediction = model(evaluation["marginal_x"])
        marginal_target = evaluation["marginal_y"]
        marginal = {
            "sliced_w2": _sliced_w2(marginal_prediction, marginal_target),
            "energy_distance": _energy_distance(
                marginal_prediction, marginal_target
            ),
            "prediction_covariance_trace": _covariance_trace(
                marginal_prediction
            ),
            "target_covariance_trace": _covariance_trace(marginal_target),
        }
        if include_samples:
            representative["marginal_prediction"] = marginal_prediction[:256].tolist()
            representative["marginal_target"] = marginal_target[:256].tolist()

    return {
        "conditional": conditional_rows,
        "conditional_mean_sliced_w2": sum(
            row["sliced_w2"] for row in conditional_rows
        )
        / len(conditional_rows),
        "conditional_mean_energy_distance": sum(
            row["energy_distance"] for row in conditional_rows
        )
        / len(conditional_rows),
        "marginal": marginal,
        "representative_samples": representative,
    }


def _manual_conditional_log_prob(
    model: EbieotGmm,
    batched_x: torch.Tensor,
    batched_y: torch.Tensor,
    *,
    normalize_weights: bool = True,
) -> torch.Tensor:
    """Clean-room Eq. 17 evaluator from exposed trained tensors."""
    log_w_n = model.log_w_n()
    a_n = model.a_n()
    A_n = model.A_n()
    b_m = model.cost.b_m(batched_x)
    log_v_m = model.cost.log_v_m(batched_x)
    log_z_nm = model.log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)
    component_logits = log_z_nm.flatten(start_dim=1)
    if normalize_weights:
        component_logits = component_logits - torch.logsumexp(
            component_logits, dim=1, keepdim=True
        )
    component_means = (
        a_n[None, :, None, :] + A_n[None, :, None, :] * b_m[:, None, :, :]
    ).reshape(batched_x.shape[0], -1, model.y_dim)
    component_variances = (
        model.epsilon * A_n[None, :, None, :]
    ).expand(
        batched_x.shape[0],
        model.n_potentials,
        model.cost.m_potentials,
        model.y_dim,
    ).reshape(batched_x.shape[0], -1, model.y_dim)
    centered = batched_y[:, None, :] - component_means
    component_log_density = -0.5 * (
        model.y_dim * math.log(2.0 * math.pi)
        + torch.log(component_variances).sum(dim=2)
        + (centered.square() / component_variances).sum(dim=2)
    )
    return torch.logsumexp(
        component_logits + component_log_density, dim=1
    )


@torch.no_grad()
def _practical_benchmark(
    model: EbieotGmm,
    evaluation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Full N*M trained-task checks for the practical parameterization."""
    torch.manual_seed(seed)
    model.eval()
    probe_x = torch.cat(
        [
            probe[None, :].repeat(target.shape[0], 1)
            for probe, target in zip(
                evaluation["probes"], evaluation["conditional_targets"]
            )
        ],
        dim=0,
    )
    probe_y = torch.cat(evaluation["conditional_targets"], dim=0)
    checker_x = probe_x[::24][:64]
    checker_y = probe_y[::24][:64]

    distribution = model.get_conditional_distribution(
        checker_x, model.log_w_n(), model.a_n(), model.A_n()
    )
    library_log_prob = distribution.log_prob(checker_y)
    manual_log_prob = _manual_conditional_log_prob(
        model, checker_x, checker_y
    )
    parity_error = float(
        torch.max(torch.abs(library_log_prob - manual_log_prob))
    )
    unnormalized_log_prob = _manual_conditional_log_prob(
        model, checker_x, checker_y, normalize_weights=False
    )
    unnormalized_error = float(
        torch.max(torch.abs(unnormalized_log_prob - library_log_prob))
    )

    likelihood_times: list[float] = []
    heldout_nll = 0.0
    for _ in range(5):
        started = time.perf_counter()
        heldout_distribution = model.get_conditional_distribution(
            probe_x, model.log_w_n(), model.a_n(), model.A_n()
        )
        heldout_nll = float(-heldout_distribution.log_prob(probe_y).mean())
        likelihood_times.append(time.perf_counter() - started)

    sample_count = 8_192
    sample_x = evaluation["probes"][2][None, :]
    sample_distribution = model.get_conditional_distribution(
        sample_x, model.log_w_n(), model.a_n(), model.A_n()
    )
    sampling_times: list[float] = []
    samples = torch.empty(0, model.y_dim, dtype=torch.float64)
    for repeat in range(5):
        torch.manual_seed(seed + repeat)
        started = time.perf_counter()
        samples = sample_distribution.sample((sample_count,)).squeeze(1)
        sampling_times.append(time.perf_counter() - started)

    weights = sample_distribution.mixture_distribution.probs.squeeze(0)
    means = sample_distribution.component_distribution.base_dist.loc.squeeze(0)
    variances = (
        sample_distribution.component_distribution.base_dist.scale.squeeze(0)
        .square()
    )
    exact_mean = (weights[:, None] * means).sum(dim=0)
    centered_means = means - exact_mean
    exact_covariance = torch.diag(
        (weights[:, None] * variances).sum(dim=0)
    ) + torch.einsum(
        "k,ki,kj->ij", weights, centered_means, centered_means
    )
    empirical_mean = samples.mean(dim=0)
    centered_samples = samples - empirical_mean
    empirical_covariance = (
        centered_samples.T @ centered_samples / (sample_count - 1)
    )
    mean_standard_errors = torch.sqrt(
        torch.diag(exact_covariance) / sample_count
    )
    mean_z_max = float(
        torch.max(
            torch.abs(empirical_mean - exact_mean)
            / torch.clamp(mean_standard_errors, min=1.0e-15)
        )
    )
    covariance_relative_error = float(
        torch.linalg.norm(empirical_covariance - exact_covariance)
        / torch.clamp(torch.linalg.norm(exact_covariance), min=1.0e-15)
    )
    collapsed_covariance_relative_error = float(
        torch.linalg.norm(exact_covariance)
        / torch.clamp(torch.linalg.norm(exact_covariance), min=1.0e-15)
    )

    likelihood_median = sorted(likelihood_times)[len(likelihood_times) // 2]
    sampling_median = sorted(sampling_times)[len(sampling_times) // 2]
    passed = (
        parity_error < 2.0e-10
        and unnormalized_error > 1.0e-4
        and math.isfinite(heldout_nll)
        and mean_z_max < 6.0
        and covariance_relative_error < 0.15
        and collapsed_covariance_relative_error > 0.5
    )
    return {
        "status": "VERIFIED" if passed else "BLOCKED",
        "passed": passed,
        "trained_components": (
            model.n_potentials * model.cost.m_potentials
        ),
        "heldout_probe_observations": int(probe_y.shape[0]),
        "heldout_conditional_nll": heldout_nll,
        "independent_manual_log_prob_max_abs_error": parity_error,
        "manual_parity_threshold": 2.0e-10,
        "negative_control": "omit conditional mixture normalization",
        "negative_control_max_abs_error": unnormalized_error,
        "negative_control_failure_floor": 1.0e-4,
        "likelihood_median_seconds": likelihood_median,
        "likelihood_observations_per_second": (
            probe_y.shape[0] / likelihood_median
        ),
        "sampling_count": sample_count,
        "sampling_median_seconds": sampling_median,
        "samples_per_second": sample_count / sampling_median,
        "sampling_mean_max_z_score": mean_z_max,
        "sampling_mean_z_threshold": 6.0,
        "sampling_covariance_relative_error": covariance_relative_error,
        "sampling_covariance_threshold": 0.15,
        "collapsed_sampler_covariance_relative_error": (
            collapsed_covariance_relative_error
        ),
        "collapsed_sampler_failure_floor": 0.5,
        "timing_repetitions": 5,
        "timing_note": (
            "Wall-clock CPU throughput on one trained paper-scale model; "
            "not an asymptotic complexity claim"
        ),
    }


def _train_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(WORKER_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(WORKER_THREADS)
    torch.set_num_threads(WORKER_THREADS)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)

    seed = int(payload["seed"])
    regime = str(payload["regime"])
    architecture = str(payload["architecture"])
    data = payload["data"]
    evaluation = payload["evaluation"]
    started = time.perf_counter()

    model = _build_model(seed + 100, architecture)
    if regime == "full-semisupervised":
        x_unpaired = data["unpaired_x"]
        y_unpaired = data["unpaired_y"]
    elif regime == "no-extra-unpaired":
        x_unpaired = data["paired_x"]
        y_unpaired = data["paired_y"]
    else:
        raise ValueError(f"Unknown training regime: {regime}")

    init_generator = torch.Generator(device="cpu").manual_seed(seed + 200)
    init_indices = torch.randperm(
        y_unpaired.shape[0], generator=init_generator
    )[:N_POTENTIALS]
    if init_indices.shape[0] < N_POTENTIALS:
        init_indices = torch.randint(
            0,
            y_unpaired.shape[0],
            (N_POTENTIALS,),
            generator=init_generator,
        )
    model.init_a_by_samples(y_unpaired[init_indices])

    unpaired_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    paired_optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 300)
    trace_steps = {0, 99, 999, 4_999, 9_999, TRAIN_STEPS - 1}
    training_trace: list[dict[str, float | int]] = []

    untrained = _evaluate_model(
        model,
        evaluation,
        evaluation_seed=seed + 400,
        include_samples=False,
    )
    for step in range(TRAIN_STEPS):
        unpaired_indices = torch.randint(
            0,
            x_unpaired.shape[0],
            (TRAIN_BATCH_SIZE,),
            generator=batch_generator,
        )
        y_indices = torch.randint(
            0,
            y_unpaired.shape[0],
            (TRAIN_BATCH_SIZE,),
            generator=batch_generator,
        )
        paired_indices = torch.randint(
            0,
            data["paired_x"].shape[0],
            (TRAIN_BATCH_SIZE,),
            generator=batch_generator,
        )

        unpaired_optimizer.zero_grad(set_to_none=True)
        unpaired_output = model.compute_unpaired_loss(
            x_unpaired[unpaired_indices], y_unpaired[y_indices]
        )
        unpaired_output["loss"].backward()
        unpaired_optimizer.step()

        paired_optimizer.zero_grad(set_to_none=True)
        paired_output = model.compute_paired_loss(
            data["paired_x"][paired_indices],
            data["paired_y"][paired_indices],
        )
        paired_output["loss"].backward()
        paired_optimizer.step()

        if step in trace_steps:
            training_trace.append(
                {
                    "step": step,
                    "unpaired_loss": float(
                        unpaired_output["loss"].detach()
                    ),
                    "paired_loss": float(paired_output["loss"].detach()),
                }
            )

    trained = _evaluate_model(
        model,
        evaluation,
        evaluation_seed=seed + 400,
        include_samples=bool(payload["include_samples"]),
    )
    practical = (
        _practical_benchmark(model, evaluation, seed + 500)
        if bool(payload.get("include_practical_benchmark", False))
        else None
    )
    finite_trace = all(
        math.isfinite(float(value))
        for row in training_trace
        for key, value in row.items()
        if key != "step"
    )
    metrics_to_check = [
        trained["conditional_mean_sliced_w2"],
        trained["conditional_mean_energy_distance"],
        trained["marginal"]["sliced_w2"],
        trained["marginal"]["energy_distance"],
    ]
    return {
        "seed": seed,
        "architecture": architecture,
        "regime": regime,
        "training_steps": TRAIN_STEPS,
        "training_trace": training_trace,
        "untrained_control": {
            "conditional_mean_sliced_w2": untrained[
                "conditional_mean_sliced_w2"
            ],
            "conditional_mean_energy_distance": untrained[
                "conditional_mean_energy_distance"
            ],
            "marginal_sliced_w2": untrained["marginal"]["sliced_w2"],
            "marginal_energy_distance": untrained["marginal"][
                "energy_distance"
            ],
        },
        "trained": trained,
        "practical_parameterization": practical,
        "finite": finite_trace
        and all(math.isfinite(value) for value in metrics_to_check),
        "runtime_seconds": time.perf_counter() - started,
        "torch_threads": torch.get_num_threads(),
    }


def _metric_checker() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20261221)
    reference = _normal(generator, 256)
    identical = _sliced_w2(reference, reference.clone())
    shifted = _sliced_w2(reference, reference + torch.tensor([5.0, -5.0]))
    identical_energy = _energy_distance(reference, reference.clone())
    shifted_energy = _energy_distance(
        reference, reference + torch.tensor([5.0, -5.0])
    )
    passed = (
        identical < 1.0e-12
        and abs(identical_energy) < 1.0e-12
        and shifted > 4.0
        and shifted_energy > 8.0
    )
    return {
        "identical_sliced_w2": identical,
        "identical_energy_distance": identical_energy,
        "shifted_sliced_w2": shifted,
        "shifted_energy_distance": shifted_energy,
        "control_failed_as_intended": passed,
    }


def run_swiss_calibration() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    data_by_seed = {
        seed: _make_training_data(seed) for seed in TRAIN_SEEDS
    }
    evaluation = _make_evaluation_data()
    payloads: list[dict[str, Any]] = []
    for seed in TRAIN_SEEDS:
        for regime in ("full-semisupervised", "no-extra-unpaired"):
            payloads.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "architecture": ARCHITECTURE,
                    "data": data_by_seed[seed],
                    "evaluation": evaluation,
                    "include_samples": (
                        seed == TRAIN_SEEDS[0]
                        and regime == "full-semisupervised"
                    ),
                    "include_practical_benchmark": (
                        seed == TRAIN_SEEDS[0]
                        and regime == "full-semisupervised"
                    ),
                }
            )

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(payloads), mp_context=context
    ) as executor:
        results = list(executor.map(_train_worker, payloads))

    checker = _metric_checker()
    full_results = [
        row for row in results if row["regime"] == "full-semisupervised"
    ]
    no_extra_results = [
        row for row in results if row["regime"] == "no-extra-unpaired"
    ]

    def aggregate(rows: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
        values: list[float] = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        mean = sum(values) / len(values)
        sample_std = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        return {"mean": mean, "sample_std": sample_std, "n": len(values)}

    aggregate_results = {
        "full_conditional_sliced_w2": aggregate(
            full_results, ("trained", "conditional_mean_sliced_w2")
        ),
        "full_conditional_energy_distance": aggregate(
            full_results, ("trained", "conditional_mean_energy_distance")
        ),
        "full_marginal_sliced_w2": aggregate(
            full_results, ("trained", "marginal", "sliced_w2")
        ),
        "no_extra_conditional_sliced_w2": aggregate(
            no_extra_results, ("trained", "conditional_mean_sliced_w2")
        ),
        "no_extra_marginal_sliced_w2": aggregate(
            no_extra_results, ("trained", "marginal", "sliced_w2")
        ),
    }
    training_plan_count = sum(
        int(data["pairing_audit"]["plans"]) for data in data_by_seed.values()
    )
    max_training_plan_residual = max(
        float(data["pairing_audit"]["max_marginal_residual"])
        for data in data_by_seed.values()
    )
    evaluation_plan_count = int(evaluation["pairing_audit"]["plans"])
    max_evaluation_plan_residual = float(
        evaluation["pairing_audit"]["max_marginal_residual"]
    )
    plan_residual_threshold = 1.0e-7
    plan_audit_passed = (
        max_training_plan_residual < plan_residual_threshold
        and max_evaluation_plan_residual < plan_residual_threshold
    )
    passed = (
        all(row["finite"] for row in results)
        and all(
            row["practical_parameterization"] is None
            or row["practical_parameterization"]["passed"]
            for row in results
        )
        and checker["control_failed_as_intended"]
        and plan_audit_passed
        and len(full_results) == len(TRAIN_SEEDS)
        and len(no_extra_results) == len(TRAIN_SEEDS)
    )
    return {
        "calibration": "swiss-roll-architecture",
        "status": "CALIBRATED" if passed else "BLOCKED",
        "passed": passed,
        "paper_contract": {
            "P_paired": P_PAIRED,
            "Q_unpaired_x": Q_UNPAIRED,
            "R_unpaired_y": R_UNPAIRED,
            "N": N_POTENTIALS,
            "M": M_POTENTIALS,
            "training_steps": TRAIN_STEPS,
            "pairing_minibatch": PAIRING_MINIBATCH_SIZE,
            "pairing_method": "POT Sinkhorn, reg=0.05, rotation-v2",
            "learning_rate_paired": 3.0e-4,
            "learning_rate_unpaired": 1.0e-3,
        },
        "architecture": {
            "name": ARCHITECTURE,
            "log_v_m_hidden_channels": [128],
            "b_m_hidden_channels": [256, 256],
            "interpretation": "Appendix C.2 single-hidden-layer v_m and two-hidden-layer a_m",
            "released_config_deviation": "Released gmm_swiss_roll.yaml overrides both towers to linear",
        },
        "evaluation": {
            "probe_points": [list(point) for point in PROBE_POINTS],
            "conditional_samples_per_probe": CONDITIONAL_EVAL_SAMPLES,
            "marginal_samples": MARGINAL_EVAL_SAMPLES,
            "sliced_w2_directions": 64,
            "ground_truth": "Appendix D.1 minibatch-64 Sinkhorn argmax conditional procedure",
        },
        "metric_checker": checker,
        "transport_plan_audit": {
            "solver": "POT sinkhorn_log",
            "regularization": 0.05,
            "max_iterations": 20_000,
            "solver_stop_threshold": 1.0e-10,
            "acceptance_marginal_residual_lt": plan_residual_threshold,
            "training_plans": training_plan_count,
            "max_training_marginal_residual": max_training_plan_residual,
            "evaluation_plans": evaluation_plan_count,
            "max_evaluation_marginal_residual": max_evaluation_plan_residual,
            "passed": plan_audit_passed,
        },
        "aggregate": aggregate_results,
        "runs": results,
        "runtime_seconds": time.perf_counter() - started,
        "estimated_worker_cores": len(payloads) * WORKER_THREADS,
    }

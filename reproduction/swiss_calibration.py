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
    )


def _sample_one_plan_pair(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    plan_sampler: OTPlanSampler,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = plan_sampler.get_map(x_batch, y_batch)
    probabilities = torch.clamp(plan.flatten(), min=0.0)
    probabilities = probabilities / probabilities.sum()
    flat_index = int(
        torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).item()
    )
    x_index = flat_index // y_batch.shape[0]
    y_index = flat_index % y_batch.shape[0]
    return x_batch[x_index], y_batch[y_index]


def _make_training_data(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    plan_sampler = _ot_sampler()
    paired_x: list[torch.Tensor] = []
    paired_y: list[torch.Tensor] = []
    for _ in range(P_PAIRED):
        x_batch = _normal(generator, PAIRING_MINIBATCH_SIZE)
        y_batch = _swiss(generator, PAIRING_MINIBATCH_SIZE)
        x_value, y_value = _sample_one_plan_pair(
            x_batch, y_batch, plan_sampler, generator
        )
        paired_x.append(x_value)
        paired_y.append(y_value)
    return {
        "paired_x": torch.stack(paired_x),
        "paired_y": torch.stack(paired_y),
        "unpaired_x": _normal(generator, Q_UNPAIRED),
        "unpaired_y": _swiss(generator, R_UNPAIRED),
    }


def _make_evaluation_data() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20261220)
    plan_sampler = _ot_sampler()
    probes = torch.tensor(PROBE_POINTS, dtype=torch.float64)
    conditional_targets: list[torch.Tensor] = []
    for probe in probes:
        values: list[torch.Tensor] = []
        for _ in range(CONDITIONAL_EVAL_SAMPLES):
            x_batch = torch.cat(
                [probe[None, :], _normal(generator, PAIRING_MINIBATCH_SIZE - 1)]
            )
            y_batch = _swiss(generator, PAIRING_MINIBATCH_SIZE)
            plan = plan_sampler.get_map(x_batch, y_batch)
            values.append(y_batch[int(torch.argmax(plan[0]).item())])
        conditional_targets.append(torch.stack(values))
    return {
        "probes": probes,
        "conditional_targets": conditional_targets,
        "marginal_x": _normal(generator, MARGINAL_EVAL_SAMPLES),
        "marginal_y": _swiss(generator, MARGINAL_EVAL_SAMPLES),
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
    passed = (
        all(row["finite"] for row in results)
        and checker["control_failed_as_intended"]
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
        "aggregate": aggregate_results,
        "runs": results,
        "runtime_seconds": time.perf_counter() - started,
        "estimated_worker_cores": len(payloads) * WORKER_THREADS,
    }

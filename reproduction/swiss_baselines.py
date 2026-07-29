"""Faithful Figure-2 baseline comparison on the audited Swiss-Roll data."""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import os
import random
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from reproduction.swiss_calibration import (
    CONDITIONAL_EVAL_SAMPLES,
    MARGINAL_EVAL_SAMPLES,
    P_PAIRED,
    Q_UNPAIRED,
    R_UNPAIRED,
    TRAIN_SEEDS,
    _covariance_trace,
    _energy_distance,
    _make_evaluation_data,
    _make_training_data,
    _metric_checker,
    _sliced_w2,
)
from src.baselines.cgan import CganTrainer
from src.baselines.regression import RegressionTrainer
from src.baselines.ugan import UganTrainer
from src.utils.samplers.data import DatasetSampler, PairedTensorBatchSampler


BASELINE_METHODS = ("regression", "ugan", "cgan")
BASELINE_STEPS = 250_000
BASELINE_BATCH_SIZE = 128
WORKER_THREADS = 6
T_95_DF2 = 4.302652729696142


def _config(method: str) -> Any:
    common = {
        "dataset": {"x_dim": 2, "y_dim": 2},
        "train": {
            "batch_size": BASELINE_BATCH_SIZE,
            "steps_to": BASELINE_STEPS,
        },
        "model": {},
    }
    if method == "regression":
        common["train"].update({"lr": 3.0e-4, "weight_decay": 0.01})
        common["model"] = {
            "input_size": 2,
            "hidden_size": 256,
            "num_hidden_layers": 4,
        }
    elif method == "ugan":
        common["train"].update(
            {
                "z_dim": 1,
                "layers_g": [256, 256, 256],
                "layers_d": [256, 256, 256],
                "lr_g": 1.0e-4,
                "lr_d": 1.0e-4,
                "beta1": 0.5,
                "beta2": 0.9,
                "r1_gamma": 0.01,
                "lazy_reg": 1,
            }
        )
    elif method == "cgan":
        common["train"].update(
            {
                "z_dim": 1,
                "num_timesteps": 1,
                "layers_g": [256, 256, 256],
                "layers_d": [256, 256, 256],
                "lr_g": 1.0e-3,
                "lr_d": 3.0e-4,
                "beta1": 0.5,
                "beta2": 0.9,
                "r1_gamma": 0.01,
                "lazy_reg": 1,
            }
        )
    else:
        raise ValueError(f"Unknown baseline: {method}")
    return OmegaConf.create(common)


def _build_trainer(method: str, cfg: Any) -> Any:
    device = torch.device("cpu")
    if method == "regression":
        trainer = RegressionTrainer(cfg, device)
    elif method == "ugan":
        trainer = UganTrainer(cfg, device)
    elif method == "cgan":
        trainer = CganTrainer(cfg, device)
    else:
        raise ValueError(f"Unknown baseline: {method}")
    trainer.build_optimizers(cfg)
    return trainer


@torch.no_grad()
def _predict(
    trainer: Any,
    method: str,
    x: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if method == "regression":
        trainer.model.eval()
        return trainer.model(x)
    latent = torch.randn(
        x.shape[0], 1, generator=generator, dtype=x.dtype
    )
    if method == "ugan":
        trainer.net_g.eval()
        return trainer.net_g(x, latent)
    trainer.net_g.eval()
    timestep = torch.zeros(x.shape[0], dtype=torch.long)
    return trainer.net_g(x, timestep, latent)


@torch.no_grad()
def _evaluate(
    trainer: Any,
    method: str,
    evaluation: dict[str, Any],
    seed: int,
    include_samples: bool,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    conditional_rows: list[dict[str, float]] = []
    representative: dict[str, Any] = {}
    for index, (probe, target64) in enumerate(
        zip(evaluation["probes"], evaluation["conditional_targets"])
    ):
        target = target64.float()
        repeated = probe.float()[None, :].repeat(target.shape[0], 1)
        prediction = _predict(trainer, method, repeated, generator)
        row = {
            "probe_x0": float(probe[0]),
            "probe_x1": float(probe[1]),
            "sliced_w2": _sliced_w2(
                prediction.double(), target.double()
            ),
            "energy_distance": _energy_distance(
                prediction.double(), target.double()
            ),
            "prediction_covariance_trace": _covariance_trace(
                prediction.double()
            ),
            "target_covariance_trace": _covariance_trace(target.double()),
        }
        conditional_rows.append(row)
        if include_samples:
            representative[f"probe_{index}_prediction"] = (
                prediction[:128].double().tolist()
            )
            representative[f"probe_{index}_target"] = (
                target[:128].double().tolist()
            )

    marginal_x = evaluation["marginal_x"].float()
    marginal_target = evaluation["marginal_y"].float()
    marginal_prediction = _predict(
        trainer, method, marginal_x, generator
    )
    marginal = {
        "sliced_w2": _sliced_w2(
            marginal_prediction.double(), marginal_target.double()
        ),
        "energy_distance": _energy_distance(
            marginal_prediction.double(), marginal_target.double()
        ),
        "prediction_covariance_trace": _covariance_trace(
            marginal_prediction.double()
        ),
        "target_covariance_trace": _covariance_trace(
            marginal_target.double()
        ),
    }
    if include_samples:
        representative["marginal_prediction"] = (
            marginal_prediction[:256].double().tolist()
        )
        representative["marginal_target"] = (
            marginal_target[:256].double().tolist()
        )
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


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(WORKER_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(WORKER_THREADS)
    torch.set_num_threads(WORKER_THREADS)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    seed = int(payload["seed"])
    method = str(payload["method"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    started = time.perf_counter()

    data = payload["data"]
    evaluation = payload["evaluation"]
    cfg = _config(method)
    trainer = _build_trainer(method, cfg)
    source = DatasetSampler(data["unpaired_x"].float(), device="cpu")
    target = DatasetSampler(data["unpaired_y"].float(), device="cpu")
    paired = PairedTensorBatchSampler(
        data["paired_x"].float(), data["paired_y"].float()
    )
    trace_steps = {
        0,
        999,
        9_999,
        49_999,
        99_999,
        BASELINE_STEPS - 1,
    }
    trace: list[dict[str, Any]] = []
    for step in range(BASELINE_STEPS):
        metrics = trainer.train_step(step, source, target, paired)
        if step in trace_steps:
            trace.append({"step": step, **metrics})

    evaluated = _evaluate(
        trainer,
        method,
        evaluation,
        seed + 700,
        include_samples=bool(payload["include_samples"]),
    )
    values = [
        value
        for row in trace
        for key, value in row.items()
        if key != "step"
    ] + [
        evaluated["conditional_mean_sliced_w2"],
        evaluated["conditional_mean_energy_distance"],
        evaluated["marginal"]["sliced_w2"],
        evaluated["marginal"]["energy_distance"],
    ]
    return {
        "method": method,
        "seed": seed,
        "training_steps": BASELINE_STEPS,
        "training_trace": trace,
        "evaluated": evaluated,
        "finite": all(math.isfinite(float(value)) for value in values),
        "runtime_seconds": time.perf_counter() - started,
        "torch_threads": torch.get_num_threads(),
    }


def _aggregate(
    rows: list[dict[str, Any]], path: tuple[str, ...]
) -> dict[str, float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    mean = sum(values) / len(values)
    sample_std = math.sqrt(
        sum((value - mean) ** 2 for value in values)
        / (len(values) - 1)
    )
    return {"mean": mean, "sample_std": sample_std, "n": len(values)}


def _paired_comparison(
    ours_by_seed: dict[int, float],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    differences = [
        row["evaluated"]["conditional_mean_sliced_w2"]
        - ours_by_seed[int(row["seed"])]
        for row in sorted(baseline_rows, key=lambda item: item["seed"])
    ]
    mean = sum(differences) / len(differences)
    sample_std = math.sqrt(
        sum((value - mean) ** 2 for value in differences)
        / (len(differences) - 1)
    )
    half_width = T_95_DF2 * sample_std / math.sqrt(len(differences))
    return {
        "baseline_minus_ours_mean": mean,
        "sample_std": sample_std,
        "n": len(differences),
        "paired_t_95_ci_low": mean - half_width,
        "paired_t_95_ci_high": mean + half_width,
    }


def run_swiss_baselines(ours_result: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    data_by_seed = {
        seed: _make_training_data(seed) for seed in TRAIN_SEEDS
    }
    evaluation = _make_evaluation_data()
    payloads = [
        {
            "method": method,
            "seed": seed,
            "data": data_by_seed[seed],
            "evaluation": evaluation,
            "include_samples": seed == TRAIN_SEEDS[0],
        }
        for method in BASELINE_METHODS
        for seed in TRAIN_SEEDS
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(payloads), mp_context=context
    ) as executor:
        rows = list(executor.map(_worker, payloads))

    by_method = {
        method: [row for row in rows if row["method"] == method]
        for method in BASELINE_METHODS
    }
    aggregates = {
        method: {
            "conditional_sliced_w2": _aggregate(
                method_rows,
                ("evaluated", "conditional_mean_sliced_w2"),
            ),
            "conditional_energy_distance": _aggregate(
                method_rows,
                ("evaluated", "conditional_mean_energy_distance"),
            ),
            "marginal_sliced_w2": _aggregate(
                method_rows, ("evaluated", "marginal", "sliced_w2")
            ),
        }
        for method, method_rows in by_method.items()
    }
    ours_rows = [
        row
        for row in ours_result["runs"]
        if row["regime"] == "full-semisupervised"
    ]
    ours_by_seed = {
        int(row["seed"]): float(
            row["trained"]["conditional_mean_sliced_w2"]
        )
        for row in ours_rows
    }
    comparisons = {
        method: _paired_comparison(ours_by_seed, method_rows)
        for method, method_rows in by_method.items()
    }
    regression_max_covariance = max(
        conditional["prediction_covariance_trace"]
        for row in by_method["regression"]
        for conditional in row["evaluated"]["conditional"]
    )
    metric_checker = _metric_checker()
    operational_pass = (
        all(row["finite"] for row in rows)
        and metric_checker["control_failed_as_intended"]
        and regression_max_covariance < 1.0e-10
        and len(rows) == len(BASELINE_METHODS) * len(TRAIN_SEEDS)
    )
    return {
        "claim": 6,
        "route": "released-standard-baselines",
        "status": "ROUTE_1_COMPLETE" if operational_pass else "BLOCKED",
        "passed": operational_pass,
        "paper_contract": {
            "figure": "Figure 2",
            "P_paired": P_PAIRED,
            "Q_unpaired_x": Q_UNPAIRED,
            "R_unpaired_y": R_UNPAIRED,
            "training_steps": BASELINE_STEPS,
            "methods": list(BASELINE_METHODS),
            "conditional_evaluation_samples_per_probe": (
                CONDITIONAL_EVAL_SAMPLES
            ),
            "marginal_evaluation_samples": MARGINAL_EVAL_SAMPLES,
        },
        "source_faithfulness": {
            "models_and_objectives": "authors' released trainer classes",
            "data_repair": (
                "all methods use Appendix D.1 fresh minibatch-64 "
                "Sinkhorn pairing rather than the released one-plan shortcut"
            ),
            "omitted_this_route": [
                "CondNF",
                "CondNF (SS)",
                "external DCPEME/parOT/OTCS/FSBM/GNOT",
                "CGMM (SS), absent from the release",
            ],
        },
        "aggregates": aggregates,
        "paired_conditional_comparisons": comparisons,
        "regression_structural_control": {
            "max_prediction_covariance_trace": regression_max_covariance,
            "acceptance_threshold": 1.0e-10,
            "failed_to_represent_distribution_as_intended": (
                regression_max_covariance < 1.0e-10
            ),
        },
        "metric_checker": metric_checker,
        "runs": rows,
        "estimated_worker_cores": (
            len(BASELINE_METHODS) * len(TRAIN_SEEDS) * WORKER_THREADS
        ),
        "runtime_seconds": time.perf_counter() - started,
        "limitations": (
            "Route 1 directly tests three released Figure-2 baselines; "
            "conditional-flow and external-method routes remain separate."
        ),
    }

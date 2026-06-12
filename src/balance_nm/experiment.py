"""Closed-loop synthetic and retrospective replay experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from .acquisition import ActionValidator, CandidateGenerator, create_policy
from .domain import (
    ExperimentState,
    MeasurementAction,
    MetricRecord,
    ObjectiveMetricRecord,
    Recommendation,
    ReplayCapabilities,
    RunConfig,
)
from .evaluation import Evaluator, summarize_benchmark
from .models import CountAwareIndependentGP
from .patterns import PatternAnalyzer
from .quality import QualityAnalyzer
from .replay import ReplayMicroscope, reference_from_dense_observations
from .simulation import VirtualMicroscope, generate_hidden_sample


@dataclass
class RunResult:
    config: RunConfig
    policy: str
    seed: int
    hidden_sample: xr.Dataset
    state: ExperimentState
    final_prediction: xr.Dataset
    metrics: list[MetricRecord]
    last_recommendation: Recommendation
    objective_metrics: list[ObjectiveMetricRecord] = field(default_factory=list)
    standardized_dataset: xr.Dataset | None = None
    capabilities: ReplayCapabilities | None = None


def _action_rng(seed: int, action_id: str) -> np.random.Generator:
    digest = sha256(f"{seed}:{action_id}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(value)


def _execute_loop(
    config: RunConfig,
    policy_name: str,
    seed: int,
    reference_sample: xr.Dataset,
    microscope: VirtualMicroscope,
    capabilities: ReplayCapabilities | None = None,
    standardized_dataset: xr.Dataset | None = None,
) -> RunResult:
    state = ExperimentState()
    initial = microscope.initial_action()
    initial_violations = ActionValidator(config, capabilities).violations(initial, state)
    if initial_violations:
        reasons = "; ".join(violation.reason for violation in initial_violations)
        raise ValueError(f"initial coarse scan is infeasible: {reasons}")
    state.add_batch(
        microscope.acquire(reference_sample, initial, _action_rng(seed, initial.action_id))
    )
    surrogate = CountAwareIndependentGP(config)
    policy = create_policy(policy_name, config, surrogate, np.random.default_rng(seed + 17_513))
    candidates = CandidateGenerator(config, microscope, capabilities)
    evaluator = Evaluator(config, policy_name, seed)
    quality_analyzer = QualityAnalyzer(config)
    pattern_analyzer = PatternAnalyzer(config)
    metrics: list[MetricRecord] = []
    objective_metrics: list[ObjectiveMetricRecord] = []
    last_recommendation = Recommendation(action=initial, utility=0.0)

    for iteration in range(config.acquisition.maximum_follow_on_actions + 1):
        if state.observations is None:
            raise RuntimeError("an experiment cannot continue without observations")
        surrogate.fit(state.observations)
        posterior = surrogate.predict()
        quality = quality_analyzer.compute(posterior, state.observations, standardized_dataset)
        patterns = pattern_analyzer.analyze(posterior, quality)
        analysis = xr.merge([posterior, quality, patterns])
        state.prediction = analysis
        state.quality_products = quality
        state.pattern_products = patterns
        metrics.append(evaluator.score(reference_sample, state, analysis, iteration))
        objective_metrics.extend(evaluator.score_objectives(reference_sample, state, analysis, iteration))
        if iteration == config.acquisition.maximum_follow_on_actions:
            last_recommendation = Recommendation(
                action=MeasurementAction(action_id="stop", action_type="stop"),
                utility=0.0,
                reasons=["Maximum follow-on action count reached."],
            )
            state.decision_trace.append(last_recommendation.as_record(iteration + 1))
            break
        feasible, rejected = candidates.generate(state, analysis, quality)
        unsupported = candidates.last_unsupported
        last_recommendation = policy.recommend(
            state, analysis, feasible, rejected, unsupported
        )
        state.decision_trace.append(last_recommendation.as_record(iteration + 1))
        state.rejected_actions.extend(
            [{"iteration": iteration + 1, **record} for record in candidates.last_rejected_records]
        )
        state.unsupported_proposals.extend(
            [{"iteration": iteration + 1, **record} for record in candidates.last_unsupported_records]
        )
        if last_recommendation.action.action_type == "stop":
            break
        state.add_batch(
            microscope.acquire(
                reference_sample,
                last_recommendation.action,
                _action_rng(seed, last_recommendation.action.action_id),
            )
        )
    if state.prediction is None:
        raise RuntimeError("experiment failed to produce a posterior prediction")
    return RunResult(
        config=config,
        policy=policy_name,
        seed=seed,
        hidden_sample=reference_sample,
        state=state,
        final_prediction=state.prediction,
        metrics=metrics,
        last_recommendation=last_recommendation,
        objective_metrics=objective_metrics,
        standardized_dataset=standardized_dataset,
        capabilities=capabilities,
    )


def run_experiment(config: RunConfig, policy_name: str, seed: int) -> RunResult:
    hidden_sample = generate_hidden_sample(config, np.random.default_rng(seed))
    return _execute_loop(
        config, policy_name, seed, hidden_sample, VirtualMicroscope(config)
    )


def run_replay_experiment(
    config: RunConfig,
    policy_name: str,
    source: xr.Dataset,
    capabilities: ReplayCapabilities,
    seed: int = 0,
) -> RunResult:
    if config.dataset.mode != "replay":
        raise ValueError("replay experiments require dataset.mode = replay")
    x_size = np.unique(source["x_nm"].values).size
    y_size = np.unique(source["y_nm"].values).size
    if x_size != config.scenario.grid_columns or y_size != config.scenario.grid_rows:
        raise ValueError(
            "replay source grid does not match scenario grid_shape/grid_size; "
            "configure the analysis grid to the imported dataset"
        )
    reference = reference_from_dense_observations(config, source)
    microscope = ReplayMicroscope(config, source, capabilities)
    return _execute_loop(
        config,
        policy_name,
        seed,
        reference,
        microscope,
        capabilities=capabilities,
        standardized_dataset=source,
    )


def run_benchmark(
    config: RunConfig, output: Path, save_run: Callable[[RunResult, Path], None]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    for seed in range(config.benchmark.seeds):
        for policy in config.benchmark.policies:
            result = run_experiment(config, policy, seed)
            run_output = output / "runs" / f"{policy}_seed_{seed:04d}"
            save_run(result, run_output)
            records.extend(record.model_dump() for record in result.metrics)
    metrics = pd.DataFrame(records)
    summary = summarize_benchmark(metrics)
    return metrics, summary

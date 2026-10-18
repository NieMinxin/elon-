from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from polymarket_bot.database import PostObservation
from polymarket_bot.market_intervals import MarketInterval
from polymarket_bot.model_experiments import (
    _best_bands,
    _write_json,
    _write_jsonl,
    paired_block_bootstrap_comparison,
)
from polymarket_bot.model_experiments_v3 import (
    _fit_model,
    _forecast_model,
    _metric_values,
    _model_artifact,
    _summary,
    generate_predictions,
)

BASELINE_MODELS = (
    "empirical_weekly",
    "analog_disjoint",
    "analog_disjoint_all",
    "analog_weight_recent",
    "analog_half_life_90",
    "gbm_conformal_0h",
)

SELECTION_OBJECTIVES = (
    ("log_loss", 3.0, False),
    ("brier_score", 2.0, False),
    ("selected_band_ece", 2.0, False),
    ("coverage_error", 1.0, False),
    ("selected_band_hit_rate", 1.0, True),
    ("mae", 1.0, False),
    ("central_80_mean_width", 0.5, False),
)


def _validation_selection(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    validation_metrics = {
        name: _metric_values([item for item in records if item["split"] == "validation"])
        for name, records in records_by_model.items()
    }
    expanded = {
        name: {**metrics, "coverage_error": abs(metrics["coverage_80"] - 0.8)}
        for name, metrics in validation_metrics.items()
    }
    scores = dict.fromkeys(validation_metrics, 0.0)
    contributions = {name: {} for name in validation_metrics}
    for metric, weight, descending in SELECTION_OBJECTIVES:
        ordered = sorted(
            validation_metrics,
            key=lambda name: (
                -expanded[name][metric] if descending else expanded[name][metric],
                name,
            ),
        )
        for rank, name in enumerate(ordered):
            contribution = rank * weight
            scores[name] += contribution
            contributions[name][metric] = contribution
    selected = min(scores, key=lambda name: (scores[name], name))
    return {
        "selected_model": selected,
        "selection_split": "validation",
        "selection_rule": "weighted rank sum; lower is better",
        "selection_weights": {
            metric: {"weight": weight, "higher_is_better": descending}
            for metric, weight, descending in SELECTION_OBJECTIVES
        },
        "rank_scores": scores,
        "rank_contributions": contributions,
        "validation_metrics": validation_metrics,
        "final_test_was_not_used_for_selection": True,
        "independent_validation": False,
        "independence_note": (
            "The time splits are ordered and the final test is sealed from selection, but the "
            "historical period overlaps earlier experiments. Future paper markets remain required."
        ),
    }


def _production_baseline(
    model: Any,
    *,
    model_name: str,
    event_start: datetime,
    event_end: datetime,
    training_cutoff: datetime,
    intervals: tuple[MarketInterval, ...],
) -> dict[str, Any]:
    forecast = _forecast_model(
        model,
        current_count=0,
        event_start=event_start,
        event_end=event_end,
        observed_at=event_start,
        observed_post_times=[],
    )
    probabilities = [float(forecast.probability(item.lower, item.upper)) for item in intervals]
    total = math.fsum(probabilities)
    probabilities = [value / total for value in probabilities]
    _, selected = _best_bands(probabilities)
    lower, upper = forecast.central_interval(0.8)
    provisional = training_cutoff < event_start
    return {
        "type": "baseline_0h_forecast",
        "model": model_name,
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "training_cutoff": training_cutoff.isoformat(),
        "status": "provisional_before_event_start" if provisional else "frozen_at_event_start",
        "requires_refresh_at_event_start": provisional,
        "warning": (
            "Pre-event context between the training cutoff and event start is not yet known; "
            "refresh and freeze P0 when the event opens."
            if provisional
            else None
        ),
        "expected_final": forecast.expected_final,
        "stddev_final": forecast.stddev_final,
        "central_80": [lower, upper],
        "integer_pmf_start": forecast.current_count,
        "integer_probabilities": list(forecast.integer_probabilities),
        "interval_probabilities": [
            {
                "label": interval.label,
                "lower": interval.lower,
                "upper": interval.upper,
                "probability": probabilities[index],
            }
            for index, interval in enumerate(intervals)
        ],
        "selected_indexes": list(selected),
        "selected_labels": [intervals[index].label for index in selected],
        "selected_probability": math.fsum(probabilities[index] for index in selected),
        "confidence_score": forecast.confidence_score,
        "low_confidence": forecast.low_confidence,
    }


def run_baseline_experiment(
    post_times: list[datetime],
    post_observations: list[PostObservation],
    *,
    history_start: datetime,
    history_end: datetime,
    event_start: datetime,
    event_end: datetime,
    intervals: tuple[MarketInterval, ...],
    interval_source: Path,
    output_root: Path,
    folds: int = 3,
    fold_days: int = 60,
    bootstrap_samples: int = 300,
    simulation_paths: int = 1200,
    seed: int = 20261017,
    data_reconciliation: dict[str, Any] | None = None,
) -> Path:
    if folds < 3:
        raise ValueError("Baseline experiment requires at least three sequential folds")
    event_duration = event_end - event_start
    if event_duration <= timedelta(0):
        raise ValueError("event_end must be after event_start")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / run_id
    _write_json(
        run_dir / "status.json",
        {"status": "running", "experiment": "baseline-0h", "run_id": run_id},
    )

    generated: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    for name in BASELINE_MODELS:
        records, weights = generate_predictions(
            post_times,
            post_observations,
            history_start=history_start,
            history_end=history_end,
            event_duration=event_duration,
            observation_fractions=(0.0,),
            intervals=intervals,
            model_name=name,
            folds=folds,
            fold_days=fold_days,
            evaluation_mode="ideal_complete_history",
            simulation_paths=simulation_paths,
            seed=seed,
        )
        generated[name] = records
        all_records.extend(records)
        all_weights.extend(weights)
        checkpoint = run_dir / "checkpoints" / name
        _write_jsonl(checkpoint / "predictions.jsonl", records)
        _write_jsonl(checkpoint / "weights.jsonl", weights)
        _write_json(
            run_dir / "status.json",
            {
                "status": "running",
                "experiment": "baseline-0h",
                "run_id": run_id,
                "last_completed_model": name,
            },
        )

    selection = _validation_selection(generated)
    selected_name = selection["selected_model"]
    summaries = {
        name: _summary(records, bootstrap_samples=bootstrap_samples, seed=seed + len(name))
        for name, records in generated.items()
    }
    baseline_test = [
        item for item in generated["empirical_weekly"] if item["split"] == "final_test"
    ]
    comparisons = {
        name: paired_block_bootstrap_comparison(
            [item for item in records if item["split"] == "final_test"],
            baseline_test,
            samples=bootstrap_samples,
            block_size=7,
            seed=seed + len(name),
        )
        for name, records in generated.items()
        if name != "empirical_weekly"
    }

    production_model = _fit_model(
        selected_name,
        post_times,
        history_start=history_start,
        model_end=history_end,
        event_duration=event_duration,
        simulation_paths=simulation_paths,
        seed=seed,
    )
    production_artifact = _model_artifact(production_model, selected_name, history_end)
    production_artifact["training_recipe"] = {
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_posts": len(post_times),
        "event_duration_hours": event_duration.total_seconds() / 3600,
        "selected_by": "validation_weighted_rank",
    }
    production_baseline = _production_baseline(
        production_model,
        model_name=selected_name,
        event_start=event_start,
        event_end=event_end,
        training_cutoff=history_end,
        intervals=intervals,
    )

    interval_bytes = interval_source.read_bytes()
    manifest = {
        "version": 1,
        "experiment": "baseline-0h",
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_posts": len(post_times),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "event_duration_hours": event_duration.total_seconds() / 3600,
        "observation_fraction": 0.0,
        "models": list(BASELINE_MODELS),
        "folds": folds,
        "fold_days": fold_days,
        "split_names": ["tuning"] * (folds - 2) + ["validation", "final_test"],
        "refit_days": 7,
        "daily_overlapping_samples": True,
        "non_overlapping_stride_days": 7,
        "analog_anchor_step_days": [4, 3],
        "uses_iso_week_groups": False,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_block_days": 7,
        "simulation_paths": simulation_paths,
        "random_seed": seed,
        "selection_uses_validation_only": True,
        "independent_validation": False,
        "interval_source": str(interval_source),
        "interval_source_sha256": hashlib.sha256(interval_bytes).hexdigest(),
        "intervals": [asdict(item) for item in intervals],
        "data_reconciliation": data_reconciliation,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "weights": "weights.jsonl",
            "summary": "summary.json",
            "selection": "selection.json",
            "checkpoints": "checkpoints/<model>/",
            "production_model": "production/model.json",
            "production_baseline": "production/baseline.json",
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "selection.json", selection)
    _write_jsonl(run_dir / "predictions.jsonl", all_records)
    _write_jsonl(run_dir / "weights.jsonl", all_weights)
    _write_json(
        run_dir / "summary.json",
        {"metrics": summaries, "vs_empirical_weekly": comparisons},
    )
    _write_json(run_dir / "production" / "model.json", production_artifact)
    _write_json(run_dir / "production" / "baseline.json", production_baseline)
    _write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "experiment": "baseline-0h",
            "run_id": run_id,
            "selected_model": selected_name,
        },
    )
    return run_dir

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import polymarket_bot.baseline_experiments as baseline_module
from polymarket_bot.baseline_experiments import _validation_selection, run_baseline_experiment
from polymarket_bot.database import PostObservation
from polymarket_bot.market_intervals import load_market_intervals
from polymarket_bot.model_experiments import GradientBoostedConformalModel
from polymarket_bot.model_experiments_v3 import _fit_model, _forecast_model


def _history(start: datetime, days: int) -> list[datetime]:
    return [start + timedelta(hours=4 * index) for index in range(days * 6)]


def _selection_record(*, split: str, probability: float, hit: bool) -> dict[str, object]:
    actual_index = 0 if hit else 1
    return {
        "split": split,
        "fold": 0,
        "window_start": "2026-01-01T00:00:00+00:00",
        "expected_final": 10.0 if hit else 30.0,
        "actual_final": 10,
        "actual_interval_index": actual_index,
        "probabilities": [probability, 1 - probability],
        "selected_indexes": [0],
        "selected_hit": int(hit),
        "selected_probability": probability,
        "top1_index": 0,
        "central_80": [0, 20],
        "central_80_probability": 0.8,
        "low_confidence": False,
        "confidence_score": 1.0,
        "novelty_score": 0.0,
        "overlap_effective_sample_size": 10.0,
        "data_count_delta": 0,
        "mean_import_delay_seconds": 0.0,
    }


def test_gbm_zero_hour_forecast_does_not_read_future_event_posts() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    event_start = start + timedelta(days=220)
    posts = _history(start, 220)
    model = _fit_model(
        "gbm_conformal_0h",
        posts,
        history_start=start,
        model_end=event_start,
        event_duration=timedelta(days=7),
        simulation_paths=100,
        seed=17,
    )

    without_future = _forecast_model(
        model,
        current_count=0,
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        observed_at=event_start,
        observed_post_times=[],
    )
    with_future = _forecast_model(
        model,
        current_count=0,
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        observed_at=event_start,
        observed_post_times=[event_start + timedelta(hours=index) for index in range(24)],
    )

    assert isinstance(model, GradientBoostedConformalModel)
    assert model.observation_fraction == 0
    assert without_future.integer_probabilities == with_future.integer_probabilities
    assert sum(with_future.integer_probabilities) == pytest.approx(1)


def test_baseline_selection_uses_validation_not_final_test() -> None:
    records = {
        "model_a": [
            _selection_record(split="validation", probability=0.9, hit=True),
            _selection_record(split="final_test", probability=0.01, hit=False),
        ],
        "model_b": [
            _selection_record(split="validation", probability=0.1, hit=False),
            _selection_record(split="final_test", probability=0.99, hit=True),
        ],
    }

    selection = _validation_selection(records)

    assert selection["selected_model"] == "model_a"
    assert selection["selection_split"] == "validation"
    assert selection["final_test_was_not_used_for_selection"] is True


def test_baseline_experiment_retains_predictions_weights_and_production_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    history_end = start + timedelta(days=240)
    posts = _history(start, 240)
    observations = [PostObservation(item, item + timedelta(minutes=1)) for item in posts]
    interval_path = Path("config/elon-musk-oct-13-20-2026-intervals.json")
    _, intervals = load_market_intervals(interval_path)
    monkeypatch.setattr(
        baseline_module,
        "BASELINE_MODELS",
        ("empirical_weekly", "gbm_conformal_0h"),
    )

    run_dir = run_baseline_experiment(
        posts,
        observations,
        history_start=start,
        history_end=history_end,
        event_start=history_end,
        event_end=history_end + timedelta(days=7),
        intervals=intervals,
        interval_source=interval_path,
        output_root=tmp_path,
        folds=3,
        fold_days=14,
        bootstrap_samples=0,
        simulation_paths=100,
        seed=17,
    )

    predictions = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    weights = (run_dir / "weights.jsonl").read_text(encoding="utf-8").splitlines()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    baseline = json.loads(
        (run_dir / "production" / "baseline.json").read_text(encoding="utf-8")
    )

    assert predictions
    assert weights
    assert selection["selected_model"] in baseline_module.BASELINE_MODELS
    assert baseline["status"] == "frozen_at_event_start"
    assert sum(baseline["integer_probabilities"]) == pytest.approx(1)
    for model in baseline_module.BASELINE_MODELS:
        assert (run_dir / "checkpoints" / model / "predictions.jsonl").exists()
        assert (run_dir / "checkpoints" / model / "weights.jsonl").exists()
    assert (run_dir / "production" / "model.json").exists()

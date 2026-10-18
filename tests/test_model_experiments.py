from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polymarket_bot.market_intervals import load_market_intervals
from polymarket_bot.model_experiments import (
    GradientBoostedRegressor,
    _metric_values,
    apply_beta_uncertainty,
    beta_quantile,
    build_routed_strategy_records,
    paired_block_bootstrap_comparison,
)


def test_real_market_interval_config_is_a_complete_partition() -> None:
    path = Path("config/elon-musk-oct-13-20-2026-intervals.json")

    slug, intervals = load_market_intervals(path)

    assert slug == "elon-musk-of-tweets-october-13-october-20-2026"
    assert len(intervals) == 26
    assert intervals[0].label == "<20"
    assert intervals[-1].label == "500+"
    assert intervals[-1].upper is None


def test_beta_quantile_matches_uniform_distribution() -> None:
    assert beta_quantile(0.10, 1, 1) == pytest.approx(0.10, abs=1e-8)


def test_beta_uncertainty_records_a_conservative_probability() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    records = []
    for index in range(30):
        window_start = start + timedelta(days=index * 8)
        records.append(
            {
                "sample_id": str(index),
                "model": "adaptive_reconciled",
                "observation_fraction": 0.5,
                "window_start": window_start.isoformat(),
                "window_end": (window_start + timedelta(days=7)).isoformat(),
                "selected_probability": 0.60,
                "decision_probability": 0.60,
                "selected_hit": int(index % 2 == 0),
                "effective_sample_size": 40.0,
                "selected_lower_probability": None,
                "calibration": {},
            }
        )

    calibrated = apply_beta_uncertainty(records)

    assert calibrated[-1]["model"] == "adaptive_beta"
    assert calibrated[-1]["calibration"]["winner_optimism"] > 0
    assert calibrated[-1]["selected_lower_probability"] < calibrated[-1]["decision_probability"]


def test_gradient_boosted_regressor_preserves_serializable_stump_weights() -> None:
    features = [(float(value),) for value in range(20)]
    targets = [2 * value + 3 for value in range(20)]

    model = GradientBoostedRegressor.fit(features, targets, estimators=40)

    assert model.stumps
    assert model.predict((18.0,)) > model.predict((2.0,))
    assert model.stumps[0].feature == 0


def test_probability_metrics_reward_a_perfect_distribution() -> None:
    record = {
        "expected_final": 25.0,
        "actual_final": 25,
        "probabilities": [0.0, 1.0, 0.0],
        "actual_interval_index": 1,
        "central_80": [20, 39],
        "top1_index": 1,
        "selected_hit": 1,
        "decision_probability": 1.0,
        "selected_lower_probability": None,
    }

    result = _metric_values([record])

    assert result["mae"] == 0
    assert result["log_loss"] == 0
    assert result["brier_score"] == 0
    assert result["ranked_probability_score"] == 0


def test_paired_comparison_uses_matching_walk_forward_samples() -> None:
    common = {
        "fold": 0,
        "observation_fraction": 0.5,
        "window_start": "2026-01-01T00:00:00+00:00",
        "expected_final": 25.0,
        "actual_final": 25,
        "actual_interval_index": 1,
        "central_80": [20, 39],
        "top1_index": 1,
        "selected_hit": 1,
        "selected_lower_probability": None,
    }
    model = {**common, "probabilities": [0.0, 1.0, 0.0], "decision_probability": 1.0}
    baseline = {
        **common,
        "expected_final": 35.0,
        "probabilities": [0.2, 0.6, 0.2],
        "decision_probability": 0.6,
    }

    result = paired_block_bootstrap_comparison([model], [baseline], samples=0, block_size=7, seed=1)

    assert result["mae_improvement"] == 10
    assert result["log_loss_improvement"] > 0
    assert result["paired_block_bootstrap_95"] == {}


def test_routed_strategy_uses_gbm_early_and_adaptive_at_126_hours() -> None:
    records_by_model = {"gbm_conformal": [], "adaptive_reconciled": []}
    for model_name, records in records_by_model.items():
        for fraction in (0.142857, 0.25, 0.5, 0.75):
            records.append(
                {
                    "sample_id": f"{model_name}-{fraction}",
                    "model": model_name,
                    "fold": 0,
                    "observation_fraction": fraction,
                    "window_start": "2026-01-01T00:00:00+00:00",
                    "calibration": {},
                }
            )

    routed = build_routed_strategy_records(records_by_model)

    assert len(routed) == 4
    assert {
        f"{item['observation_fraction']:.6f}": item["route_source_model"]
        for item in routed
    } == {
        "0.142857": "gbm_conformal",
        "0.250000": "gbm_conformal",
        "0.500000": "gbm_conformal",
        "0.750000": "adaptive_reconciled",
    }
    assert all(item["model"] == "gbm_24_84_adaptive_126" for item in routed)

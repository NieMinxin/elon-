from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polymarket_bot.database import PostObservation
from polymarket_bot.market_intervals import load_market_intervals
from polymarket_bot.model_experiments_v3 import (
    AnalogConfig,
    AnalogTrajectoryModel,
    DynamicNegativeBinomialModel,
    _overlap_effective_size,
    generate_predictions,
)


def _history(start: datetime, days: int) -> list[datetime]:
    return [
        start + timedelta(hours=4 * index)
        for index in range(days * 6)
    ]


def test_analog_disjoint_forecast_has_consistent_pmf_and_overlap_ess() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    event_start = start + timedelta(days=180)
    posts = _history(start, 180)
    model = AnalogTrajectoryModel.fit(
        posts,
        start=start,
        end=event_start,
        event_duration=timedelta(days=7),
        config=AnalogConfig(
            "test",
            "disjoint",
            ("quiet", "acceleration", "active_streak", "burst"),
        ),
    )
    observed_at = event_start + timedelta(days=3)
    observations = _history(event_start, 3)

    forecast = model.forecast(
        current_count=len(observations),
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        now=observed_at,
        observed_post_times=observations,
    )

    assert sum(forecast.integer_probabilities) == pytest.approx(1)
    assert 0 < forecast.overlap_effective_sample_size <= forecast.raw_effective_sample_size
    lower, upper = forecast.central_interval(0.8)
    assert float(forecast.probability(lower, upper)) >= 0.8


def test_analog_windows_move_backward_by_four_then_three_days() -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    history_end = datetime(2026, 1, 8, 16, tzinfo=UTC)
    posts = _history(start, (history_end - start).days)
    model = AnalogTrajectoryModel.fit(
        posts,
        start=start,
        end=history_end,
        event_duration=timedelta(days=7),
        config=AnalogConfig("test"),
    )

    starts = model._window_starts(history_end)
    ends_descending = sorted(
        (window_start + timedelta(days=7) for window_start in starts), reverse=True
    )
    backward_steps = [
        (left - right).days
        for left, right in zip(ends_descending, ends_descending[1:], strict=False)
    ]

    assert ends_descending[0] == history_end
    assert backward_steps[:6] == [4, 3, 4, 3, 4, 3]
    assert all(step in {3, 4} for step in backward_steps)
    assert all(
        window_end - window_start == timedelta(days=7)
        for window_start, window_end in zip(starts, sorted(ends_descending), strict=True)
    )


def test_overlap_ess_uses_window_overlap_not_iso_week_boundaries() -> None:
    duration = timedelta(days=7)
    independent = [
        datetime(2025, 12, 22, tzinfo=UTC),
        datetime(2025, 12, 29, tzinfo=UTC),
        datetime(2026, 1, 5, tzinfo=UTC),
    ]
    overlapping = [
        datetime(2025, 12, 29, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    ]

    assert _overlap_effective_size(independent, [1, 1, 1], duration) == pytest.approx(3)
    overlap_ess = _overlap_effective_size(overlapping, [1, 1], duration)
    shifted_ess = _overlap_effective_size(
        [item + timedelta(days=370) for item in overlapping], [1, 1], duration
    )
    assert 1 < overlap_ess < 2
    assert shifted_ess == pytest.approx(overlap_ess)


def test_dynamic_negative_binomial_is_reproducible() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    event_start = start + timedelta(days=180)
    posts = _history(start, 180)
    model = DynamicNegativeBinomialModel.fit(
        posts,
        start=start,
        end=event_start,
        event_duration=timedelta(days=7),
        simulation_paths=100,
        random_seed=17,
    )
    kwargs = {
        "current_count": 18,
        "event_start": event_start,
        "event_end": event_start + timedelta(days=7),
        "now": event_start + timedelta(days=3),
        "observed_post_times": _history(event_start, 3),
    }

    first = model.forecast(**kwargs)
    second = model.forecast(**kwargs)

    assert model.activity_center < 0
    assert first.integer_probabilities == second.integer_probabilities
    assert first.expected_final >= 18
    assert first.expected_final < 80
    assert sum(first.integer_probabilities) == pytest.approx(1)


def test_weekly_empirical_refits_more_often_than_static() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=240)
    posts = _history(start, 240)
    observations = [PostObservation(item, item + timedelta(minutes=5)) for item in posts]
    _, intervals = load_market_intervals(
        Path("config/elon-musk-oct-13-20-2026-intervals.json")
    )
    common = {
        "post_times": posts,
        "observations": observations,
        "history_start": start,
        "history_end": end,
        "event_duration": timedelta(days=7),
        "observation_fractions": (0.5,),
        "intervals": intervals,
        "folds": 3,
        "fold_days": 14,
        "evaluation_mode": "ideal_complete_history",
        "simulation_paths": 100,
        "seed": 17,
    }
    static_records, static_weights = generate_predictions(
        model_name="empirical_static", **common
    )
    weekly_records, weekly_weights = generate_predictions(
        model_name="empirical_weekly", **common
    )

    assert len(static_records) == len(weekly_records)
    assert len(weekly_weights) > len(static_weights)
    assert {item["split"] for item in weekly_records} == {
        "tuning",
        "validation",
        "final_test",
    }
    assert any(item["non_overlapping_7d"] for item in weekly_records)

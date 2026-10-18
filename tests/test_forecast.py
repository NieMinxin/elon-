from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from polymarket_bot.forecast import (
    AdaptiveWindowModel,
    CountForecast,
    EmpiricalWindowModel,
    SeasonalGammaPoisson,
    correct_baseline_forecast,
    walk_forward_backtest,
)


def make_history(start: datetime, hours: int, every_hours: int) -> list[datetime]:
    return [start + timedelta(hours=index) for index in range(0, hours, every_hours)]


def test_forecast_partition_probabilities_sum_to_one() -> None:
    history_start = datetime(2026, 1, 1, tzinfo=UTC)
    event_start = history_start + timedelta(days=60)
    model = SeasonalGammaPoisson.fit(
        make_history(history_start, 60 * 24, 2),
        start=history_start,
        end=event_start,
        prior_strength_hours=Decimal("168"),
    )
    forecast = model.forecast(
        current_count=20,
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        now=event_start + timedelta(days=3),
    )
    intervals = [(0, 19), (20, 39), (40, 59), (60, 79), (80, None)]
    total = sum((forecast.probability(low, high) for low, high in intervals), Decimal("0"))

    assert abs(total - Decimal("1")) < Decimal("0.0000001")
    assert forecast.expected_final > 20


def test_elapsed_event_has_point_mass_at_current_count() -> None:
    history_start = datetime(2026, 1, 1, tzinfo=UTC)
    event_start = history_start + timedelta(days=30)
    model = SeasonalGammaPoisson.fit(
        make_history(history_start, 30 * 24, 4),
        start=history_start,
        end=event_start,
        prior_strength_hours=Decimal("24"),
    )
    forecast = model.forecast(
        current_count=55,
        event_start=event_start,
        event_end=event_start + timedelta(days=1),
        now=event_start + timedelta(days=2),
    )

    assert forecast.probability(40, 59) == 1
    assert forecast.probability(60, 79) == 0


def test_realtime_correction_preserves_p0_then_applies_relative_update() -> None:
    baseline = CountForecast(0, 0, 0, 0, 0, 168, integer_probabilities=(0.2, 0.5, 0.3))
    reference = CountForecast(0, 0, 0, 0, 0, 168, integer_probabilities=(0.2, 0.5, 0.3))
    updated = CountForecast(0, 0, 0, 0, 0, 156, integer_probabilities=(0.1, 0.3, 0.6))

    at_open = correct_baseline_forecast(baseline, reference, updated, strength=0)
    corrected = correct_baseline_forecast(baseline, reference, updated, strength=1)

    assert at_open.integer_probabilities == pytest.approx(baseline.integer_probabilities)
    assert corrected.integer_probabilities == pytest.approx(updated.integer_probabilities)
    assert corrected.base_expected_final == pytest.approx(baseline.expected_final)


def test_walk_forward_backtest_reports_metrics() -> None:
    history_start = datetime(2025, 1, 1, tzinfo=UTC)
    history_end = history_start + timedelta(days=220)
    posts = make_history(history_start, 220 * 24, 3)

    result = walk_forward_backtest(
        posts,
        history_start=history_start,
        history_end=history_end,
        event_duration=timedelta(days=7),
        observation_fraction=0.5,
        prior_strength_hours=Decimal("168"),
        holdout_days=60,
    )

    assert result.samples == 54
    assert result.mean_absolute_error < 2
    assert 0 <= result.coverage_80 <= 1


def test_empirical_forecast_uses_observed_count_and_partitions_ranges() -> None:
    history_start = datetime(2025, 1, 1, tzinfo=UTC)
    event_start = history_start + timedelta(days=120)
    posts = make_history(history_start, 120 * 24, 3)
    model = EmpiricalWindowModel.fit(
        posts,
        start=history_start,
        end=event_start,
        event_duration=timedelta(days=7),
    )

    forecast = model.forecast(
        current_count=40,
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        now=event_start + timedelta(days=3),
    )
    intervals = [(0, 39), (40, 59), (60, 79), (80, None)]
    total = sum((forecast.probability(low, high) for low, high in intervals), Decimal("0"))

    assert abs(total - Decimal("1")) < Decimal("0.0000001")
    assert forecast.expected_final >= 40
    assert forecast.empirical_finals


def test_empirical_kernel_never_assigns_probability_below_current_count() -> None:
    empirical = CountForecast(10, 10, 2, 0, 0, 1, (10, 10, 11), 4)

    assert empirical.probability(0, 9) == 0
    assert abs(empirical.probability(10, None) - Decimal("1")) < Decimal("0.0000001")


def test_adaptive_forecast_uses_recent_rate_regime() -> None:
    history_start = datetime(2025, 1, 1, tzinfo=UTC)
    event_start = history_start + timedelta(days=150)
    history = make_history(history_start, 150 * 24, 3)
    model = AdaptiveWindowModel.fit(
        history,
        start=history_start,
        end=event_start,
        event_duration=timedelta(days=7),
    )
    now = event_start + timedelta(days=1)
    observations = [event_start + timedelta(minutes=30 * index) for index in range(48)]

    forecast = model.forecast(
        current_count=len(observations),
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        now=now,
        observed_post_times=observations,
    )

    assert forecast.realtime_multiplier > 1
    assert forecast.regime in {"high", "extreme_high"}
    assert dict(forecast.recent_counts)[24] == 48
    assert forecast.expected_final >= forecast.current_count


def test_adaptive_forecast_uses_official_count_for_reconciliation_feature() -> None:
    history_start = datetime(2025, 1, 1, tzinfo=UTC)
    event_start = history_start + timedelta(days=150)
    history = make_history(history_start, 150 * 24, 3)
    model = AdaptiveWindowModel.fit(
        history,
        start=history_start,
        end=event_start,
        event_duration=timedelta(days=7),
    )
    observations = [event_start + timedelta(minutes=30 * index) for index in range(40)]

    forecast = model.forecast(
        current_count=48,
        event_start=event_start,
        event_end=event_start + timedelta(days=7),
        now=event_start + timedelta(days=1),
        observed_post_times=observations,
    )

    assert forecast.source_count_delta == 8
    assert forecast.effective_sample_size >= 20


def test_weighted_empirical_probabilities_sum_to_one() -> None:
    forecast = CountForecast(
        10,
        20,
        5,
        0,
        0,
        1,
        (15, 25),
        0,
        (0.9, 0.1),
    )

    assert forecast.probability(10, 19) == Decimal("0.9")
    assert forecast.probability(20, None) == Decimal("0.1")


def test_central_interval_uses_the_same_smoothed_integer_distribution() -> None:
    forecast = CountForecast(
        current_count=100,
        expected_final=130,
        stddev_final=20,
        posterior_alpha=0,
        posterior_beta=0,
        remaining_exposure=24,
        empirical_finals=(100, 102, 160),
        kernel_bandwidth=12,
        empirical_weights=(0.45, 0.45, 0.10),
    )

    lower, upper = forecast.central_interval(0.8)

    assert float(forecast.probability(lower, upper)) >= 0.8
    assert forecast.expected_final == pytest.approx(
        sum(
            (forecast.current_count + offset) * probability
            for offset, probability in enumerate(forecast.integer_probabilities)
        )
    )


def test_completed_forecast_is_an_exact_point_mass() -> None:
    forecast = CountForecast(241, 999, 99, 2, 3, 0)

    assert forecast.integer_probabilities == (1.0,)
    assert forecast.expected_final == 241
    assert forecast.stddev_final == 0
    assert forecast.central_interval(0.8) == (241, 241)

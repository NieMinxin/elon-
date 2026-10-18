from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from polymarket_bot.database import PostObservation
from polymarket_bot.gru_experiments import (
    DualEncoderGRU,
    GruExperimentConfig,
    MixtureCountHead,
    _align_on_or_after,
    _config_from_manifest,
    _forecast_from_pmf,
    _load_gru_checkpoint,
    _market_probabilities,
    _mixture_pmf,
    _temperature_scale_pmfs,
    build_fold_definitions,
    clean_post_history,
)
from polymarket_bot.market_intervals import MarketInterval


def test_separated_mixture_initialization_starts_with_distinct_components() -> None:
    head = MixtureCountHead(4, 2, 100.0, separate_components=True)

    _, means, dispersion = head(torch.zeros((1, 4)))

    assert means[0, 0] < means[0, 1]
    assert dispersion[0, 0] < dispersion[0, 1]


def test_daily_pre_history_gru_accepts_hourly_input() -> None:
    model = DualEncoderGRU(
        static_size=3,
        hidden_size=4,
        components=2,
        initial_mean=100.0,
        daily_pre_history=True,
        separate_components=True,
    )

    output = model(
        torch.zeros((2, 48, 1)),
        torch.zeros((2, 168, 2)),
        torch.tensor([0, 24]),
        torch.zeros((2, 3)),
    )

    assert all(value.shape == (2, 2) for value in output)


def test_temperature_scaling_preserves_valid_pmfs() -> None:
    values = np.asarray([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], dtype=np.float64)

    sharpened = _temperature_scale_pmfs(values, 0.75)
    softened = _temperature_scale_pmfs(values, 1.5)

    assert np.allclose(sharpened.sum(axis=1), 1)
    assert np.allclose(softened.sum(axis=1), 1)
    assert sharpened[0, 2] > values[0, 2]
    assert softened[0, 2] < values[0, 2]


def test_historical_training_alignment_can_move_before_reference() -> None:
    reference = datetime(2026, 10, 16, 16, tzinfo=UTC)
    minimum = datetime(2025, 11, 30, 5, tzinfo=UTC)

    aligned = _align_on_or_after(reference, minimum)

    assert aligned == datetime(2025, 11, 30, 16, tzinfo=UTC)
    assert aligned >= minimum
    assert aligned - timedelta(days=1) < minimum


def test_gru_fold_definitions_are_non_overlapping_and_embargoed() -> None:
    duration = timedelta(days=7)
    config = GruExperimentConfig(folds=3, evaluation_events_per_fold=4)
    folds = build_fold_definitions(
        history_start=datetime(2025, 10, 1, 16, tzinfo=UTC),
        history_end=datetime(2026, 10, 16, 16, tzinfo=UTC),
        event_alignment=datetime(2026, 10, 16, 16, tzinfo=UTC),
        event_duration=duration,
        config=config,
    )

    assert len(folds) == 3
    for fold in folds:
        assert all(
            right - left == duration
            for left, right in zip(
                fold.validation_starts, fold.validation_starts[1:], strict=False
            )
        )
        assert all(
            right - left == duration
            for left, right in zip(fold.test_starts, fold.test_starts[1:], strict=False)
        )
        validation_end = fold.validation_starts[-1] + duration
        assert fold.test_starts[0] - validation_end == timedelta(days=7)
        assert fold.validation_starts[0] - fold.train_label_cutoff == timedelta(days=7)


def test_clean_post_history_keeps_valid_equal_timestamps() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    values = [
        start,
        start + timedelta(days=1, hours=12),
        start + timedelta(days=1, hours=12),
        start + timedelta(days=2),
    ]
    observations = [PostObservation(value, value + timedelta(seconds=3)) for value in values]

    cleaned, report = clean_post_history(values, observations)

    assert len(cleaned) == 3
    assert report["dropped_incomplete_prefix_rows"] == 1
    assert report["exact_timestamp_collisions_preserved"] == 1
    assert report["maximum_hourly_count"] == 2


def test_clean_post_history_drops_prefix_before_long_coverage_gap() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    values = [
        start,
        start + timedelta(hours=1),
        start + timedelta(days=20),
        start + timedelta(days=21, hours=1),
    ]
    observations = [PostObservation(value, value) for value in values]

    cleaned, report = clean_post_history(values, observations)

    assert cleaned == (start + timedelta(days=21, hours=1),)
    assert report["long_internal_gaps_detected"] == 1
    assert report["largest_internal_gap_hours"] > 24 * 19


def test_negative_binomial_mixture_maps_to_arbitrary_market_intervals() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    means = torch.tensor([[2.0, 8.0]])
    dispersion = torch.tensor([[3.0, 4.0]])
    pmf = _mixture_pmf(logits, means, dispersion, 40)
    intervals = (
        MarketInterval("<5", 0, 4),
        MarketInterval("5-8", 5, 8),
        MarketInterval("9+", 9, None),
    )

    probabilities = _market_probabilities(pmf, torch.tensor([1]), intervals)

    assert torch.allclose(pmf.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.all(probabilities >= 0)


def test_gru_forecast_has_no_probability_below_current_count() -> None:
    forecast = _forecast_from_pmf(241, [0.2, 0.3, 0.5])

    assert float(forecast.probability(0, 240)) == 0
    assert float(forecast.probability(241, 241)) == 0.2
    assert float(forecast.probability(241, None)) == 1


def test_gru_checkpoint_round_trip_preserves_source_scaler(tmp_path: Path) -> None:
    config = GruExperimentConfig(hidden_size=4, mixture_components=2, ensemble_members=2)
    model = DualEncoderGRU(
        static_size=3,
        hidden_size=config.hidden_size,
        components=config.mixture_components,
        initial_mean=25.0,
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": {"mean": [1.0, 2.0, 3.0], "scale": [2.0, 4.0, 5.0]},
            "config": asdict(config),
            "static_size": 3,
            "initial_mean": 25.0,
        },
        path,
    )

    restored, scaler, artifact = _load_gru_checkpoint(path, device=torch.device("cpu"))

    assert isinstance(restored, DualEncoderGRU)
    assert scaler.mean == (1.0, 2.0, 3.0)
    assert scaler.scale == (2.0, 4.0, 5.0)
    assert artifact["initial_mean"] == 25.0


def test_gru_manifest_config_restores_tuple_fields() -> None:
    payload = asdict(GruExperimentConfig(ensemble_members=2))
    payload["observation_hours"] = list(payload["observation_hours"])
    payload["training_step_days"] = list(payload["training_step_days"])

    restored = _config_from_manifest(payload)

    assert isinstance(restored.observation_hours, tuple)
    assert isinstance(restored.training_step_days, tuple)

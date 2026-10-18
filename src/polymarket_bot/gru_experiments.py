from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import statistics
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from polymarket_bot.database import PostObservation
from polymarket_bot.forecast import (
    AdaptiveWindowModel,
    CountForecast,
    correct_baseline_forecast,
)
from polymarket_bot.market_intervals import MarketInterval, actual_interval_index
from polymarket_bot.model_experiments import GradientBoostedConformalModel, _best_bands

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GruExperimentConfig:
    pre_event_days: int = 28
    observation_hours: tuple[int, ...] = (0, 24, 42, 84, 126)
    training_observation_step_hours: int = 6
    training_step_days: tuple[int, int] = (4, 3)
    folds: int = 3
    evaluation_events_per_fold: int = 4
    embargo_days: int = 7
    hidden_size: int = 16
    mixture_components: int = 2
    max_remaining: int = 700
    epochs: int = 180
    patience: int = 25
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    rps_weight: float = 0.5
    ensemble_members: int = 5
    seed: int = 20261017

    def validate(self, event_duration: timedelta) -> None:
        duration_hours = round(event_duration.total_seconds() / 3600)
        if event_duration != timedelta(hours=duration_hours) or duration_hours <= 0:
            raise ValueError("GRU experiments require a positive whole-hour event duration")
        if any(value < 0 or value >= duration_hours for value in self.observation_hours):
            raise ValueError("Observation hours must be inside the event duration")
        if tuple(sorted(set(self.observation_hours))) != self.observation_hours:
            raise ValueError("Observation hours must be unique and sorted")
        if not 1 <= self.training_observation_step_hours < duration_hours:
            raise ValueError("Training observation step must be inside the event duration")
        if self.pre_event_days < 7:
            raise ValueError("At least seven pre-event days are required")
        if self.folds < 1 or self.evaluation_events_per_fold < 2:
            raise ValueError("At least one fold and two evaluation events per fold are required")
        if self.embargo_days < math.ceil(event_duration.total_seconds() / 86400):
            raise ValueError("Embargo must be at least one event duration")
        if self.hidden_size < 4 or self.mixture_components < 1:
            raise ValueError("GRU hidden size and mixture component count are invalid")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("Training limits must be positive")
        if self.max_remaining < 100 or not 0 <= self.rps_weight <= 10:
            raise ValueError("Distribution support or RPS weight is invalid")
        if self.ensemble_members < 2:
            raise ValueError("Deep ensemble requires at least two members")


@dataclass(frozen=True)
class SequenceSample:
    fold: int
    split: str
    event_start: datetime
    event_end: datetime
    observed_hours: int
    pre_counts: tuple[int, ...]
    event_counts: tuple[int, ...]
    static_features: tuple[float, ...]
    current_count: int
    final_count: int
    remaining_count: int
    actual_interval_index: int
    event_weight: float

    @property
    def sample_key(self) -> str:
        return f"{self.fold}|{self.split}|{self.event_start.isoformat()}|{self.observed_hours}"


@dataclass(frozen=True)
class FoldDefinition:
    fold: int
    train_label_cutoff: datetime
    validation_starts: tuple[datetime, ...]
    test_starts: tuple[datetime, ...]


@dataclass(frozen=True)
class FeatureScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, samples: list[SequenceSample]) -> FeatureScaler:
        if not samples:
            raise ValueError("Cannot fit feature scaler without training samples")
        values = np.asarray([item.static_features for item in samples], dtype=np.float32)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(tuple(float(item) for item in mean), tuple(float(item) for item in scale))

    def transform(self, values: Iterable[float]) -> tuple[float, ...]:
        return tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(values, self.mean, self.scale, strict=True)
        )


class MixtureCountHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        components: int,
        initial_mean: float,
        *,
        separate_components: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(16, min(64, input_size * 2))
        self.components = components
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, components * 3),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias[:components].zero_()
            center = math.log(max(0.5, initial_mean))
            if separate_components and components > 1:
                final.bias[components : components * 2].copy_(
                    center + torch.linspace(-0.35, 0.35, components)
                )
                final.bias[components * 2 :].copy_(torch.linspace(1.25, 2.75, components))
            else:
                final.bias[components : components * 2].fill_(center)
                final.bias[components * 2 :].fill_(2.0)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.network(values)
        logits, log_means, raw_dispersion = raw.chunk(3, dim=-1)
        means = torch.exp(torch.clamp(log_means, min=-4.0, max=8.0))
        dispersion = F.softplus(raw_dispersion) + 0.25
        return logits, means, dispersion


class AggregateMLP(nn.Module):
    def __init__(self, static_size: int, components: int, initial_mean: float) -> None:
        super().__init__()
        self.head = MixtureCountHead(static_size, components, initial_mean)

    def forward(
        self,
        pre_counts: Tensor,
        event_values: Tensor,
        observed_hours: Tensor,
        static_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del pre_counts, event_values, observed_hours
        return self.head(static_features)


class DualEncoderGRU(nn.Module):
    def __init__(
        self,
        *,
        static_size: int,
        hidden_size: int,
        components: int,
        initial_mean: float,
        daily_pre_history: bool = False,
        separate_components: bool = False,
    ) -> None:
        super().__init__()
        self.daily_pre_history = daily_pre_history
        self.pre_encoder = nn.GRU(1, hidden_size, batch_first=True)
        self.event_encoder = nn.GRU(2, hidden_size, batch_first=True)
        self.no_event = nn.Parameter(torch.zeros(hidden_size))
        self.head = MixtureCountHead(
            hidden_size * 2 + static_size,
            components,
            initial_mean,
            separate_components=separate_components,
        )

    def forward(
        self,
        pre_counts: Tensor,
        event_values: Tensor,
        observed_hours: Tensor,
        static_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.daily_pre_history:
            if pre_counts.shape[1] % 24:
                raise ValueError("Pre-event hourly sequence must contain complete UTC days")
            raw_hourly = torch.expm1(pre_counts)
            daily_counts = raw_hourly.reshape(raw_hourly.shape[0], -1, 24, 1).sum(dim=2)
            pre_counts = torch.log1p(daily_counts)
        _, pre_hidden = self.pre_encoder(pre_counts)
        event_output, _ = self.event_encoder(event_values)
        last_indexes = torch.clamp(observed_hours - 1, min=0)
        batch_indexes = torch.arange(event_output.shape[0], device=event_output.device)
        event_hidden = event_output[batch_indexes, last_indexes]
        empty = observed_hours == 0
        if torch.any(empty):
            event_hidden = torch.where(empty[:, None], self.no_event[None, :], event_hidden)
        combined = torch.cat((pre_hidden[-1], event_hidden, static_features), dim=-1)
        return self.head(combined)


def _count(values: tuple[datetime, ...], start: datetime, end: datetime) -> int:
    return bisect_left(values, end) - bisect_left(values, start)


def _hourly_counts(values: tuple[datetime, ...], start: datetime, hours: int) -> tuple[int, ...]:
    return tuple(
        _count(values, start + timedelta(hours=index), start + timedelta(hours=index + 1))
        for index in range(hours)
    )


def _align_on_or_after(reference: datetime, minimum: datetime) -> datetime:
    steps = math.ceil((minimum - reference).total_seconds() / 86400)
    return reference + timedelta(days=steps)


def _latest_weekly_start(
    reference: datetime, *, history_end: datetime, event_duration: timedelta
) -> datetime:
    maximum = history_end - event_duration
    steps = math.floor((maximum - reference).total_seconds() / timedelta(days=7).total_seconds())
    return reference + timedelta(days=steps * 7)


def build_fold_definitions(
    *,
    history_start: datetime,
    history_end: datetime,
    event_alignment: datetime,
    event_duration: timedelta,
    config: GruExperimentConfig,
) -> tuple[FoldDefinition, ...]:
    latest = _latest_weekly_start(
        event_alignment, history_end=history_end, event_duration=event_duration
    )
    total_test_events = config.folds * config.evaluation_events_per_fold
    first_test = latest - timedelta(days=7 * (total_test_events - 1))
    folds = []
    for fold in range(config.folds):
        test_first = first_test + timedelta(days=7 * fold * config.evaluation_events_per_fold)
        test_starts = tuple(
            test_first + timedelta(days=7 * index)
            for index in range(config.evaluation_events_per_fold)
        )
        validation_last = test_first - timedelta(
            days=config.embargo_days + event_duration.total_seconds() / 86400
        )
        validation_first = validation_last - timedelta(
            days=7 * (config.evaluation_events_per_fold - 1)
        )
        validation_starts = tuple(
            validation_first + timedelta(days=7 * index)
            for index in range(config.evaluation_events_per_fold)
        )
        train_label_cutoff = validation_first - timedelta(days=config.embargo_days)
        if history_start + timedelta(days=config.pre_event_days) >= train_label_cutoff:
            raise ValueError("Not enough history before the first purged fold")
        folds.append(
            FoldDefinition(fold, train_label_cutoff, validation_starts, test_starts)
        )
    return tuple(folds)


def _training_starts(
    *,
    history_start: datetime,
    event_alignment: datetime,
    event_duration: timedelta,
    label_cutoff: datetime,
    config: GruExperimentConfig,
) -> tuple[datetime, ...]:
    minimum = history_start + timedelta(days=config.pre_event_days)
    cursor = _align_on_or_after(event_alignment, minimum)
    starts = []
    step_index = 0
    while cursor + event_duration <= label_cutoff:
        starts.append(cursor)
        cursor += timedelta(days=config.training_step_days[step_index % 2])
        step_index += 1
    return tuple(starts)


def _static_features(
    values: tuple[datetime, ...],
    *,
    event_start: datetime,
    observed_at: datetime,
    event_end: datetime,
    current_count: int,
    hourly_rate: float,
) -> tuple[float, ...]:
    elapsed = max(0.0, (observed_at - event_start).total_seconds() / 3600)
    remaining = max(0.0, (event_end - observed_at).total_seconds() / 3600)
    result = [math.log1p(current_count), elapsed / 168.0, remaining / 168.0]
    for hours in (1, 3, 6, 12, 24, 72):
        actual = _count(values, observed_at - timedelta(hours=hours), observed_at)
        expected = hourly_rate * hours
        result.append(math.log((actual + 2.0) / (expected + 2.0)))
    weekday_angle = 2 * math.pi * event_start.weekday() / 7
    hour_angle = 2 * math.pi * event_start.hour / 24
    result.extend(
        (
            math.sin(weekday_angle),
            math.cos(weekday_angle),
            math.sin(hour_angle),
            math.cos(hour_angle),
            0.0,  # tracking total minus visible sequence count in ideal-history experiments
            0.0,  # missing-data indicator in ideal-history experiments
        )
    )
    return tuple(result)


def _build_samples(
    values: tuple[datetime, ...],
    *,
    fold: int,
    split: str,
    starts: tuple[datetime, ...],
    event_duration: timedelta,
    history_start: datetime,
    training_cutoff: datetime,
    intervals: tuple[MarketInterval, ...],
    config: GruExperimentConfig,
    observation_hours: tuple[int, ...] | None = None,
) -> list[SequenceSample]:
    duration_hours = round(event_duration.total_seconds() / 3600)
    training_posts = _count(values, history_start, training_cutoff)
    training_hours = max(1.0, (training_cutoff - history_start).total_seconds() / 3600)
    hourly_rate = training_posts / training_hours
    sample_hours = observation_hours or config.observation_hours
    samples = []
    for event_start in starts:
        event_end = event_start + event_duration
        pre_start = event_start - timedelta(days=config.pre_event_days)
        pre_counts = _hourly_counts(values, pre_start, config.pre_event_days * 24)
        full_event_counts = _hourly_counts(values, event_start, duration_hours)
        final_count = sum(full_event_counts)
        interval_index = actual_interval_index(intervals, final_count)
        event_weight = 1.0 / len(sample_hours)
        for observed_hours in sample_hours:
            observed_at = event_start + timedelta(hours=observed_hours)
            current = sum(full_event_counts[:observed_hours])
            visible_counts = full_event_counts[:observed_hours] + (0,) * (
                duration_hours - observed_hours
            )
            samples.append(
                SequenceSample(
                    fold=fold,
                    split=split,
                    event_start=event_start,
                    event_end=event_end,
                    observed_hours=observed_hours,
                    pre_counts=pre_counts,
                    event_counts=visible_counts,
                    static_features=_static_features(
                        values,
                        event_start=event_start,
                        observed_at=observed_at,
                        event_end=event_end,
                        current_count=current,
                        hourly_rate=hourly_rate,
                    ),
                    current_count=current,
                    final_count=final_count,
                    remaining_count=final_count - current,
                    actual_interval_index=interval_index,
                    event_weight=event_weight,
                )
            )
    return samples


def clean_post_history(
    post_times: list[datetime], observations: list[PostObservation]
) -> tuple[tuple[datetime, ...], dict[str, Any]]:
    normalized = tuple(sorted(item.astimezone(UTC) for item in post_times))
    if not normalized:
        raise ValueError("GRU experiment needs at least one post")
    if any(not math.isfinite(item.timestamp()) for item in normalized):
        raise ValueError("Post history contains an invalid timestamp")
    timestamp_collisions = sum(
        left == right for left, right in zip(normalized, normalized[1:], strict=False)
    )
    imported = [item for item in observations if item.imported_at is not None]
    delays = [
        max(0.0, (item.imported_at - item.created_at).total_seconds())
        for item in imported
        if item.imported_at is not None
    ]
    long_gaps = [
        (left, right)
        for left, right in zip(normalized, normalized[1:], strict=False)
        if right - left > timedelta(days=7)
    ]
    contiguous_start = long_gaps[-1][1] if long_gaps else normalized[0]
    first_complete_day = (contiguous_start + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    cleaned = tuple(item for item in normalized if item >= first_complete_day)
    first_hour = cleaned[0].replace(minute=0, second=0, microsecond=0)
    last_hour = cleaned[-1].replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    hourly = _hourly_counts(
        cleaned,
        first_hour,
        round((last_hour - first_hour).total_seconds() / 3600),
    )
    report = {
        "database_rows": len(normalized),
        "cleaned_rows": len(cleaned),
        "dropped_incomplete_prefix_rows": len(normalized) - len(cleaned),
        "long_internal_gaps_detected": len(long_gaps),
        "largest_internal_gap_hours": (
            max((right - left).total_seconds() for left, right in long_gaps) / 3600
            if long_gaps
            else 0.0
        ),
        "contiguous_suffix_policy": "truncate after the last gap longer than seven days",
        "exact_timestamp_collisions_preserved": timestamp_collisions,
        "duplicate_policy": "platform_post_id is the MySQL primary key; equal timestamps are valid",
        "first_complete_at": first_complete_day.isoformat(),
        "cleaned_start": cleaned[0].isoformat(),
        "cleaned_end": cleaned[-1].isoformat(),
        "hours": len(hourly),
        "empty_hour_rate": sum(value == 0 for value in hourly) / len(hourly),
        "maximum_hourly_count": max(hourly),
        "observations_with_imported_at": len(imported),
        "observations_without_imported_at": len(observations) - len(imported),
        "import_delay_seconds_median": statistics.median(delays) if delays else None,
        "import_delay_seconds_p95": float(np.percentile(delays, 95)) if delays else None,
        "training_visibility": "ideal complete history; imported_at is audited but not a feature",
    }
    return cleaned, report


def _tensor_bundle(
    samples: list[SequenceSample], scaler: FeatureScaler, device: torch.device
) -> tuple[Tensor, ...]:
    pre = torch.tensor(
        [[[math.log1p(value)] for value in item.pre_counts] for item in samples],
        dtype=torch.float32,
        device=device,
    )
    event = torch.tensor(
        [
            [
                [math.log1p(value), float(index < item.observed_hours)]
                for index, value in enumerate(item.event_counts)
            ]
            for item in samples
        ],
        dtype=torch.float32,
        device=device,
    )
    observed = torch.tensor(
        [item.observed_hours for item in samples], dtype=torch.long, device=device
    )
    static = torch.tensor(
        [scaler.transform(item.static_features) for item in samples],
        dtype=torch.float32,
        device=device,
    )
    remaining = torch.tensor(
        [item.remaining_count for item in samples], dtype=torch.float32, device=device
    )
    current = torch.tensor(
        [item.current_count for item in samples], dtype=torch.long, device=device
    )
    interval = torch.tensor(
        [item.actual_interval_index for item in samples], dtype=torch.long, device=device
    )
    weight = torch.tensor(
        [item.event_weight for item in samples], dtype=torch.float32, device=device
    )
    return pre, event, observed, static, remaining, current, interval, weight


def _mixture_log_probability(
    target: Tensor, logits: Tensor, means: Tensor, dispersion: Tensor
) -> Tensor:
    distribution = torch.distributions.NegativeBinomial(
        total_count=dispersion,
        logits=torch.log(torch.clamp(means / dispersion, min=1e-8)),
        validate_args=False,
    )
    component = distribution.log_prob(target[:, None])
    return torch.logsumexp(F.log_softmax(logits, dim=-1) + component, dim=-1)


def _mixture_pmf(
    logits: Tensor, means: Tensor, dispersion: Tensor, maximum: int
) -> Tensor:
    support = torch.arange(maximum + 1, dtype=means.dtype, device=means.device)
    distribution = torch.distributions.NegativeBinomial(
        total_count=dispersion[:, :, None],
        logits=torch.log(torch.clamp(means[:, :, None] / dispersion[:, :, None], min=1e-8)),
        validate_args=False,
    )
    log_values = distribution.log_prob(support[None, None, :])
    pmf = torch.exp(
        torch.logsumexp(F.log_softmax(logits, dim=-1)[:, :, None] + log_values, dim=1)
    )
    tail = torch.clamp(1.0 - pmf.sum(dim=1), min=0.0)
    pmf = torch.cat((pmf[:, :-1], pmf[:, -1:] + tail[:, None]), dim=1)
    return pmf / torch.clamp(pmf.sum(dim=1, keepdim=True), min=1e-12)


def _market_probabilities(
    pmf: Tensor,
    current_counts: Tensor,
    intervals: tuple[MarketInterval, ...],
) -> Tensor:
    rows = []
    for row, current in zip(pmf, current_counts.tolist(), strict=True):
        values = []
        for interval in intervals:
            first = max(0, interval.lower - current)
            last = len(row) if interval.upper is None else max(first, interval.upper - current + 1)
            values.append(row[first:last].sum())
        probabilities = torch.stack(values)
        rows.append(probabilities / torch.clamp(probabilities.sum(), min=1e-12))
    return torch.stack(rows)


def _loss_values(
    output: tuple[Tensor, Tensor, Tensor],
    *,
    remaining: Tensor,
    current: Tensor,
    actual_interval: Tensor,
    weights: Tensor,
    intervals: tuple[MarketInterval, ...],
    max_remaining: int,
    rps_weight: float,
    rps_reduction: str = "mean",
) -> tuple[Tensor, dict[str, float]]:
    logits, means, dispersion = output
    nll = -_mixture_log_probability(remaining, logits, means, dispersion)
    if rps_weight:
        pmf = _mixture_pmf(logits, means, dispersion, max_remaining)
        probabilities = _market_probabilities(pmf, current, intervals)
        predicted_cdf = probabilities.cumsum(dim=1)[:, :-1]
        thresholds = torch.arange(len(intervals) - 1, device=remaining.device)
        observed_cdf = (thresholds[None, :] >= actual_interval[:, None]).to(probabilities.dtype)
        squared_cdf_error = (predicted_cdf - observed_cdf) ** 2
        if rps_reduction == "sum":
            rps = squared_cdf_error.sum(dim=1)
        elif rps_reduction == "mean":
            rps = squared_cdf_error.mean(dim=1)
        else:
            raise ValueError(f"Unknown RPS reduction: {rps_reduction}")
    else:
        rps = torch.zeros_like(nll)
    combined = nll + rps_weight * rps
    loss = (combined * weights).sum() / torch.clamp(weights.sum(), min=1e-12)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "nll": float(((nll * weights).sum() / weights.sum()).detach().cpu()),
        "rps": float(((rps * weights).sum() / weights.sum()).detach().cpu()),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_model(
    model: nn.Module,
    *,
    train_bundle: tuple[Tensor, ...],
    validation_bundle: tuple[Tensor, ...],
    intervals: tuple[MarketInterval, ...],
    config: GruExperimentConfig,
    rps_weight: float,
    rps_reduction: str,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    _set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    dataset = TensorDataset(*[item.detach().cpu() for item in train_bundle])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    device = next(model.parameters()).device
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch[0], batch[1], batch[2], batch[3])
            loss, values = _loss_values(
                output,
                remaining=batch[4],
                current=batch[5],
                actual_interval=batch[6],
                weights=batch[7],
                intervals=intervals,
                max_remaining=config.max_remaining,
                rps_weight=rps_weight,
                rps_reduction=rps_reduction,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(values["loss"])
        model.eval()
        with torch.no_grad():
            validation_output = model(
                validation_bundle[0],
                validation_bundle[1],
                validation_bundle[2],
                validation_bundle[3],
            )
            _, validation_values = _loss_values(
                validation_output,
                remaining=validation_bundle[4],
                current=validation_bundle[5],
                actual_interval=validation_bundle[6],
                weights=validation_bundle[7],
                intervals=intervals,
                max_remaining=config.max_remaining,
                rps_weight=rps_weight,
                rps_reduction=rps_reduction,
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": statistics.fmean(train_losses),
                "validation_loss": validation_values["loss"],
                "validation_nll": validation_values["nll"],
                "validation_rps": validation_values["rps"],
            }
        )
        if validation_values["loss"] < best_loss - 1e-5:
            best_loss = validation_values["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _forecast_from_pmf(current_count: int, probabilities: list[float]) -> CountForecast:
    return CountForecast(
        current_count=current_count,
        expected_final=0,
        stddev_final=0,
        posterior_alpha=0,
        posterior_beta=0,
        remaining_exposure=0,
        integer_probabilities=tuple(probabilities),
    )


def _prediction_record(
    *,
    model_name: str,
    sample: SequenceSample,
    forecast: CountForecast,
    intervals: tuple[MarketInterval, ...],
) -> dict[str, Any]:
    probabilities = [float(forecast.probability(item.lower, item.upper)) for item in intervals]
    total = math.fsum(probabilities)
    probabilities = [value / total for value in probabilities]
    top1, selected = _best_bands(probabilities)
    selected_probability = math.fsum(probabilities[index] for index in selected)
    central = forecast.central_interval(0.8)
    key = f"gru|{model_name}|{sample.sample_key}"
    return {
        "sample_id": hashlib.sha256(key.encode()).hexdigest()[:20],
        "model": model_name,
        "fold": sample.fold,
        "split": sample.split,
        "window_start": sample.event_start.isoformat(),
        "window_end": sample.event_end.isoformat(),
        "observed_at": (sample.event_start + timedelta(hours=sample.observed_hours)).isoformat(),
        "observed_hours": sample.observed_hours,
        "current_count": sample.current_count,
        "actual_final": sample.final_count,
        "actual_interval_index": sample.actual_interval_index,
        "expected_final": forecast.expected_final,
        "stddev_final": forecast.stddev_final,
        "central_80": list(central),
        "probabilities": probabilities,
        "top1_index": top1,
        "selected_indexes": list(selected),
        "selected_probability": selected_probability,
        "selected_hit": int(sample.actual_interval_index in selected),
    }


def _predict_model(
    model: nn.Module,
    samples: list[SequenceSample],
    *,
    scaler: FeatureScaler,
    device: torch.device,
    intervals: tuple[MarketInterval, ...],
    max_remaining: int,
    model_name: str,
) -> list[dict[str, Any]]:
    pmfs = _predict_pmfs(
        model,
        samples,
        scaler=scaler,
        device=device,
        max_remaining=max_remaining,
    )
    return _prediction_records_from_pmfs(
        model_name=model_name,
        samples=samples,
        pmfs=pmfs,
        intervals=intervals,
    )


def _predict_pmfs(
    model: nn.Module,
    samples: list[SequenceSample],
    *,
    scaler: FeatureScaler,
    device: torch.device,
    max_remaining: int,
) -> np.ndarray:
    bundle = _tensor_bundle(samples, scaler, device)
    model.eval()
    with torch.no_grad():
        output = model(bundle[0], bundle[1], bundle[2], bundle[3])
        return _mixture_pmf(*output, max_remaining).cpu().numpy()


def _prediction_records_from_pmfs(
    *,
    model_name: str,
    samples: list[SequenceSample],
    pmfs: np.ndarray,
    intervals: tuple[MarketInterval, ...],
) -> list[dict[str, Any]]:
    return [
        _prediction_record(
            model_name=model_name,
            sample=sample,
            forecast=_forecast_from_pmf(sample.current_count, probabilities.tolist()),
            intervals=intervals,
        )
        for sample, probabilities in zip(samples, pmfs, strict=True)
    ]


def _temperature_scale_pmfs(pmfs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = np.log(np.clip(pmfs, 1e-30, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _interval_log_loss_from_pmfs(
    pmfs: np.ndarray,
    samples: list[SequenceSample],
    intervals: tuple[MarketInterval, ...],
) -> float:
    losses = []
    for sample, probabilities in zip(samples, pmfs, strict=True):
        forecast = _forecast_from_pmf(sample.current_count, probabilities.tolist())
        interval = intervals[sample.actual_interval_index]
        probability = float(forecast.probability(interval.lower, interval.upper))
        losses.append(-math.log(max(1e-12, probability)))
    return statistics.fmean(losses)


def _select_temperature(
    pmfs: np.ndarray,
    samples: list[SequenceSample],
    intervals: tuple[MarketInterval, ...],
) -> tuple[float, list[dict[str, float]]]:
    candidates = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
    scores = [
        {
            "temperature": temperature,
            "validation_log_loss": _interval_log_loss_from_pmfs(
                _temperature_scale_pmfs(pmfs, temperature), samples, intervals
            ),
        }
        for temperature in candidates
    ]
    best = min(scores, key=lambda item: (item["validation_log_loss"], abs(item["temperature"] - 1)))
    return best["temperature"], scores


def _fit_route_baseline(
    post_times: tuple[datetime, ...],
    *,
    history_start: datetime,
    model_end: datetime,
    event_duration: timedelta,
) -> tuple[dict[int, GradientBoostedConformalModel], AdaptiveWindowModel]:
    training = list(post_times[: bisect_left(post_times, model_end)])
    duration_hours = event_duration.total_seconds() / 3600
    gbm = {
        hours: GradientBoostedConformalModel.fit(
            training,
            start=history_start,
            end=model_end,
            event_duration=event_duration,
            observation_fraction=hours / duration_hours,
        )
        for hours in (0, 24, 42, 84)
    }
    adaptive = AdaptiveWindowModel.fit(
        training,
        start=history_start,
        end=model_end,
        event_duration=event_duration,
    )
    return gbm, adaptive


def _predict_route_baseline(
    post_times: tuple[datetime, ...],
    samples: list[SequenceSample],
    *,
    history_start: datetime,
    model_end: datetime,
    event_duration: timedelta,
    intervals: tuple[MarketInterval, ...],
) -> list[dict[str, Any]]:
    gbm, adaptive = _fit_route_baseline(
        post_times,
        history_start=history_start,
        model_end=model_end,
        event_duration=event_duration,
    )
    records = []
    opening_forecasts: dict[datetime, tuple[CountForecast, CountForecast]] = {}
    for sample in samples:
        observed_at = sample.event_start + timedelta(hours=sample.observed_hours)
        observed = list(
            post_times[
                bisect_left(post_times, model_end) : bisect_left(post_times, observed_at)
            ]
        )
        if sample.observed_hours < 24:
            if sample.event_start not in opening_forecasts:
                opening_observed = list(
                    post_times[
                        bisect_left(post_times, model_end) : bisect_left(
                            post_times, sample.event_start
                        )
                    ]
                )
                p0 = gbm[0].forecast(
                    current_count=0,
                    event_start=sample.event_start,
                    event_end=sample.event_end,
                    now=sample.event_start,
                    observed_post_times=opening_observed,
                )
                adaptive0 = adaptive.forecast(
                    current_count=0,
                    event_start=sample.event_start,
                    event_end=sample.event_end,
                    now=sample.event_start,
                    observed_post_times=opening_observed,
                )
                opening_forecasts[sample.event_start] = (p0, adaptive0)
            p0, adaptive0 = opening_forecasts[sample.event_start]
            updated = adaptive.forecast(
                current_count=sample.current_count,
                event_start=sample.event_start,
                event_end=sample.event_end,
                now=observed_at,
                observed_post_times=observed,
            )
            forecast = correct_baseline_forecast(
                p0,
                adaptive0,
                updated,
                strength=math.sqrt(min(1.0, sample.observed_hours / 24)),
            )
        elif sample.observed_hours < 126:
            checkpoint = max(hour for hour in gbm if hour <= sample.observed_hours)
            forecast = gbm[checkpoint].forecast(
                current_count=sample.current_count,
                event_start=sample.event_start,
                event_end=sample.event_end,
                now=observed_at,
                observed_post_times=observed,
            )
        else:
            forecast = adaptive.forecast(
                current_count=sample.current_count,
                event_start=sample.event_start,
                event_end=sample.event_end,
                now=observed_at,
                observed_post_times=observed,
            )
        records.append(
            _prediction_record(
                model_name="route_frozen",
                sample=sample,
                forecast=forecast,
                intervals=intervals,
            )
        )
    return records


def _rps(probabilities: list[float], actual_index: int) -> float:
    predicted = 0.0
    observed = 0.0
    total = 0.0
    for index in range(len(probabilities) - 1):
        predicted += probabilities[index]
        observed += float(index == actual_index)
        total += (predicted - observed) ** 2
    return total / max(1, len(probabilities) - 1)


def _ece(records: list[dict[str, Any]], bins: int = 5) -> float:
    if not records:
        return math.nan
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            item
            for item in records
            if lower <= item["selected_probability"] < upper
            or (index == bins - 1 and item["selected_probability"] == 1)
        ]
        if selected:
            predicted = statistics.fmean(item["selected_probability"] for item in selected)
            observed = statistics.fmean(item["selected_hit"] for item in selected)
            result += len(selected) / len(records) * abs(predicted - observed)
    return result


def _metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {"samples": 0}
    errors = [item["expected_final"] - item["actual_final"] for item in records]
    ordered = sorted(records, key=lambda item: (item["window_start"], item["observed_hours"]))
    transitions = [
        (left, right)
        for left, right in zip(ordered, ordered[1:], strict=False)
        if left["window_start"] == right["window_start"]
    ]
    return {
        "samples": len(records),
        "events": len({item["window_start"] for item in records}),
        "mae": statistics.fmean(abs(value) for value in errors),
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "log_loss": statistics.fmean(
            -math.log(max(1e-12, item["probabilities"][item["actual_interval_index"]]))
            for item in records
        ),
        "brier_score": statistics.fmean(
            math.fsum(
                (probability - float(index == item["actual_interval_index"])) ** 2
                for index, probability in enumerate(item["probabilities"])
            )
            for item in records
        ),
        "ranked_probability_score": statistics.fmean(
            _rps(item["probabilities"], item["actual_interval_index"]) for item in records
        ),
        "selected_band_hit_rate": statistics.fmean(item["selected_hit"] for item in records),
        "selected_band_mean_probability": statistics.fmean(
            item["selected_probability"] for item in records
        ),
        "selected_band_ece": _ece(records),
        "coverage_80": statistics.fmean(
            item["central_80"][0] <= item["actual_final"] <= item["central_80"][1]
            for item in records
        ),
        "central_80_mean_width": statistics.fmean(
            item["central_80"][1] - item["central_80"][0] + 1 for item in records
        ),
        "top1_accuracy": statistics.fmean(
            item["top1_index"] == item["actual_interval_index"] for item in records
        ),
        "selected_band_switches": sum(
            left["selected_indexes"] != right["selected_indexes"] for left, right in transitions
        ),
        "selected_band_switch_rate": (
            sum(
                left["selected_indexes"] != right["selected_indexes"]
                for left, right in transitions
            )
            / max(1, len(transitions))
        ),
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"pooled": _metrics(records)}
    result["by_fold"] = {
        str(fold): _metrics([item for item in records if item["fold"] == fold])
        for fold in sorted({item["fold"] for item in records})
    }
    result["by_observed_hours"] = {
        str(hours): _metrics([item for item in records if item["observed_hours"] == hours])
        for hours in sorted({item["observed_hours"] for item in records})
    }
    return result


def _paired_event_bootstrap(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    baseline_by_key = {
        (item["fold"], item["window_start"], item["observed_hours"]): item
        for item in baseline
    }
    paired = [
        (item, baseline_by_key[(item["fold"], item["window_start"], item["observed_hours"])])
        for item in candidate
    ]
    event_keys = sorted({(item[0]["fold"], item[0]["window_start"]) for item in paired})
    by_event = {
        key: [item for item in paired if (item[0]["fold"], item[0]["window_start"]) == key]
        for key in event_keys
    }
    rng = random.Random(seed)
    deltas = {
        name: []
        for name in (
            "mae",
            "log_loss",
            "brier_score",
            "ranked_probability_score",
            "selected_band_hit_rate",
        )
    }
    for _ in range(samples):
        chosen = [rng.choice(event_keys) for _ in event_keys]
        left = [pair[0] for key in chosen for pair in by_event[key]]
        right = [pair[1] for key in chosen for pair in by_event[key]]
        left_metrics = _metrics(left)
        right_metrics = _metrics(right)
        for name in deltas:
            if name == "selected_band_hit_rate":
                value = float(left_metrics[name]) - float(right_metrics[name])
            else:
                value = float(right_metrics[name]) - float(left_metrics[name])
            deltas[name].append(value)

    def percentile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in deltas.items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hourly_rows(values: tuple[datetime, ...]) -> Iterable[dict[str, Any]]:
    cursor = values[0].replace(minute=0, second=0, microsecond=0)
    end = values[-1].replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while cursor < end:
        next_hour = cursor + timedelta(hours=1)
        yield {"hour_start": cursor.isoformat(), "count": _count(values, cursor, next_hour)}
        cursor = next_hour


def _model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _config_from_manifest(payload: dict[str, Any]) -> GruExperimentConfig:
    values = dict(payload)
    values["observation_hours"] = tuple(values["observation_hours"])
    values["training_step_days"] = tuple(values["training_step_days"])
    return GruExperimentConfig(**values)


def _load_gru_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[DualEncoderGRU, FeatureScaler, dict[str, Any]]:
    artifact = torch.load(path, map_location=device, weights_only=True)
    config = _config_from_manifest(artifact["config"])
    architecture = artifact.get("architecture") or {
        "pre_history_resolution": "hourly",
        "component_initialization": "symmetric",
    }
    model = DualEncoderGRU(
        static_size=int(artifact["static_size"]),
        hidden_size=config.hidden_size,
        components=config.mixture_components,
        initial_mean=float(artifact["initial_mean"]),
        daily_pre_history=architecture["pre_history_resolution"] == "daily",
        separate_components=architecture["component_initialization"] == "separated",
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    scaler_payload = artifact["scaler"]
    scaler = FeatureScaler(
        mean=tuple(float(item) for item in scaler_payload["mean"]),
        scale=tuple(float(item) for item in scaler_payload["scale"]),
    )
    return model, scaler, artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_gru_transfer_experiment(
    post_times: list[datetime],
    observations: list[PostObservation],
    *,
    history_start: datetime,
    history_end: datetime,
    target_handle: str,
    event_slug: str,
    event_start: datetime,
    event_duration: timedelta,
    intervals: tuple[MarketInterval, ...],
    interval_source: Path,
    source_run: Path,
    output_root: Path,
    device_name: str = "auto",
    bootstrap_samples: int = 300,
) -> Path:
    """Evaluate retained GRU weights on another account without target-account fitting."""
    manifest_path = source_run / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing source GRU manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("experiment") != "gru_remaining_count_v4_deep_ensemble":
        raise ValueError("Transfer evaluation currently requires a completed V4 GRU run")
    config = _config_from_manifest(source_manifest["config"])
    config.validate(event_duration)
    source_duration = float(source_manifest["event_duration_hours"])
    if not math.isclose(source_duration, event_duration.total_seconds() / 3600):
        raise ValueError("Source and target event durations must match")

    cleaned, quality = clean_post_history(post_times, observations)
    effective_start = max(history_start, cleaned[0])
    folds = build_fold_definitions(
        history_start=effective_start,
        history_end=history_end,
        event_alignment=event_start,
        event_duration=event_duration,
        config=config,
    )
    source_folds = {int(item["fold"]): item for item in source_manifest["folds"]}
    for definition in folds:
        source = source_folds.get(definition.fold)
        if source is None:
            raise ValueError(f"Source run is missing fold {definition.fold}")
        if source["train_label_cutoff"] != definition.train_label_cutoff.isoformat():
            raise ValueError("Source and target folds do not have matching time cutoffs")
        if source["validation_starts"] != [
            value.isoformat() for value in definition.validation_starts
        ] or source["test_starts"] != [value.isoformat() for value in definition.test_starts]:
            raise ValueError("Source and target evaluation windows do not match")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.zeros(1, device=device).add_(1)

    model_artifacts = {
        (int(item["fold"]), item["model"]): source_run / item["weights"]
        for item in source_manifest["models"]
    }
    source_temperatures = {
        int(item["fold"]): float(item["selected_temperature"])
        for item in source_manifest["calibration"]
    }
    member_names = tuple(
        f"dual_gru_dense_member_{index}" for index in range(config.ensemble_members)
    )
    transferred_base = "elon_zero_shot_dual_gru_nb_nll"
    transferred_members = tuple(f"elon_zero_shot_{item}" for item in member_names)
    transferred_ensemble = "elon_zero_shot_dense_ensemble"
    transferred_calibrated = "elon_zero_shot_dense_ensemble_source_calibrated"
    reported_models = (
        transferred_base,
        *transferred_members,
        transferred_ensemble,
        transferred_calibrated,
    )

    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "data-quality.json", quality)
    _write_jsonl(run_dir / "cleaned-hourly-counts.jsonl", _hourly_rows(cleaned))
    all_predictions: list[dict[str, Any]] = []
    dataset_rows = []
    used_artifacts = []
    duration_hours = round(event_duration.total_seconds() / 3600)

    for definition in folds:
        validation_samples = _build_samples(
            cleaned,
            fold=definition.fold,
            split="validation",
            starts=definition.validation_starts,
            event_duration=event_duration,
            history_start=effective_start,
            training_cutoff=definition.train_label_cutoff,
            intervals=intervals,
            config=config,
        )
        test_samples = _build_samples(
            cleaned,
            fold=definition.fold,
            split="final_test",
            starts=definition.test_starts,
            event_duration=event_duration,
            history_start=effective_start,
            training_cutoff=definition.train_label_cutoff,
            intervals=intervals,
            config=config,
        )
        evaluation_samples = validation_samples + test_samples
        for sample in evaluation_samples:
            dataset_rows.append(
                {
                    "fold": sample.fold,
                    "split": sample.split,
                    "window_start": sample.event_start.isoformat(),
                    "window_end": sample.event_end.isoformat(),
                    "observed_hours": sample.observed_hours,
                    "current_count": sample.current_count,
                    "remaining_count": sample.remaining_count,
                    "final_count": sample.final_count,
                    "actual_interval_index": sample.actual_interval_index,
                    "event_weight": sample.event_weight,
                }
            )
        all_predictions.extend(
            _predict_route_baseline(
                cleaned,
                evaluation_samples,
                history_start=effective_start,
                model_end=definition.train_label_cutoff,
                event_duration=event_duration,
                intervals=intervals,
            )
        )

        base_path = model_artifacts.get((definition.fold, "dual_gru_nb_nll"))
        if base_path is None or not base_path.exists():
            raise ValueError(f"Source run is missing the base GRU for fold {definition.fold}")
        base_model, source_scaler, _ = _load_gru_checkpoint(base_path, device=device)
        base_pmfs = _predict_pmfs(
            base_model,
            evaluation_samples,
            scaler=source_scaler,
            device=device,
            max_remaining=config.max_remaining,
        )
        all_predictions.extend(
            _prediction_records_from_pmfs(
                model_name=transferred_base,
                samples=evaluation_samples,
                pmfs=base_pmfs,
                intervals=intervals,
            )
        )
        used_artifacts.append(
            {
                "fold": definition.fold,
                "model": "dual_gru_nb_nll",
                "path": str(base_path),
                "sha256": _sha256(base_path),
            }
        )

        member_pmfs = []
        for source_name, target_name in zip(member_names, transferred_members, strict=True):
            artifact_path = model_artifacts.get((definition.fold, source_name))
            if artifact_path is None or not artifact_path.exists():
                raise ValueError(
                    f"Source run is missing {source_name} for fold {definition.fold}"
                )
            model, scaler, _ = _load_gru_checkpoint(artifact_path, device=device)
            pmfs = _predict_pmfs(
                model,
                evaluation_samples,
                scaler=scaler,
                device=device,
                max_remaining=config.max_remaining,
            )
            member_pmfs.append(pmfs)
            all_predictions.extend(
                _prediction_records_from_pmfs(
                    model_name=target_name,
                    samples=evaluation_samples,
                    pmfs=pmfs,
                    intervals=intervals,
                )
            )
            used_artifacts.append(
                {
                    "fold": definition.fold,
                    "model": source_name,
                    "path": str(artifact_path),
                    "sha256": _sha256(artifact_path),
                }
            )
        ensemble_pmfs = np.mean(np.stack(member_pmfs), axis=0)
        temperature = source_temperatures[definition.fold]
        all_predictions.extend(
            _prediction_records_from_pmfs(
                model_name=transferred_ensemble,
                samples=evaluation_samples,
                pmfs=ensemble_pmfs,
                intervals=intervals,
            )
        )
        all_predictions.extend(
            _prediction_records_from_pmfs(
                model_name=transferred_calibrated,
                samples=evaluation_samples,
                pmfs=_temperature_scale_pmfs(ensemble_pmfs, temperature),
                intervals=intervals,
            )
        )
        LOGGER.info(
            "GRU transfer fold %s: target=%s samples=%s source_temperature=%.2f",
            definition.fold,
            target_handle,
            len(evaluation_samples),
            temperature,
        )

    _write_jsonl(run_dir / "dataset-index.jsonl", dataset_rows)
    _write_jsonl(run_dir / "predictions.jsonl", all_predictions)
    final_test = [item for item in all_predictions if item["split"] == "final_test"]
    validation = [item for item in all_predictions if item["split"] == "validation"]
    metrics = {
        model: {
            "validation": _group_metrics(
                [item for item in validation if item["model"] == model]
            ),
            "final_test": _group_metrics(
                [item for item in final_test if item["model"] == model]
            ),
        }
        for model in ("route_frozen", *reported_models)
    }
    route_test = [item for item in final_test if item["model"] == "route_frozen"]
    comparisons = {
        model: _paired_event_bootstrap(
            [item for item in final_test if item["model"] == model],
            route_test,
            samples=bootstrap_samples,
            seed=config.seed + 1900 + index,
        )
        for index, model in enumerate(reported_models)
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "comparisons-vs-target-route.json", comparisons)
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "gru_v4_cross_account_zero_shot",
            "created_at": datetime.now(UTC).isoformat(),
            "target_handle": target_handle,
            "history_start": effective_start.isoformat(),
            "history_end": history_end.isoformat(),
            "history_posts": len(cleaned),
            "event_slug": event_slug,
            "event_alignment": event_start.isoformat(),
            "event_duration_hours": duration_hours,
            "interval_source": str(interval_source),
            "source_run": str(source_run),
            "source_manifest_sha256": _sha256(manifest_path),
            "source_weights": used_artifacts,
            "source_temperatures": source_temperatures,
            "config": asdict(config),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "folds": [
                {
                    "fold": item.fold,
                    "train_label_cutoff": item.train_label_cutoff.isoformat(),
                    "validation_starts": [value.isoformat() for value in item.validation_starts],
                    "test_starts": [value.isoformat() for value in item.test_starts],
                }
                for item in folds
            ],
            "production_candidate": False,
            "notes": [
                "Neural weights, feature scalers, and calibration temperatures come only from "
                "the source account run.",
                "Target validation and final-test labels are used only for scoring; no neural "
                "parameter, temperature, ensemble weight, or model selection is fitted on them.",
                "Target-account history available before each observation is still required to "
                "construct the model input sequences and relative-rate features.",
                "The target route baseline is trained separately on target history and is a "
                "benchmark, not part of the zero-shot neural prediction.",
                "This research experiment cannot change the production monitor.",
            ],
        },
    )
    return run_dir


def _source_artifact_path(source_run: Path, *, fold: int, model: str) -> Path:
    manifest_path = source_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        source_run / item["weights"]
        for item in manifest["models"]
        if int(item["fold"]) == fold and item["model"] == model
    ]
    if len(matches) != 1 or not matches[0].exists():
        raise ValueError(f"Could not resolve {model} fold {fold} in {source_run}")
    return matches[0]


def _source_fold_cutoff(source_run: Path, fold: int) -> str:
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    matches = [
        item["train_label_cutoff"]
        for item in manifest["folds"]
        if int(item["fold"]) == fold
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve fold {fold} cutoff in {source_run}")
    return str(matches[0])


def _live_result(
    record: dict[str, Any], intervals: tuple[MarketInterval, ...]
) -> dict[str, Any]:
    probabilities = record["probabilities"]
    return {
        "expected_final": record["expected_final"],
        "stddev_final": record["stddev_final"],
        "central_80": record["central_80"],
        "three_band": [intervals[index].label for index in record["selected_indexes"]],
        "three_band_probability": record["selected_probability"],
        "highest_intervals": [
            {
                "label": intervals[index].label,
                "probability": probabilities[index],
            }
            for index in sorted(
                range(len(intervals)), key=lambda index: probabilities[index], reverse=True
            )[:6]
        ],
        "interval_probabilities": [
            {"label": interval.label, "probability": probabilities[index]}
            for index, interval in enumerate(intervals)
        ],
    }


def run_gru_live_comparison(
    post_times: list[datetime],
    *,
    history_start: datetime,
    event_slug: str,
    event_start: datetime,
    event_end: datetime,
    as_of: datetime,
    official_current_count: int,
    intervals: tuple[MarketInterval, ...],
    original_run: Path,
    v3_run: Path,
    v4_run: Path,
    output_path: Path,
    fold: int = 2,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Compare frozen out-of-sample GRUs with a current target-specific route forecast."""
    if event_end <= event_start:
        raise ValueError("Event end must be after event start")
    if official_current_count < 0:
        raise ValueError("Official current count cannot be negative")
    duration_hours = round((event_end - event_start).total_seconds() / 3600)
    if event_end - event_start != timedelta(hours=duration_hours):
        raise ValueError("Live GRU comparison requires a whole-hour event duration")
    requested_as_of = min(max(as_of, event_start), event_end)
    observed_hours = min(
        duration_hours,
        math.floor((requested_as_of - event_start).total_seconds() / 3600),
    )
    observed_at = event_start + timedelta(hours=observed_hours)
    values = tuple(sorted(item.astimezone(UTC) for item in post_times if item < observed_at))
    if not values or values[0] > event_start - timedelta(days=28):
        raise ValueError("At least 28 days of pre-event post history are required")
    history_posts = _count(values, history_start, event_start)
    history_hours = max(1.0, (event_start - history_start).total_seconds() / 3600)
    hourly_rate = history_posts / history_hours
    event_counts = _hourly_counts(values, event_start, duration_hours)
    visible_count = sum(event_counts[:observed_hours])
    sample = SequenceSample(
        fold=fold,
        split="live",
        event_start=event_start,
        event_end=event_end,
        observed_hours=observed_hours,
        pre_counts=_hourly_counts(
            values, event_start - timedelta(days=28), 28 * 24
        ),
        event_counts=event_counts,
        static_features=_static_features(
            values,
            event_start=event_start,
            observed_at=observed_at,
            event_end=event_end,
            current_count=official_current_count,
            hourly_rate=hourly_rate,
        ),
        current_count=official_current_count,
        final_count=official_current_count,
        remaining_count=0,
        actual_interval_index=actual_interval_index(intervals, official_current_count),
        event_weight=1.0,
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.zeros(1, device=device).add_(1)

    route_record = _predict_route_baseline(
        values,
        [sample],
        history_start=history_start,
        model_end=event_start,
        event_duration=event_end - event_start,
        intervals=intervals,
    )[0]
    if observed_hours < 24:
        route_component = "GBM-0h corrected by Adaptive"
    elif observed_hours < 126:
        route_component = f"GBM-{max(hour for hour in (0, 24, 42, 84) if hour <= observed_hours)}h"
    else:
        route_component = "Adaptive-126h"

    result_models: dict[str, Any] = {
        "route_baseline": {
            **_live_result(route_record, intervals),
            "route_component": route_component,
            "weights": "fitted from account history ending at event start",
        }
    }
    source_artifacts = []
    for output_name, source_run in (
        ("original_gru_nll", original_run),
        ("v3_dense_hourly_gru", v3_run),
    ):
        artifact_path = _source_artifact_path(
            source_run, fold=fold, model="dual_gru_nb_nll"
        )
        model, scaler, artifact = _load_gru_checkpoint(artifact_path, device=device)
        record = _predict_model(
            model,
            [sample],
            scaler=scaler,
            device=device,
            intervals=intervals,
            max_remaining=int(artifact["config"]["max_remaining"]),
            model_name=output_name,
        )[0]
        result_models[output_name] = _live_result(record, intervals)
        source_artifacts.append(
            {
                "model": output_name,
                "path": str(artifact_path),
                "sha256": _sha256(artifact_path),
                "training_label_cutoff": _source_fold_cutoff(source_run, fold),
            }
        )

    v4_manifest_path = v4_run / "manifest.json"
    v4_manifest = json.loads(v4_manifest_path.read_text(encoding="utf-8"))
    v4_config = _config_from_manifest(v4_manifest["config"])
    v4_pmfs = []
    for member_index in range(v4_config.ensemble_members):
        model_name = f"dual_gru_dense_member_{member_index}"
        artifact_path = _source_artifact_path(v4_run, fold=fold, model=model_name)
        model, scaler, _ = _load_gru_checkpoint(artifact_path, device=device)
        v4_pmfs.append(
            _predict_pmfs(
                model,
                [sample],
                scaler=scaler,
                device=device,
                max_remaining=v4_config.max_remaining,
            )
        )
        source_artifacts.append(
            {
                "model": f"v4_{model_name}",
                "path": str(artifact_path),
                "sha256": _sha256(artifact_path),
                "training_label_cutoff": _source_fold_cutoff(v4_run, fold),
            }
        )
    v4_temperature = next(
        float(item["selected_temperature"])
        for item in v4_manifest["calibration"]
        if int(item["fold"]) == fold
    )
    v4_pmf = _temperature_scale_pmfs(
        np.mean(np.stack(v4_pmfs), axis=0), v4_temperature
    )
    v4_record = _prediction_records_from_pmfs(
        model_name="v4_calibrated_ensemble",
        samples=[sample],
        pmfs=v4_pmf,
        intervals=intervals,
    )[0]
    result_models["v4_calibrated_ensemble"] = {
        **_live_result(v4_record, intervals),
        "source_temperature": v4_temperature,
    }

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "event_slug": event_slug,
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "count_snapshot_at": as_of.isoformat(),
        "model_observed_at": observed_at.isoformat(),
        "observed_hours": observed_hours,
        "remaining_hours": duration_hours - observed_hours,
        "official_current_count": official_current_count,
        "visible_database_count_at_model_time": visible_count,
        "official_database_count_delta": official_current_count - visible_count,
        "history_start": history_start.isoformat(),
        "gru_checkpoint_fold": fold,
        "device": str(device),
        "models": result_models,
        "source_artifacts": source_artifacts,
        "notes": [
            "The route is fitted using this account's history available before event start.",
            "GRU checkpoints are the latest retained strict out-of-sample fold; they are not "
            "retrained or selected using either live event.",
            "The model observation time is floored to a complete UTC hour; the official count "
            "was fetched at count_snapshot_at.",
            "These are research forecasts and are not trading instructions.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    return payload


def run_gru_experiment(
    post_times: list[datetime],
    observations: list[PostObservation],
    *,
    history_start: datetime,
    history_end: datetime,
    event_slug: str,
    event_start: datetime,
    event_duration: timedelta,
    intervals: tuple[MarketInterval, ...],
    interval_source: Path,
    output_root: Path,
    config: GruExperimentConfig,
    device_name: str = "auto",
    bootstrap_samples: int = 300,
) -> Path:
    config.validate(event_duration)
    cleaned, quality = clean_post_history(post_times, observations)
    effective_start = max(history_start, cleaned[0])
    folds = build_fold_definitions(
        history_start=effective_start,
        history_end=history_end,
        event_alignment=event_start,
        event_duration=event_duration,
        config=config,
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.zeros(1, device=device).add_(1)
    LOGGER.info(
        "GRU data ready: posts=%s history=%s to %s device=%s",
        len(cleaned),
        effective_start.isoformat(),
        history_end.isoformat(),
        device,
    )

    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "data-quality.json", quality)
    _write_jsonl(run_dir / "cleaned-hourly-counts.jsonl", _hourly_rows(cleaned))

    all_predictions: list[dict[str, Any]] = []
    training_summaries = []
    calibration_summaries = []
    dataset_rows = []
    model_names = ("dual_gru_nb_nll",)
    ensemble_member_names = tuple(
        f"dual_gru_dense_member_{index}" for index in range(config.ensemble_members)
    )
    ensemble_model_names = (
        "dual_gru_dense_ensemble",
        "dual_gru_dense_ensemble_calibrated",
    )
    reported_model_names = (*model_names, *ensemble_member_names, *ensemble_model_names)
    duration_hours = round(event_duration.total_seconds() / 3600)
    training_observation_hours = tuple(
        range(0, duration_hours, config.training_observation_step_hours)
    )
    for definition in folds:
        train_starts = _training_starts(
            history_start=effective_start,
            event_alignment=event_start,
            event_duration=event_duration,
            label_cutoff=definition.train_label_cutoff,
            config=config,
        )
        train_samples = _build_samples(
            cleaned,
            fold=definition.fold,
            split="train",
            starts=train_starts,
            event_duration=event_duration,
            history_start=effective_start,
            training_cutoff=definition.train_label_cutoff,
            intervals=intervals,
            config=config,
            observation_hours=training_observation_hours,
        )
        validation_samples = _build_samples(
            cleaned,
            fold=definition.fold,
            split="validation",
            starts=definition.validation_starts,
            event_duration=event_duration,
            history_start=effective_start,
            training_cutoff=definition.train_label_cutoff,
            intervals=intervals,
            config=config,
        )
        test_samples = _build_samples(
            cleaned,
            fold=definition.fold,
            split="final_test",
            starts=definition.test_starts,
            event_duration=event_duration,
            history_start=effective_start,
            training_cutoff=definition.train_label_cutoff,
            intervals=intervals,
            config=config,
        )
        for sample in (*train_samples, *validation_samples, *test_samples):
            dataset_rows.append(
                {
                    "fold": sample.fold,
                    "split": sample.split,
                    "window_start": sample.event_start.isoformat(),
                    "window_end": sample.event_end.isoformat(),
                    "observed_hours": sample.observed_hours,
                    "current_count": sample.current_count,
                    "remaining_count": sample.remaining_count,
                    "final_count": sample.final_count,
                    "actual_interval_index": sample.actual_interval_index,
                    "event_weight": sample.event_weight,
                }
            )
        scaler = FeatureScaler.fit(train_samples)
        LOGGER.info(
            "GRU fold %s: train_events=%s validation_events=%s test_events=%s cutoff=%s",
            definition.fold,
            len(train_starts),
            len(definition.validation_starts),
            len(definition.test_starts),
            definition.train_label_cutoff.isoformat(),
        )
        train_bundle = _tensor_bundle(train_samples, scaler, device)
        validation_bundle = _tensor_bundle(validation_samples, scaler, device)
        initial_mean = statistics.fmean(item.remaining_count for item in train_samples)
        static_size = len(train_samples[0].static_features)
        fold_dir = run_dir / f"fold-{definition.fold}"
        fold_dir.mkdir()

        route_records = _predict_route_baseline(
            cleaned,
            validation_samples + test_samples,
            history_start=effective_start,
            model_end=definition.train_label_cutoff,
            event_duration=event_duration,
            intervals=intervals,
        )
        all_predictions.extend(route_records)

        for model_index, model_name in enumerate(model_names):
            seed = config.seed + definition.fold * 100 + model_index
            _set_seed(seed)
            if model_name == "mlp_nb_nll":
                model: nn.Module = AggregateMLP(
                    static_size, config.mixture_components, initial_mean
                )
                rps_weight = 0.0
                rps_reduction = "mean"
                architecture = {
                    "pre_history_resolution": "aggregate_features_only",
                    "component_initialization": "symmetric",
                    "rps_reduction": rps_reduction,
                }
            else:
                daily_pre_history = "daily" in model_name
                separate_components = "asym" in model_name or daily_pre_history
                model = DualEncoderGRU(
                    static_size=static_size,
                    hidden_size=config.hidden_size,
                    components=config.mixture_components,
                    initial_mean=initial_mean,
                    daily_pre_history=daily_pre_history,
                    separate_components=separate_components,
                )
                rps_weight = config.rps_weight if model_name.endswith("rps") else 0.0
                rps_reduction = "sum" if daily_pre_history and rps_weight else "mean"
                architecture = {
                    "pre_history_resolution": "daily" if daily_pre_history else "hourly",
                    "component_initialization": (
                        "separated" if separate_components else "symmetric"
                    ),
                    "rps_reduction": rps_reduction,
                }
            model.to(device)
            model, history, best_epoch = _train_model(
                model,
                train_bundle=train_bundle,
                validation_bundle=validation_bundle,
                intervals=intervals,
                config=config,
                rps_weight=rps_weight,
                rps_reduction=rps_reduction,
                seed=seed,
            )
            predictions = _predict_model(
                model,
                validation_samples + test_samples,
                scaler=scaler,
                device=device,
                intervals=intervals,
                max_remaining=config.max_remaining,
                model_name=model_name,
            )
            all_predictions.extend(predictions)
            artifact_path = fold_dir / f"{model_name}.pt"
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "scaler": asdict(scaler),
                    "config": asdict(config),
                    "static_size": static_size,
                    "initial_mean": initial_mean,
                    "architecture": architecture,
                    "best_epoch": best_epoch,
                    "seed": seed,
                },
                artifact_path,
            )
            _write_json(fold_dir / f"{model_name}-training.json", history)
            training_summaries.append(
                {
                    "fold": definition.fold,
                    "model": model_name,
                    "seed": seed,
                    "parameters": _model_parameter_count(model),
                    "architecture": architecture,
                    "best_epoch": best_epoch,
                    "epochs_ran": len(history),
                    "weights": str(artifact_path.relative_to(run_dir)),
                }
            )
            LOGGER.info(
                "GRU fold %s model=%s parameters=%s best_epoch=%s epochs=%s",
                definition.fold,
                model_name,
                _model_parameter_count(model),
                best_epoch,
                len(history),
            )

        evaluation_samples = validation_samples + test_samples
        member_pmfs = []
        for member_index, member_name in enumerate(ensemble_member_names):
            seed = config.seed + 10_000 + definition.fold * 100 + member_index
            _set_seed(seed)
            model = DualEncoderGRU(
                static_size=static_size,
                hidden_size=config.hidden_size,
                components=config.mixture_components,
                initial_mean=initial_mean,
            ).to(device)
            model, history, best_epoch = _train_model(
                model,
                train_bundle=train_bundle,
                validation_bundle=validation_bundle,
                intervals=intervals,
                config=config,
                rps_weight=0.0,
                rps_reduction="mean",
                seed=seed,
            )
            pmfs = _predict_pmfs(
                model,
                evaluation_samples,
                scaler=scaler,
                device=device,
                max_remaining=config.max_remaining,
            )
            member_pmfs.append(pmfs)
            all_predictions.extend(
                _prediction_records_from_pmfs(
                    model_name=member_name,
                    samples=evaluation_samples,
                    pmfs=pmfs,
                    intervals=intervals,
                )
            )
            architecture = {
                "pre_history_resolution": "hourly",
                "component_initialization": "symmetric",
                "rps_reduction": "mean",
                "ensemble_member": member_index,
            }
            artifact_path = fold_dir / f"{member_name}.pt"
            torch.save(
                {
                    "model_name": member_name,
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "scaler": asdict(scaler),
                    "config": asdict(config),
                    "static_size": static_size,
                    "initial_mean": initial_mean,
                    "architecture": architecture,
                    "best_epoch": best_epoch,
                    "seed": seed,
                },
                artifact_path,
            )
            _write_json(fold_dir / f"{member_name}-training.json", history)
            training_summaries.append(
                {
                    "fold": definition.fold,
                    "model": member_name,
                    "seed": seed,
                    "parameters": _model_parameter_count(model),
                    "architecture": architecture,
                    "best_epoch": best_epoch,
                    "epochs_ran": len(history),
                    "weights": str(artifact_path.relative_to(run_dir)),
                }
            )
            LOGGER.info(
                "GRU fold %s ensemble_member=%s best_epoch=%s epochs=%s",
                definition.fold,
                member_index,
                best_epoch,
                len(history),
            )

        ensemble_pmfs = np.mean(np.stack(member_pmfs), axis=0)
        validation_count = len(validation_samples)
        temperature, temperature_scores = _select_temperature(
            ensemble_pmfs[:validation_count], validation_samples, intervals
        )
        calibrated_pmfs = _temperature_scale_pmfs(ensemble_pmfs, temperature)
        all_predictions.extend(
            _prediction_records_from_pmfs(
                model_name=ensemble_model_names[0],
                samples=evaluation_samples,
                pmfs=ensemble_pmfs,
                intervals=intervals,
            )
        )
        all_predictions.extend(
            _prediction_records_from_pmfs(
                model_name=ensemble_model_names[1],
                samples=evaluation_samples,
                pmfs=calibrated_pmfs,
                intervals=intervals,
            )
        )
        calibration = {
            "fold": definition.fold,
            "model": ensemble_model_names[1],
            "selected_temperature": temperature,
            "validation_scores": temperature_scores,
            "members": list(ensemble_member_names),
        }
        calibration_summaries.append(calibration)
        _write_json(fold_dir / "dual_gru_dense_ensemble-calibration.json", calibration)
        LOGGER.info(
            "GRU fold %s ensemble temperature=%.2f",
            definition.fold,
            temperature,
        )

    _write_jsonl(run_dir / "dataset-index.jsonl", dataset_rows)
    _write_jsonl(run_dir / "predictions.jsonl", all_predictions)
    final_test = [item for item in all_predictions if item["split"] == "final_test"]
    validation = [item for item in all_predictions if item["split"] == "validation"]
    metrics = {
        model: {
            "validation": _group_metrics([item for item in validation if item["model"] == model]),
            "final_test": _group_metrics([item for item in final_test if item["model"] == model]),
        }
        for model in ("route_frozen", *reported_model_names)
    }
    route_test = [item for item in final_test if item["model"] == "route_frozen"]
    comparisons = {
        model: _paired_event_bootstrap(
            [item for item in final_test if item["model"] == model],
            route_test,
            samples=bootstrap_samples,
            seed=config.seed + 900 + index,
        )
        for index, model in enumerate(reported_model_names)
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "comparisons-vs-route.json", comparisons)
    manifest = {
        "experiment": "gru_remaining_count_v4_deep_ensemble",
        "created_at": datetime.now(UTC).isoformat(),
        "history_start": effective_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_posts": len(cleaned),
        "event_slug": event_slug,
        "event_alignment": event_start.isoformat(),
        "event_duration_hours": event_duration.total_seconds() / 3600,
        "interval_source": str(interval_source),
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "folds": [
            {
                "fold": item.fold,
                "train_label_cutoff": item.train_label_cutoff.isoformat(),
                "validation_starts": [value.isoformat() for value in item.validation_starts],
                "test_starts": [value.isoformat() for value in item.test_starts],
            }
            for item in folds
        ],
        "models": training_summaries,
        "calibration": calibration_summaries,
        "production_candidate": False,
        "notes": [
            "The experiment uses complete historical timestamps, not reconstructed "
            "point-in-time API visibility.",
            "Training windows advance by 4/3 days; validation and final-test events "
            "do not overlap.",
            "Every split boundary uses a seven-day embargo and model selection never "
            "reads final_test.",
            "All neural models remain research-only and are not wired into the live monitor.",
            "Training events use dense observation checkpoints while validation and final-test "
            "events retain the configured evaluation checkpoints; every event still has total "
            "sample weight one.",
            "V4 averages five independently initialized dense hourly GRUs and selects one "
            "PMF temperature per fold using validation interval log loss only.",
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir

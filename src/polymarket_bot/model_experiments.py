from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from polymarket_bot.forecast import (
    AdaptiveWindowModel,
    CountForecast,
    EmpiricalWindowModel,
    SeasonalGammaPoisson,
)
from polymarket_bot.market_intervals import MarketInterval, actual_interval_index

LOGGER = logging.getLogger(__name__)

ROUTED_STRATEGY_NAME = "gbm_24_84_adaptive_126"
ROUTED_STRATEGY_SOURCES = {
    "0.142857": "gbm_conformal",
    "0.250000": "gbm_conformal",
    "0.500000": "gbm_conformal",
    "0.750000": "adaptive_reconciled",
}


def _count(post_times: tuple[datetime, ...], start: datetime, end: datetime) -> int:
    return bisect_left(post_times, end) - bisect_left(post_times, start)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class RecentRateModel:
    baseline: SeasonalGammaPoisson
    prior_hours: float = 6.0
    lookback_hours: float = 24.0

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
    ) -> RecentRateModel:
        del event_duration
        baseline = SeasonalGammaPoisson.fit(
            post_times,
            start=start,
            end=end,
            prior_strength_hours=168,
        )
        return cls(baseline)

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
        observed_post_times: list[datetime],
    ) -> CountForecast:
        observed_at = min(max(now, event_start), event_end)
        elapsed_hours = max(0.0, (observed_at - event_start).total_seconds() / 3600)
        remaining_hours = max(0.0, (event_end - observed_at).total_seconds() / 3600)
        if remaining_hours <= 0:
            return CountForecast(current_count, float(current_count), 0, 0, 0, 0)
        window_hours = min(self.lookback_hours, elapsed_hours)
        window_start = observed_at - timedelta(hours=window_hours)
        recent_count = sum(window_start <= item < observed_at for item in observed_post_times)
        alpha = recent_count + self.prior_hours * self.baseline.hourly_rate
        beta = window_hours + self.prior_hours
        mean_remaining = alpha * remaining_hours / beta
        variance_remaining = mean_remaining + alpha * remaining_hours**2 / beta**2
        return CountForecast(
            current_count=current_count,
            expected_final=current_count + mean_remaining,
            stddev_final=math.sqrt(variance_remaining),
            posterior_alpha=alpha,
            posterior_beta=beta,
            remaining_exposure=remaining_hours,
            base_expected_final=current_count
            + self.baseline.hourly_rate * self.baseline.effective_hours(observed_at, event_end),
            recent_counts=((24, recent_count),),
            effective_sample_size=beta,
        )


@dataclass(frozen=True)
class NegativeBinomialRateModel:
    baseline: SeasonalGammaPoisson
    dispersion_per_event: float
    post_times: tuple[datetime, ...]
    event_duration: timedelta

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
    ) -> NegativeBinomialRateModel:
        ordered = tuple(sorted(item for item in post_times if start <= item < end))
        baseline = SeasonalGammaPoisson.fit(
            list(ordered),
            start=start,
            end=end,
            prior_strength_hours=168,
        )
        totals = []
        cursor = start
        while cursor + event_duration <= end:
            totals.append(_count(ordered, cursor, cursor + event_duration))
            cursor += timedelta(days=1)
        mean = statistics.fmean(totals)
        variance = statistics.variance(totals) if len(totals) > 1 else mean
        dispersion = mean * mean / max(1e-6, variance - mean) if variance > mean else 10_000.0
        return cls(baseline, min(10_000.0, max(0.5, dispersion)), ordered, event_duration)

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
        observed_post_times: list[datetime],
    ) -> CountForecast:
        observed_at = min(max(now, event_start), event_end)
        remaining = max(0.0, (event_end - observed_at).total_seconds() / 3600)
        if remaining <= 0:
            return CountForecast(current_count, float(current_count), 0, 0, 0, 0)
        known = tuple(sorted((*self.post_times, *observed_post_times)))
        ratios = []
        for hours, weight in ((24, 0.5), (72, 0.3), (168, 0.2)):
            window_start = max(event_start, observed_at - timedelta(hours=hours))
            actual = _count(known, window_start, observed_at)
            expected = self.baseline.hourly_rate * self.baseline.effective_hours(
                window_start, observed_at
            )
            ratios.append((hours, weight, (actual + 2.0) / (expected + 2.0), actual))
        multiplier = math.exp(
            math.fsum(
                weight * math.log(min(2.5, max(0.4, ratio))) for _, weight, ratio, _ in ratios
            )
            / math.fsum(weight for _, weight, _, _ in ratios)
        )
        base_remaining = self.baseline.hourly_rate * self.baseline.effective_hours(
            observed_at, event_end
        )
        mean_remaining = max(0.01, base_remaining * multiplier)
        duration_hours = self.event_duration.total_seconds() / 3600
        shape = max(0.25, self.dispersion_per_event * remaining / duration_hours)
        beta = shape / mean_remaining
        variance = mean_remaining + mean_remaining * mean_remaining / shape
        recent_24 = next((count for hours, _, _, count in ratios if hours == 24), 0)
        return CountForecast(
            current_count=current_count,
            expected_final=current_count + mean_remaining,
            stddev_final=math.sqrt(variance),
            posterior_alpha=shape,
            posterior_beta=beta,
            remaining_exposure=1.0,
            base_expected_final=current_count + base_remaining,
            realtime_multiplier=multiplier,
            regime=_regime(multiplier),
            recent_counts=((24, recent_24),),
            effective_sample_size=shape,
        )


def _regime(multiplier: float) -> str:
    if multiplier >= 1.65:
        return "extreme_high"
    if multiplier >= 1.20:
        return "high"
    if multiplier <= 0.55:
        return "extreme_low"
    if multiplier <= 0.82:
        return "low"
    return "normal"


@dataclass(frozen=True)
class RegressionStump:
    feature: int
    threshold: float
    left: float
    right: float


@dataclass(frozen=True)
class GradientBoostedRegressor:
    intercept: float
    learning_rate: float
    stumps: tuple[RegressionStump, ...]

    @classmethod
    def fit(
        cls,
        features: list[tuple[float, ...]],
        targets: list[float],
        *,
        estimators: int = 60,
        learning_rate: float = 0.05,
    ) -> GradientBoostedRegressor:
        intercept = statistics.fmean(targets)
        predictions = [intercept] * len(targets)
        stumps: list[RegressionStump] = []
        for _ in range(estimators):
            residuals = [
                target - prediction for target, prediction in zip(targets, predictions, strict=True)
            ]
            best: tuple[float, RegressionStump] | None = None
            for feature in range(len(features[0])):
                values = sorted({row[feature] for row in features})
                if len(values) < 2:
                    continue
                indexes = sorted(
                    {
                        max(1, min(len(values) - 1, round(len(values) * q / 10)))
                        for q in range(1, 10)
                    }
                )
                for index in indexes:
                    threshold = (values[index - 1] + values[index]) / 2
                    left_values = [
                        residual
                        for row, residual in zip(features, residuals, strict=True)
                        if row[feature] <= threshold
                    ]
                    right_values = [
                        residual
                        for row, residual in zip(features, residuals, strict=True)
                        if row[feature] > threshold
                    ]
                    if not left_values or not right_values:
                        continue
                    left = statistics.fmean(left_values)
                    right = statistics.fmean(right_values)
                    error = math.fsum(
                        (residual - (left if row[feature] <= threshold else right)) ** 2
                        for row, residual in zip(features, residuals, strict=True)
                    )
                    stump = RegressionStump(feature, threshold, left, right)
                    if best is None or error < best[0]:
                        best = (error, stump)
            if best is None:
                break
            stump = best[1]
            stumps.append(stump)
            for index, row in enumerate(features):
                leaf = stump.left if row[stump.feature] <= stump.threshold else stump.right
                predictions[index] += learning_rate * leaf
        return cls(intercept, learning_rate, tuple(stumps))

    def predict(self, features: tuple[float, ...]) -> float:
        value = self.intercept
        for stump in self.stumps:
            leaf = stump.left if features[stump.feature] <= stump.threshold else stump.right
            value += self.learning_rate * leaf
        return value


@dataclass(frozen=True)
class GradientBoostedConformalModel:
    post_times: tuple[datetime, ...]
    history_start: datetime
    history_end: datetime
    event_duration: timedelta
    observation_fraction: float
    baseline: SeasonalGammaPoisson
    regressor: GradientBoostedRegressor
    residuals: tuple[float, ...]
    feature_names: tuple[str, ...] = (
        "log_current_count",
        "log_event_rate_ratio",
        "log_6h_rate_ratio",
        "log_24h_rate_ratio",
        "log_72h_rate_ratio",
        "start_weekday_sin",
        "start_weekday_cos",
    )

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
        observation_fraction: float,
    ) -> GradientBoostedConformalModel:
        ordered = tuple(sorted(item for item in post_times if start <= item < end))
        baseline = SeasonalGammaPoisson.fit(
            list(ordered), start=start, end=end, prior_strength_hours=168
        )
        temporary = cls(
            ordered,
            start,
            end,
            event_duration,
            observation_fraction,
            baseline,
            GradientBoostedRegressor(0, 0.05, ()),
            (),
        )
        rows: list[tuple[datetime, tuple[float, ...], float]] = []
        cursor = start + timedelta(days=3)
        offset = (end - cursor) % timedelta(days=1)
        cursor += offset
        observed_duration = event_duration * observation_fraction
        while cursor + event_duration <= end:
            observed_at = cursor + observed_duration
            current = _count(ordered, cursor, observed_at)
            final = _count(ordered, cursor, cursor + event_duration)
            rows.append((cursor, temporary._features(cursor, observed_at, current, ordered), final))
            cursor += timedelta(days=1)
        if len(rows) < 60:
            raise ValueError("At least 60 historical windows are required for GBM + conformal")
        split = max(40, int(len(rows) * 0.75))
        training = rows[:split]
        calibration = rows[split:]
        regressor = GradientBoostedRegressor.fit(
            [row[1] for row in training], [row[2] for row in training]
        )
        residuals = tuple(row[2] - regressor.predict(row[1]) for row in calibration)
        return cls(
            ordered,
            start,
            end,
            event_duration,
            observation_fraction,
            baseline,
            regressor,
            residuals,
        )

    def _features(
        self,
        event_start: datetime,
        observed_at: datetime,
        current_count: int,
        known: tuple[datetime, ...],
    ) -> tuple[float, ...]:
        elapsed_hours = max(1.0, (observed_at - event_start).total_seconds() / 3600)
        expected_event = self.baseline.hourly_rate * self.baseline.effective_hours(
            event_start, observed_at
        )
        values = [math.log1p(current_count), math.log((current_count + 2) / (expected_event + 2))]
        for hours in (6, 24, 72):
            window_start = observed_at - timedelta(hours=hours)
            actual = _count(known, window_start, observed_at)
            expected = self.baseline.hourly_rate * self.baseline.effective_hours(
                window_start, observed_at
            )
            values.append(math.log((actual + 2) / (expected + 2)))
        angle = 2 * math.pi * event_start.weekday() / 7
        values.extend((math.sin(angle), math.cos(angle)))
        del elapsed_hours
        return tuple(values)

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
        observed_post_times: list[datetime],
    ) -> CountForecast:
        observed_at = min(max(now, event_start), event_end)
        known = tuple(
            sorted(
                (
                    *self.post_times,
                    *(item for item in observed_post_times if item >= self.history_end),
                )
            )
        )
        features = self._features(event_start, observed_at, current_count, known)
        recent_counts = tuple(
            (hours, _count(known, observed_at - timedelta(hours=hours), observed_at))
            for hours in (6, 24, 72)
        )
        point = max(float(current_count), self.regressor.predict(features))
        finals = tuple(max(current_count, round(point + residual)) for residual in self.residuals)
        expected = statistics.fmean(finals) if finals else point
        variance = statistics.variance(finals) if len(finals) > 1 else 4.0
        bandwidth = max(2.0, 1.06 * math.sqrt(variance) * max(1, len(finals)) ** -0.2)
        ratio = math.exp(features[1])
        return CountForecast(
            current_count=current_count,
            expected_final=expected,
            stddev_final=math.sqrt(variance + bandwidth**2),
            posterior_alpha=0,
            posterior_beta=0,
            remaining_exposure=max(0.0, (event_end - observed_at).total_seconds() / 3600),
            empirical_finals=finals,
            kernel_bandwidth=bandwidth,
            base_expected_final=point,
            realtime_multiplier=ratio,
            regime=_regime(ratio),
            recent_counts=recent_counts,
            effective_sample_size=float(len(finals)),
        )


def _model_artifact(model: Any, *, model_name: str, fitted_at: datetime) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "type": "fitted_model",
        "model": model_name,
        "fitted_at": fitted_at.isoformat(),
    }
    baseline = getattr(model, "baseline", None)
    if baseline is not None:
        artifact["baseline"] = {
            "hourly_rate": baseline.hourly_rate,
            "seasonality": list(baseline.seasonality),
            "history_start": baseline.history_start.isoformat(),
            "history_end": baseline.history_end.isoformat(),
            "history_posts": baseline.history_posts,
        }
    if isinstance(model, AdaptiveWindowModel):
        artifact["feature_names"] = ["event"] + [f"{hours}h" for hours in model.context_hours]
        artifact["feature_weights"] = [3.0, 0.2, 0.4, 0.7, 1.0, 1.3, 1.2, 0.8, 0.5, 0.3]
    if isinstance(model, RecentRateModel):
        artifact.update({"prior_hours": model.prior_hours, "lookback_hours": model.lookback_hours})
    if isinstance(model, NegativeBinomialRateModel):
        artifact["dispersion_per_event"] = model.dispersion_per_event
    if isinstance(model, GradientBoostedConformalModel):
        artifact["feature_names"] = list(model.feature_names)
        artifact["gbm"] = {
            "intercept": model.regressor.intercept,
            "learning_rate": model.regressor.learning_rate,
            "stumps": [stump.__dict__ for stump in model.regressor.stumps],
        }
        artifact["conformal_residuals"] = list(model.residuals)
    return artifact


def _best_bands(probabilities: list[float], size: int = 3) -> tuple[int, tuple[int, ...]]:
    top1 = max(range(len(probabilities)), key=probabilities.__getitem__)
    start = max(
        range(len(probabilities) - size + 1),
        key=lambda index: math.fsum(probabilities[index : index + size]),
    )
    return top1, tuple(range(start, start + size))


def _prediction_record(
    *,
    model_name: str,
    fold: int,
    observation_fraction: float,
    window_start: datetime,
    window_end: datetime,
    observed_at: datetime,
    current_count: int,
    actual_final: int,
    forecast: CountForecast,
    intervals: tuple[MarketInterval, ...],
) -> dict[str, Any]:
    probabilities = [float(forecast.probability(item.lower, item.upper)) for item in intervals]
    total = math.fsum(probabilities)
    probabilities = [value / total for value in probabilities]
    actual_index = actual_interval_index(intervals, actual_final)
    top1, selected = _best_bands(probabilities)
    selected_probability = math.fsum(probabilities[index] for index in selected)
    lower, upper = forecast.central_interval(0.8)
    sample_key = f"{model_name}|{fold}|{observation_fraction:.6f}|{window_start.isoformat()}"
    return {
        "sample_id": hashlib.sha256(sample_key.encode()).hexdigest()[:20],
        "model": model_name,
        "fold": fold,
        "observation_fraction": observation_fraction,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "observed_at": observed_at.isoformat(),
        "current_count": current_count,
        "actual_final": actual_final,
        "actual_interval_index": actual_index,
        "expected_final": forecast.expected_final,
        "stddev_final": forecast.stddev_final,
        "central_80": [lower, upper],
        "regime": forecast.regime,
        "effective_sample_size": forecast.effective_sample_size,
        "source_count_delta": forecast.source_count_delta,
        "probabilities": probabilities,
        "top1_index": top1,
        "selected_indexes": list(selected),
        "selected_probability": selected_probability,
        "decision_probability": selected_probability,
        "selected_lower_probability": None,
        "selected_hit": int(actual_index in selected),
        "calibration": {},
    }


def _fit_model(
    model_name: str,
    post_times: list[datetime],
    *,
    history_start: datetime,
    model_end: datetime,
    event_duration: timedelta,
    observation_fraction: float,
) -> Any:
    kwargs = {
        "start": history_start,
        "end": model_end,
        "event_duration": event_duration,
    }
    if model_name == "recent_rate":
        return RecentRateModel.fit(post_times, **kwargs)
    if model_name == "empirical_static":
        return EmpiricalWindowModel.fit(post_times, **kwargs)
    if model_name.startswith("adaptive"):
        return AdaptiveWindowModel.fit(post_times, **kwargs)
    if model_name == "negative_binomial":
        return NegativeBinomialRateModel.fit(post_times, **kwargs)
    if model_name == "gbm_conformal":
        return GradientBoostedConformalModel.fit(
            post_times, observation_fraction=observation_fraction, **kwargs
        )
    raise ValueError(f"Unknown experiment model: {model_name}")


def _forecast_model(
    model_name: str,
    model: Any,
    *,
    current_count: int,
    event_start: datetime,
    event_end: datetime,
    observed_at: datetime,
    observed_post_times: list[datetime],
) -> CountForecast:
    kwargs = {
        "current_count": current_count,
        "event_start": event_start,
        "event_end": event_end,
        "now": observed_at,
    }
    if isinstance(model, EmpiricalWindowModel):
        return model.forecast(**kwargs)
    if isinstance(model, AdaptiveWindowModel):
        return model.forecast(
            observed_post_times=observed_post_times,
            reconcile_counts="uncorrected" not in model_name,
            **kwargs,
        )
    return model.forecast(observed_post_times=observed_post_times, **kwargs)


def generate_walk_forward_predictions(
    post_times: list[datetime],
    *,
    history_start: datetime,
    history_end: datetime,
    event_duration: timedelta,
    observation_fractions: tuple[float, ...],
    intervals: tuple[MarketInterval, ...],
    model_name: str,
    folds: int,
    fold_days: int,
    refit_days: int = 7,
    simulated_endpoint_lag: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(item.astimezone(UTC) for item in post_times)
    first_test = history_end - timedelta(days=folds * fold_days)
    if first_test - history_start < timedelta(days=120):
        raise ValueError("Experiment needs at least 120 training days before the first fold")
    records: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for fraction in observation_fractions:
        for fold in range(folds):
            fold_start = first_test + timedelta(days=fold * fold_days)
            fold_end = min(history_end, fold_start + timedelta(days=fold_days))
            window_start = fold_start
            last_start = fold_end - event_duration
            model = None
            model_end = None
            next_refit = fold_start
            while window_start <= last_start:
                if model is None or window_start >= next_refit:
                    model_end = window_start
                    training_posts = [item for item in ordered if item < model_end]
                    model = _fit_model(
                        model_name,
                        training_posts,
                        history_start=history_start,
                        model_end=model_end,
                        event_duration=event_duration,
                        observation_fraction=fraction,
                    )
                    weights.append(
                        _model_artifact(model, model_name=model_name, fitted_at=model_end)
                    )
                    next_refit = window_start + timedelta(
                        days=fold_days + 1 if model_name == "empirical_static" else refit_days
                    )
                assert model_end is not None
                observed_at = window_start + event_duration * fraction
                window_end = window_start + event_duration
                current_count = bisect_left(ordered, observed_at) - bisect_left(
                    ordered, window_start
                )
                actual_final = bisect_left(ordered, window_end) - bisect_left(ordered, window_start)
                known = ordered[bisect_left(ordered, model_end) : bisect_left(ordered, observed_at)]
                if simulated_endpoint_lag > 0 and known:
                    event_known = [item for item in known if item >= window_start]
                    drop = min(
                        len(event_known), max(1, round(current_count * simulated_endpoint_lag))
                    )
                    dropped = set(event_known[-drop:])
                    known = [item for item in known if item not in dropped]
                forecast = _forecast_model(
                    model_name,
                    model,
                    current_count=current_count,
                    event_start=window_start,
                    event_end=window_end,
                    observed_at=observed_at,
                    observed_post_times=known,
                )
                record = _prediction_record(
                    model_name=model_name,
                    fold=fold,
                    observation_fraction=fraction,
                    window_start=window_start,
                    window_end=window_end,
                    observed_at=observed_at,
                    current_count=current_count,
                    actual_final=actual_final,
                    forecast=forecast,
                    intervals=intervals,
                )
                records.append(record)
                if forecast.empirical_weights:
                    weights.append(
                        {
                            "type": "prediction_weights",
                            "model": model_name,
                            "sample_id": record["sample_id"],
                            "effective_sample_size": forecast.effective_sample_size,
                            "kernel_bandwidth": forecast.kernel_bandwidth,
                            "finals": list(forecast.empirical_finals),
                            "weights": list(forecast.empirical_weights),
                        }
                    )
                window_start += timedelta(days=1)
    return records, weights


def _betacf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return result


def _beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(probability: float, a: float, b: float) -> float:
    low = 0.0
    high = 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if _beta_cdf(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def apply_beta_uncertainty(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in sorted(
        records, key=lambda item: (item["observation_fraction"], item["window_start"])
    ):
        prior = [
            item
            for item in output
            if item["observation_fraction"] == source["observation_fraction"]
            and item["window_end"] <= source["window_start"]
        ]
        optimism = (
            max(
                0.0,
                statistics.fmean(
                    item["selected_probability"] - item["selected_hit"] for item in prior
                ),
            )
            if len(prior) >= 20
            else 0.0
        )
        corrected = min(1.0, max(0.0, source["selected_probability"] - optimism))
        effective_n = max(1.0, source["effective_sample_size"])
        alpha = 1.0 + corrected * effective_n
        beta = 1.0 + (1.0 - corrected) * effective_n
        updated = dict(source)
        updated["model"] = "adaptive_beta"
        updated["decision_probability"] = alpha / (alpha + beta)
        updated["selected_lower_probability"] = beta_quantile(0.10, alpha, beta)
        updated["calibration"] = {
            "winner_optimism": optimism,
            "prior_completed_samples": len(prior),
            "beta_alpha": alpha,
            "beta_beta": beta,
        }
        output.append(updated)
    return output


def _temperature(probabilities: list[float], value: float) -> list[float]:
    powered = [max(item, 1e-12) ** (1.0 / value) for item in probabilities]
    total = math.fsum(powered)
    return [item / total for item in powered]


def apply_temperature_calibration(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    ordered = sorted(records, key=lambda item: (item["observation_fraction"], item["window_start"]))
    for source in ordered:
        prior = [
            item
            for item in records
            if item["observation_fraction"] == source["observation_fraction"]
            and item["window_end"] <= source["window_start"]
        ]
        temperature = 1.0
        if len(prior) >= 20:
            candidates = [0.50 + 0.05 * index for index in range(41)]
            temperature = min(
                candidates,
                key=lambda candidate: math.fsum(
                    -math.log(
                        max(
                            1e-12,
                            _temperature(item["probabilities"], candidate)[
                                item["actual_interval_index"]
                            ],
                        )
                    )
                    for item in prior
                ),
            )
        probabilities = _temperature(source["probabilities"], temperature)
        top1, selected = _best_bands(probabilities)
        updated = dict(source)
        updated.update(
            {
                "model": "adaptive_temperature",
                "probabilities": probabilities,
                "top1_index": top1,
                "selected_indexes": list(selected),
                "selected_probability": math.fsum(probabilities[index] for index in selected),
                "decision_probability": math.fsum(probabilities[index] for index in selected),
                "selected_hit": int(source["actual_interval_index"] in selected),
                "calibration": {
                    "temperature": temperature,
                    "prior_completed_samples": len(prior),
                },
            }
        )
        output.append(updated)
    return output


def _metric_values(records: list[dict[str, Any]]) -> dict[str, float]:
    errors = [item["expected_final"] - item["actual_final"] for item in records]
    log_losses = [
        -math.log(max(1e-12, item["probabilities"][item["actual_interval_index"]]))
        for item in records
    ]
    briers = []
    ranked_scores = []
    for item in records:
        probabilities = item["probabilities"]
        actual = item["actual_interval_index"]
        briers.append(
            math.fsum(
                (probability - (1.0 if index == actual else 0.0)) ** 2
                for index, probability in enumerate(probabilities)
            )
        )
        cumulative_probability = 0.0
        cumulative_actual = 0.0
        terms = []
        for index in range(len(probabilities) - 1):
            cumulative_probability += probabilities[index]
            cumulative_actual += 1.0 if index == actual else 0.0
            terms.append((cumulative_probability - cumulative_actual) ** 2)
        ranked_scores.append(math.fsum(terms) / max(1, len(terms)))
    selected_probabilities = [item["decision_probability"] for item in records]
    selected_hits = [item["selected_hit"] for item in records]
    selected_brier = statistics.fmean(
        (probability - hit) ** 2
        for probability, hit in zip(selected_probabilities, selected_hits, strict=True)
    )
    return {
        "samples": len(records),
        "mae": statistics.fmean(abs(item) for item in errors),
        "rmse": math.sqrt(statistics.fmean(item * item for item in errors)),
        "log_loss": statistics.fmean(log_losses),
        "brier_score": statistics.fmean(briers),
        "ranked_probability_score": statistics.fmean(ranked_scores),
        "coverage_80": statistics.fmean(
            item["central_80"][0] <= item["actual_final"] <= item["central_80"][1]
            for item in records
        ),
        "top1_accuracy": statistics.fmean(
            item["top1_index"] == item["actual_interval_index"] for item in records
        ),
        "selected_band_hit_rate": statistics.fmean(selected_hits),
        "selected_band_mean_probability": statistics.fmean(selected_probabilities),
        "selected_band_calibration_gap": statistics.fmean(selected_probabilities)
        - statistics.fmean(selected_hits),
        "selected_band_brier": selected_brier,
        "selected_lower_probability_mean": statistics.fmean(
            item["selected_lower_probability"]
            for item in records
            if item["selected_lower_probability"] is not None
        )
        if any(item["selected_lower_probability"] is not None for item in records)
        else math.nan,
    }


def _reliability(
    records: list[dict[str, Any]], bins: int = 5
) -> tuple[list[dict[str, Any]], float]:
    result = []
    total_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        items = [
            item
            for item in records
            if lower <= item["decision_probability"] < upper
            or (index == bins - 1 and item["decision_probability"] == 1)
        ]
        if not items:
            continue
        predicted = statistics.fmean(item["decision_probability"] for item in items)
        actual = statistics.fmean(item["selected_hit"] for item in items)
        total_error += len(items) / len(records) * abs(predicted - actual)
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "samples": len(items),
                "mean_probability": predicted,
                "hit_rate": actual,
            }
        )
    return result, total_error


def _block_bootstrap(
    records: list[dict[str, Any]], *, samples: int, block_size: int, seed: int
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda item: item["window_start"])
    distributions: dict[str, list[float]] = {
        name: []
        for name in (
            "mae",
            "log_loss",
            "brier_score",
            "ranked_probability_score",
            "selected_band_hit_rate",
            "selected_band_calibration_gap",
        )
    }
    starts = list(range(max(1, len(ordered) - block_size + 1)))
    for _ in range(samples):
        sampled = []
        while len(sampled) < len(ordered):
            start = rng.choice(starts)
            sampled.extend(ordered[start : start + block_size])
        values = _metric_values(sampled[: len(ordered)])
        for name in distributions:
            distributions[name].append(values[name])
    return {
        name: [_percentile(values, 0.025), _percentile(values, 0.975)]
        for name, values in distributions.items()
    }


def _comparison_values(
    model_records: list[dict[str, Any]], baseline_records: list[dict[str, Any]]
) -> dict[str, float]:
    model = _metric_values(model_records)
    baseline = _metric_values(baseline_records)
    _, model_ece = _reliability(model_records)
    _, baseline_ece = _reliability(baseline_records)
    return {
        "mae_improvement": baseline["mae"] - model["mae"],
        "log_loss_improvement": baseline["log_loss"] - model["log_loss"],
        "brier_improvement": baseline["brier_score"] - model["brier_score"],
        "rps_improvement": baseline["ranked_probability_score"] - model["ranked_probability_score"],
        "selected_hit_rate_change": model["selected_band_hit_rate"]
        - baseline["selected_band_hit_rate"],
        "selected_ece_improvement": baseline_ece - model_ece,
    }


def paired_block_bootstrap_comparison(
    model_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> dict[str, Any]:
    def key(item: dict[str, Any]) -> tuple[int, float, str]:
        return item["fold"], item["observation_fraction"], item["window_start"]

    model_by_key = {key(item): item for item in model_records}
    baseline_by_key = {key(item): item for item in baseline_records}
    common = sorted(model_by_key.keys() & baseline_by_key.keys())
    model_ordered = [model_by_key[item] for item in common]
    baseline_ordered = [baseline_by_key[item] for item in common]
    result: dict[str, Any] = _comparison_values(model_ordered, baseline_ordered)
    if samples <= 0:
        result["paired_block_bootstrap_95"] = {}
        return result
    rng = random.Random(seed)
    starts = list(range(max(1, len(common) - block_size + 1)))
    distributions = {name: [] for name in result}
    for _ in range(samples):
        indexes = []
        while len(indexes) < len(common):
            start = rng.choice(starts)
            indexes.extend(range(start, min(len(common), start + block_size)))
        indexes = indexes[: len(common)]
        values = _comparison_values(
            [model_ordered[index] for index in indexes],
            [baseline_ordered[index] for index in indexes],
        )
        for name, value in values.items():
            distributions[name].append(value)
    result["paired_block_bootstrap_95"] = {
        name: [_percentile(values, 0.025), _percentile(values, 0.975)]
        for name, values in distributions.items()
    }
    return result


def summarize_records(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    values = _metric_values(records)
    reliability, ece = _reliability(records)
    values["selected_band_ece"] = ece
    values["reliability"] = reliability
    values["block_bootstrap_95"] = _block_bootstrap(
        records, samples=bootstrap_samples, block_size=7, seed=seed
    )
    regimes = sorted({item["regime"] for item in records})
    values["by_regime"] = {
        regime: _metric_values([item for item in records if item["regime"] == regime])
        for regime in regimes
    }
    values["by_start_weekday"] = {
        str(weekday): _metric_values(
            [
                item
                for item in records
                if datetime.fromisoformat(item["window_start"]).weekday() == weekday
            ]
        )
        for weekday in range(7)
    }
    counts = [item["current_count"] for item in records]
    cuts = [_percentile([float(value) for value in counts], q) for q in (0.25, 0.5, 0.75)]
    labels = ("q1", "q2", "q3", "q4")
    groups = {label: [] for label in labels}
    for item in records:
        group = sum(item["current_count"] > cut for cut in cuts)
        groups[labels[group]].append(item)
    values["by_current_count_quartile"] = {
        label: _metric_values(items) for label, items in groups.items() if items
    }
    return values


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for item in items:
            output.write(json.dumps(_json_value(item), separators=(",", ":")) + "\n")


def _clone_model_output(
    records: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    *,
    source_name: str,
    target_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identifiers: dict[str, str] = {}
    cloned_records = []
    for item in records:
        cloned = dict(item)
        cloned["model"] = target_name
        identifier = hashlib.sha256(f"{item['sample_id']}|{target_name}".encode()).hexdigest()[:20]
        identifiers[item["sample_id"]] = identifier
        cloned["sample_id"] = identifier
        cloned_records.append(cloned)
    cloned_weights = []
    for item in weights:
        if item.get("model") != source_name:
            continue
        cloned = dict(item)
        cloned["model"] = target_name
        if "sample_id" in cloned:
            cloned["sample_id"] = identifiers[cloned["sample_id"]]
        cloned_weights.append(cloned)
    return cloned_records, cloned_weights


def build_routed_strategy_records(
    records_by_model: dict[str, list[dict[str, Any]]],
    *,
    route: dict[str, str] | None = None,
    strategy_name: str = ROUTED_STRATEGY_NAME,
) -> list[dict[str, Any]]:
    route = route or ROUTED_STRATEGY_SOURCES
    output = []
    for fraction, source_model in route.items():
        if source_model not in records_by_model:
            raise ValueError(f"Missing route source model: {source_model}")
        source_records = [
            item
            for item in records_by_model[source_model]
            if f"{float(item['observation_fraction']):.6f}" == fraction
        ]
        if not source_records:
            raise ValueError(f"No {source_model} records for observation fraction {fraction}")
        for item in source_records:
            cloned = dict(item)
            cloned["model"] = strategy_name
            cloned["route_source_model"] = source_model
            cloned["sample_id"] = hashlib.sha256(
                f"{item['sample_id']}|{strategy_name}".encode()
            ).hexdigest()[:20]
            cloned["calibration"] = {
                **item.get("calibration", {}),
                "routing_strategy": strategy_name,
                "route_source_model": source_model,
            }
            output.append(cloned)
    return sorted(
        output,
        key=lambda item: (
            item["fold"],
            item["observation_fraction"],
            item["window_start"],
        ),
    )


def run_routing_strategy_experiment(
    source_run: Path,
    *,
    output_root: Path,
    bootstrap_samples: int | None = None,
    seed: int | None = None,
) -> Path:
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    source_models = sorted(set(ROUTED_STRATEGY_SOURCES.values()) | {"recent_rate"})
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    source_hashes = {}
    for model_name in source_models:
        path = source_run / "checkpoints" / model_name / "predictions.jsonl"
        if not path.exists():
            raise ValueError(f"Missing source checkpoint: {path}")
        payload = path.read_bytes()
        source_hashes[model_name] = hashlib.sha256(payload).hexdigest()
        records_by_model[model_name] = [
            json.loads(line) for line in payload.decode("utf-8").splitlines() if line
        ]

    route_records = build_routed_strategy_records(records_by_model)
    samples = (
        int(bootstrap_samples)
        if bootstrap_samples is not None
        else int(manifest["bootstrap_samples"])
    )
    random_seed = int(seed) if seed is not None else int(manifest["random_seed"])
    fractions = tuple(float(item) for item in manifest["observation_fractions"])
    routed_keys = {
        (item["fold"], item["observation_fraction"], item["window_start"])
        for item in route_records
    }
    for model_name, records in records_by_model.items():
        source_keys = {
            (item["fold"], item["observation_fraction"], item["window_start"])
            for item in records
        }
        if routed_keys != source_keys:
            raise ValueError(f"Route samples do not match {model_name} samples")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / run_id
    by_fraction = {
        f"{fraction:.6f}": summarize_records(
            [item for item in route_records if item["observation_fraction"] == fraction],
            bootstrap_samples=samples,
            seed=random_seed + round(fraction * 10_000),
        )
        for fraction in fractions
    }
    alternatives = {
        model_name: summarize_records(
            records,
            bootstrap_samples=samples,
            seed=random_seed + len(model_name),
        )
        for model_name, records in records_by_model.items()
    }
    comparisons = {
        model_name: paired_block_bootstrap_comparison(
            route_records,
            records,
            samples=samples,
            block_size=int(manifest["bootstrap_block_days"]),
            seed=random_seed + len(model_name) + len(ROUTED_STRATEGY_NAME),
        )
        for model_name, records in records_by_model.items()
    }
    routing_manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": ROUTED_STRATEGY_NAME,
        "route": ROUTED_STRATEGY_SOURCES,
        "source_run": str(source_run),
        "source_run_id": manifest["run_id"],
        "source_checkpoint_sha256": source_hashes,
        "observation_fractions": list(fractions),
        "samples": len(route_records),
        "bootstrap_samples": samples,
        "bootstrap_block_days": int(manifest["bootstrap_block_days"]),
        "random_seed": random_seed,
        "post_hoc_exploratory": True,
        "independent_validation": False,
        "note": (
            "The routing rule was proposed after inspecting the source experiment. "
            "This replay measures the combined strategy but is not independent confirmation."
        ),
    }
    summary = {
        "strategy": ROUTED_STRATEGY_NAME,
        "overall": summarize_records(
            route_records,
            bootstrap_samples=samples,
            seed=random_seed + len(ROUTED_STRATEGY_NAME),
        ),
        "by_observation_fraction": by_fraction,
        "always_use_model": alternatives,
        "paired_comparisons": comparisons,
    }
    _write_json(run_dir / "manifest.json", routing_manifest)
    _write_jsonl(run_dir / "predictions.jsonl", route_records)
    _write_jsonl(
        run_dir / "checkpoints" / ROUTED_STRATEGY_NAME / "predictions.jsonl",
        route_records,
    )
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "run_id": run_id,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    _write_json(output_root / "latest.json", {"run_id": run_id, "path": str(run_dir)})
    return run_dir


def run_experiment_suite(
    post_times: list[datetime],
    *,
    history_start: datetime,
    history_end: datetime,
    event_duration: timedelta,
    observation_fractions: tuple[float, ...],
    intervals: tuple[MarketInterval, ...],
    interval_source: Path,
    output_root: Path,
    folds: int = 3,
    fold_days: int = 60,
    bootstrap_samples: int = 300,
    seed: int = 20261017,
    data_reconciliation: dict[str, Any] | None = None,
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / run_id
    _write_json(
        run_dir / "status.json",
        {"status": "running", "run_id": run_id, "started_at": datetime.now(UTC).isoformat()},
    )
    all_records: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    generated: dict[str, list[dict[str, Any]]] = {}
    specifications = (
        ("recent_rate", 0.0),
        ("empirical_static", 0.0),
        ("adaptive_uncorrected", 0.0),
        ("adaptive_uncorrected_lag2pct", 0.02),
        ("adaptive_reconciled_lag2pct", 0.02),
        ("negative_binomial", 0.0),
        ("gbm_conformal", 0.0),
    )
    for model_name, lag in specifications:
        LOGGER.info("Experiment model %s started", model_name)
        records, weights = generate_walk_forward_predictions(
            post_times,
            history_start=history_start,
            history_end=history_end,
            event_duration=event_duration,
            observation_fractions=observation_fractions,
            intervals=intervals,
            model_name=model_name,
            folds=folds,
            fold_days=fold_days,
            simulated_endpoint_lag=lag,
        )
        generated[model_name] = records
        all_records.extend(records)
        all_weights.extend(weights)
        checkpoint = run_dir / "checkpoints" / model_name
        _write_jsonl(checkpoint / "predictions.jsonl", records)
        _write_jsonl(checkpoint / "weights.jsonl", weights)
        LOGGER.info("Experiment model %s completed with %s predictions", model_name, len(records))
    reconciled_records, reconciled_weights = _clone_model_output(
        generated["adaptive_uncorrected"],
        all_weights,
        source_name="adaptive_uncorrected",
        target_name="adaptive_reconciled",
    )
    generated["adaptive_reconciled"] = reconciled_records
    all_records.extend(reconciled_records)
    all_weights.extend(reconciled_weights)
    checkpoint = run_dir / "checkpoints" / "adaptive_reconciled"
    _write_jsonl(checkpoint / "predictions.jsonl", reconciled_records)
    _write_jsonl(checkpoint / "weights.jsonl", reconciled_weights)
    beta_records = apply_beta_uncertainty(generated["adaptive_reconciled"])
    temperature_records = apply_temperature_calibration(generated["adaptive_reconciled"])
    generated["adaptive_beta"] = beta_records
    generated["adaptive_temperature"] = temperature_records
    all_records.extend(beta_records)
    all_records.extend(temperature_records)
    all_weights.extend(
        {
            "type": "calibration_weights",
            "model": item["model"],
            "sample_id": item["sample_id"],
            **item["calibration"],
        }
        for item in (*beta_records, *temperature_records)
    )

    summaries: dict[str, Any] = {}
    for model_name, records in generated.items():
        summaries[model_name] = {}
        for fraction in observation_fractions:
            subset = [item for item in records if item["observation_fraction"] == fraction]
            summaries[model_name][f"{fraction:.6f}"] = summarize_records(
                subset,
                bootstrap_samples=bootstrap_samples,
                seed=seed + round(fraction * 10_000) + len(model_name),
            )

    comparisons = {}
    for model_name, by_fraction in summaries.items():
        if model_name == "recent_rate":
            continue
        comparisons[model_name] = {}
        for fraction in by_fraction:
            model_subset = [
                item
                for item in generated[model_name]
                if f"{item['observation_fraction']:.6f}" == fraction
            ]
            baseline_subset = [
                item
                for item in generated["recent_rate"]
                if f"{item['observation_fraction']:.6f}" == fraction
            ]
            comparisons[model_name][fraction] = paired_block_bootstrap_comparison(
                model_subset,
                baseline_subset,
                samples=bootstrap_samples,
                block_size=7,
                seed=seed + round(float(fraction) * 10_000) + len(model_name),
            )

    interval_bytes = interval_source.read_bytes()
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_posts": len(post_times),
        "event_duration_hours": event_duration.total_seconds() / 3600,
        "observation_fractions": list(observation_fractions),
        "folds": folds,
        "fold_days": fold_days,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_block_days": 7,
        "random_seed": seed,
        "interval_source": str(interval_source),
        "interval_source_sha256": hashlib.sha256(interval_bytes).hexdigest(),
        "intervals": [item.__dict__ for item in intervals],
        "models": list(generated),
        "data_reconciliation": data_reconciliation,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "weights": "weights.jsonl",
            "summary": "summary.json",
            "model_checkpoints": "checkpoints/<model>/",
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_jsonl(run_dir / "predictions.jsonl", all_records)
    _write_jsonl(run_dir / "weights.jsonl", all_weights)
    _write_json(run_dir / "summary.json", {"metrics": summaries, "vs_recent_rate": comparisons})

    stages = {
        "01-calibration-and-baselines": {
            "models": ["recent_rate", "empirical_static", "adaptive_uncorrected"],
            "metrics": {
                name: summaries[name]
                for name in ("recent_rate", "empirical_static", "adaptive_uncorrected")
            },
            "vs_recent_rate": {
                name: comparisons[name] for name in ("empirical_static", "adaptive_uncorrected")
            },
        },
        "02-beta-uncertainty-and-selection": {
            "models": ["recent_rate", "adaptive_reconciled", "adaptive_beta"],
            "metrics": {
                name: summaries[name]
                for name in ("recent_rate", "adaptive_reconciled", "adaptive_beta")
            },
            "vs_recent_rate": {
                name: comparisons[name] for name in ("adaptive_reconciled", "adaptive_beta")
            },
        },
        "03-data-reconciliation": {
            "models": ["adaptive_uncorrected_lag2pct", "adaptive_reconciled_lag2pct"],
            "note": "The 2% endpoint-lag comparison is a declared stress test, not observed data.",
            "live_reconciliation": data_reconciliation,
            "metrics": {
                name: summaries[name]
                for name in ("adaptive_uncorrected_lag2pct", "adaptive_reconciled_lag2pct")
            },
            "vs_recent_rate": {
                name: comparisons[name]
                for name in (
                    "adaptive_uncorrected_lag2pct",
                    "adaptive_reconciled_lag2pct",
                )
            },
        },
        "04-protocol-and-bootstrap": {
            "folds": folds,
            "fold_days": fold_days,
            "block_days": 7,
            "exact_market_intervals": [item.__dict__ for item in intervals],
            "prediction_log": "../predictions.jsonl",
            "metrics": summaries,
            "vs_recent_rate": comparisons,
        },
        "05-model-upgrades": {
            "models": ["recent_rate", "adaptive_temperature", "negative_binomial", "gbm_conformal"],
            "metrics": {
                name: summaries[name]
                for name in (
                    "recent_rate",
                    "adaptive_temperature",
                    "negative_binomial",
                    "gbm_conformal",
                )
            },
            "vs_recent_rate": {
                name: comparisons[name]
                for name in ("adaptive_temperature", "negative_binomial", "gbm_conformal")
            },
        },
    }
    for name, payload in stages.items():
        _write_json(run_dir / "stages" / name / "report.json", payload)
    _write_json(
        run_dir / "status.json",
        {"status": "complete", "run_id": run_id, "finished_at": datetime.now(UTC).isoformat()},
    )
    _write_json(output_root / "latest.json", {"run_id": run_id, "path": str(run_dir)})
    return run_dir


def refresh_experiment_comparisons(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    generated: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        generated.setdefault(item["model"], []).append(item)
    weight_records = [
        json.loads(line)
        for line in (run_dir / "weights.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    weights_by_model: dict[str, list[dict[str, Any]]] = {}
    for item in weight_records:
        weights_by_model.setdefault(item["model"], []).append(item)
    for model_name, model_records in generated.items():
        checkpoint = run_dir / "checkpoints" / model_name
        _write_jsonl(checkpoint / "predictions.jsonl", model_records)
        _write_jsonl(checkpoint / "weights.jsonl", weights_by_model.get(model_name, []))
    baseline_records = generated["recent_rate"]
    comparisons: dict[str, Any] = {}
    seed = int(manifest["random_seed"])
    samples = int(manifest["bootstrap_samples"])
    for model_name, model_records in generated.items():
        if model_name == "recent_rate":
            continue
        comparisons[model_name] = {}
        for fraction in manifest["observation_fractions"]:
            key = f"{float(fraction):.6f}"
            model_subset = [
                item for item in model_records if item["observation_fraction"] == fraction
            ]
            baseline_subset = [
                item for item in baseline_records if item["observation_fraction"] == fraction
            ]
            comparisons[model_name][key] = paired_block_bootstrap_comparison(
                model_subset,
                baseline_subset,
                samples=samples,
                block_size=int(manifest["bootstrap_block_days"]),
                seed=seed + round(float(fraction) * 10_000) + len(model_name),
            )
    summary["vs_recent_rate"] = comparisons
    _write_json(summary_path, summary)
    for report_path in (run_dir / "stages").glob("*/report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        names = report.get("models")
        if names is None:
            report["vs_recent_rate"] = comparisons
        else:
            report["vs_recent_rate"] = {
                name: comparisons[name]
                for name in names
                if name != "recent_rate" and name in comparisons
            }
        _write_json(report_path, report)
    _write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "run_id": manifest["run_id"],
            "refreshed_at": datetime.now(UTC).isoformat(),
        },
    )

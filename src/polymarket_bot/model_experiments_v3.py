from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from polymarket_bot.database import PostObservation
from polymarket_bot.forecast import CountForecast, EmpiricalWindowModel, SeasonalGammaPoisson
from polymarket_bot.market_intervals import MarketInterval, actual_interval_index
from polymarket_bot.model_experiments import (
    GradientBoostedConformalModel,
    RecentRateModel,
    _best_bands,
    _write_json,
    _write_jsonl,
    paired_block_bootstrap_comparison,
)

_EXPECTED_EXPOSURE_CACHE: dict[tuple[int, datetime, datetime], float] = {}


def _count(values: tuple[datetime, ...], start: datetime, end: datetime) -> int:
    return bisect_left(values, end) - bisect_left(values, start)


def _weighted_effective_size(weights: list[float]) -> float:
    total = math.fsum(weights)
    return total * total / max(1e-15, math.fsum(value * value for value in weights))


def _overlap_effective_size(
    starts: list[datetime], weights: list[float], duration: timedelta
) -> float:
    if len(starts) != len(weights):
        raise ValueError("starts and weights must have the same length")
    if duration <= timedelta(0):
        raise ValueError("duration must be positive")
    total = math.fsum(weights)
    if total <= 0:
        return 0.0
    normalized = [value / total for value in weights]
    duration_seconds = duration.total_seconds()
    variance_inflation = math.fsum(
        left_weight
        * right_weight
        * max(
            0.0,
            1.0 - abs((left_start - right_start).total_seconds()) / duration_seconds,
        )
        for left_start, left_weight in zip(starts, normalized, strict=True)
        for right_start, right_weight in zip(starts, normalized, strict=True)
    )
    return 1.0 / max(1e-15, variance_inflation)


def _blend_forecasts(
    first: CountForecast,
    second: CountForecast,
    first_weight: float,
    *,
    confidence_score: float | None = None,
    low_confidence: bool | None = None,
) -> CountForecast:
    if first.current_count != second.current_count:
        raise ValueError("Forecasts must have the same current count")
    first_weight = min(1.0, max(0.0, first_weight))
    length = max(len(first.integer_probabilities), len(second.integer_probabilities))
    probabilities = tuple(
        first_weight
        * (first.integer_probabilities[index] if index < len(first.integer_probabilities) else 0)
        + (1 - first_weight)
        * (second.integer_probabilities[index] if index < len(second.integer_probabilities) else 0)
        for index in range(length)
    )
    return CountForecast(
        current_count=first.current_count,
        expected_final=0,
        stddev_final=0,
        posterior_alpha=0,
        posterior_beta=0,
        remaining_exposure=max(first.remaining_exposure, second.remaining_exposure),
        integer_probabilities=probabilities,
        base_expected_final=second.expected_final,
        realtime_multiplier=first.realtime_multiplier,
        regime=first.regime,
        recent_counts=first.recent_counts,
        effective_sample_size=first.effective_sample_size,
        source_count_delta=first.source_count_delta,
        raw_effective_sample_size=first.raw_effective_sample_size,
        overlap_effective_sample_size=first.overlap_effective_sample_size,
        novelty_score=first.novelty_score,
        confidence_score=confidence_score
        if confidence_score is not None
        else first.confidence_score,
        low_confidence=low_confidence
        if low_confidence is not None
        else first.low_confidence,
    )


@dataclass(frozen=True)
class AnalogConfig:
    name: str
    feature_mode: str = "cumulative"
    extras: tuple[str, ...] = ()
    weight_profile: str = "default"
    recency_half_life_days: float = 180.0
    neighbor_multiplier: float = 2.5
    confidence_fallback: bool = True
    anchor_step_days: tuple[int, int] = (4, 3)


ANALOG_CONFIGS = (
    AnalogConfig("analog_cumulative"),
    AnalogConfig("analog_disjoint", feature_mode="disjoint"),
    AnalogConfig("analog_disjoint_quiet", "disjoint", ("quiet",)),
    AnalogConfig("analog_disjoint_acceleration", "disjoint", ("acceleration",)),
    AnalogConfig("analog_disjoint_streak", "disjoint", ("active_streak",)),
    AnalogConfig("analog_disjoint_burst", "disjoint", ("burst",)),
    AnalogConfig(
        "analog_disjoint_all",
        "disjoint",
        ("quiet", "acceleration", "active_streak", "burst"),
    ),
    AnalogConfig(
        "analog_weight_recent",
        "disjoint",
        ("quiet", "acceleration", "active_streak", "burst"),
        "recent_heavy",
    ),
    AnalogConfig(
        "analog_half_life_90",
        "disjoint",
        ("quiet", "acceleration", "active_streak", "burst"),
        recency_half_life_days=90,
    ),
    AnalogConfig(
        "analog_neighbors_low",
        "disjoint",
        ("quiet", "acceleration", "active_streak", "burst"),
        neighbor_multiplier=1.75,
    ),
)


@dataclass(frozen=True)
class AnalogTrajectoryModel:
    post_times: tuple[datetime, ...]
    history_start: datetime
    history_end: datetime
    event_duration: timedelta
    baseline: SeasonalGammaPoisson
    base_model: EmpiricalWindowModel
    baseline_signature: int
    config: AnalogConfig
    _candidate_cache: dict[
        tuple[int, int, int], tuple[tuple[datetime, tuple[float, ...], int], ...]
    ] = field(default_factory=dict, compare=False, hash=False, repr=False)

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
        config: AnalogConfig,
    ) -> AnalogTrajectoryModel:
        ordered = sorted(item.astimezone(UTC) for item in post_times if start <= item < end)
        baseline = SeasonalGammaPoisson.fit(
            ordered, start=start, end=end, prior_strength_hours=168
        )
        return cls(
            tuple(ordered),
            start.astimezone(UTC),
            end.astimezone(UTC),
            event_duration,
            baseline,
            EmpiricalWindowModel.fit(
                ordered, start=start, end=end, event_duration=event_duration
            ),
            hash((baseline.hourly_rate, baseline.seasonality)),
            config,
        )

    def _expected(self, start: datetime, end: datetime) -> float:
        key = (self.baseline_signature, start, end)
        if key not in _EXPECTED_EXPOSURE_CACHE:
            _EXPECTED_EXPOSURE_CACHE[key] = (
                self.baseline.hourly_rate * self.baseline.effective_hours(start, end)
            )
        return _EXPECTED_EXPOSURE_CACHE[key]

    def _log_ratio(self, values: tuple[datetime, ...], start: datetime, end: datetime) -> float:
        return math.log((_count(values, start, end) + 2) / (self._expected(start, end) + 2))

    def _feature_names(self) -> tuple[str, ...]:
        if self.config.feature_mode == "cumulative":
            windows = (1, 3, 6, 12, 24, 72, 168, 336, 720)
            names = [f"last_{hours}h" for hours in windows]
        else:
            bounds = (
                (0, 1),
                (1, 3),
                (3, 6),
                (6, 12),
                (12, 24),
                (24, 72),
                (72, 168),
                (168, 336),
                (336, 720),
            )
            names = [f"hours_{lower}_{upper}" for lower, upper in bounds]
        return tuple(("event_elapsed", *names, *self.config.extras))

    def _features(
        self,
        values: tuple[datetime, ...],
        *,
        event_start: datetime,
        observed_at: datetime,
        current_override: int | None = None,
    ) -> tuple[float, ...]:
        current = (
            current_override
            if current_override is not None
            else _count(values, event_start, observed_at)
        )
        expected = self._expected(event_start, observed_at)
        features = [math.log((current + 2) / (expected + 2))]
        if self.config.feature_mode == "cumulative":
            for hours in (1, 3, 6, 12, 24, 72, 168, 336, 720):
                features.append(
                    self._log_ratio(values, observed_at - timedelta(hours=hours), observed_at)
                )
        else:
            for lower, upper in (
                (0, 1),
                (1, 3),
                (3, 6),
                (6, 12),
                (12, 24),
                (24, 72),
                (72, 168),
                (168, 336),
                (336, 720),
            ):
                features.append(
                    self._log_ratio(
                        values,
                        observed_at - timedelta(hours=upper),
                        observed_at - timedelta(hours=lower),
                    )
                )
        if "quiet" in self.config.extras:
            index = bisect_left(values, observed_at) - 1
            quiet = (
                min(72.0, (observed_at - values[index]).total_seconds() / 3600)
                if index >= 0
                else 72.0
            )
            features.append(math.log1p(quiet))
        if "acceleration" in self.config.extras:
            recent = self._log_ratio(values, observed_at - timedelta(hours=6), observed_at)
            prior = self._log_ratio(
                values, observed_at - timedelta(hours=24), observed_at - timedelta(hours=6)
            )
            features.append(recent - prior)
        if "active_streak" in self.config.extras:
            streak = 0
            cursor = observed_at.replace(minute=0, second=0, microsecond=0)
            for _ in range(24):
                if _count(values, cursor - timedelta(hours=1), cursor) == 0:
                    break
                streak += 1
                cursor -= timedelta(hours=1)
            features.append(streak / 6)
        if "burst" in self.config.extras:
            scores = []
            for hours in (1, 3, 6, 12, 24):
                start = observed_at - timedelta(hours=hours)
                actual = _count(values, start, observed_at)
                expected_count = self._expected(start, observed_at)
                scores.append(
                    (actual - expected_count)
                    / math.sqrt(max(1.0, expected_count + 0.35 * expected_count**2))
                )
            features.append(max(scores) / 3)
        return tuple(features)

    def _feature_weights(self) -> tuple[float, ...]:
        weights = [3.0]
        recent_weights = [1.5, 1.4, 1.3, 1.15, 1.0, 0.75, 0.55, 0.35, 0.2]
        if self.config.feature_mode == "cumulative":
            recent_weights = [0.2, 0.4, 0.7, 1.0, 1.3, 1.2, 0.8, 0.5, 0.3]
        if self.config.weight_profile == "recent_heavy":
            recent_weights = [
                value * (1.5 if index < 5 else 0.75)
                for index, value in enumerate(recent_weights)
            ]
        weights.extend(recent_weights)
        extra_weights = {
            "quiet": 0.8,
            "acceleration": 1.0,
            "active_streak": 0.6,
            "burst": 0.8,
        }
        weights.extend(extra_weights[name] for name in self.config.extras)
        return tuple(weights)

    def _window_starts(self, event_end_alignment: datetime) -> list[datetime]:
        alignment = event_end_alignment.astimezone(UTC)
        cursor_end = self.history_end.replace(
            hour=alignment.hour,
            minute=alignment.minute,
            second=alignment.second,
            microsecond=alignment.microsecond,
        )
        if cursor_end > self.history_end:
            cursor_end -= timedelta(days=1)
        earliest_start = self.history_start + timedelta(hours=720)
        starts = []
        step_index = 0
        while cursor_end - self.event_duration >= earliest_start:
            starts.append(cursor_end - self.event_duration)
            step = self.config.anchor_step_days[step_index % len(self.config.anchor_step_days)]
            cursor_end -= timedelta(days=step)
            step_index += 1
        return sorted(starts)

    def _candidate_rows(
        self,
        *,
        alignment_seconds: int,
        elapsed_seconds: int,
        duration_seconds: int,
    ) -> tuple[tuple[datetime, tuple[float, ...], int], ...]:
        key = (alignment_seconds, elapsed_seconds, duration_seconds)
        if key in self._candidate_cache:
            return self._candidate_cache[key]
        reference = self.history_end.replace(hour=0, minute=0, second=0, microsecond=0)
        reference += timedelta(seconds=alignment_seconds)
        elapsed = timedelta(seconds=elapsed_seconds)
        duration = timedelta(seconds=duration_seconds)
        rows = []
        for start in self._window_starts(reference):
            candidate_observed = start + elapsed
            candidate_end = start + duration
            features = self._features(
                self.post_times, event_start=start, observed_at=candidate_observed
            )
            remaining = _count(self.post_times, candidate_observed, candidate_end)
            rows.append((start, features, remaining))
        result = tuple(rows)
        self._candidate_cache[key] = result
        return result

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
        observed_post_times: list[datetime],
        source_count_delta: int = 0,
    ) -> CountForecast:
        observed_at = min(max(now.astimezone(UTC), event_start), event_end)
        remaining_hours = max(0.0, (event_end - observed_at).total_seconds() / 3600)
        if remaining_hours <= 0:
            return CountForecast(current_count, current_count, 0, 0, 0, 0)
        additions = tuple(
            item.astimezone(UTC)
            for item in observed_post_times
            if item.astimezone(UTC) >= self.history_end
        )
        known = tuple(sorted((*self.post_times, *additions)))
        base = self.base_model.forecast(
            current_count=current_count,
            event_start=event_start,
            event_end=event_end,
            now=observed_at,
        )
        current_features = self._features(
            known,
            event_start=event_start,
            observed_at=observed_at,
            current_override=current_count,
        )
        event_start_utc = event_start.astimezone(UTC)
        candidates = self._candidate_rows(
            alignment_seconds=(
                event_start_utc.hour * 3600
                + event_start_utc.minute * 60
                + event_start_utc.second
            ),
            elapsed_seconds=round((observed_at - event_start).total_seconds()),
            duration_seconds=round((event_end - event_start).total_seconds()),
        )
        if len(candidates) < 20:
            return base
        deviations = [
            max(0.10, statistics.pstdev(row[1][index] for row in candidates))
            for index in range(len(current_features))
        ]
        feature_weights = self._feature_weights()
        ranked = []
        for start, features, remaining in candidates:
            distance = math.sqrt(
                math.fsum(
                    feature_weights[index]
                    * ((features[index] - current_features[index]) / deviations[index]) ** 2
                    for index in range(len(features))
                )
            )
            ranked.append((distance, start, remaining))
        ranked.sort(key=lambda item: item[0])
        neighbor_count = min(
            len(ranked),
            max(20, round(math.sqrt(len(ranked)) * self.config.neighbor_multiplier)),
        )
        neighbors = ranked[:neighbor_count]
        scale = max(0.5, statistics.median(item[0] for item in neighbors))
        raw_weights = []
        finals = []
        neighbor_starts: list[datetime] = []
        for distance, start, remaining in neighbors:
            similarity = math.exp(-0.5 * (distance / scale) ** 2)
            age_days = max(
                0.0,
                (self.history_end - (start + self.event_duration)).total_seconds() / 86400,
            )
            weight = similarity * 0.5 ** (age_days / self.config.recency_half_life_days)
            raw_weights.append(weight)
            finals.append(current_count + remaining)
            neighbor_starts.append(start)
        total = math.fsum(raw_weights)
        normalized = [value / total for value in raw_weights]
        raw_n = _weighted_effective_size(raw_weights)
        overlap_n = _overlap_effective_size(
            neighbor_starts, raw_weights, self.event_duration
        )
        novelty = neighbors[0][0] / math.sqrt(max(1, len(current_features)))
        confidence = (
            min(1.0, overlap_n / 12)
            * math.exp(-max(0.0, novelty - 1.0))
            * math.exp(-abs(source_count_delta) / 5)
        )
        low_confidence = overlap_n < 8 or novelty > 1.75 or abs(source_count_delta) > 2
        mean = math.fsum(value * weight for value, weight in zip(finals, normalized, strict=True))
        variance = math.fsum(
            weight * (value - mean) ** 2
            for value, weight in zip(finals, normalized, strict=True)
        )
        bandwidth = max(2.0, 1.06 * math.sqrt(variance) * max(1.0, raw_n) ** -0.2)
        event_expected = self._expected(event_start, observed_at)
        multiplier = (current_count + 2) / (event_expected + 2)
        analog = CountForecast(
            current_count=current_count,
            expected_final=mean,
            stddev_final=math.sqrt(variance + bandwidth**2),
            posterior_alpha=0,
            posterior_beta=0,
            remaining_exposure=remaining_hours,
            empirical_finals=tuple(finals),
            kernel_bandwidth=bandwidth,
            empirical_weights=tuple(normalized),
            base_expected_final=base.expected_final,
            realtime_multiplier=multiplier,
            regime=(
                "high" if multiplier >= 1.2 else "low" if multiplier <= 0.82 else "normal"
            ),
            effective_sample_size=overlap_n,
            source_count_delta=source_count_delta,
            raw_effective_sample_size=raw_n,
            overlap_effective_sample_size=overlap_n,
            novelty_score=novelty,
            confidence_score=confidence,
            low_confidence=low_confidence,
        )
        if low_confidence and self.config.confidence_fallback:
            return _blend_forecasts(
                analog,
                base,
                max(0.25, confidence),
                confidence_score=confidence,
                low_confidence=True,
            )
        return analog


def _poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0:
        return 0
    if mean > 30:
        return max(0, round(rng.gauss(mean, math.sqrt(mean))))
    threshold = math.exp(-mean)
    product = 1.0
    value = 0
    while product > threshold:
        value += 1
        product *= rng.random()
    return value - 1


@dataclass(frozen=True)
class DynamicNegativeBinomialModel:
    post_times: tuple[datetime, ...]
    history_start: datetime
    history_end: datetime
    baseline: SeasonalGammaPoisson
    activity_phi: float
    innovation_stddev: float
    activity_center: float
    dispersion: float
    simulation_paths: int
    random_seed: int

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
        simulation_paths: int = 1200,
        random_seed: int = 20261017,
    ) -> DynamicNegativeBinomialModel:
        del event_duration
        ordered = tuple(sorted(item for item in post_times if start <= item < end))
        baseline = SeasonalGammaPoisson.fit(
            list(ordered), start=start, end=end, prior_strength_hours=168
        )
        counts = []
        means = []
        residuals = []
        cursor = start.replace(minute=0, second=0, microsecond=0)
        while cursor + timedelta(hours=1) <= end:
            count = _count(ordered, cursor, cursor + timedelta(hours=1))
            expected = max(0.05, baseline.hourly_rate * baseline.effective_hours(
                cursor, cursor + timedelta(hours=1)
            ))
            counts.append(count)
            means.append(expected)
            residuals.append(math.log((count + 0.5) / (expected + 0.5)))
            cursor += timedelta(hours=1)
        centered = [value - statistics.fmean(residuals) for value in residuals]
        denominator = math.fsum(value * value for value in centered[:-1])
        phi = (
            math.fsum(left * right for left, right in zip(centered[:-1], centered[1:], strict=True))
            / denominator
            if denominator > 0
            else 0.5
        )
        phi = min(0.97, max(0.05, phi))
        innovations = [
            right - phi * left
            for left, right in zip(centered[:-1], centered[1:], strict=True)
        ]
        innovation_stddev = max(0.05, statistics.pstdev(innovations))
        stationary_variance = innovation_stddev**2 / max(1e-6, 1 - phi**2)
        # Center the latent log activity so E[exp(activity)] = 1 in stationarity.
        # Without this correction, simulation adds a large Jensen-bias to every hour.
        activity_center = -0.5 * stationary_variance
        excess = math.fsum(
            max(0.0, (count - mean) ** 2 - count)
            for count, mean in zip(counts, means, strict=True)
        ) / len(counts)
        mean_squared = statistics.fmean(value * value for value in means)
        dispersion = min(100.0, max(0.25, mean_squared / max(1e-6, excess)))
        return cls(
            ordered,
            start,
            end,
            baseline,
            phi,
            innovation_stddev,
            activity_center,
            dispersion,
            simulation_paths,
            random_seed,
        )

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
        if observed_at >= event_end:
            return CountForecast(current_count, current_count, 0, 0, 0, 0)
        additions = tuple(item for item in observed_post_times if item >= self.history_end)
        known = tuple(sorted((*self.post_times, *additions)))
        residuals = []
        cursor = observed_at.replace(minute=0, second=0, microsecond=0)
        for offset in range(24, 0, -1):
            start = cursor - timedelta(hours=offset)
            end = start + timedelta(hours=1)
            count = _count(known, start, end)
            expected = max(
                0.05,
                self.baseline.hourly_rate * self.baseline.effective_hours(start, end),
            )
            residuals.append(math.log((count + 0.5) / (expected + 0.5)))
        observed_activity = self.activity_center
        for residual in residuals:
            observed_activity = 0.8 * observed_activity + 0.2 * residual
        activity = self.activity_center + 0.35 * (
            observed_activity - self.activity_center
        )
        seed_material = (
            f"{self.random_seed}|{event_start.isoformat()}|{observed_at.isoformat()}|{current_count}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        steps = []
        cursor = observed_at
        while cursor < event_end:
            next_hour = min(
                event_end,
                cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            )
            base_mean = self.baseline.hourly_rate * self.baseline.effective_hours(
                cursor, next_hour
            )
            duration = max(0.05, (next_hour - cursor).total_seconds() / 3600)
            steps.append((base_mean, duration))
            cursor = next_hour
        finals = []
        for _ in range(self.simulation_paths):
            state = activity
            remaining = 0
            for base_mean, duration in steps:
                state = (
                    self.activity_center
                    + self.activity_phi * (state - self.activity_center)
                    + rng.gauss(0, self.innovation_stddev)
                )
                mean = min(20.0, max(0.001, base_mean * math.exp(min(2.0, max(-2.0, state)))))
                shape = max(0.1, self.dispersion * duration)
                rate = rng.gammavariate(shape, mean / shape)
                remaining += _poisson(rng, rate)
            finals.append(current_count + remaining)
        multiplier = math.exp(min(2.0, max(-2.0, activity)))
        return CountForecast(
            current_count=current_count,
            expected_final=statistics.fmean(finals),
            stddev_final=statistics.pstdev(finals),
            posterior_alpha=0,
            posterior_beta=0,
            remaining_exposure=(event_end - observed_at).total_seconds() / 3600,
            empirical_finals=tuple(finals),
            kernel_bandwidth=0,
            realtime_multiplier=multiplier,
            regime=(
                "high" if multiplier >= 1.2 else "low" if multiplier <= 0.82 else "normal"
            ),
            effective_sample_size=float(self.simulation_paths),
        )


def _model_artifact(model: Any, name: str, fitted_at: datetime) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "type": "fitted_model",
        "model": name,
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
    if isinstance(model, AnalogTrajectoryModel):
        artifact["config"] = asdict(model.config)
        artifact["feature_names"] = list(model._feature_names())
        artifact["feature_weights"] = list(model._feature_weights())
    if isinstance(model, DynamicNegativeBinomialModel):
        artifact["dynamic_negative_binomial"] = {
            "activity_phi": model.activity_phi,
            "innovation_stddev": model.innovation_stddev,
            "activity_center": model.activity_center,
            "dispersion": model.dispersion,
            "simulation_paths": model.simulation_paths,
            "random_seed": model.random_seed,
        }
    if isinstance(model, GradientBoostedConformalModel):
        artifact["feature_names"] = list(model.feature_names)
        artifact["observation_fraction"] = model.observation_fraction
        artifact["gbm"] = {
            "intercept": model.regressor.intercept,
            "learning_rate": model.regressor.learning_rate,
            "stumps": [asdict(stump) for stump in model.regressor.stumps],
        }
        artifact["conformal_residuals"] = list(model.residuals)
    if isinstance(model, RecentRateModel):
        artifact["prior_hours"] = model.prior_hours
        artifact["lookback_hours"] = model.lookback_hours
    return artifact


def _prediction_record(
    *,
    model_name: str,
    fold: int,
    split: str,
    evaluation_mode: str,
    observation_fraction: float,
    window_start: datetime,
    window_end: datetime,
    observed_at: datetime,
    current_count: int,
    complete_current_count: int,
    actual_final: int,
    forecast: CountForecast,
    intervals: tuple[MarketInterval, ...],
    fold_start: datetime,
    mean_import_delay_seconds: float,
) -> dict[str, Any]:
    probabilities = [float(forecast.probability(item.lower, item.upper)) for item in intervals]
    total = math.fsum(probabilities)
    probabilities = [value / total for value in probabilities]
    actual_index = actual_interval_index(intervals, actual_final)
    top1, selected = _best_bands(probabilities)
    lower, upper = forecast.central_interval(0.8)
    selected_probability = math.fsum(probabilities[index] for index in selected)
    key = (
        f"v3|{model_name}|{evaluation_mode}|{fold}|{observation_fraction:.6f}|"
        f"{window_start.isoformat()}"
    )
    return {
        "sample_id": hashlib.sha256(key.encode()).hexdigest()[:20],
        "model": model_name,
        "fold": fold,
        "split": split,
        "evaluation_mode": evaluation_mode,
        "observation_fraction": observation_fraction,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "observed_at": observed_at.isoformat(),
        "non_overlapping_7d": (window_start - fold_start).days % 7 == 0,
        "current_count": current_count,
        "complete_current_count": complete_current_count,
        "data_count_delta": complete_current_count - current_count,
        "mean_import_delay_seconds": mean_import_delay_seconds,
        "actual_final": actual_final,
        "actual_interval_index": actual_index,
        "expected_final": forecast.expected_final,
        "stddev_final": forecast.stddev_final,
        "central_80": [lower, upper],
        "central_80_probability": float(forecast.probability(lower, upper)),
        "regime": forecast.regime,
        "effective_sample_size": forecast.effective_sample_size,
        "raw_effective_sample_size": forecast.raw_effective_sample_size,
        "overlap_effective_sample_size": forecast.overlap_effective_sample_size,
        "novelty_score": forecast.novelty_score,
        "confidence_score": forecast.confidence_score,
        "low_confidence": forecast.low_confidence,
        "probabilities": probabilities,
        "integer_pmf_start": forecast.current_count,
        "integer_probabilities": list(forecast.integer_probabilities),
        "top1_index": top1,
        "selected_indexes": list(selected),
        "selected_probability": selected_probability,
        "decision_probability": selected_probability,
        "signal_probability": selected_probability * forecast.confidence_score,
        "selected_lower_probability": None,
        "selected_hit": int(actual_index in selected),
        "calibration": {},
    }


def _split_names(folds: int) -> tuple[str, ...]:
    if folds < 3:
        raise ValueError("V3 requires at least three sequential folds")
    return tuple(
        "tuning" if index < folds - 2 else "validation" if index == folds - 2 else "final_test"
        for index in range(folds)
    )


def _available_times(
    observations: list[PostObservation], *, before: datetime
) -> list[datetime]:
    return [
        item.created_at
        for item in observations
        if item.created_at < before
        and item.imported_at is not None
        and item.imported_at <= before
    ]


def _fit_model(
    name: str,
    posts: list[datetime],
    *,
    history_start: datetime,
    model_end: datetime,
    event_duration: timedelta,
    simulation_paths: int,
    seed: int,
) -> Any:
    kwargs = {"start": history_start, "end": model_end, "event_duration": event_duration}
    if name == "recent_rate":
        return RecentRateModel.fit(posts, **kwargs)
    if name in {"empirical_static", "empirical_weekly"}:
        return EmpiricalWindowModel.fit(posts, **kwargs)
    if name == "dynamic_negative_binomial":
        return DynamicNegativeBinomialModel.fit(
            posts,
            simulation_paths=simulation_paths,
            random_seed=seed,
            **kwargs,
        )
    if name == "gbm_conformal_0h":
        return GradientBoostedConformalModel.fit(
            posts,
            observation_fraction=0.0,
            **kwargs,
        )
    config = next(item for item in ANALOG_CONFIGS if item.name == name)
    return AnalogTrajectoryModel.fit(posts, config=config, **kwargs)


def _forecast_model(
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
    return model.forecast(observed_post_times=observed_post_times, **kwargs)


def generate_predictions(
    post_times: list[datetime],
    observations: list[PostObservation],
    *,
    history_start: datetime,
    history_end: datetime,
    event_duration: timedelta,
    observation_fractions: tuple[float, ...],
    intervals: tuple[MarketInterval, ...],
    model_name: str,
    folds: int,
    fold_days: int,
    evaluation_mode: str,
    simulation_paths: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete = tuple(sorted(item.astimezone(UTC) for item in post_times))
    first_fold = history_end - timedelta(days=folds * fold_days)
    if first_fold - history_start < timedelta(days=120):
        raise ValueError("V3 needs at least 120 training days before the first fold")
    split_names = _split_names(folds)
    records = []
    weights = []
    visibility_cache: dict[datetime, list[datetime]] = {}

    def available(before: datetime) -> list[datetime]:
        if before not in visibility_cache:
            visibility_cache[before] = _available_times(observations, before=before)
        return visibility_cache[before]

    for fraction in observation_fractions:
        for fold, split in enumerate(split_names):
            fold_start = first_fold + timedelta(days=fold * fold_days)
            fold_end = min(history_end, fold_start + timedelta(days=fold_days))
            window_start = fold_start
            model = None
            model_end = None
            next_refit = fold_start
            while window_start <= fold_end - event_duration:
                refit_days = fold_days + 1 if model_name == "empirical_static" else 7
                if model is None or window_start >= next_refit:
                    model_end = window_start
                    training_posts = (
                        list(complete[: bisect_left(complete, model_end)])
                        if evaluation_mode == "ideal_complete_history"
                        else available(model_end)
                    )
                    model = _fit_model(
                        model_name,
                        training_posts,
                        history_start=history_start,
                        model_end=model_end,
                        event_duration=event_duration,
                        simulation_paths=simulation_paths,
                        seed=seed,
                    )
                    weights.append(_model_artifact(model, model_name, model_end))
                    next_refit = window_start + timedelta(days=refit_days)
                assert model_end is not None
                observed_at = window_start + event_duration * fraction
                window_end = window_start + event_duration
                complete_current = _count(complete, window_start, observed_at)
                if evaluation_mode == "ideal_complete_history":
                    current = complete_current
                    known = list(
                        complete[
                            bisect_left(complete, model_end) : bisect_left(complete, observed_at)
                        ]
                    )
                    delay = 0.0
                else:
                    visible = available(observed_at)
                    current = sum(window_start <= item < observed_at for item in visible)
                    known = [item for item in visible if item >= model_end]
                    delays = [
                        max(0.0, (item.imported_at - item.created_at).total_seconds())
                        for item in observations
                        if item.imported_at is not None
                        and item.imported_at <= observed_at
                        and window_start <= item.created_at < observed_at
                    ]
                    delay = statistics.fmean(delays) if delays else 0.0
                actual = _count(complete, window_start, window_end)
                forecast = _forecast_model(
                    model,
                    current_count=current,
                    event_start=window_start,
                    event_end=window_end,
                    observed_at=observed_at,
                    observed_post_times=known,
                )
                record = _prediction_record(
                    model_name=model_name,
                    fold=fold,
                    split=split,
                    evaluation_mode=evaluation_mode,
                    observation_fraction=fraction,
                    window_start=window_start,
                    window_end=window_end,
                    observed_at=observed_at,
                    current_count=current,
                    complete_current_count=complete_current,
                    actual_final=actual,
                    forecast=forecast,
                    intervals=intervals,
                    fold_start=fold_start,
                    mean_import_delay_seconds=delay,
                )
                records.append(record)
                if forecast.empirical_weights:
                    weights.append(
                        {
                            "type": "prediction_weights",
                            "model": model_name,
                            "sample_id": record["sample_id"],
                            "raw_effective_sample_size": forecast.raw_effective_sample_size,
                            "overlap_effective_sample_size": (
                                forecast.overlap_effective_sample_size
                            ),
                            "novelty_score": forecast.novelty_score,
                            "confidence_score": forecast.confidence_score,
                            "finals": list(forecast.empirical_finals),
                            "weights": list(forecast.empirical_weights),
                        }
                    )
                window_start += timedelta(days=1)
    return records, weights


def _metric_values(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"samples": 0}
    errors = [item["expected_final"] - item["actual_final"] for item in records]
    log_losses = [
        -math.log(max(1e-12, item["probabilities"][item["actual_interval_index"]]))
        for item in records
    ]
    briers = [
        math.fsum(
            (probability - (index == item["actual_interval_index"])) ** 2
            for index, probability in enumerate(item["probabilities"])
        )
        for item in records
    ]
    ordered = sorted(records, key=lambda item: (item["fold"], item["window_start"]))
    switches = sum(
        left["fold"] == right["fold"]
        and left["selected_indexes"] != right["selected_indexes"]
        for left, right in zip(ordered[:-1], ordered[1:], strict=True)
    )
    transitions = sum(
        left["fold"] == right["fold"]
        for left, right in zip(ordered[:-1], ordered[1:], strict=True)
    )
    return {
        "samples": len(records),
        "mae": statistics.fmean(abs(value) for value in errors),
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "log_loss": statistics.fmean(log_losses),
        "brier_score": statistics.fmean(briers),
        "coverage_80": statistics.fmean(
            item["central_80"][0] <= item["actual_final"] <= item["central_80"][1]
            for item in records
        ),
        "central_80_mean_width": statistics.fmean(
            item["central_80"][1] - item["central_80"][0] + 1 for item in records
        ),
        "central_80_mean_probability": statistics.fmean(
            item["central_80_probability"] for item in records
        ),
        "top1_accuracy": statistics.fmean(
            item["top1_index"] == item["actual_interval_index"] for item in records
        ),
        "selected_band_hit_rate": statistics.fmean(item["selected_hit"] for item in records),
        "selected_band_mean_probability": statistics.fmean(
            item["selected_probability"] for item in records
        ),
        "selected_band_switches": switches,
        "selected_band_switch_rate": switches / max(1, transitions),
        "low_confidence_rate": statistics.fmean(item["low_confidence"] for item in records),
        "mean_confidence_score": statistics.fmean(item["confidence_score"] for item in records),
        "mean_novelty_score": statistics.fmean(item["novelty_score"] for item in records),
        "mean_overlap_effective_sample_size": statistics.fmean(
            item.get("overlap_effective_sample_size", 0.0) for item in records
        ),
        "mean_absolute_data_count_delta": statistics.fmean(
            abs(item["data_count_delta"]) for item in records
        ),
        "mean_import_delay_seconds": statistics.fmean(
            item["mean_import_delay_seconds"] for item in records
        ),
        "selected_band_ece": _reliability(records)[1],
    }


def _reliability(
    records: list[dict[str, Any]], bins: int = 10
) -> tuple[list[dict[str, Any]], float]:
    if not records:
        return [], math.nan
    result = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        items = [
            item
            for item in records
            if lower <= item["selected_probability"] < upper
            or (index == bins - 1 and item["selected_probability"] == 1)
        ]
        if not items:
            continue
        predicted = statistics.fmean(item["selected_probability"] for item in items)
        actual = statistics.fmean(item["selected_hit"] for item in items)
        ece += len(items) / len(records) * abs(predicted - actual)
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "samples": len(items),
                "mean_probability": predicted,
                "hit_rate": actual,
            }
        )
    return result, ece


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _block_bootstrap(
    records: list[dict[str, Any]], *, samples: int, block_size: int, seed: int
) -> dict[str, list[float]]:
    if samples <= 0 or not records:
        return {}
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda item: (item["fold"], item["window_start"]))
    names = (
        "mae",
        "log_loss",
        "brier_score",
        "coverage_80",
        "central_80_mean_width",
        "selected_band_hit_rate",
        "selected_band_ece",
        "selected_band_switch_rate",
    )
    distributions = {name: [] for name in names}
    starts = list(range(max(1, len(ordered) - block_size + 1)))
    for _ in range(samples):
        sampled = []
        while len(sampled) < len(ordered):
            start = rng.choice(starts)
            sampled.extend(ordered[start : start + block_size])
        metrics = _metric_values(sampled[: len(ordered)])
        for name in names:
            distributions[name].append(metrics[name])
    return {
        name: [_percentile(values, 0.025), _percentile(values, 0.975)]
        for name, values in distributions.items()
    }


def _stratified_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "confidence": {
            "normal": _metric_values([item for item in records if not item["low_confidence"]]),
            "low": _metric_values([item for item in records if item["low_confidence"]]),
        },
        "data_count_delta": {
            "zero": _metric_values([item for item in records if item["data_count_delta"] == 0]),
            "nonzero": _metric_values(
                [item for item in records if item["data_count_delta"] != 0]
            ),
        },
        "novelty": {
            "at_most_1": _metric_values(
                [item for item in records if item["novelty_score"] <= 1]
            ),
            "above_1": _metric_values(
                [item for item in records if item["novelty_score"] > 1]
            ),
        },
        "overlap_effective_sample_size": {
            "below_8": _metric_values(
                [
                    item
                    for item in records
                    if item.get("overlap_effective_sample_size", 0.0) < 8
                ]
            ),
            "at_least_8": _metric_values(
                [
                    item
                    for item in records
                    if item.get("overlap_effective_sample_size", 0.0) >= 8
                ]
            ),
        },
    }


def _summary(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    reliability, _ = _reliability(records)
    result: dict[str, Any] = {
        "all_splits": _metric_values(records),
        "reliability": reliability,
        "strata": _stratified_metrics(records),
    }
    for split in ("tuning", "validation", "final_test"):
        subset = [item for item in records if item["split"] == split]
        result[split] = _metric_values(subset)
        result[f"{split}_block_bootstrap_95"] = _block_bootstrap(
            subset,
            samples=bootstrap_samples,
            block_size=7,
            seed=seed + len(split),
        )
        result[f"{split}_non_overlapping_7d"] = _metric_values(
            [item for item in subset if item["non_overlapping_7d"]]
        )
        result[f"{split}_non_overlapping_bootstrap_95"] = _block_bootstrap(
            [item for item in subset if item["non_overlapping_7d"]],
            samples=bootstrap_samples,
            block_size=1,
            seed=seed + len(split) + 100,
        )
    return result


def _selection_score(metrics_by_name: dict[str, dict[str, float]]) -> dict[str, float]:
    names = list(metrics_by_name)
    objectives = {
        "log_loss": False,
        "brier_score": False,
        "selected_band_hit_rate": True,
        "coverage_error": False,
        "central_80_mean_width": False,
        "selected_band_switch_rate": False,
        "selected_band_ece": False,
    }
    expanded = {
        name: {**values, "coverage_error": abs(values["coverage_80"] - 0.8)}
        for name, values in metrics_by_name.items()
    }
    scores = dict.fromkeys(names, 0.0)
    for metric, descending in objectives.items():
        ordered = sorted(names, key=lambda name: expanded[name][metric], reverse=descending)
        for rank, name in enumerate(ordered):
            scores[name] += rank
    return scores


def _pmf_forecast_from_record(record: dict[str, Any]) -> CountForecast:
    return CountForecast(
        current_count=record["integer_pmf_start"],
        expected_final=0,
        stddev_final=0,
        posterior_alpha=0,
        posterior_beta=0,
        remaining_exposure=max(
            0.0,
            (
                datetime.fromisoformat(record["window_end"])
                - datetime.fromisoformat(record["observed_at"])
            ).total_seconds()
            / 3600,
        ),
        integer_probabilities=tuple(record["integer_probabilities"]),
        regime=record["regime"],
        effective_sample_size=record["effective_sample_size"],
        raw_effective_sample_size=record["raw_effective_sample_size"],
        overlap_effective_sample_size=record.get("overlap_effective_sample_size", 0.0),
        novelty_score=record["novelty_score"],
        confidence_score=record["confidence_score"],
        low_confidence=record["low_confidence"],
    )


def _ensemble_weight(
    analog: list[dict[str, Any]], dynamic: list[dict[str, Any]], fraction: float
) -> tuple[float, float]:
    def key(item: dict[str, Any]) -> tuple[int, str]:
        return item["fold"], item["window_start"]

    left = {
        key(item): item
        for item in analog
        if item["split"] == "validation" and item["observation_fraction"] == fraction
    }
    right = {
        key(item): item
        for item in dynamic
        if item["split"] == "validation" and item["observation_fraction"] == fraction
    }
    pairs = [(left[item], right[item]) for item in sorted(left.keys() & right.keys())]
    candidates = []
    for step in range(21):
        weight = step / 20
        loss = statistics.fmean(
            -math.log(
                max(
                    1e-12,
                    weight * first["probabilities"][first["actual_interval_index"]]
                    + (1 - weight) * second["probabilities"][first["actual_interval_index"]],
                )
            )
            for first, second in pairs
        )
        candidates.append((loss, weight))
    return min(candidates)


def _make_test_ensemble(
    analog: list[dict[str, Any]],
    dynamic: list[dict[str, Any]],
    *,
    intervals: tuple[MarketInterval, ...],
    observation_fractions: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    weights = []
    for fraction in observation_fractions:
        validation_loss, analog_weight = _ensemble_weight(analog, dynamic, fraction)
        weights.append(
            {
                "type": "ensemble_weight",
                "model": "analog_dynamic_ensemble",
                "observation_fraction": fraction,
                "fit_split": "validation",
                "analog_weight": analog_weight,
                "dynamic_negative_binomial_weight": 1 - analog_weight,
                "validation_log_loss": validation_loss,
            }
        )
        dynamic_by_key = {
            (item["fold"], item["window_start"]): item
            for item in dynamic
            if item["split"] == "final_test" and item["observation_fraction"] == fraction
        }
        for source in analog:
            if source["split"] != "final_test" or source["observation_fraction"] != fraction:
                continue
            other = dynamic_by_key[(source["fold"], source["window_start"])]
            forecast = _blend_forecasts(
                _pmf_forecast_from_record(source),
                _pmf_forecast_from_record(other),
                analog_weight,
            )
            record = _prediction_record(
                model_name="analog_dynamic_ensemble",
                fold=source["fold"],
                split="final_test",
                evaluation_mode=source["evaluation_mode"],
                observation_fraction=fraction,
                window_start=datetime.fromisoformat(source["window_start"]),
                window_end=datetime.fromisoformat(source["window_end"]),
                observed_at=datetime.fromisoformat(source["observed_at"]),
                current_count=source["current_count"],
                complete_current_count=source["complete_current_count"],
                actual_final=source["actual_final"],
                forecast=forecast,
                intervals=intervals,
                fold_start=datetime.fromisoformat(source["window_start"])
                - timedelta(days=(datetime.fromisoformat(source["window_start"]).weekday() % 7)),
                mean_import_delay_seconds=source["mean_import_delay_seconds"],
            )
            record["non_overlapping_7d"] = source["non_overlapping_7d"]
            record["calibration"] = {
                "fit_split": "validation",
                "analog_model": source["model"],
                "analog_weight": analog_weight,
            }
            output.append(record)
    return output, weights


def run_experiment_suite_v3(
    post_times: list[datetime],
    post_observations: list[PostObservation],
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
    simulation_paths: int = 1200,
    seed: int = 20261017,
    data_reconciliation: dict[str, Any] | None = None,
) -> Path:
    _EXPECTED_EXPOSURE_CACHE.clear()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / "v3" / run_id
    _write_json(
        run_dir / "status.json",
        {"status": "running", "version": 3, "run_id": run_id},
    )
    first_fold = history_end - timedelta(days=folds * fold_days)
    visible_before_first = _available_times(post_observations, before=first_fold)
    as_known_available = bool(visible_before_first) and (
        max(visible_before_first) - min(visible_before_first) >= timedelta(days=120)
    )
    evaluation_modes = ["ideal_complete_history"]
    if as_known_available:
        evaluation_modes.append("as_known")
    specifications = (
        "recent_rate",
        "empirical_static",
        "empirical_weekly",
        *(item.name for item in ANALOG_CONFIGS),
        "dynamic_negative_binomial",
    )
    generated: dict[str, list[dict[str, Any]]] = {}
    all_records = []
    all_weights = []
    for mode in evaluation_modes:
        for name in specifications:
            output_name = name if mode == "ideal_complete_history" else f"{name}__as_known"
            records, weights = generate_predictions(
                post_times,
                post_observations,
                history_start=history_start,
                history_end=history_end,
                event_duration=event_duration,
                observation_fractions=observation_fractions,
                intervals=intervals,
                model_name=name,
                folds=folds,
                fold_days=fold_days,
                evaluation_mode=mode,
                simulation_paths=simulation_paths,
                seed=seed,
            )
            if output_name != name:
                for item in records:
                    item["model"] = output_name
                for item in weights:
                    item["model"] = output_name
            generated[output_name] = records
            all_records.extend(records)
            all_weights.extend(weights)
            checkpoint = run_dir / "checkpoints" / output_name
            _write_jsonl(checkpoint / "predictions.jsonl", records)
            _write_jsonl(checkpoint / "weights.jsonl", weights)
            _write_json(
                run_dir / "status.json",
                {
                    "status": "running",
                    "version": 3,
                    "run_id": run_id,
                    "last_completed_model": output_name,
                },
            )

    validation_metrics = {
        config.name: _metric_values(
            [item for item in generated[config.name] if item["split"] == "validation"]
        )
        for config in ANALOG_CONFIGS
    }
    scores = _selection_score(validation_metrics)
    selected_analog = min(scores, key=scores.__getitem__)
    ensemble, ensemble_weights = _make_test_ensemble(
        generated[selected_analog],
        generated["dynamic_negative_binomial"],
        intervals=intervals,
        observation_fractions=observation_fractions,
    )
    generated["analog_dynamic_ensemble"] = ensemble
    all_records.extend(ensemble)
    all_weights.extend(ensemble_weights)
    _write_jsonl(
        run_dir / "checkpoints" / "analog_dynamic_ensemble" / "predictions.jsonl",
        ensemble,
    )
    _write_jsonl(
        run_dir / "checkpoints" / "analog_dynamic_ensemble" / "weights.jsonl",
        ensemble_weights,
    )

    summaries = {
        name: {
            f"{fraction:.6f}": _summary(
                [item for item in records if item["observation_fraction"] == fraction],
                bootstrap_samples=bootstrap_samples,
                seed=seed + round(fraction * 10_000) + len(name),
            )
            for fraction in observation_fractions
        }
        for name, records in generated.items()
    }
    comparisons = {}
    baseline = generated["recent_rate"]
    for name, records in generated.items():
        if name == "recent_rate" or name.endswith("__as_known"):
            continue
        comparisons[name] = {}
        for fraction in observation_fractions:
            model_test = [
                item
                for item in records
                if item["split"] == "final_test" and item["observation_fraction"] == fraction
            ]
            baseline_test = [
                item
                for item in baseline
                if item["split"] == "final_test" and item["observation_fraction"] == fraction
            ]
            comparisons[name][f"{fraction:.6f}"] = paired_block_bootstrap_comparison(
                model_test,
                baseline_test,
                samples=bootstrap_samples,
                block_size=7,
                seed=seed + round(fraction * 10_000) + len(name),
            )

    selection = {
        "selected_analog_model": selected_analog,
        "selection_split": "validation",
        "selection_rule": (
            "equal-rank sum of log loss, Brier, top-three hit rate, 80% coverage "
            "error, interval width, switching rate, and selected-band ECE"
        ),
        "rank_scores": scores,
        "validation_metrics": validation_metrics,
        "ensemble_weights": ensemble_weights,
        "final_test_was_not_used_for_selection": True,
    }
    availability = {
        "historical_tracking_snapshots_available": False,
        "as_known_available": as_known_available,
        "as_known_records_before_first_fold": len(visible_before_first),
        "ideal_complete_history_is_lookahead_upper_bound": True,
        "note": (
            "Historical tracking totals are unavailable. as_known uses created_at plus "
            "imported_at; "
            "ideal_complete_history assumes every historical post was visible immediately."
        ),
    }
    interval_bytes = interval_source.read_bytes()
    manifest = {
        "version": 3,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_posts": len(post_times),
        "event_duration_hours": event_duration.total_seconds() / 3600,
        "observation_fractions": list(observation_fractions),
        "folds": folds,
        "fold_days": fold_days,
        "split_names": list(_split_names(folds)),
        "daily_overlapping_samples": True,
        "non_overlapping_stride_days": 7,
        "analog_window_sampling": {
            "window_duration_days": event_duration.total_seconds() / 86400,
            "anchor_direction": "backward_from_model_history_end",
            "anchor_step_days": list(ANALOG_CONFIGS[0].anchor_step_days),
            "uses_iso_week_groups": False,
            "effective_sample_size": (
                "inverse pairwise weighted seven-day window overlap"
            ),
        },
        "analog_configs": [asdict(item) for item in ANALOG_CONFIGS],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_block_days": 7,
        "simulation_paths": simulation_paths,
        "random_seed": seed,
        "models": list(generated),
        "interval_source": str(interval_source),
        "interval_source_sha256": hashlib.sha256(interval_bytes).hexdigest(),
        "intervals": [asdict(item) for item in intervals],
        "data_availability": availability,
        "data_reconciliation": data_reconciliation,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "weights": "weights.jsonl",
            "summary": "summary.json",
            "selection": "selection.json",
            "checkpoints": "checkpoints/<model>/",
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "selection.json", selection)
    _write_jsonl(run_dir / "predictions.jsonl", all_records)
    _write_jsonl(run_dir / "weights.jsonl", all_weights)
    _write_json(run_dir / "summary.json", {"metrics": summaries, "vs_recent_rate": comparisons})

    stages = {
        "01-unified-pmf": {
            "rule": (
                "All means, standard deviations, intervals, and market probabilities "
                "derive from one normalized integer PMF."
            ),
            "models": list(generated),
        },
        "02-fair-protocol": {
            "splits": list(_split_names(folds)),
            "models": ["empirical_static", "empirical_weekly", "recent_rate"],
            "metrics": {
                name: summaries[name]
                for name in ("recent_rate", "empirical_static", "empirical_weekly")
            },
        },
        "03-data-availability": availability,
        "04-analog-ablations": {
            "window_sampling": manifest["analog_window_sampling"],
            "configs": [asdict(item) for item in ANALOG_CONFIGS],
            "selection": selection,
            "metrics": {item.name: summaries[item.name] for item in ANALOG_CONFIGS},
        },
        "05-dynamic-negative-binomial": {
            "models": ["dynamic_negative_binomial", "analog_dynamic_ensemble"],
            "selected_analog": selected_analog,
            "ensemble_weights": ensemble_weights,
            "metrics": {
                name: summaries[name]
                for name in ("dynamic_negative_binomial", "analog_dynamic_ensemble")
            },
        },
        "06-multi-metric-selection": {
            "selection": selection,
            "final_test_metrics": {
                name: {
                    fraction: values["final_test"]
                    for fraction, values in model_summary.items()
                }
                for name, model_summary in summaries.items()
                if not name.endswith("__as_known")
            },
            "vs_recent_rate": comparisons,
        },
    }
    for name, payload in stages.items():
        _write_json(run_dir / "stages" / name / "report.json", payload)
    _write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "version": 3,
            "run_id": run_id,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    _write_json(output_root / "latest-v3.json", {"run_id": run_id, "path": str(run_dir)})
    return run_dir


def refresh_experiment_v3(run_dir: Path) -> None:
    """Rebuild V3 selection and reports from retained model checkpoints."""

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    intervals = tuple(MarketInterval(**item) for item in manifest["intervals"])
    fractions = tuple(float(item) for item in manifest["observation_fractions"])
    bootstrap_samples = int(manifest["bootstrap_samples"])
    seed = int(manifest["random_seed"])
    generated: dict[str, list[dict[str, Any]]] = {}
    for name in manifest["models"]:
        if name == "analog_dynamic_ensemble":
            continue
        path = run_dir / "checkpoints" / name / "predictions.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        for item in records:
            item["decision_probability"] = item["selected_probability"]
            item["signal_probability"] = (
                item["selected_probability"] * item["confidence_score"]
            )
        generated[name] = records
        _write_jsonl(path, records)

    validation_metrics = {
        config.name: _metric_values(
            [item for item in generated[config.name] if item["split"] == "validation"]
        )
        for config in ANALOG_CONFIGS
    }
    scores = _selection_score(validation_metrics)
    selected_analog = min(scores, key=scores.__getitem__)
    ensemble, ensemble_weights = _make_test_ensemble(
        generated[selected_analog],
        generated["dynamic_negative_binomial"],
        intervals=intervals,
        observation_fractions=fractions,
    )
    generated["analog_dynamic_ensemble"] = ensemble
    _write_jsonl(
        run_dir / "checkpoints" / "analog_dynamic_ensemble" / "predictions.jsonl",
        ensemble,
    )
    _write_jsonl(
        run_dir / "checkpoints" / "analog_dynamic_ensemble" / "weights.jsonl",
        ensemble_weights,
    )

    summaries = {
        name: {
            f"{fraction:.6f}": _summary(
                [item for item in records if item["observation_fraction"] == fraction],
                bootstrap_samples=bootstrap_samples,
                seed=seed + round(fraction * 10_000) + len(name),
            )
            for fraction in fractions
        }
        for name, records in generated.items()
    }
    comparisons = {}
    baseline = generated["recent_rate"]
    for name, records in generated.items():
        if name == "recent_rate" or name.endswith("__as_known"):
            continue
        comparisons[name] = {}
        for fraction in fractions:
            model_test = [
                item
                for item in records
                if item["split"] == "final_test" and item["observation_fraction"] == fraction
            ]
            baseline_test = [
                item
                for item in baseline
                if item["split"] == "final_test" and item["observation_fraction"] == fraction
            ]
            comparisons[name][f"{fraction:.6f}"] = paired_block_bootstrap_comparison(
                model_test,
                baseline_test,
                samples=bootstrap_samples,
                block_size=7,
                seed=seed + round(fraction * 10_000) + len(name),
            )
    selection = {
        "selected_analog_model": selected_analog,
        "selection_split": "validation",
        "selection_rule": (
            "equal-rank sum of log loss, Brier, top-three hit rate, 80% coverage "
            "error, interval width, switching rate, and selected-band ECE"
        ),
        "rank_scores": scores,
        "validation_metrics": validation_metrics,
        "ensemble_weights": ensemble_weights,
        "final_test_was_not_used_for_selection": True,
    }
    _write_json(run_dir / "selection.json", selection)
    _write_json(run_dir / "summary.json", {"metrics": summaries, "vs_recent_rate": comparisons})

    stages = {
        "01-unified-pmf": {
            "rule": (
                "All means, standard deviations, intervals, and market probabilities "
                "derive from one normalized integer PMF."
            ),
            "models": list(generated),
        },
        "02-fair-protocol": {
            "splits": manifest["split_names"],
            "models": ["empirical_static", "empirical_weekly", "recent_rate"],
            "metrics": {
                name: summaries[name]
                for name in ("recent_rate", "empirical_static", "empirical_weekly")
            },
        },
        "03-data-availability": manifest["data_availability"],
        "04-analog-ablations": {
            "window_sampling": manifest.get(
                "analog_window_sampling",
                {
                    "legacy_method": True,
                    "note": "Sampling metadata was not retained by this run.",
                },
            ),
            "configs": manifest.get("analog_configs", []),
            "selection": selection,
            "metrics": {item.name: summaries[item.name] for item in ANALOG_CONFIGS},
        },
        "05-dynamic-negative-binomial": {
            "models": ["dynamic_negative_binomial", "analog_dynamic_ensemble"],
            "selected_analog": selected_analog,
            "ensemble_weights": ensemble_weights,
            "metrics": {
                name: summaries[name]
                for name in ("dynamic_negative_binomial", "analog_dynamic_ensemble")
            },
        },
        "06-multi-metric-selection": {
            "selection": selection,
            "final_test_metrics": {
                name: {
                    fraction: values["final_test"]
                    for fraction, values in model_summary.items()
                }
                for name, model_summary in summaries.items()
                if not name.endswith("__as_known")
            },
            "vs_recent_rate": comparisons,
        },
    }
    for name, payload in stages.items():
        _write_json(run_dir / "stages" / name / "report.json", payload)

    existing_weights = [
        json.loads(line)
        for line in (run_dir / "weights.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    all_weights = [
        item for item in existing_weights if item.get("model") != "analog_dynamic_ensemble"
    ]
    all_weights.extend(ensemble_weights)
    all_records = [item for records in generated.values() for item in records]
    _write_jsonl(run_dir / "predictions.jsonl", all_records)
    _write_jsonl(run_dir / "weights.jsonl", all_weights)
    _write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "version": 3,
            "run_id": manifest["run_id"],
            "finished_at": manifest["generated_at"],
            "refreshed_at": datetime.now(UTC).isoformat(),
        },
    )

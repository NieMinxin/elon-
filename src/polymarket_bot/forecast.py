from __future__ import annotations

import math
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def _hour_start(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class CountForecast:
    current_count: int
    expected_final: float
    stddev_final: float
    posterior_alpha: float
    posterior_beta: float
    remaining_exposure: float
    empirical_finals: tuple[int, ...] = ()
    kernel_bandwidth: float = 0.0
    empirical_weights: tuple[float, ...] = ()
    base_expected_final: float | None = None
    realtime_multiplier: float = 1.0
    regime: str = "normal"
    recent_counts: tuple[tuple[int, int], ...] = ()
    effective_sample_size: float = 0.0
    source_count_delta: int = 0
    integer_probabilities: tuple[float, ...] = ()
    raw_effective_sample_size: float = 0.0
    overlap_effective_sample_size: float = 0.0
    novelty_score: float = 0.0
    confidence_score: float = 1.0
    low_confidence: bool = False

    def __post_init__(self) -> None:
        probabilities = self.integer_probabilities or self._build_integer_probabilities()
        total = math.fsum(probabilities)
        if not probabilities or not math.isfinite(total) or total <= 0:
            probabilities = (1.0,)
            total = 1.0
        normalized = tuple(max(0.0, item) / total for item in probabilities)
        mean = math.fsum(
            (self.current_count + offset) * probability
            for offset, probability in enumerate(normalized)
        )
        variance = math.fsum(
            probability * (self.current_count + offset - mean) ** 2
            for offset, probability in enumerate(normalized)
        )
        object.__setattr__(self, "integer_probabilities", normalized)
        object.__setattr__(self, "expected_final", mean)
        object.__setattr__(self, "stddev_final", math.sqrt(max(0.0, variance)))

    def _weighted_finals(self) -> tuple[tuple[int, float], ...]:
        if not self.empirical_finals:
            return ()
        if len(self.empirical_weights) == len(self.empirical_finals):
            total = math.fsum(self.empirical_weights)
            if total > 0:
                return tuple(
                    (value, weight / total)
                    for value, weight in zip(
                        self.empirical_finals, self.empirical_weights, strict=True
                    )
                )
        weight = 1.0 / len(self.empirical_finals)
        return tuple((value, weight) for value in self.empirical_finals)

    def _remaining_cdf(self, maximum: int) -> float:
        if maximum < 0:
            return 0.0
        if self.remaining_exposure <= 0:
            return 1.0
        probability = self.posterior_beta / (self.posterior_beta + self.remaining_exposure)
        log_probability = math.log(probability)
        log_failure = math.log1p(-probability)
        terms = []
        for value in range(maximum + 1):
            log_pmf = (
                math.lgamma(value + self.posterior_alpha)
                - math.lgamma(self.posterior_alpha)
                - math.lgamma(value + 1)
                + self.posterior_alpha * log_probability
                + value * log_failure
            )
            terms.append(math.exp(log_pmf))
        return min(1.0, math.fsum(terms))

    def _build_integer_probabilities(self) -> tuple[float, ...]:
        if self.remaining_exposure <= 0:
            return (1.0,)
        if self.empirical_finals:
            weighted = self._weighted_finals()
            if self.kernel_bandwidth <= 0:
                maximum = max(self.current_count, max(value for value, _ in weighted))
                probabilities = [0.0] * (maximum - self.current_count + 1)
                for value, weight in weighted:
                    probabilities[max(0, value - self.current_count)] += weight
                return tuple(probabilities)

            scale = self.kernel_bandwidth * math.sqrt(2)

            def cdf(value: float, center: int) -> float:
                return 0.5 * (1 + math.erf((value - center) / scale))

            floor = self.current_count - 0.5
            maximum_center = max(value for value, _ in weighted)
            maximum = max(
                self.current_count,
                math.ceil(maximum_center + 10 * self.kernel_bandwidth + 1),
            )
            probabilities = []
            for value in range(self.current_count, maximum + 1):
                lower = value - 0.5
                upper = value + 0.5
                probability = math.fsum(
                    weight
                    * max(0.0, cdf(upper, center) - cdf(max(lower, floor), center))
                    / max(1e-15, 1.0 - cdf(floor, center))
                    for center, weight in weighted
                )
                probabilities.append(probability)
            return tuple(probabilities)

        probability = self.posterior_beta / (self.posterior_beta + self.remaining_exposure)
        if not 0 < probability <= 1 or self.posterior_alpha <= 0:
            return (1.0,)
        log_pmf = self.posterior_alpha * math.log(probability)
        current = math.exp(log_pmf)
        probabilities = [current]
        cumulative = current
        mean_remaining = self.posterior_alpha * (1 - probability) / probability
        variance_remaining = mean_remaining / probability
        minimum_limit = math.ceil(mean_remaining + 12 * math.sqrt(max(variance_remaining, 1.0)))
        hard_limit = max(1_000, minimum_limit * 4 + 100)
        remaining = 0
        while remaining < hard_limit and (remaining < minimum_limit or cumulative < 1 - 1e-12):
            remaining += 1
            current *= (
                (remaining - 1 + self.posterior_alpha)
                / remaining
                * (1 - probability)
            )
            probabilities.append(current)
            cumulative += current
            if current == 0 and remaining >= minimum_limit:
                break
        return tuple(probabilities)

    def probability(self, lower: int, upper: int | None) -> Decimal:
        if upper is not None and (lower > upper or self.current_count > upper):
            return Decimal("0")
        first = max(0, lower - self.current_count)
        last = len(self.integer_probabilities) if upper is None else upper - self.current_count + 1
        result = math.fsum(self.integer_probabilities[first : max(first, last)])
        return Decimal(str(max(0.0, min(1.0, result))))

    def central_interval(self, mass: float) -> tuple[int, int]:
        if not 0 < mass < 1:
            raise ValueError("mass must be between 0 and 1")
        def quantile(target: float) -> int:
            cumulative = 0.0
            for offset, probability in enumerate(self.integer_probabilities):
                cumulative += probability
                if cumulative >= target:
                    return self.current_count + offset
            return self.current_count + len(self.integer_probabilities) - 1

        tail = (1 - mass) / 2
        return quantile(tail), quantile(1 - tail)


def correct_baseline_forecast(
    baseline: CountForecast,
    reference: CountForecast,
    updated: CountForecast,
    *,
    strength: float,
    maximum_likelihood_ratio: float = 20.0,
) -> CountForecast:
    """Apply only the realtime model's change since event open to a frozen baseline."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    if maximum_likelihood_ratio <= 1:
        raise ValueError("maximum_likelihood_ratio must be greater than 1")

    def probability_at(forecast: CountForecast, value: int) -> float:
        offset = value - forecast.current_count
        if offset < 0 or offset >= len(forecast.integer_probabilities):
            return 0.0
        return forecast.integer_probabilities[offset]

    current = updated.current_count
    maximum = max(
        baseline.current_count + len(baseline.integer_probabilities) - 1,
        reference.current_count + len(reference.integer_probabilities) - 1,
        updated.current_count + len(updated.integer_probabilities) - 1,
    )
    epsilon = 1e-12
    log_limit = math.log(maximum_likelihood_ratio)
    log_weights = []
    for value in range(current, maximum + 1):
        base_probability = probability_at(baseline, value)
        if base_probability <= 0:
            log_weights.append(-math.inf)
            continue
        log_ratio = math.log(probability_at(updated, value) + epsilon) - math.log(
            probability_at(reference, value) + epsilon
        )
        log_weights.append(
            math.log(base_probability) + strength * min(log_limit, max(-log_limit, log_ratio))
        )
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        return updated
    shift = max(finite)
    probabilities = tuple(
        math.exp(value - shift) if math.isfinite(value) else 0.0 for value in log_weights
    )
    return CountForecast(
        current_count=current,
        expected_final=0,
        stddev_final=0,
        posterior_alpha=0,
        posterior_beta=0,
        remaining_exposure=updated.remaining_exposure,
        integer_probabilities=probabilities,
        base_expected_final=baseline.expected_final,
        realtime_multiplier=updated.realtime_multiplier,
        regime=updated.regime,
        recent_counts=updated.recent_counts,
        effective_sample_size=min(
            value
            for value in (baseline.effective_sample_size, updated.effective_sample_size)
            if value > 0
        )
        if baseline.effective_sample_size > 0 and updated.effective_sample_size > 0
        else max(baseline.effective_sample_size, updated.effective_sample_size),
        source_count_delta=updated.source_count_delta,
        raw_effective_sample_size=updated.raw_effective_sample_size,
        overlap_effective_sample_size=updated.overlap_effective_sample_size,
        novelty_score=updated.novelty_score,
        confidence_score=updated.confidence_score,
        low_confidence=updated.low_confidence,
    )


@dataclass(frozen=True)
class BacktestResult:
    samples: int
    observation_fraction: float
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_log_loss: float
    coverage_80: float
    top1_interval_accuracy: float = 0.0
    top3_interval_coverage: float = 0.0
    model_name: str = "empirical"


@dataclass(frozen=True)
class RealtimeSignals:
    counts: tuple[tuple[int, int], ...]
    expected_counts: tuple[tuple[int, float], ...]
    rate_ratios: tuple[tuple[int, float], ...]
    rate_multiplier: float
    burst_score: float
    quiet_hours: float
    regime: str


@dataclass(frozen=True)
class SeasonalGammaPoisson:
    hourly_rate: float
    seasonality: tuple[float, ...]
    prior_strength_hours: float
    history_start: datetime
    history_end: datetime
    history_posts: int

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        prior_strength_hours: Decimal,
        shrinkage_hours: float = 24.0,
    ) -> SeasonalGammaPoisson:
        if end <= start:
            raise ValueError("History end must be after history start")
        filtered = [item for item in post_times if start <= item < end]
        total_hours = (end - start).total_seconds() / 3600
        if total_hours < 24 or not filtered:
            raise ValueError("At least 24 hours and one post are required")
        global_rate = len(filtered) / total_hours
        counts = [0] * 168
        exposures = [0] * 168
        for item in filtered:
            local = item.astimezone(EASTERN)
            counts[local.weekday() * 24 + local.hour] += 1
        cursor = _hour_start(start)
        while cursor < end:
            local = cursor.astimezone(EASTERN)
            exposures[local.weekday() * 24 + local.hour] += 1
            cursor += timedelta(hours=1)
        raw_rates = [
            (counts[index] + shrinkage_hours * global_rate) / (exposures[index] + shrinkage_hours)
            for index in range(168)
        ]
        weighted_mean = sum(raw_rates[index] * exposures[index] for index in range(168)) / sum(
            exposures
        )
        factors = tuple(rate / weighted_mean for rate in raw_rates)
        return cls(
            hourly_rate=global_rate,
            seasonality=factors,
            prior_strength_hours=float(prior_strength_hours),
            history_start=start,
            history_end=end,
            history_posts=len(filtered),
        )

    def effective_hours(self, start: datetime, end: datetime) -> float:
        if end <= start:
            return 0.0
        cursor = start.astimezone(UTC)
        finish = end.astimezone(UTC)
        total = 0.0
        while cursor < finish:
            next_hour = min(
                finish,
                cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            )
            local = cursor.astimezone(EASTERN)
            factor = self.seasonality[local.weekday() * 24 + local.hour]
            total += (next_hour - cursor).total_seconds() / 3600 * factor
            cursor = next_hour
        return total

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
    ) -> CountForecast:
        observed_until = min(max(now, event_start), event_end)
        elapsed_exposure = self.effective_hours(event_start, observed_until)
        remaining_exposure = self.effective_hours(observed_until, event_end)
        prior_alpha = self.hourly_rate * self.prior_strength_hours
        posterior_alpha = prior_alpha + current_count
        posterior_beta = self.prior_strength_hours + elapsed_exposure
        mean_remaining = posterior_alpha * remaining_exposure / posterior_beta
        variance_remaining = mean_remaining + (
            posterior_alpha * remaining_exposure**2 / posterior_beta**2
        )
        return CountForecast(
            current_count=current_count,
            expected_final=current_count + mean_remaining,
            stddev_final=math.sqrt(variance_remaining),
            posterior_alpha=posterior_alpha,
            posterior_beta=posterior_beta,
            remaining_exposure=remaining_exposure,
            effective_sample_size=self.prior_strength_hours + elapsed_exposure,
        )


@dataclass(frozen=True)
class EmpiricalWindowModel:
    post_times: tuple[datetime, ...]
    history_start: datetime
    history_end: datetime
    history_posts: int
    event_duration: timedelta

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
    ) -> EmpiricalWindowModel:
        if end <= start or event_duration <= timedelta(0):
            raise ValueError("History and event durations must be positive")
        filtered = tuple(sorted(item.astimezone(UTC) for item in post_times if start <= item < end))
        if end - start < event_duration * 5 or not filtered:
            raise ValueError("At least five event windows and one post are required")
        return cls(filtered, start, end, len(filtered), event_duration)

    def _window_starts(self, event_start: datetime) -> list[datetime]:
        event_start = event_start.astimezone(UTC)
        start = self.history_start.astimezone(UTC)
        offset = (event_start - start) % timedelta(days=1)
        cursor = start + offset
        if cursor < start:
            cursor += timedelta(days=1)
        last = self.history_end - self.event_duration
        values = []
        while cursor <= last:
            values.append(cursor)
            cursor += timedelta(days=1)
        return values

    def _count(self, start: datetime, end: datetime) -> int:
        return bisect_left(self.post_times, end) - bisect_left(self.post_times, start)

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
    ) -> CountForecast:
        duration = event_end - event_start
        observed_until = min(max(now, event_start), event_end)
        elapsed = observed_until - event_start
        remaining_hours = max(0.0, (event_end - observed_until).total_seconds() / 3600)
        if elapsed >= duration:
            return CountForecast(
                current_count,
                float(current_count),
                0.0,
                0.0,
                0.0,
                0.0,
                (current_count,),
                0.0,
            )

        pairs = [
            (
                self._count(start, start + elapsed),
                self._count(start, start + duration),
            )
            for start in self._window_starts(event_start)
        ]
        if len(pairs) < 20:
            raise ValueError("At least 20 comparable historical windows are required")

        mean_x = sum(item[0] for item in pairs) / len(pairs)
        mean_y = sum(item[1] for item in pairs) / len(pairs)
        x_variance = sum((item[0] - mean_x) ** 2 for item in pairs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in pairs) / x_variance
            if x_variance > 0
            else 0.0
        )
        intercept = mean_y - slope * mean_x
        point_forecast = intercept + slope * current_count
        samples = tuple(
            max(current_count, round(point_forecast + y - (intercept + slope * x)))
            for x, y in pairs
        )
        expected = sum(samples) / len(samples)
        sample_variance = sum((item - expected) ** 2 for item in samples) / max(1, len(samples) - 1)
        bandwidth = max(2.0, 1.06 * math.sqrt(sample_variance) * len(samples) ** -0.2)
        return CountForecast(
            current_count=current_count,
            expected_final=expected,
            stddev_final=math.sqrt(sample_variance + bandwidth**2),
            posterior_alpha=0.0,
            posterior_beta=0.0,
            remaining_exposure=remaining_hours,
            empirical_finals=samples,
            kernel_bandwidth=bandwidth,
            effective_sample_size=float(len(samples)),
        )


@dataclass(frozen=True)
class AdaptiveWindowModel:
    """Weekly-fitted analog model with online, no-refit trajectory conditioning."""

    post_times: tuple[datetime, ...]
    history_start: datetime
    history_end: datetime
    history_posts: int
    event_duration: timedelta
    baseline: SeasonalGammaPoisson
    base_model: EmpiricalWindowModel
    context_hours: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168, 336, 720)

    @classmethod
    def fit(
        cls,
        post_times: list[datetime],
        *,
        start: datetime,
        end: datetime,
        event_duration: timedelta,
    ) -> AdaptiveWindowModel:
        base_model = EmpiricalWindowModel.fit(
            post_times,
            start=start,
            end=end,
            event_duration=event_duration,
        )
        baseline = SeasonalGammaPoisson.fit(
            post_times,
            start=start,
            end=end,
            prior_strength_hours=Decimal("168"),
        )
        return cls(
            post_times=base_model.post_times,
            history_start=start.astimezone(UTC),
            history_end=end.astimezone(UTC),
            history_posts=base_model.history_posts,
            event_duration=event_duration,
            baseline=baseline,
            base_model=base_model,
        )

    @staticmethod
    def _count(post_times: tuple[datetime, ...], start: datetime, end: datetime) -> int:
        return bisect_left(post_times, end) - bisect_left(post_times, start)

    def _expected(self, start: datetime, end: datetime) -> float:
        return self.baseline.hourly_rate * self.baseline.effective_hours(start, end)

    def _known_times(self, observed_post_times: list[datetime]) -> tuple[datetime, ...]:
        additions = [
            item.astimezone(UTC)
            for item in observed_post_times
            if item.astimezone(UTC) >= self.history_end
        ]
        return tuple(sorted((*self.post_times, *additions)))

    def realtime_signals(
        self,
        *,
        observed_post_times: list[datetime],
        now: datetime,
    ) -> RealtimeSignals:
        now = now.astimezone(UTC)
        known = self._known_times(observed_post_times)
        counts: list[tuple[int, int]] = []
        expected_counts: list[tuple[int, float]] = []
        ratios: list[tuple[int, float]] = []
        for hours in self.context_hours:
            start = now - timedelta(hours=hours)
            actual = self._count(known, start, now)
            expected = self._expected(start, now)
            # Two pseudo-posts keep very short windows from dominating the state estimate.
            ratio = (actual + 2.0) / (expected + 2.0)
            counts.append((hours, actual))
            expected_counts.append((hours, expected))
            ratios.append((hours, ratio))

        ratio_map = dict(ratios)
        multiplier_weights = {3: 0.08, 6: 0.12, 12: 0.18, 24: 0.25, 72: 0.22, 168: 0.15}
        log_multiplier = math.fsum(
            weight * math.log(min(3.0, max(0.25, ratio_map[hours])))
            for hours, weight in multiplier_weights.items()
        ) / math.fsum(multiplier_weights.values())
        multiplier = math.exp(log_multiplier)

        expected_map = dict(expected_counts)
        count_map = dict(counts)
        burst_score = max(
            (
                (count_map[hours] - expected_map[hours])
                / math.sqrt(max(1.0, expected_map[hours] + 0.35 * expected_map[hours] ** 2))
                for hours in (1, 3, 6, 12, 24)
            ),
            default=0.0,
        )
        recent_ratios = [ratio_map[hours] for hours in (3, 6, 12, 24)]
        high_votes = sum(value >= 1.20 for value in recent_ratios)
        low_votes = sum(value <= 0.80 for value in recent_ratios)
        if multiplier >= 1.65 and high_votes >= 3:
            regime = "extreme_high"
        elif multiplier >= 1.20 and high_votes >= 2:
            regime = "high"
        elif multiplier <= 0.55 and low_votes >= 3:
            regime = "extreme_low"
        elif multiplier <= 0.82 and low_votes >= 2:
            regime = "low"
        else:
            regime = "normal"

        last_index = bisect_left(known, now) - 1
        quiet_hours = (
            (now - known[last_index]).total_seconds() / 3600 if last_index >= 0 else math.inf
        )
        return RealtimeSignals(
            counts=tuple(counts),
            expected_counts=tuple(expected_counts),
            rate_ratios=tuple(ratios),
            rate_multiplier=multiplier,
            burst_score=burst_score,
            quiet_hours=quiet_hours,
            regime=regime,
        )

    def _features(
        self,
        post_times: tuple[datetime, ...],
        *,
        event_start: datetime,
        observed_at: datetime,
        elapsed_count_override: int | None = None,
    ) -> tuple[float, ...]:
        expected_elapsed = self._expected(event_start, observed_at)
        elapsed_count = (
            elapsed_count_override
            if elapsed_count_override is not None
            else self._count(post_times, event_start, observed_at)
        )
        values = [math.log((elapsed_count + 2.0) / (expected_elapsed + 2.0))]
        for hours in self.context_hours:
            start = observed_at - timedelta(hours=hours)
            actual = self._count(post_times, start, observed_at)
            expected = self._expected(start, observed_at)
            values.append(math.log((actual + 2.0) / (expected + 2.0)))
        return tuple(values)

    def _window_starts(self, event_start: datetime) -> list[datetime]:
        event_start = event_start.astimezone(UTC)
        context = timedelta(hours=max(self.context_hours))
        first = self.history_start + context
        offset = (event_start - first) % timedelta(days=1)
        cursor = first + offset
        last = self.history_end - self.event_duration
        starts = []
        while cursor <= last:
            starts.append(cursor)
            cursor += timedelta(days=1)
        return starts

    def forecast(
        self,
        *,
        current_count: int,
        event_start: datetime,
        event_end: datetime,
        now: datetime,
        observed_post_times: list[datetime] | None = None,
        reconcile_counts: bool = True,
    ) -> CountForecast:
        event_start = event_start.astimezone(UTC)
        event_end = event_end.astimezone(UTC)
        observed_at = min(max(now.astimezone(UTC), event_start), event_end)
        elapsed = observed_at - event_start
        remaining_hours = max(0.0, (event_end - observed_at).total_seconds() / 3600)
        if elapsed >= event_end - event_start:
            return CountForecast(
                current_count,
                float(current_count),
                0.0,
                0.0,
                0.0,
                0.0,
                (current_count,),
                0.0,
                regime="complete",
            )

        observations = observed_post_times or []
        known = self._known_times(observations)
        signals = self.realtime_signals(observed_post_times=observations, now=observed_at)
        base_forecast = self.base_model.forecast(
            current_count=current_count,
            event_start=event_start,
            event_end=event_end,
            now=observed_at,
        )
        starts = self._window_starts(event_start)
        if len(starts) < 20:
            return CountForecast(
                **{
                    **base_forecast.__dict__,
                    "base_expected_final": base_forecast.expected_final,
                    "realtime_multiplier": signals.rate_multiplier,
                    "regime": signals.regime,
                    "recent_counts": signals.counts,
                }
            )

        current_features = self._features(
            known,
            event_start=event_start,
            observed_at=observed_at,
            # The tracking total is the settlement authority. If the posts endpoint
            # lags or omits a post, use that total for the event-level feature while
            # leaving short-window features untouched rather than inventing timestamps.
            elapsed_count_override=current_count if reconcile_counts else None,
        )
        candidates: list[tuple[datetime, tuple[float, ...], int]] = []
        current_remaining_exposure = self.baseline.effective_hours(observed_at, event_end)
        for start in starts:
            candidate_observed = start + elapsed
            candidate_end = start + (event_end - event_start)
            features = self._features(
                self.post_times,
                event_start=start,
                observed_at=candidate_observed,
            )
            remaining = self._count(self.post_times, candidate_observed, candidate_end)
            candidate_exposure = self.baseline.effective_hours(candidate_observed, candidate_end)
            exposure_scale = (
                current_remaining_exposure / candidate_exposure if candidate_exposure > 0 else 1.0
            )
            adjusted_remaining = round(remaining * min(1.25, max(0.80, exposure_scale)))
            candidates.append((start, features, adjusted_remaining))

        dimensions = len(current_features)
        deviations = []
        for index in range(dimensions):
            values = [item[1][index] for item in candidates]
            deviations.append(max(0.10, statistics.pstdev(values)))
        feature_weights = (3.0, 0.2, 0.4, 0.7, 1.0, 1.3, 1.2, 0.8, 0.5, 0.3)
        ranked = []
        for start, features, remaining in candidates:
            distance = math.sqrt(
                math.fsum(
                    feature_weights[index]
                    * ((features[index] - current_features[index]) / deviations[index]) ** 2
                    for index in range(dimensions)
                )
            )
            ranked.append((distance, start, remaining))
        ranked.sort(key=lambda item: item[0])
        neighbor_count = min(len(ranked), max(20, round(math.sqrt(len(ranked)) * 2.5)))
        neighbors = ranked[:neighbor_count]
        distance_scale = max(0.5, statistics.median(item[0] for item in neighbors))
        finals: list[int] = []
        weights: list[float] = []
        for distance, start, remaining in neighbors:
            similarity = math.exp(-0.5 * (distance / distance_scale) ** 2)
            age_days = max(0.0, (self.history_end - (start + self.event_duration)).days)
            recency = 0.5 ** (age_days / 180.0)
            finals.append(max(current_count, current_count + remaining))
            weights.append(similarity * recency)
        total_weight = math.fsum(weights)
        if total_weight <= 0:
            return base_forecast
        normalized = [item / total_weight for item in weights]
        expected = math.fsum(
            value * weight for value, weight in zip(finals, normalized, strict=True)
        )
        variance = math.fsum(
            weight * (value - expected) ** 2
            for value, weight in zip(finals, normalized, strict=True)
        )
        effective_n = 1.0 / math.fsum(weight * weight for weight in normalized)
        observed_endpoint_count = self._count(known, event_start, observed_at)
        bandwidth = max(2.0, 1.06 * math.sqrt(variance) * effective_n**-0.2)
        return CountForecast(
            current_count=current_count,
            expected_final=expected,
            stddev_final=math.sqrt(variance + bandwidth**2),
            posterior_alpha=0.0,
            posterior_beta=0.0,
            remaining_exposure=remaining_hours,
            empirical_finals=tuple(finals),
            kernel_bandwidth=bandwidth,
            empirical_weights=tuple(normalized),
            base_expected_final=base_forecast.expected_final,
            realtime_multiplier=signals.rate_multiplier,
            regime=signals.regime,
            recent_counts=signals.counts,
            effective_sample_size=effective_n,
            source_count_delta=current_count - observed_endpoint_count,
        )


def walk_forward_backtest(
    post_times: list[datetime],
    *,
    history_start: datetime,
    history_end: datetime,
    event_duration: timedelta,
    observation_fraction: float,
    prior_strength_hours: Decimal | None = None,
    holdout_days: int = 90,
) -> BacktestResult:
    if not 0 < observation_fraction < 1:
        raise ValueError("observation_fraction must be between 0 and 1")
    ordered = sorted(item.astimezone(UTC) for item in post_times)
    holdout_start = history_end - timedelta(days=holdout_days)
    train_posts = [item for item in ordered if item < holdout_start]
    model = EmpiricalWindowModel.fit(
        train_posts,
        start=history_start,
        end=holdout_start,
        event_duration=event_duration,
    )
    observation_duration = event_duration * observation_fraction
    errors: list[float] = []
    log_losses: list[float] = []
    covered = 0
    top1_hits = 0
    top3_hits = 0
    window_start = holdout_start
    last_start = history_end - event_duration
    while window_start <= last_start:
        observed_at = window_start + observation_duration
        window_end = window_start + event_duration
        start_index = bisect_left(ordered, window_start)
        observed_index = bisect_left(ordered, observed_at)
        end_index = bisect_left(ordered, window_end)
        current_count = observed_index - start_index
        actual_final = end_index - start_index
        forecast = model.forecast(
            current_count=current_count,
            event_start=window_start,
            event_end=window_end,
            now=observed_at,
        )
        error = forecast.expected_final - actual_final
        errors.append(error)
        actual_bin_start = actual_final // 20 * 20
        actual_probability = float(forecast.probability(actual_bin_start, actual_bin_start + 19))
        log_losses.append(-math.log(max(actual_probability, 1e-12)))
        lower, upper = forecast.central_interval(0.8)
        if lower <= actual_final <= upper:
            covered += 1
        top1, top3 = _best_interval_bands(forecast)
        if _in_interval(actual_final, top1):
            top1_hits += 1
        if any(_in_interval(actual_final, item) for item in top3):
            top3_hits += 1
        window_start += timedelta(days=1)
    if not errors:
        raise ValueError("Not enough holdout history for backtesting")
    return BacktestResult(
        samples=len(errors),
        observation_fraction=observation_fraction,
        mean_absolute_error=sum(abs(item) for item in errors) / len(errors),
        root_mean_squared_error=math.sqrt(sum(item * item for item in errors) / len(errors)),
        mean_log_loss=sum(log_losses) / len(log_losses),
        coverage_80=covered / len(errors),
        top1_interval_accuracy=top1_hits / len(errors),
        top3_interval_coverage=top3_hits / len(errors),
    )


def _in_interval(value: int, interval: tuple[int, int]) -> bool:
    return interval[0] <= value <= interval[1]


def _best_interval_bands(
    forecast: CountForecast,
    *,
    width: int = 20,
    band_size: int = 3,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    maximum = max(
        forecast.current_count + width * (band_size + 1),
        math.ceil(forecast.expected_final + 5 * forecast.stddev_final),
        max(forecast.empirical_finals, default=0) + width,
    )
    intervals = tuple((lower, lower + width - 1) for lower in range(0, maximum + width, width))
    probabilities = [float(forecast.probability(lower, upper)) for lower, upper in intervals]
    top1_index = max(range(len(intervals)), key=probabilities.__getitem__)
    if len(intervals) <= band_size:
        band_start = 0
    else:
        band_start = max(
            range(len(intervals) - band_size + 1),
            key=lambda index: math.fsum(probabilities[index : index + band_size]),
        )
    return intervals[top1_index], intervals[band_start : band_start + band_size]


def walk_forward_adaptive_backtest(
    post_times: list[datetime],
    *,
    history_start: datetime,
    history_end: datetime,
    event_duration: timedelta,
    observation_fraction: float,
    holdout_days: int = 90,
    refit_days: int = 7,
) -> BacktestResult:
    """Backtest a weekly-fitted model while updating live observations every forecast."""

    if not 0 < observation_fraction < 1:
        raise ValueError("observation_fraction must be between 0 and 1")
    if refit_days < 1:
        raise ValueError("refit_days must be positive")
    ordered = sorted(item.astimezone(UTC) for item in post_times)
    holdout_start = history_end - timedelta(days=holdout_days)
    observation_duration = event_duration * observation_fraction
    errors: list[float] = []
    log_losses: list[float] = []
    covered = 0
    top1_hits = 0
    top3_hits = 0
    window_start = holdout_start
    last_start = history_end - event_duration
    model: AdaptiveWindowModel | None = None
    model_end: datetime | None = None
    next_refit = holdout_start
    while window_start <= last_start:
        if model is None or window_start >= next_refit:
            model_end = window_start
            training_posts = [item for item in ordered if item < model_end]
            model = AdaptiveWindowModel.fit(
                training_posts,
                start=history_start,
                end=model_end,
                event_duration=event_duration,
            )
            next_refit = window_start + timedelta(days=refit_days)
        assert model_end is not None
        observed_at = window_start + observation_duration
        window_end = window_start + event_duration
        start_index = bisect_left(ordered, window_start)
        observed_index = bisect_left(ordered, observed_at)
        end_index = bisect_left(ordered, window_end)
        current_count = observed_index - start_index
        actual_final = end_index - start_index
        known_since_fit = ordered[bisect_left(ordered, model_end) : observed_index]
        forecast = model.forecast(
            current_count=current_count,
            event_start=window_start,
            event_end=window_end,
            now=observed_at,
            observed_post_times=known_since_fit,
        )
        error = forecast.expected_final - actual_final
        errors.append(error)
        actual_bin_start = actual_final // 20 * 20
        actual_probability = float(forecast.probability(actual_bin_start, actual_bin_start + 19))
        log_losses.append(-math.log(max(actual_probability, 1e-12)))
        lower, upper = forecast.central_interval(0.8)
        if lower <= actual_final <= upper:
            covered += 1
        top1, top3 = _best_interval_bands(forecast)
        if _in_interval(actual_final, top1):
            top1_hits += 1
        if any(_in_interval(actual_final, item) for item in top3):
            top3_hits += 1
        window_start += timedelta(days=1)
    if not errors:
        raise ValueError("Not enough holdout history for backtesting")
    return BacktestResult(
        samples=len(errors),
        observation_fraction=observation_fraction,
        mean_absolute_error=sum(abs(item) for item in errors) / len(errors),
        root_mean_squared_error=math.sqrt(sum(item * item for item in errors) / len(errors)),
        mean_log_loss=sum(log_losses) / len(log_losses),
        coverage_80=covered / len(errors),
        top1_interval_accuracy=top1_hits / len(errors),
        top3_interval_coverage=top3_hits / len(errors),
        model_name="adaptive",
    )

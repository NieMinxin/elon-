from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polymarket_bot.baseline_experiments import run_baseline_experiment
from polymarket_bot.config import ConfigError
from polymarket_bot.database import MySQLConfig, PostRepository
from polymarket_bot.forecast import (
    AdaptiveWindowModel,
    BacktestResult,
    walk_forward_adaptive_backtest,
    walk_forward_backtest,
)
from polymarket_bot.market_intervals import load_market_intervals
from polymarket_bot.model_experiments import (
    refresh_experiment_comparisons,
    run_experiment_suite,
    run_routing_strategy_experiment,
)
from polymarket_bot.model_experiments_v3 import refresh_experiment_v3, run_experiment_suite_v3
from polymarket_bot.xtracker import parse_datetime

LOGGER = logging.getLogger(__name__)
DEFAULT_EVENT_START = "2026-10-13T16:00:00Z"
DEFAULT_EVENT_END = "2026-10-20T16:00:00Z"
DEFAULT_BASELINE_EVENT_START = "2026-10-20T16:00:00Z"
DEFAULT_BASELINE_EVENT_END = "2026-10-27T16:00:00Z"
DEFAULT_GRU_EVENT_START = "2026-10-16T16:00:00Z"
DEFAULT_GRU_EVENT_END = "2026-10-23T16:00:00Z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and validate the Elon count model")
    parser.add_argument("--handle", default="elonmusk")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Audit the post history stored in MySQL")

    backtest = subparsers.add_parser("backtest", help="Run time-ordered model backtests")
    backtest.add_argument("--history-days", type=int, default=365)
    backtest.add_argument("--holdout-days", type=int, default=90)
    backtest.add_argument("--event-start", default=DEFAULT_EVENT_START)
    backtest.add_argument("--event-end", default=DEFAULT_EVENT_END)
    backtest.add_argument(
        "--fractions",
        default="0.142857,0.25,0.5,0.75",
        help="Comma-separated observed fractions of a seven-day event",
    )
    backtest.add_argument("--output", type=Path, default=Path("elon-model-report.json"))

    experiment = subparsers.add_parser(
        "experiment", help="Run retained, multi-stage model comparison experiments"
    )
    experiment.add_argument("--history-days", type=int, default=365)
    experiment.add_argument("--event-start", default=DEFAULT_EVENT_START)
    experiment.add_argument("--event-end", default=DEFAULT_EVENT_END)
    experiment.add_argument(
        "--fractions",
        default="0.142857,0.25,0.5,0.75",
        help="Comma-separated observed fractions of the event",
    )
    experiment.add_argument(
        "--interval-config",
        type=Path,
        default=Path("config/elon-musk-oct-13-20-2026-intervals.json"),
    )
    experiment.add_argument("--output-root", type=Path, default=Path("experiments"))
    experiment.add_argument("--folds", type=int, default=3)
    experiment.add_argument("--fold-days", type=int, default=60)
    experiment.add_argument("--bootstrap-samples", type=int, default=300)

    experiment_v3 = subparsers.add_parser(
        "experiment-v3", help="Run the unified-PMF, leakage-aware V3 experiments"
    )
    experiment_v3.add_argument("--history-days", type=int, default=365)
    experiment_v3.add_argument("--event-start", default=DEFAULT_EVENT_START)
    experiment_v3.add_argument("--event-end", default=DEFAULT_EVENT_END)
    experiment_v3.add_argument(
        "--fractions",
        default="0.142857,0.25,0.5,0.75",
        help="Comma-separated observed fractions of the event",
    )
    experiment_v3.add_argument(
        "--interval-config",
        type=Path,
        default=Path("config/elon-musk-oct-13-20-2026-intervals.json"),
    )
    experiment_v3.add_argument("--output-root", type=Path, default=Path("experiments"))
    experiment_v3.add_argument("--folds", type=int, default=3)
    experiment_v3.add_argument("--fold-days", type=int, default=60)
    experiment_v3.add_argument("--bootstrap-samples", type=int, default=300)
    experiment_v3.add_argument("--simulation-paths", type=int, default=1200)

    baseline = subparsers.add_parser(
        "experiment-baseline",
        help="Train and select the event-open (0h) baseline probability model",
    )
    baseline.add_argument("--history-days", type=int, default=365)
    baseline.add_argument("--event-start", default=DEFAULT_BASELINE_EVENT_START)
    baseline.add_argument("--event-end", default=DEFAULT_BASELINE_EVENT_END)
    baseline.add_argument(
        "--interval-config",
        type=Path,
        default=Path("config/elon-musk-oct-20-27-2026-intervals.json"),
    )
    baseline.add_argument("--output-root", type=Path, default=Path("experiments/baseline"))
    baseline.add_argument("--folds", type=int, default=3)
    baseline.add_argument("--fold-days", type=int, default=60)
    baseline.add_argument("--bootstrap-samples", type=int, default=300)
    baseline.add_argument("--simulation-paths", type=int, default=1200)
    baseline.add_argument("--seed", type=int, default=20261017)

    report = subparsers.add_parser(
        "experiment-report", help="Refresh reports from retained experiment predictions"
    )
    report.add_argument("--run-dir", type=Path, required=True)

    report_v3 = subparsers.add_parser(
        "experiment-v3-report", help="Refresh V3 reports from retained checkpoints"
    )
    report_v3.add_argument("--run-dir", type=Path, required=True)

    route = subparsers.add_parser(
        "experiment-route", help="Evaluate the fixed GBM/adaptive routing strategy"
    )
    route.add_argument("--source-run", type=Path, required=True)
    route.add_argument("--output-root", type=Path, default=Path("experiments/routes"))
    route.add_argument("--bootstrap-samples", type=int)
    route.add_argument("--seed", type=int)

    gru = subparsers.add_parser(
        "experiment-gru",
        help="Train leakage-aware MLP/GRU remaining-count distribution experiments",
    )
    gru.add_argument("--history-days", type=int, default=365)
    gru.add_argument("--event-start", default=DEFAULT_GRU_EVENT_START)
    gru.add_argument("--event-end", default=DEFAULT_GRU_EVENT_END)
    gru.add_argument(
        "--interval-config",
        type=Path,
        default=Path("config/elon-musk-oct-16-23-2026-intervals.json"),
    )
    gru.add_argument("--output-root", type=Path, default=Path("experiments/gru"))
    gru.add_argument("--observation-hours", default="0,24,42,84,126")
    gru.add_argument("--training-observation-step-hours", type=int, default=6)
    gru.add_argument("--folds", type=int, default=3)
    gru.add_argument("--events-per-fold", type=int, default=4)
    gru.add_argument("--pre-event-days", type=int, default=28)
    gru.add_argument("--hidden-size", type=int, default=16)
    gru.add_argument("--mixture-components", type=int, default=2)
    gru.add_argument("--max-remaining", type=int, default=700)
    gru.add_argument("--epochs", type=int, default=180)
    gru.add_argument("--patience", type=int, default=25)
    gru.add_argument("--batch-size", type=int, default=64)
    gru.add_argument("--learning-rate", type=float, default=0.003)
    gru.add_argument("--weight-decay", type=float, default=0.0001)
    gru.add_argument("--rps-weight", type=float, default=0.5)
    gru.add_argument("--ensemble-members", type=int, default=5)
    gru.add_argument("--bootstrap-samples", type=int, default=300)
    gru.add_argument("--seed", type=int, default=20261017)
    gru.add_argument("--device", default="auto", help="auto, cpu, or cuda")

    transfer = subparsers.add_parser(
        "experiment-gru-transfer",
        help="Evaluate retained GRU weights on another account without retraining",
    )
    transfer.add_argument("--source-run", type=Path, required=True)
    transfer.add_argument("--history-days", type=int, default=365)
    transfer.add_argument("--event-start", default=DEFAULT_GRU_EVENT_START)
    transfer.add_argument("--event-end", default=DEFAULT_GRU_EVENT_END)
    transfer.add_argument(
        "--interval-config",
        type=Path,
        default=Path("config/whitehouse-seven-day-generalization-intervals.json"),
    )
    transfer.add_argument(
        "--output-root", type=Path, default=Path("experiments/generalization/zero-shot")
    )
    transfer.add_argument("--bootstrap-samples", type=int, default=300)
    transfer.add_argument("--device", default="auto", help="auto, cpu, or cuda")

    compare = subparsers.add_parser(
        "forecast-gru-comparison",
        help="Compare the current route, original GRU, V3 GRU, and V4 ensemble",
    )
    compare.add_argument("--history-days", type=int, default=365)
    compare.add_argument("--event-start", required=True)
    compare.add_argument("--event-end", required=True)
    compare.add_argument("--as-of", required=True)
    compare.add_argument("--current-count", type=int, required=True)
    compare.add_argument("--interval-config", type=Path, required=True)
    compare.add_argument(
        "--original-run",
        type=Path,
        default=Path("experiments/gru/20261017T160625.192492Z"),
    )
    compare.add_argument(
        "--v3-run",
        type=Path,
        default=Path("experiments/gru-v3/20261017T162841.338782Z"),
    )
    compare.add_argument(
        "--v4-run",
        type=Path,
        default=Path("experiments/gru-v4/20261017T163347.179620Z"),
    )
    compare.add_argument("--fold", type=int, default=2)
    compare.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    compare.add_argument("--output", type=Path, required=True)

    forecast = subparsers.add_parser("forecast", help="Forecast the configured live event")
    forecast.add_argument("--history-days", type=int, default=365)
    forecast.add_argument("--event-start", default=DEFAULT_EVENT_START)
    forecast.add_argument("--event-end", default=DEFAULT_EVENT_END)
    forecast.add_argument("--as-of", help="Defaults to the current UTC time")
    return parser


def _complete_history_bounds(
    repository: PostRepository,
    handle: str,
    *,
    history_days: int,
    end: datetime,
) -> tuple[datetime, datetime, list[datetime]]:
    requested_start = end - timedelta(days=history_days)
    posts = repository.post_times(handle, start=requested_start, end=end)
    if not posts:
        raise ConfigError("No MySQL posts found for the requested training range")
    first_complete_day = (posts[0] + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = max(requested_start, first_complete_day)
    filtered = [item for item in posts if item >= start]
    if end - start < timedelta(days=120):
        raise ConfigError("At least 120 days of complete MySQL history are required")
    return start, end, filtered


def _percentile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def audit(repository: PostRepository, handle: str) -> None:
    stats = repository.stats(handle)
    earliest = stats["earliest"]
    latest = stats["latest"]
    if earliest is None or latest is None:
        raise ConfigError("MySQL contains no posts")
    start = earliest.replace(tzinfo=UTC)
    end = latest.replace(tzinfo=UTC) + timedelta(microseconds=1)
    posts = repository.post_times(handle, start=start, end=end)
    first_day = (start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = end.replace(hour=0, minute=0, second=0, microsecond=0)
    daily: list[int] = []
    cursor = first_day
    while cursor < last_day:
        next_day = cursor + timedelta(days=1)
        daily.append(sum(cursor <= item < next_day for item in posts))
        cursor = next_day
    gaps = sum(value == 0 for value in daily)
    LOGGER.info(
        "MySQL audit: posts=%s with_content=%s coverage=%s to %s complete_days=%s gaps=%s",
        len(posts),
        stats["with_content"],
        start.isoformat(),
        latest,
        len(daily),
        gaps,
    )
    LOGGER.info(
        "Daily counts: mean=%.1f median=%.1f p10=%.1f p90=%.1f min=%s max=%s",
        statistics.fmean(daily),
        statistics.median(daily),
        _percentile(daily, 0.10),
        _percentile(daily, 0.90),
        min(daily),
        max(daily),
    )
    latest_sync = repository.latest_successful_sync_end(handle)
    LOGGER.info("Latest successful sync end: %s", latest_sync or "none")


def _result_payload(result: BacktestResult) -> dict[str, float | int | str]:
    return {
        "model": result.model_name,
        "samples": result.samples,
        "observation_fraction": result.observation_fraction,
        "mae": result.mean_absolute_error,
        "rmse": result.root_mean_squared_error,
        "log_loss": result.mean_log_loss,
        "coverage_80": result.coverage_80,
        "top1_interval_accuracy": result.top1_interval_accuracy,
        "top3_interval_coverage": result.top3_interval_coverage,
    }


def backtest(repository: PostRepository, args: argparse.Namespace) -> None:
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    duration = event_end - event_start
    history_start, history_end, posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=event_start,
    )
    try:
        fractions = [float(item.strip()) for item in args.fractions.split(",")]
    except ValueError as error:
        raise ConfigError("--fractions must contain decimal numbers") from error
    results: list[BacktestResult] = []
    for fraction in fractions:
        baseline = walk_forward_backtest(
            posts,
            history_start=history_start,
            history_end=history_end,
            event_duration=duration,
            observation_fraction=fraction,
            holdout_days=args.holdout_days,
        )
        adaptive = walk_forward_adaptive_backtest(
            posts,
            history_start=history_start,
            history_end=history_end,
            event_duration=duration,
            observation_fraction=fraction,
            holdout_days=args.holdout_days,
            refit_days=7,
        )
        results.extend((baseline, adaptive))
        for result in (baseline, adaptive):
            LOGGER.info(
                "%s observed=%5.1f%% samples=%s MAE=%5.1f RMSE=%5.1f "
                "coverage80=%5.1f%% top1=%5.1f%% top3=%5.1f%% logloss=%.3f",
                result.model_name,
                result.observation_fraction * 100,
                result.samples,
                result.mean_absolute_error,
                result.root_mean_squared_error,
                result.coverage_80 * 100,
                result.top1_interval_accuracy * 100,
                result.top3_interval_coverage * 100,
                result.mean_log_loss,
            )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_posts": len(posts),
        "event_duration_hours": duration.total_seconds() / 3600,
        "holdout_days": args.holdout_days,
        "results": [_result_payload(item) for item in results],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Backtest report written to %s", args.output)


def experiment(repository: PostRepository, args: argparse.Namespace) -> None:
    if args.folds < 1 or args.fold_days < 14:
        raise ConfigError("--folds must be positive and --fold-days must be at least 14")
    if args.bootstrap_samples < 0:
        raise ConfigError("--bootstrap-samples cannot be negative")
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    if event_end <= event_start:
        raise ConfigError("--event-end must be after --event-start")
    try:
        fractions = tuple(float(item.strip()) for item in args.fractions.split(","))
    except ValueError as error:
        raise ConfigError("--fractions must contain decimal numbers") from error
    if not fractions or any(not 0 < item < 1 for item in fractions):
        raise ConfigError("Every observation fraction must be between 0 and 1")
    event_slug, intervals = load_market_intervals(args.interval_config)
    history_start, history_end, posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=event_start,
    )
    latest_reconciliation = repository.latest_reconciliation(args.handle)
    output = run_experiment_suite(
        posts,
        history_start=history_start,
        history_end=history_end,
        event_duration=event_end - event_start,
        observation_fractions=fractions,
        intervals=intervals,
        interval_source=args.interval_config,
        output_root=args.output_root,
        folds=args.folds,
        fold_days=args.fold_days,
        bootstrap_samples=args.bootstrap_samples,
        data_reconciliation=latest_reconciliation,
    )
    LOGGER.info("Experiment suite for %s written to %s", event_slug, output)


def experiment_v3(repository: PostRepository, args: argparse.Namespace) -> None:
    if args.folds < 3 or args.fold_days < 14:
        raise ConfigError("V3 requires at least three folds and --fold-days of at least 14")
    if args.bootstrap_samples < 0 or args.simulation_paths < 100:
        raise ConfigError("Bootstrap samples cannot be negative; simulation paths must be >= 100")
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    if event_end <= event_start:
        raise ConfigError("--event-end must be after --event-start")
    try:
        fractions = tuple(float(item.strip()) for item in args.fractions.split(","))
    except ValueError as error:
        raise ConfigError("--fractions must contain decimal numbers") from error
    if not fractions or any(not 0 < item < 1 for item in fractions):
        raise ConfigError("Every observation fraction must be between 0 and 1")
    event_slug, intervals = load_market_intervals(args.interval_config)
    history_start, history_end, posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=event_start,
    )
    observations = repository.post_observations(
        args.handle, start=history_start, end=history_end
    )
    output = run_experiment_suite_v3(
        posts,
        observations,
        history_start=history_start,
        history_end=history_end,
        event_duration=event_end - event_start,
        observation_fractions=fractions,
        intervals=intervals,
        interval_source=args.interval_config,
        output_root=args.output_root,
        folds=args.folds,
        fold_days=args.fold_days,
        bootstrap_samples=args.bootstrap_samples,
        simulation_paths=args.simulation_paths,
        data_reconciliation=repository.latest_reconciliation(args.handle),
    )
    LOGGER.info("V3 experiment suite for %s written to %s", event_slug, output)


def experiment_baseline(repository: PostRepository, args: argparse.Namespace) -> None:
    if args.folds < 3 or args.fold_days < 14:
        raise ConfigError(
            "Baseline experiment requires at least three folds and --fold-days of at least 14"
        )
    if args.bootstrap_samples < 0 or args.simulation_paths < 100:
        raise ConfigError("Bootstrap samples cannot be negative; simulation paths must be >= 100")
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    if event_end <= event_start:
        raise ConfigError("--event-end must be after --event-start")
    event_slug, intervals = load_market_intervals(args.interval_config)
    stats = repository.stats(args.handle)
    latest = stats["latest"]
    if latest is None:
        raise ConfigError("MySQL contains no posts")
    latest_utc = latest.replace(tzinfo=UTC) + timedelta(microseconds=1)
    training_cutoff = min(event_start, latest_utc)
    history_start, history_end, posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=training_cutoff,
    )
    observations = repository.post_observations(
        args.handle,
        start=history_start,
        end=history_end,
    )
    output = run_baseline_experiment(
        posts,
        observations,
        history_start=history_start,
        history_end=history_end,
        event_start=event_start,
        event_end=event_end,
        intervals=intervals,
        interval_source=args.interval_config,
        output_root=args.output_root,
        folds=args.folds,
        fold_days=args.fold_days,
        bootstrap_samples=args.bootstrap_samples,
        simulation_paths=args.simulation_paths,
        seed=args.seed,
        data_reconciliation=repository.latest_reconciliation(args.handle),
    )
    LOGGER.info(
        "Baseline-0h experiment for %s (training cutoff %s) written to %s",
        event_slug,
        history_end.isoformat(),
        output,
    )


def experiment_gru(repository: PostRepository, args: argparse.Namespace) -> None:
    try:
        from polymarket_bot.gru_experiments import GruExperimentConfig, run_gru_experiment
    except (ImportError, OSError) as error:
        raise ConfigError(
            "GRU dependencies are unavailable; install the optional ml dependencies"
        ) from error
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    if event_end <= event_start:
        raise ConfigError("--event-end must be after --event-start")
    try:
        observation_hours = tuple(
            int(item.strip()) for item in args.observation_hours.split(",") if item.strip()
        )
    except ValueError as error:
        raise ConfigError("--observation-hours must contain whole numbers") from error
    if args.bootstrap_samples < 1:
        raise ConfigError("--bootstrap-samples must be positive")
    event_slug, intervals = load_market_intervals(args.interval_config)
    stats = repository.stats(args.handle)
    latest = stats["latest"]
    if latest is None:
        raise ConfigError("MySQL contains no posts")
    training_cutoff = min(event_start, latest.replace(tzinfo=UTC) + timedelta(microseconds=1))
    history_start, history_end, posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=training_cutoff,
    )
    observations = repository.post_observations(
        args.handle, start=history_start, end=history_end
    )
    config = GruExperimentConfig(
        pre_event_days=args.pre_event_days,
        observation_hours=observation_hours,
        training_observation_step_hours=args.training_observation_step_hours,
        folds=args.folds,
        evaluation_events_per_fold=args.events_per_fold,
        hidden_size=args.hidden_size,
        mixture_components=args.mixture_components,
        max_remaining=args.max_remaining,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rps_weight=args.rps_weight,
        ensemble_members=args.ensemble_members,
        seed=args.seed,
    )
    try:
        output = run_gru_experiment(
            posts,
            observations,
            history_start=history_start,
            history_end=history_end,
            event_slug=event_slug,
            event_start=event_start,
            event_duration=event_end - event_start,
            intervals=intervals,
            interval_source=args.interval_config,
            output_root=args.output_root,
            config=config,
            device_name=args.device,
            bootstrap_samples=args.bootstrap_samples,
        )
    except (RuntimeError, ValueError) as error:
        raise ConfigError(f"GRU experiment failed: {error}") from error
    LOGGER.info("GRU experiment written to %s", output)


def experiment_gru_transfer(repository: PostRepository, args: argparse.Namespace) -> None:
    try:
        from polymarket_bot.gru_experiments import run_gru_transfer_experiment
    except (ImportError, OSError) as error:
        raise ConfigError(
            "GRU dependencies are unavailable; install the optional ml dependencies"
        ) from error
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    if event_end <= event_start:
        raise ConfigError("--event-end must be after --event-start")
    if args.bootstrap_samples < 1:
        raise ConfigError("--bootstrap-samples must be positive")
    event_slug, intervals = load_market_intervals(args.interval_config)
    stats = repository.stats(args.handle)
    latest = stats["latest"]
    if latest is None:
        raise ConfigError(f"MySQL contains no posts for {args.handle}")
    history_cutoff = min(event_start, latest.replace(tzinfo=UTC) + timedelta(microseconds=1))
    history_start, history_end, posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=history_cutoff,
    )
    observations = repository.post_observations(
        args.handle, start=history_start, end=history_end
    )
    try:
        output = run_gru_transfer_experiment(
            posts,
            observations,
            history_start=history_start,
            history_end=history_end,
            target_handle=args.handle,
            event_slug=event_slug,
            event_start=event_start,
            event_duration=event_end - event_start,
            intervals=intervals,
            interval_source=args.interval_config,
            source_run=args.source_run,
            output_root=args.output_root,
            device_name=args.device,
            bootstrap_samples=args.bootstrap_samples,
        )
    except (RuntimeError, ValueError) as error:
        raise ConfigError(f"GRU transfer experiment failed: {error}") from error
    LOGGER.info(
        "Zero-shot GRU transfer for %s using %s written to %s",
        args.handle,
        args.source_run,
        output,
    )


def forecast_gru_comparison(repository: PostRepository, args: argparse.Namespace) -> None:
    try:
        from polymarket_bot.gru_experiments import run_gru_live_comparison
    except (ImportError, OSError) as error:
        raise ConfigError(
            "GRU dependencies are unavailable; install the optional ml dependencies"
        ) from error
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    as_of = parse_datetime(args.as_of)
    if event_end <= event_start:
        raise ConfigError("--event-end must be after --event-start")
    if not event_start <= as_of <= event_end:
        raise ConfigError("--as-of must fall inside the event window")
    if args.current_count < 0:
        raise ConfigError("--current-count cannot be negative")
    event_slug, intervals = load_market_intervals(args.interval_config)
    history_start, _, history_posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=event_start,
    )
    observed_posts = repository.post_times(
        args.handle,
        start=event_start,
        end=as_of + timedelta(microseconds=1),
    )
    try:
        payload = run_gru_live_comparison(
            history_posts + observed_posts,
            history_start=history_start,
            event_slug=event_slug,
            event_start=event_start,
            event_end=event_end,
            as_of=as_of,
            official_current_count=args.current_count,
            intervals=intervals,
            original_run=args.original_run,
            v3_run=args.v3_run,
            v4_run=args.v4_run,
            output_path=args.output,
            fold=args.fold,
            device_name=args.device,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"Live GRU comparison failed: {error}") from error
    LOGGER.info(
        "Four-model forecast for %s at %s: count=%s observed_hours=%s",
        event_slug,
        payload["model_observed_at"],
        payload["official_current_count"],
        payload["observed_hours"],
    )
    for model, result in payload["models"].items():
        LOGGER.info(
            "%s expected=%.1f p80=%s-%s three_band=%s probability=%.1f%%",
            model,
            result["expected_final"],
            result["central_80"][0],
            result["central_80"][1],
            "/".join(result["three_band"]),
            result["three_band_probability"] * 100,
        )
    LOGGER.info("Comparison written to %s", args.output)


def forecast(repository: PostRepository, args: argparse.Namespace) -> None:
    event_start = parse_datetime(args.event_start)
    event_end = parse_datetime(args.event_end)
    now = parse_datetime(args.as_of) if args.as_of else datetime.now(UTC)
    observed_at = min(max(now, event_start), event_end)
    history_start, _, history_posts = _complete_history_bounds(
        repository,
        args.handle,
        history_days=args.history_days,
        end=event_start,
    )
    observed_posts = repository.post_times(
        args.handle,
        start=event_start,
        end=observed_at + timedelta(microseconds=1),
    )
    model = AdaptiveWindowModel.fit(
        history_posts,
        start=history_start,
        end=event_start,
        event_duration=event_end - event_start,
    )
    result = model.forecast(
        current_count=len(observed_posts),
        event_start=event_start,
        event_end=event_end,
        now=observed_at,
        observed_post_times=observed_posts,
    )
    lower, upper = result.central_interval(0.8)
    LOGGER.info(
        "Forecast at %s: count=%s expected=%.1f base=%.1f p80=%s-%s regime=%s multiplier=%.2f",
        observed_at.isoformat(),
        result.current_count,
        result.expected_final,
        result.base_expected_final or result.expected_final,
        lower,
        upper,
        result.regime,
        result.realtime_multiplier,
    )
    LOGGER.info("Recent counts: %s", dict(result.recent_counts))
    maximum = max(520, math.ceil(result.expected_final + 4 * result.stddev_final))
    intervals = [
        (lower, lower + 19, float(result.probability(lower, lower + 19)))
        for lower in range(0, maximum + 20, 20)
    ]
    best_start = max(
        range(len(intervals) - 2),
        key=lambda index: sum(item[2] for item in intervals[index : index + 3]),
    )
    best_band = intervals[best_start : best_start + 3]
    top_intervals = sorted(intervals, key=lambda item: item[2], reverse=True)[:6]
    LOGGER.info(
        "Best contiguous band: %s probability=%.1f%%",
        " / ".join(f"{lower}-{upper}" for lower, upper, _ in best_band),
        sum(item[2] for item in best_band) * 100,
    )
    LOGGER.info(
        "Highest interval probabilities: %s",
        ", ".join(
            f"{lower}-{upper}={probability * 100:.1f}%"
            for lower, upper, probability in top_intervals
        ),
    )
    signals = model.realtime_signals(observed_post_times=observed_posts, now=observed_at)
    LOGGER.info(
        "Rate ratios: %s burst_score=%.2f quiet_hours=%.2f",
        {hours: round(ratio, 2) for hours, ratio in signals.rate_ratios},
        signals.burst_score,
        signals.quiet_hours,
    )


def run(args: argparse.Namespace) -> None:
    if args.command == "experiment-report":
        refresh_experiment_comparisons(args.run_dir)
        LOGGER.info("Experiment comparisons refreshed in %s", args.run_dir)
        return
    if args.command == "experiment-v3-report":
        refresh_experiment_v3(args.run_dir)
        LOGGER.info("V3 experiment reports refreshed in %s", args.run_dir)
        return
    if args.command == "experiment-route":
        output = run_routing_strategy_experiment(
            args.source_run,
            output_root=args.output_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        LOGGER.info("Routing strategy experiment written to %s", output)
        return
    repository = PostRepository(MySQLConfig.from_env())
    repository.initialize()
    if args.command == "audit":
        audit(repository, args.handle)
    elif args.command == "backtest":
        backtest(repository, args)
    elif args.command == "experiment":
        experiment(repository, args)
    elif args.command == "experiment-v3":
        experiment_v3(repository, args)
    elif args.command == "experiment-baseline":
        experiment_baseline(repository, args)
    elif args.command == "experiment-gru":
        experiment_gru(repository, args)
    elif args.command == "experiment-gru-transfer":
        experiment_gru_transfer(repository, args)
    elif args.command == "forecast-gru-comparison":
        forecast_gru_comparison(repository, args)
    else:
        forecast(repository, args)


def cli() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        run(args)
    except ConfigError as error:
        LOGGER.error("Configuration error: %s", error)
        raise SystemExit(2) from error


if __name__ == "__main__":
    cli()

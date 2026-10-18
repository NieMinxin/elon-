#!/usr/bin/env python
"""Acceptance checks for a walk-forward experiment run.

Usage:
    python scripts/validate_experiment.py experiments/<run_id>
    python scripts/validate_experiment.py experiments/<run_id> --output acceptance

The checks are intentionally independent of the experiment code: metrics are
recomputed from ``predictions.jsonl`` and compared with ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUM_TOLERANCE = 1e-8
VALUE_TOLERANCE = 1e-6
ECE_BINS = 5


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def non_finite_paths(value: Any, prefix: str = "") -> list[str]:
    """Return paths of floats that are NaN/Inf, including nested containers."""

    found: list[str] = []
    if isinstance(value, float):
        if not math.isfinite(value):
            found.append(prefix or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(non_finite_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(non_finite_paths(item, f"{prefix}[{index}]"))
    return found


def interval_index(intervals: list[dict[str, Any]], value: int) -> int | None:
    for index, interval in enumerate(intervals):
        lower = int(interval["lower"])
        upper = interval.get("upper")
        if value >= lower and (upper is None or value <= int(upper)):
            return index
    return None


def actual_probability(probabilities: list[float], index: int) -> float:
    return float(probabilities[index])


def ranked_probability_score(
    probabilities: list[float], actual_index: int
) -> float:
    cumulative_prediction = 0.0
    score = 0.0
    for index, probability in enumerate(probabilities[:-1]):
        cumulative_prediction += probability
        cumulative_actual = 1.0 if actual_index <= index else 0.0
        score += (cumulative_prediction - cumulative_actual) ** 2
    return score / (len(probabilities) - 1)


def brier_score(probabilities: list[float], actual_index: int) -> float:
    return sum(
        (probability - (1.0 if index == actual_index else 0.0)) ** 2
        for index, probability in enumerate(probabilities)
    )


def calibration_error(pairs: list[tuple[float, float]]) -> float:
    """Expected calibration error over equal-width probability bins."""

    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for probability, outcome in pairs:
        bucket = min(int(probability * ECE_BINS), ECE_BINS - 1)
        buckets[bucket].append((probability, outcome))
    total = len(pairs)
    return sum(
        len(bucket) / total * abs(
            statistics.fmean(item[0] for item in bucket)
            - statistics.fmean(item[1] for item in bucket)
        )
        for bucket in buckets.values()
    )


def recompute_metrics(
    rows: list[dict[str, Any]], intervals: list[dict[str, Any]]
) -> dict[str, float | int]:
    errors = [float(row["expected_final"]) - int(row["actual_final"]) for row in rows]
    probabilities: list[float] = []
    brier_values: list[float] = []
    rps_values: list[float] = []
    covered = 0
    top1_hits = 0
    band_hits = 0
    raw_band_probabilities: list[float] = []
    decision_band_probabilities: list[float] = []
    raw_band_brier: list[float] = []
    decision_band_brier: list[float] = []
    raw_calibration_pairs: list[tuple[float, float]] = []
    decision_calibration_pairs: list[tuple[float, float]] = []
    for row in rows:
        distribution = [float(item) for item in row["probabilities"]]
        actual_index = int(row["actual_interval_index"])
        hit = int(row["selected_hit"])
        raw_probability = float(row["selected_probability"])
        decision_probability = float(row["decision_probability"])
        probabilities.append(actual_probability(distribution, actual_index))
        brier_values.append(brier_score(distribution, actual_index))
        rps_values.append(ranked_probability_score(distribution, actual_index))
        lower, upper = (int(item) for item in row["central_80"])
        covered += int(lower <= int(row["actual_final"]) <= upper)
        top1_hits += int(int(row["top1_index"]) == actual_index)
        band_hits += hit
        raw_band_probabilities.append(raw_probability)
        decision_band_probabilities.append(decision_probability)
        raw_band_brier.append((raw_probability - hit) ** 2)
        decision_band_brier.append((decision_probability - hit) ** 2)
        raw_calibration_pairs.append((raw_probability, float(hit)))
        decision_calibration_pairs.append((decision_probability, float(hit)))

    samples = len(rows)
    raw_mean = statistics.fmean(raw_band_probabilities)
    decision_mean = statistics.fmean(decision_band_probabilities)
    band_hit_rate = band_hits / samples
    return {
        "samples": samples,
        "mae": statistics.fmean(abs(item) for item in errors),
        "rmse": math.sqrt(statistics.fmean(item * item for item in errors)),
        "log_loss": statistics.fmean(-math.log(max(item, 1e-12)) for item in probabilities),
        "brier_score": statistics.fmean(brier_values),
        "ranked_probability_score": statistics.fmean(rps_values),
        "coverage_80": covered / samples,
        "top1_accuracy": top1_hits / samples,
        "selected_band_hit_rate": band_hit_rate,
        # The actionable probability: identical to selected_probability except
        # for adaptive_beta, which recalibrates it before the decision.
        "selected_band_mean_probability": decision_mean,
        "selected_band_calibration_gap": decision_mean - band_hit_rate,
        "selected_band_ece": calibration_error(decision_calibration_pairs),
        "selected_band_brier": statistics.fmean(decision_band_brier),
        # Raw ensemble probability, kept for transparency.
        "raw_band_mean_probability": raw_mean,
        "raw_band_calibration_gap": raw_mean - band_hit_rate,
        "raw_band_ece": calibration_error(raw_calibration_pairs),
        "raw_band_brier": statistics.fmean(raw_band_brier),
    }


def _close(left: float, right: float, tolerance: float = VALUE_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0, abs(float(left)), abs(float(right))
    )


def validate_predictions(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    fractions: list[float],
    intervals: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    checks: dict[str, Any] = {}
    errors: dict[str, list[str]] = defaultdict(list)

    expected_keys = {
        "sample_id",
        "model",
        "fold",
        "observation_fraction",
        "window_start",
        "window_end",
        "observed_at",
        "current_count",
        "actual_final",
        "actual_interval_index",
        "expected_final",
        "stddev_final",
        "central_80",
        "regime",
        "effective_sample_size",
        "source_count_delta",
        "probabilities",
        "top1_index",
        "selected_indexes",
        "selected_probability",
        "decision_probability",
        "selected_lower_probability",
        "selected_hit",
        "calibration",
    }
    missing_keys = Counter()
    for row in rows:
        missing_keys.update(expected_keys - set(row))
    checks["missing_fields"] = {key: count for key, count in missing_keys.items()}

    # Row grid: every model x fraction x sample_id should be present exactly once.
    by_model_fraction: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model_fraction[(str(row["model"]), f"{float(row['observation_fraction']):.6f}")].append(
            row
        )
    grid: dict[str, Any] = {}
    window_sets: dict[tuple[str, str], set[str]] = {}
    for model in models:
        for fraction in fractions:
            key = (model, f"{fraction:.6f}")
            group = by_model_fraction.get(key, [])
            unique_ids = {str(row["sample_id"]) for row in group}
            window_sets[key] = {str(row["window_start"]) for row in group}
            duplicate_ids = len(group) - len(unique_ids)
            grid[f"{model}/{key[1]}"] = {
                "rows": len(group),
                "unique_sample_ids": len(unique_ids),
                "unique_windows": len(window_sets[key]),
                "duplicate_sample_ids": duplicate_ids,
            }
            if duplicate_ids:
                errors["grid"].append(
                    f"{model}/{key[1]}: {duplicate_ids} duplicate sample_id rows"
                )
    reference = next(iter(window_sets.values()), set())
    for key, windows in window_sets.items():
        if windows != reference:
            errors["grid"].append(
                f"{key[0]}/{key[1]}: window set differs by "
                f"{len(windows ^ reference)} windows"
            )
    checks["row_grid"] = grid

    # Cross-model consistency for the same window and observation fraction.
    # ``sample_id`` is model-specific for some models, so it is not a join key.
    group_rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = (str(row["window_start"]), f"{float(row['observation_fraction']):.6f}")
        group_rows[identity][str(row["model"])] = row
    consistency_fields = (
        "window_start",
        "window_end",
        "observed_at",
        "current_count",
        "actual_final",
        "actual_interval_index",
        "fold",
    )
    consistency_failures = 0
    for identity, model_rows in group_rows.items():
        if len(model_rows) != len(models):
            errors["consistency"].append(
                f"{identity}: {len(model_rows)} models, expected {len(models)}"
            )
            continue
        for field in consistency_fields:
            values = {str(row[field]) for row in model_rows.values()}
            if len(values) > 1:
                consistency_failures += 1
                errors["consistency"].append(f"{identity}: field {field} differs: {sorted(values)}")
    checks["cross_model_consistency"] = {
        "sample_fraction_groups": len(group_rows),
        "failures": consistency_failures,
    }

    # Per-row numeric and structural checks.
    counters = Counter()
    max_sum_error = 0.0
    non_finite_fields: Counter[str] = Counter()
    bad_rows: list[str] = []
    for row in rows:
        row_id = f"{row.get('model')}/{row.get('sample_id')}/{row.get('observation_fraction')}"
        distribution = row.get("probabilities")
        if not isinstance(distribution, list):
            counters["probabilities_not_list"] += 1
            bad_rows.append(row_id)
        else:
            if len(distribution) != len(intervals):
                counters["probabilities_wrong_length"] += 1
                bad_rows.append(row_id)
            if any(not _is_finite(item) for item in distribution):
                counters["probabilities_non_finite"] += 1
                bad_rows.append(row_id)
            if any(
                float(item) < -SUM_TOLERANCE or float(item) > 1 + SUM_TOLERANCE
                for item in distribution
            ):
                counters["probabilities_out_of_range"] += 1
                bad_rows.append(row_id)
            sum_error = abs(sum(float(item) for item in distribution) - 1.0)
            max_sum_error = max(max_sum_error, sum_error)
            if sum_error > SUM_TOLERANCE:
                counters["probabilities_not_normalized"] += 1
                bad_rows.append(row_id)
            top1_index = int(row["top1_index"])
            if not 0 <= top1_index < len(distribution):
                counters["top1_index_out_of_range"] += 1
            elif not _close(distribution[top1_index], max(distribution), 1e-12):
                counters["top1_not_argmax"] += 1
            selected = row["selected_indexes"]
            if not isinstance(selected, list) or len(selected) != 3:
                counters["selected_wrong_length"] += 1
                bad_rows.append(row_id)
            else:
                indexes = [int(item) for item in selected]
                if any(not 0 <= item < len(distribution) for item in indexes):
                    counters["selected_out_of_range"] += 1
                pairs = zip(indexes, indexes[1:], strict=False)
                if any(right != left + 1 for left, right in pairs):
                    counters["selected_not_contiguous"] += 1
                expected_probability = sum(float(distribution[item]) for item in indexes)
                if not _close(row["selected_probability"], expected_probability, 1e-9):
                    counters["selected_probability_mismatch"] += 1
            actual_index = int(row["actual_interval_index"])
            expected_index = interval_index(intervals, int(row["actual_final"]))
            if expected_index != actual_index:
                counters["actual_interval_index_mismatch"] += 1
                errors["actual_interval_index"].append(
                    f"{row_id}: actual_final={row['actual_final']} "
                    f"index={actual_index} expected={expected_index}"
                )
            expected_hit = int(actual_index in [int(item) for item in row["selected_indexes"]])
            if int(row["selected_hit"]) != expected_hit:
                counters["selected_hit_mismatch"] += 1
            lower, upper = row["central_80"]
            if not (_is_finite(lower) and _is_finite(upper)) or int(lower) > int(upper):
                counters["central_80_invalid"] += 1
            for field in (
                "expected_final",
                "stddev_final",
                "effective_sample_size",
                "source_count_delta",
                "selected_probability",
                "decision_probability",
            ):
                if field in row and row[field] is not None and not _is_finite(row[field]):
                    counters[f"{field}_non_finite"] += 1
            if row.get("selected_lower_probability") is not None and not _is_finite(
                row["selected_lower_probability"]
            ):
                counters["selected_lower_probability_non_finite"] += 1
            for path in non_finite_paths(row.get("calibration", {})):
                non_finite_fields[f"calibration.{path}"] += 1

    checks["rows"] = {
        "total": len(rows),
        "counters": dict(counters),
        "max_abs_probability_sum_error": max_sum_error,
        "bad_row_examples": bad_rows[:10],
    }

    fold_windows: dict[int, dict[str, Any]] = {}
    for row in rows:
        fold = int(row["fold"])
        entry = fold_windows.setdefault(
            fold, {"window_starts": set(), "window_ends": set()}
        )
        entry["window_starts"].add(str(row["window_start"]))
        entry["window_ends"].add(str(row["window_end"]))
    checks["fold_windows"] = {
        str(fold): {
            "first_window_start": min(entry["window_starts"]),
            "last_window_end": max(entry["window_ends"]),
            "windows": len(entry["window_starts"]),
        }
        for fold, entry in sorted(fold_windows.items())
    }

    checks["identifier_uniqueness"] = {
        "unique_sample_ids": len({str(row["sample_id"]) for row in rows}),
        "unique_sample_ids_by_model": {
            model: len({str(row["sample_id"]) for row in rows if row["model"] == model})
            for model in models
        },
        "unique_windows": len({str(row["window_start"]) for row in rows}),
        "unique_window_fraction_pairs": len(
            {
                (str(row["window_start"]), f"{float(row['observation_fraction']):.6f}")
                for row in rows
            }
        ),
        "folds": dict(Counter(int(row["fold"]) for row in rows)),
    }

    checks["errors"] = {key: value[:20] for key, value in errors.items()}

    # Metrics per model/fraction, recomputed from raw rows.
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    fold_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for model in models:
        metrics[model] = {}
        fold_metrics[model] = {}
        for fraction in fractions:
            key = f"{fraction:.6f}"
            group = by_model_fraction[(model, key)]
            metrics[model][key] = recompute_metrics(group, intervals)
            fold_metrics[model][key] = {}
            for fold in sorted({int(row["fold"]) for row in group}):
                fold_rows = [row for row in group if int(row["fold"]) == fold]
                fold_metrics[model][key][str(fold)] = recompute_metrics(fold_rows, intervals)
    checks["fold_metrics"] = fold_metrics
    return checks, metrics


def validate_weights(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    counters: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    non_finite: Counter[str] = Counter()
    for row in rows:
        by_type[str(row.get("type"))] += 1
        for path_name in non_finite_paths(row):
            non_finite[path_name] += 1
        if row.get("type") == "prediction_weights":
            weights = row.get("weights")
            finals = row.get("finals")
            if not isinstance(weights, list) or not isinstance(finals, list):
                counters["prediction_weights_not_lists"] += 1
                continue
            if len(weights) != len(finals):
                counters["prediction_weights_length_mismatch"] += 1
            if any(not _is_finite(item) or float(item) < 0 for item in weights):
                counters["prediction_weights_invalid"] += 1
                continue
            total = sum(float(item) for item in weights)
            if total <= 0 or abs(total - 1.0) > 1e-6:
                counters["prediction_weights_not_normalized"] += 1
            effective = 1.0 / sum(float(item) ** 2 for item in weights)
            reported = row.get("effective_sample_size")
            if _is_finite(reported) and not _close(effective, reported, 1e-6):
                counters["effective_sample_size_mismatch"] += 1
        elif row.get("type") == "calibration_weights":
            alpha = row.get("beta_alpha")
            beta = row.get("beta_beta")
            temperature = row.get("temperature")
            parameters = (
                ("beta_alpha", alpha),
                ("beta_beta", beta),
                ("temperature", temperature),
            )
            for name, value in parameters:
                if value is not None and (not _is_finite(value) or float(value) <= 0):
                    counters[f"{name}_invalid"] += 1
        elif row.get("type") == "fitted_model":
            seasonality = row.get("baseline", {}).get("seasonality")
            if seasonality is not None and (
                not isinstance(seasonality, list)
                or len(seasonality) != 168
                or any(not _is_finite(item) or float(item) <= 0 for item in seasonality)
            ):
                counters["seasonality_invalid"] += 1
            residuals = row.get("baseline", {}).get("conformal_residuals")
            if residuals is not None and (
                not isinstance(residuals, list) or any(not _is_finite(item) for item in residuals)
            ):
                counters["conformal_residuals_invalid"] += 1
    return {
        "rows": len(rows),
        "by_type": dict(by_type),
        "counters": dict(counters),
        "non_finite_paths": dict(non_finite.most_common(20)),
    }


def fold_stability(
    fold_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Summarize per-fold variability and sign consistency vs recent_rate."""

    directions = {
        "mae": -1.0,
        "log_loss": -1.0,
        "brier_score": -1.0,
        "ranked_probability_score": -1.0,
        "selected_band_hit_rate": 1.0,
        "selected_band_ece": -1.0,
    }
    baseline = fold_metrics.get("recent_rate", {})
    stability: dict[str, dict[str, dict[str, Any]]] = {}
    for model, fractions in fold_metrics.items():
        stability[model] = {}
        for fraction, fold_values in fractions.items():
            fold_keys = sorted(set(fold_values) & set(baseline.get(fraction, {})))
            entry: dict[str, Any] = {}
            for metric, direction in directions.items():
                values = [float(fold_values[key][metric]) for key in fold_keys]
                base_values = [
                    float(baseline[fraction][key][metric]) for key in fold_keys
                ]
                improvements = [
                    (value - base) * direction
                    for base, value in zip(base_values, values, strict=True)
                ]
                entry[metric] = {
                    "folds": fold_keys,
                    "by_fold": values,
                    "range": max(values) - min(values) if values else None,
                    "improvement_by_fold": improvements,
                    "improved_folds": sum(item > 0 for item in improvements),
                    "compared_folds": len(improvements),
                }
            stability[model][fraction] = entry
    return stability


def compare_with_summary(
    metrics: dict[str, dict[str, dict[str, Any]]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checked = 0
    summary_metrics = summary.get("metrics", {})
    fields = (
        "samples",
        "mae",
        "rmse",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "coverage_80",
        "top1_accuracy",
        "selected_band_hit_rate",
        "selected_band_mean_probability",
        "selected_band_calibration_gap",
        "selected_band_ece",
        "selected_band_brier",
    )
    for model, fractions in metrics.items():
        for fraction, values in fractions.items():
            reported = summary_metrics.get(model, {}).get(fraction)
            if reported is None:
                mismatches.append(
                    {"model": model, "fraction": fraction, "field": "<missing>", "reported": None,
                     "recomputed": values}
                )
                continue
            for field in fields:
                if field not in reported:
                    mismatches.append(
                        {"model": model, "fraction": fraction, "field": field,
                         "reported": None, "recomputed": values.get(field)}
                    )
                    continue
                checked += 1
                if not _close(values[field], reported[field]):
                    mismatches.append(
                        {"model": model, "fraction": fraction, "field": field,
                         "reported": reported[field], "recomputed": values[field]}
                    )
            # Group sample counts should partition the full sample set.
            for group in ("by_regime", "by_current_count_quartile", "by_start_weekday"):
                group_metrics = reported.get(group)
                if isinstance(group_metrics, dict) and group_metrics:
                    group_samples = sum(
                        int(item.get("samples", 0))
                        for item in group_metrics.values()
                        if isinstance(item, dict)
                    )
                    checked += 1
                    if group_samples != int(reported.get("samples", -1)):
                        mismatches.append(
                            {"model": model, "fraction": fraction, "field": f"{group}.samples",
                             "reported": reported.get("samples"), "recomputed": group_samples}
                        )
    return {"checked_values": checked, "mismatches": mismatches}


def compare_baseline(
    metrics: dict[str, dict[str, dict[str, Any]]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    baseline = metrics.get("recent_rate")
    if baseline is None:
        return {"available": False}
    reported = summary.get("vs_recent_rate", {})
    mismatches: list[dict[str, Any]] = []
    pairs = {
        "mae_improvement": ("mae", -1.0),
        "log_loss_improvement": ("log_loss", -1.0),
        "brier_improvement": ("brier_score", -1.0),
        "rps_improvement": ("ranked_probability_score", -1.0),
        "selected_ece_improvement": ("selected_band_ece", -1.0),
        "selected_hit_rate_change": ("selected_band_hit_rate", 1.0),
    }
    checked = 0
    for model, fractions in metrics.items():
        if model == "recent_rate":
            continue
        for fraction, values in fractions.items():
            baseline_values = baseline[fraction]
            for name, (field, direction) in pairs.items():
                expected = (float(values[field]) - float(baseline_values[field])) * direction
                reported_value = reported.get(model, {}).get(fraction, {}).get(name)
                checked += 1
                if reported_value is None or not _close(expected, reported_value):
                    mismatches.append(
                        {"model": model, "fraction": fraction, "metric": name,
                         "reported": reported_value, "recomputed": expected}
                    )
    return {"available": True, "checked_values": checked, "mismatches": mismatches}


def write_metrics_csv(
    path: Path,
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> None:
    baseline = metrics.get("recent_rate", {})
    fields = [
        "model", "observation_fraction", "samples", "mae", "rmse", "log_loss",
        "brier_score", "ranked_probability_score", "coverage_80", "top1_accuracy",
        "selected_band_hit_rate", "selected_band_mean_probability",
        "selected_band_calibration_gap", "selected_band_ece", "selected_band_brier",
        "raw_band_mean_probability", "raw_band_calibration_gap", "raw_band_ece",
        "raw_band_brier",
        "mae_vs_recent_rate", "log_loss_vs_recent_rate", "brier_vs_recent_rate",
        "rps_vs_recent_rate", "coverage80_vs_recent_rate", "band_coverage_vs_recent_rate",
        "ece_vs_recent_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, fractions in metrics.items():
            for fraction, values in fractions.items():
                base = baseline.get(fraction, {})
                row = {
                    "model": model,
                    "observation_fraction": fraction,
                    **{field: values[field] for field in fields if field in values},
                }
                row["mae_vs_recent_rate"] = (
                    values["mae"] - base["mae"] if base else None
                )
                row["log_loss_vs_recent_rate"] = (
                    values["log_loss"] - base["log_loss"] if base else None
                )
                row["brier_vs_recent_rate"] = (
                    values["brier_score"] - base["brier_score"] if base else None
                )
                row["rps_vs_recent_rate"] = (
                    values["ranked_probability_score"] - base["ranked_probability_score"]
                    if base else None
                )
                row["coverage80_vs_recent_rate"] = (
                    values["coverage_80"] - base["coverage_80"] if base else None
                )
                row["band_coverage_vs_recent_rate"] = (
                    values["selected_band_hit_rate"] - base["selected_band_hit_rate"]
                    if base else None
                )
                row["ece_vs_recent_rate"] = (
                    values["selected_band_ece"] - base["selected_band_ece"] if base else None
                )
                writer.writerow({field: row.get(field) for field in fields})


def write_fold_csv(
    path: Path,
    fold_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    fields = [
        "model",
        "observation_fraction",
        "fold",
        "samples",
        "mae",
        "rmse",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "coverage_80",
        "top1_accuracy",
        "selected_band_hit_rate",
        "selected_band_mean_probability",
        "selected_band_calibration_gap",
        "selected_band_ece",
        "selected_band_brier",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, fractions in fold_metrics.items():
            for fraction, folds in fractions.items():
                for fold, values in folds.items():
                    writer.writerow(
                        {
                            "model": model,
                            "observation_fraction": fraction,
                            "fold": fold,
                            **{field: values[field] for field in fields if field in values},
                        }
                    )


def render_fold_values(
    entry: dict[str, Any], metric: str, digits: int, percent: bool = False
) -> str:
    items = entry[metric]["by_fold"]
    rendered = (
        f"{item:.{digits}%}" if percent else f"{item:.{digits}f}" for item in items
    )
    return "/".join(rendered)


def write_markdown(
    path: Path,
    *,
    experiment: Path,
    checks: dict[str, Any],
    summary_check: dict[str, Any],
    baseline_check: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, Any]]],
    stability: dict[str, dict[str, dict[str, Any]]],
    fold_windows: dict[str, dict[str, Any]],
) -> None:
    counters = checks["rows"]["counters"]
    failures = []
    if checks["missing_fields"]:
        failures.append("missing fields")
    if any(item["duplicate_sample_ids"] for item in checks["row_grid"].values()):
        failures.append("duplicate sample_id rows")
    if checks["errors"].get("grid"):
        failures.append("grid window-set differences")
    if checks["cross_model_consistency"]["failures"]:
        failures.append("cross-model consistency")
    for key, value in counters.items():
        if value:
            failures.append(key)
    if summary_check["mismatches"]:
        failures.append("summary.json mismatch")
    if baseline_check.get("mismatches"):
        failures.append("vs_recent_rate mismatch")

    lines = [
        f"# Experiment acceptance: {experiment.name}",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
        f"**Overall: {'PASS' if not failures else 'FAIL'}**",
        "",
    ]
    if failures:
        lines += ["Failures:", *[f"- {item}" for item in failures], ""]
    lines += [
        "## Validation",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Missing fields | {sum(checks['missing_fields'].values())} |",
        f"| Rows | {checks['rows']['total']} |",
        f"| Max probability-sum error | {checks['rows']['max_abs_probability_sum_error']:.3e} |",
        f"| Cross-model inconsistencies | {checks['cross_model_consistency']['failures']} |",
        f"| Summary values checked | {summary_check['checked_values']} |",
        f"| Summary mismatches | {len(summary_check['mismatches'])} |",
        f"| Baseline values checked | {baseline_check.get('checked_values', 0)} |",
        f"| Baseline mismatches | {len(baseline_check.get('mismatches', []))} |",
        "",
        "Counters:",
        "",
        "```json",
        json.dumps(counters, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Metrics by observation point",
        "",
        "Calibration and band probability use `decision_probability` (the probability the "
        "strategy acts on). For every model except `adaptive_beta` it equals "
        "`selected_probability`; `adaptive_beta` applies a Beta recalibration. Raw values are "
        "kept in `metrics.csv` and `report.json`.",
        "",
    ]
    baseline = metrics.get("recent_rate", {})
    for fraction in sorted({key for model in metrics for key in metrics[model]}, key=float):
        lines += [
            f"### Observed fraction {float(fraction):.2%}",
            "",
            "| Model | n | MAE | LogLoss | Brier | RPS | 80% cov | 3-band cov | Cal. gap | ECE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        base = baseline.get(fraction, {})
        for model, fractions in metrics.items():
            values = fractions.get(fraction)
            if values is None:
                continue
            lines.append(
                f"| {model} | {int(values['samples'])} | "
                f"{values['mae']:.2f} | {values['log_loss']:.3f} | "
                f"{values['brier_score']:.4f} | {values['ranked_probability_score']:.4f} | "
                f"{values['coverage_80']:.1%} | {values['selected_band_hit_rate']:.1%} | "
                f"{values['selected_band_calibration_gap']:+.3f} | "
                f"{values['selected_band_ece']:.3f} |"
            )
        if base:
            lines += [
                "",
                "Delta vs `recent_rate` (negative MAE/log-loss/Brier/RPS/ECE is better; "
                "positive 3-band coverage is better):",
                "",
                "| Model | ΔMAE | ΔLogLoss | ΔBrier | ΔRPS | Δ80% cov | Δ3-band cov | ΔECE |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for model, fractions in metrics.items():
                if model == "recent_rate":
                    continue
                values = fractions.get(fraction)
                if values is None:
                    continue
                delta_rps = (
                    values["ranked_probability_score"]
                    - base["ranked_probability_score"]
                )
                delta_coverage = values["coverage_80"] - base["coverage_80"]
                delta_band = (
                    values["selected_band_hit_rate"]
                    - base["selected_band_hit_rate"]
                )
                lines.append(
                    f"| {model} | "
                    f"{values['mae'] - base['mae']:+.2f} | "
                    f"{values['log_loss'] - base['log_loss']:+.3f} | "
                    f"{values['brier_score'] - base['brier_score']:+.4f} | "
                    f"{delta_rps:+.4f} | {delta_coverage:+.1%} | "
                    f"{delta_band:+.1%} | "
                    f"{values['selected_band_ece'] - base['selected_band_ece']:+.3f} |"
                )
        lines.append("")

    lines += [
        "## Fold sensitivity",
        "",
        "The protocol uses three contiguous 60-day folds. A model is only stable if its "
        "advantage over `recent_rate` has the same sign in every fold.",
        "",
        "| Fold | Window |",
        "| --- | --- |",
    ]
    for fold, window in fold_windows.items():
        lines.append(
            f"| {fold} | {window['first_window_start']} → {window['last_window_end']} |"
        )
    lines.append("")
    highlight_models = (
        "recent_rate",
        "adaptive_uncorrected",
        "adaptive_uncorrected_lag2pct",
        "adaptive_reconciled_lag2pct",
        "gbm_conformal",
        "negative_binomial",
        "adaptive_beta",
    )
    for fraction in sorted(
        {key for model in metrics for key in metrics[model]}, key=float
    ):
        lines += [
            f"### Fold sensitivity at {float(fraction):.2%}",
            "",
            "| Model | MAE f0/f1/f2 | LogLoss f0/f1/f2 | 3-band f0/f1/f2 | ECE f0/f1/f2 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for model in highlight_models:
            entry = stability.get(model, {}).get(fraction)
            if entry is None:
                continue
            lines.append(
                f"| {model} | {render_fold_values(entry, 'mae', 1)} | "
                f"{render_fold_values(entry, 'log_loss', 3)} | "
                f"{render_fold_values(entry, 'selected_band_hit_rate', 1, True)} | "
                f"{render_fold_values(entry, 'selected_band_ece', 3)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Experiment directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <experiment>/acceptance)",
    )
    args = parser.parse_args(argv)

    experiment: Path = args.experiment
    output: Path = args.output or experiment / "acceptance"
    manifest = load_json(experiment / "manifest.json")
    summary = load_json(experiment / "summary.json")
    intervals = manifest["intervals"]
    models = [str(item) for item in manifest["models"]]
    fractions = [float(item) for item in manifest["observation_fractions"]]
    predictions = load_jsonl(experiment / "predictions.jsonl")

    checks, metrics = validate_predictions(
        predictions,
        models=models,
        fractions=fractions,
        intervals=intervals,
    )
    weights_check = validate_weights(experiment / "weights.jsonl")
    summary_check = compare_with_summary(metrics, summary)
    baseline_check = compare_baseline(metrics, summary)
    stability = fold_stability(checks["fold_metrics"])

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": str(experiment),
        "generated_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "weights": weights_check,
        "summary_match": summary_check,
        "baseline_match": baseline_check,
        "metrics": metrics,
        "fold_stability": stability,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_metrics_csv(output / "metrics.csv", metrics)
    write_fold_csv(output / "metrics_by_fold.csv", checks["fold_metrics"])
    write_markdown(
        output / "summary.md",
        experiment=experiment,
        checks=checks,
        summary_check=summary_check,
        baseline_check=baseline_check,
        metrics=metrics,
        stability=stability,
        fold_windows=checks["fold_windows"],
    )

    failures = []
    if checks["missing_fields"]:
        failures.append(f"missing fields: {dict(checks['missing_fields'])}")
    if any(item["duplicate_sample_ids"] for item in checks["row_grid"].values()):
        failures.append("duplicate sample_id rows in the model/fraction grid")
    if checks["errors"].get("grid"):
        failures.append(f"grid window-set differences: {len(checks['errors']['grid'])}")
    if checks["cross_model_consistency"]["failures"]:
        failures.append(
            "cross-model inconsistencies: "
            f"{checks['cross_model_consistency']['failures']}"
        )
    if any(checks["rows"]["counters"].values()):
        failures.append(f"row checks: {checks['rows']['counters']}")
    if summary_check["mismatches"]:
        failures.append(f"summary mismatches: {len(summary_check['mismatches'])}")
    if baseline_check.get("mismatches"):
        failures.append(f"baseline mismatches: {len(baseline_check['mismatches'])}")
    if weights_check["counters"]:
        failures.append(f"weight checks: {weights_check['counters']}")

    print(f"experiment: {experiment}")
    print(f"rows: {checks['rows']['total']}")
    print(f"max |sum(prob) - 1|: {checks['rows']['max_abs_probability_sum_error']:.3e}")
    print(f"summary values checked: {summary_check['checked_values']}, "
          f"mismatches: {len(summary_check['mismatches'])}")
    print(f"baseline values checked: {baseline_check.get('checked_values', 0)}, "
          f"mismatches: {len(baseline_check.get('mismatches', []))}")
    print(f"artifact: {output}")
    if failures:
        print("STATUS: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

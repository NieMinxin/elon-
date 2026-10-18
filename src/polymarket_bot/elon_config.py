from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from polymarket_bot.config import ConfigError


def _decimal(env: Mapping[str, str], name: str, default: str) -> Decimal:
    raw = env.get(name, default).strip()
    try:
        return Decimal(raw)
    except InvalidOperation as error:
        raise ConfigError(f"{name} must be a decimal, got {raw!r}") from error


def _integer(env: Mapping[str, str], name: str, default: str) -> int:
    raw = env.get(name, default).strip()
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from error


def _boolean(env: Mapping[str, str], name: str, default: str = "false") -> bool:
    raw = env.get(name, default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class ElonArbConfig:
    event_slug: str
    xtracker_handle: str
    forecast_strategy: str
    history_days: int
    prior_strength_hours: Decimal
    min_backtest_coverage: Decimal
    max_backtest_mae: Decimal
    min_entry_edge: Decimal
    min_exit_advantage: Decimal
    min_fair_probability: Decimal
    order_amount_usd: Decimal
    max_total_cost_usd: Decimal
    max_daily_buy_usd: Decimal
    max_positions: int
    max_slippage: Decimal
    min_entry_elapsed_minutes: int
    no_new_position_minutes: int
    poll_interval_seconds: int
    trade_cooldown_seconds: int
    history_cache_path: Path
    state_path: Path
    audit_path: Path
    drawdown_alert_step: Decimal
    deterioration_warning: Decimal
    max_strategy_loss: Decimal
    label_finalization_delay_minutes: int
    four_model_alerts: bool
    four_model_python: str | None
    interval_config_path: Path
    gru_original_run: Path
    gru_v3_run: Path
    gru_v4_run: Path
    gru_fold: int
    gru_device: str
    four_model_output_path: Path
    four_model_timeout_seconds: int
    dingtalk_webhook: str | None
    dingtalk_secret: str | None
    enable_live: bool
    private_key: str | None
    wallet_address: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ElonArbConfig:
        values = os.environ if env is None else env
        config = cls(
            event_slug=values.get(
                "ELON_EVENT_SLUG",
                "elon-musk-of-tweets-october-13-october-20-2026",
            ).strip(),
            xtracker_handle=values.get("ELON_XTRACKER_HANDLE", "elonmusk").strip(),
            forecast_strategy=values.get("ELON_FORECAST_STRATEGY", "route").strip().lower(),
            history_days=_integer(values, "ELON_HISTORY_DAYS", "365"),
            prior_strength_hours=_decimal(values, "ELON_PRIOR_STRENGTH_HOURS", "168"),
            min_backtest_coverage=_decimal(values, "ELON_MIN_BACKTEST_COVERAGE", "0.65"),
            max_backtest_mae=_decimal(values, "ELON_MAX_BACKTEST_MAE", "50"),
            min_entry_edge=_decimal(values, "ELON_MIN_ENTRY_EDGE", "0.035"),
            min_exit_advantage=_decimal(values, "ELON_MIN_EXIT_ADVANTAGE", "0.015"),
            min_fair_probability=_decimal(values, "ELON_MIN_FAIR_PROBABILITY", "0.01"),
            order_amount_usd=_decimal(values, "ELON_ORDER_AMOUNT_USD", "20"),
            max_total_cost_usd=_decimal(values, "ELON_MAX_TOTAL_COST_USD", "100"),
            max_daily_buy_usd=_decimal(values, "ELON_MAX_DAILY_BUY_USD", "100"),
            max_positions=_integer(values, "ELON_MAX_POSITIONS", "3"),
            max_slippage=_decimal(values, "ELON_MAX_SLIPPAGE", "0.01"),
            min_entry_elapsed_minutes=_integer(values, "ELON_MIN_ENTRY_ELAPSED_MINUTES", "1440"),
            no_new_position_minutes=_integer(values, "ELON_NO_NEW_POSITION_MINUTES", "30"),
            poll_interval_seconds=_integer(values, "ELON_POLL_INTERVAL_SECONDS", "120"),
            trade_cooldown_seconds=_integer(values, "ELON_TRADE_COOLDOWN_SECONDS", "300"),
            history_cache_path=Path(
                values.get("ELON_HISTORY_CACHE_PATH", "elon-history-cache.json")
            ),
            state_path=Path(values.get("ELON_STATE_PATH", "elon-arb-state.json")),
            audit_path=Path(values.get("ELON_AUDIT_PATH", "elon-arb-audit.jsonl")),
            drawdown_alert_step=_decimal(values, "ELON_DRAWDOWN_ALERT_STEP", "0.05"),
            deterioration_warning=_decimal(values, "ELON_DETERIORATION_WARNING", "0.12"),
            max_strategy_loss=_decimal(values, "ELON_MAX_STRATEGY_LOSS", "0.25"),
            label_finalization_delay_minutes=_integer(
                values, "ELON_LABEL_FINALIZATION_DELAY_MINUTES", "60"
            ),
            four_model_alerts=_boolean(values, "ELON_FOUR_MODEL_ALERTS"),
            four_model_python=values.get("ELON_FOUR_MODEL_PYTHON") or None,
            interval_config_path=Path(
                values.get(
                    "ELON_INTERVAL_CONFIG_PATH",
                    "config/elon-musk-oct-13-20-2026-intervals.json",
                )
            ),
            gru_original_run=Path(
                values.get("ELON_GRU_ORIGINAL_RUN", "models/gru-original")
            ),
            gru_v3_run=Path(values.get("ELON_GRU_V3_RUN", "models/gru-v3")),
            gru_v4_run=Path(values.get("ELON_GRU_V4_RUN", "models/gru-v4")),
            gru_fold=_integer(values, "ELON_GRU_FOLD", "2"),
            gru_device=values.get("ELON_GRU_DEVICE", "auto").strip().lower(),
            four_model_output_path=Path(
                values.get(
                    "ELON_FOUR_MODEL_OUTPUT_PATH",
                    "runtime/four-model-comparison.json",
                )
            ),
            four_model_timeout_seconds=_integer(
                values, "ELON_FOUR_MODEL_TIMEOUT_SECONDS", "120"
            ),
            dingtalk_webhook=values.get("DINGTALK_WEBHOOK") or None,
            dingtalk_secret=values.get("DINGTALK_SECRET") or None,
            enable_live=_boolean(values, "ELON_ENABLE_LIVE"),
            private_key=values.get("POLYMARKET_PRIVATE_KEY") or None,
            wallet_address=values.get("POLYMARKET_WALLET_ADDRESS") or None,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.event_slug or not self.xtracker_handle:
            raise ConfigError("ELON_EVENT_SLUG and ELON_XTRACKER_HANDLE are required")
        if self.forecast_strategy not in {"route", "adaptive"}:
            raise ConfigError("ELON_FORECAST_STRATEGY must be route or adaptive")
        if not 30 <= self.history_days <= 730:
            raise ConfigError("ELON_HISTORY_DAYS must be between 30 and 730")
        if self.prior_strength_hours <= 0:
            raise ConfigError("ELON_PRIOR_STRENGTH_HOURS must be greater than 0")
        if not Decimal("0") <= self.min_backtest_coverage <= Decimal("1"):
            raise ConfigError("ELON_MIN_BACKTEST_COVERAGE must be between 0 and 1")
        if self.max_backtest_mae <= 0:
            raise ConfigError("ELON_MAX_BACKTEST_MAE must be greater than 0")
        for name, value in (
            ("ELON_MIN_ENTRY_EDGE", self.min_entry_edge),
            ("ELON_MIN_EXIT_ADVANTAGE", self.min_exit_advantage),
            ("ELON_MIN_FAIR_PROBABILITY", self.min_fair_probability),
            ("ELON_MAX_SLIPPAGE", self.max_slippage),
        ):
            if not Decimal("0") <= value <= Decimal("0.25"):
                raise ConfigError(f"{name} must be between 0 and 0.25")
        for name, value in (
            ("ELON_ORDER_AMOUNT_USD", self.order_amount_usd),
            ("ELON_MAX_TOTAL_COST_USD", self.max_total_cost_usd),
            ("ELON_MAX_DAILY_BUY_USD", self.max_daily_buy_usd),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be greater than 0")
        if self.max_positions < 1:
            raise ConfigError("ELON_MAX_POSITIONS must be at least 1")
        if self.poll_interval_seconds < 15:
            raise ConfigError("ELON_POLL_INTERVAL_SECONDS must be at least 15")
        if self.gru_fold < 0:
            raise ConfigError("ELON_GRU_FOLD cannot be negative")
        if self.gru_device not in {"auto", "cpu", "cuda"}:
            raise ConfigError("ELON_GRU_DEVICE must be auto, cpu, or cuda")
        if self.four_model_timeout_seconds < 1:
            raise ConfigError("ELON_FOUR_MODEL_TIMEOUT_SECONDS must be positive")
        if self.four_model_alerts and not self.four_model_python:
            raise ConfigError(
                "ELON_FOUR_MODEL_PYTHON is required when four-model alerts are enabled"
            )
        if (
            self.trade_cooldown_seconds < 0
            or self.no_new_position_minutes < 0
            or self.min_entry_elapsed_minutes < 0
            or self.label_finalization_delay_minutes < 0
        ):
            raise ConfigError("Cooldown values cannot be negative")
        for name, value in (
            ("ELON_DRAWDOWN_ALERT_STEP", self.drawdown_alert_step),
            ("ELON_DETERIORATION_WARNING", self.deterioration_warning),
            ("ELON_MAX_STRATEGY_LOSS", self.max_strategy_loss),
        ):
            if not Decimal("0") < value < Decimal("1"):
                raise ConfigError(f"{name} must be between 0 and 1")

    def require_live_credentials(self) -> tuple[str, str]:
        if not self.private_key or not self.wallet_address:
            raise ConfigError(
                "POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS are required in live mode"
            )
        return self.private_key, self.wallet_address

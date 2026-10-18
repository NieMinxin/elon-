from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from polymarket_bot.database import MySQLConfig, PostRepository
from polymarket_bot.elon_config import ElonArbConfig
from polymarket_bot.elon_monitor import PortfolioAssessment, assess_portfolio
from polymarket_bot.elon_state import ElonArbState
from polymarket_bot.elon_strategy import (
    ArbActionKind,
    ArbDecision,
    MarketQuote,
    PositionView,
    decide_action,
    parse_interval,
)
from polymarket_bot.forecast import (
    AdaptiveWindowModel,
    BacktestResult,
    CountForecast,
    correct_baseline_forecast,
    walk_forward_adaptive_backtest,
)
from polymarket_bot.model_experiments import GradientBoostedConformalModel
from polymarket_bot.notifier import DingTalkNotifier
from polymarket_bot.strategy import buy_price_cap, sell_price_floor
from polymarket_bot.trader import check_geoblock
from polymarket_bot.xtracker import (
    TrackingSnapshot,
    XTrackerClient,
)

LOGGER = logging.getLogger(__name__)

FOUR_MODEL_LABELS = {
    "route_baseline": "路由基线",
    "original_gru_nll": "原 GRU-NLL",
    "v3_dense_hourly_gru": "V3 密集小时 GRU",
    "v4_calibrated_ensemble": "V4 校准集成",
}


def format_four_model_comparison(payload: dict[str, Any]) -> str:
    """Render a compact DingTalk-safe summary from a retained comparison payload."""
    lines = ["\n\n#### 四模型预测"]
    for key, label in FOUR_MODEL_LABELS.items():
        result = payload["models"][key]
        if key == "route_baseline":
            component = result.get("route_component")
            if component:
                label = f"{label} {component}"
        probabilities = {
            item["label"]: float(item["probability"])
            for item in result["interval_probabilities"]
        }
        band = list(result["three_band"])
        band_detail = " / ".join(
            f"{interval} {probabilities[interval]:.1%}" for interval in band
        )
        central = result["central_80"]
        lines.extend(
            (
                f"- **{label}**：均值 {float(result['expected_final']):.1f}；"
                f"80% 区间 {central[0]}-{central[1]}",
                f"  - {band_detail}；合计 {float(result['three_band_probability']):.1%}",
            )
        )
    observed = payload.get("model_observed_at")
    if observed:
        lines.append(f"\n模型数据截点：{observed}")
    return "\n".join(lines)


class EventTradingClient(Protocol):
    async def get_event(self, *, slug: str) -> Any: ...

    async def get_price(self, *, asset_id: str, side: str) -> Decimal: ...

    async def get_order_books(self, *, asset_ids: list[str]) -> tuple[Any, ...]: ...

    def list_positions(self, **kwargs: Any) -> Any: ...

    async def place_market_order(self, **kwargs: Any) -> Any: ...

    async def wait_for_order_fill_settlement(self, order: Any) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class EventMarket:
    slug: str
    asset_id: str
    condition_id: str
    interval_label: str
    minimum_order_size: Decimal
    fee_rate: Decimal


@dataclass(frozen=True)
class CycleResult:
    snapshot: TrackingSnapshot
    forecast: CountForecast
    forecast_model: str
    quotes: tuple[MarketQuote, ...]
    decision: ArbDecision
    minutes_remaining: Decimal
    assessment: PortfolioAssessment | None = None
    alerts: tuple[str, ...] = ()


class ElonArbSystem:
    def __init__(
        self,
        client: EventTradingClient,
        xtracker: XTrackerClient,
        config: ElonArbConfig,
        *,
        live: bool,
        refresh_history: bool = False,
        notifier: DingTalkNotifier | None = None,
    ) -> None:
        self.client = client
        self.xtracker = xtracker
        self.config = config
        self.live = live
        self.refresh_history = refresh_history
        self.state = ElonArbState.load(config.state_path, config.event_slug)
        self.event_start: datetime | None = None
        self.event_end: datetime | None = None
        self.tracking_id: str | None = None
        self.markets: tuple[EventMarket, ...] = ()
        self.repository = PostRepository(MySQLConfig.from_env())
        tracker_http_client = getattr(xtracker, "client", None)
        self.notifier = notifier
        if self.notifier is None and tracker_http_client is not None:
            self.notifier = DingTalkNotifier(
                tracker_http_client,
                webhook=config.dingtalk_webhook,
                secret=config.dingtalk_secret,
            )
        self.model: AdaptiveWindowModel | None = None
        self.gbm_models: dict[int, GradientBoostedConformalModel] = {}
        self.baseline_forecast: CountForecast | None = None
        self.baseline_reference_forecast: CountForecast | None = None
        self.models_frozen_at_event_start = False
        self.history_post_times: list[datetime] = []

    async def initialize(self) -> None:
        event = await self.client.get_event(slug=self.config.event_slug)
        if event.state.closed or event.state.archived:
            raise RuntimeError("The event is closed or archived")
        starts = [
            market.sports.game_start_time or market.state.start_date
            for market in event.markets
            if market.sports.game_start_time or market.state.start_date
        ]
        ends = [market.state.end_date for market in event.markets if market.state.end_date]
        if not starts or not ends:
            raise RuntimeError("Event markets do not expose a trading window")
        self.event_start = max(starts)
        self.event_end = min(ends)
        self.tracking_id = await self.xtracker.find_tracking(
            self.config.xtracker_handle,
            start=self.event_start,
            end=self.event_end,
        )
        contexts = []
        for market in event.markets:
            if market.state.closed or market.state.accepting_orders is False:
                continue
            if market.condition_id is None or not market.group_item_title:
                continue
            outcome = market.outcomes.yes
            asset = outcome.position_id if market.version == "v2" else outcome.token_id
            if asset is None:
                continue
            fee_schedule = market.trading.fee_schedule
            fee_rate = Decimal(str(fee_schedule.rate)) if fee_schedule is not None else Decimal("0")
            contexts.append(
                EventMarket(
                    slug=market.slug or str(market.id),
                    asset_id=str(asset),
                    condition_id=str(market.condition_id),
                    interval_label=market.group_item_title,
                    minimum_order_size=market.trading.minimum_order_size or Decimal("0"),
                    fee_rate=fee_rate,
                )
            )
        if not contexts:
            raise RuntimeError("No open interval markets were found")
        self.markets = tuple(contexts)
        await self._load_model()

    async def _load_model(self) -> None:
        assert self.event_start is not None
        self.repository.initialize()
        if self.refresh_history:
            refresh_end = min(self.event_start, datetime.now(UTC))
            history_start = refresh_end - timedelta(days=self.config.history_days)
            LOGGER.info(
                "Refreshing %s days of xtracker history into MySQL",
                self.config.history_days,
            )
            posts = await self.xtracker.get_posts(
                self.config.xtracker_handle,
                start=history_start,
                end=refresh_end,
            )
            self.repository.upsert_posts(self.config.xtracker_handle, posts)
        latest = self.repository.latest_created_at(self.config.xtracker_handle)
        if latest is None:
            raise RuntimeError(
                "MySQL contains no training posts; run elon-data backfill before elon-arb"
            )
        model_end = min(self.event_start, latest + timedelta(microseconds=1))
        self._fit_models(model_end)
        self.models_frozen_at_event_start = model_end >= self.event_start

    def _fit_models(self, model_end: datetime) -> None:
        assert self.event_start is not None
        assert self.event_end is not None
        history_start = model_end - timedelta(days=self.config.history_days)
        post_times = self.repository.post_times(
            self.config.xtracker_handle,
            start=history_start,
            end=model_end,
        )
        if not post_times:
            raise RuntimeError(
                "MySQL contains no training posts; run elon-data backfill before elon-arb"
            )
        self.history_post_times = post_times
        earliest = min(post_times)
        first_complete_day = (earliest + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        effective_start = max(history_start, first_complete_day)
        assert self.event_end is not None
        self.model = AdaptiveWindowModel.fit(
            post_times,
            start=effective_start,
            end=model_end,
            event_duration=self.event_end - self.event_start,
        )
        LOGGER.info(
            "Adaptive realtime model trained from MySQL: posts=%s history=%s to %s",
            self.model.history_posts,
            self.model.history_start.date(),
            self.model.history_end.isoformat(),
        )
        if self.config.forecast_strategy == "route":
            duration_hours = (self.event_end - self.event_start).total_seconds() / 3600
            self.gbm_models = {
                hours: GradientBoostedConformalModel.fit(
                    post_times,
                    start=effective_start,
                    end=model_end,
                    event_duration=self.event_end - self.event_start,
                    observation_fraction=hours / duration_hours,
                )
                for hours in (0, 24, 42, 84)
            }
            self.baseline_forecast = self.gbm_models[0].forecast(
                current_count=0,
                event_start=self.event_start,
                event_end=self.event_end,
                now=self.event_start,
                observed_post_times=[],
            )
            self.baseline_reference_forecast = self.model.forecast(
                current_count=0,
                event_start=self.event_start,
                event_end=self.event_end,
                now=self.event_start,
                observed_post_times=[],
            )
            LOGGER.info(
                "Routed forecast models trained: baseline=GBM-0h checkpoints=%s "
                "adaptive_switch=126h",
                sorted(self.gbm_models),
            )

    async def _freeze_models_at_event_start(self, current_time: datetime) -> None:
        assert self.event_start is not None
        if self.models_frozen_at_event_start or current_time < self.event_start:
            return
        posts = await self.xtracker.get_posts(
            self.config.xtracker_handle,
            start=self.event_start - timedelta(days=7),
            end=self.event_start,
        )
        self.repository.upsert_posts(self.config.xtracker_handle, posts)
        self._fit_models(self.event_start)
        self.models_frozen_at_event_start = True
        LOGGER.info("Baseline P0 frozen at event start %s", self.event_start.isoformat())

    async def _finalize_event_outcome(
        self, snapshot: TrackingSnapshot, current_time: datetime
    ) -> tuple[str, str] | None:
        assert self.event_start is not None
        assert self.event_end is not None
        ready_at = self.event_end + timedelta(
            minutes=self.config.label_finalization_delay_minutes
        )
        if current_time < ready_at:
            return None
        existing = self.repository.event_outcome(self.config.event_slug)
        if existing is not None and existing["final_count"] == snapshot.total:
            return None

        posts = await self.xtracker.get_posts(
            self.config.xtracker_handle,
            start=self.event_start,
            end=self.event_end,
        )
        endpoint_ids = {item.platform_id for item in posts}
        self.repository.upsert_posts(self.config.xtracker_handle, posts)
        mysql_ids = self.repository.post_ids(
            self.config.xtracker_handle,
            start=self.event_start,
            end=self.event_end,
        )
        winning = None
        for market in self.markets:
            interval = parse_interval(market.interval_label)
            if snapshot.total >= interval.lower and (
                interval.upper is None or snapshot.total <= interval.upper
            ):
                winning = interval
                break
        if winning is None:
            raise RuntimeError(f"Final count {snapshot.total} is outside all event intervals")
        status = self.repository.record_event_outcome(
            {
                "event_slug": self.config.event_slug,
                "tracking_id": snapshot.tracking_id,
                "handle": self.config.xtracker_handle,
                "event_start": self.event_start,
                "event_end": self.event_end,
                "final_count": snapshot.total,
                "winning_interval_label": winning.label,
                "winning_lower": winning.lower,
                "winning_upper": winning.upper,
                "label_source": "xtracker_tracking",
                "endpoint_total": len(endpoint_ids),
                "mysql_total": len(mysql_ids),
                "tracking_posts_delta": snapshot.total - len(endpoint_ids),
                "endpoint_mysql_delta": len(endpoint_ids) - len(mysql_ids),
                "observed_at": current_time,
            }
        )
        title = "盘口最终标签已记录" if status == "inserted" else "盘口最终标签已修订"
        message = (
            f"{self.config.event_slug} 最终计数 {snapshot.total}，"
            f"命中区间 {winning.label}；posts端点 {len(endpoint_ids)}，"
            f"MySQL {len(mysql_ids)}，标签状态 {status}。"
        )
        return title, message

    def _forecast_for_time(
        self,
        *,
        current_count: int,
        current_time: datetime,
        observed_post_times: list[datetime],
    ) -> tuple[str, CountForecast]:
        assert self.model is not None
        assert self.event_start is not None
        assert self.event_end is not None
        elapsed_hours = max(0.0, (current_time - self.event_start).total_seconds() / 3600)
        if self.config.forecast_strategy == "route" and elapsed_hours < 24:
            assert self.baseline_forecast is not None
            assert self.baseline_reference_forecast is not None
            realtime = self.model.forecast(
                current_count=current_count,
                event_start=self.event_start,
                event_end=self.event_end,
                now=current_time,
                observed_post_times=observed_post_times,
            )
            strength = math.sqrt(min(1.0, elapsed_hours / 24))
            forecast = correct_baseline_forecast(
                self.baseline_forecast,
                self.baseline_reference_forecast,
                realtime,
                strength=strength,
            )
            return "gbm_conformal_0h+adaptive_realtime", forecast
        if self.config.forecast_strategy == "route" and elapsed_hours < 126:
            checkpoint = max(
                (hours for hours in self.gbm_models if hours <= elapsed_hours),
                default=0,
            )
            model = self.gbm_models[checkpoint]
            name = f"gbm_conformal_{checkpoint}h"
            forecast = model.forecast(
                current_count=current_count,
                event_start=self.event_start,
                event_end=self.event_end,
                now=current_time,
                observed_post_times=observed_post_times,
            )
            return name, forecast
        forecast = self.model.forecast(
            current_count=current_count,
            event_start=self.event_start,
            event_end=self.event_end,
            now=current_time,
            observed_post_times=observed_post_times,
        )
        return "adaptive_reconciled_126h", forecast

    def backtest(self, now: datetime | None = None) -> BacktestResult:
        if self.model is None or self.event_start is None or self.event_end is None:
            raise RuntimeError("System must be initialized before backtesting")
        current_time = now or datetime.now(UTC)
        duration = self.event_end - self.event_start
        elapsed = min(max(current_time - self.event_start, timedelta(0)), duration)
        fraction = elapsed / duration
        fraction = min(0.9, max(0.1, fraction))
        result = walk_forward_adaptive_backtest(
            self.history_post_times,
            history_start=self.model.history_start,
            history_end=self.model.history_end,
            event_duration=duration,
            observation_fraction=fraction,
        )
        LOGGER.info(
            "Backtest: samples=%s observed=%.1f%% MAE=%.1f RMSE=%.1f "
            "log_loss=%.3f coverage80=%.1f%% top1=%.1f%% top3=%.1f%%",
            result.samples,
            result.observation_fraction * 100,
            result.mean_absolute_error,
            result.root_mean_squared_error,
            result.mean_log_loss,
            result.coverage_80 * 100,
            result.top1_interval_accuracy * 100,
            result.top3_interval_coverage * 100,
        )
        return result

    async def _quote_market(
        self, market: EventMarket, forecast: CountForecast
    ) -> MarketQuote | None:
        try:
            # BUY is the highest resting bid; SELL is the lowest resting ask.
            bid_price, ask_price = await asyncio.gather(
                self.client.get_price(asset_id=market.asset_id, side="BUY"),
                self.client.get_price(asset_id=market.asset_id, side="SELL"),
            )
        except Exception as error:
            LOGGER.warning("Skipping %s quote: %s", market.interval_label, error)
            return None
        interval = parse_interval(market.interval_label)
        return MarketQuote(
            slug=market.slug,
            asset_id=market.asset_id,
            condition_id=market.condition_id,
            interval=interval,
            bid=bid_price,
            ask=ask_price,
            fee_rate=market.fee_rate,
            minimum_order_size=market.minimum_order_size,
            fair_probability=forecast.probability(interval.lower, interval.upper),
        )

    async def _quote_markets(self, forecast: CountForecast) -> tuple[MarketQuote, ...]:
        try:
            books = await self.client.get_order_books(
                asset_ids=[market.asset_id for market in self.markets]
            )
        except Exception as error:
            LOGGER.warning("Bulk order-book request failed; using price fallback: %s", error)
            results = await asyncio.gather(
                *(self._quote_market(market, forecast) for market in self.markets)
            )
            return tuple(item for item in results if item is not None)

        book_by_asset = {str(book.asset_id): book for book in books}
        quotes = []
        for market in self.markets:
            book = book_by_asset.get(market.asset_id)
            if book is None:
                LOGGER.warning("Skipping %s: order book is missing", market.interval_label)
                continue
            bids = tuple((level.price, level.size) for level in tuple(reversed(book.bids))[:10])
            asks = tuple((level.price, level.size) for level in tuple(reversed(book.asks))[:10])
            interval = parse_interval(market.interval_label)
            quotes.append(
                MarketQuote(
                    slug=market.slug,
                    asset_id=market.asset_id,
                    condition_id=market.condition_id,
                    interval=interval,
                    bid=bids[0][0] if bids else Decimal("0"),
                    ask=asks[0][0] if asks else Decimal("1"),
                    fee_rate=market.fee_rate,
                    minimum_order_size=market.minimum_order_size,
                    fair_probability=forecast.probability(interval.lower, interval.upper),
                    bids=bids,
                    asks=asks,
                    has_full_book=True,
                )
            )
        return tuple(quotes)

    async def _live_positions(self) -> dict[str, PositionView]:
        condition_ids = [market.condition_id for market in self.markets]
        page = await self.client.list_positions(
            condition_id=condition_ids, page_size=100
        ).first_page()
        assets = {market.asset_id for market in self.markets}
        return {
            str(item.asset_id): PositionView(item.current_size, item.total_cost_usdc)
            for item in page.items
            if str(item.asset_id) in assets and item.current_size > 0
        }

    async def run_once(
        self,
        now: datetime | None = None,
        *,
        execute_orders: bool = True,
    ) -> CycleResult:
        if self.model is None:
            await self.initialize()
        assert self.model is not None
        assert self.tracking_id is not None
        assert self.event_start is not None
        assert self.event_end is not None
        current_time = now or datetime.now(UTC)
        self.state.roll_day(current_time.date())
        await self._freeze_models_at_event_start(current_time)
        snapshot = await self.xtracker.get_tracking(self.tracking_id)
        observed_until = min(current_time, self.event_end)
        latest = self.repository.latest_created_at(self.config.xtracker_handle)
        sync_start = max(
            self.event_start,
            (latest - timedelta(minutes=10)) if latest is not None else self.event_start,
        )
        if sync_start < observed_until:
            latest_posts = await self.xtracker.get_posts(
                self.config.xtracker_handle,
                start=sync_start,
                end=observed_until,
            )
            self.repository.upsert_posts(self.config.xtracker_handle, latest_posts)
        observed_post_times = self.repository.post_times(
            self.config.xtracker_handle,
            start=self.event_start,
            end=observed_until,
        )
        if len(observed_post_times) != snapshot.total:
            LOGGER.warning(
                "xtracker count mismatch: tracking=%s posts_endpoint=%s; "
                "official tracking count remains authoritative",
                snapshot.total,
                len(observed_post_times),
            )
        try:
            outcome_alert = await self._finalize_event_outcome(snapshot, current_time)
            if outcome_alert is not None:
                LOGGER.warning("%s | %s", *outcome_alert)
                if self.notifier is not None:
                    await self.notifier.send(title=outcome_alert[0], text=outcome_alert[1])
        except Exception as error:
            LOGGER.error("Event outcome finalization failed: %s", error)
        forecast_model, forecast = self._forecast_for_time(
            current_count=snapshot.total,
            current_time=current_time,
            observed_post_times=observed_post_times,
        )
        quotes = await self._quote_markets(forecast)
        positions = await self._live_positions() if self.live else self.state.position_views()
        assessment = None
        if len(quotes) >= 3:
            try:
                assessment = assess_portfolio(list(quotes), positions)
            except ValueError as error:
                LOGGER.warning("Portfolio band assessment unavailable: %s", error)
        four_model_summary = None
        if (
            assessment is not None
            and assessment.target_band.labels != self.state.last_target_band
            and self.config.four_model_alerts
        ):
            four_model_summary = await self._four_model_comparison_summary(
                current_time=current_time,
                current_count=snapshot.total,
            )
        alerts = self._update_monitor_state(
            assessment=assessment,
            regime=forecast.regime,
            forecast_model=forecast_model,
            positions=positions,
            now=current_time,
            four_model_summary=four_model_summary,
        )
        for title, message in alerts:
            LOGGER.warning("%s | %s", title, message)
            try:
                if self.notifier is not None:
                    await self.notifier.send(title=title, text=message)
            except Exception as error:
                LOGGER.error("DingTalk alert failed: %s", error)
        minutes_remaining = Decimal(
            str(max(0.0, (self.event_end - current_time).total_seconds() / 60))
        )
        elapsed_minutes = Decimal(
            str(max(0.0, (current_time - self.event_start).total_seconds() / 60))
        )
        if self._cooldown_active(current_time):
            decision = ArbDecision(ArbActionKind.HOLD, "trade cooldown is active")
        elif not positions and elapsed_minutes < self.config.min_entry_elapsed_minutes:
            decision = ArbDecision(ArbActionKind.HOLD, "24-hour entry delay is active")
        else:
            decision_quotes = list(quotes)
            if assessment is not None:
                target_assets = set(assessment.target_band.asset_ids)
                decision_quotes = [
                    quote
                    for quote in quotes
                    if quote.asset_id in positions or quote.asset_id in target_assets
                ]
            decision = decide_action(
                config=self.config,
                quotes=decision_quotes,
                positions=positions,
                daily_buy_usd=self.state.daily_buy_usd,
                minutes_remaining=minutes_remaining,
            )
        result = CycleResult(
            snapshot,
            forecast,
            forecast_model,
            quotes,
            decision,
            minutes_remaining,
            assessment,
            tuple(title for title, _ in alerts),
        )
        self._log_cycle(result, positions)
        if execute_orders and decision.action is not ArbActionKind.HOLD:
            if self.live:
                await self._execute_live(decision, current_time)
            else:
                self._execute_paper(decision, current_time)
        self.state.save(self.config.state_path)
        self._write_audit(result, positions)
        try:
            self._write_db_snapshot(result, current_time)
        except Exception as error:
            LOGGER.error("MySQL monitor snapshot failed: %s", error)
        return result

    async def _four_model_comparison_summary(
        self,
        *,
        current_time: datetime,
        current_count: int,
    ) -> str:
        assert self.event_start is not None
        assert self.event_end is not None
        assert self.config.four_model_python is not None
        as_of = min(max(current_time, self.event_start), self.event_end)
        output_path = self.config.four_model_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            self.config.four_model_python,
            "-m",
            "polymarket_bot.elon_model",
            "--handle",
            self.config.xtracker_handle,
            "forecast-gru-comparison",
            "--history-days",
            str(self.config.history_days),
            "--event-start",
            self.event_start.isoformat(),
            "--event-end",
            self.event_end.isoformat(),
            "--as-of",
            as_of.isoformat(),
            "--current-count",
            str(current_count),
            "--interval-config",
            str(self.config.interval_config_path),
            "--original-run",
            str(self.config.gru_original_run),
            "--v3-run",
            str(self.config.gru_v3_run),
            "--v4-run",
            str(self.config.gru_v4_run),
            "--fold",
            str(self.config.gru_fold),
            "--device",
            self.config.gru_device,
            "--output",
            str(output_path),
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.config.four_model_timeout_seconds
            )
            if process.returncode != 0:
                detail = stderr.decode(errors="replace").strip().splitlines()
                raise RuntimeError(detail[-1] if detail else f"exit code {process.returncode}")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            return format_four_model_comparison(payload)
        except TimeoutError:
            process.kill()
            await process.communicate()
            detail = f"超过 {self.config.four_model_timeout_seconds} 秒"
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            detail = str(error)
        LOGGER.error("Four-model interval alert unavailable: %s", detail)
        return f"\n\n四模型预测暂不可用：{detail}"

    def _update_monitor_state(
        self,
        *,
        assessment: PortfolioAssessment | None,
        regime: str,
        forecast_model: str,
        positions: dict[str, PositionView],
        now: datetime,
        four_model_summary: str | None = None,
    ) -> list[tuple[str, str]]:
        alerts: list[tuple[str, str]] = []
        if (
            self.state.last_forecast_model is not None
            and forecast_model != self.state.last_forecast_model
        ):
            alerts.append(
                (
                    "预测模型切换",
                    f"路由从 {self.state.last_forecast_model} 切换到 {forecast_model}。",
                )
            )
        self.state.last_forecast_model = forecast_model
        if assessment is not None:
            target = assessment.target_band.labels
            if target != self.state.last_target_band:
                previous = self.state.last_target_band
                changed_legs = max(
                    len(set(previous) - set(target)),
                    len(set(target) - set(previous)),
                )
                message = (
                    f"预测三档从 {' / '.join(previous) if previous else '未建立'} "
                    f"移动到 {' / '.join(target)}；变化{changed_legs}档；"
                    f"新组合概率{assessment.target_band.probability:.1%}。"
                )
                if positions:
                    rotation_gain = assessment.rotation_value_gain_pct
                    gain_text = f"{rotation_gain:.1%}" if rotation_gain is not None else "不可计算"
                    message += (
                        f" 持仓组合概率{assessment.held_probability:.1%}；"
                        f"按当前Bid/Ask换仓的模型价值改善{gain_text}。"
                    )
                if four_model_summary:
                    message += four_model_summary
                alerts.append(("区间移动预警", message))
                self.state.last_target_band = target
                self.state.target_band_changed_at = now

        if self.state.last_regime is not None and regime != self.state.last_regime:
            alerts.append(
                (
                    "发帖状态变化",
                    f"实时状态从 {self.state.last_regime} 变为 {regime}。",
                )
            )
        self.state.last_regime = regime

        if assessment is None or not positions or not assessment.quotes_complete:
            if not positions:
                self.state.position_fingerprint = None
                self.state.entry_liquidation_usd = None
                self.state.peak_liquidation_usd = None
                self.state.last_drawdown_bucket = 0
                self.state.deterioration_warning_sent = False
                self.state.max_loss_warning_sent = False
            return alerts

        fingerprint = "|".join(
            f"{asset}:{positions[asset].shares.normalize()}" for asset in sorted(positions)
        )
        liquidation = assessment.liquidation_usd
        if fingerprint != self.state.position_fingerprint:
            self.state.position_fingerprint = fingerprint
            self.state.entry_liquidation_usd = liquidation
            self.state.peak_liquidation_usd = liquidation
            self.state.last_drawdown_bucket = 0
            self.state.deterioration_warning_sent = False
            self.state.max_loss_warning_sent = False
            return alerts

        entry = self.state.entry_liquidation_usd or liquidation
        peak = max(self.state.peak_liquidation_usd or liquidation, liquidation)
        self.state.peak_liquidation_usd = peak
        if peak > 0:
            drawdown = max(Decimal("0"), Decimal("1") - liquidation / peak)
            bucket = int(drawdown / self.config.drawdown_alert_step)
            if bucket > self.state.last_drawdown_bucket:
                self.state.last_drawdown_bucket = bucket
                alerts.append(
                    (
                        "组合利润回撤",
                        f"按可成交Bid计算，组合从峰值回撤{drawdown:.1%}；"
                        f"当前平仓价值${liquidation:.2f}。",
                    )
                )
        if entry > 0:
            deterioration = Decimal("1") - liquidation / entry
            if (
                deterioration >= self.config.deterioration_warning
                and not self.state.deterioration_warning_sent
            ):
                self.state.deterioration_warning_sent = True
                alerts.append(
                    (
                        "持仓恶化预警",
                        f"组合相对建仓后首次可平仓价值下降{deterioration:.1%}。",
                    )
                )
        if (
            assessment.unrealized_return is not None
            and assessment.unrealized_return <= -self.config.max_strategy_loss
            and not self.state.max_loss_warning_sent
        ):
            self.state.max_loss_warning_sent = True
            alerts.append(
                (
                    "最大亏损预警",
                    f"组合按当前Bid计算亏损{assessment.unrealized_return:.1%}，"
                    f"已达到策略{self.config.max_strategy_loss:.0%}风险上限。",
                )
            )
        return alerts

    def _cooldown_active(self, now: datetime) -> bool:
        if self.state.last_trade_at is None:
            return False
        return (now - self.state.last_trade_at).total_seconds() < self.config.trade_cooldown_seconds

    def _execute_paper(self, decision: ArbDecision, now: datetime) -> None:
        assert decision.quote is not None
        if decision.action is ArbActionKind.BUY:
            self.state.record_buy(
                decision.quote.asset_id,
                decision.amount_usd,
                decision.quote.all_in_ask,
                now,
            )
        else:
            self.state.record_sell(decision.quote.asset_id, now)
        LOGGER.warning(
            "PAPER %s %s | %s",
            decision.action.value,
            decision.quote.interval.label,
            decision.reason,
        )

    async def _execute_live(self, decision: ArbDecision, now: datetime) -> None:
        assert decision.quote is not None
        geo = await check_geoblock()
        if geo.get("blocked"):
            raise RuntimeError("Polymarket geoblock check rejected order placement")
        if decision.action is ArbActionKind.BUY:
            response = await self.client.place_market_order(
                asset_id=decision.quote.asset_id,
                side="BUY",
                amount=decision.amount_usd,
                max_spend=decision.amount_usd,
                max_price=buy_price_cap(decision.quote.ask, self.config.max_slippage),
                order_type="FAK",
            )
            if response.ok:
                self.state.daily_buy_usd += response.making_amount
        else:
            response = await self.client.place_market_order(
                asset_id=decision.quote.asset_id,
                side="SELL",
                shares=decision.shares,
                min_price=sell_price_floor(decision.quote.bid, self.config.max_slippage),
                order_type="FAK",
            )
        if not response.ok:
            raise RuntimeError(f"Order rejected ({response.code}): {response.message}")
        self.state.last_trade_at = now
        self.state.save(self.config.state_path)
        LOGGER.warning(
            "LIVE %s %s accepted: %s",
            decision.action.value,
            decision.quote.interval.label,
            response.order_id,
        )
        if response.trade_ids:
            try:
                await self.client.wait_for_order_fill_settlement(response)
            except TimeoutError:
                LOGGER.warning("Order accepted; on-chain settlement is still pending")

    def _log_cycle(self, result: CycleResult, positions: dict[str, PositionView]) -> None:
        LOGGER.info(
            "count=%s model=%s expected_final=%.1f base=%.1f std=%.1f remaining=%.1fh "
            "regime=%s multiplier=%.2f positions=%s",
            result.snapshot.total,
            result.forecast_model,
            result.forecast.expected_final,
            result.forecast.base_expected_final or result.forecast.expected_final,
            result.forecast.stddev_final,
            result.forecast.remaining_exposure,
            result.forecast.regime,
            result.forecast.realtime_multiplier,
            len(positions),
        )
        for quote in sorted(result.quotes, key=lambda item: item.entry_edge, reverse=True)[:6]:
            LOGGER.info(
                "%7s fair=%6.3f bid=%6.3f ask=%6.3f entry_edge=%+.3f exit_adv=%+.3f",
                quote.interval.label,
                quote.fair_probability,
                quote.bid,
                quote.ask,
                quote.entry_edge,
                quote.exit_advantage,
            )
        if result.assessment is not None:
            assessment = result.assessment
            LOGGER.info(
                "target_band=%s probability=%.3f held_probability=%.3f "
                "roll=%s rotation_gain=%s liquidation=$%.2f",
                "/".join(assessment.target_band.labels),
                assessment.target_band.probability,
                assessment.held_probability,
                assessment.roll_count,
                (
                    f"{assessment.rotation_value_gain_pct:.1%}"
                    if assessment.rotation_value_gain_pct is not None
                    else "n/a"
                ),
                assessment.liquidation_usd,
            )
        LOGGER.info("decision=%s | %s", result.decision.action.value, result.decision.reason)

    def _write_audit(self, result: CycleResult, positions: dict[str, PositionView]) -> None:
        path: Path = self.config.audit_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "at": datetime.now(UTC).isoformat(),
            "mode": "LIVE" if self.live else "PAPER",
            "count": result.snapshot.total,
            "forecast_model": result.forecast_model,
            "expected_final": result.forecast.expected_final,
            "stddev_final": result.forecast.stddev_final,
            "base_expected_final": result.forecast.base_expected_final,
            "realtime_multiplier": result.forecast.realtime_multiplier,
            "regime": result.forecast.regime,
            "recent_counts": dict(result.forecast.recent_counts),
            "minutes_remaining": str(result.minutes_remaining),
            "decision": {
                "action": result.decision.action.value,
                "reason": result.decision.reason,
                "interval": (
                    result.decision.quote.interval.label if result.decision.quote else None
                ),
            },
            "alerts": list(result.alerts),
            "portfolio": (
                {
                    "target_band": list(result.assessment.target_band.labels),
                    "target_probability": str(result.assessment.target_band.probability),
                    "held_band": list(result.assessment.held_labels),
                    "held_probability": str(result.assessment.held_probability),
                    "shift_steps": result.assessment.shift_steps,
                    "roll_count": result.assessment.roll_count,
                    "sell_labels": list(result.assessment.sell_labels),
                    "buy_labels": list(result.assessment.buy_labels),
                    "invested_usd": str(result.assessment.invested_usd),
                    "liquidation_usd": str(result.assessment.liquidation_usd),
                    "model_value_usd": str(result.assessment.model_value_usd),
                    "unrealized_return": (
                        str(result.assessment.unrealized_return)
                        if result.assessment.unrealized_return is not None
                        else None
                    ),
                    "rotation_value_gain_pct": (
                        str(result.assessment.rotation_value_gain_pct)
                        if result.assessment.rotation_value_gain_pct is not None
                        else None
                    ),
                }
                if result.assessment is not None
                else None
            ),
            "positions": {
                asset: {"shares": str(item.shares), "cost_usd": str(item.cost_usd)}
                for asset, item in positions.items()
            },
            "quotes": [
                {
                    "interval": quote.interval.label,
                    "fair": str(quote.fair_probability),
                    "bid": str(quote.bid),
                    "ask": str(quote.ask),
                    "entry_edge": str(quote.entry_edge),
                    "exit_advantage": str(quote.exit_advantage),
                }
                for quote in result.quotes
            ],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_db_snapshot(self, result: CycleResult, observed_at: datetime) -> None:
        assessment = result.assessment
        snapshot = {
            "event_slug": self.config.event_slug,
            "observed_at": observed_at,
            "post_count": result.snapshot.total,
            "expected_final": result.forecast.expected_final,
            "base_expected_final": result.forecast.base_expected_final,
            "stddev_final": result.forecast.stddev_final,
            "regime": result.forecast.regime,
            "realtime_multiplier": result.forecast.realtime_multiplier,
            "recent_counts": dict(result.forecast.recent_counts),
            "target_band": list(assessment.target_band.labels) if assessment else None,
            "target_probability": assessment.target_band.probability if assessment else None,
            "held_band": list(assessment.held_labels) if assessment else None,
            "held_probability": assessment.held_probability if assessment else None,
            "invested_usd": assessment.invested_usd if assessment else None,
            "liquidation_usd": assessment.liquidation_usd if assessment else None,
            "model_value_usd": assessment.model_value_usd if assessment else None,
            "unrealized_return": assessment.unrealized_return if assessment else None,
            "roll_count": assessment.roll_count if assessment else None,
            "rotation_value_gain_pct": (assessment.rotation_value_gain_pct if assessment else None),
            "alerts": list(result.alerts),
        }
        quotes = [
            {
                "asset_id": quote.asset_id,
                "market_slug": quote.slug,
                "interval_label": quote.interval.label,
                "bid": quote.bid,
                "ask": quote.ask,
                "net_bid": quote.net_bid,
                "all_in_ask": quote.all_in_ask,
                "fair_probability": quote.fair_probability,
                "bids": quote.bids,
                "asks": quote.asks,
            }
            for quote in result.quotes
        ]
        self.repository.record_monitor_snapshot(snapshot, quotes)

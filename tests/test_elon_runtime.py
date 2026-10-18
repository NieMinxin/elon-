import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from polymarket_bot.config import ConfigError
from polymarket_bot.elon_config import ElonArbConfig
from polymarket_bot.elon_main import run
from polymarket_bot.elon_state import ElonArbState
from polymarket_bot.elon_trader import (
    ElonArbSystem,
    EventMarket,
    format_four_model_comparison,
)
from polymarket_bot.forecast import CountForecast
from polymarket_bot.notifier import DingTalkNotifier
from polymarket_bot.xtracker import TrackingSnapshot


class FailingHttpClient:
    async def post(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("disabled notifier attempted an HTTP request")


class ForecastStub:
    def __init__(self, result: object) -> None:
        self.result = result

    def forecast(self, **kwargs):
        return self.result


def test_elon_defaults_match_paper_portfolio_plan() -> None:
    config = ElonArbConfig.from_env({})

    assert config.poll_interval_seconds == 120
    assert config.forecast_strategy == "route"
    assert config.order_amount_usd == Decimal("20")
    assert config.max_total_cost_usd == Decimal("100")
    assert config.max_positions == 3
    assert config.min_entry_elapsed_minutes == 1440
    assert config.label_finalization_delay_minutes == 60
    assert config.four_model_alerts is False


def test_monitoring_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = ElonArbState.empty("event")
    state.last_target_band = ("320-339", "340-359", "360-379")
    state.target_band_changed_at = datetime(2026, 10, 17, tzinfo=UTC)
    state.last_regime = "high"
    state.last_forecast_model = "gbm_conformal_84h"
    state.entry_liquidation_usd = Decimal("51.25")
    state.peak_liquidation_usd = Decimal("80")
    state.last_drawdown_bucket = 2
    state.save(path)

    loaded = ElonArbState.load(path, "event")

    assert loaded.last_target_band == state.last_target_band
    assert loaded.last_regime == "high"
    assert loaded.last_forecast_model == "gbm_conformal_84h"
    assert loaded.entry_liquidation_usd == Decimal("51.25")
    assert loaded.peak_liquidation_usd == Decimal("80")
    assert loaded.last_drawdown_bucket == 2


def test_disabled_dingtalk_notifier_is_a_noop() -> None:
    notifier = DingTalkNotifier(FailingHttpClient(), webhook=None, secret=None)  # type: ignore[arg-type]

    asyncio.run(notifier.send(title="test", text="test"))


@pytest.mark.parametrize(
    ("elapsed_hours", "expected_name", "expected_result"),
    [
        (24, "gbm_conformal_24h", "gbm-24"),
        (42, "gbm_conformal_42h", "gbm-42"),
        (84, "gbm_conformal_84h", "gbm-84"),
        (126, "adaptive_reconciled_126h", "adaptive"),
    ],
)
def test_realtime_route_uses_expected_checkpoint(
    elapsed_hours: int,
    expected_name: str,
    expected_result: str,
) -> None:
    start = datetime(2026, 10, 13, 16, tzinfo=UTC)
    system = ElonArbSystem.__new__(ElonArbSystem)
    system.config = ElonArbConfig.from_env({"ELON_FORECAST_STRATEGY": "route"})
    system.event_start = start
    system.event_end = datetime(2026, 10, 20, 16, tzinfo=UTC)
    system.model = ForecastStub("adaptive")
    system.gbm_models = {
        24: ForecastStub("gbm-24"),
        42: ForecastStub("gbm-42"),
        84: ForecastStub("gbm-84"),
    }

    name, result = system._forecast_for_time(
        current_count=200,
        current_time=start + timedelta(hours=elapsed_hours),
        observed_post_times=[],
    )

    assert name == expected_name
    assert result == expected_result


def test_realtime_route_uses_frozen_baseline_correction_before_24h() -> None:
    start = datetime(2026, 10, 13, 16, tzinfo=UTC)
    baseline = CountForecast(0, 0, 0, 0, 0, 168, integer_probabilities=(0.2, 0.5, 0.3))
    updated = CountForecast(0, 0, 0, 0, 0, 156, integer_probabilities=(0.1, 0.3, 0.6))
    system = ElonArbSystem.__new__(ElonArbSystem)
    system.config = ElonArbConfig.from_env({"ELON_FORECAST_STRATEGY": "route"})
    system.event_start = start
    system.event_end = start + timedelta(days=7)
    system.model = ForecastStub(updated)
    system.gbm_models = {}
    system.baseline_forecast = baseline
    system.baseline_reference_forecast = baseline

    name, result = system._forecast_for_time(
        current_count=0,
        current_time=start + timedelta(hours=12),
        observed_post_times=[],
    )

    assert name == "gbm_conformal_0h+adaptive_realtime"
    assert result.integer_probabilities[2] > baseline.integer_probabilities[2]


def test_model_switch_alert_is_emitted_once() -> None:
    system = ElonArbSystem.__new__(ElonArbSystem)
    system.config = ElonArbConfig.from_env({})
    system.state = ElonArbState.empty("event")
    now = datetime(2026, 10, 17, tzinfo=UTC)

    first = system._update_monitor_state(
        assessment=None,
        regime="normal",
        forecast_model="gbm_conformal_84h",
        positions={},
        now=now,
    )
    switched = system._update_monitor_state(
        assessment=None,
        regime="normal",
        forecast_model="adaptive_reconciled_126h",
        positions={},
        now=now,
    )
    repeated = system._update_monitor_state(
        assessment=None,
        regime="normal",
        forecast_model="adaptive_reconciled_126h",
        positions={},
        now=now,
    )

    assert first == []
    assert [title for title, _ in switched] == ["预测模型切换"]
    assert repeated == []


def test_formats_four_model_interval_alert() -> None:
    model = {
        "expected_final": 312.1,
        "central_80": [252, 394],
        "three_band": ["260-279", "280-299", "300-319"],
        "three_band_probability": 0.485,
        "interval_probabilities": [
            {"label": "260-279", "probability": 0.139},
            {"label": "280-299", "probability": 0.174},
            {"label": "300-319", "probability": 0.171},
        ],
    }
    payload = {
        "model_observed_at": "2026-10-18T01:00:00+00:00",
        "models": {
            "route_baseline": {**model, "route_component": "GBM-24h"},
            "original_gru_nll": model,
            "v3_dense_hourly_gru": model,
            "v4_calibrated_ensemble": model,
        },
    }

    message = format_four_model_comparison(payload)

    assert "路由基线 GBM-24h" in message
    assert "均值 312.1；80% 区间 252-394" in message
    assert "260-279 13.9% / 280-299 17.4% / 300-319 17.1%" in message
    assert "合计 48.5%" in message
    assert "原 GRU-NLL" in message
    assert "V3 密集小时 GRU" in message
    assert "V4 校准集成" in message


def test_event_outcome_is_labeled_from_official_tracking_total() -> None:
    class RepositoryStub:
        def __init__(self) -> None:
            self.payload = None

        def event_outcome(self, event_slug: str):
            return None

        def upsert_posts(self, handle: str, posts: list[object]) -> int:
            return len(posts)

        def post_ids(self, handle: str, *, start: datetime, end: datetime) -> set[str]:
            return {"1", "2"}

        def record_event_outcome(self, payload: dict[str, object]) -> str:
            self.payload = payload
            return "inserted"

    class XTrackerStub:
        async def get_posts(self, handle: str, *, start: datetime, end: datetime):
            return [SimpleNamespace(platform_id="1"), SimpleNamespace(platform_id="2")]

    start = datetime(2026, 10, 16, 16, tzinfo=UTC)
    end = start + timedelta(days=7)
    repository = RepositoryStub()
    system = ElonArbSystem.__new__(ElonArbSystem)
    system.config = ElonArbConfig.from_env(
        {
            "ELON_EVENT_SLUG": "test-event",
            "ELON_LABEL_FINALIZATION_DELAY_MINUTES": "0",
        }
    )
    system.event_start = start
    system.event_end = end
    system.repository = repository
    system.xtracker = XTrackerStub()
    system.markets = (
        EventMarket("low", "1", "c1", "280-299", Decimal("0"), Decimal("0")),
        EventMarket("winner", "2", "c2", "300-319", Decimal("0"), Decimal("0")),
        EventMarket("high", "3", "c3", "320-339", Decimal("0"), Decimal("0")),
    )
    snapshot = TrackingSnapshot("tracking", "test", start, end, 317, end, end)

    alert = asyncio.run(system._finalize_event_outcome(snapshot, end))

    assert alert is not None
    assert alert[0] == "盘口最终标签已记录"
    assert repository.payload is not None
    assert repository.payload["final_count"] == 317
    assert repository.payload["winning_interval_label"] == "300-319"
    assert repository.payload["tracking_posts_delta"] == 315


def test_route_strategy_rejects_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELON_ENABLE_LIVE", "true")
    args = SimpleNamespace(live=True, backtest=True)

    with pytest.raises(ConfigError, match="observe-only"):
        asyncio.run(run(args))

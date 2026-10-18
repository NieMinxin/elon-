from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from polymarket_bot.elon_strategy import PositionView


@dataclass
class PaperPosition:
    shares: Decimal
    cost_usd: Decimal


@dataclass
class ElonArbState:
    event_slug: str
    spending_date: str
    daily_buy_usd: Decimal = Decimal("0")
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    last_trade_at: datetime | None = None
    last_target_band: tuple[str, ...] = ()
    target_band_changed_at: datetime | None = None
    last_regime: str | None = None
    last_forecast_model: str | None = None
    position_fingerprint: str | None = None
    entry_liquidation_usd: Decimal | None = None
    peak_liquidation_usd: Decimal | None = None
    last_drawdown_bucket: int = 0
    deterioration_warning_sent: bool = False
    max_loss_warning_sent: bool = False

    @classmethod
    def empty(cls, event_slug: str) -> ElonArbState:
        return cls(event_slug, datetime.now(UTC).date().isoformat())

    @classmethod
    def load(cls, path: Path, event_slug: str) -> ElonArbState:
        if not path.exists():
            return cls.empty(event_slug)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("event_slug") != event_slug:
            return cls.empty(event_slug)
        state = cls(
            event_slug=event_slug,
            spending_date=str(raw["spending_date"]),
            daily_buy_usd=Decimal(str(raw.get("daily_buy_usd", "0"))),
            positions={
                asset_id: PaperPosition(
                    Decimal(str(item["shares"])), Decimal(str(item["cost_usd"]))
                )
                for asset_id, item in raw.get("positions", {}).items()
            },
            last_trade_at=(
                datetime.fromisoformat(raw["last_trade_at"]) if raw.get("last_trade_at") else None
            ),
            last_target_band=tuple(str(item) for item in raw.get("last_target_band", [])),
            target_band_changed_at=(
                datetime.fromisoformat(raw["target_band_changed_at"])
                if raw.get("target_band_changed_at")
                else None
            ),
            last_regime=(str(raw["last_regime"]) if raw.get("last_regime") else None),
            last_forecast_model=(
                str(raw["last_forecast_model"]) if raw.get("last_forecast_model") else None
            ),
            position_fingerprint=raw.get("position_fingerprint"),
            entry_liquidation_usd=(
                Decimal(str(raw["entry_liquidation_usd"]))
                if raw.get("entry_liquidation_usd") is not None
                else None
            ),
            peak_liquidation_usd=(
                Decimal(str(raw["peak_liquidation_usd"]))
                if raw.get("peak_liquidation_usd") is not None
                else None
            ),
            last_drawdown_bucket=int(raw.get("last_drawdown_bucket", 0)),
            deterioration_warning_sent=bool(raw.get("deterioration_warning_sent", False)),
            max_loss_warning_sent=bool(raw.get("max_loss_warning_sent", False)),
        )
        state.roll_day()
        return state

    def roll_day(self, today: date | None = None) -> None:
        value = (today or datetime.now(UTC).date()).isoformat()
        if value != self.spending_date:
            self.spending_date = value
            self.daily_buy_usd = Decimal("0")

    def position_views(self) -> dict[str, PositionView]:
        return {
            asset_id: PositionView(item.shares, item.cost_usd)
            for asset_id, item in self.positions.items()
            if item.shares > 0
        }

    def record_buy(
        self, asset_id: str, amount: Decimal, all_in_price: Decimal, now: datetime
    ) -> None:
        shares = amount / all_in_price
        current = self.positions.get(asset_id, PaperPosition(Decimal("0"), Decimal("0")))
        self.positions[asset_id] = PaperPosition(current.shares + shares, current.cost_usd + amount)
        self.daily_buy_usd += amount
        self.last_trade_at = now

    def record_sell(self, asset_id: str, now: datetime) -> None:
        self.positions.pop(asset_id, None)
        self.last_trade_at = now

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_slug": self.event_slug,
            "spending_date": self.spending_date,
            "daily_buy_usd": str(self.daily_buy_usd),
            "positions": {
                asset_id: {"shares": str(item.shares), "cost_usd": str(item.cost_usd)}
                for asset_id, item in self.positions.items()
            },
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "last_target_band": list(self.last_target_band),
            "target_band_changed_at": (
                self.target_band_changed_at.isoformat() if self.target_band_changed_at else None
            ),
            "last_regime": self.last_regime,
            "last_forecast_model": self.last_forecast_model,
            "position_fingerprint": self.position_fingerprint,
            "entry_liquidation_usd": (
                str(self.entry_liquidation_usd) if self.entry_liquidation_usd is not None else None
            ),
            "peak_liquidation_usd": (
                str(self.peak_liquidation_usd) if self.peak_liquidation_usd is not None else None
            ),
            "last_drawdown_bucket": self.last_drawdown_bucket,
            "deterioration_warning_sent": self.deterioration_warning_sent,
            "max_loss_warning_sent": self.max_loss_warning_sent,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

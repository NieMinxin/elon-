from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path


@dataclass
class BotState:
    spending_date: str
    daily_buy_usd: Decimal = Decimal("0")
    paper_market_key: str | None = None
    paper_shares: Decimal = Decimal("0")
    paper_cost_usd: Decimal = Decimal("0")
    last_trade_at: datetime | None = None

    @classmethod
    def empty(cls, today: date | None = None) -> BotState:
        return cls(spending_date=(today or datetime.now(UTC).date()).isoformat())

    def roll_day(self, today: date | None = None) -> None:
        current = (today or datetime.now(UTC).date()).isoformat()
        if self.spending_date != current:
            self.spending_date = current
            self.daily_buy_usd = Decimal("0")

    def select_paper_market(self, market_slug: str, outcome: str) -> None:
        key = f"{market_slug}:{outcome}"
        if self.paper_market_key != key:
            self.paper_market_key = key
            self.paper_shares = Decimal("0")
            self.paper_cost_usd = Decimal("0")

    def record_buy(self, usd: Decimal, shares: Decimal, when: datetime) -> None:
        self.daily_buy_usd += usd
        self.paper_cost_usd += usd
        self.paper_shares += shares
        self.last_trade_at = when

    def record_sell(self, shares: Decimal, proceeds: Decimal, when: datetime) -> None:
        if shares <= 0 or self.paper_shares <= 0:
            return
        sold = min(shares, self.paper_shares)
        cost_reduction = self.paper_cost_usd * sold / self.paper_shares
        self.paper_shares -= sold
        self.paper_cost_usd = max(Decimal("0"), self.paper_cost_usd - cost_reduction)
        if self.paper_shares == 0:
            self.paper_cost_usd = Decimal("0")
        self.last_trade_at = when

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["daily_buy_usd"] = str(self.daily_buy_usd)
        payload["paper_shares"] = str(self.paper_shares)
        payload["paper_cost_usd"] = str(self.paper_cost_usd)
        payload["last_trade_at"] = (
            self.last_trade_at.astimezone(UTC).isoformat() if self.last_trade_at else None
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> BotState:
        if not path.exists():
            return cls.empty()
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = cls(
            spending_date=str(raw["spending_date"]),
            daily_buy_usd=Decimal(str(raw.get("daily_buy_usd", "0"))),
            paper_market_key=raw.get("paper_market_key"),
            paper_shares=Decimal(str(raw.get("paper_shares", "0"))),
            paper_cost_usd=Decimal(str(raw.get("paper_cost_usd", "0"))),
            last_trade_at=(
                datetime.fromisoformat(raw["last_trade_at"]) if raw.get("last_trade_at") else None
            ),
        )
        state.roll_day()
        return state

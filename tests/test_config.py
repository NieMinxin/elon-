from decimal import Decimal

import pytest

from polymarket_bot.config import BotConfig, ConfigError


def valid_env() -> dict[str, str]:
    return {
        "POLYMARKET_MARKET_SLUG": "example-market",
        "POLYMARKET_BUY_BELOW": "0.40",
        "POLYMARKET_TAKE_PROFIT_AT": "0.60",
        "POLYMARKET_STOP_LOSS_AT": "0.25",
    }


def test_loads_safe_defaults() -> None:
    config = BotConfig.from_env(valid_env())

    assert config.outcome == "YES"
    assert config.order_amount_usd == Decimal("10")
    assert config.enable_live is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("POLYMARKET_OUTCOME", "MAYBE"),
        ("POLYMARKET_BUY_BELOW", "1.5"),
        ("POLYMARKET_STOP_LOSS_AT", "0.45"),
        ("POLYMARKET_TAKE_PROFIT_AT", "0.30"),
        ("POLYMARKET_MAX_SLIPPAGE", "0.30"),
    ],
)
def test_rejects_unsafe_values(name: str, value: str) -> None:
    env = valid_env()
    env[name] = value

    with pytest.raises(ConfigError):
        BotConfig.from_env(env)

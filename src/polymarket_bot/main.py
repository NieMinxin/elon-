from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from polymarket_bot.config import BotConfig, ConfigError
from polymarket_bot.trader import TradingBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polymarket threshold trading bot")
    parser.add_argument("--once", action="store_true", help="Evaluate once and exit")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow real orders; POLYMARKET_ENABLE_LIVE=true is also required",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


@asynccontextmanager
async def create_client(config: BotConfig, live: bool) -> AsyncIterator[Any]:
    if live:
        from polymarket import AsyncSecureClient

        private_key, wallet = config.require_live_credentials()
        async with await AsyncSecureClient.create(private_key=private_key, wallet=wallet) as client:
            yield client
    else:
        from polymarket import AsyncPublicClient

        async with AsyncPublicClient() as client:
            yield client


async def run(args: argparse.Namespace) -> None:
    config = BotConfig.from_env()
    if args.live and not config.enable_live:
        raise ConfigError(
            "--live also requires POLYMARKET_ENABLE_LIVE=true; this prevents accidental orders"
        )
    live = bool(args.live and config.enable_live)
    logging.warning(
        "Starting in %s mode for %s %s",
        "LIVE" if live else "PAPER",
        config.outcome,
        config.market_slug,
    )
    async with create_client(config, live) as client:
        bot = TradingBot(client, config, live=live)
        context = await bot.initialize()
        logging.info("Market: %s | asset_id=%s", context.question, context.asset_id)
        while True:
            try:
                await bot.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Bot cycle failed")
                if args.once:
                    raise
            if args.once:
                return
            await asyncio.sleep(config.poll_interval_seconds)


def cli() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(run(args))
    except (ConfigError, KeyboardInterrupt) as error:
        if isinstance(error, ConfigError):
            logging.error("Configuration error: %s", error)
            raise SystemExit(2) from error
        logging.info("Stopped")


if __name__ == "__main__":
    cli()

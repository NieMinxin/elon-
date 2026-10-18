from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from polymarket_bot.config import ConfigError
from polymarket_bot.elon_config import ElonArbConfig
from polymarket_bot.elon_trader import ElonArbSystem
from polymarket_bot.xtracker import XTrackerClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-driven Elon post-count interval trader")
    parser.add_argument("--once", action="store_true", help="Evaluate once and exit")
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Fetch, forecast, quote, and alert without recording or submitting orders",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow real orders; ELON_ENABLE_LIVE=true is also required",
    )
    parser.add_argument(
        "--refresh-history",
        action="store_true",
        help="Download the historical training set again",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a 90-day holdout backtest before evaluating the live event",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


@asynccontextmanager
async def create_client(config: ElonArbConfig, live: bool) -> AsyncIterator[Any]:
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
    config = ElonArbConfig.from_env()
    if args.live and not config.enable_live:
        raise ConfigError("--live also requires ELON_ENABLE_LIVE=true")
    if args.live and not args.backtest:
        raise ConfigError("--live requires --backtest so model calibration is checked first")
    live = bool(args.live and config.enable_live)
    if live and config.forecast_strategy == "route":
        raise ConfigError(
            "The GBM/adaptive route is observe-only until an independent validation passes"
        )
    logging.warning("Starting Elon interval system in %s mode", "LIVE" if live else "PAPER")
    async with httpx.AsyncClient(timeout=30) as http_client:
        xtracker = XTrackerClient(http_client)
        async with create_client(config, live) as client:
            system = ElonArbSystem(
                client,
                xtracker,
                config,
                live=live,
                refresh_history=args.refresh_history,
            )
            await system.initialize()
            if args.backtest:
                result = system.backtest()
                if live and (
                    result.coverage_80 < float(config.min_backtest_coverage)
                    or result.mean_absolute_error > float(config.max_backtest_mae)
                ):
                    raise ConfigError(
                        "Live trading blocked: backtest calibration is outside configured limits"
                    )
            while True:
                try:
                    await system.run_once(execute_orders=not args.observe_only)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logging.exception("Elon interval cycle failed")
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
    # httpx logs complete request URLs, including DingTalk access tokens.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args))
    except ConfigError as error:
        logging.error("Configuration error: %s", error)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        logging.info("Stopped")


if __name__ == "__main__":
    cli()

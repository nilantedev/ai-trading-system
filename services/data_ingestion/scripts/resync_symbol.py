#!/usr/bin/env python3
"""CLI utility to (re)sync a symbol's daily historical range.

Respects existing Redis progress key unless --force specified.
Dry-run supported via ENABLE_HIST_DRY_RUN env var (no writes/persist).
"""
import asyncio
import os
import click
from datetime import datetime
from services.data_ingestion.market_data_service import MarketDataService

@click.command()
@click.argument('symbol')
@click.option('--start', 'start_date', required=True, help='Start date YYYY-MM-DD')
@click.option('--end', 'end_date', required=True, help='End date YYYY-MM-DD')
@click.option('--force', is_flag=True, default=False, help='Ignore existing progress key and refetch')
async def main_async(symbol: str, start_date: str, end_date: str, force: bool):
    svc = MarketDataService()
    await svc.start()
    s = datetime.strptime(start_date, '%Y-%m-%d')
    e = datetime.strptime(end_date, '%Y-%m-%d')
    if s > e:
        raise click.ClickException('Start date after end date')

    progress_key = f"hist:progress:daily:{symbol.upper()}"
    if not force and svc.cache and hasattr(svc.cache, 'client'):
        try:
            existing = await svc.cache.client.get(progress_key)  # type: ignore
            if existing:
                click.echo(f"Existing progress {existing.decode() if isinstance(existing, bytes) else existing}; continuing from there unless outside requested window.")
        except Exception:
            pass

    bars = await svc.get_bulk_daily_historical(symbol, s, e)
    click.echo(f"Fetched {len(bars)} bars (dry-run={svc.enable_hist_dry_run})")
    if bars:
        click.echo(f"First bar: {bars[0].timestamp} last bar: {bars[-1].timestamp}")

@click.command()
@click.argument('symbol')
@click.option('--start', 'start_date', required=True, help='Start date YYYY-MM-DD')
@click.option('--end', 'end_date', required=True, help='End date YYYY-MM-DD')
@click.option('--force', is_flag=True, default=False, help='Ignore existing progress key and refetch')
def main(symbol: str, start_date: str, end_date: str, force: bool):
    asyncio.run(main_async(symbol, start_date, end_date, force))

if __name__ == '__main__':
    main()

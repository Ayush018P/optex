"""Lightweight Binance WebSocket client for paper trading display."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Dict

import websockets

BINANCE_WS = "wss://stream.binance.com:9443/ws"


async def depth_stream(symbol: str = "btcusdt") -> AsyncIterator[Dict]:
    """Yield order book depth snapshots for a symbol."""
    stream = f"{symbol.lower()}@depth5@100ms"
    url = f"{BINANCE_WS}/{stream}"
    async with websockets.connect(url, ping_interval=None) as ws:
        async for msg in ws:
            yield json.loads(msg)


async def trade_stream(symbol: str = "btcusdt") -> AsyncIterator[Dict]:
    """Yield trade prints for a symbol."""
    stream = f"{symbol.lower()}@trade"
    url = f"{BINANCE_WS}/{stream}"
    async with websockets.connect(url, ping_interval=None) as ws:
        async for msg in ws:
            yield json.loads(msg)


async def main() -> None:
    async for book in depth_stream():
        print(book)
        break


if __name__ == "__main__":
    asyncio.run(main())

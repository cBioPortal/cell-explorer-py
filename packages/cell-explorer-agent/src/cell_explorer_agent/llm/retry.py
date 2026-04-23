"""Exponential-backoff retry for LLM transport errors."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Retry `func` with exponential backoff on exceptions in `retry_on`.

    Raises the last exception if all attempts fail.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await func()
        except retry_on:
            if attempt >= max_attempts:
                raise
            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            await asyncio.sleep(delay)

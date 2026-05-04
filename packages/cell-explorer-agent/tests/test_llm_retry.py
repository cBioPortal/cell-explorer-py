import asyncio

import pytest

from cell_explorer_agent.llm.retry import retry_async


async def test_returns_on_first_success():
    calls = 0

    async def f():
        nonlocal calls
        calls += 1
        return 42

    r = await retry_async(f, max_attempts=3, base_delay_s=0.01)
    assert r == 42
    assert calls == 1


async def test_retries_until_success():
    calls = 0

    async def f():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient")
        return "ok"

    r = await retry_async(
        f,
        max_attempts=5,
        base_delay_s=0.001,
        retry_on=(RuntimeError,),
    )
    assert r == "ok"
    assert calls == 3


async def test_gives_up_after_max_attempts():
    async def f():
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="always"):
        await retry_async(
            f,
            max_attempts=3,
            base_delay_s=0.001,
            retry_on=(RuntimeError,),
        )


async def test_does_not_retry_unexpected_errors():
    calls = 0

    async def f():
        nonlocal calls
        calls += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        await retry_async(
            f,
            max_attempts=5,
            base_delay_s=0.001,
            retry_on=(RuntimeError,),
        )
    assert calls == 1

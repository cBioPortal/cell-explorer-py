"""Bounded-concurrency helper for tools that scan many gene columns."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from cell_explorer_agent.tools.zarr_protocol import ZarrAccess

T = TypeVar("T")


async def reduce_gene_columns(
    z: ZarrAccess,
    genes: list[str],
    reduce_fn: Callable[[str, np.ndarray], T],
    *,
    max_concurrency: int,
) -> list[T]:
    """Fetch each gene column with bounded concurrency, reduce, return aligned list.

    Args:
        z: ZarrAccess Protocol implementation (real adapter or test fake).
        genes: gene names to fetch, in the order results should be returned.
        reduce_fn: synchronous callable ``(gene_name, expression_array) -> T``.
            Called once per gene as soon as that gene's column is available.
            Returns a small per-gene value (tuple, float, etc.) that does not
            retain the expression array.
        max_concurrency: maximum number of in-flight ``gene_column`` calls.

    Returns:
        list of length ``len(genes)`` aligned positionally with the input.

    Memory: at most ``max_concurrency`` expression arrays are held in memory
    simultaneously. Each array is released after ``reduce_fn`` returns.

    Errors: any exception from ``z.gene_column`` or ``reduce_fn`` propagates;
    other in-flight tasks are cancelled by ``asyncio.gather``'s default behavior.

    Cancellation: if the calling task is cancelled, all in-flight tasks cancel cleanly.
    """
    if not genes:
        return []

    sem = asyncio.Semaphore(max_concurrency)

    async def _task(gene: str) -> T:
        async with sem:
            expr = await z.gene_column(gene)
            return reduce_fn(gene, expr)

    return await asyncio.gather(*(_task(g) for g in genes))

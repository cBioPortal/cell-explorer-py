"""Benchmark: compare_groups via X-scan vs strata fast path.

Times both code paths for a single (obs_column, group_a, group_b) query and
reports wall-clock + top-N gene-ranking agreement. Useful for regression
checks after touching the strata code or after re-converting a dataset.

Example invocations
-------------------

Comparing T cells vs B cells on the cell_type axis::

    uv run python scripts/eval_compare_groups_strata.py \\
        --dataset /path/to/dataset.zarr \\
        --obs-column cell_type \\
        --group-a "T cell" \\
        --group-b "B cell"

A dataset where the obs_column is only in the atomic table::

    uv run python scripts/eval_compare_groups_strata.py \\
        --dataset /path/to/dataset.zarr \\
        --obs-column Phase \\
        --group-a G1 \\
        --group-b S \\
        --n 30

Outputs a JSON dump alongside this script (``<basename>.last.json``) for
post-hoc inspection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import time
from pathlib import Path

from aiohttp import web

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.tools import build_v1_catalog
from cell_explorer_api.services.zarr_adapter import (
    AnnDataZarrAccess,
    StrataZarrAccess,
)
from zarr_access import AnnDataStore, StrataStore, ZarrStore


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def _serve(dataset_dir: Path) -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_static("/", dataset_dir, show_index=True)
    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return f"http://127.0.0.1:{port}", runner


def _summarize(label: str, elapsed: float, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  wall_clock_s: {elapsed:.3f}")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  method: {result['method']}")
    print(f"  top genes ({len(result['genes'])}):")
    for i, g in enumerate(result["genes"], 1):
        d = g.get("cohens_d") if "cohens_d" in g else g.get("log2_fold_change", float("nan"))
        print(f"    {i:2d}. {g['symbol']:<20} stat={d:+.3f}")


def _compare_top(result_x: dict, result_s: dict, n: int) -> None:
    print("\n=== AGREEMENT ===")
    if "error" in result_x or "error" in result_s:
        print("  (at least one tool errored — agreement not computed)")
        return
    genes_x = [g["symbol"] for g in result_x["genes"][:n]]
    genes_s = [g["symbol"] for g in result_s["genes"][:n]]
    overlap = set(genes_x) & set(genes_s)
    print(f"  top-{n} set overlap: {len(overlap)}/{n} ({100 * len(overlap) / n:.0f}%)")
    print(f"  top-1 match: {genes_x[0] == genes_s[0]} ({genes_x[0]!r} vs {genes_s[0]!r})")


async def _run(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    base_url, runner = await _serve(dataset_path.parent)
    try:
        url = f"{base_url}/{dataset_path.name}"
        print(f"serving from {url}")

        zarr_store = await ZarrStore.open(url)
        anndata = await AnnDataStore.open(zarr_store)
        strata_store = await StrataStore.open(zarr_store)
        adapter = AnnDataZarrAccess(anndata)
        strata_adapter = StrataZarrAccess(strata_store)

        with_strata = build_v1_catalog(adapter, config=AgentConfig(), strata=strata_adapter)
        without_strata = build_v1_catalog(adapter, config=AgentConfig(), strata=None)

        call_args = dict(
            obs_column=args.obs_column,
            group_a=args.group_a,
            group_b=args.group_b,
            n=args.n,
        )
        print(f"\nQuery: {call_args}")
        print(f"coarse slugs: {strata_adapter.coarse_slugs()}")
        print(f"atomic axes:  {strata_adapter.store.atomic_axes()}")

        t0 = time.perf_counter()
        result_s = await with_strata.get("compare_groups").func(**call_args)
        elapsed_s = time.perf_counter() - t0
        _summarize("compare_groups (with strata)", elapsed_s, result_s)

        t0 = time.perf_counter()
        result_x = await without_strata.get("compare_groups").func(**call_args)
        elapsed_x = time.perf_counter() - t0
        _summarize("compare_groups (X-scan)", elapsed_x, result_x)

        if elapsed_s > 0 and elapsed_x > 0:
            print("\n=== TIMING ===")
            print(f"  xscan:  {elapsed_x:.3f}s")
            print(f"  strata: {elapsed_s:.3f}s")
            print(f"  speedup: {elapsed_x / elapsed_s:.1f}x")

        _compare_top(result_x, result_s, n=args.n)

        out_path = Path(__file__).parent / f"{Path(__file__).stem}.last.json"
        out_path.write_text(json.dumps({
            "query": call_args,
            "strata": {"elapsed_s": elapsed_s, "result": result_s},
            "xscan": {"elapsed_s": elapsed_x, "result": result_x},
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await runner.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to a local zarr store (v3, AnnData-shaped, with uns/strata/ populated).",
    )
    p.add_argument("--obs-column", required=True, help="Categorical obs column to compare on.")
    p.add_argument("--group-a", required=True, help="First group value.")
    p.add_argument("--group-b", required=True, help="Second group value.")
    p.add_argument("--n", type=int, default=20, help="Top-N genes to compare (default 20).")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

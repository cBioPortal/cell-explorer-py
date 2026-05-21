"""Benchmark: row-selective atomic read vs full atomic read.

For a given (obs_column, group_a, group_b) query, times three things:

  1. ``read_atomic_stratum_keys()`` — the cheap bookkeeping read used by the
     two-phase pattern in compare_groups.
  2. ``read_atomic()`` — full atomic table load (the old path).
  3. ``read_atomic_rows(needed_rows)`` — row-selective read for the rows that
     match group_a / group_b on the queried axis.

Reports wall-clock, bytes transferred, % of atomic rows touched (selectivity),
and verifies (3) produces byte-identical sums to a numpy slice of (2).

Useful for:
  - Verifying a re-converted dataset's sharded atomic actually delivers
    selectivity-proportional reads.
  - Tuning the selectivity threshold in
    ``zarr_access.strata_store.ROWS_SLICED_SELECTIVITY_THRESHOLD`` (see #133).
  - Investigating perf regressions in the atomic read path.

Example invocations
-------------------

High-selectivity case (a query that touches a small fraction of atomic rows)::

    uv run python scripts/eval_atomic_row_selective.py \\
        --dataset /path/to/dataset.zarr \\
        --obs-column donor_id \\
        --group-a donor-001 \\
        --group-b donor-002

Low-selectivity case (a query that touches a large fraction of atomic rows)::

    uv run python scripts/eval_atomic_row_selective.py \\
        --dataset /path/to/dataset.zarr \\
        --obs-column Phase \\
        --group-a G1 \\
        --group-b S
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import time
from pathlib import Path

import numpy as np
from aiohttp import web

from zarr_access import StrataStore, ZarrStore


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


async def _run(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    base_url, runner = await _serve(dataset_path.parent)
    try:
        url = f"{base_url}/{dataset_path.name}"
        zs = await ZarrStore.open(url)
        strata = await StrataStore.open(zs)

        if not strata.has_atomic():
            raise SystemExit("dataset has no atomic strata table")
        axes = strata.atomic_axes() or []
        if args.obs_column not in axes:
            raise SystemExit(
                f"obs column {args.obs_column!r} not in atomic axes {axes}"
            )
        axis_idx = axes.index(args.obs_column)
        print(f"atomic axes:    {axes}")
        print(f"atomic n_strata: {strata.atomic_strata_count()}")

        # 1. Stratum-keys-only read (cheap bookkeeping)
        t0 = time.perf_counter()
        keys, n_cells = await strata.read_atomic_stratum_keys()
        t_keys = time.perf_counter() - t0
        print(f"\nstratum_keys read: {t_keys * 1000:.0f} ms")

        a_rows = np.where(keys[:, axis_idx] == args.group_a)[0].tolist()
        b_rows = np.where(keys[:, axis_idx] == args.group_b)[0].tolist()
        if not a_rows:
            raise SystemExit(
                f"no atomic rows match {args.obs_column}={args.group_a!r}"
            )
        if not b_rows:
            raise SystemExit(
                f"no atomic rows match {args.obs_column}={args.group_b!r}"
            )

        print(
            f"\nQuery: {args.obs_column}={args.group_a!r} "
            f"({len(a_rows)} atomic rows, {int(n_cells[a_rows].sum())} cells) "
            f"vs {args.obs_column}={args.group_b!r} "
            f"({len(b_rows)} rows, {int(n_cells[b_rows].sum())} cells)"
        )
        needed = a_rows + b_rows
        selectivity_pct = 100 * len(needed) / keys.shape[0]
        print(
            f"Need {len(needed)} of {keys.shape[0]} atomic rows "
            f"({selectivity_pct:.1f}% selectivity)"
        )

        # 2. Full read (the old path)
        print("\n--- full read_atomic() ---")
        t0 = time.perf_counter()
        full = await strata.read_atomic()
        t_full = time.perf_counter() - t0
        print(f"  wall_clock: {t_full:.2f}s")
        print(f"  sum_x shape: {full.sum_x.shape}, ~{full.sum_x.nbytes / 1e6:.0f} MB")

        # 3. Row-selective read
        print("\n--- two-phase read_atomic_rows() ---")
        t0 = time.perf_counter()
        slim = await strata.read_atomic_rows(needed)
        t_slim = time.perf_counter() - t0
        print(f"  wall_clock: {t_slim:.2f}s")
        print(f"  sum_x shape: {slim.sum_x.shape}, ~{slim.sum_x.nbytes / 1e6:.0f} MB")

        # Speedup
        print("\n--- speedup ---")
        print(f"  full read:   {t_full:.2f}s")
        print(f"  row-selective: {t_slim:.2f}s")
        if t_slim > 0:
            ratio = t_full / t_slim
            verdict = "faster" if ratio >= 1 else "SLOWER"
            print(f"  row-selective is {ratio:.1f}x {verdict} than full read")

        # Equivalence (regression guard)
        np.testing.assert_array_equal(slim.sum_x, full.sum_x[needed])
        np.testing.assert_array_equal(slim.sum_xx, full.sum_xx[needed])
        np.testing.assert_array_equal(slim.n_cells, full.n_cells[needed])
        print("\n=== EQUIVALENCE: row-selective matches full sliced byte-for-byte ===")
    finally:
        await runner.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to a local zarr store with uns/strata/atomic/ populated.",
    )
    p.add_argument(
        "--obs-column",
        required=True,
        help="Atomic axis to query (must be one of strata.atomic_axes()).",
    )
    p.add_argument("--group-a", required=True, help="First group value.")
    p.add_argument("--group-b", required=True, help="Second group value.")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

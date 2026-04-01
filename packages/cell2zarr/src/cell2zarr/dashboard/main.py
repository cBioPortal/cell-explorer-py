"""FastAPI dashboard for zarr conversion runs."""
import json
import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from cell2zarr.models import (
    RunEntry, RunDataset, RunPerformance, RunZarrConfig, RunConversionConfig,
)

APP_DIR = Path(__file__).resolve().parent

if "CELL2ZARR_RUN_DB" not in os.environ:
    raise RuntimeError("CELL2ZARR_RUN_DB environment variable is required. Use 'cell2zarr dashboard --run-db <path>' or set the env var directly.")

RUNS_FILE = Path(os.environ["CELL2ZARR_RUN_DB"])
LOGS_DIR = RUNS_FILE.parent / "logs"

app = FastAPI(title="Zarr Conversion Dashboard")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


# ── Data ─────────────────────────────────────────────────────────


def _read_runs() -> list[RunEntry]:
    if not RUNS_FILE.exists():
        return []
    with open(RUNS_FILE) as f:
        return [RunEntry.model_validate(r) for r in json.load(f)]


# ── Jinja2 filters ──────────────────────────────────────────────


def fmt_time(s):
    if s is None:
        return "—"
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def fmt_size(gb):
    if gb is None:
        return "—"
    return f"{gb * 1024:.0f} MB" if gb < 1 else f"{gb} GB"


def fmt_number(n):
    if n is None:
        return "—"
    return f"{n:,}" if isinstance(n, (int, float)) else str(n)


def fmt_datetime(iso):
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%b %-d, %I:%M %p")
    except Exception:
        return iso


def fmt_shape(arr):
    if not arr:
        return "—"
    parts = []
    for v in arr:
        if isinstance(v, (int, float)) and v > 9999:
            parts.append(f"{v / 1000:.0f}k")
        else:
            parts.append(str(v))
    return "(" + ", ".join(parts) + ")"


def fmt_size_auto(obj):
    if not obj or not isinstance(obj, dict):
        return "—"
    if obj.get("size_gb") is not None:
        return f"{obj['size_gb']} GB"
    if obj.get("size_mb") is not None:
        return f"{obj['size_mb']} MB"
    return "—"


def fmt_rate(perf):
    if perf is None:
        return "—"
    r = perf.phase2_rate_batches_per_min
    return f"{r} bat/min" if r else "—"


def build_cmd(args):
    if not args:
        return ""
    parts = ["cell2zarr", args.get("input", "")]
    if args.get("output"):
        parts.append(args["output"])
    for k, v in args.items():
        if k in ("input", "output"):
            continue
        if v is True:
            parts.append(k)
        elif v is not False and v is not None:
            parts.append(f"{k} {v}")
    return " ".join(parts)


def fmt_compressor(comp):
    if comp is None:
        return "—"
    if isinstance(comp, dict):
        name, level = comp.get("name", "?"), comp.get("level")
    else:
        name, level = getattr(comp, "name", "?"), getattr(comp, "level", None)
    return f"{name} L{level}" if level else name


for _name, _fn in [
    ("fmt_time", fmt_time), ("fmt_size", fmt_size), ("fmt_number", fmt_number),
    ("fmt_datetime", fmt_datetime), ("fmt_shape", fmt_shape), ("fmt_size_auto", fmt_size_auto),
    ("fmt_rate", fmt_rate), ("build_cmd", build_cmd), ("fmt_compressor", fmt_compressor),
]:
    templates.env.filters[_name] = _fn


# ── Routes ───────────────────────────────────────────────────────


@app.get("/")
async def index(request: Request):
    runs = _read_runs()
    completed = [r for r in runs if r.status == "completed"]
    times = [r.performance.total_time_s for r in completed if r.performance and r.performance.total_time_s]
    ratios = [r.performance.compression_ratio for r in completed if r.performance and r.performance.compression_ratio]

    summary = {
        "total": len(runs),
        "completed": len(completed),
        "fastest": min(times) if times else None,
        "best_ratio": max(ratios) if ratios else None,
    }

    runs_json = json.dumps([r.model_dump(exclude_none=True) for r in runs])

    return templates.TemplateResponse(request, "index.html", context={
        "runs": runs,
        "runs_reversed": list(reversed(runs)),
        "summary": summary,
        "runs_json": runs_json,
    })


@app.get("/api/runs")
async def api_runs():
    runs = _read_runs()
    return [r.model_dump(exclude_none=True) for r in runs]


def _get_log_path(run_id: int) -> Path | None:
    """Look up the log file path from the run-db entry."""
    runs = _read_runs()
    for r in runs:
        if r.run == run_id:
            if r.log_file:
                log_path = Path(r.log_file)
                if log_path.is_absolute():
                    return log_path
                return RUNS_FILE.parent / log_path
            # Fallback to legacy path
            return LOGS_DIR / f"run-{run_id}.log"
    return None


@app.get("/api/logs/{run_id}")
async def api_logs(run_id: int):
    log_path = _get_log_path(run_id)
    if log_path is None or not log_path.exists():
        return PlainTextResponse(f"Log not found for run {run_id}", status_code=404)
    return PlainTextResponse(
        log_path.read_text(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/logs/{run_id}/stream")
async def api_logs_stream(run_id: int):
    """SSE endpoint that tails the log file for a running conversion."""
    import asyncio
    from starlette.responses import StreamingResponse

    log_path = _get_log_path(run_id)
    if log_path is None or not log_path.exists():
        return PlainTextResponse(f"Log not found for run {run_id}", status_code=404)

    async def stream():
        with open(log_path) as f:
            # Send existing content first
            for line in f:
                yield f"data: {line.rstrip()}\n\n"

            # Then tail for new lines
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    # Check if run is still active
                    runs = _read_runs()
                    run = next((r for r in runs if r.run == run_id), None)
                    if run and run.status in ("completed", "failed", "cancelled", "killed"):
                        yield "event: done\ndata: Run finished\n\n"
                        break
                    await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")

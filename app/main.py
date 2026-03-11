"""FastAPI dashboard for zarr conversion runs."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RUNS_FILE = PROJECT_ROOT / "docs" / "conversion-runs.json"
LOGS_DIR = PROJECT_ROOT / "docs" / "logs"

app = FastAPI(title="Zarr Conversion Dashboard")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


# ── Pydantic models (mirrored from convert_h5ad_to_zarr.py) ────

RunStatus = Literal["running", "phase2", "completed", "failed", "cancelled", "killed"]


class RunDataset(BaseModel):
    model_config = {"extra": "allow"}
    n_obs: int | None = None
    n_vars: int | None = None
    input_size_gb: float | None = None
    input_format: str | None = None


class RunPerformance(BaseModel):
    model_config = {"extra": "allow"}
    start_time: str | None = None
    end_time: str | None = None
    phase1_time_s: int | None = None
    phase2_time_s: int | None = None
    total_time_s: int | None = None
    output_size_gb: float | None = None
    avg_chunk_size_mb: float | None = None
    compression_ratio: float | None = None
    phase2_rate_batches_per_min: float | None = None


class RunZarrConfig(BaseModel):
    model_config = {"extra": "allow"}
    format: str | None = None
    chunk_shape: list[int] | None = None
    sharding: list[int] | None = None
    dtype: str | None = None
    compression: str | None = None
    target_encoding: dict[str, Any] | None = None
    actual_encoding: dict[str, Any] | None = None


class RunConversionConfig(BaseModel):
    model_config = {"extra": "allow"}
    approach: str | None = None
    cell_chunk_size: int | None = None
    temp_var_chunk: int | None = None
    codec_pipeline: str | None = None
    threads: int | None = None
    skip_layers: bool | None = None
    obsm_keys: list[str] | None = None
    phase2_read_batch: int | None = None


class RunEntry(BaseModel):
    model_config = {"extra": "allow"}
    run: int
    date: str | None = None
    status: RunStatus = "running"
    script_args: dict[str, Any] | None = None
    zarr_config: RunZarrConfig | None = None
    conversion_config: RunConversionConfig | None = None
    dataset: RunDataset | None = None
    performance: RunPerformance | None = None
    notes: str = ""
    log_file: str | None = None


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
    parts = ["python scripts/convert_h5ad_to_zarr.py", args.get("input", "")]
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

    return templates.TemplateResponse("index.html", {
        "request": request,
        "runs": runs,
        "runs_reversed": list(reversed(runs)),
        "summary": summary,
        "runs_json": runs_json,
    })


@app.get("/api/runs")
async def api_runs():
    runs = _read_runs()
    return [r.model_dump(exclude_none=True) for r in runs]


@app.get("/api/logs/{run_id}")
async def api_logs(run_id: int):
    log_file = LOGS_DIR / f"run-{run_id}.log"
    if not log_file.exists():
        return PlainTextResponse(f"Log not found: run-{run_id}.log", status_code=404)
    return PlainTextResponse(
        log_file.read_text(),
        headers={"Cache-Control": "no-store"},
    )

"""Small FastAPI surface for health checks and local profile inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from barellm.config import SETTINGS, Settings

_STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True)
class ProfileRun:
    """A profile directory exposed by the local profile API."""

    run_id: str
    model: str
    path: Path


def _is_run_directory(path: Path) -> bool:
    return path.is_dir() and any(
        (path / filename).is_file()
        for filename in ("metrics.json", "engine.trace.json", "torch.trace.json")
    )


def _is_within(root: Path, path: Path) -> bool:
    root = root.resolve()
    resolved = path.resolve()
    return resolved != root and root in resolved.parents


def discover_profiles(root: Path) -> list[ProfileRun]:
    """Discover timestamped runs and the older flat profile layout."""
    if not root.is_dir():
        return []

    runs: list[ProfileRun] = []
    for model_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        if not _is_within(root, model_dir):
            continue
        if _is_run_directory(model_dir):
            runs.append(ProfileRun(model_dir.name, model_dir.name, model_dir))
            continue
        for run_dir in sorted(
            (item for item in model_dir.iterdir() if item.is_dir()),
            reverse=True,
        ):
            if _is_within(root, run_dir) and _is_run_directory(run_dir):
                runs.append(
                    ProfileRun(
                        f"{model_dir.name}/{run_dir.name}", model_dir.name, run_dir
                    )
                )
    return sorted(
        runs,
        key=lambda run: (run.path.stat().st_mtime, run.run_id),
        reverse=True,
    )


def _public_run(run: ProfileRun) -> dict[str, Any]:
    files = {
        name: (run.path / filename).is_file()
        for name, filename in (
            ("metrics", "metrics.json"),
            ("engine_trace", "engine.trace.json"),
            ("torch_trace", "torch.trace.json"),
        )
    }
    encoded_id = quote(run.run_id, safe="")
    return {
        "id": run.run_id,
        "model": run.model,
        "name": run.path.name,
        "files": files,
        "urls": {
            "metrics": f"/api/profiles/{encoded_id}/metrics",
            "engine_trace": f"/api/profiles/{encoded_id}/trace/engine",
            "torch_trace": f"/api/profiles/{encoded_id}/trace/torch",
        },
    }


def _find_profile(root: Path, run_id: str) -> ProfileRun:
    """Resolve an API id while keeping it inside the configured root."""
    normalized_id = run_id.replace("%2F", "/").replace("%2f", "/")
    root = root.resolve()
    candidate = (root / normalized_id).resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="profile run not found")
    for run in discover_profiles(root):
        if run.run_id == normalized_id and run.path.resolve() == candidate:
            return run
    raise HTTPException(status_code=404, detail="profile run not found")


def _read_metrics(run: ProfileRun) -> dict[str, Any]:
    path = run.path / "metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="metrics not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="invalid metrics file") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="invalid metrics file")
    return payload


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the BareLLM HTTP application without starting a server."""
    current = settings or SETTINGS
    app = FastAPI(title="BareLLM")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    if current.enable_profile_api:
        root = current.profile_root
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/api/profiles")
        def profiles(
            limit: int = Query(default=50, ge=1, le=100),
            offset: int = Query(default=0, ge=0),
        ) -> dict[str, Any]:
            all_runs = discover_profiles(root)
            page = all_runs[offset : offset + limit]
            return {
                "runs": [_public_run(run) for run in page],
                "total": len(all_runs),
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(page) < len(all_runs),
            }

        @app.get("/api/profiles/{run_id:path}/metrics")
        def metrics(run_id: str) -> dict[str, Any]:
            return _read_metrics(_find_profile(root, run_id))

        @app.get("/api/profiles/{run_id:path}/trace/{trace_name}")
        def trace(run_id: str, trace_name: str) -> FileResponse:
            filename = {"engine": "engine.trace.json", "torch": "torch.trace.json"}.get(
                trace_name
            )
            if filename is None:
                raise HTTPException(status_code=404, detail="trace not found")
            path = _find_profile(root, run_id).path / filename
            if not path.is_file():
                raise HTTPException(status_code=404, detail="trace not found")
            return FileResponse(path, media_type="application/json", filename=filename)

        @app.get("/profiles")
        def dashboard() -> FileResponse:
            return FileResponse(_STATIC_DIR / "dashboard.html", media_type="text/html")

    return app


def run_server(settings: Settings | None = None) -> None:
    """Start the configured Uvicorn server."""
    import uvicorn

    current = settings or SETTINGS
    uvicorn.run(create_app(current), host=current.host, port=current.port)

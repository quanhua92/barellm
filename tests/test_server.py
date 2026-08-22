import json
import os
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from barellm.config import Settings
from barellm.web import create_app


def _settings(root: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        model_id="test/model",
        device="cpu",
        dtype=torch.float32,
        host="127.0.0.1",
        port=8000,
        profile_root=root,
        enable_profile_api=enabled,
    )


def test_profile_api_discovers_metrics_and_trace(tmp_path: Path) -> None:
    run_dir = tmp_path / "qwen3-0.6b" / "2026-08-21T12-00-00-cpu"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"metrics": {"generated_tokens": 3}, "metadata": {"device": "cpu"}}),
        encoding="utf-8",
    )
    (run_dir / "engine.trace.json").write_text('{"traceEvents": []}', encoding="utf-8")

    client = TestClient(create_app(_settings(tmp_path)))
    assert client.get("/health").json() == {"status": "ok"}

    response = client.get("/api/profiles")
    assert response.status_code == 200
    run = response.json()["runs"][0]
    assert run["id"] == "qwen3-0.6b/2026-08-21T12-00-00-cpu"
    assert run["files"] == {
        "metrics": True,
        "engine_trace": True,
        "torch_trace": False,
    }

    metrics = client.get(run["urls"]["metrics"])
    assert metrics.status_code == 200
    assert metrics.json()["metrics"]["generated_tokens"] == 3

    trace = client.get(run["urls"]["engine_trace"])
    assert trace.status_code == 200
    assert trace.json() == {"traceEvents": []}
    dashboard = client.get("/profiles").text
    assert "BareLLM Profiles" in dashboard
    assert "/static/profiles.css" in dashboard
    assert "/static/profiles.js" in dashboard
    javascript = client.get("/static/profiles.js").text
    stylesheet = client.get("/static/profiles.css").text
    assert "setInterval(sendPing, 100)" in javascript
    assert "keepApiOpen: true" in javascript
    assert "if (run.files.engine_trace) await openTrace('engine')" in javascript
    assert "Time to first token" in dashboard
    assert 'class="panel recent-panel"' in dashboard
    assert 'class="panel sidebar"' not in dashboard
    assert "Raw metrics JSON" in dashboard
    assert "@media (max-width: 520px)" in stylesheet


def test_profile_api_supports_legacy_flat_profile_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "qwen3"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    client = TestClient(create_app(_settings(tmp_path)))
    assert client.get("/api/profiles").json()["runs"][0]["id"] == "qwen3"


def test_profile_api_returns_newest_first_with_pagination(tmp_path: Path) -> None:
    for index in range(3):
        run_dir = tmp_path / "qwen3" / f"run-{index}"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        os.utime(run_dir, (index, index))

    client = TestClient(create_app(_settings(tmp_path)))
    first_page = client.get("/api/profiles?limit=2").json()
    second_page = client.get("/api/profiles?limit=2&offset=2").json()

    assert [run["id"] for run in first_page["runs"]] == ["qwen3/run-2", "qwen3/run-1"]
    assert first_page["total"] == 3
    assert first_page["has_more"] is True
    assert [run["id"] for run in second_page["runs"]] == ["qwen3/run-0"]
    assert second_page["has_more"] is False


def test_profile_api_can_be_disabled(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path, enabled=False)))

    assert client.get("/health").status_code == 200
    assert client.get("/api/profiles").status_code == 404
    assert client.get("/profiles").status_code == 404


def test_profile_api_rejects_unknown_and_traversal_runs(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    assert client.get("/api/profiles/missing/metrics").status_code == 404
    assert client.get("/api/profiles/%2E%2E/metrics").status_code == 404
    assert client.get("/api/profiles/missing/trace/unknown").status_code == 404

from __future__ import annotations

from fastapi.testclient import TestClient

from agiresearch.api.main import app
from agiresearch.config import settings

AUTH = {"Authorization": f"Bearer {settings.api_bearer_token}"}


def test_health_endpoint_requires_no_auth():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_endpoints_reject_missing_or_bad_auth():
    with TestClient(app) as client:
        assert client.get("/research").status_code == 401
        assert client.get("/research", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_submit_get_and_list_research_run_flow():
    with TestClient(app) as client:
        submit_response = client.post(
            "/research", json={"objective": "AGI progress in meta-learning"}, headers=AUTH
        )
        assert submit_response.status_code == 202
        body = submit_response.json()
        request_id = body["request_id"]
        assert request_id

        # BackgroundTasks complete before TestClient returns control here.
        detail_response = client.get(f"/research/{request_id}", headers=AUTH)
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["status"] == "completion"
        assert detail["paper_count"] > 0
        assert detail["average_agi_score"] is not None
        assert detail["final_report"] is not None
        assert len(detail["evaluated_papers"]) == detail["paper_count"]

        list_response = client.get("/research", headers=AUTH)
        assert list_response.status_code == 200
        summaries = list_response.json()
        assert any(s["request_id"] == request_id for s in summaries)


def test_get_unknown_run_is_404():
    with TestClient(app) as client:
        response = client.get("/research/does-not-exist", headers=AUTH)
    assert response.status_code == 404


def test_submit_rejects_empty_objective():
    with TestClient(app) as client:
        response = client.post("/research", json={"objective": ""}, headers=AUTH)
    assert response.status_code == 422

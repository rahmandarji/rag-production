from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_pipeline
from app.main import app


def test_query_rejects_empty_query() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: object()

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/query",
            json={"query": ""},
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_query_rejects_whitespace_query() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: object()

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/query",
            json={"query": "   "},
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_query_rejects_zero_retrieval_limit() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: object()

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/query",
            json={
                "query": "authentication",
                "retrieval_limit": 0,
            },
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_query_rejects_retrieval_limit_above_max() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: object()

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/query",
            json={
                "query": "authentication",
                "retrieval_limit": 21,
            },
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_query_rejects_excessively_long_query() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: object()

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/query",
            json={
                "query": "x" * 2001,
            },
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_unknown_route_returns_404() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404

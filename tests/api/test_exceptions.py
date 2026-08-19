from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.exceptions import internal_exception_handler


def test_internal_exception_handler_hides_internal_error() -> None:
    app = FastAPI()
    app.add_exception_handler(Exception, internal_exception_handler)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret database failure")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal server error occurred."
    }

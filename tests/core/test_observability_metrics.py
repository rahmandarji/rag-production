import pytest

from app.core.metrics import Metrics


def test_metrics_increment() -> None:
    metrics = Metrics()

    metrics.increment("requests_total")
    metrics.increment("requests_total")
    metrics.increment("rag_queries_total")

    assert metrics.snapshot() == {
        "requests_total": 2,
        "requests_failed": 0,
        "rag_queries_total": 1,
        "rag_refusals_total": 0,
        "rag_grounded_total": 0,
        "retrieval_empty_total": 0,
    }


def test_metrics_reject_unknown_metric() -> None:
    metrics = Metrics()

    with pytest.raises(ValueError, match="Unknown metric"):
        metrics.increment("does_not_exist")


def test_metrics_snapshot_is_independent() -> None:
    metrics = Metrics()

    snapshot = metrics.snapshot()
    snapshot["requests_total"] = 999

    assert metrics.snapshot()["requests_total"] == 0

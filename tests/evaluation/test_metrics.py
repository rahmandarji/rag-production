from app.evaluation.metrics import (
    reciprocal_rank,
    retrieval_recall,
    token_f1,
)


def test_retrieval_recall() -> None:
    assert retrieval_recall(
        ["chunk-1", "chunk-2"],
        ["chunk-2"],
    ) == 1.0


def test_retrieval_recall_partial() -> None:
    assert retrieval_recall(
        ["chunk-1"],
        ["chunk-1", "chunk-2"],
    ) == 0.5


def test_retrieval_recall_empty_expected() -> None:
    assert retrieval_recall(
        [],
        [],
    ) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(
        ["chunk-3", "chunk-2", "chunk-1"],
        ["chunk-2"],
    ) == 0.5


def test_reciprocal_rank_miss() -> None:
    assert reciprocal_rank(
        ["chunk-1"],
        ["chunk-2"],
    ) == 0.0


def test_token_f1_exact_match() -> None:
    assert token_f1(
        "Path parameters are declared.",
        "Path parameters are declared.",
    ) == 1.0


def test_token_f1_no_overlap() -> None:
    assert token_f1(
        "OAuth authentication",
        "database indexing",
    ) == 0.0

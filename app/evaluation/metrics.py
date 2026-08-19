import re


def normalize_text(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def token_f1(
    prediction: str,
    reference: str,
) -> float:
    prediction_tokens = normalize_text(prediction)
    reference_tokens = normalize_text(reference)

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts: dict[str, int] = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(token, 0) + 1
        )

    reference_counts: dict[str, int] = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(token, 0) + 1
        )

    overlap = sum(
        min(
            prediction_counts.get(token, 0),
            count,
        )
        for token, count in reference_counts.items()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def retrieval_recall(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    if not expected_ids:
        return 1.0

    retrieved = set(retrieved_ids)
    expected = set(expected_ids)

    return len(retrieved & expected) / len(expected)


def reciprocal_rank(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    expected = set(expected_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank

    return 0.0

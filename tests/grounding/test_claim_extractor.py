from app.grounding.claim_extractor import ClaimExtractor


def test_claim_extractor_splits_sentences() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        "FastAPI uses path parameters. They are declared in the route."
    )

    assert result == [
        "FastAPI uses path parameters.",
        "They are declared in the route.",
    ]


def test_claim_extractor_handles_empty_answer() -> None:
    extractor = ClaimExtractor()

    assert extractor.extract("   ") == []

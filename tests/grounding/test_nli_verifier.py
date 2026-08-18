from app.grounding.nli_verifier import NLIVerifier


def test_nli_verifier_detects_supported_claim() -> None:
    verifier = NLIVerifier()

    result = verifier.verify(
        claim="You define path parameters inside the route.",
        evidence=[
            "Path parameters are declared directly in the route path.",
            "FastAPI was created in 2018.",
        ],
    )

    assert result
    assert result[0][0] == (
        "Path parameters are declared directly in the route path."
    )


def test_nli_verifier_returns_empty_for_no_evidence() -> None:
    verifier = NLIVerifier()

    assert verifier.verify(
        claim="FastAPI was created in 2018.",
        evidence=[],
    ) == []

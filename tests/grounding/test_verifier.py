from app.grounding.verifier import ClaimVerifier


def test_claim_verifier_is_abstract() -> None:
    assert ClaimVerifier.__abstractmethods__ == {"verify"}

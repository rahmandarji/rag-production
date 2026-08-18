from app.grounding.models import ClaimVerification, GroundingResult


def test_grounding_result_preserves_support() -> None:
    result = GroundingResult(
        grounded=True,
        claims=[
            ClaimVerification(
                claim="Path parameters are declared in the route.",
                grounded=True,
                supporting_chunk_ids=["doc-001:chunk-0001"],
            )
        ],
    )

    assert result.grounded is True
    assert len(result.claims) == 1
    assert result.claims[0].supporting_chunk_ids == [
        "doc-001:chunk-0001"
    ]

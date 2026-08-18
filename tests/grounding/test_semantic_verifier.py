from app.grounding.semantic_verifier import SemanticVerifier


def test_semantic_verifier_is_an_abstract_interface() -> None:
    assert SemanticVerifier.__abstractmethods__ == {"verify"}

import pytest

from app.generation.qwen import QwenGenerationProvider


def test_qwen_rejects_invalid_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.generation.qwen.AutoTokenizer.from_pretrained",
        lambda _: None,
    )
    monkeypatch.setattr(
        "app.generation.qwen.AutoModelForCausalLM.from_pretrained",
        lambda _: None,
    )

    with pytest.raises(ValueError, match="max_new_tokens"):
        QwenGenerationProvider(max_new_tokens=0)


def test_qwen_returns_refusal_without_evidence() -> None:
    provider = QwenGenerationProvider.__new__(QwenGenerationProvider)

    result = provider.generate(
        query="Who founded the company?",
        evidence=[],
    )

    assert "don't have enough information" in result.answer
    assert result.sources == []


def test_qwen_rejects_empty_query() -> None:
    provider = QwenGenerationProvider.__new__(QwenGenerationProvider)

    with pytest.raises(ValueError):
        provider.generate(
            query="   ",
            evidence=[],
        )

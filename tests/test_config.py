from app.core.config import settings


def test_settings_defaults() -> None:
    assert settings.app_name == "rag-production"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_settings_reject_invalid_limits() -> None:
    import pytest
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(embedding_dimension=0)

    with pytest.raises(ValidationError):
        Settings(max_document_size_bytes=0)

    with pytest.raises(ValidationError):
        Settings(max_documents_per_batch=0)

    with pytest.raises(ValidationError):
        Settings(max_chunks_per_document=0)

    with pytest.raises(ValidationError):
        Settings(max_query_length=0)


def test_settings_reject_retrieval_limit_above_retrieval_k() -> None:
    import pytest

    from app.core.config import Settings

    with pytest.raises(ValueError, match="max_retrieval_limit"):
        Settings(
            retrieval_k=5,
            max_retrieval_limit=10,
        )


def test_environment_is_validated() -> None:
    from pydantic import ValidationError

    from app.core.config import Settings

    try:
        Settings(environment="invalid")
    except ValidationError:
        return

    raise AssertionError("Invalid environment should be rejected.")


def test_log_level_is_normalized() -> None:
    from app.core.config import Settings

    config = Settings(log_level="warning")

    assert config.log_level == "WARNING"


def test_retrieval_limits_are_validated() -> None:
    from pydantic import ValidationError

    from app.core.config import Settings

    try:
        Settings(
            retrieval_k=5,
            max_retrieval_limit=10,
        )
    except ValidationError:
        return

    raise AssertionError(
        "max_retrieval_limit should not exceed retrieval_k."
    )


def test_environment_variable_prefix(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.setenv("RAG_ENVIRONMENT", "production")

    config = Settings()

    assert config.environment == "production"

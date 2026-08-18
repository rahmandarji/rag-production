from app.core.config import settings


def test_settings_defaults() -> None:
    assert settings.app_name == "rag-production"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"

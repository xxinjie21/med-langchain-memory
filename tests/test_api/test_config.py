"""全局配置单元测试（D30）。

覆盖：默认值与包版本一致性、环境变量覆盖、非法值拒绝、文档 URL 开关分支、
不可变语义，以及 logger / 环境变量前缀常量。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

import med_langchain_memory
from med_langchain_memory.config import (
    ACCESS_LOGGER_NAME,
    ENV_PREFIX,
    PACKAGE_LOGGER_NAME,
    MedMemorySettings,
)


def test_defaults_are_stable() -> None:
    """无环境变量时使用默认值，且版本号与包版本一致。"""
    settings = MedMemorySettings()
    assert settings.app_name == "med-langchain-memory"
    assert settings.app_version == med_langchain_memory.__version__
    assert settings.log_level == "INFO"
    assert settings.request_id_header == "X-Request-ID"
    assert settings.docs_enabled is True


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MED_MEMORY_`` 前缀环境变量可覆盖全部字段。"""
    monkeypatch.setenv("MED_MEMORY_APP_NAME", "hospital-gateway")
    monkeypatch.setenv("MED_MEMORY_APP_VERSION", "9.9.9")
    monkeypatch.setenv("MED_MEMORY_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MED_MEMORY_REQUEST_ID_HEADER", "X-Trace-ID")
    monkeypatch.setenv("MED_MEMORY_DOCS_ENABLED", "false")

    settings = MedMemorySettings()
    assert settings.app_name == "hospital-gateway"
    assert settings.app_version == "9.9.9"
    assert settings.log_level == "DEBUG"
    assert settings.request_id_header == "X-Trace-ID"
    assert settings.docs_enabled is False


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """非法日志级别（边界：非 Literal 成员）直接抛校验异常。"""
    monkeypatch.setenv("MED_MEMORY_LOG_LEVEL", "LOUD")
    with pytest.raises(PydanticValidationError):
        MedMemorySettings()


def test_blank_app_name_is_rejected() -> None:
    """空服务名（边界：min_length 下界）被拒绝。"""
    with pytest.raises(PydanticValidationError):
        MedMemorySettings(app_name="")


def test_blank_request_id_header_is_rejected() -> None:
    """空请求 ID 头名（边界：min_length 下界）被拒绝。"""
    with pytest.raises(PydanticValidationError):
        MedMemorySettings(request_id_header="")


def test_doc_urls_when_docs_enabled() -> None:
    """``docs_enabled=True`` 时三个文档路径均可用。"""
    settings = MedMemorySettings(docs_enabled=True)
    assert settings.docs_url == "/docs"
    assert settings.redoc_url == "/redoc"
    assert settings.openapi_url == "/openapi.json"


def test_doc_urls_when_docs_disabled() -> None:
    """``docs_enabled=False`` 时三个文档路径均为 ``None``（关闭挂载）。"""
    settings = MedMemorySettings(docs_enabled=False)
    assert settings.docs_url is None
    assert settings.redoc_url is None
    assert settings.openapi_url is None


def test_settings_are_frozen() -> None:
    """配置对象不可变（边界：字段赋值应被拒绝）。"""
    settings = MedMemorySettings()
    with pytest.raises(PydanticValidationError):
        settings.app_name = "other"  # type: ignore[misc]


def test_logger_and_env_prefix_constants() -> None:
    """logger 命名与访问 logger 均以包级 logger 为前缀。"""
    assert PACKAGE_LOGGER_NAME == "med_langchain_memory"
    assert ACCESS_LOGGER_NAME == PACKAGE_LOGGER_NAME + ".api.access"
    assert ENV_PREFIX == "MED_MEMORY_"

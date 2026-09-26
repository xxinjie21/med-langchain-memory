"""全局配置（pydantic-settings）。

配置来源优先级：显式构造参数 > 环境变量（前缀 ``MED_MEMORY_``）> 字段默认值。
**不读取 ``.env`` 文件**，避免测试环境与部署环境之间出现隐式差异（部署侧请通过
环境变量或容器编排注入）。

设计取舍：只承载「进程级、可静态校验」的少量开关；不引入配置中心、不做热更新、
不做多环境 profile。本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from med_langchain_memory import __version__

#: 包级 logger 名（所有子模块 logger 均以它为前缀）。
PACKAGE_LOGGER_NAME = "med_langchain_memory"

#: 访问日志 logger 名。
ACCESS_LOGGER_NAME = f"{PACKAGE_LOGGER_NAME}.api.access"

#: 环境变量前缀。
ENV_PREFIX = "MED_MEMORY_"

#: 合法日志级别（直接复用标准库 ``logging`` 的级别名）。
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class MedMemorySettings(BaseSettings):
    """服务级配置。

    所有字段均可通过 ``MED_MEMORY_<字段名大写>`` 形式的环境变量覆盖，
    例如 ``MED_MEMORY_LOG_LEVEL=DEBUG``、``MED_MEMORY_DOCS_ENABLED=false``。

    Attributes:
        app_name: 服务名，用于 OpenAPI 标题与健康检查响应。
        app_version: 服务版本，默认与包版本保持一致。
        log_level: 包级 logger 的日志级别。
        request_id_header: 请求追踪 ID 的 HTTP 头名（透传已有值或自动生成）。
        docs_enabled: 是否挂载 ``/docs``、``/redoc`` 与 OpenAPI 文档。
    """

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore", frozen=True)

    app_name: str = Field(default="med-langchain-memory", min_length=1)
    app_version: str = Field(default=__version__, min_length=1)
    log_level: LogLevel = "INFO"
    request_id_header: str = Field(default="X-Request-ID", min_length=1)
    docs_enabled: bool = True

    @property
    def docs_url(self) -> str | None:
        """Swagger UI 路径；``docs_enabled`` 为假时返回 ``None`` 表示关闭。"""
        return "/docs" if self.docs_enabled else None

    @property
    def redoc_url(self) -> str | None:
        """ReDoc 路径；``docs_enabled`` 为假时返回 ``None`` 表示关闭。"""
        return "/redoc" if self.docs_enabled else None

    @property
    def openapi_url(self) -> str | None:
        """OpenAPI JSON 路径；``docs_enabled`` 为假时返回 ``None`` 表示关闭。"""
        return "/openapi.json" if self.docs_enabled else None

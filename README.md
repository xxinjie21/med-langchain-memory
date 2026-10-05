# med-langchain-memory

[![CI](https://github.com/xxinjie21/med-langchain-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/xxinjie21/med-langchain-memory/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/types-mypy%20strict-blue.svg)](https://mypy-lang.org/)

> 医疗专属 LangChain 分布式会话存储中间件 · Medical-grade distributed chat message history middleware for LangChain（Python 3.11+ / LCEL）

面向医院医患问诊会话的**存储中间件**——只负责对话历史的「存、管、取」，不内置任何 NLP / 中文分词 / 实体抽取逻辑。上层由 [`springai-med-qa`](https://github.com/xxinjie21/springai-med-qa)（Java 问诊后端）消费，两仓库存储字段、键规范、序列化协议严格对齐，数据可互通。

📖 **文档**：[对外存储规范](./docs/storage-spec.md) ｜ [架构与设计决策](./docs/architecture.md) ｜ [迭代路线图](./ROADMAP.md)

---

## 核心定位

| 维度 | 说明 |
|---|---|
| 角色 | AI 问诊系统的「记忆系统」——LLM 无状态，本库负责把对话规范地存好、管好、按需取回 |
| 职责边界 | 数据「肉身」始终躺在 Redis / MySQL / ES 等真实数据库里，本库只做调度与规则层（中间件） |
| 合规 | 字段级正则脱敏、多租户科室隔离、到期自动归档，满足医疗数据留存要求 |
| 约束 | 全程零 NLP 依赖，隐私处理仅字段级正则规则 |

---

## 特性矩阵

| 能力 | 说明 |
|---|---|
| 多存储适配 | 五种已注册后端（内存 / 文件 / Redis 单机 / Redis 集群 / ES 归档）+ MySQL 16 张分表结构定义与 hash 路由，装饰器注册、热插拔 |
| 统一序列化协议 | Protobuf 二进制 + 冻结字段号，跨语言（Java / Go / Python）互通 |
| 会话生命周期 | TTL 自动归档、软删除与合规保留期、快照备份/恢复、跨存储迁移（断点续传） |
| 上下文工程 | 时序窗口裁剪 → LLM 摘要压缩 → Token 预算裁剪，三级可组合流水线 |
| 并发与可用性 | Redis 分布式会话锁（看门狗续期 + 本地降级）、主备降级 + 熔断器 |
| 隐私合规 | 字段级正则脱敏（手机号 / 身份证 / 病历号 / 床号），策略可按租户组合 |
| API 层 | FastAPI 会话/消息/管理端点、API Key 鉴权 + 科室 scope 校验、统一异常与请求日志 |
| 工程质量 | 全量 pytest（fakeredis / SQLite 内存库替身，无需真实中间件）+ GitHub Actions CI |

---

## 技术栈

| 层次 | 技术 | 用途 |
|---|---|---|
| 框架基础 | LangChain（`langchain-core`） | 实现 `BaseChatMessageHistory`，增强 `RunnableWithMessageHistory` |
| 数据建模 | Pydantic v2 + pydantic-settings | 消息 / 会话强类型实体、配置管理 |
| 序列化 | Protobuf | 统一二进制编码，跨语言与 Java 项目互通 |
| 存储引擎 | redis-py（Cluster） / SQLAlchemy + MySQL / elasticsearch-py | 热会话、持久化分表、冷归档 |
| Token 计算 | tiktoken | 上下文按 Token 预算裁剪 |
| API 层 | FastAPI | 轻量会话管理接口（增删查、迁移、归档触发） |
| 测试 | pytest + fakeredis + sqlite 内存库 | 全量单测，不依赖真实中间件 |
| CI | GitHub Actions | 多版本矩阵测试 + 覆盖率门禁 + lint/type-check |

---

## 存储后端

后端通过 `@StoreFactory.register("<name>")` 装饰器自注册，`StoreFactory.available()` 返回当前
进程实际可用的后端（可选依赖缺失时该后端不注册，其余后端不受影响）。

| 后端名 | 实现 | 说明 | 依赖 |
|---|---|---|---|
| `memory` | `InMemoryMedHistory` | 进程内内存存储，测试与本地开发的行为基准 | 无 |
| `file` | `FileMedHistory` | 本地文件，支持 JSONL 追加与 protobuf 二进制两种模式 + 跨进程文件锁 | 无 |
| `redis` | `RedisMedHistory` | Redis 单机热会话存储，pipeline 批量写 + 原生 TTL 滑动续期 | `[redis]` |
| `redis-cluster` | `RedisClusterMedHistory` | Redis 集群，`{session_id}` hash tag 保证同会话同 slot | `[redis]` |
| `elasticsearch` | `EsArchiveMedHistory` | 冷归档，按月滚动索引 + bulk 批量写入 | `[es]` |

```python
from med_langchain_memory.stores import StoreFactory

StoreFactory.available()  # 例：['elasticsearch', 'file', 'memory', 'redis', 'redis-cluster']
```

> **MySQL 分表**当前提供**结构定义与 hash 路由**（`stores/mysql_schema.py` 的 16 张同构分表
> DDL、`stores/mysql_shard_router.py` 的 `crc32(session_id) % 16` 路由），
> 尚未注册为独立的 history 适配器；分表列定义见 [存储规范 §7.2](./docs/storage-spec.md)。

---

## 架构一览

```
api/        FastAPI 接入层（health · sessions · messages · admin，可选依赖）
  ▲
runnable/   LCEL 上下文工程：租户隔离 · 时序裁剪 · Token 预算 · 摘要压缩 · 会话锁 · 降级熔断
  ▲
privacy/    字段级正则脱敏（零 NLP）
lifecycle/  TTL 归档 · 合规保留 · 快照 · 跨存储迁移
  ▲
stores/     存储适配层：内存 / 文件 / Redis 单机 / Redis 集群 / MySQL 分表 / ES 归档
  ▲
serde/      Serializer 抽象 + ProtobufSerializer
  ▲
domain/     MedMessage · SessionMeta · AuditEvent
```

依赖方向自上而下单向。`api/` 为可选依赖层，未安装 FastAPI 时整包仍可用。
详细分层说明与设计决策见 [docs/architecture.md](./docs/architecture.md)。

---

## 模块结构

```
src/med_langchain_memory/
├── domain/        # 领域模型：MedMessage、SessionMeta（Pydantic 强类型实体）
├── serde/         # 序列化层：Protobuf 编解码（med_session_pb2）、统一序列化接口
├── stores/        # 存储适配器：内存 / 文件 / Redis / MySQL / ES，统一接口 + 工厂注册
├── lifecycle/     # 会话生命周期：TTL 归档、快照备份、跨存储迁移、合规保留
├── privacy/       # 字段级正则脱敏（手机号 / 身份证 / 病历号 / 床号）
├── runnable/      # 医疗增强 Runnable：租户隔离、Token 裁剪、LLM 摘要、并发锁、降级熔断
├── api/           # FastAPI 会话管理接口
├── config.py      # 全局配置（pydantic-settings）
└── exceptions.py  # 统一异常体系
```

---

## 安装

核心包仅依赖 `langchain-core` / `pydantic` / `protobuf`（内存与文件后端开箱可用），
其余后端与能力按需安装 extras：

```bash
pip install med-langchain-memory                  # 核心（memory / file 后端）
pip install "med-langchain-memory[redis]"         # + Redis 单机 / 集群
pip install "med-langchain-memory[mysql]"         # + MySQL 分表
pip install "med-langchain-memory[es]"            # + Elasticsearch 归档
pip install "med-langchain-memory[api]"           # + FastAPI 接口层
pip install "med-langchain-memory[token]"         # + tiktoken Token 计数
pip install "med-langchain-memory[dev]"           # 开发/测试全套（含测试替身与 lint 工具）
```

> 也可组合安装：`pip install "med-langchain-memory[redis,api,token]"`。

---

## 快速开始

```python
from med_langchain_memory.stores import StoreFactory

history = StoreFactory.create(
    "memory",
    session_id="s-20261005-001",
    tenant_id="hospital-a",
    dept_id="cardiology",
    patient_id="p-0001",
)

history.add_med_messages([...])  # 写入医疗消息（校验命名空间归属）
for msg in history.get_med_messages():  # 按时序读取
    print(msg.role.value, msg.content)
```

LCEL 上下文工程（时序裁剪 → 摘要压缩 → Token 预算）：

```python
from med_langchain_memory.runnable import (
    ContextWindowPolicy,
    MedRunnableWithMessageHistory,
    SummaryPolicy,
    TokenBudgetPolicy,
)

runnable = MedRunnableWithMessageHistory(
    store_factory=StoreFactory,
    window_policy=ContextWindowPolicy(max_messages=40),
    summary_policy=SummaryPolicy(trigger_messages=30),
    token_budget=TokenBudgetPolicy(max_tokens=4000),
)
```

启动 API 服务：

```bash
uvicorn med_langchain_memory.api.app:create_app --factory
```

---

## API 端点

应用工厂：`create_app(...)`；交互式文档：`/docs`（Swagger UI）与 `/openapi.json`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET /health` | 存活检查 | 进程级探针，恒返回 `ok` |
| `GET /health/ready` | 就绪检查 | 执行已注册探针，返回各组件状态 |
| `POST /sessions` | 创建会话 | 建立会话元数据（DTO 校验） |
| `GET /sessions` | 分页查询会话 | 按命名空间分页列出会话 |
| `GET /sessions/{session_id}` | 查询会话详情 | 不存在或不可见返回 404 |
| `POST /sessions/{session_id}/close` | 关闭会话 | `ACTIVE -> CLOSED` |
| `POST /sessions/{session_id}/archive` | 归档会话 | `ACTIVE/CLOSED -> ARCHIVED` |
| `DELETE /sessions/{session_id}` | 软删除会话 | `ARCHIVED -> DELETED`，不物理清除 |
| `POST /sessions/{session_id}/messages` | 追加消息 | 批量追加 + 可选脱敏，非活跃会话返回 409 |
| `GET /sessions/{session_id}/messages` | 游标分页查询消息 | 支持脱敏开关参数 |
| `GET /admin/stats` | 归档/会话统计 | 按命名空间聚合统计（自动翻页） |
| `POST /admin/sessions/{session_id}/migrate` | 跨存储迁移 | 触发源→目标后端迁移 |
| `POST /admin/sessions/{session_id}/snapshot` | 快照导出 | 内联返回 `payload_base64` + `sha256` + `size_bytes` |

**鉴权**：不注入认证器时，命名空间由查询参数（`tenant_id` / `dept_id`）给出；
注入 `ApiKeyAuthenticator` 后，请求须携带 API Key 请求头，并校验其科室 scope（支持 `"*"` 通配）。
缺失/无效密钥返回 401，越权访问返回 403。

---

## 统一存储对接规范（与 springai-med-qa 互通）

为保证异构系统数据互通，两仓库严格遵循同一套规范（完整版见 [docs/storage-spec.md](./docs/storage-spec.md)）：

| 项 | 规则 |
|---|---|
| Redis 键 | `med:chat:{tenant_id}:{dept_id}:{session_id}`（`:messages` 列表 + `:meta` 哈希，hash tag 用 `{session_id}`） |
| MySQL 分表 | `med_message_{crc32(session_id) % 16}`（共 16 张表，编号补零 `med_message_00`~`med_message_15`） |
| ES 归档索引 | `med-chat-archive-{yyyy.MM}`（按月滚动，UTC） |
| 消息字段 | `message_id` / `session_id` / `tenant_id` / `dept_id` / `patient_id` / `role` / `content` / `token_count` / `masked` / `created_at` / `metadata` |
| 主键 | `message_id` 为 UUIDv7（RFC 9562），时序有序 |
| 时间单位 | 一律 UTC epoch 毫秒（`int64`） |
| 序列化 | Protobuf 二进制（`protos/med_session.proto`），存储层禁止 JSON |

---

## 本地开发

```bash
pip install -e ".[dev]"
pytest                 # 全量单测（fakeredis / sqlite 内存库替身，无需真实中间件）
ruff check . && ruff format --check .
mypy
```

---

## 每日迭代节奏

项目按 [ROADMAP.md](./ROADMAP.md) 的分阶段任务表，由每日自动化任务完成「编码 → 单测 → 提交 → 推送 GitHub」闭环，每个迭代点 30–60 分钟可独立提交。当前已完成阶段 0–4 全部迭代（D1–D35）。

---

## License

Apache-2.0

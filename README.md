# med-langchain-memory

> 医疗专属 LangChain 分布式会话存储中间件 · Medical-grade distributed chat message history middleware for LangChain (Python 3.11+ / LCEL)

面向医院医患问诊会话的**存储中间件**——只负责对话历史的「存、管、取」，不内置任何 NLP / 中文分词 / 实体抽取逻辑。上层由 [`springai-med-qa`](https://github.com/xxinjie21/springai-med-qa)（Java 问诊后端）消费，两仓库存储字段、键规范、序列化协议严格对齐，数据可互通。

---

## 核心定位

| 维度 | 说明 |
|---|---|
| 角色 | AI 问诊系统的「记忆系统」——LLM 无状态，本库负责把对话规范地存好、管好、按需取回 |
| 职责边界 | 数据「肉身」始终躺在 Redis / MySQL / ES 等真实数据库里，本库只做调度与规则层（中间件） |
| 合规 | 字段级正则脱敏、多租户科室隔离、到期自动归档，满足医疗数据留存要求 |
| 约束 | 全程零 NLP 依赖，隐私处理仅字段级正则规则 |

---

## 技术栈

| 层次 | 技术 | 用途 |
|---|---|---|
| 框架基础 | LangChain（`langchain-core`） | 实现 `BaseChatMessageHistory`，增强 `RunnableWithMessageHistory` |
| 数据建模 | Pydantic v2 + pydantic-settings | 消息 / 会话强类型实体（≈ Java DTO + @Valid + Jackson 三合一）、配置管理 |
| 序列化 | Protobuf | 统一二进制编码，跨语言与 Java 项目互通 |
| 存储引擎 | redis-py（Cluster） / SQLAlchemy + MySQL / elasticsearch-py | 热会话、持久化分表、冷归档 |
| Token 计算 | tiktoken | 上下文按 Token 预算裁剪 |
| API 层 | FastAPI | 轻量会话管理接口（增删查、迁移、归档触发） |
| 测试 | pytest + fakeredis + sqlite 内存库 | 全量单测，不依赖真实中间件 |
| CI | GitHub Actions | 自动测试 + 覆盖率 |

---

## 模块结构

```
src/med_langchain_memory/
├── domain/        # 领域模型：ChatMessage、SessionMeta（Pydantic 强类型实体）
├── serde/         # 序列化层：Protobuf 编解码（med_session_pb2）、统一序列化接口
├── stores/        # 存储适配器：内存 / 文件 / Redis / MySQL / ES，统一接口 + 工厂注册
├── lifecycle/     # 会话生命周期：TTL 归档、快照备份、跨存储迁移
├── privacy/       # 字段级正则脱敏（手机号 / 身份证 / 病历号）
├── runnable/      # 医疗增强 Runnable：租户隔离、Token 裁剪、LLM 摘要、并发锁、降级熔断
├── api/           # FastAPI 会话管理接口
└── exceptions.py  # 统一异常
```

---

## 统一存储对接规范（与 springai-med-qa 互通）

为保证异构系统数据互通，两仓库严格遵循同一套规范：

| 项 | 规则 |
|---|---|
| Redis 键 | `med:chat:{tenant}:{dept}:{session}` |
| MySQL 分表 | `med_message_{crc32(session_id) % 16}`（共 16 张表） |
| 消息字段 | `session_id` / `tenant` / `dept` / `patient_id` / `role` / `content` / `created_at`（epoch millis，UUIDv7 主键） |
| 序列化 | Protobuf（`med_session.proto`，跨语言统一协议） |

---

## 安装

```bash
pip install med-langchain-memory            # 核心（内存/文件存储）
pip install "med-langchain-memory[redis]"   # + Redis 集群
pip install "med-langchain-memory[mysql]"   # + MySQL 分表
pip install "med-langchain-memory[es]"      # + Elasticsearch 归档
pip install "med-langchain-memory[api]"     # + FastAPI 管理接口
```

---

## 本地开发

```bash
pip install -e ".[dev]"
pytest                 # 全量单测（fakeredis / sqlite 内存库替身，无需真实中间件）
```

---

## 每日迭代节奏

项目按 `ROADMAP.md` 的分阶段任务表，由每日自动化任务完成「编码 → 单测 → 分模块 commit → 推送 GitHub」闭环，每个迭代点 30–60 分钟可独立提交。

---

## License

Apache-2.0

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
| 多存储适配 | 六种已注册后端（内存 / 文件 / Redis 单机 / Redis 集群 / MySQL 16 张分表 / ES 归档），装饰器注册、热插拔 |
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
| CI | GitHub Actions | 多版本矩阵测试 + 覆盖率门禁 + lint/type-check + tag 发布流水线（版本守卫 + 构建 + Release） |

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
| `mysql` | `MySQLMedHistory` | MySQL 16 张同构分表，`crc32(session_id) % 16` 一致性 hash 路由 + `med_session` 会话行 upsert | `[mysql]` |
| `elasticsearch` | `EsArchiveMedHistory` | 冷归档，按月滚动索引 + bulk 批量写入 | `[es]` |

```python
from med_langchain_memory.stores import StoreFactory

StoreFactory.available()  # 例：['elasticsearch', 'file', 'memory', 'mysql', 'redis', 'redis-cluster']
```

> **MySQL 分表**：写入按 `crc32(session_id) % 16` 路由到唯一分表，同一次
> `add_med_messages` 的批量消息单次 `executemany` 落库；存储层为每条消息分配会话内
> 单调递增的 `ordinal`，读取按 `(created_at, ordinal)` 排序，保证同毫秒消息的写入顺序
> 与其余后端一致。表结构与列定义见 [存储规范 §7.2](./docs/storage-spec.md)。

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
├── testing/       # 集成测试支撑：真实中间件探针、开关与集群故障转移演练（纯标准库）
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
pip install "med-langchain-memory[mysql]"         # + MySQL 分表（另需 DBAPI 驱动，如 pymysql）
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

## 集成测试（真实中间件，可选）

默认 `pytest` **不需要任何真实中间件**。真实后端用例单独放在 `tests/test_integration/`，
统一打 `integration` 标记，并且需要同时满足两个条件才会真正执行：

1. 环境变量 `MED_MEMORY_IT=1`（显式开关）；
2. 目标服务端口可连（Redis Cluster 还要求 `cluster_state:ok`，见下）。

任一条件不满足时用例自动 `skip`，跳过原因里带可复制的启动命令。

```bash
docker compose -f docker-compose.integration.yml up -d      # Redis + MySQL + ES + Redis Cluster(3主3从)
MED_MEMORY_IT=1 pytest -m integration                       # 只跑真实后端用例
docker compose -f docker-compose.integration.yml down -v
```

| 项 | 说明 |
|---|---|
| 服务地址 | 默认 `redis://localhost:16379/15` · `mysql+pymysql://root:med@localhost:13306/med_memory` · `http://localhost:19200` · `redis://localhost:17001/0`（集群种子节点） |
| 端口约定 | 宿主端口刻意避开标准端口（6379 / 3306 / 9200），避免与本机既有中间件抢占 |
| 地址覆盖 | `MED_MEMORY_IT_REDIS_URL` / `MED_MEMORY_IT_MYSQL_URL` / `MED_MEMORY_IT_ELASTICSEARCH_URL` / `MED_MEMORY_IT_REDIS_CLUSTER_URL` |
| 用例范围 | 复用 `tests/test_stores/behavior.py` 跨后端行为基准套件，再补服务端特有断言（键类型、原生 TTL、分表落表、按月索引、hash tag 同 slot、集群 pipeline 语义、集群故障转移） |
| MySQL 前置 | 需宿主侧自备 DBAPI 驱动（如 `pip install pymysql`），缺失时 MySQL 用例跳过 |
| CI | 可选工作流 `.github/workflows/integration.yml`（手动触发 + **每日夜间定时**），不阻塞 `ci.yml` 必过门禁；每次运行都会产出趋势摘要并滚动缓存报告供下次对比（见下） |

探针与开关逻辑在 `med_langchain_memory/testing/services.py`，只用标准库，
因此「能不能跑」的判断本身不依赖任何中间件客户端。

### 夜间 CI 与趋势上报

`integration.yml` 每日 18:00 UTC（北京时间次日 02:00）跑一次，除了执行用例还做三件事：

| 步骤 | 动作 |
|---|---|
| 报告 | `pytest -m integration --junit-xml=reports/integration.xml`，失败也照常出报告 |
| 摘要 | `python -m med_langchain_memory.testing.trend reports/integration.xml --baseline reports/previous.xml` 渲染 Markdown 写进 Job Summary（用例数 / 通过率 / **新增失败** / **已修复** / 最慢用例） |
| 滚动 | `reports/` 目录走 `actions/cache`，key 带 `run_id`、`restore-keys` 前缀命中上一次；恢复出的报告改名 `previous.xml` 作为本次基线 |

趋势渲染逻辑在 `med_langchain_memory/testing/trend.py`，**纯标准库**
（`xml.etree.ElementTree`），汇总口径一律从 `<testcase>` 子节点推导，
不信任 `<testsuite>` 上的 `tests=` / `failures=` 属性。本地可直接复现：

```bash
MED_MEMORY_IT=1 pytest -m integration --junit-xml=reports/integration.xml
python -m med_langchain_memory.testing.trend reports/integration.xml --output reports/trend.md
```

首次运行（没有基线）时基线列显示 `—`，不会报错。


### Redis Cluster（三主三从）

编排会起 6 个节点（宿主端口 17001–17006 + 集群总线端口 27001–27006），
再由一次性容器 `redis-cluster-init` 组装成 3 主 3 从、16384 槽全覆盖的集群。

集群节点必须向客户端与对端广播一个**双向可达**的地址（`--cluster-announce-ip`），
否则客户端会拿到不可达的容器内网 IP，集群也永远停在 `cluster_state:fail`：

| 环境 | 广播地址 |
|---|---|
| Linux（含 GitHub Actions runner） | 默认 `172.31.240.1`（编排固定子网的网关），双向可达，**无需设置** |
| Docker Desktop（Windows / macOS） | 容器 IP 与子网网关都不可从宿主机访问，**必须**显式指定宿主机局域网 IP |

```bash
# Docker Desktop 用户：先探测本机局域网 IP，再起服务
export MED_MEMORY_IT_CLUSTER_IP=$(python -c "import socket;s=socket.socket(2,2);s.connect(('192.0.2.1',9));print(s.getsockname()[0])")
docker compose -f docker-compose.integration.yml up -d
```

> 该变量只影响「怎么起集群」，不影响用例本身：用例侧只需 `MED_MEMORY_IT_REDIS_CLUSTER_URL`。

集群用例还覆盖了一条**替身测不出**的真机约束：redis-py 的集群客户端弃用了 `MULTI`
（`pipeline(transaction=True)` 抛 `RedisClusterException`），因此 `RedisClusterMedHistory`
必须覆写 `_pipeline()` 使用非事务 pipeline。详见 `docs/architecture.md` ADR-11。

### 故障转移演练（破坏性用例）

`tests/test_integration/test_redis_cluster_failover.py` 是唯一会**主动破坏环境**的用例：
它用 `docker stop` 停掉一个主节点，验证集群自动把其从节点晋升为新主、读写仍可用，
再 `docker start` 把节点拉回并等到集群重新收敛为 3 主 16384 槽全覆盖。
无论断言成败都会在 `finally` 里恢复环境，避免把降级集群留给后续用例。

演练依赖两个前提：

| 前提 | 说明 |
|---|---|
| 显式容器名 | 6 个节点在编排里固定为 `med-memory-integration-redis-cluster-N`，演练据此定位目标节点（默认命名会漂移） |
| `docker` CLI | 宿主机需有可用的 `docker` 命令；缺失时该模块整体跳过 |

拓扑解析与目标选择（`CLUSTER NODES` 文本解析、主/从配对、容器名映射、可注入时钟的轮询）
全部抽在 `med_langchain_memory/testing/cluster.py` 里，**纯标准库**，
因此这部分逻辑在没有 Docker 与 Redis 的机器上也有完整的离线单测。

---

## 发布与版本

**版本单一事实源**：`pyproject.toml` 的 `[project].version` 与包内
`med_langchain_memory.__version__` 必须逐字一致（由 `tests/test_package.py` 守住），
发布 tag 也必须与之逐字一致。

发布流程由 `.github/workflows/release.yml` 承载，只由 `v*` 形态的 tag 触发
（如 `v0.1.0`），分支推送与 PR **不会**触发发布：

```bash
git tag v0.1.0
git push origin v0.1.0        # 触发 release.yml
```

工作流分三个作业，逐级卡关：

| 作业 | 动作 |
|---|---|
| `verify` | ① 版本守卫：tag ↔ pyproject ↔ `__version__` 三方比对；② 全量 `pytest`；③ `python -m build` 产出 sdist + wheel；④ 产物校验：sdist/wheel 齐全、wheel 内必需成员与元数据版本正确；⑤ 上传 `distributions` artifact |
| `github-release` | 用官方 `gh release create` 建 Release，附上 sdist/wheel 并自动生成说明（`--verify-tag` 保证 tag 真实存在） |
| `publish-pypi` | **默认关闭**；仅当仓库变量 `PUBLISH_TO_PYPI` 设为 `true` 时执行，走 PyPI Trusted Publishing（OIDC），不需要任何 token secret |

权限按最小化授予：顶层 `contents: read`，只有建 Release 的作业提升为
`contents: write`，只有 PyPI 作业持有 `id-token: write`。

守卫本身是一个纯标准库模块，可在本地预演同样的检查：

```bash
python -m med_langchain_memory.release --tag v0.1.0                     # 版本一致性
python -m med_langchain_memory.release --dist dist --expected-version v0.1.0   # 产物完整性
```

两条命令通过返回 `0`，失败返回 `1` 并逐条打印差异（CI 日志里可直接定位）。

---

## 每日迭代节奏

项目按 [ROADMAP.md](./ROADMAP.md) 的分阶段任务表，由每日自动化任务完成「编码 → 单测 → 提交 → 推送 GitHub」闭环，每个迭代点 30–60 分钟可独立提交。当前已完成阶段 0–4（D1–D35），以及阶段 5 的存储补齐与发布工程（D36 MySQL 适配器、D37 真实中间件集成测试、D38 发布流水线、D39 Redis Cluster 真机用例、D40 集群故障转移演练）。

---

## License

Apache-2.0

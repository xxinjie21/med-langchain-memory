# med-langchain-memory 开发路线图

> 医疗专属 LangChain 分布式会话存储中间件（Python / LCEL）
> 仓库路径：`D:\javaproject\med-langchain-memory` ｜ 独立仓库，独立推送 GitHub

---

## 一、项目简介

**定位**：基于 LangChain `BaseChatMessageHistory` 抽象的医疗专属分布式会话存储中间件，面向医院医患问诊会话的持久化、多存储适配、医疗合规与上下文优化。

**核心能力**：
- 医患对话消息持久化：内存 / 本地文件 / Redis 单机 / Redis 集群 / MySQL 分表 / Elasticsearch 归档，六种存储适配器 + 工厂注册
- 统一 Protobuf 序列化协议，支持会话 TTL 自动归档、快照备份、跨存储迁移
- 医疗增强版 `RunnableWithMessageHistory`：多租户科室权限隔离、时序上下文裁剪、Token 预算控制、长会话 LLM 摘要压缩、并发会话锁、读写降级兜底
- 医疗隐私字段级规则脱敏（手机号/身份证/病历号等结构化字段，纯正则规则策略，**不含任何文本解析预处理逻辑**）
- FastAPI 轻量会话管理接口 + Pydantic 强类型 + 全量单元测试 + GitHub Actions CI（测试+覆盖率）

**明确不做**：不引入任何第三方文本预处理/语义解析库，不做文本内容理解，只做存储、协议、生命周期与上下文工程。

**技术栈**：Python 3.11+ / LangChain Core / Pydantic v2 / Protobuf / redis-py / SQLAlchemy / elasticsearch-py / FastAPI / pytest / ruff / mypy / GitHub Actions

---

## 二、完整分层目录结构

```
med-langchain-memory/
├── pyproject.toml                  # 打包与依赖（hatchling / PEP621）
├── README.md
├── ROADMAP.md                      # 本文件
├── .pre-commit-config.yaml         # ruff + mypy 钩子
├── .github/
│   └── workflows/
│       ├── ci.yml                  # pytest + coverage + lint（必过门禁）
│       ├── integration.yml         # 真实中间件集成用例（手动 + 每周定时，可选）
│       └── release.yml             # tag 发布：版本守卫 + 构建 + 产物校验 + GitHub Release
├── protos/
│   └── med_session.proto           # 统一序列化协议（跨语言规范源）
├── docker-compose.integration.yml  # 集成测试本地编排：Redis/MySQL/ES + Redis Cluster 三主三从
├── src/med_langchain_memory/
│   ├── __init__.py
│   ├── config.py                   # 全局配置（pydantic-settings）
│   ├── exceptions.py               # 统一异常体系
│   ├── domain/                     # 领域模型层
│   │   ├── message.py              # MedMessage（对齐跨语言字段规范）
│   │   ├── session.py              # SessionMeta / SessionStatus
│   │   └── audit.py                # 审计事件模型
│   ├── serde/                      # 序列化层
│   │   ├── base.py                 # Serializer 抽象
│   │   └── protobuf_serializer.py
│   ├── stores/                     # 存储适配层（适配器模式）
│   │   ├── base.py                 # MedChatMessageHistory 抽象（扩展 BaseChatMessageHistory）
│   │   ├── factory.py              # StoreFactory + 注册器
│   │   ├── memory_store.py
│   │   ├── file_store.py           # JSONL/二进制 + 文件锁
│   │   ├── redis_store.py          # 单机 + 集群 + TTL
│   │   ├── mysql_store.py          # SQLAlchemy + hash 分表路由
│   │   └── es_store.py             # 归档存储
│   ├── lifecycle/                  # 会话生命周期层
│   │   ├── ttl_archiver.py         # 活跃→归档调度
│   │   ├── snapshot.py             # 快照备份/恢复
│   │   ├── migrator.py             # 跨存储迁移
│   │   └── retention.py            # 软删除与合规保留期
│   ├── privacy/                    # 隐私合规层（纯规则，无文本解析）
│   │   ├── masker.py               # 字段级正则脱敏引擎
│   │   └── policies.py             # 可插拔脱敏策略（策略模式）
│   ├── runnable/                   # LCEL 增强层
│   │   ├── med_history_runnable.py # MedRunnableWithMessageHistory
│   │   ├── tenant.py               # 多租户/科室命名空间隔离
│   │   ├── trimmer.py              # 时序窗口 + Token 预算裁剪
│   │   ├── summarizer.py           # 长会话 LLM 摘要压缩
│   │   ├── lock.py                 # 并发会话锁（Redis 分布式锁+本地降级）
│   │   └── fallback.py             # 读写降级兜底（熔断器）
│   ├── api/                        # FastAPI 接口层
│   │   ├── app.py
│   │   ├── deps.py                 # 鉴权/依赖注入
│   │   ├── routers/
│   │   │   ├── sessions.py
│   │   │   ├── messages.py
│   │   │   └── admin.py            # 归档/迁移管理
│   │   └── schemas.py              # 请求/响应 DTO
│   ├── testing/                    # 集成测试支撑（纯标准库探针与开关）
│   └── release.py                  # 发布守卫（tag/版本一致性 + 产物校验，纯标准库）
├── tests/                          # 与 src 镜像的全量单测
│   ├── conftest.py
│   ├── test_domain/ test_serde/ test_stores/
│   ├── test_lifecycle/ test_privacy/ test_runnable/
│   ├── test_api/ test_ci/ test_docs/
│   └── test_integration/           # 真实中间件用例（integration 标记，默认跳过）
└── docs/
    ├── architecture.md             # 架构图与设计决策
    └── storage-spec.md             # 存储字段/序列化跨语言规范（对外发布）
```

---

## 三、分阶段每日迭代任务（每个 30–60 分钟，单一功能，独立 commit）

### 阶段 0：工程基建（D1–D5）

| Day | 任务 | 实现要点 | Commit 信息 |
|---|---|---|---|
| D1 | 项目脚手架 | pyproject.toml、src 布局、ruff+mypy+pre-commit、LICENSE、README 骨架 | `chore: bootstrap project skeleton with pyproject, lint and type-check tooling` |
| D2 | 领域模型 | Pydantic v2 定义 `MedMessage`/`SessionMeta`/`SessionStatus` 枚举 + 字段校验单测 | `feat(domain): add medical session and message models with validation` |
| D3 | Protobuf 协议 | 编写 `med_session.proto`（消息/会话/租户字段），编译脚本 `make proto` | `feat(proto): define cross-language session serialization schema` |
| D4 | 序列化器 | `Serializer` 抽象 + Protobuf 实现，round-trip 单测（model↔bytes） | `feat(serde): implement protobuf serializer with round-trip tests` |
| D5 | CI 流水线 | GitHub Actions：多 Python 版本矩阵、pytest、coverage 上报、lint 检查 | `ci: add test, lint and coverage workflow` |

### 阶段 1：多存储适配层（D6–D17）

| Day | 任务 | 实现要点 | Commit 信息 |
|---|---|---|---|
| D6 | 存储抽象 | `MedChatMessageHistory(BaseChatMessageHistory)`：新增 TTL/归档/租户钩子方法 | `feat(stores): define medical chat history abstract interface` |
| D7 | 工厂+注册器 | `StoreFactory.register("redis")` 装饰器注册模式，配置驱动实例化 | `feat(stores): add store factory with decorator-based registry` |
| D8 | 内存存储 | `InMemoryMedHistory` + 完整单测（作为其余实现的行为基准套件） | `feat(stores): implement in-memory store with shared behavior test suite` |
| D9 | 文件存储 | JSONL 追加写 + protobuf 二进制模式 + 跨进程文件锁 | `feat(stores): implement file-based store with append log and file lock` |
| D10 | Redis 单机 | List+Hash 结构存消息与元数据，pipeline 批量写 | `feat(stores): implement redis store with pipelined writes` |
| D11 | Redis 集群 | cluster 客户端适配、hash tag 保证同会话同 slot、连接池配置 | `feat(stores): support redis cluster with hash-tag slot affinity` |
| D12 | Redis TTL | 会话级 TTL 策略、滑动续期、过期回调钩子 | `feat(stores): add session ttl with sliding expiration for redis store` |
| D13 | MySQL 模型 | SQLAlchemy ORM、建表 DDL、Alembic 迁移初始化 | `feat(stores): add mysql schema and alembic migration baseline` |
| D14 | MySQL 分表 | `session_id` 一致性 hash 分表路由（如 msg_00~msg_15），路由单测 | `feat(stores): implement hash-based table sharding router for mysql` |
| D15 | ES 归档 | 归档索引模板（按月滚动）、批量 bulk 写入、归档查询 | `feat(stores): implement elasticsearch archive store with monthly indices` |
| D16 | 跨存储迁移 | `Migrator`：源→目标批量迁移、断点续传游标、迁移校验 | `feat(lifecycle): add cross-store migration tool with resumable cursor` |
| D17 | 快照备份 | 会话快照导出/导入（protobuf 文件包）、校验和 | `feat(lifecycle): add session snapshot export/import with checksum` |

### 阶段 2：生命周期与合规（D18–D21）

| Day | 任务 | 实现要点 | Commit 信息 |
|---|---|---|---|
| D18 | TTL 归档调度 | 后台调度器：活跃会话超期→自动迁移至 ES 归档层 | `feat(lifecycle): add ttl-driven auto archiver from hot store to archive` |
| D19 | 合规保留期 | 软删除标记、保留期策略（如病历会话保留 N 年）、到期物理清理 | `feat(lifecycle): add soft-delete and compliance retention policy` |
| D20 | 审计事件 | 读/写/删/迁移操作审计事件模型 + 审计落盘接口 | `feat(domain): add audit event model and audit sink interface` |
| D21 | 脱敏引擎 | 字段级正则脱敏（手机号/身份证/病历号/床号），纯规则策略 | `feat(privacy): add rule-based field masking engine` |

### 阶段 3：医疗增强 Runnable（D22–D29）

| Day | 任务 | 实现要点 | Commit 信息 |
|---|---|---|---|
| D22 | 脱敏策略化 | 策略模式：可插拔 MaskPolicy、按租户配置组合策略 | `feat(privacy): support pluggable per-tenant masking policies` |
| D23 | Runnable 骨架 | `MedRunnableWithMessageHistory` 继承封装，注入存储工厂 | `feat(runnable): add medical runnable with pluggable history stores` |
| D24 | 多租户隔离 | `tenant_id:dept_id:session_id` 命名空间键设计、越权访问拒绝 | `feat(runnable): enforce tenant and department namespace isolation` |
| D25 | 时序裁剪 | 滑动窗口裁剪（按条数/时间窗），保留首条主诉消息 | `feat(runnable): add time-ordered context window trimmer` |
| D26 | Token 预算 | tiktoken 计数、预算内贪心保留最近消息、预算超限告警 | `feat(runnable): add token-budget aware context trimmer` |
| D27 | LLM 摘要压缩 | 超长会话触发摘要链，摘要写回 system 槽位并标记已压缩区间 | `feat(runnable): add llm summary compression for long sessions` |
| D28 | 并发会话锁 | Redis SETNX+看门狗续期分布式锁，本地线程锁降级 | `feat(runnable): add distributed session lock with local fallback` |
| D29 | 读写降级 | 主存储故障→备存储兜底、简易熔断器（失败计数+半开恢复） | `feat(runnable): add read/write fallback with circuit breaker` |

### 阶段 4：API 与收尾（D30–D35）

| Day | 任务 | 实现要点 | Commit 信息 |
|---|---|---|---|
| D30 | FastAPI 骨架 | app 工厂、健康检查、统一异常处理、请求日志中间件 | `feat(api): bootstrap fastapi app with health check and error handler` |
| D31 | 会话接口 | 会话创建/查询/关闭/归档 CRUD + DTO 校验 | `feat(api): add session crud endpoints` |
| D32 | 消息接口 | 消息追加、游标分页查询、脱敏开关参数 | `feat(api): add message append and cursor pagination endpoints` |
| D33 | 管理接口 | 迁移触发、快照导出、归档统计管理端点 | `feat(api): add admin endpoints for migration and snapshot` |
| D34 | API 鉴权 | API Key + 科室 scope 校验依赖、401/403 单测 | `feat(api): add api-key auth with department scope enforcement` |
| D35 | 文档收尾 | README 完善（徽章/快速开始/架构图）、docs/storage-spec.md 发布 | `docs: complete readme, architecture and storage spec` |

### 阶段 5：存储补齐与发布工程（D36–）

| Day | 任务 | 实现要点 | Commit 信息 |
|---|---|---|---|
| D36 | MySQL 适配器 | `MySQLMedHistory` 注册 `mysql` 后端：`crc32(session_id) % 16` 分表写入 + `med_session` 会话行 upsert + `ordinal` 会话内保序 | `feat(stores): implement mysql sharded store with ordered writes` |
| D37 | 真实中间件集成测试 | `docker-compose.integration.yml` 三服务编排 + `testing/services.py`（纯标准库 TCP 探针 + `MED_MEMORY_IT` 显式开关）+ `tests/test_integration/`（复用行为基准套件，打 `integration` 标记，默认跳过）+ 可选工作流 `integration.yml` | `feat(stores): add opt-in real middleware integration test harness` |
| D38 | 发布流水线 | `release.py`（纯标准库发布守卫：tag ↔ pyproject ↔ `__version__` 三方比对 + sdist/wheel 完整性与元数据版本校验 + CLI）+ `release.yml`（`v*` tag 触发：守卫 → 全量测试 → 构建 → 产物校验 → GitHub Release，PyPI 走 OIDC 可信发布且默认关闭）+ `tests/test_release.py`、`tests/test_ci/test_release_workflow.py` | `feat: daily iteration D38 - tag-driven release pipeline with version guard` |
| D39 | Redis Cluster 真机用例 | `docker-compose.integration.yml` 增三主三从 6 节点 + 一次性初始化容器（固定子网 + `MED_MEMORY_IT_CLUSTER_IP` 广播地址）；`testing/services.py` 增 `redis-cluster` 服务、`cluster_startup_nodes`、`detect_host_address`；`tests/test_integration/test_redis_cluster_live.py`（复用行为基准套件 + slot 亲和 + **真机回归：集群弃用 `MULTI`**）；`RedisClusterMedHistory._pipeline()` 覆写为非事务 pipeline | `feat: daily iteration D39 - real redis cluster integration tests` |
| D40 | 集群故障转移演练 | `testing/cluster.py`（纯标准库：`CLUSTER NODES` 解析 `parse_cluster_nodes`/`parse_slots`、确定性目标选择 `plan_failover`→`FailoverPlan`、端口↔容器名映射 `cluster_node_index`/`cluster_container_name`、可注入时钟的 `wait_until`、可注入执行器的 `DockerContainerController`/`run_docker`）；`tests/test_integration/test_redis_cluster_failover.py`（停主节点 → 从节点晋升 → 读写可用 → 原主恢复收敛）；编排 6 节点显式固定 `container_name` | `feat: daily iteration D40 - redis cluster failover drill` |

> 阶段 5 后续方向（**待用户决策**，不预设顺序）：① 集成测试纳入夜间 CI 并上报趋势；
> ② 正式推送 `v0.1.0` tag 触发首次发布（需维护者执行 `git push origin v0.1.0`）；
> ③ MySQL 8 的 `pymysql` 驱动纳入 dev extras，让 MySQL 真机用例在 CI 默认可跑。

---

## 四、对外存储规范（本库为规范定义方）

任何下游服务（任意语言）对接本库存储时，必须遵循 `protos/med_session.proto` 与 `docs/storage-spec.md`：

**统一消息字段**（protobuf `MedMessage`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `message_id` | string | UUIDv7，时序有序 |
| `session_id` | string | 会话ID |
| `tenant_id` | string | 医院/机构租户 |
| `dept_id` | string | 科室ID |
| `patient_id` | string | 患者ID（脱敏存储） |
| `role` | enum | PATIENT / DOCTOR / ASSISTANT / SYSTEM |
| `content` | string | 消息正文（可被字段级脱敏） |
| `token_count` | int32 | 内容 token 数 |
| `masked` | bool | 是否已脱敏 |
| `created_at` | int64 | epoch millis |
| `metadata` | map<string,string> | 扩展标签 |

**统一键/表规范**：
- Redis 键：`med:chat:{tenant_id}:{dept_id}:{session_id}`（hash tag 用 `{session_id}`）
- MySQL 分表：`med_message_{crc32(session_id) % 16}`，16 张分表
- ES 归档索引：`med-chat-archive-{yyyy.MM}`
- 序列化：一律 protobuf 二进制，禁止 JSON 落库（API 层 DTO 除外）

---

## 五、Commit 提交规范（Conventional Commits）

```
<type>(<scope>): <subject 英文小写祈使句，≤72字符>

[body 可选：动机 + 方案要点]
[footer 可选：BREAKING CHANGE / issue 引用]
```

- type：`feat` / `fix` / `refactor` / `perf` / `test` / `docs` / `ci` / `chore`
- scope：`domain` / `serde` / `stores` / `lifecycle` / `privacy` / `runnable` / `api`
- 铁律：**每个迭代必须配套单元测试，测试全绿才允许提交**

### 每日分模块批次提交与推送流程

每天迭代完成后，按模块目录分批提交，形成清晰提交历史，最后统一推送：

```bash
# 1. 检查变更
git status
# 2. 按模块分批提交（示例：某迭代同时改了 stores 与 tests）
git add src/med_langchain_memory/stores/ && git commit -m "feat(stores): implement redis store with pipelined writes"
git add tests/test_stores/            && git commit -m "test(stores): add redis store unit tests with fakeredis"
git add .github/ pyproject.toml       && git commit -m "chore: update dependencies for redis store"   # 如有
# 3. 推送
git push origin main
```

分批顺序约定：`domain → serde → stores → lifecycle → privacy → runnable → api → tests → ci/配置`。单测可随功能模块同 commit，也可独立 `test(scope)` commit；禁止一天所有变更混在一个大 commit 里推送。

> **自动化每日迭代的例外**：由无人值守任务执行的每日迭代（D1 起）采用
> **单次 `git add -A` + 单个 commit + 单次 push** 的收敛式提交，
> 便于逐迭代回溯与失败重放；人工迭代仍按上述分批约定执行。

---

## 六、简历技术亮点（本项目）

1. 设计并开源基于 LangChain `BaseChatMessageHistory` 的医疗会话存储中间件，通过适配器+工厂+注册器模式支持内存/文件/Redis 单机/Redis 集群/MySQL 分表/ES 归档六种存储引擎热插拔
2. 制定跨语言 Protobuf 会话序列化规范，实现会话 TTL 自动归档、快照备份、断点续传式跨存储迁移
3. 自研医疗增强版 `RunnableWithMessageHistory`：多租户科室命名空间隔离、时序窗口+Token 预算双层上下文裁剪、长会话 LLM 摘要压缩，长会话场景下上下文 Token 成本显著降低
4. 实现 Redis 分布式会话锁（看门狗续期+本地锁降级）与主备存储熔断兜底，保障并发问诊场景数据一致性与可用性
5. 落地医疗合规能力：字段级规则脱敏策略引擎、操作审计事件、合规保留期与软删除
6. 全量 pytest 单测 + GitHub Actions CI（多版本矩阵、覆盖率门禁），Conventional Commits 规范化提交

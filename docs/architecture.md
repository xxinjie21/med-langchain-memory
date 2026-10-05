# 架构与设计决策

> **med-langchain-memory** — 医疗专属 LangChain 分布式会话存储中间件
> 配套阅读：[对外存储规范](./storage-spec.md) ｜ 迭代计划：`ROADMAP.md`

本文件说明系统的分层结构、一次请求的完整链路，以及若干关键设计决策（ADR 风格）。
存储字段、键名与序列化的对外契约以 [storage-spec.md](./storage-spec.md) 为准。

---

## 1. 定位与职责边界

LLM 本身无状态，对话的「记忆」必须由外部存储承担。本库是医患问诊系统的**记忆层中间件**：

| 本库负责 | 本库不负责 |
|---|---|
| 对话历史的持久化、多后端适配、生命周期管理 | 任何 NLP / 中文分词 / 实体识别 / 语义解析 |
| 统一 protobuf 序列化协议与跨语言字段规范 | 业务编排、问诊流程、Prompt 工程 |
| 上下文工程（时序裁剪 / Token 预算 / 摘要压缩） | 模型调用本身（不内置 LLM 客户端） |
| 并发一致性（会话锁）与可用性（主备降级熔断） | 权限体系的身份源（只做 scope 校验） |
| 字段级正则脱敏（合规最小集） | 数据「肉身」的最终归属——数据始终躺在 Redis / MySQL / ES |

**数据归属**：本库只做调度与规则层，真实数据始终落在外部中间件中；本库自身不引入数据库。

---

## 2. 分层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  api/            FastAPI 接入层（可选依赖 fastapi）                    │
│  ├─ app.py       应用工厂 create_app()，不产生 import 副作用            │
│  ├─ auth.py      API Key 鉴权 + 科室 scope 校验（SHA-256 摘要查表）      │
│  ├─ deps.py      依赖注入：命名空间 / 仓储 / 脱敏器 / 历史解析器          │
│  ├─ errors.py    统一异常 → HTTP 状态码映射 + 统一错误响应体             │
│  ├─ middleware.py 纯 ASGI 请求日志 / request_id 透传                    │
│  ├─ cursor.py    游标编解码 + 分页纯函数                                │
│  ├─ session_guards.py 会话守卫（存在 / 可见 / 活跃）                     │
│  └─ routers/     health · sessions · messages · admin                  │
├──────────────────────────────────────────────────────────────────────┤
│  runnable/       LCEL 上下文工程层                                     │
│  ├─ med_history_runnable.py  MedRunnableWithMessageHistory             │
│  ├─ tenant.py    租户/科室命名空间隔离与越权守卫                         │
│  ├─ trimmer.py   时序滑动窗口裁剪（保留首条主诉）                        │
│  ├─ token_budget.py  Token 计数 + 预算内贪心保留 + 超限告警              │
│  ├─ summarizer.py   长会话摘要压缩（摘要写回 system 槽位 + 区间标记）      │
│  ├─ lock.py      会话锁：Redis SETNX + 看门狗续期，本地线程锁降级         │
│  └─ fallback.py  主备存储降级 + 每后端熔断器（失败计数 / 半开探测 / 闭合） │
├──────────────────────────────────────────────────────────────────────┤
│  privacy/        隐私合规层（纯正则，零 NLP）                           │
│  ├─ masker.py    字段级正则脱敏引擎（手机号/身份证/病历号/床号）           │
│  └─ policies.py  可插拔策略：按租户组合规则集                            │
├──────────────────────────────────────────────────────────────────────┤
│  lifecycle/      会话生命周期层                                        │
│  ├─ ttl_archiver.py 活跃超期 → 自动迁移至归档层                          │
│  ├─ retention.py    合规保留期：软删除 + 宽限期后物理清理                 │
│  ├─ snapshot.py     快照导出/恢复（带 SHA-256 校验和的二进制文件包）       │
│  └─ migrator.py     跨存储迁移（断点续传游标 + 校验）                     │
├──────────────────────────────────────────────────────────────────────┤
│  stores/         存储适配层（适配器 + 工厂 + 注册器）                    │
│  ├─ base.py      MedChatMessageHistory（扩展 BaseChatMessageHistory）   │
│  ├─ factory.py   StoreFactory.register("redis") 装饰器注册             │
│  ├─ history_resolver.py  会话 → 历史实例解析（主/备存储）                │
│  ├─ memory_store.py / file_store.py / redis_store.py                  │
│  ├─ redis_cluster_store.py / mysql_store.py                           │
│  ├─ mysql_schema.py / mysql_shard_router.py   分表结构 + hash 路由     │
│  ├─ es_store.py      冷归档（按月滚动索引 + bulk）                      │
│  └─ *_repository.py  会话索引 / 消息仓储（补 history 回答不了的列表查询）  │
├──────────────────────────────────────────────────────────────────────┤
│  serde/          序列化层：Serializer 抽象 + ProtobufSerializer          │
├──────────────────────────────────────────────────────────────────────┤
│  domain/         领域模型：MedMessage · SessionMeta · AuditEvent         │
└──────────────────────────────────────────────────────────────────────┘
        ▲ 依赖方向自上而下单向；下层绝不 import 上层
```

### 2.1 依赖方向

`api → runnable → lifecycle → stores → serde → domain`

* `domain` / `serde` 为叶子层，无内部依赖；
* 下层不得反向 import 上层（如 `stores` 不得引用 `api`）；
* `api` 是**可选依赖层**：未安装 FastAPI 时整包仍可正常使用，因此
  `med_langchain_memory/__init__.py` 不导入 `api`。

---

## 3. 一次问诊请求的完整链路

```
HTTP POST /sessions/{id}/messages
   │
   ├─ RequestLoggingMiddleware      生成/透传 request_id，记录耗时
   ├─ resolve_session_scope         鉴权（API Key → 租户/科室 scope），得到命名空间
   ├─ session_guards.require_active_session   会话存在 + 可见 + 处于 ACTIVE
   ├─ MessageCreate → MedMessage    DTO 校验 → 领域模型（Pydantic v2）
   ├─ FieldMasker.mask_message      命中脱敏规则则改写 content 并置 masked=True
   ├─ MessageRepository.append      写入消息仓储（真实后端为分表/Redis）
   └─ 统一响应 MessageAppendResponse
```

读取链路则叠加 LCEL 上下文工程（`MedRunnableWithMessageHistory.build_context`）：

```
全量历史
   └─ ① 时序窗口裁剪 (trimmer)     按条数/时间窗裁剪，始终保留首条主诉
        └─ ② 摘要压缩 (summarizer)  超阈值区间折叠为摘要，写回 system 槽位
             └─ ③ Token 预算裁剪   预算内贪心保留最近消息，超限产生告警
                  └─ ④ 会话锁 (lock)  写入路径由分布式锁串行化，本地锁降级
                       └─ ⑤ 主备降级  主存储异常 → 备存储兜底 + 熔断器保护
```

三级上下文流水线（①②③）是可组合的：任一环节未配置即透传，配置后按序叠加。

---

## 4. 关键设计决策（ADR）

### ADR-1 · 存储适配用「适配器 + 工厂 + 装饰器注册器」

**决策**：`MedChatMessageHistory` 在 LangChain `BaseChatMessageHistory` 之上补充租户 / TTL /
归档三类钩子；子类只需实现 `_append` / `_read` / `clear` 三个存储原语，其余契约方法由基类提供。
后端通过 `@StoreFactory.register("redis")` 自注册。

**理由**：新增后端零侵入（不改工厂代码）；可选依赖（redis / SQLAlchemy / elasticsearch）缺失时
用 `contextlib.suppress(ImportError)` 静默跳过注册，其余后端不受影响。

**代价**：类级注册表是进程级全局状态，测试需注意隔离。

### ADR-2 · 序列化一律 protobuf 二进制

**决策**：存储层禁止 JSON，统一 protobuf 二进制；`protos/med_session.proto` 为跨语言单一事实源。

**理由**：字段号冻结带来强向后兼容；二进制体积小于 JSON；与 Java 后端（`springai-med-qa`）
共享同一 `.proto` 即可互通，无需手写映射。

**代价**：人工排查需先解码，可读性差；ES 侧以 base64 存储。

### ADR-3 · 隐私脱敏只做字段级正则，绝不引入 NLP

**决策**：手机号 / 身份证 / 病历号 / 床号通过 `MaskRule` 正则规则在字段粒度脱敏；
未纳管字段原样透传。

**理由**：医疗合规要求处理逻辑可审计、可复现；正则规则无模型不确定性，零第三方依赖，
不会因分词/实体识别错误而漏脱敏。规则按「具体优先」排序（身份证先于手机号）。

**代价**：无法处理非结构化表述（如「我的手机尾号是…」），这是明确接受的能力边界。

### ADR-4 · 鉴权开关 = 是否注入认证器

**决策**：`create_app(authenticator=...)` 不传认证器时，命名空间退回查询参数模式。

**理由**：让 D31–D33 的既有接口契约与测试保持零改动，鉴权作为可插拔增强而非破坏性变更；
后续接入真实身份源只需替换 `api/deps.py::resolve_session_scope`。

### ADR-5 · 管理端点内联返回，服务端不落盘

**决策**：快照导出端点返回 `payload_base64` + `sha256` + `size_bytes`，而非写服务器文件。

**理由**：避免任意路径写文件带来的安全面；由调用方决定落盘位置。
为此给 `SessionSnapshotter` 增加不落盘的 `prepare_snapshot()`。

### ADR-6 · 降级「尽力而为 + 不静默」

**决策**：主存储异常时切换备存储，每个后端各持一份熔断器（失败计数打开 → 恢复窗口半开探测
→ 探测成功闭合）。全部后端失败时抛携带完整报告的 `FallbackExhaustedError`。

**理由**：可用性优先，但绝不把「全挂」伪装成「空结果」——静默返回空列表会让上层误判为
「会话无消息」，是医疗数据场景不可接受的错误。

### ADR-7 · 无原生 TTL 的存储用逻辑过期

**决策**：`supports_ttl` 类变量标记后端能力；内存 / 文件 / MySQL 不支持原生 TTL，
通过 `updated_at + ttl_seconds` 做 `is_expired()` 惰性判定；Redis 走原生 `EXPIRE` + 滑动续期。

**理由**：统一的 TTL 语义（`set_ttl` / `refresh_ttl` / `is_expired`）跨后端一致，
上层无需感知底层差异；不支持的组合在设置时即抛 `StorageError`，而非静默失效。

### ADR-8 · 关系型分表用存储层 `ordinal` 保序

**决策**：消息分表额外带一列存储层自增序号 `ordinal`（不属于 `MedMessage` 字段集），
由 `MySQLMedHistory` 在写入时按「会话内最大序号 + 批内偏移」分配；读取按
`(created_at, ordinal)` 排序。ES 归档侧同样写入 `ordinal`，语义对齐。

**理由**：`message_id` 是 UUIDv7，**同毫秒内随机**，仅按 `created_at` 排序无法还原写入顺序
（关系型表也不保证同值行的返回顺序）；而内存 / 文件 / Redis 后端靠「稳定排序 + 追加序」
天然保序，若 MySQL 不显式保序就会出现跨后端语义漂移。归档层同理，
排序键必须含 `ordinal`。

**代价**：`ordinal` 的「读最大值 + 批量插入」不是原子操作，跨进程并发写同一会话
需由上层会话锁（`runnable/lock.py`）串行化；`med_session` 行的 upsert 同理。

---

## 5. 并发与可用性

| 机制 | 实现 | 降级路径 |
|---|---|---|
| 会话互斥 | `RedisSessionLock`：`SET NX PX` + 看门狗续期 + 事务化 compare-and-delete | 无 Redis 时 `LocalSessionLock`（进程级按锁键共享互斥量） |
| 存储容错 | `CircuitBreaker`（失败计数打开 / 恢复窗口半开 / 成功阈值闭合） | 主备双写降级；全挂抛 `FallbackExhaustedError` |
| 单会话写入 | 一次 `add_messages` 打包进单个 pipeline 事务（Redis） | 非事务后端按序追加，失败即抛 `StorageError` |

> 本地锁降级仅在单进程内有效，多实例部署必须配置 Redis；这是可用性降级而非一致性保证。

---

## 6. 错误体系

所有业务异常继承 `MedMemoryError`，API 层通过 `STATUS_MAP` 做 `isinstance` 匹配映射 HTTP 状态码：

| 异常 | HTTP | 场景 |
|---|---|---|
| `ValidationError` | 422 | 领域模型 / 请求参数校验失败 |
| `AuthenticationError` | 401 | API Key 缺失、无效或已停用 |
| `AuthorizationError` | 403 | 已认证但无权访问目标租户 / 科室 |
| `SessionNotFoundError` | 404 | 目标会话在命名空间下不存在 |
| `SessionNotActiveError` | 409 | 会话非 `ACTIVE`，拒绝写入 |
| `StateTransitionError` | 409 | 会话状态非法流转 |
| `TenantIsolationError` | 403 | 跨租户 / 跨科室越权访问 |
| `StorageError` / `FallbackExhaustedError` | 503 | 存储读写失败 / 主备全部不可用 |
| `IntegrityError` | 422 | 快照校验和不匹配等完整性失败 |

基础设施与配置类异常（一般由启动期或后台任务暴露，不直接映射用户请求）：

| 异常 | 场景 |
|---|---|
| `StoreRegistrationError` | 存储适配器注册失败（名称非法、重复注册或类型不合法） |
| `StoreNotFoundError` | 按名称查找存储适配器失败（未注册的后端） |
| `LockError` / `LockAcquisitionError` | 会话锁基础设施异常 / 等待超时未能获取锁 |
| `AuditSinkError` | 审计事件落盘失败（路径不可写、磁盘 IO 异常等） |

错误码由类名按 CamelCase → snake_case 推导（如 `SessionNotFoundError` → `session_not_found_error`）。

---

## 7. 测试策略

| 手段 | 说明 |
|---|---|
| 替身中间件 | `fakeredis`（无 Lua，用 `WATCH`/`MULTI`/`EXEC` 事务实现锁）、SQLite 内存库、自写 fake ES |
| 无真实依赖 | 全量单测不需要真实 Redis / MySQL / ES，CI 无需起容器 |
| 行为基准套件 | `tests/test_stores/behavior.py` 供各后端复用，保证语义一致 |
| 覆盖率门禁 | CI `--cov-fail-under=85`，实际维持在 99% |
| 文档一致性 | `tests/test_docs/` 校验文档字段 / 端点 / 后端清单与代码、proto、pyproject 保持同步 |

---

_本文件描述设计意图；实现细节以 `src/med_langchain_memory/` 源码为准。_

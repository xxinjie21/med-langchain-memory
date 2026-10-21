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
│  testing/        测试支撑层（纯标准库，不参与生产链路）                 │
│  ├─ services.py  集成测试服务探针 + 显式开关 + 集群启动节点/广播地址解析  │
│  ├─ cluster.py   集群拓扑解析 + 故障转移目标选择 + docker 容器控制 + 轮询  │
│  ├─ trend.py     JUnit 报告汇总 + 趋势对比 + Markdown 渲染             │
│  ├─ preflight.py 必需服务前置守卫：不可达即失败（防「集体跳过」假绿灯）│
│  ├─ shards.py    集成套件按服务分片：用例路径/服务/compose 名/环境变量的单一真值源│
│  └─ aggregate.py 分片结果聚合：五片总账 + 基线对比（含逐分片趋势）+ 结论退出码    │
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
  `med_langchain_memory/__init__.py` 不导入 `api`；
* `testing` 是**旁路层**：只依赖 `exceptions`，不依赖任何中间件客户端，
  也不被其它层引用（仅 `tests/` 使用）。

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

### ADR-9 · 真实中间件集成测试用「显式开关 + TCP 探针」，默认离线全绿

**决策**：真实后端用例集中在 `tests/test_integration/` 并统一打 `integration` 标记，
执行前须同时满足「环境变量 `MED_MEMORY_IT=1`」与「目标服务端口可连」；
不满足时用例 `skip` 并把可复制的 `docker compose` 启动命令写进跳过原因。
探针与开关放在 `testing/services.py`，**只依赖标准库**（`socket` / `urllib.parse`），
不导入任何中间件客户端。

**理由**：

* 日常 `pytest` 与必过 CI（`ci.yml`）不能被容器依赖拖慢或拖红——集成测试放在
  可选工作流 `integration.yml`（手动触发 + 每周定时）里；
* 若探针模块自己 import `redis` / `elasticsearch`，缺包时连「跳过」都做不到，
  测试支撑模块会反向依赖可选后端；纯标准库的 TCP 探针没有这个约束；
* 断言复用 `tests/test_stores/behavior.py` 的跨后端行为基准套件，
  替身单测与真实后端走**同一份语义契约**，避免两套断言各自漂移。

**代价**：TCP 可连不等于协议就绪（MySQL 冷启动、ES 集群初始化），
因此夹具在探针之后还要做一次客户端级握手重试，就绪判定比纯端口探测更慢；
这也是默认 `WAIT_SECONDS=120` 的来源。

### ADR-10 · 发布门禁用「可执行模块」而非 YAML 里的内联脚本

**决策**：tag 发布（`.github/workflows/release.yml`）把版本一致性与产物完整性
两件事都收敛到 `med_langchain_memory/release.py` 一个纯标准库模块里，
工作流只负责按顺序调用：

```
release.py --tag "${{ github.ref_name }}"        # 构建前：tag ↔ pyproject ↔ __version__
python -m build
release.py --dist dist --expected-version "${{ github.ref_name }}"   # 构建后：sdist/wheel 完整性
```

**理由**：

* 内联 shell / python 片段无法被 `pytest` 覆盖，也容易在 YAML 缩进里悄悄写错；
  抽成模块后，两条门禁各有正向与边界用例（合成 wheel/sdist 即可全量验证，无需真实构建）；
* 版本漂移（改了代码忘了改版本、tag 打错、dist 里躺着上一次构建的旧 wheel）是
  发布事故最常见来源，必须在**构建前**和**发布前**各卡一道；
* 只有 `contents: read` 的顶层权限 + 写权限下放到单个作业，配合
  「PyPI 发布默认关闭（仓库变量开关）+ OIDC 可信发布」把误发布面降到最小。

**代价**：守卫随包一起发布（多一个几十行的小模块）；工作流里的 CLI 选项需要
与模块保持同步——为此 `tests/test_ci/test_release_workflow.py` 直接调用
`build_parser()` 断言「工作流用到的选项真实存在」。

### ADR-11 · Redis 集群用 hash tag + 非事务 pipeline，并用真机用例守护

**决策**：`RedisClusterMedHistory` 与单机版有两处刻意差异：

1. 键布局用 `{session_id}` 作 hash tag，使同一会话的 `:messages` 与 `:meta`
   落在同一 slot（租户 / 科室**不**进 tag，否则整个租户塌缩到单一 slot）；
2. 覆写 `_pipeline()` 返回 `pipeline(transaction=False)`。

配套新增真机套件 `tests/test_integration/test_redis_cluster_live.py`，
以 `docker-compose.integration.yml` 里的三主三从编排为环境。

**理由**：

* redis-py 的集群客户端**弃用了 `MULTI`**：`pipeline(transaction=True)` 直接抛
  `RedisClusterException("transaction is deprecated in cluster mode")`。
  单机实现的 `clear()` / `_append()` / `_apply_ttl()` / `_persist()` 默认都走事务
  pipeline，因此在真集群上会**整体不可用**（`clear()` 首当其冲）；
* 这类缺陷**替身永远测不出**：`fakeredis.FakeRedis` 没有集群语义，
  集群单测传入的是普通替身，事务 pipeline 一路绿灯。真机用例是唯一防线，
  因此套件里保留一条「本集群确实拒绝 `transaction=True`」的断言，
  一旦 redis-py 改变行为即红灯，提示重新评估该适配；
* hash tag 已把两条键固定在同一 slot，非事务 pipeline 依旧把批量命令压成
  一次网络往返，只放弃 `MULTI/EXEC` 的原子性——集群模式下本来也换不来跨 slot 原子性。

**代价**：集群写入不再有事务原子性（`RPUSH` 与 `HSET` 之间可能被其它客户端观测到），
需由上层会话锁（`runnable/lock.py`）串行化；集群集成栈需要 6 个容器 + 1 个初始化容器，
且必须解决「节点广播地址要双向可达」的网络问题（见下）。

**广播地址**：集群节点必须广播一个**同时**能被容器内对端与宿主机客户端访问的地址
（`--cluster-announce-ip`），否则客户端拿到不可达的容器内网 IP、节点间 gossip 也断链，
集群永远停在 `cluster_state:fail`。编排用 `MED_MEMORY_IT_CLUSTER_IP` 控制，
默认取固定子网网关 `172.31.240.1`（Linux / CI runner 双向可达）；
Docker Desktop（Windows / macOS）上容器 IP 与子网网关都不可从宿主机访问，
必须显式覆盖为宿主机局域网 IP。`testing/services.py::detect_host_address()` 可探测该地址。

---

### ADR-12 · 高可用靠「可复现的故障转移演练」验证，而非靠拓扑断言

**决策**：新增破坏性集成用例 `tests/test_integration/test_redis_cluster_failover.py`：
停掉一个主节点容器 → 等到其从节点晋升并接管槽位 → 用**存活节点**重建客户端验证读写 →
把节点拉回并等到集群重新收敛为 3 主 16384 槽全覆盖。

演练的「非 Redis 部分」全部下沉到 `testing/cluster.py`（纯标准库）：

| 关注点 | 实现 | 为什么下沉 |
|---|---|---|
| 拓扑解析 | `parse_cluster_nodes` / `parse_slots` | 解析规则是纯函数，可在无 Redis 环境完整单测 |
| 目标选择 | `plan_failover` → `FailoverPlan` | 选择规则必须确定性可复现，否则演练不可回归 |
| 节点定位 | `cluster_node_index` / `cluster_container_name` | 端口 ↔ 容器名映射是纯计算，且要与编排文件对齐 |
| 环境破坏 | `DockerContainerController` | 执行器可注入，编排逻辑用假执行器即可单测 |
| 收敛等待 | `wait_until`（时钟与 sleep 可注入） | 真机要等几十秒，离线单测必须零等待 |

**理由**：

* 「3 主 3 从、16384 槽全覆盖」只是**静态拓扑断言**，证明不了主节点消失后集群还能
  自动收敛——而这恰恰是「高可用」的全部含义。只有真停一个容器才能覆盖；
* 把演练逻辑写成 `testing/cluster.py` 的纯函数后，**失败模式可以离线复现**：
  「只有主没有从」「从节点链路断开」「主节点已标记 fail」「端口不属于本编排」
  都有对应的离线用例，真机用例只负责最后那一步不可替代的破坏与观察；
* 读取拓扑刻意用 `execute_command("CLUSTER", "NODES")` 而非 redis-py 的
  `cluster_nodes()`：后者只为字面量 `"CLUSTER NODES"` 注册了解析回调并返回
  **按 IP 为键的 dict**（同广播 IP 的多节点会互相覆盖），拿原始文本才与自研解析口径一致。

**代价**：用例会主动破坏环境，必须保证恢复（恢复动作放在 `finally`，
并在恢复后强制 `nodes_manager.initialize()` 刷新会话级客户端的槽位映射）；
6 个节点在编排里必须显式固定 `container_name`，否则 `docker stop` 会打错目标；
演练使集成套件整体耗时增加约 1 分钟。

---

### ADR-13 · 集成测试报告用「JUnit + 滚动缓存」做趋势对比，而不是只看单次红绿

**决策**：夜间集成工作流每次运行都产出 JUnit 报告（`--junit-xml`），
由 `testing/trend.py` 渲染 Markdown 摘要写进 Job Summary，
并把报告经 `actions/cache` 滚动给下一次运行当基线：

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 汇总口径 | `parse_junit_xml` 逐条读 `<testcase>` | `<testsuite>` 的 `tests=` / `failures=` 属性可能与明细不一致，只信明细 |
| 结果归一化 | `CaseResult.outcome` + `failing` | 只区分 passed / failed / error / skipped，`skipped` 不算失败 |
| 通过率 | `SuiteReport.pass_rate` | 分母排除 `skipped`（集成套件大量条件跳过，否则通过率被稀释） |
| 趋势 | `TrendReport.new_failures` / `fixed` / `pass_rate_delta` | 按 `classname::name` 做集合差，回答「这次是不是我弄坏的」 |
| 输出 | `render_markdown` | 直接追加进 `$GITHUB_STEP_SUMMARY`，无需第三方报告平台 |
| 基线来源 | `actions/cache` + `restore-keys` 前缀 | 不引入外部存储，也不需要跨 run 读 artifact 的 API 调用 |

**理由**：

* 「这次红了」信息量很低——集成套件依赖外部容器，**偶发失败与真实回归必须区分开**，
  只有和上一次运行对比才能回答「是不是新引入的问题」；
* 摘要步骤与 `pytest` 步骤解耦：`pytest` 的退出码负责卡关，摘要步骤 `if: always()` 负责可读化，
  因此**失败时也能拿到报告**（失败信息恰恰最需要被看到）；
* CLI 只做「解析 → 渲染 → 落盘」，退出码不含「有失败用例」这一维，
  避免摘要步骤自己变成第二个卡关点，掩盖 pytest 的真实状态；
* 解析只用标准库 `xml.etree.ElementTree`，与项目「零文本预处理依赖」的边界一致。

**代价**：GitHub 缓存条目按 `run_id` 累积（保留期受仓库缓存配额约束）；
`reports/` 目录已加入 `.gitignore`，本地跑不会污染工作区。

---

### ADR-14 · 集成测试用「必需服务前置守卫」防假绿灯，而不是相信 skip

**决策**：`integration.yml` 在执行用例**之前**先跑
`python -m med_langchain_memory.testing.preflight`，由 `MED_MEMORY_IT_REQUIRED`
声明「必须可用」的服务清单；任一服务不可用即整步失败。
地址不再手写在工作流里：每个分片的清单与地址由
`med_langchain_memory.testing.shards --format env` 生成（见 ADR-15），
用例步骤与守卫步骤读同一份 `$GITHUB_ENV`：

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 必需清单 | `parse_required`（逗号分隔、去空白、去重保序） | 未声明时返回空元组 → 守卫对本地手工 `pytest` 无副作用 |
| 判定 | `PreflightReport.ok` = 无解析错误 且 全部服务 `ready` | 复用 `ServiceStatus.ready`，与用例夹具同一套判定口径 |
| 服务名拼错 | 记进 `errors` 而非抛异常 | 守卫的价值在于把问题**一次列全**，而不是首个错误就退出 |
| 地址来源 | 分片模块生成后写进 `$GITHUB_ENV`（缺省值来自 `DEFAULT_SERVICES`） | 用例与守卫读同一份地址，杜绝「步骤里改了端口、守卫还在探旧端口」 |
| 入口 | `python -m med_langchain_memory.testing.preflight`（模块路径） | 包根上的 `main` / `build_parser` 已属于 `trend` 的 CLI，导出会静默覆盖 |

**理由**：

* 集成用例对不可达服务 `skip` 是**本地开发**的正确行为，但在 CI 上等价于
  「跑了个寂寞还是绿的」——服务名拼错、容器没起来、端口映射写错，
  全部只会安静地变成一堆 `skipped`，真实回归反而被掩盖；
* 守卫只做 TCP 可达性判断（不握手、不导入中间件客户端），
  因此「环境是否具备」这件事本身**不依赖任何可选后端**；
* 协议级就绪（集群是否 `cluster_state:ok`、ES 是否加载模板）仍由用例自己的夹具轮询，
  职责边界清晰：守卫管「端口通不通」，夹具管「服务能不能用」。

**代价**：多一个步骤、多一个环境变量；守卫与容器健康检查有部分重叠
（健康检查保证容器已起，守卫额外覆盖「compose 起的 Redis Cluster 与地址覆盖是否生效」）。

---

### ADR-15 · 集成套件按服务分片并行，分片表是「单一真值来源」

**决策**：把真机套件按服务切成五个**互不依赖**的分片
（`redis` / `mysql` / `elasticsearch` / `redis-cluster` / `redis-cluster-failover`），
`integration.yml` 用 `strategy.matrix` 并行执行；每个分片在自己的 runner 上
**只拉起自己需要的容器**。分片表（用例路径 / 必需服务 / compose 服务名 /
初始化容器 / 环境变量 / 墙钟预算）全部由纯标准库模块
`med_langchain_memory.testing.shards` 给出，工作流只调用它的 CLI：

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 分片定义 | `SHARDS`（冻结 dataclass 元组）+ `resolve_shard` / `select_shards` | 新增分片只改一处；矩阵取值由 `shard_names()` 交叉校验 |
| 用例路径 | `--format paths` | 工作流不写路径，避免「分片表加了用例、工作流没跟上」 |
| 必需服务 | `--format required` → `$GITHUB_ENV` | 与 ADR-14 的守卫共用，分片化后守卫粒度也更准 |
| 拉起容器 | `--format compose` → `docker compose up -d --wait <服务>` | 只起本分片需要的服务，冷启动时间不叠加 |
| 一次性初始化 | `--format init` → `run --rm <容器>` | 集群分片才需要；非集群分片拿到空串即整步跳过，工作流无需 `if:` 判断分片名 |
| 服务地址 | `--format env`（复用 `services.resolve_service`） | 工作流里**不出现任何 `MED_MEMORY_IT_*_URL` 字面量** |
| 失败语义 | `fail-fast: false` | 一个分片红不代表其它分片没有价值，五个分片各自出结论 |

**理由**：

* 单作业串行跑全部真机套件时，墙钟时间 = 四个服务的启动时间与用例耗时之和
  （ES 冷启动最慢，6 节点集群还要额外组集群）；分片并行后墙钟时间由
  「最慢的那个分片」决定，且任一服务的启动抖动只影响自己那一片；
* 中间件统一改走 `docker compose`：GitHub Actions 的 `services:` **无法按矩阵分片**
  （声明了就会让每个分片都拉起全部容器），而 Redis Cluster 本来就只能用 compose。
  compose 里每个长驻服务都有 `healthcheck`，`up -d --wait` 与原先 `services:`
  的健康检查语义等价；
* 「配置即数据」而非「配置即 YAML」：分片表是普通 Python 数据，
  因此矩阵取值、compose 服务名、服务地址都能被 pytest 直接断言，
  工作流静态测试可以做成**交叉校验**（工作流传的 CLI 选项必须能被真实解析器解析、
  分片要的 compose 服务名必须在编排文件里存在）。

**代价**：每个分片重复一遍 checkout / setup-python / pip install（矩阵作业的固有开销）；
报告、缓存与 artifact 都按分片隔离，Job Summary 从一份变成五份。

---

### ADR-16 · 分片结果用一个汇总作业聚合成总账，而不是让人点开五个作业

**决策**：在矩阵作业之后追加 `aggregate` 作业（`needs: integration` + `if: always()`），
把五个分片的 JUnit 报告聚合成一份 Markdown 总账写进 Job Summary，
并用**单一退出码**给出整轮集成测试的结论。聚合逻辑放在纯标准库模块
`med_langchain_memory.testing.aggregate`：

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 期望分片清单 | `collect_shard_reports` 默认取 `shards.shard_names()` | 与矩阵取值同源，工作流不重复声明分片清单 |
| 用例明细解析 | 复用 `trend.parse_junit_report` | 不重复实现一遍 XML 解析，口径与单分片摘要逐字一致 |
| 报告目录布局 | `shard_report_path`：先认 `reports/<分片>/`，再认 `reports/<artifact 名>/` | 本地跑出的布局与 `download-artifact` 解出的布局都能吃下 |
| 结论退出码 | `0` 齐备无失败 / `1` 缺报告或解析失败 / `2` 有失败用例 / `3` 参数非法 | 「缺报告」与「有失败」是两类问题，必须能分开定位 |
| 摘要与退出码解耦 | 先写 Markdown 再返回退出码；工作流「聚合」与「贴摘要」分两步，后者 `if: always()` | 聚合失败时最需要看的那份总账不会因为步骤中断而丢失 |
| 聚合作业不拉容器 | 只读 artifact | 汇总不需要中间件，避免重复支付冷启动成本 |

**理由**：

* 分片并行把墙钟时间压到「最慢的那一片」，代价是结论被切成五份；
  人要知道「今晚到底绿没绿」得点开五个作业，反而比串行更贵——聚合把这份成本收回来；
* 「缺报告」在单分片视角里不可见（没报告就是没跑过），但在总账里它是
  **CI 基础设施故障**的明确信号：矩阵作业在 pytest 之前挂掉（compose 起不来、
  守卫失败）时不会有 JUnit 文件，聚合据此给出退出码 `1`，而不是安静地少算一片；
* 聚合只是**读文件 + 计数**，因此可以完全离线单测（合成 JUnit XML），
  不引入任何新的中间件依赖。

**代价**：多一个作业（多一次 checkout / setup-python / 下载 artifact 的开销）；
Job Summary 里总账与五份分片摘要并存，需要一点阅读顺序上的约定（总账在最后）。

---

### ADR-17 · 总账自带「与上一轮对比」，基线靠滚动缓存而不是外部存储

**决策**：在 ADR-16 的总账之上再加一层对比——`testing/aggregate.py` 的
`AggregateTrend` / `build_aggregate_trend` 拿上一轮的分片报告算出**整轮合计的变化**
（用例数 / 通过 / 失败 / 跳过 / 通过率 / 耗时）、**逐分片变化**（失败数、通过率与状态标记）
以及跨分片的**新增失败 / 已修复**清单；`render_markdown(..., baseline=...)` 把它渲染成
「与上一轮对比」小节。基线的取得方式是 `--snapshot-dir`（把本轮报告按规范布局写进
`baseline/<分片>/integration.xml`）+ `actions/cache` 滚动（`path: baseline`，
key 带 `run_id`、`restore-keys` 前缀命中上一次），由汇总作业自己完成闭环。

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 对比粒度 | 整轮合计 + 逐分片 + 跨分片用例清单 | D41 的趋势对比只到「单分片 vs 该分片上一次」，要回答「整轮比昨晚好了还是差了、是哪一片退化的」得自己拼五份摘要 |
| 基线来源 | `--snapshot-dir` 写、`actions/cache` 传、`--baseline-dir` 读 | 不引入任何外部存储（对象存储 / DB / artifact 追溯），复用 D41 已经验证过的滚动缓存模式 |
| 基线布局 | 快照**只按规范布局**写出 `baseline/<分片>/integration.xml` | 基线目录只有一种结构，读取方不必再猜布局（读入侧仍兼容两种） |
| 基线不齐 | 对比照常，`baseline_complete` 为假并在小节里列出缺哪几片 | 上一轮有分片在 pytest 之前挂掉时，把「少了一片」误读成「用例变少了」是最危险的错 |
| 无基线 | `baseline=None` / 基线一份报告都没解析出来 → 不渲染对比小节 | 首次运行与缓存未命中都是这种情形，输出与 D44 逐字一致（纯增量，不改既有契约） |
| 快照时机 | 与写 Markdown 一样，在返回退出码**之前** | 一次基础设施故障不该把基线链断掉，下一轮仍能拿到上一轮的报告 |

**理由**：

* 总账回答的是「今晚绿没绿」，而人真正要的是「比昨晚好了还是差了」——两句话的成本
  几乎相同（都是读同一批 JUnit），但后者才能驱动行动（谁引入了回退、哪一片在退化）；
* 对比是**纯计数**：分片名参与比对（同名用例出现在不同分片不算「已修复」），
  因此完全离线可测，不需要任何中间件；
* 「基线不齐」必须显式：整轮对比的分母一旦悄悄少一片，结论就会反向。

**代价**：汇总作业多一个缓存步骤与一次快照写盘；对比小节让总账更长一点
（因此 `--slowest` 之类的可裁剪项保留）。本地跑 `--snapshot-dir` 会在仓库根建
`baseline/`，已加入 `.gitignore`。

---

### ADR-18 · 破坏性演练独立成片，矩阵取值与超时预算由 `plan` 作业生成

**决策**：在 ADR-15 的五分片之上再做两件事——① 把停主节点的故障转移演练从
`redis-cluster` 分片拆成独立的 `redis-cluster-failover` 分片（各自拉起一套
6 节点集群，互不影响）；② 矩阵本身不再以字面量写在工作流里，而是由 `plan` 作业
调用 `shards --format matrix` 渲染成单行 JSON（分片名 + 逐片 `timeout_minutes`），
写进 `$GITHUB_OUTPUT` 后经 `fromJSON` 消费，墙钟预算随之逐片生效。

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 演练隔离 | `redis-cluster-failover` 独立分片，自带 6 节点 + 初始化容器 | 演练会停主节点、改变槽位归属，与只读套件同片跑会互相污染；独立成片后也不再拖住别的分片 |
| 矩阵取值 | `plan` 作业 → `--format matrix` → `outputs.matrix` → `fromJSON` | 工作流里**连分片名清单都不出现**，与「服务地址不出现」是同一条单一真值原则 |
| 超时预算 | `Shard.timeout_minutes`（默认 `DEFAULT_TIMEOUT_MINUTES`）→ `timeout-minutes: ${{ matrix.timeout_minutes }}` | 演练与 ES 冷启动明显更慢，统一超时要么误杀慢分片、要么放任卡死的分片耗光 runner 的 6 小时上限 |
| plan 作业成本 | 只装运行期依赖（`pip install -e .`）、不拉容器 | 它只 import 分片表并打印 JSON，几秒结束；与 integration 作业并行不同的依赖集，互不影响 |

**理由**：

* 演练与只读套件的资源冲突是**语义级**的（一个在停节点，一个在读数据），
  分片表加一行就能拆开，比在同一分片里做顺序编排更可靠；
* `timeout-minutes` 在 GitHub Actions 里是**作业级**表达式，只有矩阵值带得进去，
  再一次说明「矩阵是数据而不是 YAML」；
* 期望分片清单仍由 `shard_names()` 给出（ADR-16 的总账、ADR-17 的基线都基于它），
  拆片对下游零改动：总账自动多一行，基线缓存自动多一份。

**代价**：多一个 `plan` 作业（checkout + setup-python + `pip install -e .`，约数十秒）；
集群类资源从「一片跑两遍套件」变成「两片各跑一遍」，夜间墙钟略增但确定性更好。

---

### ADR-19 · 两套基线合一：分片粒度的用例级趋势上收汇总作业

**决策**：撤销 D41「每个分片各自用 `trend` CLI 渲染趋势、各自滚动一份
`actions/cache` 基线」的形态，改为**一套基线、一个渲染点**——汇总作业用同一份
`baseline/` 快照同时渲染整轮总账对比（ADR-17）与**逐分片用例级趋势**：
`testing/aggregate.py` 的 `ShardTrend` / `AggregateTrend.shard_trends` 对两侧
都有报告的分片逐片调用 `trend.build_trend`（口径与 `trend` CLI 逐字一致），
渲染进「与上一轮对比」小节的「逐分片趋势」小节。`trend` 的 CLI 保留给本地
单报告场景，工作流不再调用它。

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 缓存份数 | 工作流只剩汇总作业一份 `actions/cache`（`path: baseline`） | 分片作业是并行矩阵，各自缓存 = 缓存翻倍且口径可能漂移；而分片要的基线数据本来就在汇总作业手里 |
| 口径一致性 | 分片趋势直接复用 `trend.build_trend`，不重写集合差 | 「新增失败 / 已修复」的判定规则只有一处实现，离线单测正对照钉死 |
| 覆盖范围 | 只覆盖本轮与基线**都有报告**的分片 | 任一侧缺报告由 `ShardDelta` 状态表表达（`⚠️ 缺报告`），趋势行不伪造 0 数据 |
| 退场路径 | `trend` 模块与 CLI 原样保留 | 本地单跑一份报告时它仍是最短路径；删除反而要动包根导出与文档 |

**理由**：

* 两份缓存 = 两份真相：分片缓存按 `run_id` 滚动、汇总缓存也按 `run_id` 滚动，
  任何一边缓存未命中时两侧的「趋势」表现不一致（分片无对比、总账有对比），
  读摘要的人要靠「这行趋势是哪来的」脑内归因；
* 汇总作业天然持有全量数据：它下载五个分片的 artifact，逐分片趋势需要的信息
  （本轮的 `reports/<分片>/` + 上一轮的 `baseline/<分片>/`）**已经在同一台 runner 上**，
  上收不增加任何新的数据搬运，只是把渲染从五个作业挪到最后一个；
* 缓存条目数从「五分片 + 一基线」降为「一基线」，`actions/cache` 的 10 GB
  租户限制与 nightly 清理压力同步下降。

**代价**：分片作业的 Job Summary 不再有「自己 vs 自己」的趋势段，需要看趋势的
人改看汇总作业的总账（信息没有丢，只是位置变了）；若未来某分片想在自己作业内
即时看到变化，需要重新引入分片级基线——届时应以「汇总作业把上一轮分片摘要
写回 artifact」的方式回填，而不是恢复第二套缓存。

---

### ADR-20 · 夜间报告发布为 GitHub Pages 看板，而不是只活在 Job Summary

**决策**：新增 `testing/dashboard.py`（纯标准库）与工作流 `dashboard` 作业，
把汇总作业已经得出的同一份总账（`AggregateReport` / `AggregateTrend`）渲染成
一份**自包含** HTML 页面（内联 CSS、零外部资源、零 JS、浅色主题、全量
`html.escape`），由 `peaceiris/actions-gh-pages@v4` 发布到 `gh-pages` 分支，
给夜间集成报告一个固定 URL 的日常入口。

| 关注点 | 实现 | 为什么这样切 |
|---|---|---|
| 数据同源 | 看板直接消费 `aggregate` 模块的聚合对象，不重新解析 XML、不重新定义通过率 / 状态判定 | 「页面与 Job Summary 不一致」比「没有页面」更糟；复用同一渲染口径只需维护一套 |
| 数据搬运 | `aggregate` 作业在**聚合之前**上传 `integration-ledger` artifact（本轮报告 + 刚恢复的上一轮基线），看板作业只下载这一个包 | 聚合的 `--snapshot-dir` 会把 `baseline/` 覆盖成本轮快照，之后再上传就只剩「本轮 vs 本轮」；顺序由 `TestDashboardJob` 静态钉死 |
| 门禁关系 | 看板作业 `if: always()`、不持门禁状态、退出码仅本地排查用 | 看板是附加视图：矩阵或汇总红了**更要**发布，否则最需要看板的时候反而没有看板 |
| 权限 | `contents: write` 只授予 `dashboard` 作业（其余作业默认只读） | 最小权限：能写分支的作业越少越好 |
| 注入面 | 页面无表单、无 JS，所有动态文本（用例名 / 分片名 / 运行链接）一律转义 | 节点 ID 理论上是不可信输入；零 JS 也意味着没有 XSS 执行环境 |

**理由**：D41–D47 把报告越做越细，但结论只存在于 GitHub Actions 的 Job Summary
与 artifact 里——「今晚绿没绿」需要人主动登进 GitHub、找到当次运行、点开作业。
看板把这个动作变成一个书签；同时它只读 artifact、不拉容器、不碰门禁，
新增它的边际风险接近零。

**代价**：多一个第三方动作依赖（`peaceiris/actions-gh-pages`，已锁大版本，
静态测试校验）；仓库需要开启 Pages（Settings → Pages → `gh-pages`），
未开启时发布步骤会失败——但这只让看板作业红，不影响任何测试结论。

---

## 5. 并发与可用性

| 机制 | 实现 | 降级路径 |
|---|---|---|
| 会话互斥 | `RedisSessionLock`：`SET NX PX` + 看门狗续期 + 事务化 compare-and-delete | 无 Redis 时 `LocalSessionLock`（进程级按锁键共享互斥量） |
| 存储容错 | `CircuitBreaker`（失败计数打开 / 恢复窗口半开 / 成功阈值闭合） | 主备双写降级；全挂抛 `FallbackExhaustedError` |
| 单会话写入 | 一次 `add_messages` 打包进单个 pipeline（Redis 单机为 `MULTI/EXEC` 事务） | 集群后端用非事务 pipeline（hash tag 保证同 slot）；无 pipeline 的后端按序追加，失败即抛 `StorageError` |

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
| 真实后端集成 | `tests/test_integration/` + `integration` 标记：`MED_MEMORY_IT=1` 且服务可达才执行，否则跳过；同一份行为基准套件在真机上再跑一遍 |
| 真机守护的缺陷 | 集群弃用 `MULTI`（ES 索引模板、MySQL 同毫秒保序同理）——替身测不出的服务端行为由真机套件兜底 |
| 高可用演练 | `test_redis_cluster_failover.py` 停掉主节点容器验证从节点晋升 + 读写可用 + 自动收敛；编排逻辑在 `testing/cluster.py` 中纯函数化，离线可测 |
| 集成环境编排 | `docker-compose.integration.yml`（Redis / MySQL / ES 带健康检查 + Redis Cluster 三主三从 6 节点 + 一次性初始化容器）；可选工作流 `.github/workflows/integration.yml`（手动 + 每日夜间），跑用例前先过前置守卫 |
| 分片并行 | `testing/shards.py` 把真机套件按服务切成五个矩阵分片（`redis` / `mysql` / `elasticsearch` / `redis-cluster` / `redis-cluster-failover`），每个分片只拉起自己的容器；分片表是单一真值来源，工作流不硬编码任何服务地址 / 分片名 / 容器名 / 分片数量（连矩阵 JSON 都由 `plan` 作业的 `--format matrix` 生成，ADR-18）；每个分片声明自己的 `timeout_minutes` 墙钟预算；离线单测在 `tests/test_integration/test_shards.py` |
| 假绿灯防护 | `testing/preflight.py` 按 `MED_MEMORY_IT_REQUIRED` 逐个探针，任一必需服务不可用即退出码 1——否则「用例集体 skip」会让环境故障伪装成通过；离线单测在 `tests/test_integration/test_preflight.py` |
| 趋势上报 | `testing/trend.py` 解析 JUnit 报告渲染 Markdown 摘要，是本地单报告场景的入口（CLI 保留）；夜间工作流的趋势自 D47 起统一由汇总作业渲染（见下一行），离线单测在 `tests/test_integration/test_trend.py` |
| 分片结果聚合 | `testing/aggregate.py` 把五个分片的报告聚合成一份总账（分片明细 + 合计 + 缺失报告 + 失败用例 + 最慢用例）写入 Job Summary，并以退出码 `0/1/2/3` 区分「齐备无失败 / 缺报告 / 有失败 / 参数非法」；离线单测在 `tests/test_integration/test_aggregate.py` |
| 总账基线对比 | 同一模块的 `AggregateTrend` / `build_aggregate_trend` 拿 `--baseline-dir` 的上一轮报告算出整轮合计变化、逐分片状态、跨分片的新增失败 / 已修复，以及**逐分片用例级趋势**（`ShardTrend` / `shard_trends`，口径复用 `trend.build_trend`，即 ADR-19 的「两套基线合一」）；`--snapshot-dir` 把本轮报告写进 `baseline/`，由汇总作业唯一的 `actions/cache` 滚动给下一轮；无基线时对比小节整体消失（纯增量）；离线单测同上，工作流契约在 `tests/test_ci/test_integration_workflow.py::TestAggregateBaseline` 与 `TestTrendReporting` |
| 夜间看板 | `testing/dashboard.py` 把同一份总账渲染成自包含 HTML（内联 CSS、零外部资源、零 JS、全量转义）并由 `dashboard` 作业发布到 GitHub Pages（ADR-20）；数据来自 `aggregate` 作业在聚合前上传的 `integration-ledger` artifact（顺序由静态测试钉死）；看板不持门禁状态，`contents: write` 只授予该作业；离线单测在 `tests/test_integration/test_dashboard.py`，工作流契约在 `tests/test_ci/test_integration_workflow.py::TestDashboardJob` |
| 覆盖率门禁 | CI `--cov-fail-under=85`，实际维持在 99% |
| 发布门禁 | `med_langchain_memory/release.py`（纯标准库）校验 tag ↔ pyproject ↔ `__version__` 与 sdist/wheel 完整性；`tests/test_ci/test_release_workflow.py` 静态校验 `release.yml`（触发条件、作业顺序、最小权限、动作锁版本、CLI 选项真实存在） |
| 文档一致性 | `tests/test_docs/` 校验文档字段 / 端点 / 后端清单与代码、proto、pyproject 保持同步 |

---

_本文件描述设计意图；实现细节以 `src/med_langchain_memory/` 源码为准。_

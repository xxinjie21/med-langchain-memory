# 医疗会话存储规范（Storage Specification）

> **状态**：已发布（Published） ｜ **规范版本**：`1` ｜ **协议源文件**：`protos/med_session.proto`
> **适用对象**：任何需要读写本库医疗会话数据的下游服务（Java / Go / Python / …）

本文件是 **med-langchain-memory** 对外发布的存储契约。本库是规范定义方，下游服务必须严格
遵循本文件与 `protos/med_session.proto` 的字段、键名与序列化约定，方可与本库数据互通。

---

## 1. 适用范围与设计原则

| 原则 | 说明 |
|---|---|
| 单一事实源 | 字段名、字段号、枚举取值以 `protos/med_session.proto` 为准；本文件为可读性说明，冲突时以 `.proto` 为准 |
| 二进制优先 | 落库载荷一律 protobuf 二进制；**存储层禁止 JSON**（API 层 DTO 除外） |
| 字段号冻结 | 已发布字段号**永不复用、永不重编号**；新增字段一律使用未占用的更大编号 |
| 无文本解析 | 规范不定义任何分词 / 实体识别 / 语义解析行为；隐私处理仅为字段级正则替换 |
| 时间统一 | 所有时间戳均为 **UTC epoch 毫秒**（`int64`），禁止秒级或本地时区时间 |

---

## 2. 通用约定

### 2.1 业务 ID 规范（`IdStr`）

租户 / 科室 / 会话 / 患者 / 消息五类 ID 均遵循同一约束：

| 约束项 | 取值 |
|---|---|
| 长度 | `1` ~ `64` 字符 |
| 字符集 | `^[A-Za-z0-9_.-]+$` |
| 禁止字符 | `:` 与 `{}`（会破坏 Redis 键规范与集群 hash tag） |

> `message_id` 例外：必须为合法 **UUID 字符串**（推荐 UUIDv7，见 2.2），存储列宽固定 36。

### 2.2 消息主键（UUIDv7）

`message_id` 遵循 RFC 9562 的 UUIDv7 布局，保证**按生成时间字典序单调递增**，便于时序分页与范围扫描：

```
 0                   48        52    64                                   128 bit
 +--------------------+---------+-----+------------------------------------+
 |  unix_ts_ms (48)   | ver(4)  | var |           rand (62)                |
 +--------------------+---------+-----+------------------------------------+
                      version=7  variant=RFC4122
```

同毫秒内的消息由随机位排序，因此 **`message_id` 不能作为同毫秒内的时序 tiebreaker**；
归档检索的排序键必须使用 `created_at` + `archived_at` + 批内序号（见 7.3）。

### 2.3 编码

| 项 | 规范 |
|---|---|
| 文本编码 | UTF-8 |
| protobuf 语法 | `proto3` |
| 包名 | `med.session.v1` |
| Java 包 | `com.medlangchain.memory.proto.v1`（`java_multiple_files = true`） |
| Go 包 | `github.com/xxinjie21/med-langchain-memory/gen/go/medsessionv1` |

---

## 3. 消息模型 `MedMessage`

一次医患问诊中的单条消息。所有存储后端落库的消息均使用该字段集。

| # | 字段 | proto 类型 | 存储类型 | 必填 | 说明 |
|---|---|---|---|---|---|
| 1 | `message_id` | `string` | `VARCHAR(36)` PK | 是 | UUIDv7 字符串，时序有序 |
| 2 | `session_id` | `string` | `VARCHAR(64)` | 是 | 归属会话 ID |
| 3 | `tenant_id` | `string` | `VARCHAR(64)` | 是 | 医院 / 机构租户 ID |
| 4 | `dept_id` | `string` | `VARCHAR(64)` | 是 | 科室 ID |
| 5 | `patient_id` | `string` | `VARCHAR(64)` | 是 | 患者 ID（命中脱敏规则时存脱敏值） |
| 6 | `role` | `MessageRole` | `VARCHAR(16)` | 是 | 发送方角色，见 5.1 |
| 7 | `content` | `string` | `TEXT` | 是 | 消息正文，长度 ≥ 1，可被字段级脱敏 |
| 8 | `token_count` | `int32` | `INT` | 是 | 正文 token 数，未计算时为 `0` |
| 9 | `masked` | `bool` | `BOOL` | 是 | 正文是否已被隐私引擎脱敏 |
| 10 | `created_at` | `int64` | `BIGINT` | 是 | 创建时间，UTC epoch 毫秒，必须 > 0 |
| 11 | `metadata` | `map<string,string>` | `JSON` | 是 | 扩展标签；键值均强制为字符串，缺省为空表 |

领域层校验（Pydantic v2）额外约定：

* 模型为**不可变**（`frozen`）且**禁止未声明字段**（`extra = "forbid"`）——下游写入多余键会被拒绝；
* `content` 最短长度 1，禁止空串；
* `token_count >= 0`；`metadata` 键值一律 `str`（非字符串值在转换层被强制字符串化）。

---

## 4. 会话模型 `SessionMeta`

一次问诊会话的元数据，一会话一行。

| # | 字段 | proto 类型 | 存储类型 | 必填 | 说明 |
|---|---|---|---|---|---|
| 1 | `session_id` | `string` | `VARCHAR(64)` PK | 是 | 会话 ID |
| 2 | `tenant_id` | `string` | `VARCHAR(64)` | 是 | 租户 ID |
| 3 | `dept_id` | `string` | `VARCHAR(64)` | 是 | 科室 ID |
| 4 | `patient_id` | `string` | `VARCHAR(64)` | 是 | 患者 ID |
| 5 | `status` | `SessionStatus` | `VARCHAR(16)` | 是 | 生命周期状态，见 5.2，缺省 `active` |
| 6 | `message_count` | `int32` | `INT` | 是 | 已持久化消息条数，≥ 0，缺省 `0` |
| 7 | `created_at` | `int64` | `BIGINT` | 是 | 创建时间，UTC epoch 毫秒，> 0 |
| 8 | `updated_at` | `int64` | `BIGINT` | 是 | 最后更新时间，UTC epoch 毫秒；**不得早于 `created_at`** |
| 9 | `metadata` | `map<string,string>` | `JSON` | 是 | 扩展标签，缺省为空表 |

`updated_at` 缺省（`0`）时自动对齐 `created_at`；写入小于 `created_at` 的值会被拒绝。

### 4.1 会话状态机

```
                 ┌─────────────┐
   ┌────────────►│   ACTIVE    │◄────────────┐
   │             └──────┬──────┘             │
   │                    │                    │
   │        ┌───────────┴───────────┐        │
   │        ▼                       ▼        │
   │  ┌──────────┐           ┌────────────┐  │
   └──│  CLOSED  │           │  ARCHIVED  │──┘
      └────┬─────┘           └──────┬─────┘
           │                        │
           └───────────┬────────────┘
                       ▼
                 ┌───────────┐
                 │  DELETED  │  ← 终态（软删除，不可再流转）
                 └───────────┘
```

| 起始状态 | 允许流转到 |
|---|---|
| `ACTIVE` | `CLOSED`、`ARCHIVED` |
| `CLOSED` | `ACTIVE`（复诊重开）、`ARCHIVED` |
| `ARCHIVED` | `DELETED`（软删除） |
| `DELETED` | 无（终态） |

非法流转（如 `ACTIVE -> DELETED`）必须被拒绝。**软删除不物理清除消息**，物理清理由保留期
策略在宽限期结束后执行（见 7.4）。

---

## 5. 枚举定义

### 5.1 `MessageRole`

| 名称 | 值 | 说明 |
|---|---|---|
| `MESSAGE_ROLE_UNSPECIFIED` | 0 | proto3 零值，**禁止持久化** |
| `MESSAGE_ROLE_PATIENT` | 1 | 患者 |
| `MESSAGE_ROLE_DOCTOR` | 2 | 医生 |
| `MESSAGE_ROLE_ASSISTANT` | 3 | AI 助手 |
| `MESSAGE_ROLE_SYSTEM` | 4 | 系统消息（如摘要槽位） |

字符串形式落库（`VARCHAR(16)`）：`patient` / `doctor` / `assistant` / `system`。

### 5.2 `SessionStatus`

| 名称 | 值 | 字符串落库值 |
|---|---|---|
| `SESSION_STATUS_UNSPECIFIED` | 0 | 禁止持久化 |
| `SESSION_STATUS_ACTIVE` | 1 | `active` |
| `SESSION_STATUS_CLOSED` | 2 | `closed` |
| `SESSION_STATUS_ARCHIVED` | 3 | `archived` |
| `SESSION_STATUS_DELETED` | 4 | `deleted` |

---

## 6. 批量与快照消息

| 消息 | 字段 | 用途 |
|---|---|---|
| `MedMessageBatch` | `session_id = 1`、`repeated MedMessage messages = 2` | 批量写入、跨存储迁移、归档载荷 |
| `SessionSnapshot` | `SessionMeta meta = 1`、`repeated MedMessage messages = 2`、`int64 snapshot_at = 3`、`string schema_version = 4` | 快照导出 / 恢复（配外部校验和，见 8.2） |

`MedMessageBatch.messages` 的顺序即写入顺序，接收方必须按序追加，不得重排。

---

## 7. 物理存储规范

### 7.1 Redis（热会话）

每个会话使用两个键，均以统一存储键为前缀：

| 键 | 结构 | 说明 |
|---|---|---|
| `med:chat:{tenant_id}:{dept_id}:{session_id}:messages` | **List** | `RPUSH` 追加 protobuf 二进制消息体，天然保序，追加 O(1) |
| `med:chat:{tenant_id}:{dept_id}:{session_id}:meta` | **Hash** | 命名空间、状态、消息条数、时间戳；条数用 `HINCRBY` 原子累加 |

* **统一存储键**：`med:chat:{tenant_id}:{dept_id}:{session_id}`
* **集群 hash tag**：以 `{session_id}` 作为 hash tag，保证同一会话的 `messages` / `meta`
  两个键落在同一 slot（否则集群下无法把两者的命令放进同一个 pipeline）。
  租户 / 科室**不进** hash tag —— 否则整个租户会塌缩到单一 slot，丧失分片能力
* **批量写**：一次 `add_messages` 的全部命令打包进单个 pipeline，仅一次网络往返。
  单机用 `MULTI/EXEC` 事务 pipeline；**集群必须用非事务 pipeline** ——
  redis-py 的集群客户端弃用了 `MULTI`（`pipeline(transaction=True)` 抛
  `RedisClusterException`），集群侧由 hash tag 保证同 slot，放弃跨键原子性
* **TTL**：会话级过期走 Redis 原生 `EXPIRE`；写入后滑动续期（可选读取也续期）；
  `set_ttl(None)` 下发 `PERSIST` 恢复永不过期
* **⚠️ 客户端禁止开启 `decode_responses`**：消息体是 protobuf 二进制，解码会破坏数据

### 7.2 MySQL（持久化分表）

| 表 | 说明 |
|---|---|
| `med_message_00` ~ `med_message_15` | 消息分表，共 **16** 张同构表 |
| `med_session` | 会话元数据表，一会话一行 |
| `med_schema_version` | 迁移基线版本表（`revision` + `applied_at` 两列） |

**分表路由**：`med_message_{crc32(session_id) % 16}`，表名编号补零到两位（`00`…`15`）。

分表列定义（结构对齐 `MedMessage`，外加存储层保序列 `ordinal`）：

```sql
CREATE TABLE med_message_00 (
    message_id  VARCHAR(36)  NOT NULL,  -- PK
    session_id  VARCHAR(64)  NOT NULL,
    tenant_id   VARCHAR(64)  NOT NULL,
    dept_id     VARCHAR(64)  NOT NULL,
    patient_id  VARCHAR(64)  NOT NULL,
    role        VARCHAR(16)  NOT NULL,
    content     TEXT         NOT NULL,
    token_count INT          NOT NULL DEFAULT 0,
    masked      BOOL         NOT NULL DEFAULT 0,
    created_at  BIGINT       NOT NULL,
    ordinal     BIGINT       NOT NULL DEFAULT 0,  -- 会话内单调递增写入序号
    metadata    JSON         NOT NULL,
    PRIMARY KEY (message_id),
    INDEX (session_id, created_at, ordinal),   -- 会话时序扫描（保序）
    INDEX (tenant_id, dept_id)                 -- 租户/科室维度统计
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

`ordinal` **不属于** `MedMessage` 字段集，是存储层为保序附加的列：`message_id` 为 UUIDv7，
同毫秒内随机，仅按 `created_at` 排序无法还原写入顺序，因此由写入方按
「会话内最大 `ordinal` + 批内偏移」分配，读取一律按 `(created_at, ordinal)` 排序。
跨进程并发写同一会话时，`ordinal` 分配需由上层会话锁串行化（与 ES 归档侧同一约定）。

会话表索引：`(tenant_id, dept_id, status)`（租户科室维度列表）与 `(updated_at)`（TTL 扫描）。

迁移基线版本号：`0001_baseline`，建表后登记进 `med_schema_version`；重复初始化幂等。

### 7.3 Elasticsearch（冷归档）

| 项 | 规范 |
|---|---|
| 索引名 | `med-chat-archive-{yyyy.MM}`（按月滚动，月份按 **UTC** 计算） |
| 索引模板 | `med-chat-archive` |
| 检索模式 | `med-chat-archive-*` |
| 单次 bulk 上限 | `500` 文档 |
| 单次 search 上限 | `10000` 命中（ES `index.max_result_window` 默认值） |
| 排序键 | `created_at` → `archived_at` → `ordinal`（批内序号） |

排序键必须包含 `ordinal`：`message_id` 同毫秒内随机，单独使用会破坏与热存储一致的写入顺序语义。

### 7.4 保留期与软删除

| 参数 | 含义 |
|---|---|
| `retention_days` | `ARCHIVED` 会话保留天数，超期后打软删除标记 |
| `grace_days` | 软删除后的宽限天数，超期后**物理清除**消息数据（撤销窗口） |

清理流程：`ARCHIVED` --(超 retention_days)--> 软删除 `DELETED` --(超 grace_days)--> 物理清除。

---

## 8. 序列化规范

### 8.1 载荷编码

| 场景 | 编码 |
|---|---|
| 消息 / 会话 / 批量落库 | protobuf 二进制（**禁止 JSON**） |
| 归档文档字段 | protobuf 二进制的 base64 字符串（ES `binary` 字段） |
| API 请求 / 响应 DTO | JSON（**仅限 API 层**，不落库） |
| 快照文件包 | 自描述二进制容器（见 8.2） |

### 8.2 快照文件包格式

快照文件包为自描述二进制容器，布局如下（小端）：

```
magic(8B) | schema_version_len(1B) | schema_version(UTF-8) |
payload_len(4B, LE) | payload(protobuf SessionSnapshot) | sha256(32B)
```

| 项 | 规范 |
|---|---|
| magic | `MEDSNAP1`（8 字节 ASCII，用于快速识别文件类型与版本） |
| `schema_version` | 当前为 `1`，随序列化协议演进而递增 |
| `payload` | protobuf `SessionSnapshot` 二进制 |
| `sha256` | 覆盖 **从 magic 到 payload 末尾** 的全部字节 |

导入时校验顺序：长度 → magic → 头部完整性 → payload 长度自洽 → SHA-256 校验和。
任一失败必须拒绝导入（校验和不匹配视为文件损坏或被篡改）。恢复时还须校验目标会话与
文件包**命名空间一致**，杜绝跨租户越权写入。

---

## 9. 隐私脱敏字段规范

脱敏为 **字段级纯正则规则**，不含任何分词 / 实体识别 / 语义解析逻辑。默认纳管字段为消息正文
`content`，未纳管字段一律原样透传。

| 规则名 | 目标 | 保留策略 | 示例 |
|---|---|---|---|
| `id_card` | 18 位居民身份证号 | 保留前 6 位地址码 + 后 4 位 | `110101********1234` |
| `phone` | 中国大陆手机号（`1[3-9]` 开头 11 位） | 保留前 3 位 + 后 4 位 | `138****5678` |
| `medical_record_no` | 带标签的病历号 / 病案号 / 住院号 / 门诊号 | 仅脱敏号码本体，保留标签，末 4 位可读 | `病历号: ****1234` |
| `bed_no` | 带标签的床号（`床号: 12-3`） | 仅脱敏号码本体，保留标签 | `床号: **-*` |
| `bed_no_suffix` | 后缀式床号（`12床`） | 仅脱敏号码本体，保留「床」字 | `**床` |

规则按「具体优先」顺序串行应用（身份证先于手机号，避免长号被截断匹配）。

**`masked` 字段语义**：为 `true` 时表示 `content` 已按上述规则改写，**原始值不可从存储中还原**。
下游若需保留原文，必须在写入前自行留存于合规通道，不得依赖本库。

---

## 10. 兼容性与演进规则

1. **字段号冻结**：已发布字段号永不复用、永不重编号；删除字段时保留编号并标记 `reserved`。
2. **向后兼容新增**：仅允许新增字段与新增枚举值；接收方必须容忍未知字段（proto3 默认行为）。
3. **枚举零值**：所有枚举必须保留 `0` 号 `*_UNSPECIFIED` 值，且禁止持久化该值。
4. **快照 schema 版本**：文件包 `schema_version` 与 `SessionSnapshot.schema_version` 双写；
   版本不匹配时导入方须显式拒绝，不得静默降级解析。
5. **存储键不变**：Redis 键格式、MySQL 表名与 ES 索引前缀属于对外契约，变更等同破坏性升级。

---

## 11. 下游接入检查清单

- [ ] 落库载荷使用 protobuf 二进制，未使用 JSON
- [ ] 时间戳统一 UTC epoch 毫秒
- [ ] 业务 ID 满足 `^[A-Za-z0-9_.-]+$` 且不含 `:` / `{}`
- [ ] `message_id` 为合法 UUID（推荐 UUIDv7）
- [ ] Redis 客户端未开启 `decode_responses`；集群模式使用 `{session_id}` hash tag
- [ ] Redis 集群模式未使用事务 pipeline（`MULTI` 在集群下不可用），批量写走非事务 pipeline
- [ ] MySQL 写入按 `crc32(session_id) % 16` 路由到正确分表，并写入会话内单调递增的 `ordinal`
- [ ] MySQL / ES 读取按 `(created_at, ordinal)` 排序，未使用同毫秒内随机的 `message_id` 做 tiebreaker
- [ ] ES 归档按 UTC 月份滚动写入 `med-chat-archive-{yyyy.MM}`，排序键含 `ordinal`
- [ ] 会话状态流转符合 4.1 状态机，非法流转被拒绝
- [ ] 快照导入前校验 SHA-256 与命名空间一致性
- [ ] 未持久化任何 `*_UNSPECIFIED`（0 值）枚举

---

_本规范由 med-langchain-memory 维护；协议源文件：`protos/med_session.proto`。_

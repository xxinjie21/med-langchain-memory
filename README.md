# med-langchain-memory

> 医疗专属 LangChain 分布式会话存储中间件 · Medical-grade distributed chat message
> history middleware for LangChain (Python 3.11+ / LCEL)

## 定位

基于 LangChain `BaseChatMessageHistory` 抽象的医疗会话存储中间件，面向医院医患
问诊会话的持久化、多存储适配、医疗合规与上下文优化。

## 核心能力

- **多存储适配**：内存 / 本地文件 / Redis 集群 / MySQL 分表 / Elasticsearch 归档，
  适配器 + 工厂注册模式热插拔
- **统一序列化**：跨语言 Protobuf 协议，会话 TTL 自动归档、快照备份、跨存储迁移
- **医疗增强 Runnable**：多租户科室隔离、时序 + Token 预算双层裁剪、长会话摘要压缩、
  分布式会话锁、读写降级熔断
- **隐私合规**：字段级正则规则脱敏（手机号 / 身份证 / 病历号），操作审计、合规保留期

## 安装

```bash
pip install med-langchain-memory            # 核心（内存/文件存储）
pip install "med-langchain-memory[redis]"   # + Redis
pip install "med-langchain-memory[mysql]"   # + MySQL
pip install "med-langchain-memory[es]"      # + Elasticsearch 归档
pip install "med-langchain-memory[api]"     # + FastAPI 管理接口
```

## 快速开始

> 开发中，接口以 [ROADMAP.md](./ROADMAP.md) 迭代计划为准。

## 开发

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## License

[Apache-2.0](./LICENSE)

"""med-langchain-memory.

Medical-grade distributed chat message history middleware built on top of
LangChain's ``BaseChatMessageHistory`` abstraction.

Provides pluggable storage adapters (memory / file / Redis / MySQL / ES),
a unified protobuf serialization protocol, session lifecycle management,
rule-based field-level privacy masking, and a medical-enhanced
``RunnableWithMessageHistory``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

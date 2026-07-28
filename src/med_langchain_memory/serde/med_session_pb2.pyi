from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MessageRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MESSAGE_ROLE_UNSPECIFIED: _ClassVar[MessageRole]
    MESSAGE_ROLE_PATIENT: _ClassVar[MessageRole]
    MESSAGE_ROLE_DOCTOR: _ClassVar[MessageRole]
    MESSAGE_ROLE_ASSISTANT: _ClassVar[MessageRole]
    MESSAGE_ROLE_SYSTEM: _ClassVar[MessageRole]

class SessionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SESSION_STATUS_UNSPECIFIED: _ClassVar[SessionStatus]
    SESSION_STATUS_ACTIVE: _ClassVar[SessionStatus]
    SESSION_STATUS_CLOSED: _ClassVar[SessionStatus]
    SESSION_STATUS_ARCHIVED: _ClassVar[SessionStatus]
    SESSION_STATUS_DELETED: _ClassVar[SessionStatus]
MESSAGE_ROLE_UNSPECIFIED: MessageRole
MESSAGE_ROLE_PATIENT: MessageRole
MESSAGE_ROLE_DOCTOR: MessageRole
MESSAGE_ROLE_ASSISTANT: MessageRole
MESSAGE_ROLE_SYSTEM: MessageRole
SESSION_STATUS_UNSPECIFIED: SessionStatus
SESSION_STATUS_ACTIVE: SessionStatus
SESSION_STATUS_CLOSED: SessionStatus
SESSION_STATUS_ARCHIVED: SessionStatus
SESSION_STATUS_DELETED: SessionStatus

class MedMessage(_message.Message):
    __slots__ = ("message_id", "session_id", "tenant_id", "dept_id", "patient_id", "role", "content", "token_count", "masked", "created_at", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    DEPT_ID_FIELD_NUMBER: _ClassVar[int]
    PATIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    MASKED_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    session_id: str
    tenant_id: str
    dept_id: str
    patient_id: str
    role: MessageRole
    content: str
    token_count: int
    masked: bool
    created_at: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, message_id: _Optional[str] = ..., session_id: _Optional[str] = ..., tenant_id: _Optional[str] = ..., dept_id: _Optional[str] = ..., patient_id: _Optional[str] = ..., role: _Optional[_Union[MessageRole, str]] = ..., content: _Optional[str] = ..., token_count: _Optional[int] = ..., masked: bool = ..., created_at: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SessionMeta(_message.Message):
    __slots__ = ("session_id", "tenant_id", "dept_id", "patient_id", "status", "message_count", "created_at", "updated_at", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    DEPT_ID_FIELD_NUMBER: _ClassVar[int]
    PATIENT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    tenant_id: str
    dept_id: str
    patient_id: str
    status: SessionStatus
    message_count: int
    created_at: int
    updated_at: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, session_id: _Optional[str] = ..., tenant_id: _Optional[str] = ..., dept_id: _Optional[str] = ..., patient_id: _Optional[str] = ..., status: _Optional[_Union[SessionStatus, str]] = ..., message_count: _Optional[int] = ..., created_at: _Optional[int] = ..., updated_at: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MedMessageBatch(_message.Message):
    __slots__ = ("session_id", "messages")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    messages: _containers.RepeatedCompositeFieldContainer[MedMessage]
    def __init__(self, session_id: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[MedMessage, _Mapping]]] = ...) -> None: ...

class SessionSnapshot(_message.Message):
    __slots__ = ("meta", "messages", "snapshot_at", "schema_version")
    META_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_AT_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    meta: SessionMeta
    messages: _containers.RepeatedCompositeFieldContainer[MedMessage]
    snapshot_at: int
    schema_version: str
    def __init__(self, meta: _Optional[_Union[SessionMeta, _Mapping]] = ..., messages: _Optional[_Iterable[_Union[MedMessage, _Mapping]]] = ..., snapshot_at: _Optional[int] = ..., schema_version: _Optional[str] = ...) -> None: ...

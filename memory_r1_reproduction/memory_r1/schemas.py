from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryEvent(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


@dataclass
class DialogueTurn:
    speaker: str
    text: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionAnswer:
    question: str
    answer: str
    turn_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    id: str
    speaker: str
    text: str
    timestamp: str = ""
    source_turn: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryDecision:
    id: str
    text: str
    event: MemoryEvent
    speaker: str = ""
    old_memory: str = ""


@dataclass
class RetrievedMemory:
    entry: MemoryEntry
    score: float


@dataclass
class AnswerTrace:
    selected_memory_ids: List[str]
    answer: str
    raw_text: str = ""


@dataclass
class ManagerSample:
    dialogue_id: str
    participants: List[str]
    memory_snapshot: List[MemoryEntry]
    current_turn: DialogueTurn
    question: str
    answer: str


@dataclass
class AnswerSample:
    dialogue_id: str
    participants: List[str]
    question: str
    answer: str
    retrieved_memories: List[MemoryEntry]

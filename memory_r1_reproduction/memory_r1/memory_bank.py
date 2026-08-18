from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from .schemas import MemoryDecision, MemoryEntry, MemoryEvent


class MemoryBank:
    def __init__(self, entries: Optional[Iterable[MemoryEntry]] = None):
        self._entries: Dict[str, MemoryEntry] = {}
        if entries:
            for entry in entries:
                self._entries[entry.id] = entry
        self._next_id = self._compute_next_id()

    def _compute_next_id(self) -> int:
        numeric_ids = []
        for key in self._entries:
            if str(key).isdigit():
                numeric_ids.append(int(key))
        return (max(numeric_ids) + 1) if numeric_ids else 0

    def clone(self) -> "MemoryBank":
        return MemoryBank(self.entries())

    def entries(self) -> List[MemoryEntry]:
        return list(self._entries.values())

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(memory_id)

    def add(self, speaker: str, text: str, timestamp: str = "", source_turn: Optional[int] = None) -> MemoryEntry:
        memory_id = str(self._next_id)
        self._next_id += 1
        entry = MemoryEntry(
            id=memory_id,
            speaker=speaker,
            text=text,
            timestamp=timestamp,
            source_turn=source_turn,
        )
        self._entries[memory_id] = entry
        return entry

    def update(self, memory_id: str, text: str, timestamp: str = "") -> None:
        if memory_id not in self._entries:
            return
        entry = self._entries[memory_id]
        entry.text = text
        if timestamp:
            entry.timestamp = timestamp

    def delete(self, memory_id: str) -> None:
        self._entries.pop(memory_id, None)

    def apply(self, decisions: List[MemoryDecision], default_speaker: str, timestamp: str = "") -> None:
        for decision in decisions:
            if decision.event == MemoryEvent.ADD:
                speaker = decision.speaker or default_speaker
                self.add(speaker=speaker, text=decision.text, timestamp=timestamp)
            elif decision.event == MemoryEvent.UPDATE:
                self.update(memory_id=decision.id, text=decision.text, timestamp=timestamp)
            elif decision.event == MemoryEvent.DELETE:
                self.delete(memory_id=decision.id)

    def to_dict(self) -> List[dict]:
        return [asdict(entry) for entry in self.entries()]

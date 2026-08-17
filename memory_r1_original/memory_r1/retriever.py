from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .schemas import MemoryEntry, RetrievedMemory
from .utils import tokenize_for_retrieval


class LexicalMemoryRetriever:
    def __init__(self):
        pass

    def score(self, query: str, document: str) -> float:
        q_tokens = set(tokenize_for_retrieval(query))
        d_tokens = set(tokenize_for_retrieval(document))
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = len(q_tokens & d_tokens)
        return overlap / (len(q_tokens) ** 0.5 * len(d_tokens) ** 0.5)

    def retrieve(self, query: str, memories: Iterable[MemoryEntry], top_k: int = 5) -> List[RetrievedMemory]:
        scored = []
        for memory in memories:
            score = self.score(query, memory.text)
            if score > 0:
                scored.append(RetrievedMemory(entry=memory, score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def retrieve_per_speaker(
        self,
        query: str,
        memories: Iterable[MemoryEntry],
        participants: List[str],
        per_speaker_top_k: int = 30,
    ) -> List[MemoryEntry]:
        grouped: Dict[str, List[MemoryEntry]] = defaultdict(list)
        for memory in memories:
            grouped[memory.speaker].append(memory)

        merged: List[RetrievedMemory] = []
        for participant in participants:
            speaker_memories = grouped.get(participant, [])
            merged.extend(self.retrieve(query, speaker_memories, top_k=per_speaker_top_k))
        merged.sort(key=lambda x: x.score, reverse=True)
        return [item.entry for item in merged]
